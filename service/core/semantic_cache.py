"""语义缓存模块（Semantic Cache）

基于 redisvl 官方 SemanticCache + Redis Stack（带 RediSearch 向量搜索模块）实现：
  https://redis.io/docs/latest/develop/ai/redisvl/user_guide/llmcache/

核心流程：
  1. lookup()：将 query 转为向量，Redis 在服务端做 KNN 向量检索
              余弦距离 <= distance_threshold（默认 0.1）→ 命中，直接返回缓存答案
              distance_threshold 与余弦相似度的换算：distance = 1 - cosine_similarity
              即 distance_threshold=0.1 ↔ cosine_similarity >= 0.9
  2. save()  ：RAG 完成后将 (query_vector, answer) 写入 Redis，由 redisvl 自动管理 TTL

依赖要求（⚠️ 重要）：
  - Redis Stack（含 RediSearch 模块）：docker image redis/redis-stack-server
  - redisvl>=0.16.0：pip install redisvl
  - 自定义 DashScopeVectorizer：包装项目现有的 DashScopeEmbeddings，无需额外向量服务

环境变量：
  SEMANTIC_CACHE_ENABLED      设为 "false" 可全局禁用，默认 true
  SEMANTIC_CACHE_TTL          缓存 TTL（秒），默认 3600
  SEMANTIC_CACHE_THRESHOLD    余弦距离阈值（越小越严格），默认 0.1（对应相似度 >= 0.9）
  REDIS_URL                   Redis 连接地址，默认 redis://127.0.0.1:6379/0
"""

import asyncio
import logging
import os
import time
from typing import Any, Callable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# 默认配置
_DEFAULT_TTL: int = 3600          # 缓存 TTL（秒）
_DEFAULT_DISTANCE_THRESHOLD: float = 0.1   # 余弦距离阈值（0.1 ↔ 相似度 >= 0.9）
_DEFAULT_CACHE_NAME: str = "kb_semantic_cache"   # Redis 索引名称前缀


# ─────────────────────────── DashScope 向量适配器 ────────────────────────────

def _build_dashscope_vectorizer(embeddings, dims: int):
    """
    动态创建 DashScopeVectorizer 类（继承 redisvl BaseVectorizer）。

    为避免在模块加载时强制 import redisvl（Redis 不可用时不应报错），
    采用函数内动态创建子类的方式进行懒加载。
    """
    from redisvl.utils.vectorize.base import BaseVectorizer

    class DashScopeVectorizer(BaseVectorizer):
        """将 DashScopeEmbeddings 包装成 redisvl 兼容的向量化器。

        redisvl BaseVectorizer 是 Pydantic 模型，必须通过 model 字段构造。
        实现 _embed / _embed_many（同步）和 _aembed / _aembed_many（异步）即可。
        """

        class Config:
            # 允许存储非 Pydantic 字段（_embeddings）
            arbitrary_types_allowed = True

        # 私有属性：持有 DashScopeEmbeddings 实例
        _embeddings_client: Any = None

        def __init__(self, embeddings_client):
            # Pydantic 模型初始化：传入 model 和 dims 字段
            super().__init__(model=embeddings_client.model, dims=dims)
            # 绕过 Pydantic 的字段管理直接设置私有属性
            object.__setattr__(self, "_embeddings_client", embeddings_client)

        def _embed(self, content: Any = "", **kwargs) -> List[float]:
            """同步生成单条文本的向量（redisvl 抽象方法实现）。"""
            text = str(content) if content else ""
            _t = time.perf_counter()
            result = self._embeddings_client.embed_query(text)
            logger.info(
                f"🔢 [语义缓存] Embedding API 耗时: {(time.perf_counter() - _t)*1000:.1f}ms"
            )
            return result

        def _embed_many(self, content: Any = None, **kwargs) -> List[List[float]]:
            """同步批量生成向量（redisvl 抽象方法实现）。"""
            texts = content if isinstance(content, list) else [str(content)]
            return self._embeddings_client.embed_documents(texts)

        async def _aembed(self, content: Any = "", **kwargs) -> List[float]:
            """异步生成单条文本向量（通过线程池避免阻塞事件循环）。"""
            text = str(content) if content else ""
            return await asyncio.to_thread(self._embeddings_client.embed_query, text)

        async def _aembed_many(self, content: Any = None, **kwargs) -> List[List[float]]:
            """异步批量生成向量。"""
            texts = content if isinstance(content, list) else [str(content)]
            return await asyncio.to_thread(self._embeddings_client.embed_documents, texts)

    return DashScopeVectorizer(embeddings)


# ───────────────────────── SemanticCacheWrapper ──────────────────────────────

class SemanticCacheWrapper:
    """对 redisvl.SemanticCache 的轻量封装，提供：
    - lookup() / save() 异步接口（与 graph.py 无缝集成）
    - 缓存失败自动降级（不阻断主流程）
    - 租户隔离（每个 tenant_id 使用独立 Redis 索引）
    """

    def __init__(
        self,
        redis_url: str,
        embeddings,
        embedding_dims: int,
        distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD,
        ttl: Optional[int] = _DEFAULT_TTL,
        cache_name: str = _DEFAULT_CACHE_NAME,
    ):
        self._redis_url = redis_url
        self._embeddings = embeddings
        self._embedding_dims = embedding_dims
        self._distance_threshold = distance_threshold
        self._ttl = ttl
        self._cache_name = cache_name
        # 每个 tenant_id 对应一个独立的 SemanticCache 实例（懒加载）
        self._caches: dict = {}
        self._vectorizer = None  # 懒加载
        # Redis 不支持 RediSearch 时置为 True，后续调用直接静默跳过（不再重试）
        self._unavailable: bool = False

        logger.info(
            f"语义缓存初始化：distance_threshold={distance_threshold}（相似度>="
            f"{1 - distance_threshold:.1f}），TTL={ttl}s，"
            f"redis_url={redis_url}"
        )

    def _get_vectorizer(self):
        """延迟创建向量化器（单例复用）。"""
        if self._vectorizer is None:
            self._vectorizer = _build_dashscope_vectorizer(
                self._embeddings, self._embedding_dims
            )
        return self._vectorizer

    def _get_cache(self, tenant_id: str):
        """获取指定租户的 SemanticCache 实例（首次访问时懒加载创建）。

        若 Redis 不支持 RediSearch（缺少 FT.* 命令），标记 _unavailable=True，
        后续所有调用均立即跳过，不再重试，避免持续报错。
        """
        if self._unavailable:
            return None

        if tenant_id not in self._caches:
            from redisvl.extensions.cache.llm import SemanticCache

            # 每个租户用独立的索引名，实现数据隔离
            name = f"{self._cache_name}_{tenant_id}"
            try:
                cache = SemanticCache(
                    name=name,
                    distance_threshold=self._distance_threshold,
                    ttl=self._ttl,
                    vectorizer=self._get_vectorizer(),
                    redis_url=self._redis_url,
                )
                self._caches[tenant_id] = cache
                logger.info(f"语义缓存：为租户 {tenant_id} 创建 Redis 索引 '{name}'")
            except Exception as e:
                err_msg = str(e).lower()
                # FT.* 命令不存在 → 当前 Redis 缺少 RediSearch 模块
                if "ft." in err_msg or "unknown command" in err_msg:
                    self._unavailable = True
                    logger.warning(
                        "⚠️ 语义缓存已禁用：当前 Redis 不支持 RediSearch 向量搜索模块。\n"
                        "   原因：redisvl SemanticCache 需要 Redis Stack（含 RediSearch）。\n"
                        "   修复：将 docker-compose.yml 中的 Redis 镜像改为 "
                        "redis/redis-stack-server:latest 并重启容器。\n"
                        f"   底层错误：{e}"
                    )
                    return None
                # 其他未知错误：向上抛出，由调用方 try-except 处理
                raise

        return self._caches.get(tenant_id)

    async def lookup(self, query: str, tenant_id: str) -> Tuple[Optional[str], float]:
        """语义查找：返回 (命中的答案 or None, 余弦相似度)。

        redisvl 在 Redis 服务端完成 KNN 向量检索，
        匹配时返回 [{response, prompt, vector_distance, ...}, ...]。
        vector_distance 是余弦距离（越小越相似），转换为相似度 = 1 - distance。
        """
        _t0 = time.perf_counter()
        try:
            # ── 阶段①：获取缓存实例（首次创建索引，后续从内存字典直接返回）────
            _t1 = time.perf_counter()
            cache = await asyncio.to_thread(self._get_cache, tenant_id)
            _get_cache_ms = (time.perf_counter() - _t1) * 1000

            # Redis 不支持 RediSearch 时 _get_cache 返回 None，静默跳过
            if cache is None:
                return None, 0.0

            # ── 阶段②：cache.check()（内含 Embedding API + Redis KNN 两步）─────
            # 注意：DashScopeVectorizer._embed() 中已单独打印 Embedding 耗时
            _t2 = time.perf_counter()
            results = await asyncio.to_thread(cache.check, prompt=query)
            _check_ms = (time.perf_counter() - _t2) * 1000
            _total_ms = (time.perf_counter() - _t0) * 1000

            logger.info(
                f"⏱️  [语义缓存] lookup 分段耗时 | "
                f"获取实例: {_get_cache_ms:.1f}ms | "
                f"cache.check(embed+KNN): {_check_ms:.1f}ms | "
                f"合计: {_total_ms:.1f}ms"
            )

            if results:
                best = results[0]
                # vector_distance 字段由 redisvl 注入，表示余弦距离
                distance = float(best.get("vector_distance", self._distance_threshold))
                similarity = round(1.0 - distance, 4)
                answer = best.get("response", "")
                logger.info(
                    f"✅ 语义缓存命中！租户={tenant_id}，余弦距离={distance:.4f}，"
                    f"相似度={similarity:.4f}，答案长度={len(answer)} 字符"
                )
                return answer, similarity

            logger.info(f"语义缓存未命中。租户={tenant_id}，query='{query[:40]}...'")
            return None, 0.0

        except Exception as e:
            logger.error(f"语义缓存 lookup 异常（降级处理）：{e}", exc_info=True)
            return None, 0.0

    async def save(self, query: str, answer: str, tenant_id: str) -> bool:
        """将 (query, answer) 写入语义缓存，由 redisvl 自动管理向量索引和 TTL。"""
        if not answer or not answer.strip():
            logger.debug("语义缓存：答案为空，跳过写入")
            return False
        try:
            cache = await asyncio.to_thread(self._get_cache, tenant_id)
            # Redis 不支持 RediSearch 时 _get_cache 返回 None，静默跳过
            if cache is None:
                return False
            await asyncio.to_thread(cache.store, prompt=query, response=answer)
            logger.info(
                f"语义缓存写入成功。租户={tenant_id}，"
                f"query 长度={len(query)} 字符，TTL={self._ttl}s"
            )
            return True
        except Exception as e:
            logger.error(f"语义缓存 save 异常（降级处理）：{e}", exc_info=True)
            return False

    async def invalidate_tenant(self, tenant_id: str) -> bool:
        """清除指定租户的所有缓存条目（知识库更新后调用）。"""
        try:
            cache = await asyncio.to_thread(self._get_cache, tenant_id)
            if cache is None:
                return False
            await asyncio.to_thread(cache.clear)
            logger.info(f"语义缓存清除成功：租户={tenant_id}")
            return True
        except Exception as e:
            logger.error(f"语义缓存 invalidate_tenant 异常：{e}", exc_info=True)
            return False


# ── 模块级单例（延迟初始化）────────────────────────────────────────────────────
_SEMANTIC_CACHE: Optional[SemanticCacheWrapper] = None


def get_semantic_cache() -> Optional[SemanticCacheWrapper]:
    """获取全局语义缓存单例。

    首次调用时初始化（依赖 config.get_embeddings() 和 REDIS_URL）。
    若 Redis 不可用或 redisvl 未安装，返回 None（主流程不受影响）。

    ⚠️ Redis 需要 redis-stack-server 镜像（支持 RediSearch 向量搜索模块）。
    """
    global _SEMANTIC_CACHE
    if _SEMANTIC_CACHE is not None:
        return _SEMANTIC_CACHE

    # 支持通过环境变量全局禁用
    if os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "false":
        logger.info("语义缓存：已通过环境变量 SEMANTIC_CACHE_ENABLED=false 禁用")
        return None

    try:
        import core.config as config

        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        embeddings = config.get_embeddings()
        ttl = int(os.getenv("SEMANTIC_CACHE_TTL", str(_DEFAULT_TTL)))
        # distance_threshold：余弦距离阈值（0.1 ↔ 相似度 >= 0.9）
        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", str(_DEFAULT_DISTANCE_THRESHOLD)))
        cache_name = os.getenv("SEMANTIC_CACHE_NAME", _DEFAULT_CACHE_NAME)

        _SEMANTIC_CACHE = SemanticCacheWrapper(
            redis_url=redis_url,
            embeddings=embeddings,
            embedding_dims=config.EMBEDDING_DIM,
            distance_threshold=threshold,
            ttl=ttl,
            cache_name=cache_name,
        )
        return _SEMANTIC_CACHE

    except Exception as e:
        logger.error(f"语义缓存初始化失败（降级：禁用缓存）：{e}", exc_info=True)
        return None
