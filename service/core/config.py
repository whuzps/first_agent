"""配置与资源加载模块

负责：
- 读取环境变量与默认值
- 初始化对话模型与向量检索组件
- 提供向量索引、订单数据库与图检查点的获取函数
"""
import os
import logging

logger = logging.getLogger(__name__)
from typing import Optional
import dotenv
from collections import deque
import threading
import json
import time
from pathlib import Path
import redis.asyncio as redis
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from core.prompts import SUMMARIZATION_PROMPT

dotenv.load_dotenv()
# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = Path(BASE_DIR) / "logs"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

KB_INDEX_DIR = os.getenv("KB_INDEX_DIR")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/first_agent")
CHECKPOINT_POSTGRES_DSN = os.getenv("CHECKPOINT_POSTGRES_DSN", POSTGRES_DSN)
ORDERS_DB_PATH = os.getenv("ORDERS_DB_PATH", POSTGRES_DSN)
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", CHECKPOINT_POSTGRES_DSN)
SUPPORT_DB_PATH = os.getenv("SUPPORT_DB_PATH", POSTGRES_DSN)
HUMAN_SUPPORT_URL = os.getenv("HUMAN_SUPPORT_URL")
TENANTS_BASE_DIR = os.getenv("TENANTS_BASE_DIR")
PRODUCT_TENANT_MAP = os.getenv("PRODUCT_TENANT_MAP")
LONG_TERM_MEMORY_DB_PATH = os.getenv("LONG_TERM_MEMORY_DB_PATH")

SUPPORTED_MODELS = ("qwen-turbo", "qwen-plus", "qwen-omni-turbo", "qwen-flash")
DEFAULT_MODEL_PARAMS = {"qwen-turbo": {}, "qwen-plus": {}, "qwen-omni-turbo": {}, "qwen-flash": {}}
_CURRENT_MODEL = MODEL_NAME
_MODEL_LOCK = threading.RLock()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_test_v1")
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "memory_test_v1")

EMBEDDING_DIM = 1024
RERANK_MODEL = "gte-rerank-v2"

# 最大重试次数
MAX_ATTEMPTS = 3
# 最大移交人工重试次数
MAX_HANDOFF_RETRY = 3 

# 定义订单号正则规则（可根据实际业务调整，示例：ORD开头+8-16位数字）
ORDER_ID_PATTERN = r"ORD\d{8,16}"

# 使用langSmith做Tracing
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "first-agent")


def _base_dir() -> str:
    return os.path.dirname(__file__)

def _norm_tenant(tid: Optional[str]) -> str:
    t = str(tid or "").strip()
    if not t:
        return "default"
    import re
    if re.fullmatch(r"[A-Za-z0-9_]{1,64}", t):
        return t
    return "default"

def _tenants_root() -> str:
    base = _base_dir()
    if TENANTS_BASE_DIR:
        p = TENANTS_BASE_DIR
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        return p
    return os.path.normpath(os.path.join(base, "tenants"))

def _tenant_dir(tid: Optional[str]) -> str:
    t = _norm_tenant(tid)
    return os.path.normpath(os.path.join(_tenants_root(), t))


VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "qwen-vl-max")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=_CURRENT_MODEL, api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)


def get_vl_llm() -> ChatOpenAI:
    """获取视觉语言模型（qwen-vl-max），用于包含图片的多模态对话"""
    return ChatOpenAI(model=VL_MODEL_NAME, api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)


def get_small_llm() -> ChatOpenAI:
    """获取小模型（qwen-turbo），用于简单任务如query改写"""
    return ChatOpenAI(model="qwen-turbo", api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)


def get_embeddings() -> DashScopeEmbeddings:
    """创建嵌入模型，用于向量检索。"""
    return DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)


def get_collection_name(tenant_id: Optional[str] = None) -> str:
    """根据租户 ID 获取对应的集合名称。"""
    return (_norm_tenant(tenant_id) if tenant_id is not None else "default") + "_"  + COLLECTION_NAME


def get_memory_collection_name(tenant_id: Optional[str] = None) -> str:
    """根据租户 ID 获取对应的记忆集合名称。"""
    return (_norm_tenant(tenant_id) if tenant_id is not None else "default") + "_" + MEMORY_COLLECTION_NAME


def get_orders_db_path(tenant_id: Optional[str] = None) -> Optional[str]:
    """兼容旧接口：返回订单 PostgreSQL DSN。"""
    return get_postgres_dsn(tenant_id)


def get_postgres_dsn(tenant_id: Optional[str] = None) -> str:
    """返回业务 PostgreSQL DSN（按租户可覆盖）。"""
    tid = _norm_tenant(tenant_id)
    env_key = f"POSTGRES_DSN_{tid.upper()}"
    return os.getenv(env_key, POSTGRES_DSN)


def get_checkpointer_dsn(tenant_id: Optional[str] = None) -> str:
    """返回 LangGraph checkpointer PostgreSQL DSN。"""
    tid = _norm_tenant(tenant_id)
    env_key = f"CHECKPOINT_POSTGRES_DSN_{tid.upper()}"
    return os.getenv(env_key, CHECKPOINT_POSTGRES_DSN)


def get_checkpointer_path(tenant_id: Optional[str] = None) -> str:
    """兼容旧接口：返回 checkpointer PostgreSQL DSN。"""
    return get_checkpointer_dsn(tenant_id)


class _Stats:
    def __init__(self, maxlen: int = 1000):
        self.lock = threading.Lock()
        self.window = deque(maxlen=maxlen)
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None
    def update(self, v: float):
        with self.lock:
            self.window.append(v)
            self.count += 1
            self.sum += v
            self.min = v if self.min is None or v < self.min else self.min
            self.max = v if self.max is None or v > self.max else self.max
    def snapshot(self) -> dict:
        with self.lock:
            n = self.count
            avg = (self.sum / n) if n else 0.0
            mn = self.min if self.min is not None else 0.0
            mx = self.max if self.max is not None else 0.0
            arr = list(self.window)
        p95 = 0.0
        if arr:
            arr.sort()
            idx = max(int(len(arr) * 0.95) - 1, 0)
            p95 = arr[idx]
        return {"count": n, "min_ms": mn, "max_ms": mx, "avg_ms": avg, "p95_ms": p95}

class Metrics:
    def __init__(self):
        self._stats = {}
        self._lock = threading.Lock()
    def update(self, key: str, v: float):
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = _Stats()
                self._stats[key] = s
        s.update(v)
    def snapshot(self, key: str) -> dict:
        s = self._stats.get(key)
        return s.snapshot() if s else {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}
    def snapshot_all(self) -> dict:
        with self._lock:
            keys = list(self._stats.keys())
        return {k: self.snapshot(k) for k in keys}

_METRICS: Optional[Metrics] = None
_REDIS = None
_SESSIONS = {}

# ========== 短期记忆配置 ==========
# 最近 N 轮对话直接放入 prompt 上下文（1轮 = 1条用户消息 + 1条助手消息）
SESSION_WINDOW_TURNS: int = int(os.getenv("SESSION_WINDOW_TURNS", "5"))
# 每新增 N 轮后触发一次后台增量摘要更新
SUMMARY_TRIGGER_EVERY: int = int(os.getenv("SUMMARY_TRIGGER_EVERY", "5"))
# Redis/内存中最多保留的消息条数（足够大，摘要机制负责压缩）
SESSION_MAX_MESSAGES: int = int(os.getenv("SESSION_MAX_MESSAGES", "200"))

# 本地内存摘要缓存（Redis 不可用时的降级存储）
# 结构：{summary_key -> {"summary": str, "summarized_until": int, "updated_at": int}}
_SUMMARIES: dict = {}

def get_metrics() -> Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS

async def get_redis():
    global _REDIS
    if _REDIS is not None:
        return _REDIS
    try:
        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        _REDIS = redis.from_url(url, decode_responses=True)
        await _REDIS.ping()
        logger.info("Redis 异步客户端连接成功: %s", url)
        return _REDIS
    except Exception as e:
        logger.warning("Redis 异步客户端连接失败，将回退内存会话: %s", e)
        _REDIS = None
        return None

def _sess_key(thread_id: str) -> str:
    return f"session:{thread_id}"


def _summary_key(thread_id: str, user_id: Optional[str] = None) -> str:
    """生成摘要缓存 key，绑定 thread_id + user_id"""
    uid = user_id or "anon"
    return f"summary:{thread_id}:{uid}"


async def get_session_summary(thread_id: str, user_id: Optional[str] = None) -> dict:
    """从缓存（Redis 优先，本地内存回退）中获取会话历史摘要

    Returns:
        dict 包含 summary(str)、summarized_until(int)、updated_at(int)
    """
    key = _summary_key(thread_id, user_id)
    r = await get_redis()
    if r is not None:
        try:
            val = await r.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
    return dict(_SUMMARIES.get(key) or {})


async def set_session_summary(
    thread_id: str,
    user_id: Optional[str] = None,
    summary: str = "",
    summarized_until: int = 0,
    ttl_seconds: int = 86400,
):
    """保存会话摘要到缓存（Redis + 本地内存双写）

    Args:
        summarized_until: 已摘要到的消息列表下标（不含），下次只取 [summarized_until:] 的新消息
        ttl_seconds: Redis key 有效期，默认 24 小时
    """
    key = _summary_key(thread_id, user_id)
    data = {
        "summary": summary,
        "summarized_until": summarized_until,
        "updated_at": int(time.time()),
    }
    r = await get_redis()
    if r is not None:
        try:
            await r.setex(key, ttl_seconds, json.dumps(data, ensure_ascii=False))
        except Exception:
            pass
    _SUMMARIES[key] = data


async def get_context_for_prompt(
    thread_id: str,
    user_id: Optional[str] = None,
    n_window: Optional[int] = None,
) -> tuple:
    """构建用于 prompt 的上下文：最近 N 轮对话 + 历史摘要

    最近 N 轮直接放入 prompt（原文），N 轮之前的内容以摘要形式呈现。

    Args:
        n_window: 保留的最近轮数，None 时使用 SESSION_WINDOW_TURNS

    Returns:
        (recent_context, summary_context):
            recent_context  - 最近 N 轮对话的格式化文本（"role: content" 每行一条）
            summary_context - 更早对话的摘要文本（空字符串表示无摘要）
    """
    if n_window is None:
        n_window = SESSION_WINDOW_TURNS

    all_messages = await get_all_session_messages(thread_id)
    window_size = n_window * 2  # N 轮 = 2N 条消息

    recent_msgs = all_messages[-window_size:] if len(all_messages) > window_size else all_messages
    recent_context = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in recent_msgs
    )

    summary_data = await get_session_summary(thread_id, user_id)
    summary_context = summary_data.get("summary", "")

    return recent_context, summary_context


async def maybe_trigger_summary_update(thread_id: str, user_id: Optional[str] = None):
    """检查是否到达触发点，若是则在后台启动增量摘要更新

    每新增 SUMMARY_TRIGGER_EVERY 轮对话触发一次，整个过程异步非阻塞。
    """
    try:
        all_messages = await get_all_session_messages(thread_id)
        total_turns = len(all_messages) // 2  # 向下取整得到完整轮数
        if total_turns > 0 and total_turns % SUMMARY_TRIGGER_EVERY == 0:
            logger.info(f"触发增量摘要更新: thread_id={thread_id}, total_turns={total_turns}")
            await _do_incremental_summary_update(thread_id, user_id)
    except Exception as e:
        logger.warning(f"摘要触发检查失败: thread_id={thread_id}, error={e}")


async def _do_incremental_summary_update(thread_id: str, user_id: Optional[str] = None):
    """执行增量摘要更新（后台任务）

    策略：
    1. 取「窗口之外」的所有消息作为待摘要范围
    2. 对比 summarized_until，只取「新增」部分
    3. 若已有摘要，用 INCREMENTAL_SUMMARY_PROMPT 合并；否则用 SUMMARIZATION_PROMPT 全量生成
    4. 写回缓存
    """
    try:
        from core.prompts import SUMMARIZATION_PROMPT, INCREMENTAL_SUMMARY_PROMPT

        all_messages = await get_all_session_messages(thread_id)
        window_size = SESSION_WINDOW_TURNS * 2

        # 只摘要窗口之外的消息
        msgs_to_consider = all_messages[:-window_size] if len(all_messages) > window_size else []
        if not msgs_to_consider:
            logger.info(f"增量摘要无需更新: thread_id={thread_id}，消息均在窗口内")
            return

        summary_data = await get_session_summary(thread_id, user_id)
        existing_summary: str = summary_data.get("summary", "")
        summarized_until: int = summary_data.get("summarized_until", 0)

        # 只取上次摘要之后新增的消息
        new_msgs = msgs_to_consider[summarized_until:]
        if not new_msgs:
            logger.info(f"增量摘要无需更新: thread_id={thread_id}，无新消息")
            return

        new_msgs_str = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in new_msgs
        )

        summarize_llm = get_llm()
        if existing_summary:
            # 增量更新：将新消息合并进现有摘要
            prompt = INCREMENTAL_SUMMARY_PROMPT.format(
                existing_summary=existing_summary,
                new_messages=new_msgs_str,
            )
        else:
            # 首次生成摘要
            prompt = SUMMARIZATION_PROMPT.format(chat_history=new_msgs_str)

        result = await summarize_llm.ainvoke(prompt)
        new_summary = str(getattr(result, "content", result)).strip()

        new_summarized_until = len(msgs_to_consider)
        await set_session_summary(thread_id, user_id, new_summary, new_summarized_until)
        logger.info(
            f"增量摘要更新完成: thread_id={thread_id}, "
            f"new_msgs={len(new_msgs)}, summarized_until={new_summarized_until}"
        )
    except Exception as e:
        logger.warning(f"增量摘要更新失败: thread_id={thread_id}, error={e}")

async def get_session_messages(thread_id: str, maxlen: int = 5) -> list:
    r = await get_redis()
    if r is not None:
        try:
            key = _sess_key(thread_id)
            vals = await r.lrange(key, -maxlen, -1)
            out = []
            for v in vals:
                try:
                    out.append(json.loads(v))
                except Exception:
                    pass
            return out
        except Exception:
            pass
    now = int(time.time())
    sess = _SESSIONS.get(thread_id) or {"ts": now, "arr": []}
    arr = sess["arr"][-maxlen:]
    return list(arr)

async def get_all_session_messages(thread_id: str) -> list:
    """读取会话所有消息，优先 Redis，失败则回退内存（无长度限制）。"""
    r = await get_redis()
    if r is not None:
        try:
            key = _sess_key(thread_id)
            vals = await r.lrange(key, 0, -1)
            out = []
            for v in vals:
                try:
                    out.append(json.loads(v))
                except Exception:
                    pass
            return out
        except Exception:
            pass
    now = int(time.time())
    sess = _SESSIONS.get(thread_id) or {"ts": now, "arr": []}
    return list(sess["arr"])


async def delete_session_messages(thread_id: str) -> None:
    """删除指定会话的消息历史及摘要缓存（内存 + Redis）。"""
    # 清除内存中的消息
    _SESSIONS.pop(thread_id, None)
    # 清除内存中的摘要（key 形如 "summary:{thread_id}:{user_id}"）
    keys_to_del = [k for k in list(_SUMMARIES.keys()) if k.startswith(f"summary:{thread_id}:")]
    for k in keys_to_del:
        _SUMMARIES.pop(k, None)
    # 清除 Redis 中的消息与摘要
    r = await get_redis()
    if r is not None:
        try:
            await r.delete(_sess_key(thread_id))
            async for k in r.scan_iter(match=f"summary:{thread_id}:*"):
                await r.delete(k)
        except Exception:
            pass


async def summarize_messages(thread_id: str) -> list:
    """剪辑和摘要上下文历史记录：保留最新两条，其余形成摘要（异步版本，向后兼容）"""
    logger.info(f"🔎 获取上下文历史, session_id: {thread_id}")
    stored_messages = await get_all_session_messages(thread_id)
    if len(stored_messages) <= 4:
        logger.info(f"🔎 获取上下文历史完成: {stored_messages}")
        return stored_messages
    last_two_messages = stored_messages[-4:]
    messages_to_summarize = stored_messages[:-4]

    summarize_llm = get_llm()
    summary = await summarize_llm.ainvoke(SUMMARIZATION_PROMPT.format(chat_history=messages_to_summarize))
    logger.info(f"Summary: {summary.content}")

    history = []
    history.append(last_two_messages[0])
    history.append(last_two_messages[1])
    history.append({"role": "chat_history_summarization", "content": summary.content})
    logger.info(f"🔎 获取上下文历史完成: {history}")

    return history

async def append_session_message(thread_id: str, role: str, content: str, ttl_seconds: int = 86400, maxlen: int = SESSION_MAX_MESSAGES):
    item = {"role": role, "content": content}
    r = await get_redis()
    if r is not None:
        try:
            key = _sess_key(thread_id)
            await r.rpush(key, json.dumps(item, ensure_ascii=False))
            await r.ltrim(key, -maxlen, -1)
            await r.expire(key, ttl_seconds)
            return True
        except Exception:
            pass
    now = int(time.time())
    sess = _SESSIONS.get(thread_id)
    if sess is None:
        sess = {"ts": now, "arr": []}
        _SESSIONS[thread_id] = sess
    sess["ts"] = now
    arr = sess["arr"]
    arr.append(item)
    if len(arr) > maxlen:
        del arr[0:len(arr)-maxlen]
    return True

async def reset_session(thread_id: str):
    r = await get_redis()
    if r is not None:
        try:
            await r.delete(_sess_key(thread_id))
            return True
        except Exception:
            pass
    _SESSIONS.pop(thread_id, None)
    return True

def get_supported_models() -> list:
    return list(SUPPORTED_MODELS)

_TENANT_MODELS: dict = {}
_PRODUCT_MAP_CACHE: Optional[dict] = None

def get_current_model_name(tenant_id: Optional[str] = None) -> str:
    if tenant_id is None:
        return _CURRENT_MODEL
    t = _norm_tenant(tenant_id)
    v = _TENANT_MODELS.get(t)
    return v or _CURRENT_MODEL

def validate_model(name: str) -> dict:
    n = str(name or "").strip()
    if n not in SUPPORTED_MODELS:
        return {"ok": False, "code": "unsupported", "message": "模型不受支持"}
    try:
        _ = ChatOpenAI(model=n, api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)
        return {"ok": True, "code": "ok", "message": "可用"}
    except Exception as e:
        return {"ok": False, "code": "init_error", "message": str(e)}

def switch_model(name: str, tenant_id: Optional[str] = None) -> dict:
    n = str(name or "").strip()
    with _MODEL_LOCK:
        v = validate_model(n)
        if not v.get("ok"):
            return {"ok": False, "code": v.get("code"), "message": v.get("message")}
        if tenant_id is None:
            old = _CURRENT_MODEL
            if n == old:
                return {"ok": True, "code": "noop", "message": "已是当前模型", "old": old, "new": old}
            globals()["MODEL_NAME"] = n
            globals()["_CURRENT_MODEL"] = n
            logger.info("model switched: %s -> %s", old, n)
            try:
                base = os.path.dirname(__file__)
                logs = os.path.join(base, "logs")
                os.makedirs(logs, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                line = f"{ts} | INFO | root | model switch old={old} new={n}"
                with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
            return {"ok": True, "code": "switched", "message": "切换成功", "old": old, "new": n}
        t = _norm_tenant(tenant_id)
        old = _TENANT_MODELS.get(t) or _CURRENT_MODEL
        _TENANT_MODELS[t] = n
        logger.info("tenant model switched: %s -> %s for %s", old, n, t)
        try:
            base = os.path.dirname(__file__)
            logs = os.path.join(base, "logs")
            os.makedirs(logs, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            line = f"{ts} | INFO | root | tenant model switch tenant={t} old={old} new={n}"
            with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        return {"ok": True, "code": "switched", "message": "切换成功", "old": old, "new": n}

def get_model_lock():
    return _MODEL_LOCK

def get_long_term_memory_db_path(tenant_id: Optional[str] = None) -> str:
    """返回长期记忆数据库 DSN（迁移到 PostgreSQL）。"""
    if LONG_TERM_MEMORY_DB_PATH and tenant_id is None:
        return LONG_TERM_MEMORY_DB_PATH
    return get_postgres_dsn(tenant_id)

