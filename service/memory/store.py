"""长期记忆模块（PostgreSQL 主存储 + Milvus 可选向量索引）

设计目标：
- 跨会话持久化用户相关信息（画像、偏好、关键事实、重要事件）
- 每次对话后异步提取并智能更新记忆
- 冲突解决：LLM 决策（ADD / UPDATE / DELETE / NONE）避免重复和矛盾
- 遗忘策略：
    1. 时间衰减（指数衰减）：长期未访问的记忆有效重要度自动下降
    2. 访问加成：被检索到的记忆保持较高有效重要度
    3. LLM 智能审查：每 N 次对话触发一次大模型清理（过时 / 空洞 / 冲突）
"""

import logging

logger = logging.getLogger(__name__)
import psycopg
import json
import os
import math
import traceback
import threading
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from typing_extensions import Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
from psycopg.rows import dict_row

import core.config as config  # 供本模块内 config.xxx 使用
import core.postgres as postgres

try:
    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType, Collection, utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus 未安装，Milvus 向量检索不可用，将降级为关键词检索")


# ==================== 枚举 & 数据模型 ====================

class MemoryType(str, Enum):
    USER_PROFILE      = "user_profile"        # 用户画像
    PREFERENCE        = "preference"          # 用户偏好
    FACT              = "fact"                # 关键事实
    CONVERSATION_SUMMARY = "conversation_summary"  # 对话摘要
    IMPORTANT_EVENT   = "important_event"     # 重要事件


class ExtractedMemoryItem(BaseModel):
    """LLM 提取的单条记忆结构"""
    memory_type: Literal["user_profile", "preference", "fact", "important_event"] = Field(
        None, description="记忆类型"
    )
    content: str = Field(description="记忆内容，具体简洁")
    importance: int = Field(description="重要程度 1-10", ge=1, le=10)


class ExtractedMemoryList(BaseModel):
    """LLM 提取的记忆列表（用于结构化输出）"""
    memories: List[ExtractedMemoryItem] = Field(default_factory=list)


class MemoryUpdateOperation(BaseModel):
    """LLM 更新决策（用于结构化输出）"""
    operation: Literal["ADD", "UPDATE", "DELETE", "NONE"] = Field(description="操作类型")
    memory_id: Optional[int] = Field(None, description="UPDATE/DELETE 时指定的记忆 ID")
    content: Optional[str] = Field(None, description="ADD/UPDATE 时的新内容")
    memory_type: Optional[str] = Field(None, description="ADD 时的记忆类型")
    importance: Optional[int] = Field(None, description="ADD/UPDATE 时的重要程度")


class MemoryReviewItem(BaseModel):
    """LLM 审查结果的单条（用于结构化输出）"""
    memory_id: int = Field(description="被审查的记忆 ID")
    action: Literal["DELETE", "KEEP"] = Field(description="处置结论")
    reason: str = Field(description="简短原因")


class MemoryReviewResult(BaseModel):
    """LLM 审查结果列表（用于结构化输出）"""
    items: List[MemoryReviewItem] = Field(default_factory=list)


@dataclass
class MemoryItem:
    """内存中的记忆数据结构"""
    id: Optional[int] = None
    user_id: str = None
    memory_type: str = None
    content: str = None
    metadata: Dict[str, Any] = None
    importance: int = 5
    access_count: int = 0
    created_at: str = None
    updated_at: str = None
    last_accessed_at: str = None
    is_active: bool = True


# ==================== 遗忘策略参数（可通过环境变量调整）====================

# 有效重要度低于此阈值时软删除
FORGET_THRESHOLD: float = float(os.getenv("MEMORY_FORGET_THRESHOLD", "1.5"))
# 时间衰减半衰期（天）：半衰期越短，遗忘越快
DECAY_HALF_LIFE_DAYS: float = float(os.getenv("MEMORY_DECAY_HALF_LIFE_DAYS", "30"))
# 访问频率加成上限（每次访问 +0.3，最多 +ACCESS_BONUS_CAP）
ACCESS_BONUS_CAP: float = float(os.getenv("MEMORY_ACCESS_BONUS_CAP", "2.0"))
# 每 N 次对话触发一次 LLM 审查（设为 0 禁用）
LLM_REVIEW_EVERY: int = int(os.getenv("MEMORY_LLM_REVIEW_EVERY", "20"))


def compute_effective_importance(
    importance: int, updated_at: str, access_count: int
) -> float:
    """计算记忆的有效重要度（考虑时间衰减 + 访问加成）

    公式：有效重要度 = 基础重要度 × exp(-days / 半衰期×1.44) + 访问加成
    - 时间衰减：指数衰减，距上次更新越久衰减越多
    - 访问加成：每次被检索 +0.3，上限 ACCESS_BONUS_CAP
    """
    try:
        last_active = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
    except Exception:
        last_active = datetime.now()

    days_inactive = max(0, (datetime.now() - last_active).days)
    # 指数衰减系数（ln2 ≈ 0.693，half_life × 1.44 = half_life / ln2）
    decay_factor = math.exp(-days_inactive * math.log(2) / max(DECAY_HALF_LIFE_DAYS, 1))
    time_decayed = importance * decay_factor
    access_bonus = min(access_count * 0.3, ACCESS_BONUS_CAP)
    return time_decayed + access_bonus


# ==================== 长期记忆管理器 ====================

class LongTermMemory:
    """长期记忆管理器

    PostgreSQL 是主存储；Milvus 是可选的语义索引，不可用时自动降级为关键词匹配。
    """

    def __init__(
        self,
        db_path: str,
        tenant_id: Optional[str] = None,
        embeddings=None,
    ):
        self.db_path = db_path
        self.tenant_id = tenant_id
        self.embeddings = embeddings
        self._milvus_connected = False
        self._collection = None

        self._init_db()
        self._init_milvus()
        self._init_llm_tools()

        logger.info(
            f"长期记忆模块初始化完成: postgres={db_path}, milvus={self._milvus_connected}"
        )

    # ---------- 初始化 ----------

    def _init_llm_tools(self):
        """初始化 LLM 结构化输出工具（提取、更新决策、审查）"""
        try:
            llm = config.get_llm()
            self.memory_extractor = llm.with_structured_output(ExtractedMemoryList)
            self.memory_updater   = llm.with_structured_output(MemoryUpdateOperation)
            self.memory_reviewer  = llm.with_structured_output(MemoryReviewResult)
        except Exception as e:
            logger.warning(f"LLM 工具初始化失败（记忆模块降级运行）: {e}")
            self.memory_extractor = None
            self.memory_updater   = None
            self.memory_reviewer  = None

    def _init_db(self):
        """初始化 PostgreSQL 表结构（支持平滑升级旧表）"""
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()

            # 记忆主表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id               BIGSERIAL PRIMARY KEY,
                    user_id          TEXT    NOT NULL,
                    memory_type      TEXT    NOT NULL,
                    content          TEXT    NOT NULL,
                    metadata         TEXT,
                    importance       INTEGER DEFAULT 5,
                    access_count     INTEGER DEFAULT 0,
                    created_at       TEXT    NOT NULL,
                    updated_at       TEXT    NOT NULL,
                    last_accessed_at TEXT,
                    is_active        INTEGER DEFAULT 1
                )
            """)

            # 平滑升级：兼容旧版缺失字段
            for col, col_def in [
                ("access_count",     "INTEGER DEFAULT 0"),
                ("last_accessed_at", "TEXT"),
            ]:
                savepoint_name = f"sp_add_col_{col}"
                try:
                    # 使用保存点隔离可忽略异常，避免整笔事务进入 aborted 状态
                    cursor.execute(f"SAVEPOINT {savepoint_name}")
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_def}")
                    cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except Exception:
                    # 列已存在等场景可忽略，但必须回滚到保存点恢复事务可用性
                    cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    logger.debug(f"memory 表字段已存在或无需升级，跳过: {col}")

            # 用户元数据表（记录对话次数、上次 LLM 审查时间）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_meta (
                    user_id              TEXT PRIMARY KEY,
                    conversation_count   INTEGER DEFAULT 0,
                    last_llm_review_at   TEXT,
                    updated_at           TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id, memory_type, is_active)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(user_id, importance DESC)
            """)
            conn.commit()

    def ensure_db_schema(self):
        """确保 PostgreSQL 表结构完整（含 memory_meta）。

        用于修复：旧版代码未建 memory_meta、或外部拷贝的仅含 memories 的库文件，
        以及热重载后单例未重建等场景导致的 no such table。
        """
        self._init_db()

    def _init_milvus(self):
        """初始化 Milvus 连接（失败时静默降级）"""
        if not MILVUS_AVAILABLE:
            return
        if self.embeddings is None:
            try:
                self.embeddings = config.get_embeddings()
            except Exception as e:
                logger.warning(f"嵌入模型获取失败，Milvus 不可用: {e}")
                return
        try:
            connections.connect(
                alias="memory_default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                token=config.MILVUS_TOKEN,
            )
            self._milvus_connected = True
            self._init_milvus_collection()
            logger.info(f"Milvus 连接成功: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        except Exception as e:
            logger.warning(f"Milvus 连接失败，记忆将使用关键词检索: {e}")

    def _memory_milvus_collection_compatible(self, coll: Collection) -> bool:
        """判断集合是否为「长期记忆」专用 schema，而非 RAG 混合检索等其它集合。

        若 MEMORY_COLLECTION_NAME 与 RAG 同名或误用旧集合，会出现 md5/text/sparse_vector
        等字段，此时 insert 列数不匹配（如 expect 3 list, got 6）。
        """
        try:
            field_list = list(coll.schema.fields)
            names = {f.name for f in field_list}
        except Exception:
            return False

        required = {
            "memory_id",
            "user_id",
            "memory_type",
            "dense_vector",
            "importance",
            "created_at",
        }
        if not required.issubset(names):
            logger.warning(
                "Milvus 记忆集合字段不完整: 需要 %s，实际 %s",
                sorted(required),
                sorted(names),
            )
            return False

        # 误绑 RAG Hybrid（BM25 + dense）集合时的典型字段
        rag_markers = {"md5", "text", "sparse_vector"}
        if names & rag_markers:
            logger.warning(
                "当前 Milvus 集合疑似 RAG 知识库 schema（含 %s），不能用作长期记忆",
                sorted(names & rag_markers),
            )
            return False

        try:
            want_dim = int(config.EMBEDDING_DIM)
        except (TypeError, ValueError):
            want_dim = 1024
        for f in field_list:
            if f.name != "dense_vector" or f.dtype != DataType.FLOAT_VECTOR:
                continue
            dim = None
            try:
                if getattr(f, "params", None):
                    dim = f.params.get("dim")
            except Exception:
                pass
            if dim is None:
                logger.warning("dense_vector 缺少 dim 元数据")
                return False
            try:
                if int(dim) != want_dim:
                    logger.warning(
                        "dense_vector 维度 %s 与 EMBEDDING_DIM=%s 不一致",
                        dim,
                        want_dim,
                    )
                    return False
            except (TypeError, ValueError):
                return False
            return True
        return False

    def _create_memory_milvus_collection(self, collection_name: str) -> None:
        """创建长期记忆向量集合（与 RAG 集合 schema 完全独立）。"""
        fields = [
            FieldSchema(name="memory_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
            FieldSchema(name="importance", dtype=DataType.INT32),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields=fields, description="长期记忆向量索引")
        self._collection = Collection(
            name=collection_name, schema=schema, using="memory_default"
        )
        self._collection.create_index(
            field_name="dense_vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "ef_construction": 200},
            },
        )
        self._collection.load()

    def _init_milvus_collection(self):
        """初始化或加载 Milvus 记忆集合；若已存在集合为 RAG 等错误 schema 则删建。"""
        collection_name = config.get_memory_collection_name(self.tenant_id)
        logger.info(f"初始化或加载 Milvus 记忆集合: {collection_name}")
        try:
            if utility.has_collection(collection_name, using="memory_default"):
                coll = Collection(collection_name, using="memory_default")
                coll.load()
                if self._memory_milvus_collection_compatible(coll):
                    self._collection = coll
                    logger.info("已加载兼容的长期记忆 Milvus 集合: %s", collection_name)
                    return
                logger.warning(
                    "Milvus 集合 %s 与长期记忆 schema 不兼容（常见于与 RAG 集合同名或误用），"
                    "将删除后重建；PostgreSQL 记忆仍保留。",
                    collection_name,
                )
                utility.drop_collection(collection_name, using="memory_default")

            self._create_memory_milvus_collection(collection_name)
            logger.info(f"Milvus 记忆集合初始化完成: {collection_name}")
        except Exception as e:
            logger.warning(f"Milvus 集合初始化失败: {e}")
            self._milvus_connected = False

    # ---------- 工具方法 ----------

    def _get_now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _row_to_item(self, row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            importance=row["importance"],
            access_count=row["access_count"] or 0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            is_active=bool(row["is_active"]),
        )

    # ==================== CRUD ====================

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: int = 5,
    ) -> int:
        """新增一条记忆（同时写入 PostgreSQL 和 Milvus）"""
        now = self._get_now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories
                  (user_id, memory_type, content, metadata, importance,
                   access_count, created_at, updated_at, is_active)
                VALUES (%s, %s, %s, %s, %s, 0, %s, %s, 1)
                RETURNING id
                """,
                (user_id, memory_type, content, metadata_json, importance, now, now),
            )
            conn.commit()
            memory_id = cursor.fetchone()[0] if cursor.description else None

        self._milvus_add(memory_id, user_id, memory_type, content, importance, now)
        logger.info(f"添加记忆: id={memory_id}, user={user_id}, type={memory_type}, content={content}")
        return memory_id

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[int] = None,
    ) -> bool:
        """更新记忆内容（同时更新 Milvus）"""
        now = self._get_now()
        updates, params = [], []
        if content is not None:
            updates.append("content = %s");   params.append(content)
        if metadata is not None:
            updates.append("metadata = %s");  params.append(json.dumps(metadata, ensure_ascii=False))
        if importance is not None:
            updates.append("importance = %s"); params.append(importance)
        if not updates:
            return False
        updates.append("updated_at = %s"); params.append(now)
        params.append(memory_id)

        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = %s", params
            )
            conn.commit()
            success = cursor.rowcount > 0

        if success:
            mem = self.get_memory_by_id(memory_id)
            if mem:
                self._milvus_update(mem)
        return success

    def delete_memory(self, memory_id: int) -> bool:
        """软删除（将 is_active 置为 0，同时从 Milvus 移除）"""
        now = self._get_now()
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET is_active = 0, updated_at = %s WHERE id = %s",
                (now, memory_id),
            )
            conn.commit()
            success = cursor.rowcount > 0
        if success:
            self._milvus_delete(memory_id)
        return success

    def get_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        min_importance: int = 0,
        limit: int = 30,
    ) -> List[MemoryItem]:
        """按用户 ID 查询记忆（PostgreSQL）"""
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor(row_factory=dict_row)
            sql = "SELECT * FROM memories WHERE user_id = %s AND is_active = 1 AND importance >= %s"
            params: list = [user_id, min_importance]
            if memory_type:
                sql += " AND memory_type = %s"; params.append(memory_type)
            sql += " ORDER BY importance DESC, updated_at DESC LIMIT %s"
            params.append(limit)
            cursor.execute(sql, params)
            return [self._row_to_item(row) for row in cursor.fetchall()]

    def get_memory_by_id(self, memory_id: int) -> Optional[MemoryItem]:
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor(row_factory=dict_row)
            cursor.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
            row = cursor.fetchone()
            return self._row_to_item(row) if row else None

    async def update_access_stats(self, memory_ids: List[int]):
        """记录记忆被访问（+1 access_count，更新 last_accessed_at）"""
        if not memory_ids:
            return
        now = self._get_now()
        placeholders = ",".join(["%s"] * len(memory_ids))
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE memories SET access_count = access_count + 1, "
                f"last_accessed_at = %s WHERE id IN ({placeholders})",
                [now] + list(memory_ids),
            )
            conn.commit()

    # ==================== 遗忘策略 ====================

    def apply_time_decay_and_cleanup(self, user_id: Optional[str] = None):
        """时间衰减清理：计算所有记忆的有效重要度，低于阈值的执行软删除

        该操作轻量，建议每次对话后都执行（仅针对当前用户）。
        """
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor(row_factory=dict_row)
            sql = "SELECT * FROM memories WHERE is_active = 1"
            params: list = []
            if user_id:
                sql += " AND user_id = %s"; params.append(user_id)
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        to_delete = []
        for row in rows:
            eff_imp = compute_effective_importance(
                row["importance"],
                row["updated_at"] or row["created_at"],
                row["access_count"] or 0,
            )
            if eff_imp < FORGET_THRESHOLD:
                to_delete.append(row["id"])

        if to_delete:
            now = self._get_now()
            placeholders = ",".join(["%s"] * len(to_delete))
            with postgres.get_conn(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE memories SET is_active = 0, updated_at = %s "
                    f"WHERE id IN ({placeholders})",
                    [now] + to_delete,
                )
                conn.commit()
            for mid in to_delete:
                self._milvus_delete(mid)
            logger.info(
                f"时间衰减清理: user_id={user_id}, 软删除 {len(to_delete)} 条低重要度记忆"
            )

    def increment_conversation_count(self, user_id: str) -> int:
        """递增对话计数，返回更新后的计数值"""
        self.ensure_db_schema()
        now = self._get_now()
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            # 查询是否存在
            cursor.execute(
                "SELECT conversation_count FROM memory_meta WHERE user_id = %s", (user_id,)
            )
            row = cursor.fetchone()
            if row:
                new_count = row[0] + 1
                cursor.execute(
                    "UPDATE memory_meta SET conversation_count = %s, updated_at = %s WHERE user_id = %s",
                    (new_count, now, user_id),
                )
            else:
                new_count = 1
                cursor.execute(
                    "INSERT INTO memory_meta(user_id, conversation_count, updated_at) VALUES (%s, 1, %s)",
                    (user_id, now),
                )
            conn.commit()
        return new_count

    def should_llm_review(self, user_id: str) -> bool:
        """是否应触发 LLM 审查（每 LLM_REVIEW_EVERY 次对话触发一次）"""
        if LLM_REVIEW_EVERY <= 0:
            return False
        self.ensure_db_schema()
        with postgres.get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT conversation_count FROM memory_meta WHERE user_id = %s", (user_id,)
            )
            row = cursor.fetchone()
            count = row[0] if row else 0
        return count > 0 and count % LLM_REVIEW_EVERY == 0

    async def llm_review_and_cleanup(self, user_id: str):
        """LLM 智能审查：识别过时/空洞/冲突记忆并软删除（重量级，异步）"""
        if self.memory_reviewer is None:
            return
        memories = self.get_memories(user_id, limit=60)
        if not memories:
            return

        try:
            from core.prompts import MEMORY_REVIEW_PROMPT

            memories_str = "\n".join(
                f"ID={m.id} [{m.memory_type}] 重要度={m.importance} "
                f"访问={m.access_count}次 更新于={m.updated_at}: {m.content}"
                for m in memories
            )
            prompt = MEMORY_REVIEW_PROMPT.format(memories=memories_str)
            result = await self.memory_reviewer.ainvoke(prompt)

            deleted_count = 0
            for item in result.items or []:
                if item.action == "DELETE":
                    self.delete_memory(item.memory_id)
                    deleted_count += 1

            if deleted_count:
                self.ensure_db_schema()
                now = self._get_now()
                with postgres.get_conn(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE memory_meta SET last_llm_review_at = %s, updated_at = %s "
                        "WHERE user_id = %s",
                        (now, now, user_id),
                    )
                    conn.commit()
                logger.info(
                    f"LLM 记忆审查完成: user_id={user_id}, 删除 {deleted_count} 条"
                )
        except Exception as e:
            logger.warning(f"LLM 记忆审查失败: user_id={user_id}, error={e}")

    # ==================== 记忆检索 ====================

    def get_relevant_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[MemoryItem]:
        """检索与当前 query 最相关的记忆

        优先使用 Milvus 语义检索；Milvus 不可用时降级为关键词匹配。
        检索到的记忆会自动更新访问统计。
        """
        import time as _time

        memories: List[MemoryItem] = []

        if self._milvus_connected and self._collection is not None and self.embeddings is not None:
            try:
                _t_milvus = _time.perf_counter()
                memories = self._milvus_search(user_id, query, limit)
                logger.info(f"[长期记忆耗时]   Milvus 语义检索(命中{len(memories)}条): {_time.perf_counter() - _t_milvus:.4f}s")
            except Exception as e:
                logger.warning(f"Milvus 语义检索失败，降级关键词: {e}")

        if not memories:
            _t_kw = _time.perf_counter()
            memories = self._keyword_search(user_id, query, limit)
            logger.info(f"[长期记忆耗时]   关键词检索(命中{len(memories)}条): {_time.perf_counter() - _t_kw:.4f}s")

        if memories:
            asyncio.create_task(self.update_access_stats([m.id for m in memories if m.id]))
        return memories

    def _milvus_search(self, user_id: str, query: str, limit: int) -> List[MemoryItem]:
        import time as _time

        _t0 = _time.perf_counter()
        query_vec = self.embeddings.embed_query(query)
        _t1 = _time.perf_counter()
        logger.info(f"[长期记忆耗时]     Embedding 向量化: {_t1 - _t0:.4f}s")

        results = self._collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=f'user_id == "{user_id}"',
            output_fields=["memory_id"],
        )
        _t2 = _time.perf_counter()
        logger.info(f"[长期记忆耗时]     Milvus 向量搜索: {_t2 - _t1:.4f}s")

        ids = [hit.id for hit in results[0]]
        mems = [self.get_memory_by_id(mid) for mid in ids]
        _t3 = _time.perf_counter()
        logger.info(f"[长期记忆耗时]     PG 加载记忆详情({len(ids)}条): {_t3 - _t2:.4f}s")

        return [m for m in mems if m and m.is_active]

    def _keyword_search(self, user_id: str, query: str, limit: int) -> List[MemoryItem]:
        import time as _time

        _t0 = _time.perf_counter()
        all_mems = self.get_memories(user_id, min_importance=1, limit=100)
        _t1 = _time.perf_counter()
        logger.info(f"[长期记忆耗时]     PG 加载全部记忆({len(all_mems)}条): {_t1 - _t0:.4f}s")

        keywords = [k.strip() for k in query.lower().split() if len(k.strip()) > 1]
        scored = []
        for mem in all_mems:
            score = mem.importance
            for kw in keywords:
                if kw in mem.content.lower():
                    score += 3
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        _t2 = _time.perf_counter()
        logger.info(f"[长期记忆耗时]     关键词评分排序: {_t2 - _t1:.4f}s")

        return [m for _, m in scored[:limit]]

    # ==================== 格式化输出 ====================

    def format_for_prompt(self, memories: List[MemoryItem]) -> str:
        """格式化为 prompt 可用的字符串"""
        if not memories:
            return ""
        lines = ["【用户长期记忆】"]
        for m in memories:
            lines.append(f"- [{m.memory_type}] {m.content}")
        return "\n".join(lines)

    def format_for_update_decision(self, memories: List[MemoryItem]) -> str:
        """格式化为更新决策用的字符串（带 ID 便于 LLM 指定操作目标）"""
        if not memories:
            return "（暂无现有记忆）"
        return "\n".join(
            f"ID={m.id} [{m.memory_type}] 重要度={m.importance}: {m.content}"
            for m in memories
        )

    # ==================== LLM 决策更新 ====================

    async def decide_memory_update(
        self,
        existing_memories: List[MemoryItem],
        new_item: ExtractedMemoryItem,
    ) -> MemoryUpdateOperation:
        """调用 LLM 决定对新记忆应执行 ADD / UPDATE / DELETE / NONE"""
        if self.memory_updater is None:
            # LLM 不可用时默认直接新增
            return MemoryUpdateOperation(
                operation="ADD",
                content=new_item.content,
                memory_type=new_item.memory_type,
                importance=new_item.importance,
            )
        try:
            from core.prompts import UPDATE_MEMORY_PROMPT

            existing_str = self.format_for_update_decision(existing_memories)
            new_str = (
                f"类型={new_item.memory_type}, 内容={new_item.content}, "
                f"重要度={new_item.importance}"
            )
            prompt = UPDATE_MEMORY_PROMPT.format(
                existing_memories=existing_str, new_memory=new_str
            )
            result = await self.memory_updater.ainvoke(prompt)
            logger.debug(f"记忆更新决策: {result}")
            return result
        except Exception as e:
            logger.warning(f"记忆更新决策 LLM 调用失败，默认 ADD: {e}")
            return MemoryUpdateOperation(
                operation="ADD",
                content=new_item.content,
                memory_type=new_item.memory_type,
                importance=new_item.importance,
            )

    async def update_memory_with_decision(
        self, user_id: str, decision: MemoryUpdateOperation
    ) -> Optional[int]:
        """执行 LLM 决策"""
        try:
            if decision.operation == "ADD":
                if not decision.content or not decision.memory_type:
                    return None
                return self.add_memory(
                    user_id,
                    decision.memory_type,
                    decision.content,
                    importance=decision.importance or 5,
                )
            elif decision.operation == "UPDATE":
                if not decision.memory_id:
                    return None
                self.update_memory(
                    decision.memory_id,
                    content=decision.content,
                    importance=decision.importance,
                )
                return decision.memory_id
            elif decision.operation == "DELETE":
                if not decision.memory_id:
                    return None
                self.delete_memory(decision.memory_id)
                return decision.memory_id
            elif decision.operation == "NONE":
                return None
        except Exception as e:
            logger.error(f"执行记忆更新操作失败: {e}")
        return None

    # ==================== Milvus 内部方法 ====================

    def _milvus_add(
        self,
        memory_id: int,
        user_id: str,
        memory_type: str,
        content: str,
        importance: int,
        created_at: str,
    ):
        if not self._milvus_connected or self._collection is None or self.embeddings is None:
            return
        try:
            embedding = self.embeddings.embed_query(content)
            self._collection.insert(
                [[memory_id], [user_id], [memory_type], [embedding], [importance], [created_at]]
            )
            self._collection.flush()
        except Exception as e:
            logger.warning(f"Milvus 记忆写入失败: {e}")

    def _milvus_update(self, mem: MemoryItem):
        if not self._milvus_connected or self._collection is None:
            return
        try:
            self._collection.delete(f"memory_id == {mem.id}")
            self._milvus_add(
                mem.id, mem.user_id, mem.memory_type,
                mem.content, mem.importance, mem.created_at,
            )
        except Exception as e:
            logger.warning(f"Milvus 记忆更新失败: {e}")

    def _milvus_delete(self, memory_id: int):
        if not self._milvus_connected or self._collection is None:
            return
        try:
            self._collection.delete(f"memory_id == {memory_id}")
            self._collection.flush()
        except Exception as e:
            logger.warning(f"Milvus 记忆删除失败: {e}")


# ==================== 单例工厂（按租户隔离）====================

_memory_instances: Dict[str, LongTermMemory] = {}
_memory_lock = threading.Lock()


def get_memory_manager(
    db_path: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> LongTermMemory:
    """获取长期记忆管理器（按 tenant_id 隔离的单例）"""
    key = tenant_id or "default"
    if key not in _memory_instances:
        with _memory_lock:
            if key not in _memory_instances:
                if db_path is None:
                    db_path = config.get_long_term_memory_db_path(tenant_id)
                _memory_instances[key] = LongTermMemory(
                    db_path=db_path, tenant_id=tenant_id
                )
    return _memory_instances[key]


# ==================== 对外异步接口 ====================

async def load_relevant_memories(
    user_id: str, tenant_id: str, query: str, limit: int = 5
) -> str:
    """为当前 query 加载最相关的长期记忆，返回格式化字符串（供 prompt 使用）"""
    try:
        import time as _time

        _t0 = _time.perf_counter()
        mem_manager = get_memory_manager(tenant_id=tenant_id)
        _t1 = _time.perf_counter()
        logger.info(f"[长期记忆耗时] 获取记忆管理器: {_t1 - _t0:.4f}s")

        memories = mem_manager.get_relevant_memories(user_id, query, limit=limit)
        _t2 = _time.perf_counter()
        logger.info(f"[长期记忆耗时] 检索相关记忆(共{len(memories)}条): {_t2 - _t1:.4f}s")

        result = mem_manager.format_for_prompt(memories)
        _t3 = _time.perf_counter()
        logger.info(f"[长期记忆耗时] 格式化输出: {_t3 - _t2:.4f}s")

        logger.info(f"[长期记忆耗时] 总耗时: {_t3 - _t0:.4f}s, user_id={user_id}")
        return result
    except Exception as e:
        logger.warning(f"加载长期记忆失败: user_id={user_id}, error={e}")
        return ""


async def extract_and_save_memory(
    user_id: str, tenant_id: str, query: str, answer: str
):
    """从对话中异步提取并智能保存长期记忆

    流程：
    1. LLM 从 Q&A 中提取记忆项
    2. 对每条记忆，LLM 决策 ADD / UPDATE / DELETE / NONE（避免冲突和重复）
    3. 时间衰减清理（每次都执行，轻量）
    4. LLM 审查清理（每 LLM_REVIEW_EVERY 次对话执行一次，重量级）
    """
    try:
        from core.prompts import EXTRACT_MEMORY_PROMPT

        mem_manager = get_memory_manager(tenant_id=tenant_id)
        # 确保含 memory_meta 在内的表已就绪（兼容旧库/仅含 memories 的文件）
        mem_manager.ensure_db_schema()
        if mem_manager.memory_extractor is None:
            logger.warning("记忆提取器不可用，跳过记忆保存")
            return

        # Step 1：提取记忆
        extract_prompt = EXTRACT_MEMORY_PROMPT.format(query=query, answer=answer)
        try:
            result = await mem_manager.memory_extractor.ainvoke(extract_prompt)
            memories_to_save = result.memories if result else []
        except Exception as e:
            logger.warning(f"记忆提取 LLM 调用失败: {e}")
            return

        if not memories_to_save:
            logger.info(f"本轮对话无值得保存的记忆: user_id={user_id}")
        else:
            # Step 2：获取现有记忆，逐条决策更新
            existing_memories = mem_manager.get_memories(user_id, limit=50)
            stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}

            for mem_item in memories_to_save:
                if isinstance(mem_item, dict):
                    mem_item = ExtractedMemoryItem(**mem_item)
                decision = await mem_manager.decide_memory_update(existing_memories, mem_item)
                await mem_manager.update_memory_with_decision(user_id, decision)
                stats[decision.operation] = stats.get(decision.operation, 0) + 1

            logger.info(
                f"记忆更新完成: user_id={user_id}, stats={stats}"
            )

        # Step 3：对话计数 + 时间衰减清理（轻量，每轮都执行）
        conv_count = mem_manager.increment_conversation_count(user_id)
        mem_manager.apply_time_decay_and_cleanup(user_id)

        # Step 4：LLM 审查（每 N 次对话，重量级，异步）
        if mem_manager.should_llm_review(user_id):
            logger.info(f"触发 LLM 记忆审查: user_id={user_id}, conv_count={conv_count}")
            await mem_manager.llm_review_and_cleanup(user_id)
        
        logger.info(f"长期记忆提取和保存完成: user_id={user_id}")

    except Exception:
        logger.warning(f"extract_and_save_memory 异常: {traceback.format_exc()}")
