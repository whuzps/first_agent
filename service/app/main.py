"""FastAPI 应用入口

提供 `/chat` 对话接口与 `/health` 健康检查，支持请求 ID 中间件。
"""
# ── 标准库 ──────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, List, Optional

# ── 第三方依赖 ──────────────────────────────────────────────────────────────
import jwt as _jwt
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ── 兼容 `python3 service/app/main.py` 直接运行 ─────────────────────────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# `service/` 目录：包含 core/、graph/、tools/ 等包
_SERVICE_DIR = os.path.dirname(_CURRENT_DIR)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

# 注意：必须在 sys.path 注入完成后，才能导入 core/*
import core.config as config
from core.config import BASE_DIR, LOG_DIR
from core.logging_config import setup_logging
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── 加载本地 MCP 服务 ─────────────────────────────────────────────────────────
_MCP_SERVER_PATH = os.path.join(str(BASE_DIR), "mcp_server", "server.py")
_MCP_SPEC = importlib.util.spec_from_file_location("local_mcp_server", _MCP_SERVER_PATH)
if _MCP_SPEC is None or _MCP_SPEC.loader is None:
    raise RuntimeError(f"无法加载本地MCP服务模块: {_MCP_SERVER_PATH}")
_MCP_MODULE = importlib.util.module_from_spec(_MCP_SPEC)
_MCP_SPEC.loader.exec_module(_MCP_MODULE)
_mcp = _MCP_MODULE.mcp

# ── 项目内部模块 ─────────────────────────────────────────────────────────────
from core.hander import determine_answer, handle_command
import memory.store as memory_module
import core.postgres as postgres
from core.observability import (
    start_prometheus,
    TTFT_DURATION,
    CHAT_STREAM_REQUESTS_TOTAL,
    CHAT_STREAM_ACTIVE_REQUESTS,
    CHAT_STREAM_MESSAGES_TOTAL,
    CHAT_STREAM_CONSUME_DURATION,
)
from core.preprocessing import transcribe_audio
from core.suggest import gen_suggest_questions
import graph.graph as graph
from graph.graph import run_graph, run_graph_stream, resume_graph, warmup_graph_templates
import core.mq as _mq
import app.gradio_ui as gradio_ui
from auth.security_middleware import RedactionMiddleware, build_default_config, sanitize_dict
from tools.service_tools import getdb
from core.state import State
from core.token_bucket_rate_limit import get_remote_address, token_bucket_limit

# ── 全局常量 ─────────────────────────────────────────────────────────────────
_REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
_JWT_SECRET = os.environ.get("JWT_SECRET", "cs-agent-jwt-secret-2025")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_DAYS = 7
_KB_UPLOADS_DIR = os.path.join(BASE_DIR, "data/uploads")
_KB_ALLOWED_EXT = {".txt", ".md", ".pdf", ".docx"}
os.makedirs(_KB_UPLOADS_DIR, exist_ok=True)

# ── 链路追踪 ─────────────────────────────────────────────────────────────────
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def get_trace_id() -> str:
    return _trace_id_var.get()


class _TraceIdFilter(logging.Filter):
    """向每条日志记录注入当前请求的 trace_id。"""
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()  # type: ignore[attr-defined]
        return True


# 使用统一日志配置模块，携带请求链路追踪格式
setup_logging(with_trace=True, trace_filter=_TraceIdFilter())

# 模块级 logger
logger = logging.getLogger(__name__)

# ── 全局可变状态 ──────────────────────────────────────────────────────────────
SUGGEST_QUEUES: dict = {}
VECTORS_LOCK: asyncio.Lock  # 在应用初始化后赋值
_chain = None


# =========================================================================
# ── Pydantic 请求/响应模型 ─────────────────────────────────────────────────────
# =========================================================================

class ChatRequest(BaseModel):
    query: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    images: Optional[list[str]] = None
    audio: Optional[str] = None
    asr_language: Optional[str] = None
    asr_itn: Optional[bool] = None
    quoted_message: Optional[str] = None

class SwitchRequest(BaseModel):
    name: str

class VectorItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

class VectorsAddRequest(BaseModel):
    items: List[VectorItem]

class VectorsDeleteRequest(BaseModel):
    ids: List[str]

class MilvusCollectionDeleteRequest(BaseModel):
    collection_name: str

class MilvusFileIngestRequest(BaseModel):
    file_path: str
    collection_name: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"
    tenant_id: str = "default"

class PublicSignupRequest(BaseModel):
    username: str
    password: str

class RenameSessionRequest(BaseModel):
    title: str

class EvaluationCaseRequest(BaseModel):
    name: str
    query: str
    ground_truth: Optional[str] = None

class BatchEvaluationRequest(BaseModel):
    cases: List[EvaluationCaseRequest]
    faith_threshold: Optional[float] = 0.85
    relev_threshold: Optional[float] = 0.7

class HitlConfirmRequest(BaseModel):
    """HITL 高危操作确认请求"""
    thread_id: str
    decision: str  # "approved" 或 "rejected"

class SingleEvaluationRequest(BaseModel):
    case: EvaluationCaseRequest
    answer: str
    route: Optional[str] = None


# =========================================================================
# ── 工具函数 ──────────────────────────────────────────────────────────────────
# =========================================================================

def _ok(data: Any) -> Dict[str, Any]:
    return {"code": 0, "message": "OK", "data": data}

def _err(code: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"code": code, "message": message, "data": data or {}}

def _stable_id_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _audit(op: str, data: Dict[str, Any]):
    try:
        logs = LOG_DIR
        os.makedirs(logs, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        line = f"{ts} | INFO | audit | {op} " + json.dumps(sanitize_dict(data), ensure_ascii=False)
        with open(os.path.join(logs, "requests.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── 数据库 ───────────────────────────────────────────────────────────────────

def _pg_dsn() -> str:
    return config.get_postgres_dsn("default")

# ── JWT 认证 ─────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)

def _make_token(user_id: int, username: str, role: str, tenant_id: str) -> str:
    payload = {
        "sub": str(user_id), "username": username, "role": role, "tenant_id": tenant_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=_JWT_EXPIRE_DAYS),
    }
    return _jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)

def _decode_token(token: str) -> dict:
    try:
        return _jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的 Token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    return _decode_token(credentials.credentials)


# ── 用户身份提取 ──────────────────────────────────────────────────────────────

def _canonical_user_id_from_token_payload(payload: Optional[dict]) -> str:
    """从 JWT payload 解析业务用户标识（优先 sub，其次 username）"""
    if not payload:
        return "default"
    sub = payload.get("sub")
    if sub is not None and str(sub).strip():
        return str(sub).strip()
    un = payload.get("username")
    return str(un).strip() if un and str(un).strip() else "default"

def _get_user_id_from_request(request: Request) -> str:
    """从请求提取 user_id（JWT sub → X-User-ID → query user → default）"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = _jwt.decode(auth[7:], _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
            uid = _canonical_user_id_from_token_payload(payload)
            if uid != "default":
                return uid
        except Exception:
            pass
    return request.headers.get("X-User-ID") or request.query_params.get("user") or "default"


# ── 限流 key 函数 ─────────────────────────────────────────────────────────────

def get_user_id_or_ip(request: Request) -> str:
    """限流 key：JWT 账号 → X-User-ID → X-Forwarded-For → remote address"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = _jwt.decode(auth[7:], _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
            uid = _canonical_user_id_from_token_payload(payload)
            if uid != "default":
                return f"user:{uid}"
        except Exception:
            pass
    user_id = request.headers.get("X-User-ID") or request.query_params.get("user")
    if user_id:
        return f"user:{user_id}"
    xff = request.headers.get("X-Forwarded-For")
    return f"ip:{xff.split(',')[0].strip()}" if xff else f"ip:{get_remote_address(request)}"

def get_tenant_id_key(request: Request) -> str:
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    return f"tenant:{tenant_id}"


# ── 流式监控辅助 ──────────────────────────────────────────────────────────────

def _record_stream_metric(status: str, elapsed: float, task_id: str = ""):
    """记录流式消息消费指标（成功/失败 + 端到端耗时）"""
    if CHAT_STREAM_MESSAGES_TOTAL is not None:
        CHAT_STREAM_MESSAGES_TOTAL.labels(status=status).inc()
    if CHAT_STREAM_CONSUME_DURATION is not None:
        CHAT_STREAM_CONSUME_DURATION.observe(elapsed)
    verb = "成功" if status == "success" else "失败"
    logger.info(f"[监控] 消息消费{verb}, 耗时={elapsed*1000:.2f}ms, task={task_id}")

def _record_ttft(ttft_start: float, label: str) -> float:
    """上报首字响应时间（TTFT），返回耗时秒数"""
    ttft_seconds = time.perf_counter() - ttft_start
    logger.info(f"📊 TTFT（{label}）: {ttft_seconds*1000:.2f}ms")
    if TTFT_DURATION is not None:
        TTFT_DURATION.observe(ttft_seconds)
    return ttft_seconds


# ── 建议问题推送 ──────────────────────────────────────────────────────────────

async def push_suggest(thread_id: str, query: Optional[str], answer: str):
    """异步推送建议问题：先发 react_start，再生成建议，失败发 error"""
    try:
        q = SUGGEST_QUEUES.setdefault(thread_id, asyncio.Queue())
        await asyncio.sleep(0.05)
        await q.put({"suggestions": [], "event": "react_start"})
        try:
            suggestions = await gen_suggest_questions(thread_id, query, answer)
            await q.put({"suggestions": suggestions, "event": "react", "final": True})
        except Exception:
            await q.put({"error": {"code": "react_error", "message": "建议生成异常"}, "event": "error", "final": True})
    except Exception:
        pass


# ── 对话后处理（写历史 / 更新会话 / 触发摘要 / 提取记忆 / 推送建议）────────────

def _post_process_tasks(thread_id: str, user_id: str, tenant_id: str, query: str, answer: str):
    """批量启动异步后处理任务，不阻塞主流程"""

    async def _run():
        if query:
            await config.append_session_message(thread_id, "user", query)
        if answer:
            await config.append_session_message(thread_id, "assistant", answer)
        if query:
            title = (query[:20] + "…") if len(query) > 20 else query
            msg_count = len(await config.get_all_session_messages(thread_id)) + 2
            asyncio.create_task(asyncio.to_thread(_upsert_session, thread_id, user_id, title, msg_count))
        if query and answer:
            await config.maybe_trigger_summary_update(thread_id, user_id)
            asyncio.create_task(memory_module.extract_and_save_memory(user_id, tenant_id, query, answer))
        asyncio.create_task(push_suggest(thread_id, query, answer))

    asyncio.create_task(_run())


# =========================================================================
# ── 数据库初始化 & 操作 ────────────────────────────────────────────────────────
# =========================================================================

def _init_users_db():
    """初始化用户表，写入默认管理员账号"""
    dsn = _pg_dsn()
    with postgres.get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         BIGSERIAL PRIMARY KEY,
                    username   TEXT   NOT NULL UNIQUE,
                    password   TEXT   NOT NULL,
                    role       TEXT   NOT NULL DEFAULT 'user',
                    tenant_id  TEXT   NOT NULL DEFAULT 'default',
                    created_at BIGINT NOT NULL
                )
            """)
            default_pw = hashlib.sha256("admin123".encode()).hexdigest()
            cur.execute(
                "INSERT INTO users (username, password, role, tenant_id, created_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                ("admin", default_pw, "admin", "default", int(time.time())),
            )
        conn.commit()


def _init_kb_tasks_db():
    """初始化知识库解析任务表"""
    dsn = _pg_dsn()
    with postgres.get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kb_tasks (
                    id              TEXT    PRIMARY KEY,
                    user_id         TEXT    NOT NULL DEFAULT 'default',
                    filename        TEXT    NOT NULL,
                    file_size       INTEGER DEFAULT 0,
                    file_path       TEXT    NOT NULL,
                    collection_name TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'pending',
                    error           TEXT,
                    created_at      INTEGER NOT NULL,
                    updated_at      INTEGER NOT NULL,
                    chunk_count     INTEGER DEFAULT 0
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_tasks_user ON kb_tasks(user_id, created_at DESC)")
        conn.commit()


def _init_sessions_registry_db():
    """初始化会话注册表"""
    dsn = _pg_dsn()
    with postgres.get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id     TEXT    PRIMARY KEY,
                    user_id       TEXT    NOT NULL,
                    title         TEXT    NOT NULL DEFAULT '新对话',
                    created_at    INTEGER NOT NULL,
                    updated_at    INTEGER NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC)")
        conn.commit()


_init_users_db()
_init_kb_tasks_db()
_init_sessions_registry_db()

_KB_TASKS_DB = _pg_dsn()
_SESSIONS_REGISTRY_DB = _pg_dsn()


def _upsert_session(thread_id: str, user_id: str, title: str = "新对话", message_count: int = 0):
    """注册或更新会话元信息（title 仅首次写入）"""
    now = int(time.time())
    postgres.execute(_pg_dsn(),
        """INSERT INTO sessions (thread_id, user_id, title, created_at, updated_at, message_count)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at, message_count=excluded.message_count""",
        (thread_id, user_id, title, now, now, message_count),
    )


def _delete_session_registry(thread_id: str, user_id: str) -> bool:
    with postgres.get_conn(_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE thread_id=%s AND user_id=%s", (thread_id, user_id))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def _update_session_title(thread_id: str, user_id: str, title: str) -> bool:
    with postgres.get_conn(_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE sessions SET title=%s WHERE thread_id=%s AND user_id=%s", (title, thread_id, user_id))
            updated = cur.rowcount > 0
        conn.commit()
    return updated


# =========================================================================
# ── 知识库入库工作进程（必须在模块顶层，供 ProcessPoolExecutor pickle）────────────
# =========================================================================

_KB_EXECUTOR = ProcessPoolExecutor(max_workers=2)


def _kb_ingestion_worker(task_id: str, file_path: str, collection_name: str, db_path: str, service_dir: str) -> dict:
    """在独立进程中执行知识库文档入库，直接更新 PostgreSQL 任务状态"""
    import sys, time as _time, traceback as _tb, logging as _log
    if service_dir not in sys.path:
        sys.path.insert(0, service_dir)
    import core.postgres as _postgres
    # 子进程独立初始化统一日志（ProcessPoolExecutor 不继承父进程 handler）
    from core.logging_config import setup_logging as _setup_logging
    _setup_logging()
    logger = _log.getLogger("kb_worker")

    def _set_status(status, error=None, chunk_count=0):
        _postgres.execute(db_path,
            "UPDATE kb_tasks SET status=%s, error=%s, updated_at=%s, chunk_count=%s WHERE id=%s",
            (status, error, int(_time.time()), chunk_count, task_id),
        )

    try:
        _set_status("processing")
        logger.info(f"[kb_worker] 开始入库: task={task_id}, file={file_path}")
        from rag.ingestion import Ingestion
        success = Ingestion().ingest_file(file_path, collection_name)
        _set_status("done" if success else "failed", error=None if success else "ingest_file 返回 False")
        logger.info(f"[kb_worker] 入库{'完成' if success else '失败'}: task={task_id}")
        return {"success": success, "task_id": task_id}
    except Exception as exc:
        _set_status("failed", error=_tb.format_exc()[-800:])
        logger.error(f"[kb_worker] 入库异常: task={task_id}, error={exc}")
        return {"success": False, "error": str(exc), "task_id": task_id}


async def _dispatch_kb_task(task_id: str, file_path: str, collection_name: str, service_dir: str):
    """在事件循环中把入库任务提交给进程池，失败时写回 DB"""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _KB_EXECUTOR, _kb_ingestion_worker,
            task_id, file_path, collection_name, _KB_TASKS_DB, service_dir,
        )
    except Exception as exc:
        logger.error(f"入库任务提交失败: task_id={task_id}, error={exc}")
        postgres.execute(_pg_dsn(),
            "UPDATE kb_tasks SET status='failed', error=%s, updated_at=%s WHERE id=%s",
            (str(exc)[:500], int(time.time()), task_id),
        )


# =========================================================================
# ── 服务连通性探测 ─────────────────────────────────────────────────────────────
# =========================================================================

def _check_milvus() -> dict:
    try:
        from pymilvus import MilvusClient
        uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
        client = MilvusClient(uri=uri, token=config.MILVUS_TOKEN or "")
        return {"ok": True, "uri": uri, "collections": len(client.list_collections())}
    except Exception as e:
        return {"ok": False, "uri": f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}", "error": str(e)[:200]}

async def _check_redis() -> dict:
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        r = await config.get_redis()
        if r is None:
            return {"ok": False, "url": url, "error": "连接失败或未配置"}
        await r.ping()
        return {"ok": True, "url": url}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)[:200]}

def _check_postgres() -> dict:
    dsn = _pg_dsn()
    try:
        with postgres.get_conn(dsn) as conn:
            conn.execute("SELECT 1")
        return {"ok": True, "dsn": dsn}
    except Exception as e:
        return {"ok": False, "dsn": dsn, "error": str(e)[:200]}


# =========================================================================
# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
# =========================================================================

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """应用生命周期：启动 Prometheus，探测依赖服务，预热图模板"""
    global _chain
    try:
        start_prometheus(port=9090)
        milvus_s, redis_s, postgres_s = await asyncio.gather(
            asyncio.to_thread(_check_milvus),
            _check_redis(),
            asyncio.to_thread(_check_postgres),
        )
        _tag = lambda s: "OK" if s.get("ok") else "FAIL"
        _detail = lambda s: s.get("error", "") if not s.get("ok") else ""
        logger.info(
            "========== 启动健康探测 ==========\n"
            "  Milvus   : [%s] %s %s\n"
            "  Redis    : [%s] %s %s\n"
            "  Postgres : [%s] %s %s\n"
            "===================================",
            _tag(milvus_s), milvus_s.get("uri", ""), _detail(milvus_s),
            _tag(redis_s), redis_s.get("url", ""), _detail(redis_s),
            _tag(postgres_s), postgres_s.get("dsn", ""), _detail(postgres_s),
        )
        _chain = warmup_graph_templates(force_rebuild=False)
        logger.info("服务启动预编译完成：graph 模板已加载到全局缓存")
        yield
    finally:
        # 服务关闭阶段统一释放 PostgreSQL 连接池，避免连接泄漏
        try:
            postgres.close_all_pools()
            logger.info("服务关闭：已优雅关闭所有 PostgreSQL 连接池")
        except Exception as e:
            logger.warning(f"服务关闭：关闭 PostgreSQL 连接池失败（非致命）: {e}")


app = FastAPI(lifespan=_lifespan)
app.add_middleware(RedactionMiddleware, config=build_default_config())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5174",
                   "http://localhost:5176", "http://localhost:5175", "http://localhost:5178"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
gradio_ui.mount_gradio(app)
try:
    app.mount("/mcp", _mcp.sse_app())
except Exception:
    pass
VECTORS_LOCK = asyncio.Lock()


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """链路追踪中间件：为每个请求生成唯一 trace_id，注入 ContextVar 并贯穿全链路日志"""
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    _token = _trace_id_var.set(trace_id)
    request.state.request_id = trace_id
    logger.info("→ %s %s | user=%s | tenant=%s",
        request.method, request.url.path,
        _get_user_id_from_request(request),
        request.headers.get("X-Tenant-ID", "default"),
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        logging.exception("请求处理异常: %s %s", request.method, request.url.path)
        raise exc
    finally:
        _trace_id_var.reset(_token)
    logger.info("← %s %s | status=%d | trace=%s",
        request.method, request.url.path, response.status_code, trace_id)
    response.headers["X-Request-Id"] = trace_id
    response.headers["X-Trace-Id"] = trace_id
    return response


def require_api_key(request: Request):
    key = "test"
    if not key or request.headers.get("X-API-Key") != key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _write_request_log_sync(term_line: str, req_str: str):
    logs = LOG_DIR
    os.makedirs(logs, exist_ok=True)
    with open(logs / "requests.log", "a", encoding="utf-8") as f:
        f.write(term_line + " | request=" + req_str + "\n")


def measure_latency(func):
    """计时与指标装饰器：统计耗时、更新路由分类指标、写请求日志"""
    @wraps(func)
    async def _wrap(*args, **kwargs):
        _start = time.perf_counter()
        result = await func(*args, **kwargs)
        if isinstance(result, dict) and "route" in result:
            _elapsed_ms = (time.perf_counter() - _start) * 1000.0
            _route = result.get("route")
            m = config.get_metrics()
            m.update("overall", _elapsed_ms)
            m.update({"faq": "kb", "order": "order", "human": "handoff"}.get(_route, "direct"), _elapsed_ms)
            logger.info("latency route=%s cost=%.2fms", _route, _elapsed_ms)
            try:
                _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                _term_line = f"{_ts} | INFO     | root | trace={get_trace_id()} | latency route={_route} cost={_elapsed_ms:.2f}ms"
                _req = args[0] if args else kwargs.get("req")
                _safe = sanitize_dict({"query": getattr(_req, "query", None) if _req else None})
                
                # 使用 asyncio 的默认线程池执行日志写入，避免高并发下线程创建风暴
                asyncio.get_running_loop().run_in_executor(
                    None, 
                    _write_request_log_sync, 
                    _term_line, 
                    json.dumps(_safe, ensure_ascii=False)
                )
            except Exception:
                pass
        return result
    return _wrap


# =========================================================================
# ── 对话接口 ──────────────────────────────────────────────────────────────────
# =========================================================================


async def _auto_reject_stale_hitl(thread_id: str, tenant_id: str) -> None:
    """前置检查：若指定 thread 有未处理的 HITL 挂起状态，自动拒绝并恢复图执行。

    场景：
      - 用户收到确认弹窗后未操作，直接发了新消息
      - HITL 超时后用户才回来（Redis key 已过期，但 checkpoint 中图仍 interrupted）
    两种情况都需要先把图从 interrupted 状态恢复（以 rejected 身份），
    才能正常处理新的用户输入。
    """
    pending = await _mq.get_hitl_pending(thread_id)
    if pending is None:
        return

    logger.info(
        f"[HITL自动拒绝] 检测到 thread_id={thread_id} 存在未处理的 HITL 挂起，"
        f"order_id={pending.get('order_id')}，自动拒绝并恢复图"
    )
    try:
        await resume_graph(
            thread_id=thread_id,
            tenant_id=tenant_id,
            user_decision="rejected",
        )
        logger.info(f"[HITL自动拒绝] 图已恢复: thread_id={thread_id}")
    except Exception as e:
        logger.warning(f"[HITL自动拒绝] 恢复图失败（非致命，将尝试正常处理）: {e}")
    finally:
        await _mq.clear_hitl_pending(thread_id)


@app.post("/chat")
@token_bucket_limit("60/minute", key_func=get_user_id_or_ip)
@token_bucket_limit("1000/hour", key_func=get_tenant_id_key)
@measure_latency
async def chat(req: ChatRequest, request: Request):
    """同步对话接口：调用图执行并返回完整答案"""
    thread_id = req.thread_id or request.state.request_id
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    user_id = _get_user_id_from_request(request)
    audio_text = transcribe_audio(req.audio, req.asr_language, True if req.asr_itn is None else bool(req.asr_itn)) if getattr(req, "audio", None) else None
    query_text = (req.query or audio_text or "").strip()

    cmd = await handle_command(query_text, thread_id)
    if cmd is not None:
        return cmd

    # ── 前置检查：若该 thread 存在未处理的 HITL 挂起，自动拒绝并恢复图 ──
    await _auto_reject_stale_hitl(thread_id, tenant_id)

    state = State(thread_id=thread_id, query=query_text, history=None,
                  tenant_id=tenant_id, user_id=user_id,
                  quoted_message=req.quoted_message, images=req.images or None)
    result = await run_graph(state)

    # ── HITL 中断检测：图被 interrupt() 暂停，需要用户二次确认 ──
    if isinstance(result, dict) and result.get("__interrupted__"):
        interrupt_info = result.get("__interrupt_info__") or {}
        logger.info(f"HITL 中断：thread_id={thread_id}，等待用户确认")
        _msg = interrupt_info.get("message", "即将执行高危操作，请确认是否继续？")
        _ops = interrupt_info.get("operations", [])
        _oid = interrupt_info.get("order_id", "")
        hitl_state = await _mq.register_hitl_pending(
            thread_id=thread_id, order_id=_oid,
            operations=_ops, message=_msg,
        )
        return {
            "route": "hitl_confirm",
            "answer": _msg,
            "sources": [],
            "trace_id": get_trace_id(),
            "hitl": {
                "thread_id": thread_id,
                "operations": _ops,
                "order_id": _oid,
                "requires_confirmation": True,
                "expires_at": hitl_state.get("expires_at"),
                "timeout_seconds": _mq.HITL_TIMEOUT,
            },
        }

    route, answer, sources = determine_answer(result)
    _post_process_tasks(thread_id, user_id, tenant_id, query_text, answer)
    return {"route": route, "answer": answer, "sources": sources, "trace_id": get_trace_id()}


@app.post("/chat/confirm")
@token_bucket_limit("60/minute", key_func=get_user_id_or_ip)
async def chat_confirm(req: HitlConfirmRequest, request: Request):
    """HITL 高危操作确认接口：用户对挂起的高危操作做出确认或拒绝决定。

    前端收到 route="hitl_confirm" 的响应后，展示确认对话框。
    用户点击「确认」或「取消」后，调用本接口恢复图执行。

    Request Body:
        thread_id: 会话线程 ID（必须与被中断的请求一致）
        decision: "approved"（确认执行）或 "rejected"（取消操作）
    """
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    user_id = _get_user_id_from_request(request)
    decision = req.decision.strip().lower()

    if decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision 必须是 approved 或 rejected")

    # ── 过期校验：HITL 挂起状态是否已超时 ──
    pending = await _mq.get_hitl_pending(req.thread_id)
    if pending is None:
        logger.warning(f"HITL 确认接口：挂起状态不存在或已过期，thread_id={req.thread_id}")
        # Redis key 已过期，但 checkpoint 中图可能仍 interrupted，自动 reject 恢复
        try:
            await resume_graph(thread_id=req.thread_id, tenant_id=tenant_id, user_decision="rejected")
        except Exception:
            pass
        return {
            "route": "fallback",
            "answer": "操作确认已超时，系统已自动取消本次操作。如需继续，请重新发起。",
            "sources": [],
            "trace_id": get_trace_id(),
        }

    logger.info(
        f"HITL 确认接口：thread_id={req.thread_id}，decision={decision}，user_id={user_id}"
    )

    try:
        result = await resume_graph(
            thread_id=req.thread_id,
            tenant_id=tenant_id,
            user_decision=decision,
        )
    except Exception as e:
        logger.error(f"HITL 恢复执行失败: {e}", exc_info=True)
        return {
            "route": "fallback",
            "answer": "抱歉，操作处理异常，请稍后重试或联系人工客服。",
            "sources": [],
            "trace_id": get_trace_id(),
        }
    finally:
        await _mq.clear_hitl_pending(req.thread_id)

    route, answer, sources = determine_answer(result)
    _post_process_tasks(req.thread_id, user_id, tenant_id, "", answer)
    return {"route": route, "answer": answer, "sources": sources, "trace_id": get_trace_id()}


@app.post("/chat/stream")
@token_bucket_limit("60/minute", key_func=get_user_id_or_ip)
@token_bucket_limit("1000/hour", key_func=get_tenant_id_key)
async def chat_stream(req: ChatRequest, request: Request):
    """流式对话接口（SSE）：通过消息队列异步削峰，立即返回 SSE 流。

    路由策略（优先级从高到低）：
      ① MQ 路径  — Celery Worker 处理，结果写入 Redis Stream，SSE 从 Stream 读取
      ② 降级路径 — MQ 不可用时，FastAPI 后台 Task 执行，同样写 Redis Stream
      ③ 兜底路径 — Redis 不可用时，完全内联执行

    SSE 事件格式：
    - {"type": "token", "content": "..."}
    - {"type": "done", "route": "...", "answer": "...", "sources": [...], "trace_id": "..."}
    - {"type": "error", "message": "..."}
    """
    thread_id = req.thread_id or request.state.request_id
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    user_id = _get_user_id_from_request(request)

    # Prometheus：QPS 计数 & 堆积量
    if CHAT_STREAM_REQUESTS_TOTAL is not None:
        CHAT_STREAM_REQUESTS_TOTAL.labels(path="/chat/stream", tenant_id=tenant_id).inc()
    if CHAT_STREAM_ACTIVE_REQUESTS is not None:
        CHAT_STREAM_ACTIVE_REQUESTS.inc()
    _consume_start = time.perf_counter()
    _gauge_released = [False]   # 防止 _sse_inline / _sse_generator 双重释放

    audio_text = transcribe_audio(
        req.audio, req.asr_language, True if req.asr_itn is None else bool(req.asr_itn),
    ) if getattr(req, "audio", None) else None
    query_text = (req.query or audio_text or "").strip()

    cmd = await handle_command(query_text, thread_id)
    if cmd is not None:
        return cmd

    # ── 前置检查：若该 thread 存在未处理的 HITL 挂起，自动拒绝并恢复图 ──
    await _auto_reject_stale_hitl(thread_id, tenant_id)

    images = req.images or None
    trace_id = get_trace_id()
    task_id = trace_id
    state_dict = {
        "thread_id": thread_id, "query": query_text,
        "tenant_id": tenant_id, "user_id": user_id,
        "quoted_message": req.quoted_message,
        "images": images,
    }

    # ── ③ 兜底路径：Redis 不可用时，完全内联执行 ─────────────────────────────
    async def _sse_inline():
        """图在当前进程 asyncio Task 中运行，直接从内存队列推送 token。

        兜底优化：若 Redis 实际可用，则同步将事件写入 Redis Stream，
        使前端刷新后仍可通过 GET /chat/stream/{task_id} 重放已输出内容。
        """
        stream_queue: asyncio.Queue = asyncio.Queue()
        streamed_tokens = []
        ttft_start = time.perf_counter()
        ttft_recorded = False
        finished = False

        # 尝试连接 Redis，以便支持有限的断点续传（兜底路径触发时 Redis 可能刚恢复）
        _r_inline = None
        _inline_stream_key = f"chat:stream:{task_id}"
        try:
            import redis.asyncio as aioredis
            _r_inline = aioredis.from_url(_REDIS_URL, decode_responses=True)
            await asyncio.wait_for(_r_inline.ping(), timeout=1.0)
            logger.info(f"[内联兜底] Redis 可用，将同步写入 Stream 支持续传: task_id={task_id}")
        except Exception:
            if _r_inline is not None:
                try:
                    await _r_inline.aclose()
                except Exception:
                    pass
            _r_inline = None
            logger.info(f"[内联兜底] Redis 不可用，纯内联模式（不支持续传）: task_id={task_id}")

        async def _write_inline_stream(event: dict) -> None:
            """将事件写入 Redis Stream（仅在 _r_inline 可用时执行）"""
            if _r_inline is None:
                return
            try:
                await _r_inline.xadd(
                    _inline_stream_key,
                    {"event": json.dumps(event, ensure_ascii=False)},
                    maxlen=_mq.STREAM_MAXLEN,
                    approximate=True,
                )
                await _r_inline.expire(_inline_stream_key, _mq.STREAM_TTL)
            except Exception as we:
                logger.debug(f"[内联兜底] Stream 写入失败（非致命）: {we}")

        async def _run_graph_task():
            state = State(thread_id=thread_id, query=query_text, history=None,
                          tenant_id=tenant_id, user_id=user_id,
                          quoted_message=req.quoted_message, images=images)
            return await run_graph_stream(state, stream_queue)

        graph_task = asyncio.create_task(_run_graph_task())
        try:
            # 在 SSE 开头告知客户端断线后 3 秒自动重连
            yield "retry: 3000\n\n"

            while not graph_task.done():
                try:
                    event = await asyncio.wait_for(stream_queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue
                if event.get("type") == "token" and not ttft_recorded:
                    ttft_recorded = True
                    _record_ttft(ttft_start, f"内联兜底 thread={thread_id}")
                if event.get("type") == "token":
                    streamed_tokens.append(event.get("content", ""))
                await _write_inline_stream(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            while not stream_queue.empty():
                event = stream_queue.get_nowait()
                if event.get("type") == "token":
                    if not ttft_recorded:
                        ttft_recorded = True
                        _record_ttft(ttft_start, f"内联兜底 thread={thread_id}")
                    streamed_tokens.append(event.get("content", ""))
                await _write_inline_stream(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            graph_result = graph_task.result()

            # ── HITL 中断检测（流式模式） ──
            if isinstance(graph_result, dict) and graph_result.get("__interrupted__"):
                interrupt_info = graph_result.get("__interrupt_info__") or {}
                _msg = interrupt_info.get("message", "即将执行高危操作，请确认是否继续？")
                _ops = interrupt_info.get("operations", [])
                _oid = interrupt_info.get("order_id", "")
                # 在 Redis 中注册 HITL 挂起状态（带 TTL，超时自动过期）
                hitl_state = await _mq.register_hitl_pending(
                    thread_id=thread_id, order_id=_oid,
                    operations=_ops, message=_msg,
                )
                hitl_event = {
                    "type": "hitl_confirm",
                    "message": _msg,
                    "operations": _ops,
                    "order_id": _oid,
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                    "expires_at": hitl_state.get("expires_at"),
                    "timeout_seconds": _mq.HITL_TIMEOUT,
                }
                await _write_inline_stream(hitl_event)
                yield f"data: {json.dumps(hitl_event, ensure_ascii=False)}\n\n"
                finished = True
                _record_stream_metric("success", time.perf_counter() - _consume_start, thread_id)
            else:
                route, answer, sources = determine_answer(graph_result)
                if not streamed_tokens and answer:
                    if not ttft_recorded:
                        ttft_recorded = True
                        _record_ttft(ttft_start, "内联兜底非流式")
                    token_event = {"type": "token", "content": answer}
                    await _write_inline_stream(token_event)
                    yield f"data: {json.dumps(token_event, ensure_ascii=False)}\n\n"

                done_event = {"type": "done", "route": route, "answer": answer, "sources": sources, "trace_id": trace_id}
                await _write_inline_stream(done_event)
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

                finished = True
                _record_stream_metric("success", time.perf_counter() - _consume_start, thread_id)
                _post_process_tasks(thread_id, user_id, tenant_id, query_text, answer)

        except Exception as e:
            logger.error(f"/chat/stream 内联兜底异常: {e}", exc_info=True)
            if not finished:
                _record_stream_metric("error", time.perf_counter() - _consume_start, thread_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]}, ensure_ascii=False)}\n\n"
        finally:
            if not _gauge_released[0] and CHAT_STREAM_ACTIVE_REQUESTS is not None:
                _gauge_released[0] = True
                CHAT_STREAM_ACTIVE_REQUESTS.dec()
            if _r_inline is not None:
                try:
                    await _r_inline.aclose()
                except Exception:
                    pass

    # ── ① MQ + ② 降级路径：通过 Redis Stream 推送 token ─────────────────────
    async def _sse_generator():
        ttft_start = time.perf_counter()
        ttft_recorded = False
        metrics_recorded = [False]

        def _record_if_needed(status: str):
            if not metrics_recorded[0]:
                metrics_recorded[0] = True
                _record_stream_metric(status, time.perf_counter() - _consume_start, task_id)

        # 步骤 1：初始化 Redis 任务；不可用时回退内联兜底
        if not _mq.init_task_in_redis(task_id, state_dict):
            logger.warning(f"[/chat/stream] Redis 不可用，切换内联兜底: task_id={task_id}")
            async for chunk in _sse_inline():
                yield chunk
            return

        async def _post_process_fallback(sd: dict, answer: str):
            _post_process_tasks(sd["thread_id"], sd.get("user_id"), sd.get("tenant_id", "default"), sd.get("query", ""), answer)

        _fallback_triggered = [False]

        def _launch_fallback(reason: str):
            """在 FastAPI 进程内启动降级执行，结果写入同一个 Redis Stream（幂等）"""
            if _fallback_triggered[0]:
                return
            _fallback_triggered[0] = True
            logger.info(f"[/chat/stream] 触发降级（{reason}）: task_id={task_id}")
            asyncio.create_task(_mq.run_direct_to_stream(
                task_id=task_id, state_dict=state_dict, trace_id=trace_id,
                run_graph_stream_fn=run_graph_stream, determine_answer_fn=determine_answer,
                post_process_fn=_post_process_fallback,
            ))

        # 步骤 2：尝试投递 MQ；失败则直接降级
        mq_ok = _mq.submit_to_celery(task_id, state_dict, trace_id)
        if not mq_ok:
            _launch_fallback("MQ 不可用")
        else:
            logger.info(f"[/chat/stream] 任务已投递到 MQ: task_id={task_id}")

        # 步骤 3：从 Redis Stream 读取事件并实时转发给客户端
        _HEARTBEAT_TIMEOUT = 10   # Worker 无响应超时阈值（秒）
        _worker_responded = False
        _submission_time = time.time()

        # 断点续传：客户端可通过 X-Last-Event-ID 头携带上次收到的最后一条消息 ID
        # 浏览器 EventSource 断线重连时会自动携带该头，这里也兼容手动 POST 续传场景
        _resume_from = (
            request.headers.get("X-Last-Event-ID")
            or request.headers.get("last-event-id")
            or "0"
        )
        if _resume_from != "0":
            logger.info(f"[/chat/stream] 检测到断点续传请求: task_id={task_id}, last_event_id={_resume_from}")

        try:
            import redis.asyncio as aioredis
            r_async = aioredis.from_url(_REDIS_URL, decode_responses=True)
            stream_key = f"chat:stream:{task_id}"
            last_id = _resume_from   # 从断点处或从头开始
            timeout_at = time.time() + 60  # 最长等待 1 分钟

            # 在 SSE 开头告知客户端断线后 3 秒自动重连（EventSource 协议标准指令）
            yield "retry: 3000\n\n"

            try:
                while True:
                    if time.time() > timeout_at:
                        logger.warning(f"[/chat/stream] SSE 读取超时: task_id={task_id}")
                        _record_if_needed("error")
                        yield f"data: {json.dumps({'type': 'error', 'message': '响应超时，请重试'}, ensure_ascii=False)}\n\n"
                        break

                    if mq_ok and not _worker_responded and not _fallback_triggered[0]:
                        if time.time() - _submission_time > _HEARTBEAT_TIMEOUT:
                            logger.warning(f"[/chat/stream] Worker 心跳超时，触发降级: task_id={task_id}")
                            _mq.on_worker_suspected_dead(task_id)
                            _launch_fallback("Worker 心跳超时")

                    block_ms = min(_HEARTBEAT_TIMEOUT * 1000, max(int((timeout_at - time.time()) * 1000), 100))
                    try:
                        results = await r_async.xread({stream_key: last_id}, count=50, block=block_ms)
                    except Exception as e:
                        logger.error(f"[/chat/stream] Redis Stream 读取异常: {e}")
                        _record_if_needed("error")
                        yield f"data: {json.dumps({'type': 'error', 'message': '服务异常，请稍后重试'}, ensure_ascii=False)}\n\n"
                        break

                    if not results:
                        continue

                    for _, messages in results:
                        for msg_id, data in messages:
                            last_id = msg_id
                            try:
                                event = json.loads(data.get("event", "{}"))
                            except Exception:
                                continue
                            if event.get("type") == "init":
                                continue
                            _worker_responded = True

                            if event.get("type") == "token" and not ttft_recorded:
                                ttft_recorded = True
                                _record_ttft(ttft_start, f"MQ/降级 task={task_id} mq={'yes' if mq_ok else 'fallback'}")

                            # 携带 SSE id 字段：浏览器断线重连时会自动将此值作为
                            # Last-Event-ID 头发回，实现断点续传
                            yield f"id: {msg_id}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

                            evt_type = event.get("type")
                            if evt_type in ("done", "error", "hitl_confirm"):
                                _record_if_needed("success" if evt_type in ("done", "hitl_confirm") else "error")
                                return
            finally:
                try:
                    await r_async.aclose()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"[/chat/stream] SSE 顶层异常: {e}", exc_info=True)
            _record_if_needed("error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]}, ensure_ascii=False)}\n\n"
        finally:
            if not _gauge_released[0] and CHAT_STREAM_ACTIVE_REQUESTS is not None:
                _gauge_released[0] = True
                CHAT_STREAM_ACTIVE_REQUESTS.dec()

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Request-Id": trace_id, "X-Trace-Id": trace_id, "X-Task-Id": task_id,
        },
    )


@app.get("/chat/task/{task_id}")
async def get_chat_task_status(task_id: str):
    """查询聊天任务状态（轮询接口，SSE 不可用时使用）"""
    status = await _mq.read_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"任务不存在或已过期: {task_id}")
    return status


@app.get("/chat/stream/{task_id}")
async def chat_stream_resume(task_id: str, request: Request):
    """断点续传接口（GET SSE）

    适用场景：
      - 前端刷新页面后通过 task_id 重新订阅流式输出
      - 浏览器 EventSource 断线后携带 Last-Event-ID 自动重连
      - 网络中断恢复后从指定位置续传

    断点位置确定优先级（高→低）：
      1. Last-Event-ID 请求头（EventSource 断线重连时浏览器自动携带）
      2. last_id 查询参数（客户端手动指定，用于非 EventSource 场景）
      3. "0"（从头开始，即完整重放）

    前端使用示例：
      const es = new EventSource(`/chat/stream/${taskId}`);
      es.onmessage = (e) => { ... };
      // 断线后 EventSource 会自动携带 Last-Event-ID 重连，无需额外处理
    """
    # 读取断点 ID：优先使用浏览器自动携带的 Last-Event-ID
    last_event_id = (
        request.headers.get("Last-Event-ID")
        or request.headers.get("last-event-id")
        or request.query_params.get("last_id")
        or "0"
    )
    trace_id = get_trace_id()
    logger.info(f"[断点续传] 收到重连请求: task_id={task_id}, last_event_id={last_event_id}")

    async def _resume_sse():
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(_REDIS_URL, decode_responses=True)
            stream_key = f"chat:stream:{task_id}"
            task_key = f"chat:task:{task_id}"

            # 校验 Stream 是否存在（已过期则提示用户重新发起）
            try:
                stream_exists = await r.exists(stream_key)
            except Exception:
                stream_exists = False

            if not stream_exists:
                logger.warning(f"[断点续传] Stream 不存在或已过期: task_id={task_id}")
                yield (
                    f"data: {json.dumps({'type': 'error', 'message': '会话已过期，请重新发起对话'}, ensure_ascii=False)}\n\n"
                )
                await r.aclose()
                return

            # 告知浏览器 3 秒后自动重连
            yield "retry: 3000\n\n"

            last_id = last_event_id
            timeout_at = time.time() + 120   # 最长等待 2 分钟
            _POLL_BLOCK_MS = 5000            # 每次 xread 最长阻塞 5 秒

            try:
                while True:
                    if time.time() > timeout_at:
                        logger.warning(f"[断点续传] 续传超时: task_id={task_id}")
                        yield (
                            f"data: {json.dumps({'type': 'error', 'message': '续传超时，请重新发起对话'}, ensure_ascii=False)}\n\n"
                        )
                        break

                    try:
                        results = await r.xread({stream_key: last_id}, count=50, block=_POLL_BLOCK_MS)
                    except Exception as e:
                        logger.error(f"[断点续传] Redis Stream 读取异常: task_id={task_id}, error={e}")
                        yield (
                            f"data: {json.dumps({'type': 'error', 'message': '读取异常，请稍后重试'}, ensure_ascii=False)}\n\n"
                        )
                        break

                    if not results:
                        # 无新消息时检查任务是否已完成（防止任务完成后客户端永久阻塞）
                        try:
                            task_val = await r.get(task_key)
                            if task_val:
                                task_status = json.loads(task_val).get("status", "")
                                if task_status in ("completed", "failed"):
                                    logger.info(
                                        f"[断点续传] 任务已完成无新事件，断开连接: "
                                        f"task_id={task_id}, status={task_status}"
                                    )
                                    break
                        except Exception:
                            pass
                        continue

                    for _, messages in results:
                        for msg_id, data in messages:
                            last_id = msg_id
                            try:
                                event = json.loads(data.get("event", "{}"))
                            except Exception:
                                continue
                            # 跳过初始化占位事件
                            if event.get("type") == "init":
                                continue
                            # 携带 id 字段，支持下次断线时精确续传
                            yield f"id: {msg_id}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

                            if event.get("type") in ("done", "error"):
                                logger.info(
                                    f"[断点续传] 流式输出结束: task_id={task_id}, type={event.get('type')}"
                                )
                                return
            finally:
                try:
                    await r.aclose()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"[断点续传] 顶层异常: task_id={task_id}, error={e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _resume_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Task-Id": task_id,
            "X-Trace-Id": trace_id,
        },
    )


# =========================================================================
# ── 基础接口 ──────────────────────────────────────────────────────────────────
# =========================================================================

@app.get("/health")
async def health():
    """健康检查：并发探测 Milvus/Redis/PostgreSQL，返回熔断器状态与延迟快照"""
    from core.circuit_breaker import get_all_snapshots
    milvus_s, redis_s, postgres_s = await asyncio.gather(
        asyncio.to_thread(_check_milvus),
        _check_redis(),
        asyncio.to_thread(_check_postgres),
    )
    m = config.get_metrics()
    return {
        "model": config.MODEL_NAME,
        "services": {"milvus": milvus_s, "redis": redis_s, "postgres": postgres_s},
        "metrics": {k: m.snapshot(k) for k in ("overall", "kb", "order", "direct", "handoff", "vectors_add", "vectors_delete")},
        "circuit_breakers": get_all_snapshots(),
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus 指标拉取接口"""
    from core.observability import generate_latest, CONTENT_TYPE_LATEST
    if generate_latest is None:
        return JSONResponse({"error": "prometheus_client 未安装"}, status_code=503)
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/greet")
async def greet():
    return {
        "message": "您好，请问有什么可以帮您？",
        "options": [
            {"key": "faq",   "title": "产品咨询", "desc": "售前售后、账户、物流等常见问题"},
            {"key": "order", "title": "订单查询", "desc": "查询订单状态、取消、退款等操作"},
            {"key": "human", "title": "人工转接", "desc": "直接转人工客服"},
        ],
    }


@app.get("/suggest/{thread_id}")
async def suggest(thread_id: str):
    """建议问题流（SSE），前端实时展示"""
    async def _gen():
        q = SUGGEST_QUEUES.setdefault(thread_id, asyncio.Queue())
        deadline = time.perf_counter() + 15.0
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.perf_counter() > deadline:
                    err = {"route": None, "error": {"code": "timeout", "message": "建议生成超时"}, "event": "error", "final": True}
                    yield f"id: {thread_id}\nevent: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
                    break
                continue
            yield f"id: {thread_id}\nevent: {item.get('event', 'suggest')}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            if bool(item.get("final")):
                break
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/models/list")
async def models_list():
    t0 = time.perf_counter()
    data = {"current": config.get_current_model_name(), "models": config.get_supported_models()}
    logger.info("models list cost=%.2fms", (time.perf_counter() - t0) * 1000.0)
    return _ok(data)


@app.post("/models/switch")
async def models_switch(req: SwitchRequest):
    name = str(req.name or "").strip()
    lock = config.get_model_lock()
    with lock:
        v = config.validate_model(name)
        if not v.get("ok"):
            return _err(v.get("code") or "invalid", v.get("message") or "无效模型")
        res = config.switch_model(name, None)
        if not res.get("ok"):
            return _err(res.get("code") or "error", res.get("message") or "切换失败")
        try:
            graph.llm = config.get_llm()
            graph.router_llm = graph.llm.with_structured_output(graph.Route)
            graph.sql_llm = graph.llm.with_structured_output(graph.SQLSpec)
            graph._react_agent = None
            global _chain
            _chain = warmup_graph_templates(force_rebuild=True)
        except Exception as e:
            return _err("graph_reload_error", str(e))
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            with open(LOG_DIR / "requests.log", "a", encoding="utf-8") as f:
                f.write(f"{ts} | INFO | root | model switch apply new={name}\n")
        except Exception:
            pass
        return _ok({"current": config.get_current_model_name(), "models": config.get_supported_models()})


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str, request: Request):
    """根据订单 ID 查询订单详情"""
    try:
        payload = getdb(order_id)
        sql, params = payload["sql"], payload["params"]
        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
        row = postgres.fetchone(config.get_postgres_dsn(tenant_id), sql, params)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        return {
            "order_id": str(row[0]), "status": str(row[1]),
            "amount": float(row[2]) if row[2] is not None else None,
            "create_time": str(row[3]) if row[3] is not None else None,
            "update_time": str(row[4]) if len(row) > 4 and row[4] is not None else None,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


# =========================================================================
# ── 认证接口 ──────────────────────────────────────────────────────────────────
# =========================================================================

def _validate_public_signup_username(username: str) -> str:
    u = (username or "").strip()
    if len(u) < 3 or len(u) > 32:
        raise HTTPException(status_code=400, detail="用户名长度需在 3～32 个字符之间")
    if not re.match(r"^[\w\u4e00-\u9fff]+$", u):
        raise HTTPException(status_code=400, detail="用户名仅支持字母、数字、下划线与中文")
    return u


@app.post("/auth/signup")
@token_bucket_limit("20/hour")
async def auth_public_signup(request: Request, req: PublicSignupRequest):
    """普通用户公开注册（角色固定为 user，租户固定为 default）"""
    username = _validate_public_signup_username(req.username)
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if len(req.password) > 128:
        raise HTTPException(status_code=400, detail="密码过长")
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    tenant_id = os.environ.get("PUBLIC_SIGNUP_TENANT_ID", "default")
    try:
        with postgres.get_conn(_pg_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password, role, tenant_id, created_at) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                    (username, pw_hash, "user", tenant_id, int(time.time())),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        raise HTTPException(status_code=409, detail="该用户名已被注册")
    return _ok({
        "token": _make_token(new_id, username, "user", tenant_id),
        "user": {"id": new_id, "username": username, "role": "user", "tenant_id": tenant_id},
    })


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """用户登录，返回 JWT Token"""
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    row = postgres.fetchone(_pg_dsn(), "SELECT id, username, role, tenant_id FROM users WHERE username=%s AND password=%s",
                            (req.username, pw_hash))
    if not row:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    user_id, username, role, tenant_id = row
    return _ok({
        "token": _make_token(user_id, username, role, tenant_id),
        "user": {"id": user_id, "username": username, "role": role, "tenant_id": tenant_id},
    })


@app.post("/auth/register")
async def auth_register(req: RegisterRequest, _user: dict = Depends(get_current_user)):
    """注册新用户（仅管理员可操作）"""
    if _user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="无权限，仅管理员可注册新用户")
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    try:
        with postgres.get_conn(_pg_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password, role, tenant_id, created_at) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                    (req.username, pw_hash, req.role, req.tenant_id, int(time.time())),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return _ok({"id": new_id, "username": req.username, "role": req.role, "tenant_id": req.tenant_id})


@app.get("/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    """获取当前登录用户信息"""
    row = postgres.fetchone(_pg_dsn(), "SELECT id, username, role, tenant_id, created_at FROM users WHERE id=%s",
                            (int(current_user["sub"]),))
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    return _ok({"id": row[0], "username": row[1], "role": row[2], "tenant_id": row[3], "created_at": row[4]})


# =========================================================================
# ── 会话管理接口 ───────────────────────────────────────────────────────────────
# =========================================================================

@app.get("/sessions")
@token_bucket_limit("120/minute", key_func=get_user_id_or_ip)
async def list_sessions(request: Request, current_user: dict = Depends(get_current_user)):
    """列出当前用户的会话（按最近更新时间倒序，最多 200 条）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    rows = postgres.fetchall(_pg_dsn(),
        "SELECT thread_id, title, created_at, updated_at, message_count FROM sessions WHERE user_id=%s ORDER BY updated_at DESC LIMIT 200",
        (user_id,),
    )
    sessions = [{"thread_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3], "message_count": r[4]} for r in rows]
    return _ok({"sessions": sessions})


@app.get("/sessions/{thread_id}/messages")
async def get_session_messages_api(thread_id: str, current_user: dict = Depends(get_current_user)):
    """获取指定会话的历史消息（校验用户归属权）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    row = postgres.fetchone(_pg_dsn(), "SELECT title FROM sessions WHERE thread_id=%s AND user_id=%s", (thread_id, user_id))
    if not row:
        raise HTTPException(status_code=404, detail="会话不存在或无权限访问")
    return _ok({"thread_id": thread_id, "title": row[0], "messages": await config.get_all_session_messages(thread_id)})


@app.delete("/sessions/{thread_id}")
async def delete_session_api(thread_id: str, current_user: dict = Depends(get_current_user)):
    """删除指定会话（元信息 + 消息历史 + 摘要缓存）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    if not _delete_session_registry(thread_id, user_id):
        raise HTTPException(status_code=404, detail="会话不存在或无权限删除")
    await config.delete_session_messages(thread_id)
    return _ok({"deleted": thread_id})


@app.patch("/sessions/{thread_id}")
async def rename_session_api(thread_id: str, req: RenameSessionRequest, current_user: dict = Depends(get_current_user)):
    """重命名会话标题"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="标题不能为空")
    if not _update_session_title(thread_id, user_id, title):
        raise HTTPException(status_code=404, detail="会话不存在或无权限修改")
    return _ok({"thread_id": thread_id, "title": title})


# =========================================================================
# ── 知识库文档上传接口 ─────────────────────────────────────────────────────────
# =========================================================================

@app.post("/api/v1/kb/upload")
@token_bucket_limit("30/hour", key_func=get_user_id_or_ip)
async def kb_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """批量上传知识库文档，后台多进程解析入库（支持 .txt/.md/.pdf/.docx）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    tenant_id = request.headers.get("X-Tenant-ID") or "default"
    collection_name = config.get_collection_name(tenant_id)
    service_dir = os.path.dirname(os.path.abspath(__file__))

    task_records = []
    for upload in files:
        orig_name = upload.filename or "unknown"
        ext = os.path.splitext(orig_name)[1].lower()
        if ext not in _KB_ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"不支持 {ext}，仅支持: {', '.join(_KB_ALLOWED_EXT)}")

        task_id = str(uuid.uuid4())
        file_path = os.path.join(_KB_UPLOADS_DIR, f"{task_id}{ext}")
        content = await upload.read()
        with open(file_path, "wb") as fp:
            fp.write(content)

        now = int(time.time())
        postgres.execute(_pg_dsn(),
            "INSERT INTO kb_tasks (id, user_id, filename, file_size, file_path, collection_name, status, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (task_id, user_id, orig_name, len(content), file_path, collection_name, "pending", now, now),
        )

        asyncio.create_task(_dispatch_kb_task(task_id, file_path, collection_name, service_dir))
        task_records.append({"task_id": task_id, "filename": orig_name, "status": "pending", "collection_name": collection_name})

    return _ok({"tasks": task_records, "collection_name": collection_name})


@app.get("/api/v1/kb/tasks")
@token_bucket_limit("120/minute", key_func=get_user_id_or_ip)
async def kb_task_list(request: Request, current_user: dict = Depends(get_current_user)):
    """查询当前用户的知识库解析任务列表（最近 200 条）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    rows = postgres.fetchall(_pg_dsn(),
        "SELECT id, filename, file_size, collection_name, status, error, created_at, updated_at, chunk_count FROM kb_tasks WHERE user_id=%s ORDER BY created_at DESC LIMIT 200",
        (user_id,),
    )
    tasks = [{"task_id": r[0], "filename": r[1], "file_size": r[2], "collection_name": r[3],
              "status": r[4], "error": r[5], "created_at": r[6], "updated_at": r[7], "chunk_count": r[8]} for r in rows]
    return _ok({"tasks": tasks})


@app.delete("/api/v1/kb/tasks/{task_id}")
async def kb_task_delete(task_id: str, current_user: dict = Depends(get_current_user)):
    """删除指定任务记录（不删除已入库的向量）"""
    user_id = _canonical_user_id_from_token_payload(current_user)
    with postgres.get_conn(_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_tasks WHERE id=%s AND user_id=%s", (task_id, user_id))
            deleted = cur.rowcount > 0
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="任务不存在或无权限删除")
    return _ok({"deleted": task_id})


# =========================================================================
# ── Milvus 管理接口 ────────────────────────────────────────────────────────────
# =========================================================================

@app.get("/api/v1/milvus/collections")
@token_bucket_limit("30/minute", key_func=get_user_id_or_ip)
async def milvus_collections_list(request: Request):
    """列出所有 Milvus 集合及其统计信息"""
    try:
        from rag.milvus_store import MilvusStore
        store = MilvusStore()
        return _ok({"collections": [store.get_collection_stats(n) for n in store.list_collections()]})
    except Exception as e:
        logger.error(f"列出 Milvus 集合失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列出集合失败: {str(e)}")


@app.post("/api/v1/milvus/collections/delete")
@token_bucket_limit("10/minute", key_func=get_user_id_or_ip)
async def milvus_collection_delete(req: MilvusCollectionDeleteRequest, request: Request, _auth: Any = Depends(require_api_key)):
    """删除指定的 Milvus 集合"""
    t0 = time.perf_counter()
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    try:
        from rag.milvus_store import MilvusStore
        MilvusStore(collection_name=req.collection_name).clear()
        logger.info(f"删除 Milvus 集合: {req.collection_name}, cost={(time.perf_counter()-t0)*1000:.2f}ms")
        _audit("milvus_collection_delete", {"request_id": request.state.request_id,
                                             "collection_name": req.collection_name, "tenant_id": tenant_id})
        return _ok({"message": f"集合 {req.collection_name} 已删除并重建", "collection_name": req.collection_name})
    except Exception as e:
        logger.error(f"删除 Milvus 集合失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除集合失败：{str(e)}")


@app.post("/api/v1/milvus/ingest/file")
@token_bucket_limit("20/hour", key_func=get_user_id_or_ip)
async def milvus_file_ingest(req: MilvusFileIngestRequest, request: Request, _auth: Any = Depends(require_api_key)):
    """将指定文件入库到 Milvus"""
    t0 = time.perf_counter()
    tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=400, detail=f"文件不存在：{req.file_path}")
    try:
        from rag.ingestion import Ingestion
        success = Ingestion().ingest_file(req.file_path, req.collection_name)
        if not success:
            return _err("no_chunks", "文件未生成有效切片")
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(f"文件入库成功: {req.file_path} → {req.collection_name}, cost={elapsed:.2f}ms")
        _audit("milvus_file_ingest", {"request_id": request.state.request_id,
                                      "file_path": req.file_path, "collection_name": req.collection_name, "tenant_id": tenant_id})
        return _ok({"message": f"{req.file_path} 已入库到 {req.collection_name}",
                    "file_path": req.file_path, "collection_name": req.collection_name})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件入库失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件入库失败：{str(e)}")


# =========================================================================
# ── RAGAS 评估接口 ─────────────────────────────────────────────────────────────
# =========================================================================

@app.post("/api/ragas/evaluate/single")
async def evaluate_single(req: SingleEvaluationRequest, request: Request):
    """评估单个用例"""
    try:
        from scripts.ragas_evaluator import RagasEvaluator, EvaluationCase
        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
        evaluator = RagasEvaluator()
        case = EvaluationCase(name=req.case.name, query=req.case.query, ground_truth=req.case.ground_truth)
        result = evaluator.evaluate_case(case=case, answer=req.answer, route=req.route, tenant_id=tenant_id)
        return _ok({
            "case_name": result.case_name, "query": result.query, "answer": result.answer,
            "metrics": {"faithfulness": result.faithfulness, "answer_relevancy": result.answer_relevancy,
                        "context_precision": result.context_precision, "context_recall": result.context_recall},
            "passed": result.passed, "problem_description": result.problem_description,
        })
    except Exception as e:
        logger.error(f"单个评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ragas/evaluate/batch")
async def evaluate_batch(req: BatchEvaluationRequest, request: Request):
    """批量评估（后台异步执行）；直接调用知识库节点(kb_node)获取回答和检索上下文，避免走完整图流程"""
    try:
        from scripts.ragas_evaluator import RagasEvaluator, EvaluationCase
        from graph.graph import kb_node
        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"
        evaluator = RagasEvaluator(faith_threshold=req.faith_threshold or 0.85, relev_threshold=req.relev_threshold or 0.7)
        cases = [EvaluationCase(name=c.name, query=c.query, ground_truth=c.ground_truth) for c in req.cases]

        async def answer_provider_async(query: str):
            """
            直接调用 kb_node 获取知识库回答和检索上下文。
            返回三元组 (answer, contexts, route)，contexts 来自 kb_node 的 sources 字段。
            """
            thread_id = f"eval_{uuid.uuid4()}"
            # 构造只包含必要字段的 State，intent 设为 faq 直接走知识库路径
            state = State(
                thread_id=thread_id,
                query=query,
                history="",
                tenant_id=tenant_id,
                intent="faq",
            )
            logger.info(f"RAGAS 评估：调用 kb_node，query='{query[:50]}'，tenant={tenant_id}")
            result_state = await kb_node(state)

            # 提取回答
            answer = result_state.kb_answer or ""

            # 从 sources 提取文本作为 RAGAS 上下文（与 kb_node 实际检索内容一致）
            contexts = [
                s["content"]
                for s in (result_state.sources or [])
                if s.get("content")
            ]
            logger.info(
                f"RAGAS 评估：kb_node 返回 answer 长度={len(answer)}，contexts 条数={len(contexts)}"
            )
            return answer, contexts, "kb"

        loop = asyncio.get_running_loop()

        def sync_answer_provider(query: str):
            """同步包装器：在线程池中通过 event loop 调用异步 answer_provider_async"""
            future = asyncio.run_coroutine_threadsafe(answer_provider_async(query), loop)
            return future.result()

        def _run_evaluation_sync():
            results, summary = evaluator.evaluate_batch(
                cases=cases,
                answer_provider=sync_answer_provider,
                tenant_id=tenant_id,
            )
            return results, summary, evaluator.save_report(results, summary)

        async def _run_evaluation():
            await asyncio.to_thread(_run_evaluation_sync)

        task = asyncio.create_task(_run_evaluation())
        return _ok({"message": "评估任务已启动", "task_id": str(id(task)), "cases_count": len(cases)})
    except Exception as e:
        logger.error(f"批量评估启动失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ragas/health")
async def ragas_health():
    try:
        from scripts.ragas_evaluator import get_evaluator
        evaluator = get_evaluator()
        return _ok({"available": evaluator.is_ragas_available(), "evaluator_initialized": True})
    except Exception as e:
        return _ok({"available": False, "evaluator_initialized": False, "error": str(e)})


@app.get("/api/ragas/test-cases")
async def get_ragas_test_cases(request: Request):
    try:
        from dataclasses import asdict
        from scripts.ragas_evaluator import get_evaluator
        from graph.graph import kb_node

        tenant_id = request.headers.get("X-Tenant-ID") or request.query_params.get("tenant") or "default"

        # 1) 获取默认测试用例
        evaluator = get_evaluator()
        cases = evaluator.get_default_test_cases()
        logger.info(
            f"RAGAS 测试用例评估接口触发：tenant={tenant_id}，cases_count={len(cases)}"
        )

        # 2) 定义 answer_provider：只调用知识库节点 kb_node 获取 answer + contexts
        async def answer_provider_async(query: str):
            # 说明：kb_node 的输出里包含 kb_answer 和 sources（其中 content 就是检索上下文）
            thread_id = f"test_eval_{uuid.uuid4()}"
            state = State(
                thread_id=thread_id,
                query=query,
                history="",
                tenant_id=tenant_id,
                intent="faq",
            )
            logger.info(f"RAGAS 测试评估：调用 kb_node，query='{query[:50]}'，tenant={tenant_id}")
            result_state = await kb_node(state)

            answer = result_state.kb_answer or ""
            contexts = [
                s["content"]
                for s in (result_state.sources or [])
                if s.get("content")
            ]

            logger.info(
                f"RAGAS 测试评估：kb_node 返回 answer_len={len(answer)}，contexts={len(contexts)}"
            )
            return answer, contexts, "kb"

        loop = asyncio.get_running_loop()

        def sync_answer_provider(query: str):
            # 在线程池中同步等待 asyncio 的异步 kb_node 调用结果
            future = asyncio.run_coroutine_threadsafe(answer_provider_async(query), loop)
            return future.result()

        # 3) 直接跑一次评估（默认测试用例很少：5 条），返回结果给前端调试查看
        def _run_evaluation_sync():
            results, summary = evaluator.evaluate_batch(
                cases=cases,
                answer_provider=sync_answer_provider,
                tenant_id=tenant_id,
            )
            report_path = evaluator.save_report(results, summary)
            return results, summary, report_path

        results, summary, report_path = await asyncio.to_thread(_run_evaluation_sync)

        return _ok({
            "tenant_id": tenant_id,
            "summary": asdict(summary),
            "cases": [asdict(r) for r in results],
            "report_path": report_path,
        })
    except Exception as e:
        logger.error(f"RAGAS 测试用例评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ragas/reports")
async def list_ragas_reports():
    try:
        from scripts.ragas_evaluator import get_evaluator
        import glob
        evaluator = get_evaluator()
        report_files = sorted(glob.glob(os.path.join(evaluator.reports_dir, "*.json")),
                              key=os.path.getmtime, reverse=True)
        reports = []
        for rf in report_files[:20]:
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append({"filename": os.path.basename(rf),
                                 "timestamp": data.get("timestamp"), "summary": data.get("summary", {})})
            except Exception:
                continue
        return _ok({"reports": reports})
    except Exception as e:
        logger.error(f"列出报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
