"""Celery 聊天任务

在独立 Worker 进程中执行 LangGraph 图，通过 Redis Stream 实时推送 token。

核心流程：
  1. 从 RabbitMQ 接收任务参数（task_id, state_dict, trace_id）
  2. 构建 State，调用 run_graph_stream 执行图
  3. 每产生一个 token，立即通过 XADD 写入 Redis Stream（chat:stream:{task_id}）
  4. 图执行完成后写入 done 事件；异常时写入 error 事件并更新任务状态
  5. 执行后处理（保存消息历史、增量摘要、长期记忆）

兜底策略：
  - 幂等检查：若 Stream 中已存在 done/error 事件，跳过重复执行
  - 任务内置重试（最多 2 次），仅针对基础设施异常（非业务逻辑错误）
  - 超时保护：soft_time_limit=120s（SoftTimeLimitExceeded）发布超时 error；hard limit=180s 强杀
"""
import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

import redis as _redis_sync

# 模块级 logger
logger = logging.getLogger(__name__)

# 确保 service 目录在路径中（Worker 独立进程启动时需要）
_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

import dotenv
dotenv.load_dotenv()

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery_app import celery_app
from core.state import State
from graph.graph import run_graph_stream
from core.hander import determine_answer
import core.config as config
import core.postgres as postgres

# ========== Redis Stream 常量 ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
STREAM_KEY_PREFIX = "chat:stream:"   # Redis Stream key 前缀
TASK_KEY_PREFIX = "chat:task:"       # 任务元数据 key 前缀
STREAM_MAXLEN = 2000                 # Stream 最大条目数（近似裁剪，防内存溢出）
STREAM_TTL = 1800                    # Stream key 存活时间 30 分钟
TASK_TTL = 1800                      # 任务元数据存活时间 30 分钟


def _get_redis() -> _redis_sync.Redis:
    """获取同步 Redis 客户端（Celery Worker 中使用同步客户端）"""
    return _redis_sync.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)


def _write_stream(r: _redis_sync.Redis, task_id: str, event: dict) -> None:
    """将事件写入 Redis Stream，并刷新过期时间
    
    使用 XADD + maxlen 近似裁剪，兼顾实时性与内存安全。
    """
    stream_key = f"{STREAM_KEY_PREFIX}{task_id}"
    r.xadd(
        stream_key,
        {"event": json.dumps(event, ensure_ascii=False)},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    r.expire(stream_key, STREAM_TTL)
    logger.debug(f"[Worker] Stream 写入事件: task_id={task_id}, type={event.get('type')}")


def _set_task_status(r: _redis_sync.Redis, task_id: str, status: str, extra: Optional[dict] = None) -> None:
    """更新 Redis 中的任务状态元数据"""
    task_key = f"{TASK_KEY_PREFIX}{task_id}"
    data = {
        "status": status,
        "task_id": task_id,
        "updated_at": int(time.time()),
    }
    if extra:
        data.update(extra)
    r.setex(task_key, TASK_TTL, json.dumps(data, ensure_ascii=False))


def _is_already_done(r: _redis_sync.Redis, task_id: str) -> bool:
    """幂等检查：若 Stream 中最后一条事件是 done，说明已成功处理完毕，跳过重复执行
    
    注意：仅检查 done，不检查 error。
    error 表示上一次执行失败，重试时应该重新执行（需先清理旧 Stream 避免 token 重复）。
    """
    try:
        stream_key = f"{STREAM_KEY_PREFIX}{task_id}"
        last = r.xrevrange(stream_key, count=1)
        if last:
            event = json.loads(last[0][1].get("event", "{}"))
            if event.get("type") == "done":
                return True
    except Exception:
        pass
    return False


def _clear_stream_for_retry(r: _redis_sync.Redis, task_id: str) -> None:
    """重试前清空 Redis Stream，防止 SSE 客户端收到重复 token
    
    重新写入一条 retrying 事件，让仍在等待的客户端知道正在重试。
    """
    stream_key = f"{STREAM_KEY_PREFIX}{task_id}"
    try:
        r.delete(stream_key)
        r.xadd(
            stream_key,
            {"event": json.dumps({"type": "retrying", "message": "正在重试..."})},
            maxlen=STREAM_MAXLEN,
        )
        r.expire(stream_key, STREAM_TTL)
        logger.info(f"[Worker] Stream 已清空，等待重试: task_id={task_id}")
    except Exception as e:
        logger.warning(f"[Worker] 清空 Stream 失败（非致命）: task_id={task_id}, error={e}")


def _upsert_session_sync(thread_id: str, user_id: Optional[str], title: str, message_count: int) -> None:
    """同步保存会话元信息到 PostgreSQL（Worker 进程内调用）"""
    try:
        dsn = config.get_postgres_dsn()
        now = int(time.time())
        with postgres.get_conn(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (thread_id, user_id, title, created_at, updated_at, message_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        updated_at    = excluded.updated_at,
                        message_count = excluded.message_count
                    """,
                    (thread_id, user_id, title, now, now, message_count),
                )
            conn.commit()
    except Exception as e:
        logger.warning(f"[Worker] 更新会话元信息失败: {e}")


@celery_app.task(
    name="tasks.chat_task.execute_chat",
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=120,
    time_limit=180,
    max_retries=2,
    default_retry_delay=2,
)
def execute_chat(self: Task, task_id: str, state_dict: dict, trace_id: str) -> dict:
    """执行聊天任务：运行 LangGraph 图并将 token 流写入 Redis Stream

    多模态支持已迁移至 graph 层的 per-request VL 模型选择，
    images 通过 state_dict → State.images 透传到图节点。
    """
    query_preview = state_dict.get("query", "")[:50]
    logger.info(f"[Worker] 开始执行聊天任务: task_id={task_id}, query={query_preview!r}")

    r = _get_redis()
    try:
        # ---------- 幂等检查：防止重复处理 ----------
        if _is_already_done(r, task_id):
            logger.warning(f"[Worker] 任务已处理完成，跳过重复执行: task_id={task_id}")
            return {"status": "duplicate", "task_id": task_id}

        _set_task_status(r, task_id, "processing")

        state = State(
            thread_id=state_dict["thread_id"],
            query=state_dict.get("query", ""),
            history=None,
            tenant_id=state_dict.get("tenant_id", "default"),
            user_id=state_dict.get("user_id"),
            quoted_message=state_dict.get("quoted_message"),
            images=state_dict.get("images"),
        )

        # ---------- 在事件循环中运行异步图 ----------
        async def _run() -> dict:
            """异步执行流式图，将每个 token 实时写入 Redis Stream"""
            stream_queue: asyncio.Queue = asyncio.Queue()
            streamed_tokens = []

            # 启动图执行（独立 Task）
            graph_task = asyncio.create_task(run_graph_stream(state, stream_queue))
            logger.info(f"[Worker] 图执行 Task 已启动: task_id={task_id}")

            # 持续消费 token 并写入 Redis Stream
            while not graph_task.done():
                try:
                    event = await asyncio.wait_for(stream_queue.get(), timeout=0.3)
                    if event.get("type") == "token":
                        streamed_tokens.append(event.get("content", ""))
                    _write_stream(r, task_id, event)
                except asyncio.TimeoutError:
                    continue  # 正常等待，继续轮询

            # 排空队列中剩余事件（图执行完毕但队列可能还有积压）
            while not stream_queue.empty():
                event = stream_queue.get_nowait()
                if event.get("type") == "token":
                    streamed_tokens.append(event.get("content", ""))
                _write_stream(r, task_id, event)

            # 获取最终结果
            result = graph_task.result()
            route, answer, sources = determine_answer(result)

            # 非流式路径（订单查询/人工接待等）：一次性写入完整回答
            if not streamed_tokens and answer:
                _write_stream(r, task_id, {"type": "token", "content": answer})

            # 写入 done 事件（完成信号）
            done_event = {
                "type": "done",
                "route": route,
                "answer": answer or "",
                "sources": sources or [],
                "trace_id": trace_id,
            }
            _write_stream(r, task_id, done_event)
            logger.info(f"[Worker] 图执行完成，done 事件已写入: task_id={task_id}, route={route}")

            return {"answer": answer, "route": route}

        # Celery 同步任务中运行 asyncio 代码
        run_result = asyncio.run(_run())

        # ---------- 后处理（消息历史、摘要、长期记忆）----------
        async def _post_process():
            thread_id = state_dict["thread_id"]
            query_text = state_dict.get("query", "")
            user_id = state_dict.get("user_id")
            tenant_id = state_dict.get("tenant_id", "default")
            answer = run_result.get("answer", "")

            try:
                if query_text:
                    await config.append_session_message(thread_id, "user", query_text)
                if answer:
                    await config.append_session_message(thread_id, "assistant", answer)
                if query_text:
                    title = (query_text[:20] + "…") if len(query_text) > 20 else query_text
                    msg_count = len(await config.get_all_session_messages(thread_id)) + 2
                    # 同步写 PostgreSQL（无需等待，直接调用）
                    _upsert_session_sync(thread_id, user_id, title, msg_count)
                if query_text and answer:
                    await config.maybe_trigger_summary_update(thread_id, user_id)
                if query_text and answer:
                    try:
                        import memory.store as memory_module
                        await memory_module.extract_and_save_memory(user_id, tenant_id, query_text, answer)
                    except Exception as e:
                        logger.warning(f"[Worker] 长期记忆保存失败（非致命）: {e}")
                logger.info(f"[Worker] 后处理完成: task_id={task_id}")
            except Exception as e:
                logger.warning(f"[Worker] 后处理部分失败（非致命）: {e}")

        asyncio.run(_post_process())

        _set_task_status(r, task_id, "completed")
        logger.info(f"[Worker] 任务完成: task_id={task_id}")
        return {"status": "completed", "task_id": task_id}

    except SoftTimeLimitExceeded:
        # 软超时（120s）：属于不可恢复的超时，直接通知客户端，不重试
        # （硬超时 180s 由 Celery 强杀进程，此时已无法写 Redis）
        logger.error(f"[Worker] 任务软超时（120s）: task_id={task_id}")
        try:
            _write_stream(r, task_id, {"type": "error", "message": "处理超时，请稍后重试"})
            _set_task_status(r, task_id, "timeout")
        except Exception:
            pass
        raise  # 让 Celery 继续处理硬超时

    except Exception as exc:
        retry_count = self.request.retries          # 当前已经是第几次重试（0-based）
        max_retries = self.max_retries              # = 2，即最多重试 2 次
        has_retry_left = retry_count < max_retries

        logger.error(
            f"[Worker] 任务执行异常: task_id={task_id}, "
            f"retry={retry_count}/{max_retries}, error={exc}",
            exc_info=True,
        )

        if has_retry_left:
            # ── 还有重试机会：不写 error 事件（避免 SSE 客户端提前关闭），清空旧 Stream 后重新入队 ──
            # 清空旧 Stream 防止重试成功后客户端收到重复 token
            _clear_stream_for_retry(r, task_id)
            _set_task_status(r, task_id, "retrying", {
                "retry": retry_count + 1,
                "max_retries": max_retries,
                "error": str(exc)[:100],
            })
            countdown = 2 ** retry_count   # 指数退避：1s → 2s → 4s
            logger.warning(
                f"[Worker] 任务将在 {countdown}s 后重试 "
                f"({retry_count + 1}/{max_retries}): task_id={task_id}"
            )
            # raise Retry（不会走到后面的 raise）
            raise self.retry(exc=exc, countdown=countdown)
        else:
            # ── 重试次数已耗尽：写 error 事件通知客户端，消息将进入 DLQ ──
            logger.error(f"[Worker] 所有重试已耗尽，任务最终失败: task_id={task_id}")
            try:
                _write_stream(r, task_id, {
                    "type": "error",
                    "message": "服务暂时不可用，请稍后重试",
                })
                _set_task_status(r, task_id, "failed", {"error": str(exc)[:200]})
            except Exception:
                pass
            raise  # 让 Celery 将消息路由到 DLQ

    finally:
        try:
            r.close()
        except Exception:
            pass
