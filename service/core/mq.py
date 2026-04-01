"""消息队列客户端模块（熔断器 + 降级 + Redis Stream）

职责：
  1. submit_to_celery()     — 向 RabbitMQ/Celery 投递聊天任务（主路径）
  2. run_direct_to_stream() — MQ 不可用时降级，直接在 FastAPI 事件循环中运行图（兜底路径）
  3. init_task_in_redis()   — 初始化 Redis 任务元数据和 Stream（任务投递前必须调用）
  4. is_mq_healthy()        — 查询熔断器状态，供业务层决策使用

熔断器设计（三态机：closed → open → half_open → closed）：
  - closed    : 正常，所有任务走 MQ
  - open      : 熔断，所有任务走降级路径（直接执行）
  - half_open : 恢复探测，允许一次试探性 MQ 调用；成功则回 closed，失败则回 open

兜底优先级（高到低）：
  ① MQ + Redis Stream  （Celery Worker 异步处理，最大吞吐）
  ② 直接执行 + Redis Stream（FastAPI 进程内，性能稍低但实时性好）
  ③ 纯内联执行（Redis 不可用时，完全回退到原始行为，无任何改动）
"""
import asyncio
import json
import logging

logger = logging.getLogger(__name__)
import os
import threading
import time
from typing import Any, Callable, Optional

import redis as _redis_sync

# ========== 常量 ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
STREAM_KEY_PREFIX = "chat:stream:"
TASK_KEY_PREFIX = "chat:task:"
STREAM_MAXLEN = 500
STREAM_TTL = 1800   # 30 分钟
TASK_TTL = 3600     # 1 小时

# 是否启用 MQ（可通过环境变量 MQ_ENABLED=false 全局关闭，回退到直接执行模式）
MQ_ENABLED = os.getenv("MQ_ENABLED", "true").lower() == "true"

# ========== 熔断器状态（进程级单例）==========
_CB_LOCK = threading.Lock()
_CB_STATE = "closed"          # closed | open | half_open
_CB_FAILURES = 0              # 连续失败次数（成功时清零）
_CB_THRESHOLD = 5             # 连续失败达到此值触发熔断
_CB_RECOVERY_TIMEOUT = 60.0  # 熔断后等待多少秒再尝试恢复（half_open）
_CB_LAST_FAILURE_TS = 0.0     # 最后一次失败时间戳


def is_mq_healthy() -> bool:
    """查询 MQ 熔断器状态：True = 可以投递，False = 已熔断（应降级）"""
    global _CB_STATE
    if not MQ_ENABLED:
        return False
    with _CB_LOCK:
        if _CB_STATE == "closed":
            return True
        if _CB_STATE == "open":
            # 超过恢复等待时间，切换为 half_open 进行一次探测
            if time.time() - _CB_LAST_FAILURE_TS > _CB_RECOVERY_TIMEOUT:
                _CB_STATE = "half_open"
                logger.info("[MQ熔断器] 切换为 half_open，将发起恢复探测")
                return True
            return False
        # half_open：允许一次试探
        return True


def _on_mq_success() -> None:
    """MQ 调用成功 → 重置熔断器"""
    global _CB_STATE, _CB_FAILURES
    with _CB_LOCK:
        if _CB_STATE != "closed":
            logger.info(f"[MQ熔断器] 恢复成功，状态回到 closed（失败次数清零）")
        _CB_STATE = "closed"
        _CB_FAILURES = 0


def _on_mq_failure() -> None:
    """MQ 调用失败 → 更新熔断器，达到阈值则触发熔断"""
    global _CB_STATE, _CB_FAILURES, _CB_LAST_FAILURE_TS
    with _CB_LOCK:
        _CB_FAILURES += 1
        _CB_LAST_FAILURE_TS = time.time()
        if _CB_FAILURES >= _CB_THRESHOLD or _CB_STATE == "half_open":
            if _CB_STATE != "open":
                logger.warning(
                    f"[MQ熔断器] 熔断触发！连续失败 {_CB_FAILURES} 次，"
                    f"切换为 open，{_CB_RECOVERY_TIMEOUT}s 后尝试恢复"
                )
            _CB_STATE = "open"
        else:
            logger.warning(
                f"[MQ熔断器] MQ 调用失败（{_CB_FAILURES}/{_CB_THRESHOLD}），"
                f"当前状态仍为 closed"
            )


def on_worker_suspected_dead(task_id: str) -> None:
    """Worker 进程疑似挂起（投递成功但长时间无任何响应）→ 计入熔断失败次数

    由 SSE 生成器在检测到 Worker 心跳超时后调用。
    多次调用后达到阈值会触发熔断，后续请求直接走降级路径，避免用户持续等待。
    """
    logger.warning(f"[MQ熔断器] Worker 心跳超时，计入失败（task_id={task_id}）")
    _on_mq_failure()


# ========== Redis 工具 ==========

def _get_redis_sync() -> Optional[_redis_sync.Redis]:
    """获取同步 Redis 客户端（用于初始化任务元数据），失败返回 None"""
    try:
        r = _redis_sync.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=3)
        r.ping()
        return r
    except Exception as e:
        logger.warning(f"[MQ] Redis 同步连接失败: {e}")
        return None


def init_task_in_redis(task_id: str, state_dict: dict) -> bool:
    """在 Redis 中初始化任务元数据和 Stream 的第一条 init 事件

    必须在 submit_to_celery / run_direct_to_stream 之前调用，确保 SSE 客户端
    连接时 Stream 已存在（避免 xread 因 key 不存在而立即返回空）。

    Returns:
        True = 初始化成功；False = Redis 不可用（可以继续，兜底路径不依赖此函数）
    """
    r = _get_redis_sync()
    if r is None:
        logger.warning(f"[MQ] Redis 不可用，跳过任务初始化: task_id={task_id}")
        return False
    try:
        task_key = f"{TASK_KEY_PREFIX}{task_id}"
        r.setex(
            task_key,
            TASK_TTL,
            json.dumps({
                "status": "queued",
                "task_id": task_id,
                "created_at": int(time.time()),
                "query_preview": state_dict.get("query", "")[:100],
            }, ensure_ascii=False),
        )
        # 写入 init 事件，确保 Stream key 存在；SSE 端读取到 init 后跳过
        stream_key = f"{STREAM_KEY_PREFIX}{task_id}"
        r.xadd(
            stream_key,
            {"event": json.dumps({"type": "init", "task_id": task_id})},
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )
        r.expire(stream_key, STREAM_TTL)
        r.close()
        logger.info(f"[MQ] Redis 任务初始化完成: task_id={task_id}")
        return True
    except Exception as e:
        logger.warning(f"[MQ] Redis 任务初始化失败: task_id={task_id}, error={e}")
        return False


def submit_to_celery(
    task_id: str,
    state_dict: dict,
    trace_id: str,
) -> bool:
    """向 Celery/RabbitMQ 投递聊天任务（含熔断保护）

    Returns:
        True = 投递成功；False = 投递失败（调用方应降级到 run_direct_to_stream）
    """
    if not is_mq_healthy():
        logger.warning(f"[MQ] 熔断器已开启，跳过 MQ 投递，执行降级: task_id={task_id}")
        return False

    try:
        from tasks.chat_task import execute_chat
        execute_chat.apply_async(
            kwargs={
                "task_id": task_id,
                "state_dict": state_dict,
                "trace_id": trace_id,
            },
            task_id=task_id,
            queue="chat_queue",
            expires=300,
        )
        _on_mq_success()
        logger.info(f"[MQ] 任务投递成功: task_id={task_id}")
        return True
    except Exception as exc:
        _on_mq_failure()
        logger.error(f"[MQ] 任务投递失败: task_id={task_id}, error={exc}")
        return False


async def run_direct_to_stream(
    task_id: str,
    state_dict: dict,
    trace_id: str,
    run_graph_stream_fn: Callable,
    determine_answer_fn: Callable,
    post_process_fn: Optional[Callable] = None,
) -> None:
    """降级方案：直接在 FastAPI 事件循环中运行图，将结果写入 Redis Stream

    与 Celery Worker 路径产出完全相同的 Redis Stream 事件，
    SSE 客户端对两条路径完全透明。
    """
    import redis.asyncio as aioredis
    from core.state import State

    r_async = aioredis.from_url(REDIS_URL, decode_responses=True)
    stream_key = f"{STREAM_KEY_PREFIX}{task_id}"
    task_key = f"{TASK_KEY_PREFIX}{task_id}"

    logger.info(f"[MQ降级] 开始直接执行: task_id={task_id}")

    async def _write(event: dict) -> None:
        """异步写入 Redis Stream"""
        try:
            await r_async.xadd(
                stream_key,
                {"event": json.dumps(event, ensure_ascii=False)},
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            await r_async.expire(stream_key, STREAM_TTL)
        except Exception as e:
            logger.warning(f"[MQ降级] Redis Stream 写入失败: task_id={task_id}, error={e}")

    try:
        await r_async.setex(
            task_key,
            TASK_TTL,
            json.dumps({
                "status": "processing",
                "task_id": task_id,
                "updated_at": int(time.time()),
                "fallback": True,
            }, ensure_ascii=False),
        )

        state = State(
            thread_id=state_dict["thread_id"],
            query=state_dict.get("query", ""),
            history=None,
            tenant_id=state_dict.get("tenant_id", "default"),
            user_id=state_dict.get("user_id"),
            quoted_message=state_dict.get("quoted_message"),
            images=state_dict.get("images"),
        )

        stream_queue: asyncio.Queue = asyncio.Queue()
        streamed_tokens = []

        graph_task = asyncio.create_task(run_graph_stream_fn(state, stream_queue))
        logger.info(f"[MQ降级] 图执行 Task 已启动: task_id={task_id}")

        # 持续消费 token 并异步写入 Redis Stream
        while not graph_task.done():
            try:
                event = await asyncio.wait_for(stream_queue.get(), timeout=0.3)
                if event.get("type") == "token":
                    streamed_tokens.append(event.get("content", ""))
                await _write(event)
            except asyncio.TimeoutError:
                continue

        # 排空剩余队列
        while not stream_queue.empty():
            event = stream_queue.get_nowait()
            if event.get("type") == "token":
                streamed_tokens.append(event.get("content", ""))
            await _write(event)

        result = graph_task.result()

        # ── HITL 中断检测：图被 interrupt() 暂停，需将确认提示写入 Stream ──
        if isinstance(result, dict) and result.get("__interrupted__"):
            interrupt_info = result.get("__interrupt_info__") or {}
            _msg = interrupt_info.get("message", "即将执行高危操作，请确认是否继续？")
            _ops = interrupt_info.get("operations", [])
            _oid = interrupt_info.get("order_id", "")
            _tid = state_dict["thread_id"]
            hitl_state = await register_hitl_pending(
                thread_id=_tid, order_id=_oid,
                operations=_ops, message=_msg,
            )
            hitl_event = {
                "type": "hitl_confirm",
                "message": _msg,
                "operations": _ops,
                "order_id": _oid,
                "thread_id": _tid,
                "trace_id": trace_id,
                "expires_at": hitl_state.get("expires_at"),
                "timeout_seconds": HITL_TIMEOUT,
            }
            await _write(hitl_event)
            logger.info(f"[MQ降级] HITL 中断，hitl_confirm 事件已写入: task_id={task_id}")

            await r_async.setex(
                task_key, TASK_TTL,
                json.dumps({"status": "interrupted", "task_id": task_id, "updated_at": int(time.time())},
                           ensure_ascii=False),
            )
            return

        route, answer, sources = determine_answer_fn(result)

        # 非流式路径（订单/工单等）
        if not streamed_tokens and answer:
            await _write({"type": "token", "content": answer})

        # 写入完成事件
        await _write({
            "type": "done",
            "route": route,
            "answer": answer or "",
            "sources": sources or [],
            "trace_id": trace_id,
        })
        logger.info(f"[MQ降级] 图执行完成: task_id={task_id}, route={route}")

        await r_async.setex(
            task_key,
            TASK_TTL,
            json.dumps({"status": "completed", "task_id": task_id, "updated_at": int(time.time())},
                       ensure_ascii=False),
        )

        # 调用后处理（消息历史、摘要、记忆等）
        if post_process_fn:
            try:
                await post_process_fn(state_dict, answer or "")
            except Exception as e:
                logger.warning(f"[MQ降级] 后处理失败（非致命）: task_id={task_id}, error={e}")

    except Exception as exc:
        logger.error(f"[MQ降级] 执行失败: task_id={task_id}, error={exc}", exc_info=True)
        await _write({"type": "error", "message": f"处理失败，请稍后重试"})
        try:
            await r_async.setex(
                task_key,
                TASK_TTL,
                json.dumps({
                    "status": "failed",
                    "task_id": task_id,
                    "updated_at": int(time.time()),
                    "error": str(exc)[:200],
                }, ensure_ascii=False),
            )
        except Exception:
            pass

    finally:
        try:
            await r_async.aclose()
        except Exception:
            pass


async def read_task_status(task_id: str) -> Optional[dict]:
    """查询任务状态（用于 /chat/task/{task_id} 接口）

    Returns:
        任务状态字典，或 None（任务不存在）
    """
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            val = await r.get(f"{TASK_KEY_PREFIX}{task_id}")
            if val is None:
                return None
            return json.loads(val)
        finally:
            await r.aclose()
    except Exception as e:
        logger.warning(f"[MQ] 查询任务状态失败: task_id={task_id}, error={e}")
        return None


# ========== HITL 挂起状态管理（Redis）==========

HITL_KEY_PREFIX = "chat:hitl:"
HITL_TIMEOUT = int(os.getenv("HITL_TIMEOUT_SECONDS", "300"))  # 默认 5 分钟


async def register_hitl_pending(
    thread_id: str,
    order_id: str = "",
    operations: list = None,
    message: str = "",
) -> dict:
    """HITL 中断发生时，在 Redis 中注册挂起状态（带 TTL 自动过期）。

    Returns:
        包含 created_at / expires_at 的状态字典（供 hitl_confirm 事件携带给前端）
    """
    import redis.asyncio as aioredis

    now = int(time.time())
    expires_at = now + HITL_TIMEOUT
    hitl_state = {
        "thread_id": thread_id,
        "order_id": order_id,
        "operations": operations or [],
        "message": message,
        "created_at": now,
        "expires_at": expires_at,
        "status": "pending",
    }
    key = f"{HITL_KEY_PREFIX}{thread_id}"
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.setex(key, HITL_TIMEOUT, json.dumps(hitl_state, ensure_ascii=False))
            logger.info(f"[HITL] 注册挂起状态: thread_id={thread_id}, expires_at={expires_at}")
        finally:
            await r.aclose()
    except Exception as e:
        logger.warning(f"[HITL] Redis 注册失败（非致命）: thread_id={thread_id}, error={e}")
    return hitl_state


async def get_hitl_pending(thread_id: str) -> Optional[dict]:
    """查询指定 thread 是否有 HITL 挂起状态（未过期）。"""
    import redis.asyncio as aioredis

    key = f"{HITL_KEY_PREFIX}{thread_id}"
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            val = await r.get(key)
            if val is None:
                return None
            return json.loads(val)
        finally:
            await r.aclose()
    except Exception as e:
        logger.warning(f"[HITL] 查询挂起状态失败: thread_id={thread_id}, error={e}")
        return None


async def clear_hitl_pending(thread_id: str) -> None:
    """清除 HITL 挂起状态（用户已确认/拒绝/自动过期 后调用）。"""
    import redis.asyncio as aioredis

    key = f"{HITL_KEY_PREFIX}{thread_id}"
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.delete(key)
            logger.info(f"[HITL] 清除挂起状态: thread_id={thread_id}")
        finally:
            await r.aclose()
    except Exception as e:
        logger.warning(f"[HITL] 清除挂起状态失败: thread_id={thread_id}, error={e}")
