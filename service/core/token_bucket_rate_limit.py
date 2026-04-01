"""基于 Redis + Lua 的分布式令牌桶限流

使用单 key 哈希存储 tokens 与 last_ms，在 Lua 中原子完成补充与扣减，
多实例共享同一 Redis 即可实现分布式限流。
"""
from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)
import os
import time
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request

# ========== Lua：令牌桶（连续补充速率 rate tokens/s，桶容量 capacity）==========
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
if capacity == nil or rate == nil or cost == nil or now == nil then
  return {0, 0, -1}
end
local data = redis.call('HMGET', key, 'tokens', 'last_ms')
local tokens = tonumber(data[1])
local last_ms = tonumber(data[2])
if not tokens then
  tokens = capacity
  last_ms = now
end
local elapsed_ms = now - last_ms
if elapsed_ms < 0 then
  elapsed_ms = 0
end
local add = (elapsed_ms / 1000.0) * rate
tokens = math.min(capacity, tokens + add)
last_ms = now
if tokens >= cost then
  tokens = tokens - cost
  redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_ms', tostring(last_ms))
  local ttl = math.ceil((capacity / rate) * 1000) + 120000
  if ttl < 60000 then ttl = 60000 end
  redis.call('PEXPIRE', key, ttl)
  return {1, tokens, 0}
else
  redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_ms', tostring(last_ms))
  local ttl = math.ceil((capacity / rate) * 1000) + 120000
  if ttl < 60000 then ttl = 60000 end
  redis.call('PEXPIRE', key, ttl)
  local deficit = cost - tokens
  local retry_after = 0
  if rate > 0 then
    retry_after = math.ceil((deficit / rate) * 1000)
  else
    retry_after = 60000
  end
  return {0, tokens, retry_after}
end
"""


def get_remote_address(request: Request) -> str:
    """获取客户端 IP（与 Starlette/slowapi 常见实现一致）。"""
    if request.client is not None:
        return request.client.host or "127.0.0.1"
    return "127.0.0.1"


def _parse_limit_spec(spec: str) -> tuple[float, float]:
    """解析限流表达式。

    将 '60/minute' 转为 (桶容量, 每秒补充速率)。
    容量表示突发上限，平均速率 = 容量/时间窗口，与常见「窗口配额」语义一致。
    """
    spec = (spec or "").strip().lower().replace(" ", "")
    if "/" not in spec:
        raise ValueError(f"无效的限流表达式: {spec!r}")
    left, right = spec.split("/", 1)
    count = float(left)
    unit = right.strip()
    unit_sec = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}.get(unit)
    if unit_sec is None:
        raise ValueError(f"不支持的时间单位: {unit}")
    if count <= 0 or unit_sec <= 0:
        raise ValueError(f"无效的限流数值: {spec!r}")
    rate = count / float(unit_sec)
    return count, rate


def _rule_id(limit_spec: str, key_func_name: str) -> str:
    """生成短规则 ID，用于 Redis key 前缀。"""
    raw = f"{limit_spec}|{key_func_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _bucket_redis_key(scope: str, limit_spec: str, key_func_name: str) -> str:
    """根据限流维度字符串与规则生成 Redis 键名（哈希缩短、防注入）。"""
    scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:32]
    rid = _rule_id(limit_spec, key_func_name)
    return f"rl:tb:v1:{rid}:{scope_hash}"


def _find_request(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Request:
    """从 *args/**kwargs 中取出 Starlette Request（装饰器不注入，需从路由参数解析）。"""
    r = kwargs.get("request")
    if isinstance(r, Request):
        return r
    for a in args:
        if isinstance(a, Request):
            return a
    raise RuntimeError("token_bucket_limit: 未找到 Request 参数，请在路由中声明 request: Request")


async def _consume_token(
    redis_client: Any,
    redis_key: str,
    capacity: float,
    rate: float,
    cost: float = 1.0,
) -> tuple[bool, float, int]:
    """异步执行 EVAL，避免阻塞事件循环。

    返回：(是否放行, 剩余令牌数, 拒绝时建议等待毫秒数；-1 表示脚本异常)。
    """
    now_ms = int(time.time() * 1000)

    try:
        result = await redis_client.eval(
            _TOKEN_BUCKET_LUA,
            1,
            redis_key,
            str(capacity),
            str(rate),
            str(cost),
            str(now_ms),
        )
    except Exception as e:
        logger.warning("令牌桶 Lua 执行失败: key=%s, error=%s", redis_key, e)
        return False, 0.0, -1
    if not result or len(result) < 3:
        return False, 0.0, -1
    allowed = int(result[0]) == 1
    remaining = float(result[1])
    retry_ms = int(result[2])
    return allowed, remaining, retry_ms


def token_bucket_limit(
    limit_spec: str,
    key_func: Optional[Callable[[Request], str]] = None,
    *,
    cost: float = 1.0,
) -> Callable[[Any], Any]:
    """装饰异步视图：进入路由前按令牌桶扣减（Redis + Lua 原子操作）。

    参数:
        limit_spec: 如 '60/minute'、'1000/hour'
        key_func: 从 Request 生成限流维度字符串，未传时按客户端 IP
        cost: 单次请求消耗令牌数，默认 1

    Redis 不可用: 环境变量 RATE_LIMIT_FAIL_OPEN 默认为开启（放行并记警告日志）；
    设为 0/false/no 时返回 503。
    """
    if key_func is None:
        key_func = get_remote_address

    capacity, rate = _parse_limit_spec(limit_spec)
    kname = getattr(key_func, "__name__", "key")

    def decorator(func: Any) -> Any:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _find_request(args, kwargs)
            scope = key_func(request)
            redis_key = _bucket_redis_key(scope, limit_spec, kname)

            import core.config as _config

            r = await _config.get_redis()
            fail_open = os.environ.get("RATE_LIMIT_FAIL_OPEN", "1").lower() not in (
                "0",
                "false",
                "no",
            )
            if r is None:
                msg = "令牌桶限流需要 Redis，当前未连接"
                logger.warning("%s，scope=%s key=%s", msg, scope, redis_key)
                if fail_open:
                    return await func(*args, **kwargs)
                raise HTTPException(status_code=503, detail=msg)

            allowed, remaining, retry_ms = await _consume_token(
                r, redis_key, capacity, rate, cost=cost
            )
            if not allowed:
                if retry_ms < 0:
                    raise HTTPException(
                        status_code=500,
                        detail="限流脚本执行异常",
                    )
                retry_sec = max(1, (retry_ms + 999) // 1000)
                cap_int = int(capacity) if capacity == int(capacity) else capacity
                raise HTTPException(
                    status_code=429,
                    detail="请求过于频繁，请稍后再试",
                    headers={
                        "Retry-After": str(retry_sec),
                        "X-RateLimit-Limit": str(cap_int),
                        "X-RateLimit-Remaining": str(max(0, int(remaining))),
                    },
                )

            return await func(*args, **kwargs)

        return async_wrapper

    return decorator
