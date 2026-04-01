"""熔断器模块

实现标准三态熔断器（Circuit Breaker）：
- CLOSED（闭合）：正常放行请求
- OPEN（断路）：快速失败，拒绝请求
- HALF_OPEN（半开）：有限放行，探测服务是否恢复

使用方式：
    breaker = get_breaker("llm_main", failure_threshold=5, recovery_timeout=60.0)
    result = await breaker.call(llm.ainvoke, prompt)
    # 同步函数用 asyncio.to_thread 包装：
    result = await breaker.call(asyncio.to_thread, retrieve_kb, query, tenant_id)
"""

import asyncio
import time
import logging

logger = logging.getLogger(__name__)
from enum import Enum
from typing import Any, Callable


class CircuitState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"        # 闭合：正常请求
    OPEN = "open"            # 断路：拒绝请求
    HALF_OPEN = "half_open"  # 半开：探测恢复


class CircuitBreakerError(Exception):
    """熔断器断路时抛出；上层节点可据此走降级逻辑，不应触发重试"""
    pass


class CircuitBreaker:
    """单个资源的熔断器

    Args:
        name: 熔断器名称（用于日志和监控）
        failure_threshold: 连续失败 N 次后断路（默认 5）
        recovery_timeout: 断路后 N 秒进入半开探测（默认 60s）
        success_threshold: 半开期间连续成功 N 次后恢复闭合（默认 2）
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        # 仅用于状态切换的互斥，不在函数执行期间持有
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用 func。

        断路时直接抛出 CircuitBreakerError（快速失败），
        不持有锁执行 func，允许并发请求正常通过。
        """
        # ── 1. 状态检查（持锁，仅做判断与状态跳转，不执行 func）──
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    # 到达恢复探测时间，进入半开
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"[熔断器:{self.name}] OPEN → HALF_OPEN"
                        f"（已断路 {elapsed:.0f}s，开始探测）"
                    )
                else:
                    remaining = int(self.recovery_timeout - elapsed)
                    raise CircuitBreakerError(
                        f"熔断器 [{self.name}] 断路中，预计 {remaining}s 后进入探测期"
                    )

        # ── 2. 执行 func（不持锁，允许并发）──
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitBreakerError:
            # 自身抛出的断路异常不再计入失败
            raise
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self):
        """请求成功后更新状态"""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        f"[熔断器:{self.name}] HALF_OPEN → CLOSED"
                        f"（连续成功 {self.success_threshold} 次，服务已恢复）"
                    )
            elif self._state == CircuitState.CLOSED:
                # 成功时重置连续失败计数
                self._failure_count = 0

    async def _on_failure(self):
        """请求失败后更新状态"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开探测失败，重新断路
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[熔断器:{self.name}] HALF_OPEN → OPEN（探测失败，重新断路）"
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"[熔断器:{self.name}] CLOSED → OPEN"
                        f"（连续失败 {self._failure_count} 次，触发熔断）"
                    )

    def snapshot(self) -> dict:
        """返回熔断器当前状态快照（供监控/健康检查使用）"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
        }


# ── 全局熔断器注册表（进程内单例） ──
_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2,
) -> CircuitBreaker:
    """获取或创建指定名称的熔断器（懒加载单例）

    多次调用同一 name 时，仅首次创建实例并复用，
    后续调用的 threshold 参数会被忽略。
    """
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )
    return _BREAKERS[name]


def get_all_snapshots() -> dict:
    """返回所有已注册熔断器的状态快照（供健康检查接口使用）"""
    return {name: cb.snapshot() for name, cb in _BREAKERS.items()}
