"""PostgreSQL 连接与通用 SQL 执行封装。

说明：
- 统一管理连接池，减少高并发下反复建连开销。
- 所有 SQL 统一使用 `%s` 占位符。
"""

import logging

logger = logging.getLogger(__name__)
from contextlib import contextmanager
import threading
from typing import Any, Dict, Iterable, List, Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


_POOLS: Dict[str, ConnectionPool] = {}
_POOL_LOCK = threading.Lock()


def init_pool(dsn: str) -> ConnectionPool:
    """初始化或复用 PostgreSQL 连接池（按 DSN 隔离）。"""
    key = str(dsn or "").strip()
    if not key:
        raise ValueError("PostgreSQL DSN 不能为空")

    pool = _POOLS.get(key)
    if pool is not None:
        return pool

    # 双重检查锁：减少高并发下重复建池
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is not None:
            return pool
        pool = ConnectionPool(conninfo=key, min_size=2, max_size=20, timeout=10)
        _POOLS[key] = pool
        logger.info("PostgreSQL 连接池初始化完成: dsn=%s", key)
        return pool


def get_pool(dsn: str) -> ConnectionPool:
    return init_pool(dsn)


def close_all_pools() -> None:
    """关闭所有连接池（进程退出或测试清理时可调用）。"""
    with _POOL_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        try:
            pool.close()
        except Exception:
            pass


@contextmanager
def get_conn(dsn: str):
    """获取连接池连接（自动归还）。"""
    pool = get_pool(dsn)
    with pool.connection() as conn:
        yield conn


def execute(dsn: str, sql: str, params: Optional[Iterable[Any]] = None) -> None:
    """执行写操作 SQL。"""
    with get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
        conn.commit()


def fetchone(dsn: str, sql: str, params: Optional[Iterable[Any]] = None) -> Optional[tuple]:
    """执行查询并返回单行 tuple。"""
    with get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchone()


def fetchall(dsn: str, sql: str, params: Optional[Iterable[Any]] = None) -> List[tuple]:
    """执行查询并返回多行 tuple。"""
    with get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()


def fetchall_dict(dsn: str, sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    """执行查询并返回字典行。"""
    with get_conn(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params or [])
            return list(cur.fetchall())
