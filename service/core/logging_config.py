"""统一日志配置模块

全项目所有入口点（FastAPI app、Celery worker、脚本）均应调用 setup_logging()，
以确保日志格式、级别、Handler 在整个进程中保持一致。

统一格式：
    %(asctime)s | %(levelname)-8s | %(name)s | %(message)s

带链路追踪时（FastAPI 入口专用）：
    %(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s

使用示例::

    # 普通入口（Celery worker、脚本等）
    from core.logging_config import setup_logging
    setup_logging()

    # FastAPI 入口（含请求追踪 ID）
    from core.logging_config import setup_logging
    setup_logging(with_trace=True, trace_filter=MyTraceFilter())

    # 评估/测试脚本（静默第三方库噪音）
    from core.logging_config import setup_logging
    setup_logging(silence_libs=True, script_logger_name="eval")
"""

import logging
from typing import Optional

# ── 统一格式常量 ─────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_WITH_TRACE = "%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Celery worker_log_format / worker_task_log_format 保持与 LOG_FORMAT 一致
CELERY_WORKER_LOG_FORMAT = LOG_FORMAT
CELERY_TASK_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | task=%(task_name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    with_trace: bool = False,
    trace_filter: Optional[logging.Filter] = None,
    silence_libs: bool = False,
    script_logger_name: Optional[str] = None,
) -> None:
    """配置全局日志格式与级别。

    应在进程入口处调用一次，内部使用 ``force=True`` 确保覆盖任何已存在的配置
    （例如被 import 时提前触发的 basicConfig）。

    Parameters
    ----------
    level:
        根 logger 级别，默认 ``logging.INFO``。
    with_trace:
        是否在格式中包含 ``trace_id`` 字段（FastAPI 请求入口专用）。
    trace_filter:
        ``logging.Filter`` 实例，用于向每条日志记录注入 ``trace_id``；
        仅在 ``with_trace=True`` 时有意义。
    silence_libs:
        评估/脚本模式：将根 logger 级别设为 ``WARNING``，屏蔽第三方库噪音；
        可配合 ``script_logger_name`` 单独保留某个 logger 的 INFO 输出。
    script_logger_name:
        仅在 ``silence_libs=True`` 时有效，将指定名称的 logger 单独抬高至 INFO。
    """
    fmt = LOG_FORMAT_WITH_TRACE if with_trace else LOG_FORMAT
    # 脚本模式下根 logger 静默第三方库，其余情况使用传入级别
    root_level = logging.WARNING if silence_libs else level

    logging.basicConfig(
        level=root_level,
        format=fmt,
        datefmt=DATE_FORMAT,
        force=True,
    )

    # 注入链路追踪 filter（为已绑定的所有 handler 添加）
    if trace_filter is not None:
        for handler in logging.getLogger().handlers:
            handler.addFilter(trace_filter)

    # 脚本模式：单独抬高指定 logger 到 INFO
    if silence_libs and script_logger_name:
        logging.getLogger(script_logger_name).setLevel(logging.INFO)

    logging.getLogger(__name__).debug("日志系统初始化完成: level=%s, with_trace=%s", root_level, with_trace)
