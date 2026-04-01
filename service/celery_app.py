"""Celery 应用配置

使用 RabbitMQ 作为 Broker（消息队列），Redis 作为 Result Backend。

队列设计：
  - chat_queue   : 主聊天队列，含死信路由、消息 TTL 和最大长度限制（防消息堆积）
  - chat_dlq     : 死信队列（DLQ），接收超时/拒绝的任务，供告警和人工介入使用

可靠性配置：
  - task_acks_late=True               : 任务执行完毕后才 ACK，防止 worker 崩溃丢消息
  - task_reject_on_worker_lost=True   : worker 异常断开时自动重新入队
  - worker_prefetch_multiplier=1      : 每次仅预取 1 个任务，避免消息堆积在 worker 内存中
  - expires=300                       : 任务在队列中最多存活 5 分钟，防止过期任务堆积
"""
import os
import sys
import logging

# 将 service 目录加入模块路径，确保任务模块能正确导入业务代码
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from celery import Celery
from kombu import Exchange, Queue
from core.logging_config import setup_logging, CELERY_WORKER_LOG_FORMAT, CELERY_TASK_LOG_FORMAT

# 初始化统一日志（Celery worker 进程入口）
setup_logging()

# 模块级 logger
logger = logging.getLogger(__name__)

# ========== 连接配置 ==========
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672//")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

logger.info("[Celery] 初始化: broker=%s..., backend=redis", RABBITMQ_URL[:30])

# ========== 交换机定义 ==========
# 主交换机：路由正常任务
_chat_exchange = Exchange("chat", type="direct", durable=True)
# 死信交换机（DLX）：接收超时/被拒绝的任务
_dlx_exchange = Exchange("chat_dlx", type="direct", durable=True)

# ========== Celery App ==========
celery_app = Celery(
    "cs_agent",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
    include=["tasks.chat_task"],
)

celery_app.conf.update(
    # ---------- 序列化 ----------
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # ---------- 时区 ----------
    timezone="Asia/Shanghai",
    enable_utc=True,
    # ---------- 超时（客服实时性要求：软超时 2min，硬超时 3min）----------
    task_soft_time_limit=120,
    task_time_limit=180,
    # ---------- 可靠性 ----------
    task_acks_late=True,               # 任务执行完毕后再 ACK，防止 worker 崩溃丢消息
    task_reject_on_worker_lost=True,   # worker 异常断开时重新入队（而非丢弃）
    worker_prefetch_multiplier=1,      # 每次仅预取 1 个任务，防止消息大量堆积在 worker 端
    # ---------- 结果存储 ----------
    result_expires=3600,               # Celery Result 在 Redis 中保留 1 小时
    # ---------- 队列路由 ----------
    task_routes={
        "tasks.chat_task.execute_chat": {"queue": "chat_queue"},
    },
    # ---------- 队列定义（含 DLQ）----------
    task_queues=(
        # 主聊天队列
        Queue(
            "chat_queue",
            _chat_exchange,
            routing_key="chat",
            queue_arguments={
                # 死信路由：超时/拒绝的消息自动流转到 DLQ
                "x-dead-letter-exchange": "chat_dlx",
                "x-dead-letter-routing-key": "dlq",
                # 队列最大容量（超出后新消息发往 DLQ，防止无限堆积）
                "x-max-length": 1000,
                # 消息最大存活时间 5 分钟（毫秒），过期消息进 DLQ
                "x-message-ttl": 300_000,
            },
        ),
        # 死信队列（DLQ）：接收所有"问题任务"，供运营告警和人工复核
        Queue(
            "chat_dlq",
            _dlx_exchange,
            routing_key="dlq",
        ),
    ),
    task_default_queue="chat_queue",
    task_default_exchange="chat",
    task_default_routing_key="chat",
    # ---------- Broker 连接重试 ----------
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    # ---------- Worker 日志（与全局格式保持一致）----------
    worker_log_format=CELERY_WORKER_LOG_FORMAT,
    worker_task_log_format=CELERY_TASK_LOG_FORMAT,
)
