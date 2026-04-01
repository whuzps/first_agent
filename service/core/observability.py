from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)
import os
from typing import List

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"
    logger.warning("prometheus_client not installed, using fallback mock")

# ── 全局静态定义 Prometheus 指标（模块级单例，避免重复注册）────────────────
if PROMETHEUS_AVAILABLE:
    NODE_DURATION = Histogram(
        'langgraph_node_duration_seconds',
        '节点执行耗时',
        ['node']                    # 去掉 success 标签，降低 Histogram 基数
    )
    NODE_EXECUTIONS = Counter(
        'langgraph_node_executions_total',
        '节点执行次数',
        ['node', 'status']
    )
    GRAPH_DURATION = Histogram(
        'langgraph_graph_duration_seconds',
        '图执行耗时'
    )
    GRAPH_COST = Counter(           # 改为 Counter：累计成本只增不减
        'langgraph_graph_cost_total',
        '图执行成本'
    )
    TTFT_DURATION = Histogram(
        'chat_stream_ttft_seconds',
        '流式对话首字响应时间（TTFT, Time To First Token）',
        buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    )

    # ── /chat/stream 专项监控指标 ──────────────────────────────────────────────

    # QPS：请求总数计数器，可通过 rate() 函数换算出每秒/每分钟请求量
    CHAT_STREAM_REQUESTS_TOTAL = Counter(
        'chat_stream_requests_total',
        '/chat/stream 接口接收到的请求总数（用于 QPS 监控）',
        ['path', 'tenant_id'],          # path 固定为 "/chat/stream"，tenant_id 区分租户
    )

    # 消息堆积量：当前正在处理中（未完成）的请求数，实时反映系统压力
    CHAT_STREAM_ACTIVE_REQUESTS = Gauge(
        'chat_stream_active_requests',
        '/chat/stream 当前正在处理中的请求数（消息堆积量）',
    )

    # 发送消息成功率：按 status=success/error 分类计数，用于计算成功率
    CHAT_STREAM_MESSAGES_TOTAL = Counter(
        'chat_stream_messages_total',
        '/chat/stream 消息发送结果统计（status=success 表示成功，error 表示失败）',
        ['status'],                      # success | error
    )

    # 消息消费耗时：从请求到达到收到 done/error 事件的端到端耗时
    CHAT_STREAM_CONSUME_DURATION = Histogram(
        'chat_stream_consume_duration_seconds',
        '/chat/stream 消息消费端到端耗时（从请求到达到 done/error 事件）',
        buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0],
    )

    # ── KB 语义缓存专项监控指标 ────────────────────────────────────────────

    # 缓存请求总数（命中 + 未命中），用于计算命中率
    KB_CACHE_REQUESTS_TOTAL = Counter(
        'kb_semantic_cache_requests_total',
        'KB 语义缓存请求总数（result=hit 表示命中，miss 表示未命中）',
        ['tenant_id', 'result'],   # result: "hit" | "miss"
    )

    # KB 节点端到端响应耗时，按缓存状态区分（便于对比有缓存 vs 无缓存延迟）
    KB_RESPONSE_DURATION = Histogram(
        'kb_response_duration_seconds',
        'KB 节点端到端响应耗时（cache_status=hit 时跳过了 RAG 检索，miss 时走完整 RAG 流程）',
        ['cache_status', 'node'],  # cache_status: "hit" | "miss"；node: "kb_node" | "kb_node_stream"
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0],
    )

    # 缓存相似度分布直方图（仅未命中时记录最高相似度，用于调优阈值）
    KB_CACHE_SIMILARITY = Histogram(
        'kb_semantic_cache_similarity',
        'KB 语义缓存未命中时的最高余弦相似度分布（用于评估阈值是否合理）',
        ['tenant_id'],
        buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.92, 0.95, 1.0],
    )

else:
    NODE_DURATION = None
    NODE_EXECUTIONS = None
    GRAPH_DURATION = None
    GRAPH_COST = None
    TTFT_DURATION = None
    CHAT_STREAM_REQUESTS_TOTAL = None
    CHAT_STREAM_ACTIVE_REQUESTS = None
    CHAT_STREAM_MESSAGES_TOTAL = None
    CHAT_STREAM_CONSUME_DURATION = None
    KB_CACHE_REQUESTS_TOTAL = None
    KB_RESPONSE_DURATION = None
    KB_CACHE_SIMILARITY = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"


_SERVER_STARTED = False

def start_prometheus(port: int = 9090):
    """Start Prometheus HTTP server if available and not already started."""
    global _SERVER_STARTED
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available, skipping server start.")
        return

    if _SERVER_STARTED:
        logger.warning(f"Prometheus server already started (or attempted).")
        return

    try:
        start_http_server(port)
        _SERVER_STARTED = True
        logger.info(f"✅ Prometheus metrics server started on port {port}")
    except OSError as e:
        logger.warning(f"⚠️ Failed to start Prometheus server on port {port} (Address already in use?): {e}")
    except Exception as e:
        logger.error(f"❌ Failed to start Prometheus metrics server on port {port}: {e}")


@dataclass
class GraphMetrics:
    """图执行指标"""
    thread_id: str
    start_time: datetime
    end_time: datetime
    total_nodes_executed: int
    failed_nodes: List[str]
    retry_count: int
    total_tokens: int
    total_cost: float


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics_buffer = []
    
    async def track_node_execution(self, node_name: str, duration: float, success: bool):
        """追踪节点执行（NODE_DURATION 已去掉 success 标签，降低 Histogram 基数；duration 参数单位为秒，上报时转换为毫秒）"""
        # duration_ms = duration * 1000  # 秒转毫秒
        if PROMETHEUS_AVAILABLE:
            try:
                NODE_DURATION.labels(node=node_name).observe(duration)
                status = "success" if success else "failure"
                NODE_EXECUTIONS.labels(node=node_name, status=status).inc()
            except Exception as e:
                logger.error(f"Error tracking node execution metrics: {e}")

        logger.info(f"✅ Tracked node execution: {node_name}, duration: {duration * 1000:.2f}ms, success: {success}")

    async def track_kb_cache(
        self,
        tenant_id: str,
        hit: bool,
        duration: float,
        node: str = "kb_node",
        similarity: float = 0.0,
    ):
        """追踪 KB 语义缓存命中/未命中事件。

        Args:
            tenant_id:  租户 ID（用于分标签统计）
            hit:        True 表示缓存命中，False 表示未命中
            duration:   KB 节点端到端耗时（秒）
            node:       节点名称（kb_node 或 kb_node_stream）
            similarity: 缓存未命中时的最高余弦相似度（用于阈值调优）
        """
        result = "hit" if hit else "miss"
        cache_status = "hit" if hit else "miss"
        duration_ms = duration * 1000  # 秒转毫秒，与 NODE_DURATION 保持一致

        if PROMETHEUS_AVAILABLE:
            try:
                # 缓存请求计数（命中率分子/分母）
                KB_CACHE_REQUESTS_TOTAL.labels(
                    tenant_id=tenant_id, result=result
                ).inc()
                # KB 节点响应耗时（有/无缓存对比）
                KB_RESPONSE_DURATION.labels(
                    cache_status=cache_status, node=node
                ).observe(duration)
                # 未命中时记录相似度分布（辅助阈值调优）
                if not hit and similarity > 0:
                    KB_CACHE_SIMILARITY.labels(tenant_id=tenant_id).observe(similarity)
            except Exception as e:
                logger.error(f"记录 KB 缓存指标异常：{e}")

        logger.info(
            f"📊 KB 缓存指标：node={node}，tenant={tenant_id}，"
            f"result={result}，duration={duration_ms:.1f}ms，similarity={similarity:.4f}"
        )

    async def track_graph_execution(self, metrics: GraphMetrics):
        """追踪整个图的执行（GRAPH_COST 改为 Counter.inc，记录本次执行增量成本）"""
        if PROMETHEUS_AVAILABLE:
            try:
                duration = (metrics.end_time - metrics.start_time).total_seconds()
                GRAPH_DURATION.observe(duration)
                if metrics.total_cost > 0:
                    GRAPH_COST.inc(metrics.total_cost)  # Counter：累加增量，不覆盖
            except Exception as e:
                logger.error(f"Error tracking graph execution metrics: {e}")
        
        # 存储详细日志用于分析
        self.metrics_buffer.append(metrics)
        
        # 定期批量写入
        if len(self.metrics_buffer) >= 5:
            await self.flush_metrics()
    
    async def flush_metrics(self):
        """批量写入指标"""
        if not self.metrics_buffer:
            return
        
        # 写入数据仓库或日志系统
        try:
            batch_data = [
                json.dumps(m.__dict__, default=str) 
                for m in self.metrics_buffer
            ]
            # 实际写入逻辑
            await self.write_to_database(batch_data)
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
        finally:
            self.metrics_buffer.clear()

    async def write_to_database(self, batch_data: List[str]):
        """写入数据库/数据仓库"""
        try:
            _logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(_logs_dir, exist_ok=True)
            with open(os.path.join(_logs_dir, "graph_metrics.json"), "a") as f:
                for item in batch_data:
                    f.write(item + "\n")
        except Exception as e:
            logger.error(f"Error writing metrics to file: {e}")
