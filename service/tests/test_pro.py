from prometheus_client import start_http_server, Counter, Gauge
import time
import random

# --------------------------
# 定义指标（固定不动）
# --------------------------
REQUEST_COUNT = Counter(
    "app_http_requests_total",
    "HTTP 请求总数",
    ["endpoint"]
)

ACTIVE_CONNECTIONS = Gauge(
    "app_active_connections",
    "当前活跃连接数"
)

# --------------------------
# 主程序：强制生成数据
# --------------------------
if __name__ == "__main__":
    # 启动 metrics 服务（端口 8000）
    start_http_server(9090)
    print("✅ 监控已启动：http://127.0.0.1:9090/metrics")
    print("✅ 正在生成指标数据...")

    # 死循环：一直生成数据
    while True:
        # 关键：必须执行这行，指标才会出现！
        REQUEST_COUNT.labels(endpoint="/api/user").inc()
        
        # 更新 Gauge
        ACTIVE_CONNECTIONS.set(random.randint(10, 50))
        
        time.sleep(1)