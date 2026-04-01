"""
初始化订单 PostgreSQL 数据库脚本，创建 `orders` 表并插入示例记录
"""
from pathlib import Path
import sys
import psycopg

# graph、core、tools 等包在 service/ 下，需把 service 根目录加入 path（不是 app/）
_APP_DIR = Path(__file__).resolve().parent
_SERVICE_ROOT = _APP_DIR.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))
    
import core.config as config

dsn = config.get_postgres_dsn("default")
conn = psycopg.connect(dsn)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, status TEXT, amount REAL, create_time TEXT, update_time TEXT)")

# 插入/更新一条示例订单，便于联调
c.execute(
    "INSERT INTO orders(order_id,status,amount,create_time,update_time) VALUES(%s,%s,%s,%s,%s) ON CONFLICT (order_id) DO UPDATE SET status=EXCLUDED.status, amount=EXCLUDED.amount, create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time",
    ("ORD20260101001", "待付款", 199.0, "2025-12-15 12:00:00", "2025-12-15 12:00:00"),
)
c.execute(
    "INSERT INTO orders(order_id,status,amount,create_time,update_time) VALUES(%s,%s,%s,%s,%s) ON CONFLICT (order_id) DO UPDATE SET status=EXCLUDED.status, amount=EXCLUDED.amount, create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time",
    ("ORD20260101002", "已发货", 19.0, "2025-12-15 12:00:00", "2025-12-15 12:00:00"),
)
c.execute(
    "INSERT INTO orders(order_id,status,amount,create_time,update_time) VALUES(%s,%s,%s,%s,%s) ON CONFLICT (order_id) DO UPDATE SET status=EXCLUDED.status, amount=EXCLUDED.amount, create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time",
    ("ORD20260101003", "已取消", 69.0, "2025-12-15 12:00:00", "2025-12-15 12:00:00"),
)
c.execute(
    "INSERT INTO orders(order_id,status,amount,create_time,update_time) VALUES(%s,%s,%s,%s,%s) ON CONFLICT (order_id) DO UPDATE SET status=EXCLUDED.status, amount=EXCLUDED.amount, create_time=EXCLUDED.create_time, update_time=EXCLUDED.update_time",
    ("ORD20260101004", "已完成", 55.0, "2025-12-15 12:00:00", "2025-12-15 12:00:00"),
)
conn.commit()
c.close()
conn.close()

print("orders initialized in PostgreSQL:", dsn)
