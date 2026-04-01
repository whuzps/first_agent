import logging

logger = logging.getLogger(__name__)
import traceback
from typing import List, Any

try:
    from . import config
except Exception:
    import core.config as config
import core.postgres as postgres

import gradio as gr


def load_orders(tenant_id: str = "default") -> List[List[Any]]:
    try:
        with postgres.get_conn(config.get_postgres_dsn(tenant_id)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT order_id, status, amount, create_time, update_time FROM orders"
                )
                rows = cur.fetchall()
    except Exception:
        logger.error("❎ error loading orders db")
        # 打印错误堆栈信息
        logger.error(traceback.format_exc())
        return []
    return [
        [
            str(order_id or ""),
            str(status or ""),
            float(amount) if amount is not None else None,
            str(create_time or ""),
            str(update_time or ""),
        ]
        for (order_id, status, amount, create_time, update_time) in rows
    ]


def build_orders_ui():
    try:
        headers = ["order_id", "status", "amount", "create_time", "update_time"]
        with gr.Blocks() as demo:
            gr.Markdown("订单数据库")
            tenant = gr.Textbox(label="tenant", value="default")
            # 如果数据为空列表，Gradio可能会报错，传入None更安全
            data = load_orders("default")
            if not data:
                data = None
                
            df = gr.Dataframe(headers=headers, value=data, interactive=False)
            btn = gr.Button("刷新")
            btn.click(fn=load_orders, inputs=tenant, outputs=df, api_name="lambda", show_progress="minimal")
    except Exception:
        logger.error("❎ error")
        # 打印错误堆栈信息
        logger.error(traceback.format_exc())
        return False
    return demo


def mount_gradio(app):
    try:
        demo = build_orders_ui()
    except Exception:
        return False
    try:
        if hasattr(gr, "mount_gradio_app"):
            gr.mount_gradio_app(app, demo, path="/listdb")
        else:
            from gradio.routes import App as GradioApp
            app.mount("/listdb", GradioApp.create_app(demo))
        return True
    except Exception:
        return False
