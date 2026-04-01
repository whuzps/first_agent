from typing import Any, Dict
import core.config as config  # 供本模块内 config.xxx 使用
import re


async def handle_command(query_text: str, thread_id: str):
    """处理 /help /history /reset 指令；未匹配返回 None"""
    if query_text.strip().startswith("/"):
        cmd = query_text.strip().lower()
        if cmd.startswith("/help"):
            return {
                "commands": [
                    {"cmd": "/help", "desc": "查看所有快捷指令"},
                    {"cmd": "/history", "desc": "查看最近5轮对话"},
                    {"cmd": "/reset", "desc": "重置当前会话上下文"},
                ]
            }
        if cmd.startswith("/history"):
            msgs = await config.get_session_messages(thread_id, maxlen=5)
            return {"history": msgs}
        if cmd.startswith("/reset"):
            await config.reset_session(thread_id)
            return {"reset": True}
    return None


def determine_answer(result: Dict[str, Any]):
    """根据模型输出确定输出类型，优先级：clarify > order > human > kb > ask_human > fallback"""
    route = result.get("route") or result.get("intent")
    sources = result.get("sources")
    clarify = result.get("clarify_question")
    ask_human = result.get("ask_human")
    order = result.get("order_summary")
    human = result.get("human_handoff")
    kb = result.get("kb_answer")
    fallback = result.get("fallback")
    if clarify:
        answer = clarify
        route = route or "clarify"
    elif order:
        answer = order
    elif human:
        answer = human
    elif kb:
        answer = kb
    elif ask_human:
        answer = ask_human
    elif fallback:
        answer = fallback
    else:
        answer = ""
    return route, answer, sources


def validate_order_id_format(order_id: str) -> bool:
    """订单号格式校验（示例：纯数字+固定长度，可自定义）"""
    if not order_id:
        return False
    # 规则：8-16位数字，以"ORD"开头（示例规则，可自定义）
    pattern = r"^ORD\d{8,16}$"
    return bool(re.match(pattern, order_id.strip()))

