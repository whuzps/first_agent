from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import Tool
import core.config as config  # 供本模块内 config.xxx 使用
from core.prompts import SUGGEST_QUESTIONS_PROMPT_TEMPLATE, DEFAULT_SUGGEST_QUESTIONS
from tools.service_tools import (retrieve_kb, getdb, exec_sql)
import asyncio


_react_agent = None


def get_react_agent():
    global _react_agent
    if _react_agent is not None:
        return _react_agent
    try:
        llm = config.get_llm()
        def _kb_tool(q: str) -> str:
            s, _docs = retrieve_kb(q)
            return s or ""
        def _order_tool(text: str) -> str:
            p = getdb(text)
            if exec_sql is None:
                import json as _json
                return _json.dumps(p.get("mock"), ensure_ascii=False)
            res = exec_sql(p.get("sql"), p.get("params"))
            import json as _json
            return _json.dumps(res or p.get("mock"), ensure_ascii=False)
        tools = []
        if Tool is not None:
            tools = [
                Tool(name="kb_search", description="检索产品知识库并返回参考片段", func=_kb_tool),
                Tool(name="order_lookup", description="查询订单并返回结构化结果JSON", func=_order_tool),
            ]
        if create_agent is None or InMemorySaver is None or not tools:
            return None
        _react_agent = create_agent(
            model=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            system_prompt="你是建议生成助手。根据用户原问题与客服回答，生成3-5个可能继续追问的相关问题。这些问题需与上下文直接相关，促使用户深入了解具体细节，避免对回答内容的评价或建议，以开放式问句输出。"
        )
        return _react_agent
    except Exception:
        _react_agent = None
        return None


async def gen_suggest_questions(thread_id: str, question: str, answer: str) -> list:
    try:
        agent = get_react_agent()
        if agent is None:
            return DEFAULT_SUGGEST_QUESTIONS[:5]
        prompt = SUGGEST_QUESTIONS_PROMPT_TEMPLATE.format(question=str(question or ""), answer=str(answer or ""))
        messages = {"messages": [{"role": "user", "content": prompt}]}
        out = await asyncio.wait_for(agent.ainvoke(messages, {"configurable": {"thread_id": thread_id}}), timeout=10.0)
        msgs_out = out.get("messages", []) if isinstance(out, dict) else []
        final_text = ""
        for m in reversed(msgs_out):
            c = getattr(m, "content", None)
            if isinstance(c, str) and c.strip():
                final_text = c
                break
        if not final_text:
            content = getattr(out, "content", out)
            final_text = str(content) if isinstance(content, (str, bytes)) else ""
        if final_text.strip():
            lines = [s.strip() for s in final_text.splitlines() if s.strip()]
            cleaned = []
            for s in lines:
                t = s
                if t.startswith(("1.", "2.", "3.", "4.", "5.")):
                    t = t.split(".", 1)[1].strip()
                if t.startswith(("- ", "* ")):
                    t = t[2:].strip()
                cleaned.append(t)
            if not cleaned:
                cleaned = [final_text.strip()]
            if len(cleaned) < 3:
                fb = DEFAULT_SUGGEST_QUESTIONS
                cleaned = (cleaned + fb)[:5]
            return cleaned[:5]
        return DEFAULT_SUGGEST_QUESTIONS[:5]
    except Exception:
        return DEFAULT_SUGGEST_QUESTIONS[:5]
    

