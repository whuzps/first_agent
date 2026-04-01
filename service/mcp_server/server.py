from typing import Any, Dict, Optional
import os

from graph.graph import run_graph
from core.state import State
from tools.service_tools import retrieve_kb, getdb, exec_sql

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FirstAgent MCP")

@mcp.tool()
def kb_search(query: str, k: int = 2, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    serialized, docs = retrieve_kb(query, tenant_id)
    try:
        sources = [getattr(d, "metadata", {}) for d in (docs or [])]
    except Exception:
        sources = []
    return {"context": serialized, "sources": sources[:k]}

@mcp.tool()
def order_lookup(text: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    payload = getdb(text)
    sql = payload.get("sql")
    params = payload.get("params")
    result = None
    try:
        result = exec_sql(sql, params, tenant_id)
    except Exception:
        result = None
    if result is None:
        result = payload.get("mock") or {}
    return result

@mcp.tool()
async def chat(query: str, thread_id: Optional[str] = None, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    state = State(thread_id=thread_id or "mcp", query=(query or "").strip(), history="", tenant_id=tenant_id or "default")
    result = await run_graph(state)
    route = result.get("route") or result.get("intent")
    sources = result.get("sources")
    order = result.get("order_summary")
    kb = result.get("kb_answer")
    human = result.get("human_handoff")

    answer = order or human or kb or ""
    return {"route": route, "answer": answer, "sources": sources}

@mcp.resource("kb://{tenant_id}/{query}")
def kb_resource(tenant_id: str, query: str) -> str:
    serialized, _ = retrieve_kb(query, tenant_id)
    return serialized

@mcp.resource("orders://{tenant_id}/{order_id}")
def order_resource(tenant_id: str, order_id: str) -> Dict[str, Any]:
    p = getdb(order_id)
    res = exec_sql(p.get("sql"), p.get("params"), tenant_id)
    if res is None:
        res = p.get("mock") or {}
    return res

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_PORT", "6278"))
    if transport == "http":
        mcp.run(transport="http", host="0.0.0.0", port=port)
    elif transport == "sse":
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        mcp.run()
