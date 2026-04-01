"""工具函数集合

包含：
- 知识库检索
- 订单数据库查询
- 未命中问题记录（PostgreSQL）
- 订单信息的自然语言格式化
- 转人工渠道封装
"""
import datetime
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# 模块级 logger
logger = logging.getLogger(__name__)

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
try:
    from service.rag.retrieval import Retrieval 
except Exception:
    from rag.retrieval import Retrieval 

try:
    from . import config
except Exception:
    import core.config as config
import core.postgres as postgres


class SafeToolExecutor:
    """安全的工具执行器"""
    
    def __init__(self, tools: List, fallback_model=None):
        self.tools = {tool.name: tool for tool in tools}
        self.fallback_model = fallback_model
        self.execution_history = []  # 记录执行历史
    
    def validate_args(self, tool_name: str, args: dict) -> Optional[str]:
        """在执行前用工具的 args_schema（Pydantic）校验参数。
        
        返回 None 表示校验通过；返回字符串表示校验失败的错误描述。
        """
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"工具 {tool_name} 不存在"
        
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            # 没有 schema 则跳过校验
            return None
        
        try:
            schema(**args)
            return None
        except Exception as e:
            return str(e)

    async def execute_with_fallback(self, tool_calls: List[Dict[str, Any]], state: Dict) -> List[ToolMessage]:
        """执行工具调用，执行前校验参数，失败时有降级策略"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id") or f"{tool_name}_call"
            
            # 第一步：验证工具是否存在
            if tool_name not in self.tools:
                logger.warning(f"SafeToolExecutor: 工具 {tool_name} 不存在")
                results.append(ToolMessage(
                    content=f"未找到工具: {tool_name}",
                    tool_call_id=tool_call_id,
                    additional_kwargs={"error": "ToolNotFound"}
                ))
                continue

            # 第二步：执行前校验参数（利用工具的 Pydantic args_schema）
            validation_error = self.validate_args(tool_name, tool_args)
            if validation_error:
                logger.warning(f"SafeToolExecutor: 工具 {tool_name} 参数校验失败: {validation_error}, args={tool_args}")
                # 参数有误时直接走降级，避免带着错误参数去调工具
                fallback_result = await self.try_fallback(
                    tool_name, tool_args,
                    ValueError(f"参数校验失败: {validation_error}"),
                    state
                )
                results.append(ToolMessage(
                    content=fallback_result,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"error": validation_error, "fallback_used": True}
                ))
                continue
            
            # 第三步：执行工具
            try:
                result = await self.execute_single_tool(tool_name, tool_args, state)
                results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                ))
                
            except Exception as e:
                # 记录错误历史
                self.execution_history.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "error": str(e),
                    "timestamp": datetime.datetime.now()
                })
                logger.error(f"SafeToolExecutor: 工具 {tool_name} 执行失败: {e}")
                
                # 尝试降级策略
                fallback_result = await self.try_fallback(tool_name, tool_args, e, state)
                
                results.append(ToolMessage(
                    content=fallback_result,
                    tool_call_id=tool_call_id,
                    additional_kwargs={
                        "error": str(e),
                        "fallback_used": True
                    }
                ))
        
        return results
    
    async def try_fallback(self, tool_name: str, args: dict, error: Exception, state: dict) -> str:
        """降级策略：依次尝试备用工具 → LLM模拟 → 兜底文案"""
        # 策略1：使用备用工具
        backup_tool = await self.get_backup_tool(tool_name)
        if backup_tool:
            try:
                return str(await backup_tool.ainvoke(args))
            except Exception as be:
                logger.warning(f"SafeToolExecutor: 备用工具也执行失败: {be}")
        
        # 策略2：使用LLM模拟
        if self.fallback_model:
            try:
                prompt = (
                    f"工具 {tool_name} 执行失败。\n"
                    f"参数：{args}\n"
                    f"错误：{error}\n"
                    f"请基于当前上下文提供一个合理的替代回答。\n"
                    f"上下文：{state.get('context', '')}"
                )
                resp = await self.fallback_model.ainvoke(prompt)
                return str(getattr(resp, "content", resp))
            except Exception as le:
                logger.warning(f"SafeToolExecutor: LLM降级也失败: {le}")
        
        # 策略3：返回有意义的错误信息
        return f"工具执行失败，请尝试其他方式：{error}"
    
    async def execute_single_tool(self, tool_name: str, args: dict, state: dict) -> Any:
        """执行单个工具（使用 ainvoke 保持与 LangChain tool 的兼容性）"""
        tool = self.tools[tool_name]
        return await tool.ainvoke(args)

    async def get_backup_tool(self, tool_name: str):
        """获取备用工具（约定命名为 {tool_name}_backup）"""
        backup_name = f"{tool_name}_backup"
        return self.tools.get(backup_name)


def retrieve_kb(query: str, tenant_id: Optional[str] = None) -> Tuple[str, List[Any]]:
    """根据查询在向量库中检索相似文档。

    返回值：
    - serialized：将检索到的文档以 "Source/Content" 形式串联的文本
    - docs：原始文档对象列表（用于提取来源 metadata）
    """
    # 根据租户获取集合名称
    collection_name = config.get_collection_name(tenant_id)
    k = 3

    logger.info(f"🔍 [Query Start] Q: {query} | Collection: {collection_name} | k: {k}")
    
    # 创建 Retrieval 实例
    retriever = Retrieval()
    docs = retriever.retrieve(query, collection_name=collection_name, top_k=k)
    
    serialized = "\n\n".join(
        [f"【来源: {d.metadata.get('source')}】\n{d.page_content}" for d in docs]
    )
    return serialized, docs


if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    query = "退货"
    serialized, docs = retrieve_kb(query)
    print("🔍 [Retrieval Result]:\n" + serialized)


def _parse_order_id(text: str) -> Optional[str]:
    """从文本中提取订单号，支持带或不带 `#` 前缀。

    规则：匹配 3~20 位数字，统一返回形如 `#20251114001` 的格式。
    """
    m = re.search(r"#?\d{3,20}", (text or ""))
    if not m:
        return None
    s = m.group(0)
    return s if s.startswith("#") else f"#{s}"


def getdb(order_text: str) -> Dict[str, Any]:
    """生成订单查询所需的 SQL 与参数"""
    oid = _parse_order_id(order_text)
    sql = (
        # PostgreSQL 使用 %s 占位符
        "SELECT order_id, status, amount, create_time, update_time FROM orders WHERE order_id = %s LIMIT 1"
    )
    params = [oid.lstrip("#")]
    return {"sql": sql, "params": params}


def exec_sql(sql: str, params: List[Any], tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """执行订单查询 SQL 并返回结构化结果。

    当数据库路径缺失或执行失败时返回 None。
    """
    dsn = config.get_postgres_dsn(tenant_id)
    try:
        with postgres.get_conn(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                logger.debug("exec_sql: sql=%s params=%s row=%s", sql, params, row)
        if row:
            return {
                "order_id": str(row[0]),
                "status": str(row[1]),
                "amount": float(row[2]) if row[2] is not None else None,
                "create_time": str(row[3]) if row[3] is not None else None,
                "update_time": str(row[4]) if len(row) > 4 and row[4] is not None else None,
            }
        return None
    except Exception:
        return None


def record_unanswered(text: str, user_id: Optional[str] = None, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """将未命中的用户问题记录到 PostgreSQL，便于人工回溯。"""
    dsn = config.get_postgres_dsn(tenant_id)
    ts = int(time.time())
    with postgres.get_conn(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS unanswered_questions (id BIGSERIAL PRIMARY KEY, tenant_id TEXT, user_id TEXT, text TEXT, created_at BIGINT)"
            )
            cur.execute(
                "INSERT INTO unanswered_questions(tenant_id, user_id, text, created_at) VALUES(%s, %s, %s, %s)",
                [tenant_id or "default", user_id, text, ts],
            )
        conn.commit()
    return {"ok": True, "db": "postgres"}


def format_order_nlg(item: Dict[str, Any]) -> str:
    """将订单信息整理为中文客服话术，便于直接回复用户。"""
    oid = item.get("order_id") or ""
    status = item.get("status") or "未知"
    amount = item.get("amount")
    create_time = item.get("create_time")
    update_time = item.get("update_time")
    parts: List[str] = []
    first = f"您的订单 {oid} 当前状态为{status}"
    if amount is not None:
        first += f"，订单金额为{amount}元"
    parts.append(first)
    if create_time:
        parts.append(f"订单创建时间为{create_time}")
    if update_time:
        parts.append(f"订单最近更新时间为{update_time}")
    return "。".join(parts)


def handoff_to_human(payload: Dict[str, Any]) -> Dict[str, Any]:
    """封装转人工的渠道与载荷。"""
    url = config.HUMAN_SUPPORT_URL
    return f"接入人工中，url = {url or 'default_url'}, payload = {payload}"


def validate_order_id_exists(order_id: str, tenant_id: str) -> bool:
    """订单号有效性校验（查DB）"""
    if not order_id:
        return False
    try:
        # 调用DB工具查询订单号是否存在
        payload = getdb(order_id)  # 传入有效订单号
        sql_text = payload.get("sql")
        params = [order_id]  # 已校验的订单号作为参数
        result = exec_sql(sql_text, list(params), tenant_id)
        return result is not None
    except Exception:
        return False


# -------------------------------------------------------------------------
# 新增订单操作工具
# -------------------------------------------------------------------------

class CancelOrderArgs(BaseModel):
    order_id: str = Field(..., description="订单号")
    reason: str = Field("无理由", description="取消原因")


@tool(args_schema=CancelOrderArgs)
def cancel_order(order_id: str, reason: str = "无理由") -> str:
    """取消订单。
    
    Args:
        order_id: 订单号
        reason: 取消原因
    """
    logger.info(f"取消订单: {order_id}, 原因: {reason}")
    # Mock data
    return f"订单 {order_id} 已成功取消。原因：{reason}"


class ModifyOrderAddressArgs(BaseModel):
    order_id: str = Field(..., description="订单号")
    new_address: str = Field(..., description="新的收货地址")


@tool(args_schema=ModifyOrderAddressArgs)
def modify_order_address(order_id: str, new_address: str) -> str:
    """修改订单收货地址。
    
    Args:
        order_id: 订单号
        new_address: 新的收货地址
    """
    logger.info(f"修改订单地址: {order_id}, 新地址: {new_address}")
    # Mock data
    return f"订单 {order_id} 的收货地址已修改为：{new_address}"


class ApplyRefundArgs(BaseModel):
    order_id: str = Field(..., description="订单号")
    reason: str = Field(..., description="退款原因")
    amount: Optional[float] = Field(None, description="退款金额（可选，默认全额）")


@tool(args_schema=ApplyRefundArgs)
def apply_refund(order_id: str, reason: str, amount: Optional[float] = None) -> str:
    """申请退款。
    
    Args:
        order_id: 订单号
        reason: 退款原因
        amount: 退款金额（可选，默认全额）
    """
    logger.info(f"申请退款: {order_id}, 原因: {reason}, 金额: {amount}")
    # Mock data
    return f"订单 {order_id} 的退款申请已提交。原因：{reason}，金额：{amount if amount else '全额'}"


class GetLogisticsInfoArgs(BaseModel):
    order_id: str = Field(..., description="订单号")


@tool(args_schema=GetLogisticsInfoArgs)
def get_logistics_info(order_id: str) -> str:
    """查询物流信息。
    
    Args:
        order_id: 订单号
    """
    logger.info(f"查询物流: {order_id}")
    # Mock data
    return f"订单 {order_id} 的物流状态：已发货，当前位置：北京分拣中心，预计明天送达。"


class GetOrderDetailArgs(BaseModel):
    order_id: str = Field(..., description="订单号")
    tenant_id: str = Field("default", description="租户ID")


@tool(args_schema=GetOrderDetailArgs)
def get_order_detail(order_id: str, tenant_id: str = "default") -> str:
    """查询订单详情。
    
    Args:
        order_id: 订单号
        tenant_id: 租户ID
    """
    logger.info(f"查询订单详情: {order_id}")
    # Reuse existing logic
    payload = getdb(order_id)
    sql_text = payload.get("sql")
    params = [order_id]
    result = exec_sql(sql_text, params, tenant_id)
    
    if not result:
        return f"未找到订单 {order_id} 的信息。"
        
    return format_order_nlg(result)
