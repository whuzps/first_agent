import logging
import asyncio
import os
import re
import inspect
import threading
import time
from typing import Any, Dict, Optional
from contextvars import ContextVar as _CtxVar
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from core.hander import validate_order_id_format
from core.observability import MetricsCollector
from core.preprocessing import clean_input
from core.retry_policy import SmartRetryPolicy
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import json
import traceback
from datetime import datetime
from core.robust import Robust
from core.state import State, StateStatus
from core.prompts import (
    RAG_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT, ORDER_AGENT_PROMPT_TEMPLATE,
    REACT_ORDER_PROMPT_TEMPLATE,
    INTENT_SLOT_PROMPT, UNIFIED_INTENT_PROMPT,
    CLARIFY_INTENT_PROMPT, CLARIFY_SLOT_PROMPT,
)
import core.config as config
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage as LCToolMessage
from tools.service_tools import (retrieve_kb, record_unanswered, handoff_to_human, validate_order_id_exists, cancel_order, modify_order_address, apply_refund, get_logistics_info, get_order_detail, SafeToolExecutor)
from tools.skill_tool import lookup_skill, read_reference, SkillRegistry
import memory.store as memory_module
from core.circuit_breaker import get_breaker, CircuitBreakerError
from core.semantic_cache import get_semantic_cache

# 模块级 logger，避免直接使用根 logger
logger = logging.getLogger(__name__)

"""对话路由与节点执行图

使用 LangGraph 构建状态机，利用并行 fan-out/fan-in 机制优化记忆加载性能：

并行分支结构：
  START ──┬── load_memory(短期记忆) → intent(意图识别) ──┬── intent_post(汇合) → 路由分发
          └── load_long_term_memory(长期记忆) ──────────┘

节点说明：
- load_memory 节点：加载短期记忆（最近 N 轮原文 + 历史摘要）
- load_long_term_memory 节点：加载长期记忆（语义检索用户画像/偏好/历史事实），与分支 1 并行执行
- intent 节点：多阶段意图识别（规则快速路径 → LLM 联合识别 → 意图转移感知 → 置信度校准 → 槽位校验）
  意图分类：faq（知识库问答）/ order（订单操作）/ chitchat（闲聊）/ human（转人工）
- intent_post 节点：并行分支汇合点，合并意图识别结果与长期记忆
- kb 节点：检索知识库并回答（faq 意图）
- order 节点：查询订单并生成客服话术（order 意图，含 order_id/action/address 槽位）
- handoff 节点：转人工时输出渠道（human 意图）
- direct 节点：闲聊/问候直接回答（chitchat 意图）
"""
class Route(BaseModel):
    """结构化路由输出模型，用于约束 LLM 返回的意图标签。"""
    step: Literal["faq", "order", "chitchat", "human"] = Field(None)


llm = config.get_llm()
small_llm = config.get_small_llm()
vl_llm = config.get_vl_llm()

VALID_INTENTS = {"faq", "order", "chitchat", "human"}
metrics_collector = MetricsCollector()


# ── 多模态辅助函数 ──────────────────────────────────────────────────────
def _has_images(state: State) -> bool:
    """判断当前请求是否包含图片"""
    return bool(getattr(state, "images", None))


def _get_effective_llm(state: State):
    """根据是否包含图片选择 LLM：有图片 → VL 模型，无图片 → 默认模型"""
    return vl_llm if _has_images(state) else llm


def _build_llm_input(prompt: str, state: State):
    """构建 LLM 输入：有图片时生成多模态 HumanMessage，无图片时返回纯文本。

    Qwen VL 通过 OpenAI 兼容接口接收图片，格式为 HumanMessage(content=[
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/...;base64,..." 或 https://..."}},
    ])
    """
    if not _has_images(state):
        return prompt

    content_parts = [{"type": "text", "text": prompt}]
    for img in state.images:
        if img.startswith(("http://", "https://")):
            url = img
        else:
            url = f"data:image/jpeg;base64,{img}"
        content_parts.append({"type": "image_url", "image_url": {"url": url}})

    return HumanMessage(content=content_parts)

# ── 熔断器：各外部资源独立一个实例（进程内单例）──
# 主 LLM（kb/direct/order 节点使用）：连续失败 5 次后断路 60s
_llm_main_breaker = get_breaker("llm_main", failure_threshold=5, recovery_timeout=60.0, success_threshold=2)
# 小 LLM（intent/rewrite/clarify 节点使用）：连续失败 5 次后断路 60s
_llm_small_breaker = get_breaker("llm_small", failure_threshold=5, recovery_timeout=60.0, success_threshold=2)
# KB 检索：连续失败 3 次后断路 30s
_kb_retrieval_breaker = get_breaker("kb_retrieval", failure_threshold=3, recovery_timeout=30.0, success_threshold=2)
# 订单工具调用：连续失败 5 次后断路 30s
_order_tools_breaker = get_breaker("order_tools", failure_threshold=5, recovery_timeout=30.0, success_threshold=2)

# 模糊意图反问阈值：低于此值触发反问
_INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.55"))

# ── ReAct 推理配置 ──
# 是否启用 ReAct 推理模式（可通过环境变量关闭，降级回单次调用模式）
_REACT_ENABLED = os.getenv("REACT_ENABLED", "true").lower() in ("true", "1", "yes")
# ReAct 推理最大迭代次数（防止无限循环）
_REACT_MAX_ITERATIONS = int(os.getenv("REACT_MAX_ITERATIONS", "3"))

# 各意图的必需槽位定义
_REQUIRED_SLOTS: Dict[str, list] = {
    "order": ["order_id", "action"],
}
def _is_valid_slot_value(v) -> bool:
    """判断槽位值是否有效：排除 None、空字符串、LLM 产出的字面量 "null" """
    if v is None:
        return False
    if isinstance(v, str) and (not v.strip() or v.strip().lower() == "null"):
        return False
    return True


# 各槽位的中文描述，用于反问话术
_SLOT_DISPLAY: Dict[str, str] = {
    "order_id": "订单号（格式示例：ORD12345678）",
    "action": "要执行的操作（如：查询详情、取消订单、申请退款、修改地址、查物流）",
    "address": "新的收货地址",
}


_SWITCH_SIGNALS = [
    "另外", "对了", "换个话题", "顺便问下", "顺便问一下",
    "还有个问题", "还想问", "还有一个问题",
    "算了", "不用了", "那个不管了",
    "我想问", "我还想",
]


def _detect_switch_signal(text: str) -> bool:
    """检测用户输入中是否包含意图转移信号词"""
    t = (text or "").lower()
    return any(s in t for s in _SWITCH_SIGNALS)


def _keywords_intent(text: str) -> Optional[str]:
    """基于规则的快速意图分类（faq / order / chitchat / human）。

    优先级：human > order（含单号）> order（操作动词）> faq > chitchat
    不确定场景返回 None，交由 LLM 判断。
    """
    t = (text or "").lower()

    # 1. 转人工（最高优先级）
    if any(k in t for k in ["人工", "转人工", "人工客服", "找客服", "人工服务", "接入人工", "找主管", "投诉主管"]):
        return "human"

    # 2. 含订单号 → order（支持 ORD12345678 和 #20251114001 两种格式）
    if re.search(r"ord\d{6,16}", t, re.IGNORECASE) or re.search(r"#\d{8,}", t):
        return "order"

    # 3. 明确订单操作动词（无订单号，节点会继续追问）
    order_action_keywords = [
        "取消订单", "取消我的订单",
        "申请退款", "发起退款", "提交退款",
        "修改地址", "改地址", "修改收货地址", "改收货地址",
        "查看物流", "查物流", "物流到哪了", "追踪包裹",
        "帮我取消", "帮我退款", "帮我修改", "帮我改",
        "给我取消", "给我退款", "给我修改",
        "怎么还没发货", "订单还没发",
    ]
    if any(k in t for k in order_action_keywords):
        return "order"

    # 4. FAQ —— 合并售前/售后/通用知识库查询
    faq_keywords = [
        # 售后政策
        "退货政策", "换货政策", "退换货政策", "售后政策", "退款政策",
        "退货流程", "换货流程", "退款流程", "怎么退货", "如何退货",
        "怎么换货", "如何换货", "申请换货", "提交换货",
        "退货运费", "退货邮费", "谁承担运费",
        "商品坏了", "商品损坏", "商品有缺陷", "商品有问题",
        "收到破损", "收到损坏", "质量问题", "产品故障", "不能正常使用",
        "保修", "在保修期", "维修", "保修政策",
        # 售前咨询
        "如何购买", "怎么买", "如何下单", "怎么下单", "如何订购", "购买流程",
        "付款方式", "支付方式", "怎么付款", "如何支付",
        "支持哪些付款", "支持哪些支付", "可以用什么付款",
        "有没有优惠", "有优惠吗", "打折", "折扣", "优惠活动", "促销活动",
        "能便宜", "能优惠", "给个折扣",
        "企业采购", "批量采购", "批量下单", "批量购买",
        # 账户类
        "忘记密码", "找回密码", "重置密码", "忘记账号", "找回账号",
        "账号注册", "如何注册", "怎么注册", "更改邮箱", "修改邮箱",
        "绑定手机", "注销账号", "删除账号",
        # 物流/运费 FAQ
        "运费怎么算", "运费如何计算", "怎么计算运费",
        "发货时间", "处理时间", "多久发货", "几天发货",
        "运输方式", "配送方式", "配送范围", "能否配送", "运送到哪些国家",
        "关税", "需要缴关税", "清关",
        "如何追踪", "怎么查物流轨迹",
        # 积分/优惠券 FAQ
        "积分怎么用", "怎么使用积分", "积分有效期", "奖励积分",
        "优惠券怎么用", "优惠券失效",
        # 订单 FAQ（无订单号）
        "如何查看订单", "怎么查看订单", "找不到订单", "看不到订单", "订单在哪查",
        "订单状态说明", "取消订单流程", "如何取消订单",
        "多久收到退款", "退款多久到账", "退款什么时候",
        # 支付 FAQ
        "付款失败", "无法支付", "支付失败", "为什么扣款", "如何退款", "如何申请退款",
    ]
    if any(k in t for k in faq_keywords):
        return "faq"

    # 5. 纯闲聊/问候（仅当输入较短且匹配问候模式时命中，长文本交给 LLM）
    chitchat_patterns = [
        "你好", "您好", "hello", "hi ", "嗨",
        "谢谢", "感谢", "thanks", "thank you",
        "再见", "拜拜", "bye",
        "哈哈", "呵呵", "嘿嘿", "哦哦",
        "好的", "ok", "知道了", "明白了", "了解了",
    ]
    if len(t) < 20 and any(k in t for k in chitchat_patterns):
        return "chitchat"

    return None


def validate_and_fix_intent(intent: str) -> str:
    """校验意图标签有效性，无效则兜底为 chitchat，并记录异常。"""
    clean_intent = intent.strip().lower() if intent else ""
    if clean_intent in VALID_INTENTS:
        return clean_intent
    logger.warning(f"意图识别结果无效({clean_intent})，兜底为 chitchat")
    return "chitchat"


def init_node_state(state: State, node_name: str):
    """初始化节点通用状态：仅设置运行标志和清空错误。
    
    [优化] 不再在此处清空所有输出字段（kb_answer/order_summary/sources 等），
    避免节点 A 误清节点 B 的结果，各节点应自行管理自己负责的输出字段。
    """
    state.status = StateStatus.RUNNING
    state.error = None
    # state.ask_human = None
    # state.kb_answer = None
    # state.human_handoff = None
    # state.order_summary = None
    # state.sources = None
    # state.fallback = None
    # state.rewritten_query = None
    # state.long_term_memory = None
    # state.clarify_question = None
    # state.reasoning_trace = []
    logger.info(f"进入节点: {node_name}")

async def load_memory_node(state: State) -> Dict[str, Any]:
    """加载短期记忆节点：最近 N 轮原文 + 历史摘要。
    
    长期记忆由独立的 load_long_term_memory_node 并行加载，
    通过 LangGraph fan-out/fan-in 机制在 intent_post 节点汇合。
    """
    _start = time.perf_counter()
    init_node_state(state, "加载短期记忆节点")

    try:
        user_id = state.user_id or "default"

        _sub = time.perf_counter()
        recent_context, summary_context = await config.get_context_for_prompt(state.thread_id, user_id)
        logger.info(f"获取短期记忆 耗时: {time.perf_counter() - _sub:.4f}s")

        history_parts = []
        if summary_context:
            history_parts.append(f"【历史摘要】\n{summary_context}")
        if recent_context:
            history_parts.append(f"【最近对话】\n{recent_context}")
        state.history = "\n\n".join(history_parts) if history_parts else None

        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0 
        recent_turns = len(recent_context.splitlines()) // 2 if recent_context else 0
        logger.info(
            f"加载短期记忆节点完成: user_id={user_id}, "
            f"recent_turns={recent_turns}, "
            f"has_summary={bool(summary_context)}"
        )
        await metrics_collector.track_node_execution(
            "load_memory_node", (time.perf_counter() - _start), success=True
        )
        return state

    except Exception as e:
        state.error = str(e)[:200]
        state.status = StateStatus.SUCCESS  # 短期记忆加载失败不影响主流程
        state.retry_attempts = 0 
        state.history = None
        logger.warning(f"加载短期记忆节点失败: {e}")
        await metrics_collector.track_node_execution(
            "load_memory_node", (time.perf_counter() - _start), success=False
        )
        return state


async def load_long_term_memory_node(state: State) -> Dict[str, Any]:
    """加载长期记忆节点：根据当前 query 语义检索用户画像/偏好/历史事实。
    
    与 load_memory_node + intent_node 并行执行（LangGraph fan-out），
    在 intent_post_node 汇合（fan-in）。
    
    返回值只包含 long_term_memory 字段，避免与并行分支的 state 字段冲突。
    """
    _start = time.perf_counter()
    logger.info("进入节点: 加载长期记忆节点")

    try:
        user_id = state.user_id or "default"
        tenant_id = state.tenant_id or "default"

        long_term_str = await memory_module.load_relevant_memories(
            user_id=user_id, tenant_id=tenant_id, query=state.query, limit=5
        )
        logger.info(
            f"加载长期记忆节点完成: user_id={user_id}, "
            f"has_long_term={bool(long_term_str)}, "
            f"耗时: {time.perf_counter() - _start:.4f}s"
        )
        await metrics_collector.track_node_execution(
            "load_long_term_memory_node", (time.perf_counter() - _start), success=True
        )
        # 仅返回 long_term_memory，避免与并行分支（intent_node）写入相同 channel 导致冲突
        return {"long_term_memory": long_term_str or None}

    except Exception as e:
        logger.warning(f"加载长期记忆节点失败: {e}，降级为无长期记忆")
        await metrics_collector.track_node_execution(
            "load_long_term_memory_node", (time.perf_counter() - _start), success=False
        )
        return {"long_term_memory": None}


_PRONOUN_PATTERN = re.compile(
    r"(它|他/她|这个|那个|这款|那款|这种|那种|此产品|该产品|上面|之前|刚才|前面说|前面提)"
)

_QUESTION_STARTERS = ("什么", "怎么", "为什么", "哪里", "多少", "几时", "啥", "咋", "怎样", "如何")


def _needs_rewrite_heuristic(query: str) -> bool:
    """
    启发式判断 query 是否可能需要改写。
    返回 True  → 必须调用 LLM（无论长短）
    返回 False → 可跳过改写（query 已足够清晰且不含歧义信号）

    触发条件：
      1. 指代词：含 它/他/她/这个/那个/这款/那款/这种/那种/此/该/上面/之前/刚才/前面 等
      2. 极简输入：去空格后 ≤ 8 个字符（几乎一定省略了主体）
      3. 无主语的疑问句：以疑问词开头但全句很短（≤ 20 字）
    """
    # 1. 指代词检测（无论长度，使用模块级预编译正则）
    if _PRONOUN_PATTERN.search(query):
        return True

    stripped = query.strip()

    # 2. 极简输入
    if len(stripped) <= 8:
        return True

    # 3. 无主语短疑问句（≤20字且以疑问词开头，使用模块级元组）
    if len(stripped) <= 20 and any(stripped.startswith(s) for s in _QUESTION_STARTERS):
        return True

    return False


async def query_rewrite_node(state: State) -> Dict[str, Any]:
    """Query 改写节点：智能判断是否需要改写，将简短/模糊/省略/指代式输入改写为完整清晰的标准 query。
    
    改进点：
    - 基于歧义信号的启发式判断
    - 仅当 query 既长（>20字）又无歧义信号时才跳过，避免漏改
    - 准确判断 query_rewritten 标志（改写结果与原文一致视为未改写）
    - 对改写结果做合理性校验，防止 LLM 输出异常内容
    """
    _start = time.perf_counter()
    init_node_state(state, "Query改写节点")
    state.rewritten_query = None
    
    try:
        original_query = state.query

        # ── 启发式快速判断：是否跳过改写 ──
        # 仅当 query 足够长（>20字）且不含歧义信号时跳过，节省推理开销
        has_ambiguity = _needs_rewrite_heuristic(original_query)
        if len(original_query) > 20 and not has_ambiguity:
            state.rewritten_query = original_query
            state.query_rewritten = False
            state.status = StateStatus.SUCCESS
            state.retry_attempts = 0
            logger.info(
                f"Query改写节点：查询较长且无歧义信号，跳过改写，"
                f"原始查询={original_query[:60]}{'...' if len(original_query) > 60 else ''}"
            )
            await metrics_collector.track_node_execution("query_rewrite_node", (time.perf_counter() - _start), success=True)
            return state

        logger.info(
            f"Query改写节点：准备改写，"
            f"长度={len(original_query)}，含歧义信号={has_ambiguity}，"
            f"原始查询={original_query}"
        )

        # ── 调用小模型改写，受熔断器保护 ──
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            history=state.history or "（无历史对话）",
            query=original_query
        )
        response = await _llm_small_breaker.call(small_llm.ainvoke, rewrite_prompt)
        rewritten_query = str(getattr(response, "content", response)).strip()

        # ── 改写结果合理性校验 ──
        # 1. 空结果 → 降级到原查询
        if not rewritten_query:
            logger.warning("Query改写节点：LLM 返回空结果，降级使用原始查询")
            rewritten_query = original_query

        # 2. 结果异常过长（超过原始 3 倍）→ 可能是 LLM 幻觉，降级
        elif len(rewritten_query) > max(len(original_query) * 3, 150):
            logger.warning(
                f"Query改写节点：改写结果异常过长（{len(rewritten_query)}字），降级使用原始查询"
            )
            rewritten_query = original_query

        # ── 比较改写前后，准确设置 query_rewritten 标志 ──
        state.query_rewritten = rewritten_query != original_query
        state.rewritten_query = rewritten_query
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0
        logger.info(
            f"Query改写节点完成：原始查询=【{original_query}】，"
            f"改写结果=【{rewritten_query}】，"
            f"是否实际改写={state.query_rewritten}"
        )
        await metrics_collector.track_node_execution("query_rewrite_node", (time.perf_counter() - _start), success=True)
        return state

    except Exception as e:
        state.error = str(e)[:200]
        state.status = StateStatus.SUCCESS  # 改写失败不阻断主流程，降级到原查询
        state.retry_attempts = 0
        state.rewritten_query = state.query
        state.query_rewritten = False
        logger.warning(f"Query改写节点异常: {e}，降级使用原始查询")
        await metrics_collector.track_node_execution("query_rewrite_node", (time.perf_counter() - _start), success=False)
        return state


def _parse_intent_slot_response(raw: str) -> dict:
    """解析 LLM 返回的意图+槽位 JSON，容错处理。"""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # 尝试从文本中提取 JSON 块
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _extract_slots_from_keywords(intent: str, query: str) -> dict:
    """关键词命中时，用规则快速提取槽位（不调 LLM）。"""
    slots = {}
    if intent == "order":
        oid = re.search(r"ORD\d{8,16}", query, re.IGNORECASE)
        if oid:
            slots["order_id"] = oid.group()
        # 动作推断
        action_map = [
            (["取消订单", "帮我取消", "给我取消"], "cancel"),
            (["申请退款", "发起退款", "提交退款", "帮我退款", "给我退款"], "refund"),
            (["修改地址", "改地址", "修改收货地址", "改收货地址", "帮我修改", "帮我改", "给我修改"], "modify_address"),
            (["查看物流", "查物流", "物流到哪了", "追踪包裹"], "logistics"),
            (["怎么还没发货", "订单还没发"], "logistics"),
        ]
        q = query.lower()
        for keywords, action in action_map:
            if any(k in q for k in keywords):
                slots["action"] = action
                break
        # 注意：不在此处设置 "query_detail" 兜底默认值。
        # 原因：多轮对话场景下，若用户第1轮已确认 action（如 cancel），
        # 第2轮只补充了订单号，此处若强制写入 "query_detail" 会在槽位合并时
        # 覆盖第1轮保留的 action，导致意图丢失。
        # 兜底逻辑统一由路由函数 _post_intent_branch 处理（只缺 action 时补默认值）。
    return slots


async def intent_node(state: State) -> Dict[str, Any]:
    """多阶段意图识别节点：规则快速路径 + LLM 联合识别 + 意图转移感知 + 槽位提取。

    流水线：
    Stage 1 — 保存上轮意图、检测转移信号词
    Stage 2 — 规则快速路径（关键词匹配 + 正则，零 LLM 调用）
    Stage 3 — LLM 联合识别（意图 + 槽位 + 置信度 + 转移判断，一次调用）
    Stage 4 — 置信度校准（意图转移时适当降权）
    Stage 5 — 必需槽位校验（order 意图需要 order_id + action）
    """
    _start = time.perf_counter()

    # ── Stage 1：保存上轮意图 + 初始化 ──
    prev_intent = state.intent
    state.prev_intent = prev_intent
    init_node_state(state, "意图识别节点")
    state.intent_switched = False
    state.switch_from = None
    state.recognition_source = None

    try:
        q = clean_input(state.query)
        has_context = bool(prev_intent and (state.history or "").strip())
        has_switch_signal = _detect_switch_signal(q)

        # ── Stage 2：规则快速路径 ──
        kw_intent = _keywords_intent(q)

        if kw_intent:
            intent = kw_intent
            confidence = 0.95
            new_slots = _extract_slots_from_keywords(intent, q)
            state.recognition_source = "keyword"

            if prev_intent and intent == prev_intent and not has_switch_signal:
                # 未转移：同意图延续，合并槽位（保留上轮已收集的槽位，补充本轮新提取的）
                if intent == "order": 
                    if not new_slots:
                        llm_result = await _llm_intent_slot(q, state.history)
                        if validate_and_fix_intent(llm_result.get("intent", "")) == intent:
                            new_slots = llm_result.get("slots") or {}
                    logger.debug(f"intent_node: new_slots={new_slots}, state.slots={state.slots}")
                    state.slots = {**(state.slots or {}), **{k: v for k, v in new_slots.items() if _is_valid_slot_value(v)}}
                else:
                    state.slots = {k: v for k, v in new_slots.items() if _is_valid_slot_value(v)}
            else:
                # 意图转移：新意图 / 转移信号词 / 首轮对话
                state.slots = new_slots
                if has_context and (has_switch_signal or intent != prev_intent):
                    state.intent_switched = True
                    state.switch_from = prev_intent
        else:
            # ── Stage 3：LLM 联合识别 ──
            llm_result = await _llm_unified_intent(q, state.history, prev_intent if has_context else None)
            state.recognition_source = "llm"

            llm_switched = llm_result.get("intent_switched", False)

            if not llm_switched and prev_intent:
                # 未转移：继承上轮意图，补充槽位
                intent = prev_intent
                confidence = 0.90
                new_slots = llm_result.get("slots") or {}
                state.slots = {**(state.slots or {}), **{k: v for k, v in new_slots.items() if _is_valid_slot_value(v)}}
                state.recognition_source = "context_inherit"
                state.intent_switched = False
            else:
                # 意图转移：全新识别
                intent = validate_and_fix_intent(llm_result.get("intent", ""))
                confidence = float(llm_result.get("confidence", 0.7))
                state.slots = {k: v for k, v in (llm_result.get("slots") or {}).items() if _is_valid_slot_value(v)}

                if has_context and intent != prev_intent:
                    state.intent_switched = True
                    state.switch_from = prev_intent

        # ── Stage 4：置信度校准 ──
        # 意图转移时小幅降权，表示新话题识别的额外不确定性
        if state.intent_switched and confidence > 0.85:
            confidence = round(confidence * 0.95, 4)

        # ── Stage 5：必需槽位校验 ──
        required = _REQUIRED_SLOTS.get(intent, [])
        missing = [s for s in required if not state.slots.get(s)]
        state.missing_slots = missing

        # 意图转移时清理不相关的历史槽位（从 order 切换到其他意图时清空订单槽位）
        if state.intent_switched and state.switch_from == "order" and intent != "order":
            state.slots = {}
            state.missing_slots = []

        # [优化] 将原本在路由函数 _post_intent_branch 中的副作用提前到此处：
        # 只缺 action 但有 order_id → 默认 query_detail，不反问
        if intent == "order" and missing == ["action"] and state.slots.get("order_id"):
            state.slots["action"] = "query_detail"
            state.missing_slots = []

        state.intent = intent
        state.route = intent
        state.intent_confidence = confidence
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0

        log_parts = [
            f"intent={intent}",
            f"confidence={confidence:.2f}",
            f"switched={state.intent_switched}",
            f"source={state.recognition_source}",
            f"slots={state.slots}",
            f"missing={missing}",
        ]
        if state.intent_switched:
            log_parts.append(f"SWITCHED from={state.switch_from}")
        logger.info(f"意图识别节点：{', '.join(log_parts)}")

        await metrics_collector.track_node_execution(
            "intent_node", time.perf_counter() - _start, success=True
        )
        return state

    except Exception as e:
        state.error = str(e)[:200]
        state.retry_attempts += 1
        logger.info(f"意图识别节点错误, retry_attempts: {state.retry_attempts}, error: {traceback.format_exc()}")
        await Robust.log_error(state.thread_id, "意图识别节点", state.error, state.retry_attempts)

        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_llm(e):
            state.status = StateStatus.RETRYING
            logger.info(f"意图识别节点重试: {traceback.format_exc()}")
            raise

        await metrics_collector.track_node_execution("intent_node", time.perf_counter() - _start, success=False)
        state.status = StateStatus.FAILED
        return state


async def _llm_intent_slot(query: str, history: str = None) -> dict:
    """调用 LLM 做意图+槽位联合识别（无上下文续接判断），受熔断器保护。"""
    prompt = INTENT_SLOT_PROMPT.format(
        query=query,
        history=history or "（无历史对话）",
    )
    resp = await _llm_small_breaker.call(llm.ainvoke, prompt)
    raw = str(getattr(resp, "content", resp)).strip()
    return _parse_intent_slot_response(raw)


async def _llm_unified_intent(query: str, history: str = None, prev_intent: str = None) -> dict:
    """调用 LLM 做意图转移检测 + 意图 + 槽位联合识别（一次调用），受熔断器保护。"""
    prompt = UNIFIED_INTENT_PROMPT.format(
        query=query,
        history=history or "（无历史对话）",
        prev_intent=prev_intent or "（无上轮意图）",
    )
    resp = await _llm_small_breaker.call(small_llm.ainvoke, prompt)
    raw = str(getattr(resp, "content", resp)).strip()
    return _parse_intent_slot_response(raw)


async def clarify_node(state: State) -> Dict[str, Any]:
    """统一反问节点：处理模糊意图反问 + 槽位缺失反问。
    
    优先级：
    1. 模糊意图（confidence 低）→ 生成选项式反问
    2. 槽位缺失 → 生成针对性追问
    """
    _start = time.perf_counter()
    init_node_state(state, "反问节点")
    state.clarify_question = None
    state.ask_retry_times += 1

    try:
        # 情况1：模糊意图反问（受熔断器保护）
        if state.intent_confidence < _INTENT_CONFIDENCE_THRESHOLD:
            prompt = CLARIFY_INTENT_PROMPT.format(
                query=state.query,
                reason=state.slots.get("ambiguity_reason", "意图不够明确"),
                candidates="faq(产品/售前/售后FAQ)、order(订单操作)、human(转人工)",
            )
            resp = await _llm_small_breaker.call(small_llm.ainvoke, prompt)
            reply = str(getattr(resp, "content", resp)).strip()
            state.clarify_question = reply or "请问您具体想了解什么呢？可以描述得更详细一些吗？"

        # 情况2：槽位缺失反问（受熔断器保护）
        elif state.missing_slots:
            filled_desc = ", ".join(
                f"{_SLOT_DISPLAY.get(k, k)}: {v}" for k, v in state.slots.items() if v
            ) or "暂无"
            missing_desc = ", ".join(
                _SLOT_DISPLAY.get(s, s) for s in state.missing_slots
            )
            prompt = CLARIFY_SLOT_PROMPT.format(
                intent=state.intent,
                filled_slots=filled_desc,
                missing_slots=missing_desc,
            )
            resp = await _llm_small_breaker.call(small_llm.ainvoke, prompt)
            reply = str(getattr(resp, "content", resp)).strip()
            state.clarify_question = reply or f"请提供以下信息：{missing_desc}"

        else:
            state.clarify_question = "请问您具体想了解什么呢？"

        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0 
        logger.info(f"反问节点：clarify_question={state.clarify_question}")
        await metrics_collector.track_node_execution(
            "clarify_node", time.perf_counter() - _start, success=True
        )
        return state

    except Exception as e:
        # 反问生成失败时使用固定话术
        state.clarify_question = "请问您具体想了解什么呢？可以提供更多信息吗？"
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0 
        logger.warning(f"反问节点失败: {e}，使用默认话术")
        await metrics_collector.track_node_execution(
            "clarify_node", time.perf_counter() - _start, success=False
        )
        return state


async def order_id_ask_node(state: State) -> Dict[str, Any]:
    """反问节点：提示用户补充/修正订单号"""
    _start = time.perf_counter()
    init_node_state(state, "订单号反问节点")
    state.ask_human = None
    
    # 根据不同场景生成反问话术（排除 LLM 产出的 "null" 字面量）
    if not _is_valid_slot_value(state.order_id):
        reply = "请问你要查询的订单号是多少？（格式示例：ORD12345678）"
    else:
        reply = f"你输入的订单号【{state.order_id}】格式不正确/不存在，请重新提供正确的订单号（格式示例：ORD12345678）。"
    
    # 记录反问次数
    state.ask_retry_times += 1
    state.ask_human = reply  # 反问话术返回给用户
    state.status = StateStatus.SUCCESS
    state.retry_attempts = 0 
    await metrics_collector.track_node_execution("order_id_ask_node", (time.perf_counter() - _start), success=True)
    return state


async def order_id_check_node(state: State) -> Dict[str, Any]:
    """订单号校验节点：优先从 slots 获取 order_id，否则从 query 提取，再做格式+有效性校验"""
    _start = time.perf_counter()
    init_node_state(state, "订单号校验节点")

    # 优先使用槽位中已提取的 order_id（过滤 LLM 产出的 "null" 字面量）
    order_id = (state.slots or {}).get("order_id")
    if not _is_valid_slot_value(order_id):
        order_id = None
    if not order_id:
        m = re.search(r"ORD\d{8,16}", state.query)
        order_id = m.group() if m else None

    state.order_id = order_id

    if not validate_order_id_format(order_id):
        state.order_id_valid = False
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0 
        return state

    state.order_id_valid = validate_order_id_exists(order_id, state.tenant_id)
    state.status = StateStatus.SUCCESS
    state.retry_attempts = 0 
    await metrics_collector.track_node_execution(
        "order_id_check_node", time.perf_counter() - _start, success=True
    )
    return state


def _build_sources(docs) -> list:
    """从检索文档列表构建前端友好的 sources 列表，每条包含 title / content / url / metadata。
    
    [优化] 抽取为独立函数，消除 kb_node 和 kb_node_stream 中的重复代码。
    """
    sources = []
    for d in (docs or []):
        meta = getattr(d, "metadata", {}) or {}
        h1 = str(meta.get("h1") or "").strip()
        h2 = str(meta.get("h2") or "").strip()
        h3 = str(meta.get("h3") or "").strip()
        title = h3 or h2 or h1 or meta.get("source", "知识库")
        original_text = (
            meta.get("original_text")
            or meta.get("original_child_content")
            or d.page_content
            or ""
        )
        sources.append({
            "title": title,
            "content": original_text[:500].strip(),
            "url": meta.get("source", ""),
            "metadata": {
                "source": meta.get("source", ""),
                "h1": h1, "h2": h2, "h3": h3,
                "rerank_score": round(float(meta.get("rerank_score") or 0), 4),
                "is_expanded": meta.get("is_expanded", False),
            },
        })
    return sources


async def kb_node(state: State) -> Dict[str, Any]:
    """知识库节点：先查语义缓存，命中则直接返回；未命中则走完整 RAG 流程后写入缓存。"""
    _start = time.perf_counter()
    init_node_state(state, "知识库节点")
    state.kb_answer = None
    state.sources = None
    
    try:
        # 优先使用改写后的 query，如果没有则使用原始 query
        query = state.rewritten_query or state.query
        tenant_id = state.tenant_id or "default"

        # ── 语义缓存查找（余弦相似度 > 阈值则命中，跳过 RAG 检索和 LLM 推理）──
        _t_cache_start = time.perf_counter()
        _cache = get_semantic_cache()
        cached_answer, best_sim = None, 0.0
        if _cache is not None:
            logger.info(f"知识库节点：开始语义缓存查找，query='{query[:50]}...'，tenant={tenant_id}")
            cached_answer, best_sim = await _cache.lookup(query, tenant_id)
        _t_cache_ms = (time.perf_counter() - _t_cache_start) * 1000

        if cached_answer is not None:
            # ── 缓存命中：直接返回，无需 RAG 检索和 LLM 推理 ──────────────────
            _elapsed = time.perf_counter() - _start
            logger.info(
                f"⏱️  知识库节点 分段耗时 | "
                f"节点初始化: {(_t_cache_start - _start)*1000:.1f}ms | "
                f"缓存查找(embed+KNN): {_t_cache_ms:.1f}ms | "
                f"合计: {_elapsed*1000:.1f}ms"
            )
            logger.info(
                f"✅ 知识库节点缓存命中！相似度={best_sim:.4f}，"
                f"耗时={_elapsed*1000:.1f}ms（已跳过 RAG+LLM）"
            )
            state.status = StateStatus.SUCCESS
            state.retry_attempts = 0
            state.kb_answer = cached_answer
            # 缓存命中时 sources 为空（缓存中不存储 sources，前端降级展示）
            state.sources = []
            await metrics_collector.track_node_execution("kb_node", _elapsed, success=True)
            await metrics_collector.track_kb_cache(
                tenant_id=tenant_id, hit=True, duration=_elapsed,
                node="kb_node", similarity=best_sim,
            )
            return state

        # ── 缓存未命中：走完整 RAG 流程 ─────────────────────────────────────
        logger.info(
            f"知识库节点：缓存未命中（最高相似度={best_sim:.4f}），"
            f"启动 RAG 检索，tenant={tenant_id}"
        )

        # retrieve_kb 是同步函数，通过 asyncio.to_thread 在线程池执行，受熔断器保护
        serialized, docs = await _kb_retrieval_breaker.call(asyncio.to_thread, retrieve_kb, query, tenant_id)

        # [优化] 使用抽取的 _build_sources 函数构建前端 sources 列表
        sources = _build_sources(docs)
        
        if not docs:
            logger.warning("⚠️ 检索结果为空! 请检查 Collection 是否有数据，或 Embeddings 是否匹配。")
            content = "抱歉，没有找到相关文档，无法回答您的问题。"
        else:
            _llm_start = time.perf_counter()
            prefix = f"最近对话摘要：\n{state.history}\n\n" if state.history else ""
            prompt = prefix + RAG_PROMPT_TEMPLATE.format(
                context=serialized, 
                question=query
            )
            effective_llm = _get_effective_llm(state)
            llm_input = _build_llm_input(prompt, state)
            msg = await _llm_main_breaker.call(effective_llm.ainvoke, llm_input)
            _llm_end = time.perf_counter()
            logger.info(f"知识库节点：LLM 耗时: {_llm_end - _llm_start:.4f}s, 多模态={_has_images(state)}")
            content = str(getattr(msg, "content", msg)).strip()

        # ── RAG 完成后写入语义缓存（仅检索到文档时才缓存，兜底回复不缓存）──
        # [优化] ensure_future → create_task + 异常回调，避免异常被静默吞掉
        if docs and _cache is not None:
            _task = asyncio.create_task(_cache.save(query, content, tenant_id))
            _task.add_done_callback(
                lambda t: t.exception() and logger.warning(f"知识库缓存写入失败: {t.exception()}")
            )
            logger.info(f"知识库节点：已异步提交缓存写入任务，tenant={tenant_id}")
            
        # 统一处理成功状态
        _elapsed = time.perf_counter() - _start
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0 
        state.kb_answer = content
        state.sources = sources
        await metrics_collector.track_node_execution("kb_node", _elapsed, success=True)
        await metrics_collector.track_kb_cache(
            tenant_id=tenant_id, hit=False, duration=_elapsed,
            node="kb_node", similarity=best_sim,
        )
        return state
        
    except Exception as e:
        state.error = str(e)[:200]
        state.retry_attempts += 1
        
        await Robust.log_error(state.thread_id, "知识库节点", state.error, state.retry_attempts)
        
        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_api(e):
            state.status = StateStatus.RETRYING  # 还有重试次数，且需要重试，标记为「重试中」
            raise
            
        state.status = StateStatus.FAILED    # 标记为「失败」
        await metrics_collector.track_node_execution("kb_node", (time.perf_counter() - _start), success=False)
        return state


async def handoff_node(state: State) -> Dict[str, Any]:
    """人工客服节点：记录未命中问题，并返回转人工渠道信息。"""
    _start = time.perf_counter()
    state.ask_retry_times = 0
    init_node_state(state, "人工客服节点")
    state.human_handoff = None
    try:
        # 记录未命中问题
        q = state.rewritten_query or state.query
        record_unanswered(q, None, state.tenant_id)

        # 返回转人工渠道信息
        payload = {"query": q}
        res = handoff_to_human(payload)
    except Exception as e:
        state.error = str(e)[:200]
        state.retry_attempts += 1

        await Robust.log_error(state.thread_id, "人工客服节点", state.error, state.retry_attempts)
        
        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_llm(e):
            state.status = StateStatus.RETRYING  # 还有重试次数，且需要重试，标记为「重试中」
            raise
        else:
            await metrics_collector.track_node_execution("handoff_node", (time.perf_counter() - _start), success=False)
            state.status = StateStatus.FAILED    # 标记为「失败」
            return state
    state.status = StateStatus.SUCCESS
    state.retry_attempts = 0 
    state.human_handoff = res
    await metrics_collector.track_node_execution("handoff_node", (time.perf_counter() - _start), success=True)
    logger.info(f"人工客服节点：state={state}")
    return state


_REPLY_MARKER = "【回复】"


def _extract_user_reply(text: str) -> str:
    """从 LLM 输出中提取面向用户的回复（【回复】标记之后的内容）。

    ReAct 模式下 LLM 的输出包含「内部推理 + 【回复】+ 用户可见回复」，
    本函数剥离推理部分，只保留用户可见的回复。
    如果没有找到标记则原样返回（兜底，保证不丢内容）。
    """
    idx = text.rfind(_REPLY_MARKER)
    if idx >= 0:
        reply = text[idx + len(_REPLY_MARKER):].strip()
        if reply:
            return reply
    return text.strip()


async def _react_order_loop(
    llm_with_tools,
    tools: list,
    executor: SafeToolExecutor,
    system_prompt: str,
    user_query: str,
    order_id: str,
    tenant_id: str,
    state_ctx: dict,
    max_iterations: int = _REACT_MAX_ITERATIONS,
    stream_queue=None,
) -> tuple:
    """ReAct 推理循环：思考(Thought) → 行动(Action) → 观察(Observation)，重复直到得出最终回答。

    返回:
        (final_answer, reasoning_trace)
        - final_answer: 最终回复文本
        - reasoning_trace: 推理链路列表，每项 {"step", "type", "content"}
    """
    # 必须同时包含 SystemMessage 和 HumanMessage，否则部分 LLM 不会触发 function call
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]
    reasoning_trace = []
    tool_map = {(getattr(tool, "name", None) or tool.__name__): tool for tool in tools}

    _loop_start = time.perf_counter()
    # 各阶段累计耗时（秒）
    _t_llm_total = 0.0          # LLM 推理（Thought + 决策）
    _t_tool_exec_total = 0.0    # 工具执行（Observation）
    _t_arg_inject_total = 0.0   # 参数注入 + 预处理
    _step_timings = []          # 每步明细

    for step in range(1, max_iterations + 1):
        _step_start = time.perf_counter()
        logger.info(f"ReAct 推理循环 第{step}/{max_iterations}步")

        # ── Thought + Action：调用 LLM（受熔断器保护）──
        _t0 = time.perf_counter()
        response = await _llm_main_breaker.call(llm_with_tools.ainvoke, messages)
        _t_llm = time.perf_counter() - _t0
        _t_llm_total += _t_llm
        messages.append(response)

        # 记录思考过程（LLM 的 content 部分即为 Thought）
        thought = str(getattr(response, "content", "")).strip()
        if thought:
            reasoning_trace.append({"step": step, "type": "thought", "content": thought})
            logger.info(f"ReAct 第{step}步 [思考] ({_t_llm*1000:.0f}ms): {thought[:150]}...")
            if stream_queue is not None:
                await stream_queue.put({
                    "type": "react_step", "step": step, "phase": "thought", "content": thought
                })

        # 检查是否有工具调用（Action）
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            _step_timings.append({
                "step": step,
                "llm_ms": round(_t_llm * 1000, 1),
                "tools_ms": 0,
                "total_ms": round((time.perf_counter() - _step_start) * 1000, 1),
                "tool_names": [],
            })
            logger.info(f"ReAct 在第{step}步得出最终回答（无更多工具调用），LLM耗时={_t_llm*1000:.0f}ms")
            break

        # ── Action：强制注入 order_id / tenant_id，记录行动 ──
        _t0 = time.perf_counter()
        called_tool_names = []
        for call in tool_calls:
            args = call.get("args", {})
            tool_name = call.get("name")
            called_tool_names.append(tool_name)
            if not args.get("order_id"):
                args["order_id"] = order_id
            target_tool = tool_map.get(tool_name)
            if target_tool is not None and not args.get("tenant_id"):
                try:
                    params = inspect.signature(target_tool).parameters
                    if "tenant_id" in params:
                        args["tenant_id"] = tenant_id
                except Exception:
                    pass

            action_desc = f"调用 {tool_name}({json.dumps(args, ensure_ascii=False)})"
            reasoning_trace.append({"step": step, "type": "action", "content": action_desc})
            logger.info(f"ReAct 第{step}步 [行动]: {action_desc}")
            if stream_queue is not None:
                await stream_queue.put({
                    "type": "react_step", "step": step, "phase": "action", "content": action_desc
                })
        _t_inject = time.perf_counter() - _t0
        _t_arg_inject_total += _t_inject

        # ── Observation：执行工具（受熔断器保护），记录观察 ──
        _t0 = time.perf_counter()
        tool_messages = await _order_tools_breaker.call(
            executor.execute_with_fallback, tool_calls, state_ctx
        )
        _t_tool = time.perf_counter() - _t0
        _t_tool_exec_total += _t_tool

        for tc, tm in zip(tool_calls, tool_messages):
            observation = tm.content
            reasoning_trace.append({"step": step, "type": "observation", "content": observation})
            logger.info(f"ReAct 第{step}步 [观察] ({_t_tool*1000:.0f}ms): {observation[:150]}...")
            if stream_queue is not None:
                await stream_queue.put({
                    "type": "react_step", "step": step, "phase": "observation", "content": observation
                })
            messages.append(LCToolMessage(
                content=observation,
                tool_call_id=tc.get("id", f"react_{step}")
            ))

        _step_elapsed = time.perf_counter() - _step_start
        _step_timings.append({
            "step": step,
            "llm_ms": round(_t_llm * 1000, 1),
            "tools_ms": round(_t_tool * 1000, 1),
            "total_ms": round(_step_elapsed * 1000, 1),
            "tool_names": called_tool_names,
        })
    else:
        # for-else：达到最大迭代次数仍未结束，强制收尾
        logger.warning(f"ReAct 达到最大迭代次数 {max_iterations}，强制总结")
        reasoning_trace.append({
            "step": max_iterations, "type": "thought",
            "content": f"已达最大推理步数({max_iterations})，基于已有信息给出回答"
        })
        messages.append(SystemMessage(
            content="你已完成所有工具调用，请直接用专业客服语气给出最终回答，不要再调用工具。回复必须以 【回复】 开头。"
        ))
        try:
            _t0 = time.perf_counter()
            base_llm = llm_with_tools.kwargs.get("llm", llm_with_tools)
            final_resp = await _llm_main_breaker.call(base_llm.ainvoke, messages)
            _t_llm_total += time.perf_counter() - _t0
            final_text = str(getattr(final_resp, "content", "")).strip()
            if final_text:
                messages.append(final_resp)
        except Exception:
            pass

    # ── ReAct 耗时汇总日志 ──
    _loop_elapsed = time.perf_counter() - _loop_start
    _t_overhead = _loop_elapsed - _t_llm_total - _t_tool_exec_total
    logger.info(
        f"⏱️  ReAct 耗时汇总 | "
        f"总耗时: {_loop_elapsed*1000:.0f}ms | "
        f"LLM推理: {_t_llm_total*1000:.0f}ms ({_t_llm_total/_loop_elapsed*100:.0f}%) | "
        f"工具执行: {_t_tool_exec_total*1000:.0f}ms ({_t_tool_exec_total/_loop_elapsed*100:.0f}%) | "
        f"其他开销: {_t_overhead*1000:.0f}ms ({_t_overhead/_loop_elapsed*100:.0f}%) | "
        f"共 {len(_step_timings)} 步"
    )
    for st in _step_timings:
        logger.info(
            f"  ├─ 第{st['step']}步: LLM={st['llm_ms']:.0f}ms, "
            f"工具={st['tools_ms']:.0f}ms, "
            f"步总计={st['total_ms']:.0f}ms, "
            f"调用={st['tool_names'] or '(无工具调用)'}"
        )

    # 提取最终回答
    final_answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not isinstance(msg, LCToolMessage):
            text = str(msg.content).strip()
            if text:
                final_answer = text
                break

    # 兜底：从推理链中取最后一条有意义的内容
    if not final_answer and reasoning_trace:
        for item in reversed(reasoning_trace):
            if item["type"] in ("observation", "thought") and item["content"]:
                final_answer = item["content"]
                break

    # 剥离内部推理，只保留 【回复】 标记之后的用户可见内容
    final_answer = _extract_user_reply(final_answer)

    return final_answer, reasoning_trace


async def order_node(state: State) -> Dict[str, Any]:
    """订单节点（ReAct 模式）：通过 思考→行动→观察 推理循环处理用户的订单问题。

    启用 ReAct 后，LLM 会在每次工具调用前先输出思考过程，根据工具返回结果决定下一步行动，
    支持多步推理和自我纠错。可通过 REACT_ENABLED=false 降级回单次调用模式。"""
    _start = time.perf_counter()
    init_node_state(state, "订单节点(ReAct)" if _REACT_ENABLED else "订单节点")
    state.order_summary = None
    state.reasoning_trace = []
    state.ask_retry_times = 0
    # [优化] 防御性检查：order_id 为 None 时不应走到此节点，但加保护避免 AttributeError
    if not state.order_id:
        logger.error("订单节点：order_id 为空，不应到达此节点，降级返回")
        state.order_summary = "抱歉，缺少订单号信息，请提供订单号后重试。"
        state.status = StateStatus.FAILED
        state.retry_attempts = 0
        await metrics_collector.track_node_execution("order_node", (time.perf_counter() - _start), success=False)
        return state
    order_id = state.order_id.strip()
    tenant_id = state.tenant_id or "default"

    # 定义可用工具列表（含渐进式披露的 skill 工具）
    tools = [
        get_order_detail,
        cancel_order,
        modify_order_address,
        apply_refund,
        get_logistics_info,
        lookup_skill,       # L1 渐进式披露：按需加载技能详细指南
        read_reference,     # L2 渐进式披露：按需加载参考文档
    ]

    # 创建安全工具执行器，LLM 作为降级兜底
    executor = SafeToolExecutor(tools, fallback_model=llm)

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 将意图识别槽位中的 action 转换为中文意图描述，注入 Prompt 辅助 LLM 选择工具
    _ACTION_TO_HINT = {
        "cancel":          "取消订单",
        "refund":          "申请退款",
        "modify_address":  "修改收货地址",
        "query_detail":    "查询订单详情",
        "logistics":       "查询物流信息",
    }
    _slot_action = (state.slots or {}).get("action", "")
    _action_label = _ACTION_TO_HINT.get(_slot_action, "")
    action_hint = _action_label if _action_label else ""
    if action_hint:
        logger.debug(f"订单节点：从槽位 action='{_slot_action}' 转换意图提示 → {_action_label}")

    # 构造传递给 SafeToolExecutor 的上下文 dict
    state_ctx = {
        "context": state.order_summary or "",
        "order_id": order_id,
        "tenant_id": tenant_id,
    }

    # 获取流式队列（在流式图中可用，普通图中为 None）
    stream_queue = _stream_queue_var.get(None)

    # L0 层：获取技能索引摘要，注入 System Prompt
    skill_index = SkillRegistry.get().get_index_summary()

    try:
        if _REACT_ENABLED:
            # ══ ReAct 推理模式：思考 → 行动 → 观察 循环 ══
            prompt = REACT_ORDER_PROMPT_TEMPLATE.format(
                order_id=order_id,
                query=state.query,
                rewritten_query=state.rewritten_query or state.query,
                action_hint=action_hint,
                tenant_id=tenant_id,
                skill_index=skill_index,
            )
            logger.info(f"订单节点启用 ReAct 推理，最大迭代 {_REACT_MAX_ITERATIONS} 步")

            final_answer, reasoning_trace = await _react_order_loop(
                llm_with_tools=llm_with_tools,
                tools=tools,
                executor=executor,
                system_prompt=prompt,
                user_query=state.query,
                order_id=order_id,
                tenant_id=tenant_id,
                state_ctx=state_ctx,
                max_iterations=_REACT_MAX_ITERATIONS,
                stream_queue=stream_queue,
            )

            state.order_summary = final_answer
            state.reasoning_trace = reasoning_trace
            state.status = StateStatus.SUCCESS

            logger.info(
                f"订单节点 ReAct 完成：共 {len([t for t in reasoning_trace if t['type'] == 'action'])} 次工具调用，"
                f"{len(reasoning_trace)} 步推理"
            )
        else:
            # ══ 降级模式：单次 LLM 调用（原有逻辑）══
            prompt = ORDER_AGENT_PROMPT_TEMPLATE.format(
                order_id=order_id,
                query=state.query,
                rewritten_query=state.rewritten_query or state.query,
                tenant_id=tenant_id,
                action_hint=action_hint,
                skill_index=skill_index,
            )

            response = await _llm_main_breaker.call(llm_with_tools.ainvoke, prompt)
            tool_calls = getattr(response, "tool_calls", [])

            if tool_calls:
                tool_map = {tool.__name__: tool for tool in tools}
                for call in tool_calls:
                    args = call.get("args", {})
                    tool_name = call.get("name")
                    target_tool = tool_map.get(tool_name)
                    if not args.get("order_id"):
                        args["order_id"] = order_id
                    if target_tool is not None and not args.get("tenant_id"):
                        try:
                            params = inspect.signature(target_tool).parameters
                            if "tenant_id" in params:
                                args["tenant_id"] = tenant_id
                        except Exception as e:
                            logger.warning(f"订单节点工具签名解析失败: tool={tool_name}, error={e}")
                    logger.info(f"订单节点准备调用工具: {call.get('name')}, 参数: {args}")

                tool_messages = await _order_tools_breaker.call(
                    executor.execute_with_fallback, tool_calls, state_ctx
                )
                results = [msg.content for msg in tool_messages]
            else:
                content = _extract_user_reply(str(getattr(response, "content", "")))
                if content:
                    results = [content]
                else:
                    logger.info(f"无工具调用，兜底查询详情: {order_id}")
                    fallback_calls = [{
                        "name": "get_order_detail",
                        "args": {"order_id": order_id, "tenant_id": tenant_id},
                        "id": "fallback_detail",
                    }]
                    tool_messages = await _order_tools_breaker.call(
                        executor.execute_with_fallback, fallback_calls, state_ctx
                    )
                    results = [msg.content for msg in tool_messages]

            state.order_summary = "\n".join(results)
            state.status = StateStatus.SUCCESS

    except CircuitBreakerError as e:
        state.error = str(e)[:200]
        logger.warning(f"订单节点熔断: {e}")
        state.order_summary = "抱歉，订单服务暂时不可用，请稍后重试或联系人工客服。"
        state.status = StateStatus.FAILED
    except Exception as e:
        state.error = str(e)[:200]
        logger.error(f"订单节点执行失败: {e}, {traceback.format_exc()}")
        # 降级：兜底查询订单详情
        try:
            fallback_calls = [{
                "name": "get_order_detail",
                "args": {"order_id": order_id, "tenant_id": tenant_id},
                "id": "error_fallback_detail",
            }]
            tool_messages = await executor.execute_with_fallback(fallback_calls, state_ctx)
            state.order_summary = "\n".join(msg.content for msg in tool_messages)
            state.status = StateStatus.SUCCESS
        except Exception:
            state.status = StateStatus.FAILED
    state.retry_attempts = 0
    _node_success = (state.status == StateStatus.SUCCESS)
    await metrics_collector.track_node_execution("order_node", (time.perf_counter() - _start), success=_node_success)
    return state


async def direct_node(state: State) -> Dict[str, Any]:
    """直答节点：不依赖 KB 的简要回答。"""
    _start = time.perf_counter()
    init_node_state(state, "直答节点")
    state.kb_answer = None
    try:  
        q = state.rewritten_query or state.query
        h = state.history
        prefix = ("最近对话摘要：\n" + h + "\n\n") if h else ""
        prompt = prefix + DIRECT_PROMPT_TEMPLATE.format(
            long_term_memory=state.long_term_memory or "（暂无长期记忆）",
            question=q
        )
        effective_llm = _get_effective_llm(state)
        llm_input = _build_llm_input(prompt, state)
        msg = await _llm_main_breaker.call(effective_llm.ainvoke, llm_input)
        content = str(getattr(msg, "content", msg)).strip()
    except Exception as e:
        state.error = str(e)[:200]
        state.retry_attempts += 1

        await Robust.log_error(state.thread_id, "直答节点", state.error, state.retry_attempts)
        
        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_llm(e):
            state.status = StateStatus.RETRYING  # 还有重试次数，且需要重试，标记为「重试中」
            raise
        else:
            await metrics_collector.track_node_execution("direct_node", (time.perf_counter() - _start), success=False)
            state.status = StateStatus.FAILED    # 标记为「失败」
            return state
    state.status = StateStatus.SUCCESS
    state.retry_attempts = 0 
    state.kb_answer = content
    await metrics_collector.track_node_execution("direct_node", (time.perf_counter() - _start), success=True)
    return state


async def fallback_node(state: State) -> Dict[str, Any]:
    """兜底节点：失败后返回友好结果，更新 status 为FAILED"""
    _start = time.perf_counter()
    state.status = StateStatus.FALLBACK  # 标记为「兜底」
    state.fallback = None
    logger.info(f"兜底节点：state={state}")
    reply = "抱歉，请求暂时无法处理，请稍后重试~"
    # 兜底逻辑
    await Robust.log_error(state.thread_id, "兜底节点", state.error, state.retry_attempts)
    # await Robust.send_to_dead_letter(state.thread_id, state.query)  
    state.fallback = reply
    await metrics_collector.track_node_execution("fallback_node", (time.perf_counter() - _start), success=True)
    return state


async def intent_post_node(state: State) -> Dict[str, Any]:
    """意图后处理节点：并行分支（intent_node + load_long_term_memory_node）的汇合点。
    
    LangGraph fan-in 机制确保本节点在两个并行分支都完成后才执行：
    - 分支 1（load_memory → intent）：短期记忆加载 + 意图识别
    - 分支 2（load_long_term_memory）：长期记忆加载
    
    本节点负责记录汇合状态，后续由条件边路由到对应处理节点。
    """
    logger.info(
        f"意图后处理节点（并行汇合）：intent={state.intent}, "
        f"confidence={state.intent_confidence:.2f}, "
        f"has_long_term_memory={bool(state.long_term_memory)}, "
        f"has_history={bool(state.history)}"
    )
    return state


# ── [优化] 路由函数提升为模块级，消除 construct_graph / construct_graph_stream 中的重复定义 ──

def _post_intent_branch(state: State) -> str:
    """意图识别后路由：根据意图和槽位状态分发到对应处理节点（纯判断，无副作用）"""
    if state.status == StateStatus.FAILED:
        return "fallback"
    if state.intent_confidence < _INTENT_CONFIDENCE_THRESHOLD:
        return "clarify"
    intent = state.intent
    if intent == "order":
        if state.missing_slots:
            return "clarify"
        return "order_id_check"
    if intent == "faq":
        return "query_rewrite"
    if intent == "human":
        return "handoff"
    return "direct"


def _post_order_id_check(state: State) -> str:
    """订单号校验后路由"""
    if state.ask_retry_times >= config.MAX_HANDOFF_RETRY:
        logger.info(f"订单号校验后路由：超过最大反问次数{config.MAX_HANDOFF_RETRY}，转人工")
        return "handoff"
    if state.order_id_valid:
        return "order"
    return "order_id_ask"


def _post_kb(state: State) -> str:
    """KB 节点后路由：有回答 → END，无回答 → 转人工"""
    return "has_kb" if (state.kb_answer or "").strip() else "no_kb"


_INTENT_BRANCH_MAP = {
    "order_id_check": "order_id_check",
    "query_rewrite": "query_rewrite",
    "handoff": "handoff",
    "direct": "direct",
    "fallback": "fallback",
    "clarify": "clarify",
}

_ORDER_ID_CHECK_MAP = {
    "order": "order",
    "order_id_ask": "order_id_ask",
    "handoff": "handoff",
}

_KB_BRANCH_MAP = {"has_kb": END, "no_kb": "handoff"}


def _construct_graph_core(*, streaming: bool = False) -> StateGraph:
    """统一构图核心函数。
    
    [优化] 将 construct_graph 和 construct_graph_stream 的公共逻辑合并，
    仅通过 streaming 参数选择 kb / direct 节点是流式版本还是普通版本。

    利用 LangGraph 并行 fan-out/fan-in 机制，长期记忆加载与意图识别并行执行：

    START ──┬── load_memory(短期记忆) → intent(意图识别) ──┬── intent_post(汇合) → 路由分发
            └── load_long_term_memory(长期记忆) ──────────┘
      ├─ 模糊意图 / 槽位缺失 → clarify → END（反问用户）
      ├─ order → order_id_check → order / order_id_ask / handoff
      ├─ faq → query_rewrite → kb → END / handoff
      ├─ human → handoff → END
      ├─ chitchat → direct → END
      └─ FAILED → fallback → END
    """
    g = StateGraph(State)

    # ── 并行分支节点 ──
    g.add_node("load_memory", load_memory_node)
    g.add_node("load_long_term_memory", load_long_term_memory_node)
    g.add_node("intent", intent_node, retry_policy=SmartRetryPolicy.create_policy("intent_llm"))
    g.add_node("intent_post", intent_post_node)

    # ── 后续处理节点 ──
    g.add_node("clarify", clarify_node)
    g.add_node("query_rewrite", query_rewrite_node)
    g.add_node("order_id_check", order_id_check_node)
    g.add_node("order_id_ask", order_id_ask_node)
    g.add_node("kb", kb_node_stream if streaming else kb_node, retry_policy=SmartRetryPolicy.create_policy("kb_api"))
    g.add_node("handoff", handoff_node, retry_policy=SmartRetryPolicy.create_policy("handoff_api"))
    g.add_node("order", order_node, retry_policy=SmartRetryPolicy.create_policy("order_api"))
    g.add_node("direct", direct_node_stream if streaming else direct_node, retry_policy=SmartRetryPolicy.create_policy("direct_llm"))
    g.add_node("fallback", fallback_node)

    # ── 并行 fan-out：从 START 同时启动两个分支 ──
    # 分支 1：短期记忆加载 → 意图识别（意图识别阶段不需要长期记忆）
    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "intent")
    # 分支 2：长期记忆加载（与分支 1 并行执行）
    g.add_edge(START, "load_long_term_memory")

    # ── 并行 fan-in：两个分支在 intent_post 节点汇合 ──
    g.add_edge("intent", "intent_post")
    g.add_edge("load_long_term_memory", "intent_post")

    # ── 路由分发（从 intent_post 节点开始，原 intent 节点的路由逻辑不变）──
    g.add_conditional_edges("intent_post", _post_intent_branch, _INTENT_BRANCH_MAP)
    g.add_edge("clarify", END)
    g.add_edge("query_rewrite", "kb")
    g.add_conditional_edges("order_id_check", _post_order_id_check, _ORDER_ID_CHECK_MAP)
    g.add_conditional_edges("kb", _post_kb, _KB_BRANCH_MAP)
    g.add_edge("order", END)
    g.add_edge("order_id_ask", END)
    g.add_edge("handoff", END)
    g.add_edge("direct", END)
    g.add_edge("fallback", END)

    mode_label = "流式" if streaming else "普通"
    logger.info(f"graph构建完毕（{mode_label}模式，含并行记忆加载）")
    return g


def construct_graph():
    """构建普通对话图（调用统一核心函数）"""
    return _construct_graph_core(streaming=False)


# [优化] 以下是原始 construct_graph 的完整实现，已被 _construct_graph_core 替代
# def construct_graph():
#     """构建对话图
#     节点流程：
#     START → load_memory → intent → 路由分发
#       ├─ 模糊意图 / 槽位缺失 → clarify → END
#       ├─ order → order_id_check → order / order_id_ask / handoff
#       ├─ faq → query_rewrite → kb → END / handoff
#       ├─ human → handoff → END
#       ├─ chitchat → direct → END
#       └─ FAILED → fallback → END
#     """
#     g = StateGraph(State)
#     g.add_node("load_memory", load_memory_node)
#     g.add_node("intent", intent_node, retry_policy=SmartRetryPolicy.create_policy("intent_llm"))
#     g.add_node("clarify", clarify_node)
#     g.add_node("query_rewrite", query_rewrite_node)
#     g.add_node("order_id_check", order_id_check_node)
#     g.add_node("order_id_ask", order_id_ask_node)
#     g.add_node("kb", kb_node, retry_policy=SmartRetryPolicy.create_policy("kb_api"))
#     g.add_node("handoff", handoff_node, retry_policy=SmartRetryPolicy.create_policy("handoff_api"))
#     g.add_node("order", order_node, retry_policy=SmartRetryPolicy.create_policy("order_api"))
#     g.add_node("direct", direct_node, retry_policy=SmartRetryPolicy.create_policy("direct_llm"))
#     g.add_node("fallback", fallback_node)
#     g.add_edge(START, "load_memory")
#     g.add_edge("load_memory", "intent")
#     def _post_intent_branch(state: State) -> str:
#         ...（路由逻辑已提到模块级）
#     g.add_conditional_edges("intent", _post_intent_branch, {...})
#     g.add_edge("clarify", END)
#     g.add_edge("query_rewrite", "kb")
#     def _post_order_id_check(state: State) -> str:
#         ...（路由逻辑已提到模块级）
#     g.add_conditional_edges("order_id_check", _post_order_id_check, {...})
#     def _post_kb(state: State) -> str:
#         ...（路由逻辑已提到模块级）
#     g.add_conditional_edges("kb", _post_kb, {...})
#     g.add_edge("order", END)
#     g.add_edge("order_id_ask", END)
#     g.add_edge("handoff", END)
#     g.add_edge("direct", END)
#     g.add_edge("fallback", END)
#     logger.info("graph构建完毕")
#     return g


# 整图执行的全局超时（秒），防止某个 LLM 节点卡住导致请求永久挂起
_GRAPH_TIMEOUT = float(os.getenv("GRAPH_TIMEOUT_SECONDS", "300"))

# 预编译图模板缓存：避免每次请求都重新构图 + 编译
_GRAPH_TEMPLATE_LOCK = threading.Lock()
_COMPILED_GRAPH_TEMPLATE = None
_COMPILED_STREAM_GRAPH_TEMPLATE = None


def _get_compiled_graph_template():
    """获取普通图的编译模板（全局单例，懒加载）。"""
    global _COMPILED_GRAPH_TEMPLATE
    if _COMPILED_GRAPH_TEMPLATE is None:
        with _GRAPH_TEMPLATE_LOCK:
            if _COMPILED_GRAPH_TEMPLATE is None:
                logger.info("图模板缓存未命中，开始构建并编译普通图模板")
                _COMPILED_GRAPH_TEMPLATE = construct_graph().compile()
                logger.info("普通图模板编译完成，已写入全局缓存")
    return _COMPILED_GRAPH_TEMPLATE


def _get_compiled_graph_stream_template():
    """获取流式图的编译模板（全局单例，懒加载）。"""
    global _COMPILED_STREAM_GRAPH_TEMPLATE
    if _COMPILED_STREAM_GRAPH_TEMPLATE is None:
        with _GRAPH_TEMPLATE_LOCK:
            if _COMPILED_STREAM_GRAPH_TEMPLATE is None:
                logger.info("图模板缓存未命中，开始构建并编译流式图模板")
                _COMPILED_STREAM_GRAPH_TEMPLATE = construct_graph_stream().compile()
                logger.info("流式图模板编译完成，已写入全局缓存")
    return _COMPILED_STREAM_GRAPH_TEMPLATE


def _bind_checkpointer(compiled_template, checkpointer):
    """基于编译模板复制一个运行时链，并动态绑定本次请求的 checkpointer。"""
    return compiled_template.copy(update={"checkpointer": checkpointer})


def warmup_graph_templates(force_rebuild: bool = False):
    """预热图模板缓存（可选强制重建）。

    - force_rebuild=False：仅在缓存为空时编译
    - force_rebuild=True：先清空缓存，再重新编译
    """
    global _COMPILED_GRAPH_TEMPLATE, _COMPILED_STREAM_GRAPH_TEMPLATE
    if force_rebuild:
        with _GRAPH_TEMPLATE_LOCK:
            logger.info("收到强制重建请求，清空图模板缓存")
            _COMPILED_GRAPH_TEMPLATE = None
            _COMPILED_STREAM_GRAPH_TEMPLATE = None

    normal_template = _get_compiled_graph_template()
    stream_template = _get_compiled_graph_stream_template()
    logger.info("图模板预热完成：普通图与流式图均已就绪")
    return {
        "normal_template_ready": bool(normal_template),
        "stream_template_ready": bool(stream_template),
    }


async def _run_graph_core(state: State, *, streaming: bool = False, stream_queue: asyncio.Queue = None):
    """[优化] 统一图执行核心，消除 run_graph 和 run_graph_stream 之间的重复代码。
    
    Args:
        state: 对话状态
        streaming: 是否使用流式图模板
        stream_queue: 流式输出队列（仅 streaming=True 时有效）
    """
    from core.observability import GraphMetrics

    if streaming and stream_queue is not None:
        _stream_queue_var.set(stream_queue)

    mode_label = "流式" if streaming else "普通"
    _graph_start = datetime.now()

    compiled_template = _get_compiled_graph_stream_template() if streaming else _get_compiled_graph_template()
    checkpointer_dsn = config.get_checkpointer_dsn(state.tenant_id)
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn) as memory:
        await memory.setup()
        chain = _bind_checkpointer(compiled_template, memory)
        robust_runner = Robust(chain, memory)
        try:
            turn_input = {
                "thread_id": state.thread_id,
                "query": state.query,
                "user_id": state.user_id,
                "tenant_id": state.tenant_id,
                "quoted_message": state.quoted_message,
                "images": state.images,
            }
            logger.info(f"run_graph({mode_label}) 本轮输入字段: {list(turn_input.keys())}, 含图片={bool(state.images)}")
            result = await asyncio.wait_for(
                robust_runner.run_with_recovery(turn_input, state.thread_id, max_recovery_attempts=3),
                timeout=_GRAPH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"{mode_label} graph 全局超时 ({_GRAPH_TIMEOUT}s) thread_id={state.thread_id}")
            result = {
                "route": "fallback",
                "fallback": "抱歉，当前响应超时，请稍后重试或联系人工客服。",
                "sources": [],
            }
        except Exception:
            logger.error(f"{mode_label} graph 全局重试最终失败 {traceback.format_exc()}")
            result = {
                "route": "fallback",
                "fallback": "抱歉，系统暂时繁忙，请稍后再试或直接联系人工客服。",
                "sources": [],
            }

        _graph_end = datetime.now()
        graph_metrics = GraphMetrics(
            thread_id=state.thread_id or "",
            start_time=_graph_start,
            end_time=_graph_end,
            total_nodes_executed=0,
            failed_nodes=[],
            retry_count=result.get("retry_attempts", 0) if isinstance(result, dict) else 0,
            total_tokens=0,
            total_cost=0.0,
        )
        asyncio.create_task(metrics_collector.track_graph_execution(graph_metrics))

        logger.info(f"{mode_label} graph 执行完毕")
        return result


async def run_graph(state: State):
    """普通模式执行图"""
    return await _run_graph_core(state, streaming=False)


# [优化] 以下是原始 run_graph 的完整实现，已被 _run_graph_core 替代
# async def run_graph(state: State):
#     from core.observability import GraphMetrics
#     _graph_start = datetime.now()
#     compiled_template = _get_compiled_graph_template()
#     checkpointer_dsn = config.get_checkpointer_dsn(state.tenant_id)
#     async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn) as memory:
#         await memory.setup()
#         chain = _bind_checkpointer(compiled_template, memory)
#         robust_runner = Robust(chain, memory)
#         try:
#             turn_input = {...}
#             result = await asyncio.wait_for(...)
#         except asyncio.TimeoutError: ...
#         except Exception: ...
#         graph_metrics = GraphMetrics(...)
#         asyncio.create_task(metrics_collector.track_graph_execution(graph_metrics))
#         return result


# ═══════════════════════════════════════════════════════════════
#  流式输出支持（SSE）：新增流式节点、流式图构建、流式图执行
# ═══════════════════════════════════════════════════════════════

# 流式输出队列：流式节点在生成 token 时逐步推送到此队列，由 SSE 生成器消费
_stream_queue_var: _CtxVar[Optional[asyncio.Queue]] = _CtxVar("stream_queue", default=None)


async def _stream_llm_and_collect(llm_obj, llm_input, stream_queue):
    """流式调用 LLM 并收集完整结果，中间 token 通过 stream_queue 实时推送。

    llm_input 可以是纯文本字符串，也可以是多模态 HumanMessage。
    """
    chunks = []
    async for chunk in llm_obj.astream(llm_input):
        # 兼容不同 LLM 的 chunk 格式
        if hasattr(chunk, "content"):
            text = str(chunk.content)
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = str(chunk)
        if text:
            chunks.append(text)
            if stream_queue is not None:
                await stream_queue.put({"type": "token", "content": text})
    return "".join(chunks)


async def kb_node_stream(state: State) -> Dict[str, Any]:
    """知识库节点（流式版本）：先查语义缓存，命中则直接推送全文 token；未命中则走 RAG+流式 LLM 后写入缓存。"""
    _start = time.perf_counter()
    init_node_state(state, "知识库节点(流式)")
    state.kb_answer = None
    state.sources = None
    stream_queue = _stream_queue_var.get(None)

    try:
        query = state.rewritten_query or state.query
        tenant_id = state.tenant_id or "default"

        # ── 语义缓存查找（与 kb_node 共享同一缓存层）────────────────────────
        _t_cache_start = time.perf_counter()
        _cache = get_semantic_cache()
        cached_answer, best_sim = None, 0.0
        if _cache is not None:
            logger.info(
                f"知识库节点(流式)：开始语义缓存查找，query='{query[:50]}...'，tenant={tenant_id}"
            )
            cached_answer, best_sim = await _cache.lookup(query, tenant_id)
        _t_cache_ms = (time.perf_counter() - _t_cache_start) * 1000

        if cached_answer is not None:
            # ── 缓存命中：将整段答案一次性推入 stream_queue（前端逐字符渲染）──
            _t_push_start = time.perf_counter()
            _elapsed = time.perf_counter() - _start
            logger.info(
                f"⏱️  知识库节点(流式) 分段耗时 | "
                f"节点初始化: {(_t_cache_start - _start)*1000:.1f}ms | "
                f"缓存查找(embed+KNN): {_t_cache_ms:.1f}ms | "
                f"合计: {_elapsed*1000:.1f}ms"
            )
            logger.info(
                f"✅ 知识库节点(流式)缓存命中！相似度={best_sim:.4f}，"
                f"耗时={_elapsed*1000:.1f}ms（已跳过 RAG+LLM）"
            )
            if stream_queue is not None:
                await stream_queue.put({"type": "token", "content": cached_answer})
            state.status = StateStatus.SUCCESS
            state.retry_attempts = 0
            state.kb_answer = cached_answer
            state.sources = []
            await metrics_collector.track_node_execution("kb_node_stream", _elapsed, success=True)
            await metrics_collector.track_kb_cache(
                tenant_id=tenant_id, hit=True, duration=_elapsed,
                node="kb_node_stream", similarity=best_sim,
            )
            return state

        # ── 缓存未命中：走完整 RAG + 流式 LLM 流程 ─────────────────────────
        logger.info(
            f"知识库节点(流式)：缓存未命中（最高相似度={best_sim:.4f}），"
            f"启动 RAG 检索，tenant={tenant_id}"
        )

        # KB 检索（同步函数，通过线程池执行），受熔断器保护
        serialized, docs = await _kb_retrieval_breaker.call(
            asyncio.to_thread, retrieve_kb, query, tenant_id
        )

        # [优化] 使用抽取的 _build_sources 函数构建前端 sources 列表
        sources = _build_sources(docs)

        if not docs:
            logger.warning("⚠️ 检索结果为空! 请检查 Collection 是否有数据，或 Embeddings 是否匹配。")
            content = "抱歉，没有找到相关文档，无法回答您的问题。"
            if stream_queue is not None:
                await stream_queue.put({"type": "token", "content": content})
        else:
            _llm_start = time.perf_counter()
            prefix = f"最近对话摘要：\n{state.history}\n\n" if state.history else ""
            prompt = prefix + RAG_PROMPT_TEMPLATE.format(
                context=serialized, question=query
            )
            effective_llm = _get_effective_llm(state)
            llm_input = _build_llm_input(prompt, state)
            content = await _llm_main_breaker.call(
                _stream_llm_and_collect, effective_llm, llm_input, stream_queue
            )
            _llm_end = time.perf_counter()
            logger.info(f"知识库节点(流式)：LLM 耗时: {_llm_end - _llm_start:.4f}s, 多模态={_has_images(state)}")

        # ── RAG 完成后异步写入语义缓存（不阻塞流式返回）───────────────────────
        # [优化] ensure_future → create_task + 异常回调
        if docs and _cache is not None:
            _task = asyncio.create_task(_cache.save(query, content, tenant_id))
            _task.add_done_callback(
                lambda t: t.exception() and logger.warning(f"知识库(流式)缓存写入失败: {t.exception()}")
            )
            logger.info(f"知识库节点(流式)：已异步提交缓存写入任务，tenant={tenant_id}")

        _elapsed = time.perf_counter() - _start
        state.status = StateStatus.SUCCESS
        state.retry_attempts = 0
        state.kb_answer = content
        state.sources = sources
        await metrics_collector.track_node_execution("kb_node_stream", _elapsed, success=True)
        await metrics_collector.track_kb_cache(
            tenant_id=tenant_id, hit=False, duration=_elapsed,
            node="kb_node_stream", similarity=best_sim,
        )
        return state

    except Exception as e:
        state.error = str(e)[:200]
        state.retry_attempts += 1
        await Robust.log_error(
            state.thread_id, "知识库节点(流式)", state.error, state.retry_attempts
        )
        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_api(e):
            state.status = StateStatus.RETRYING
            raise
        state.status = StateStatus.FAILED
        await metrics_collector.track_node_execution(
            "kb_node_stream", (time.perf_counter() - _start), success=False
        )
        return state


async def direct_node_stream(state: State) -> Dict[str, Any]:
    """直答节点（流式版本）：与 direct_node 逻辑完全一致，LLM 回答部分使用 astream 实现逐 token 输出"""
    _start = time.perf_counter()
    init_node_state(state, "直答节点(流式)")
    state.kb_answer = None
    stream_queue = _stream_queue_var.get(None)

    try:
        q = state.rewritten_query or state.query
        h = state.history
        prefix = ("最近对话摘要：\n" + h + "\n\n") if h else ""
        prompt = prefix + DIRECT_PROMPT_TEMPLATE.format(
            long_term_memory=state.long_term_memory or "（暂无长期记忆）",
            question=q,
        )
        effective_llm = _get_effective_llm(state)
        llm_input = _build_llm_input(prompt, state)
        content = await _llm_main_breaker.call(
            _stream_llm_and_collect, effective_llm, llm_input, stream_queue
        )
    except Exception as e:
        logger.error(f"直答节点(流式)：{traceback.format_exc()}")
        state.error = str(e)[:200]
        state.retry_attempts += 1
        await Robust.log_error(
            state.thread_id, "直答节点(流式)", state.error, state.retry_attempts
        )
        if state.retry_attempts < config.MAX_ATTEMPTS and SmartRetryPolicy.should_retry_llm(e):
            state.status = StateStatus.RETRYING
            raise
        await metrics_collector.track_node_execution(
            "direct_node_stream", (time.perf_counter() - _start), success=False
        )
        state.status = StateStatus.FAILED
        return state

    state.status = StateStatus.SUCCESS
    state.retry_attempts = 0
    state.kb_answer = content
    await metrics_collector.track_node_execution(
        "direct_node_stream", (time.perf_counter() - _start), success=True
    )
    return state


def construct_graph_stream():
    """构建流式对话图（调用统一核心函数，streaming=True）"""
    return _construct_graph_core(streaming=True)


# [优化] 以下是原始 construct_graph_stream 的完整实现，已被 _construct_graph_core(streaming=True) 替代
# def construct_graph_stream():
#     """构建流式对话图
#     节点流程与 construct_graph 完全一致，仅将 kb / direct 节点替换为流式版本
#     """
#     g = StateGraph(State)
#     g.add_node("load_memory", load_memory_node)
#     g.add_node("intent", intent_node, retry_policy=SmartRetryPolicy.create_policy("intent_llm"))
#     g.add_node("clarify", clarify_node)
#     g.add_node("query_rewrite", query_rewrite_node)
#     g.add_node("order_id_check", order_id_check_node)
#     g.add_node("order_id_ask", order_id_ask_node)
#     g.add_node("kb", kb_node_stream, retry_policy=SmartRetryPolicy.create_policy("kb_api"))
#     g.add_node("handoff", handoff_node, retry_policy=SmartRetryPolicy.create_policy("handoff_api"))
#     g.add_node("order", order_node, retry_policy=SmartRetryPolicy.create_policy("order_api"))
#     g.add_node("direct", direct_node_stream, retry_policy=SmartRetryPolicy.create_policy("direct_llm"))
#     g.add_node("fallback", fallback_node)
#     g.add_edge(START, "load_memory")
#     g.add_edge("load_memory", "intent")
#     def _post_intent_branch(state): ...
#     g.add_conditional_edges("intent", _post_intent_branch, {...})
#     g.add_edge("clarify", END)
#     g.add_edge("query_rewrite", "kb")
#     def _post_order_id_check(state): ...
#     g.add_conditional_edges("order_id_check", _post_order_id_check, {...})
#     def _post_kb(state): ...
#     g.add_conditional_edges("kb", _post_kb, {...})
#     g.add_edge("order", END)
#     g.add_edge("order_id_ask", END)
#     g.add_edge("handoff", END)
#     g.add_edge("direct", END)
#     g.add_edge("fallback", END)
#     logger.info("流式 graph 构建完毕")
#     return g


async def run_graph_stream(state: State, stream_queue: asyncio.Queue = None):
    """流式执行图（调用统一核心函数，streaming=True）"""
    return await _run_graph_core(state, streaming=True, stream_queue=stream_queue)


# [优化] 以下是原始 run_graph_stream 的完整实现，已被 _run_graph_core 替代
# async def run_graph_stream(state: State, stream_queue: asyncio.Queue = None):
#     """流式执行图：与 run_graph 逻辑一致，额外通过 stream_queue 实时推送 LLM token"""
#     from core.observability import GraphMetrics
#     if stream_queue is not None:
#         _stream_queue_var.set(stream_queue)
#     _graph_start = datetime.now()
#     compiled_template = _get_compiled_graph_stream_template()
#     checkpointer_dsn = config.get_checkpointer_dsn(state.tenant_id)
#     async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn) as memory:
#         await memory.setup()
#         chain = _bind_checkpointer(compiled_template, memory)
#         robust_runner = Robust(chain, memory)
#         try:
#             turn_input = {...}
#             result = await asyncio.wait_for(...)
#         except asyncio.TimeoutError: ...
#         except Exception: ...
#         graph_metrics = GraphMetrics(...)
#         asyncio.create_task(metrics_collector.track_graph_execution(graph_metrics))
#         return result
