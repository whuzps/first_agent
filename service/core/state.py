"""
状态定义模块：用于在 LangGraph 流程中传递会话状态。
包含用户查询、意图路由、知识库回答、RAG 来源、订单摘要、人工交接信息与最终路由等字段。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import Literal


# ── 并行分支 reducer（用于 LangGraph fan-out/fan-in）────────────────────

def _last_value(_current: Any, new: Any) -> Any:
    """默认 reducer：后写入的值覆盖前值。

    等效于 LangGraph 默认的 LastValue channel 行为，
    但允许同一 step 内来自多个并行分支的写入
    （默认 LastValue 在此场景下会抛出 InvalidUpdateError）。
    """
    return new


def _keep_non_none(current: Optional[str], new: Optional[str]) -> Optional[str]:
    """合并 reducer：优先保留非 None 值。

    无论并行分支完成顺序如何，最终结果一致：
      - reducer("memories", None) → "memories"
      - reducer(None, "memories") → "memories"
      - reducer(None, None)       → None
    """
    return new if new is not None else current


def _parallel_safe(cls):
    """类装饰器：为 State 的所有字段自动添加并行安全的 reducer。

    LangGraph fan-out/fan-in 场景下，多个并行分支可能同时写入同一个 state 字段。
    默认的 LastValue channel 在收到多个值时会抛出 InvalidUpdateError。

    本装饰器自动将未标注 reducer 的字段包裹为 Annotated[type, _last_value]，
    使 channel 切换为 BinaryOperatorAggregate，从而允许并行写入。
    已有自定义 Annotated reducer 的字段（如 long_term_memory）保持不变。

    装饰顺序：必须放在 @dataclass 之前（Python 装饰器从内到外执行）。
    """
    new_annotations = {}
    for name, type_hint in cls.__annotations__.items():
        if hasattr(type_hint, '__metadata__'):
            new_annotations[name] = type_hint
        else:
            new_annotations[name] = Annotated[type_hint, _last_value]
    cls.__annotations__ = new_annotations
    return cls


class StateStatus(str, Enum):
    """流程执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    FALLBACK = "fallback"


@dataclass
@_parallel_safe
class State:
    thread_id: str
    query: str
    history: str = None
    intent: Literal["faq", "order", "chitchat", "human"] = None
    route: str = None
    user_id: str = None
    tenant_id: str = "default"
    status: StateStatus = StateStatus.PENDING
    error: Optional[str] = None
    retry_attempts: int = 0

    kb_answer: str = None
    sources: List[Dict[str, Any]] = None
    order_summary: str = None
    ask_human: str = None
    human_handoff: Dict[str, Any] = None
    fallback: str = None

    order_id: Optional[str] = None
    order_id_valid: Optional[bool] = None
    ask_retry_times: int = 0

    # 使用自定义 reducer：并行分支合并时优先保留非 None 值
    long_term_memory: Annotated[Optional[str], _keep_non_none] = None

    # 用户引用的历史消息内容
    quoted_message: Optional[str] = None

    # 多模态图片列表（base64 或 URL），非空时自动切换 VL 模型
    images: Optional[List[str]] = None

    # Query 改写
    rewritten_query: str = None
    query_rewritten: bool = False

    # ── 意图+槽位识别 ──────────────────────────────────────────
    # 意图置信度（0~1），低于阈值触发模糊意图反问
    intent_confidence: float = 1.0

    # 槽位字典：不同意图有不同必需槽位（如 order 需要 order_id + action）
    slots: Dict[str, Any] = field(default_factory=dict)

    # 缺失的必需槽位名称列表（非空时触发反问）
    missing_slots: List[str] = field(default_factory=list)

    # 反问话术（由 clarify_node 填充，优先级高于其他回复字段）
    clarify_question: Optional[str] = None

    # ── 意图转移检测 ───────────────────────────────────────────
    # 上一轮的意图（用于判断是否发生转移）
    prev_intent: Optional[str] = None

    # 是否检测到意图从上轮发生了转移（含话题转换、转折信号词等场景）
    intent_switched: bool = False
    
    # 转移前的意图（仅 intent_switched=True 时有值）
    switch_from: Optional[str] = None
    
    # 识别来源：keyword（规则命中）/ llm（模型判断）/ context_inherit（上下文继承）
    recognition_source: Optional[str] = None

    # ── ReAct 推理链路追踪 ────────────────────────────────────────
    # 记录订单节点 ReAct 推理的每一步：思考(thought)、行动(action)、观察(observation)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
