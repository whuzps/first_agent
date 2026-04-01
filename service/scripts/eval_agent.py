#!/usr/bin/env python3
"""客服 Agent 自动化评估脚本

评估指标：
  1. 回答准确率  (Accuracy)       — LLM-as-Judge 对比标准答案，0-100 分
  2. 幻觉率      (Hallucination)  — KB 回答中编造内容的比例（OutputGuard 双层检测）
  3. 工具调用正确率 (Tool Accuracy) — 订单操作是否调用预期工具（答案内容推断）
  4. 意图路由准确率 (Route Accuracy) — 是否路由到预期意图节点
  5. 响应时间    (Latency)        — avg / p50 / p95 / max（ms）
  6. 首字响应时间 (TTFT)          — 通过流式接口测量首个 token 到达时间（ms）

用法：
  python eval_agent.py                          # 内置测试集，默认并发 2
  python eval_agent.py --dataset my.json        # 自定义测试集（JSON 数组）
  python eval_agent.py --concurrency 3          # 并发数（不宜过高，会打爆 API）
  python eval_agent.py --output my_report       # 报告文件名前缀
  python eval_agent.py --skip-hallucination     # 跳过幻觉检测（节省约 600ms/条）
  python eval_agent.py --skip-accuracy          # 跳过 LLM 准确率评分（仅指标统计）
  python eval_agent.py --skip-ttft              # 跳过 TTFT 首字响应时间（使用非流式接口）
  python eval_agent.py --tenant default         # 指定租户
"""

import asyncio
import argparse
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.state import State
from core.config import LOG_DIR

# graph、core、tools 等包在 service/ 下，需把 service 根目录加入 path（不是 app/）
_APP_DIR = Path(__file__).resolve().parent
_SERVICE_ROOT = _APP_DIR.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from core.logging_config import setup_logging
# 评估脚本模式：静默第三方库噪音，只保留 eval logger 的 INFO 输出
setup_logging(silence_libs=True, script_logger_name="eval")
logger = logging.getLogger("eval")


# ============================================================
# 内置测试集
# ============================================================

BUILT_IN_CASES: List[Dict] = [
    # ── KB 类：产品 FAQ（product）───────────────────────────────
    {
        "name": "FAQ-运费计算",
        "query": "运费是怎么计算的？",
        "case_type": "kb",
        "expected_route": "product",
        "ground_truth": "运费根据商品重量、目的地和运输方式计算，结算时会显示具体费用。",
    },
    {
        "name": "FAQ-密码找回",
        "query": "忘记密码了怎么找回？",
        "case_type": "kb",
        "expected_route": "product",
        "ground_truth": "可以通过注册邮箱接收重置链接来找回密码。",
    },
    {
        "name": "FAQ-发货时间",
        "query": "下单后多久发货？",
        "case_type": "kb",
        "expected_route": "product",
        "ground_truth": "一般在付款确认后1-3个工作日内发货，具体以页面显示为准。",
    },
    # ── KB 类：售前咨询（presale）───────────────────────────────
    {
        "name": "PRESALE-付款方式",
        "query": "支持哪些付款方式？",
        "case_type": "kb",
        "expected_route": "presale",
        "ground_truth": "支持信用卡、PayPal 等多种国际付款方式。",
    },
    {
        "name": "PRESALE-批量采购",
        "query": "可以企业批量采购吗？有折扣吗？",
        "case_type": "kb",
        "expected_route": "presale",
        "ground_truth": "支持企业批量采购，批量订单可能享有折扣，建议联系客服咨询具体政策。",
    },
    # ── KB 类：售后服务（postsale）──────────────────────────────
    {
        "name": "POSTSALE-退货政策",
        "query": "退货政策是什么？怎么退货？",
        "case_type": "kb",
        "expected_route": "postsale",
        "ground_truth": "支持30天内退货，商品需保持原包装未使用状态，通过账户中心申请退货。",
    },
    {
        "name": "POSTSALE-收到破损",
        "query": "收到的商品是坏的，怎么处理？",
        "case_type": "kb",
        "expected_route": "postsale",
        "ground_truth": "请拍照记录破损情况并通过售后渠道提交，我们将安排换货或退款处理。",
    },
    # ── 订单类：查询详情（order → get_order_detail）────────────
    {
        "name": "ORDER-查询待付款",
        "query": "ORD20260101001 这个订单的状态是什么？",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "get_order_detail",
        "ground_truth": None,
    },
    {
        "name": "ORDER-查询已发货",
        "query": "帮我查一下 ORD20260101002 的订单",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "get_order_detail",
        "ground_truth": None,
    },
    # ── 订单类：取消订单（order → cancel_order）────────────────
    {
        "name": "ORDER-取消订单",
        "query": "帮我取消订单 ORD20260101001",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "cancel_order",
        "ground_truth": None,
    },
    # ── 订单类：申请退款（order → apply_refund）────────────────
    {
        "name": "ORDER-申请退款",
        "query": "我要申请退款，订单号是 ORD20260101004",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "apply_refund",
        "ground_truth": None,
    },
    # ── 订单类：查询物流（order → get_logistics_info）──────────
    {
        "name": "ORDER-查物流",
        "query": "ORD20260101002 的物流到哪里了？",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "get_logistics_info",
        "ground_truth": None,
    },
    # ── 订单类：修改地址（order → modify_order_address）────────
    {
        "name": "ORDER-修改地址",
        "query": "ORD20260101001 的收货地址改成上海市浦东新区",
        "case_type": "order",
        "expected_route": "order",
        "expected_tool": "modify_order_address",
        "ground_truth": None,
    },
    # ── 直答类（direct）────────────────────────────────────────
    {
        "name": "DIRECT-问候",
        "query": "你好",
        "case_type": "direct",
        "expected_route": "direct",
        "ground_truth": None,
    },
    {
        "name": "DIRECT-感谢",
        "query": "谢谢你的帮助！",
        "case_type": "direct",
        "expected_route": "direct",
        "ground_truth": None,
    },
    {
        "name": "DIRECT-闲聊",
        "query": "今天天气真好",
        "case_type": "direct",
        "expected_route": "direct",
        "ground_truth": None,
    },
    # ── 转人工（human）─────────────────────────────────────────
    {
        "name": "HUMAN-转人工",
        "query": "我要找人工客服",
        "case_type": "human",
        "expected_route": "human",
        "ground_truth": None,
    },
    # ── 边界 / 鲁棒性 ──────────────────────────────────────────
    {
        "name": "EDGE-空查询兜底",
        "query": "？？？",
        "case_type": "edge",
        "expected_route": None,  # 任意路由，只检查不崩溃
        "ground_truth": None,
    },
    {
        "name": "EDGE-无订单号追问",
        "query": "帮我取消订单",   # 无订单号，应触发反问
        "case_type": "edge",
        "expected_route": "order",
        "ground_truth": None,
    },
]


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TestCase:
    """单条测试用例"""
    name: str
    query: str
    case_type: str                      # kb / order / direct / human / edge
    expected_route: Optional[str]       # 预期意图路由
    expected_tool: Optional[str] = None # 预期工具（仅 order 类）
    ground_truth: Optional[str] = None  # 标准答案（用于准确率评分）
    user_id: str = "__eval__"
    tenant_id: str = "default"


@dataclass
class CaseResult:
    """单条用例评估结果"""
    name: str
    query: str
    case_type: str

    # 基础输出
    answer: str = ""
    route_actual: str = ""
    route_expected: Optional[str] = None
    route_correct: Optional[bool] = None

    # 准确率
    accuracy_score: float = -1.0        # -1 表示无标准答案，未评分
    accuracy_label: str = "N/A"         # 优秀/良好/一般/差/N/A

    # 幻觉
    hallucinated: Optional[bool] = None # None = 未检测
    hallucination_reason: str = ""

    # 工具调用
    tool_actual: Optional[str] = None
    tool_expected: Optional[str] = None
    tool_correct: Optional[bool] = None

    # 响应时间
    latency_ms: float = 0.0

    # 首字响应时间（TTFT），-1 表示未测量
    ttft_ms: float = -1.0

    # 异常
    error: Optional[str] = None


@dataclass
class EvalSummary:
    """评估汇总"""
    total: int = 0
    error_count: int = 0

    # 1. 准确率（有 ground_truth 的用例）
    accuracy_cases: int = 0
    avg_accuracy: float = 0.0           # 0-100
    accuracy_pass_rate: float = 0.0     # 分数 >= 70 视为通过

    # 2. 幻觉率（KB 类用例）
    kb_cases: int = 0
    hallucinated_cases: int = 0
    hallucination_rate: float = 0.0

    # 3. 工具调用正确率（order 类用例）
    order_cases: int = 0
    tool_correct_cases: int = 0
    tool_accuracy: float = 0.0

    # 4. 意图路由准确率（有 expected_route 的用例）
    routable_cases: int = 0
    route_correct_cases: int = 0
    route_accuracy: float = 0.0

    # 5. 响应时间
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # 6. 首字响应时间（TTFT）
    ttft_cases: int = 0                  # 成功测量 TTFT 的用例数
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    max_ttft_ms: float = 0.0

    timestamp: str = ""


# ============================================================
# 工具调用推断（从答案内容反推实际调用的工具）
# ============================================================

_TOOL_SIGNALS: Dict[str, List[str]] = {
    "cancel_order": [
        "已为您取消", "取消成功", "订单已取消", "取消申请",
        "cancelled", "cancel success",
    ],
    "apply_refund": [
        "退款申请", "已为您申请退款", "退款成功", "退款金额",
        "refund applied", "refund request",
    ],
    "modify_order_address": [
        "地址已修改", "收货地址已更新", "地址修改成功", "新地址",
        "address updated", "address modified",
    ],
    "get_logistics_info": [
        "物流", "快递", "运输", "派送", "已签收", "在途",
        "logistics", "tracking", "shipped",
    ],
    "get_order_detail": [
        "订单状态", "订单详情", "订单信息", "待付款", "已发货",
        "已完成", "已取消", "order status", "order detail",
        "创建时间", "订单金额",
    ],
}


def infer_tool_from_answer(answer: str) -> Optional[str]:
    """从回答内容推断实际调用的工具（按特异性从高到低匹配）"""
    if not answer:
        return None
    t = answer.lower()
    # cancel / refund / address 特异性高，优先
    priority_order = [
        "cancel_order",
        "apply_refund",
        "modify_order_address",
        "get_logistics_info",
        "get_order_detail",
    ]
    for tool_name in priority_order:
        if any(kw.lower() in t for kw in _TOOL_SIGNALS[tool_name]):
            return tool_name
    return None


# ============================================================
# LLM-as-Judge：准确率评分
# ============================================================

_judge_llm = None


def _get_judge_llm():
    """延迟加载评判 LLM（qwen-turbo，节省成本）"""
    global _judge_llm
    if _judge_llm is None:
        import core.config as _cfg
        _judge_llm = _cfg.get_llm()
    return _judge_llm


_ACCURACY_JUDGE_PROMPT = """\
你是客服回答质量评估专家。对比"标准答案"与"实际回答"的准确性，输出 0-100 的整数分。

评分参考：
90-100：核心信息完全正确，无误导内容
70-89 ：基本正确，有少量细节遗漏或表述差异
40-69 ：部分正确，缺失重要信息或有错误表述
10-39 ：大部分错误，只有极少信息匹配
0-9  ：完全错误、无关或拒绝回答

只输出一个整数，不要任何解释。

标准答案：{ground_truth}
实际回答：{answer}
分数："""


async def judge_accuracy(answer: str, ground_truth: str) -> Tuple[float, str]:
    """LLM 评判准确率，返回 (score 0-100, label)。超时时降级为关键词匹配。"""
    try:
        prompt = _ACCURACY_JUDGE_PROMPT.format(
            ground_truth=ground_truth[:400],
            answer=answer[:400],
        )
        llm = _get_judge_llm()
        resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=5.0)
        text = str(getattr(resp, "content", resp)).strip()
        # 提取第一个数字
        import re
        m = re.search(r"\d+", text)
        score = float(m.group()) if m else 0.0
        score = max(0.0, min(100.0, score))
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"LLM 评分超时或失败，降级关键词匹配: {e}")
        score = _keyword_accuracy(answer, ground_truth)

    if score >= 90:
        label = "优秀"
    elif score >= 70:
        label = "良好"
    elif score >= 40:
        label = "一般"
    else:
        label = "差"
    return score, label


def _keyword_accuracy(answer: str, ground_truth: str) -> float:
    """基于关键词重叠的简化准确率（LLM 不可用时的降级方案）"""
    import re
    ans_words = set(re.findall(r"[\w\u4e00-\u9fff]+", answer.lower()))
    gt_words = set(re.findall(r"[\w\u4e00-\u9fff]+", ground_truth.lower()))
    if not gt_words:
        return 50.0
    overlap = ans_words & gt_words
    return round(len(overlap) / len(gt_words) * 100, 1)


# ============================================================
# 幻觉检测（LLM-as-Judge，对比 sources 与回答）
# ============================================================

_HALLUCINATION_JUDGE_PROMPT = """\
你是一位严谨的事实核查专家。请判断"实际回答"中是否存在知识库来源文档**未提及**的编造内容（幻觉）。

判断规则：
- 如果回答中的核心事实信息都能在来源文档中找到依据，则判定为"无幻觉"。
- 如果回答中包含来源文档完全未提及的具体事实（如数据、政策、流程），则判定为"有幻觉"。
- 语气词、礼貌用语、合理的概括性表述不算幻觉。

严格按以下 JSON 格式输出，不要输出其他内容：
{{"hallucinated": true/false, "reason": "简要说明幻觉内容或无幻觉"}}

用户问题：{query}
知识库来源文档：{sources}
实际回答：{answer}
"""


async def detect_hallucination(
    answer: str,
    query: str,
    sources: Optional[List[Dict]],
    is_kb: bool,
) -> Tuple[Optional[bool], str]:
    """
    LLM-as-Judge 幻觉检测：对比回答与 sources，判断是否存在编造内容。
    返回 (is_hallucinated, reason)。None 表示不适用（非 KB 类或无来源）。
    """
    if not is_kb or not sources:
        return None, ""
    try:
        import re as _re
        # 将 sources 拼接为文本（取 content / page_content 字段）
        source_texts = []
        for s in sources[:5]:
            txt = s.get("content") or s.get("page_content") or str(s)
            source_texts.append(txt)
        sources_str = "\n---\n".join(source_texts)

        prompt = _HALLUCINATION_JUDGE_PROMPT.format(
            query=query,
            sources=sources_str,
            answer=answer,
        )
        llm = _get_judge_llm()
        resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=8.0)
        text = str(getattr(resp, "content", resp)).strip()
        
        # 解析 JSON 结果
        json_match = _re.search(r'\{.*\}', text, _re.DOTALL)
        if json_match:
            import json as _json
            obj = _json.loads(json_match.group())
            hallucinated = bool(obj.get("hallucinated", False))
            reason = obj.get("reason", "")
            if hallucinated is True:
                logger.info(f"========检测到幻觉: {reason}\n, 用户问题: {query}\n, 回答: {answer}\n, 来源文档: {sources_str}\n=========")
            return hallucinated, reason

        # JSON 解析失败时，简单关键词降级
        lower = text.lower()
        if "true" in lower:
            return True, text[:200]
        return False, ""
    except Exception as e:
        logger.warning(f"幻觉检测异常: {e}")
        return None, str(e)


# ============================================================
# 单条用例执行
# ============================================================

async def run_case(
    tc: TestCase,
    semaphore: asyncio.Semaphore,
    enable_hallucination: bool,
    enable_accuracy: bool,
    enable_ttft: bool = False,
) -> CaseResult:
    """执行一条测试用例并返回评估结果"""
    async with semaphore:
        result = CaseResult(
            name=tc.name,
            query=tc.query,
            case_type=tc.case_type,
            route_expected=tc.expected_route,
            tool_expected=tc.expected_tool,
        )

        thread_id = f"eval_{uuid.uuid4().hex[:8]}"
        state = State(thread_id=thread_id, query=tc.query, history=None, tenant_id=tc.tenant_id, user_id=tc.user_id)

        # ── 调用 Agent（直接调用 graph，不走 HTTP）────────────────
        t_start = time.perf_counter()
        try:
            if enable_ttft:
                # 使用流式接口执行，同时测量 TTFT 和总延迟
                raw = await _run_case_stream(result, tc, state, t_start)
            else:
                # 非流式执行（原有逻辑）
                try:
                    from service.graph.graph import run_graph
                except Exception:
                    from graph.graph import run_graph

                raw = await run_graph(state)

        except Exception as e:
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            result.error = str(e)[:200]
            logger.error(f"[{tc.name}] 执行异常: {traceback.format_exc()}")
            return result

        result.latency_ms = (time.perf_counter() - t_start) * 1000

        # ── 提取 answer / route / sources ─────────────────────────
        from core.hander import determine_answer
        route, answer, sources = determine_answer(raw)
        result.answer = answer or ""
        result.route_actual = route or ""

        # ── 意图路由准确率 ────────────────────────────────────────
        if tc.expected_route is not None:
            result.route_correct = (result.route_actual == tc.expected_route)

        # ── 工具调用正确率（order 类）────────────────────────────
        if tc.expected_tool:
            result.tool_actual = infer_tool_from_answer(result.answer)
            result.tool_correct = (result.tool_actual == tc.expected_tool)

        # ── 幻觉检测（KB 类）────────────────────────────────────
        if enable_hallucination:
            is_kb = (tc.case_type == "kb")
            result.hallucinated, result.hallucination_reason = await detect_hallucination(
                result.answer, tc.query, sources, is_kb
            )

        # ── 准确率评分（有标准答案的用例）───────────────────────
        if enable_accuracy and tc.ground_truth and result.answer:
            result.accuracy_score, result.accuracy_label = await judge_accuracy(
                result.answer, tc.ground_truth
            )

        # 如果启用了 TTFT 但未收到流式 token，将 TTFT 设为总延迟（整体返回的场景）
        if enable_ttft and result.ttft_ms < 0 and not result.error:
            result.ttft_ms = result.latency_ms

        ttft_info = f"ttft={result.ttft_ms:.0f}ms " if result.ttft_ms >= 0 else ""
        logger.info(
            f"[{tc.name}] ✓  latency={result.latency_ms:.0f}ms {ttft_info}"
            f"route={result.route_actual}({'✓' if result.route_correct else '✗' if result.route_correct is not None else '-'}) "
            f"acc={result.accuracy_score:.0f}{'/' + result.accuracy_label if result.accuracy_score >= 0 else '/N/A'} "
            f"hall={'✓' if result.hallucinated is False else '✗' if result.hallucinated else '-'} "
            f"tool={result.tool_actual or '-'}({'✓' if result.tool_correct else '✗' if result.tool_correct is not None else '-'})"
        )
        return result


async def _run_case_stream(
    result: CaseResult,
    tc: TestCase,
    state: State,
    t_start: float,
) -> dict:
    """通过流式接口执行图，并从队列中测量首字响应时间（TTFT）"""
    try:
        from service.graph.graph import run_graph_stream
    except Exception:
        from graph.graph import run_graph_stream

    stream_queue = asyncio.Queue()
    graph_task = asyncio.create_task(
        run_graph_stream(state, stream_queue)
    )

    ttft_recorded = False
    # 持续从队列读取事件，直到首个 token 到达或图执行完毕
    while not graph_task.done():
        try:
            event = await asyncio.wait_for(stream_queue.get(), timeout=0.3)
            if event.get("type") == "token" and not ttft_recorded:
                result.ttft_ms = (time.perf_counter() - t_start) * 1000
                ttft_recorded = True
                logger.debug(f"[{tc.name}] TTFT 首字响应: {result.ttft_ms:.1f}ms")
                break
        except asyncio.TimeoutError:
            continue

    # 图执行完毕后排空队列，补充检测首 token（极端情况：图极快完成）
    if not ttft_recorded:
        while not stream_queue.empty():
            event = stream_queue.get_nowait()
            if event.get("type") == "token":
                result.ttft_ms = (time.perf_counter() - t_start) * 1000
                ttft_recorded = True
                break

    raw = await graph_task
    return raw


# ============================================================
# 批量评估
# ============================================================

async def run_eval(
    cases: List[TestCase],
    concurrency: int,
    enable_hallucination: bool,
    enable_accuracy: bool,
    enable_ttft: bool = False,
) -> Tuple[List[CaseResult], EvalSummary]:
    """并发执行所有测试用例，返回结果列表和汇总"""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        run_case(tc, semaphore, enable_hallucination, enable_accuracy, enable_ttft)
        for tc in cases
    ]
    mode_desc = "流式（含 TTFT）" if enable_ttft else "非流式"
    logger.info(f"开始评估，共 {len(cases)} 条用例，并发={concurrency}，模式={mode_desc}")
    results: List[CaseResult] = await asyncio.gather(*tasks)

    # ── 汇总统计 ─────────────────────────────────────────────────
    summary = _compute_summary(results)
    return results, summary


def _compute_summary(results: List[CaseResult]) -> EvalSummary:
    s = EvalSummary(total=len(results), timestamp=datetime.now().isoformat())

    latencies = []
    acc_scores = []
    kb_hallu = []
    tool_results = []
    route_results = []

    for r in results:
        if r.error:
            s.error_count += 1

        latencies.append(r.latency_ms)

        # 准确率
        if r.accuracy_score >= 0:
            acc_scores.append(r.accuracy_score)

        # 幻觉
        if r.case_type == "kb":
            s.kb_cases += 1
            if r.hallucinated is not None:
                kb_hallu.append(r.hallucinated)

        # 工具调用
        if r.tool_expected is not None:
            s.order_cases += 1
            if r.tool_correct is not None:
                tool_results.append(r.tool_correct)

        # 路由
        if r.route_expected is not None:
            s.routable_cases += 1
            if r.route_correct is not None:
                route_results.append(r.route_correct)

    # ── 计算各指标 ──────────────────────────────────────────────
    if acc_scores:
        s.accuracy_cases = len(acc_scores)
        s.avg_accuracy = round(sum(acc_scores) / len(acc_scores), 1)
        s.accuracy_pass_rate = round(
            sum(1 for x in acc_scores if x >= 70) / len(acc_scores) * 100, 1
        )

    if kb_hallu:
        s.hallucinated_cases = sum(kb_hallu)
        s.hallucination_rate = round(s.hallucinated_cases / len(kb_hallu) * 100, 1)

    if tool_results:
        s.tool_correct_cases = sum(tool_results)
        s.tool_accuracy = round(s.tool_correct_cases / len(tool_results) * 100, 1)

    if route_results:
        s.route_correct_cases = sum(route_results)
        s.route_accuracy = round(s.route_correct_cases / len(route_results) * 100, 1)

    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        s.avg_latency_ms = round(sum(latencies) / n, 1)
        s.p50_latency_ms = round(latencies_sorted[int(n * 0.50)], 1)
        s.p95_latency_ms = round(latencies_sorted[min(int(n * 0.95), n - 1)], 1)
        s.max_latency_ms = round(max(latencies), 1)

    # ── TTFT 统计（仅统计成功测量的用例，ttft_ms >= 0）──────────
    ttft_values = [r.ttft_ms for r in results if r.ttft_ms >= 0]
    if ttft_values:
        s.ttft_cases = len(ttft_values)
        ttft_sorted = sorted(ttft_values)
        nt = len(ttft_sorted)
        s.avg_ttft_ms = round(sum(ttft_values) / nt, 1)
        s.p50_ttft_ms = round(ttft_sorted[int(nt * 0.50)], 1)
        s.p95_ttft_ms = round(ttft_sorted[min(int(nt * 0.95), nt - 1)], 1)
        s.max_ttft_ms = round(max(ttft_values), 1)

    return s


# ============================================================
# 报告输出
# ============================================================

def _fmt(val, fmt=".1f") -> str:
    """格式化数值，None/负数 → '-'"""
    if val is None or (isinstance(val, float) and val < 0):
        return "-"
    return format(val, fmt)


def print_report(results: List[CaseResult], summary: EvalSummary) -> None:
    """在控制台打印格式化评估报告"""
    W = 120
    DIVIDER = "─" * W
    THICK = "═" * W

    print(f"\n{THICK}")
    print(f"  客服 Agent 自动化评估报告   {summary.timestamp}")
    print(THICK)

    # ── 每条用例明细表 ────────────────────────────────────────────
    has_ttft = any(r.ttft_ms >= 0 for r in results)
    ttft_header = f" {'TTFT(ms)':>9}" if has_ttft else ""
    header = (
        f"{'用例名称':<22} {'类型':<7} {'路由(预→实)':<16} "
        f"{'准确率':>7} {'幻觉':>5} {'工具调用':>16} {'耗时(ms)':>9}{ttft_header} {'状态':>5}"
    )
    print(f"\n{header}")
    print(DIVIDER)

    for r in results:
        # 路由列
        exp = r.route_expected or "-"
        act = r.route_actual or "-"
        route_flag = "✓" if r.route_correct else ("✗" if r.route_correct is not None else " ")
        route_col = f"{exp}→{act}[{route_flag}]"

        # 准确率列
        acc_col = f"{r.accuracy_score:.0f}({r.accuracy_label})" if r.accuracy_score >= 0 else "N/A"

        # 幻觉列
        if r.hallucinated is None:
            hall_col = "  -"
        elif r.hallucinated:
            hall_col = "  ✗"
        else:
            hall_col = "  ✓"

        # 工具调用列
        if r.tool_expected:
            actual_t = r.tool_actual or "无"
            flag = "✓" if r.tool_correct else "✗"
            tool_col = f"{actual_t}[{flag}]"
        else:
            tool_col = "  -"

        # TTFT 列
        ttft_col = f" {r.ttft_ms:>9.0f}" if has_ttft and r.ttft_ms >= 0 else (" {:>9}".format("-") if has_ttft else "")

        # 状态列
        status = "ERR" if r.error else "OK "

        print(
            f"{r.name:<22} {r.case_type:<7} {route_col:<16} "
            f"{acc_col:>7} {hall_col:>5} {tool_col:>16} {r.latency_ms:>9.0f}{ttft_col} {status:>5}"
        )

    print(DIVIDER)

    # ── 汇总指标 ──────────────────────────────────────────────────
    print(f"\n{'汇总指标':^{W}}")
    print(DIVIDER)

    def metric_line(label, value, unit="", note=""):
        note_str = f"  ({note})" if note else ""
        print(f"  {label:<24} {value}{unit}{note_str}")

    metric_line(
        "1. 回答准确率（avg）",
        _fmt(summary.avg_accuracy, ".1f"),
        "/100",
        f"通过率(>=70分) {_fmt(summary.accuracy_pass_rate)}%，共 {summary.accuracy_cases} 条有标准答案",
    )
    metric_line(
        "2. 幻觉率",
        _fmt(summary.hallucination_rate),
        "%",
        f"{summary.hallucinated_cases}/{summary.kb_cases} 条 KB 回答触发幻觉告警",
    )
    metric_line(
        "3. 工具调用正确率",
        _fmt(summary.tool_accuracy),
        "%",
        f"{summary.tool_correct_cases}/{summary.order_cases} 条订单用例",
    )
    metric_line(
        "4. 意图路由准确率",
        _fmt(summary.route_accuracy),
        "%",
        f"{summary.route_correct_cases}/{summary.routable_cases} 条可路由用例",
    )
    print(f"  {'5. 响应时间':<24} "
          f"avg={summary.avg_latency_ms:.0f}ms  "
          f"p50={summary.p50_latency_ms:.0f}ms  "
          f"p95={summary.p95_latency_ms:.0f}ms  "
          f"max={summary.max_latency_ms:.0f}ms")

    if summary.ttft_cases > 0:
        print(f"  {'6. 首字响应时间(TTFT)':<24} "
              f"avg={summary.avg_ttft_ms:.0f}ms  "
              f"p50={summary.p50_ttft_ms:.0f}ms  "
              f"p95={summary.p95_ttft_ms:.0f}ms  "
              f"max={summary.max_ttft_ms:.0f}ms  "
              f"(共 {summary.ttft_cases} 条)")

    print(DIVIDER)
    print(f"  总用例: {summary.total}  |  异常: {summary.error_count}")
    print(f"{THICK}\n")


def save_report(
    results: List[CaseResult],
    summary: EvalSummary,
    output_prefix: str,
) -> str:
    """保存 JSON 格式的完整评估报告，返回文件路径"""
    reports_dir = LOG_DIR / "eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_prefix}_{ts}.json"
    path = reports_dir / filename

    data = {
        "timestamp": summary.timestamp,
        "summary": asdict(summary),
        "cases": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"评估报告已保存: {path}")
    return str(path)


# ============================================================
# 入口
# ============================================================

def _load_cases(dataset_path: Optional[str], tenant: str) -> List[TestCase]:
    """加载测试用例（外部 JSON 或内置集）"""
    raw = []
    if dataset_path:
        with open(dataset_path, encoding="utf-8") as f:
            raw = json.load(f)
        logger.info(f"从 {dataset_path} 加载了 {len(raw)} 条用例")
    else:
        raw = BUILT_IN_CASES
        logger.info(f"使用内置测试集，共 {len(raw)} 条用例")

    cases = []
    for d in raw:
        cases.append(TestCase(
            name=d.get("name", "未命名"),
            query=d["query"],
            case_type=d.get("case_type", "kb"),
            expected_route=d.get("expected_route"),
            expected_tool=d.get("expected_tool"),
            ground_truth=d.get("ground_truth"),
            tenant_id=d.get("tenant_id", tenant),
        ))
    return cases


async def main(args: argparse.Namespace) -> None:
    cases = _load_cases(args.dataset, args.tenant)

    results, summary = await run_eval(
        cases=cases,
        concurrency=args.concurrency,
        enable_hallucination=not args.skip_hallucination,
        enable_accuracy=not args.skip_accuracy,
        enable_ttft=not args.skip_ttft,
    )

    print_report(results, summary)
    path = save_report(results, summary, args.output)
    print(f"  报告已保存至: {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="客服 Agent 自动化评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", default=None,
        help="自定义测试集 JSON 文件路径（默认使用内置测试集）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="并发评估数量（默认 2，建议不超过 4 避免 API 限速）",
    )
    parser.add_argument(
        "--output", default="eval_report",
        help="报告文件名前缀（默认 eval_report）",
    )
    parser.add_argument(
        "--skip-hallucination", action="store_true",
        help="跳过幻觉检测（节省约 600ms/条 KB 用例）",
    )
    parser.add_argument(
        "--skip-accuracy", action="store_true",
        help="跳过 LLM 准确率评分（节省约 300ms/条有标准答案用例）",
    )
    parser.add_argument(
        "--skip-ttft", action="store_true",
        help="跳过 TTFT 首字响应时间测量（使用非流式接口，速度更快）",
    )
    parser.add_argument(
        "--tenant", default="default",
        help="评估使用的租户 ID（默认 default）",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
