"""
RAGAS 自动化评估模块

用于评估 RAG 系统的质量，包括：
- 忠实度 (faithfulness)：回答是否忠实于检索上下文
- 回答相关性 (answer_relevancy)：回答是否与问题相关
- 上下文精确度 (context_precision)：检索上下文的精确度
- 上下文召回率 (context_recall)：检索上下文的召回率
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# 确保导入路径正确（指向 service/ 目录）
base = Path(__file__).resolve().parent.parent
if str(base) not in sys.path:
    sys.path.append(str(base))

import core.config as config  
from tools.service_tools import retrieve_kb

from core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class EvaluationCase:
    """评估用例"""
    name: str
    query: str
    ground_truth: Optional[str] = None  # 标准答案（可选）


@dataclass
class EvaluationResult:
    """单个评估结果"""
    case_name: str
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    passed: bool = False
    problem_description: str = ""
    route: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class EvaluationSummary:
    """评估汇总"""
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    elapsed_seconds: float
    thresholds: Dict[str, float]
    timestamp: str


class RagasEvaluator:
    """RAGAS 评估器"""
    
    def __init__(
        self,
        faith_threshold: float = 0.85,
        relev_threshold: float = 0.7,
        precision_threshold: float = 0.7,
        recall_threshold: float = 0.7,
        reports_dir: Optional[str] = None
    ):
        """
        初始化评估器
        
        Args:
            faith_threshold: 忠实度阈值
            relev_threshold: 回答相关性阈值
            precision_threshold: 上下文精确度阈值
            recall_threshold: 上下文召回率阈值
            reports_dir: 报告存储目录
        """
        self.faith_threshold = faith_threshold
        self.relev_threshold = relev_threshold
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        
        # 报告目录
        if reports_dir is None:
            reports_dir = os.path.join(str(base), "logs", "ragas_reports")
        os.makedirs(reports_dir, exist_ok=True)
        self.reports_dir = reports_dir
        
        # 尝试加载 RAGAS
        self._load_ragas()
        
        # 初始化 LLM 和 Embeddings
        self.llm = config.get_llm()
        self.embeddings = config.get_embeddings()
        
        logger.info(f"RAGAS 评估器初始化完成，阈值: faith={faith_threshold}, relev={relev_threshold}")
    
    def _load_ragas(self):
        """加载 RAGAS 库"""
        self.evaluate = None
        self.RagasLLM = None
        self.RagasEmb = None
        self.faithfulness = None
        self.answer_relevancy = None
        self.context_precision = None
        self.context_recall = None
        
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            
            # 尝试导入 LLM 和 Embeddings 包装器
            ragas_llm = None
            ragas_emb = None
            
            try:
                from ragas.llms import LangchainLLMWrapper as RagasLLM
                ragas_llm = RagasLLM
            except ImportError:
                try:
                    from ragas.llms import LangchainLLM as RagasLLM
                    ragas_llm = RagasLLM
                except ImportError:
                    try:
                        from ragas.llms import LangChainLLM as RagasLLM
                        ragas_llm = RagasLLM
                    except ImportError:
                        pass
            
            try:
                from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmb
                ragas_emb = RagasEmb
            except ImportError:
                try:
                    from ragas.embeddings import LangchainEmbeddings as RagasEmb
                    ragas_emb = RagasEmb
                except ImportError:
                    try:
                        from ragas.embeddings import LangChainEmbeddings as RagasEmb
                        ragas_emb = RagasEmb
                    except ImportError:
                        pass
            
            self.evaluate = evaluate
            self.RagasLLM = ragas_llm
            self.RagasEmb = ragas_emb
            self.faithfulness = faithfulness
            self.answer_relevancy = answer_relevancy
            self.context_precision = context_precision
            self.context_recall = context_recall
            
            logger.info("RAGAS 库加载成功")
        except ImportError as e:
            logger.warning(f"RAGAS 未安装或版本不兼容: {e}，将使用简化评估")
        except Exception as e:
            logger.warning(f"RAGAS 初始化失败: {e}")
    
    def is_ragas_available(self) -> bool:
        """检查 RAGAS 是否可用"""
        return all([
            self.evaluate is not None,
            self.faithfulness is not None,
            self.answer_relevancy is not None
        ])
    
    def get_contexts(self, query: str, tenant_id: Optional[str] = None) -> List[str]:
        """
        获取检索上下文
        
        Args:
            query: 查询
            tenant_id: 租户ID
            
        Returns:
            上下文列表
        """
        try:
            serialized, docs = retrieve_kb(query, tenant_id)
            ctxs = []
            for d in docs or []:
                c = getattr(d, "page_content", "")
                if isinstance(c, str) and c.strip():
                    ctxs.append(c)
            return ctxs
        except Exception as e:
            logger.error(f"获取检索上下文失败: {e}")
            return []
    
    def _evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        使用 RAGAS 进行评估
        
        Args:
            query: 查询
            answer: 回答
            contexts: 上下文列表
            ground_truth: 标准答案（可选）
            
        Returns:
            评估指标字典
        """
        if not self.is_ragas_available():
            return self._evaluate_simple(query, answer, contexts, ground_truth)
        
        try:
            from datasets import Dataset
            
            # 准备数据集
            data_dict = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts]
            }
            if ground_truth:
                data_dict["ground_truth"] = [ground_truth]
            
            ds = Dataset.from_dict(data_dict)
            
            # 准备指标
            metrics = [self.faithfulness, self.answer_relevancy]
            if ground_truth:
                if self.context_precision:
                    metrics.append(self.context_precision)
                if self.context_recall:
                    metrics.append(self.context_recall)
            
            # 准备 LLM 和 Embeddings
            ragas_llm = self.RagasLLM(self.llm) if self.RagasLLM else self.llm
            ragas_emb = self.RagasEmb(self.embeddings) if self.RagasEmb else self.embeddings
            
            # 执行评估
            res = self.evaluate(
                ds,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb
            )
            
            # 解析结果
            result = {}
            if hasattr(res, "to_dict"):
                result_dict = res.to_dict()
            elif isinstance(res, dict):
                result_dict = res
            elif hasattr(res, "to_pandas"):
                df = res.to_pandas()
                row = df.iloc[0].to_dict()
                result_dict = row
            else:
                result_dict = {}
            
            # 提取指标
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if key in result_dict:
                    val = result_dict[key]
                    if isinstance(val, list) and val:
                        result[key] = float(val[0])
                    else:
                        result[key] = float(val)
                else:
                    result[key] = 0.0
            
            return result
            
        except Exception as e:
            logger.warning(f"RAGAS 评估失败，使用简化评估: {e}")
            return self._evaluate_simple(query, answer, contexts, ground_truth)
    
    def _evaluate_simple(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        简化评估（当 RAGAS 不可用时使用）
        
        Args:
            query: 查询
            answer: 回答
            contexts: 上下文列表
            ground_truth: 标准答案（可选）
            
        Returns:
            评估指标字典
        """
        import re
        
        result = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }
        
        # 1. 简化的忠实度：检查回答是否包含上下文中的关键词
        if contexts and answer:
            context_text = " ".join(contexts).lower()
            answer_text = answer.lower()
            
            # 简单的关键词匹配
            words = set(re.findall(r'\w+', answer_text))
            context_words = set(re.findall(r'\w+', context_text))
            overlap = words & context_words
            
            if words:
                result["faithfulness"] = min(1.0, len(overlap) / len(words))
        
        # 2. 简化的回答相关性：检查回答是否包含查询中的关键词
        if query and answer:
            query_words = set(re.findall(r'\w+', query.lower()))
            answer_words = set(re.findall(r'\w+', answer.lower()))
            overlap = query_words & answer_words
            
            if query_words:
                result["answer_relevancy"] = min(1.0, len(overlap) / len(query_words))
        
        # 3. 如果有标准答案，计算上下文精确度和召回率
        if ground_truth and contexts:
            gt_words = set(re.findall(r'\w+', ground_truth.lower()))
            context_text = " ".join(contexts).lower()
            context_words = set(re.findall(r'\w+', context_text))
            
            # 精确度：上下文中有多少标准答案的词
            gt_in_context = gt_words & context_words
            if gt_words:
                result["context_recall"] = min(1.0, len(gt_in_context) / len(gt_words))
            
            # 召回率：标准答案有多少在上下文中
            if context_words:
                result["context_precision"] = min(1.0, len(gt_in_context) / len(context_words))
        
        return result
    
    def evaluate_case(
        self,
        case: EvaluationCase,
        answer: str,
        route: Optional[str] = None,
        tenant_id: Optional[str] = None,
        contexts: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        评估单个用例

        Args:
            case: 评估用例
            answer: 系统回答
            route: 路由标签（可选）
            tenant_id: 租户ID（可选）
            contexts: 检索上下文列表（可选）；若已由 kb_node 提供则直接使用，
                      否则内部调用 retrieve_kb 自行检索
        Returns:
            评估结果
        """
        # 优先使用外部传入的 contexts（来自 kb_node.sources），减少重复检索
        if contexts is None:
            logger.info(f"evaluate_case: contexts 未传入，自行调用 retrieve_kb 检索，query='{case.query[:40]}'")
            contexts = self.get_contexts(case.query, tenant_id)
        else:
            logger.info(f"evaluate_case: 使用外部传入的 contexts，共 {len(contexts)} 条，query='{case.query[:40]}'")
        
        # 执行评估
        if self.is_ragas_available():
            metrics = self._evaluate_with_ragas(
                case.query,
                answer,
                contexts,
                case.ground_truth
            )
        else:
            metrics = self._evaluate_simple(
                case.query,
                answer,
                contexts,
                case.ground_truth
            )
        
        # 构建问题描述
        problem_parts = []
        if metrics["faithfulness"] < self.faith_threshold:
            problem_parts.append("忠实度偏低，可能未严格依据检索上下文作答")
        if metrics["answer_relevancy"] < self.relev_threshold:
            problem_parts.append("回答相关性不足，未充分贴合用户问题")
        if case.ground_truth and metrics["context_precision"] < self.precision_threshold:
            problem_parts.append("检索精确度偏低，相关片段未充分命中")
        if case.ground_truth and metrics["context_recall"] < self.recall_threshold:
            problem_parts.append("检索召回率偏低，可能遗漏部分相关信息")
        
        problem_description = "；".join(problem_parts) if problem_parts else "无明显问题"
        
        # 判断是否通过
        passed = (
            metrics["faithfulness"] >= self.faith_threshold and
            metrics["answer_relevancy"] >= self.relev_threshold
        )
        if case.ground_truth:
            passed = passed and (
                metrics["context_precision"] >= self.precision_threshold or
                metrics["context_recall"] >= self.recall_threshold
            )
        
        # 创建评估结果
        result = EvaluationResult(
            case_name=case.name,
            query=case.query,
            answer=answer,
            contexts=contexts,
            ground_truth=case.ground_truth,
            faithfulness=metrics.get("faithfulness", 0.0),
            answer_relevancy=metrics.get("answer_relevancy", 0.0),
            context_precision=metrics.get("context_precision", 0.0),
            context_recall=metrics.get("context_recall", 0.0),
            passed=passed,
            problem_description=problem_description,
            route=route,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"用例评估完成: {case.name} | "
            f"faith={result.faithfulness:.3f} | "
            f"relev={result.answer_relevancy:.3f} | "
            f"结果={'通过' if passed else '失败'}"
        )
        
        return result
    
    def evaluate_batch(
        self,
        cases: List[EvaluationCase],
        answer_provider,
        tenant_id: Optional[str] = None
    ) -> tuple[List[EvaluationResult], EvaluationSummary]:
        """
        批量评估
        
        Args:
            cases: 评估用例列表
            answer_provider: 回答提供者函数，签名: (query: str) -> Tuple[str, Optional[str]]
            tenant_id: 租户ID（可选）
            
        Returns:
            (评估结果列表, 评估汇总)
        """
        t0 = time.time()
        results = []
        pass_count = 0
        fail_count = 0
        
        logger.info(f"开始批量评估，共 {len(cases)} 个用例")
        
        for case in cases:
            try:
                # 获取回答；支持两种返回格式：
                #   (answer, route)              — 旧格式，contexts 由内部检索
                #   (answer, contexts, route)    — 新格式，contexts 来自 kb_node.sources
                raw = answer_provider(case.query)
                if len(raw) == 3:
                    answer, kb_contexts, route = raw
                    logger.info(f"answer_provider 返回 3-tuple，使用 kb_node 上下文，共 {len(kb_contexts)} 条")
                else:
                    answer, route = raw
                    kb_contexts = None
                
                # 评估
                result = self.evaluate_case(case, answer, route, tenant_id, contexts=kb_contexts)
                results.append(result)
                
                if result.passed:
                    pass_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"用例 {case.name} 评估失败: {e}")
                fail_count += 1
                results.append(EvaluationResult(
                    case_name=case.name,
                    query=case.query,
                    answer="",
                    contexts=[],
                    ground_truth=case.ground_truth,
                    passed=False,
                    problem_description=f"评估异常: {str(e)}"
                ))
        
        # 计算汇总
        elapsed = time.time() - t0
        total = len(results)
        
        avg_faithfulness = sum(r.faithfulness for r in results) / total if results else 0.0
        avg_answer_relevancy = sum(r.answer_relevancy for r in results) / total if results else 0.0
        avg_context_precision = sum(r.context_precision for r in results) / total if results else 0.0
        avg_context_recall = sum(r.context_recall for r in results) / total if results else 0.0
        
        summary = EvaluationSummary(
            total_cases=total,
            passed_cases=pass_count,
            failed_cases=fail_count,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_precision=avg_context_precision,
            avg_context_recall=avg_context_recall,
            elapsed_seconds=elapsed,
            thresholds={
                "faithfulness": self.faith_threshold,
                "answer_relevancy": self.relev_threshold,
                "context_precision": self.precision_threshold,
                "context_recall": self.recall_threshold
            },
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"批量评估完成: 总={total}, 通过={pass_count}, 失败={fail_count}, "
            f"耗时={elapsed:.2f}s"
        )
        
        return results, summary
    
    def save_report(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        report_name: Optional[str] = None
    ) -> str:
        """
        保存评估报告
        
        Args:
            results: 评估结果列表
            summary: 评估汇总
            report_name: 报告名称（可选）
            
        Returns:
            报告文件路径
        """
        if report_name is None:
            report_name = f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = os.path.join(self.reports_dir, report_name)
        
        # 构建报告数据
        report_data = {
            "timestamp": summary.timestamp,
            "summary": asdict(summary),
            "cases": [asdict(r) for r in results]
        }
        
        # 保存报告
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存: {report_path}")
        return report_path
    
    def get_default_test_cases(self) -> List[EvaluationCase]:
        """获取默认测试用例"""
        return [
            EvaluationCase(
                name='退货与退款',
                query='退款将在哪里发放？',
                ground_truth='如果您在线支付，退款将根据所使用的支付方式进行。它将存入您的 LightInTheBox 信用帐户或返回到用于付款的原始帐户/卡。请按照客户支持团队提供的说明查看退款进度。 如果您是线下支付，我们的客服部门将与您联系并解释详细的退款流程。请按照他们的说明查看退款的方式和地点。'
            ),
            EvaluationCase(
                name='退货与退款',
                query='定制产品的退货/取消政策',
                ground_truth='定制产品的退货/取消政策 交货时间声明 付款确认后，所有连衣裙（包括标准尺码）均由裁缝手工制作。通常，裁缝师需要 7-19 个工作日从头开始制作并将其运送到我们的仓库。然后我们会将其运送给您，运送时间因运送方式而异。 您可以在商品页面查看“剪裁时间”和“发货时间”。 例如： 剪裁加工时间：7-19个工作日。 如果您选择加急快递运送，运送时间约为 4-7 个工作日（6~9 个日历日）。 尺寸声明 请测量您的身体并根据您的准确测量值选择尺码。 （我们的尺寸可能与您当地的尺寸不同）。 视频测量指南：点击这里！ 取消政策 付款确认后24小时内免费取消！ 24 小时后取消可获 50% 退款！ 订单发出后无法取消！ 退货政策 您的满意对我们非常重要。建议您在包裹到达后先检查产品状态，以确保其符合您的订单要求。请尽快试穿衣服，不要更换、清洗或撕掉标签。 1、具体政策 如果您的商品有缺陷、损坏或运输不当，您有资格获得全额退款。请在收到订单后 30 天内联系我们的客户服务。我们要求每项索赔都有理由，例如拍照、视频证明等。 由于所有产品（包括标准尺寸）均为订单生产，因此我们不接受因客户个人原因而退货。如果您有任何疑问或希望退回订单中的部分或部分产品，请在收到订单后 7 天内联系我们的客服并附上图片说明退货原因。 *客户原因：不再需要、买错、不符合预期、改变主意等 您提交票据后，我们的客户服务将根据我们的政策、保修、产品状态以及您提供的证明批准您的退货请求。在我们事先不知情的情况下，我们将无法处理退货。 2. 退货要求 2.1 尺寸偏差 由于我们所有的连衣裙都是手工缝制和定制的，因此成品连衣裙在指定尺寸的任何方向上都可能存在尺寸偏差。如果您的连衣裙尺码符合您的订单规格，胸围、腰围偏差在3CM以内，臀围偏差在5CM以内，属于定制连衣裙的正常尺码范围。如果超出此标准，请联系我们的客服寻求指导并提供支持证明（照片或视频）。 测量指南 PDF：http://litb.cc/l/wLKN 测量指南视频：http://litb.cc/l/taFe 视频1-前半身像：http://litb.cc/l/wLKB 视频2 - 后胸围：http://litb.cc/l/wLKV 2.2 颜色不匹配 由于电脑或其他设备屏幕分辨率的差异，色差属于正常现象。轻微的色差可能并不意味着衣服有缺陷或发送了错误的物品。但如果您确定您收到的产品颜色有误，请联系客服。 2.3 买很多件但只保留一件 由于所有产品（包括标准尺寸）均为订单生产，因此我们不接受“买多只留一件”的退货。我们的客户服务团队可能会拒绝为此目的的退款请求。 如果您退回2件以上商品（包括相同产品或款式、同时订购多种尺寸或颜色），您将需要支付50%的购买费用。 3. 兑换政策 我们目前不提供任何产品交换服务。由于所有商品均按订单生产，因此我们没有现成的连衣裙或产品可用于更换您退回的商品。 4、注意事项 如果不能接受退货，我们有权不处理退款。'
            ),
            EvaluationCase(
                name='退货与退款',
                query='如果我改变主意，我可以拒绝包裹吗？',
                ground_truth='无论出于何种原因，如果您在包裹运输过程中想要拒收包裹，您需要等到您收到包裹后再提出退货请求。 如果您拒绝从邮递员处领取包裹或未到当地自提店领取包裹，我们的客服将无法判断包裹的情况，因此无法处理您的退货请求。 如果由于客户个人原因（请查看下面的详细信息）而导致包裹被退回我们的仓库，我们将联系您重新支付重寄邮费（通过PayPal）并安排重寄。但请您理解，在这种情况下，我们不会退款。  客户个人原因详情： * 地址错误/无收货人 * 联系信息无效/送货电话和电子邮件无人接听 * 客户拒绝接受包裹/缴纳税费/完成清关 * 未在截止日期前领取包裹'
            ),
            EvaluationCase(
                name='退货与退款',
                query='取消政策',
                ground_truth='对于除定制商品外的所有商品，在订单发货之前，您可以随时取消订单并获得全额退款。 对于定制产品： 付款确认后 24 小时内取消的订单可以获得全额退款。 生产开始后取消的订单可能会被取消，但您将承担产品价格的 50%。 已发货的定制产品无法取消。'
            ),
        ]


# 单例模式
_evaluator_instance: Optional[RagasEvaluator] = None


def get_evaluator() -> RagasEvaluator:
    """获取评估器单例"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RagasEvaluator()
    return _evaluator_instance


if __name__ == "__main__":
    # 简单测试
    evaluator = get_evaluator()
    print(f"RAGAS 可用: {evaluator.is_ragas_available()}")
    
    # 获取默认测试用例
    cases = evaluator.get_default_test_cases()
    print(f"默认测试用例数: {len(cases)}")
