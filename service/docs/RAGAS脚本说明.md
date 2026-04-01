# RAGAS 自动化评估脚本说明

## 1. 概述

RAGAS（Retrieval Augmented Generation Assessment）评估模块用于自动化评估 RAG 系统的回答质量。模块位于 `service/rag/ragas_evaluator.py`，对应 API 接口在 `service/app.py` 中注册。

核心能力：

- 对单个问答用例进行质量评估
- 批量评估多个用例并生成报告
- 当 RAGAS 库不可用时自动降级为简化评估（基于关键词匹配）
- 评估报告以 JSON 格式持久化存储

---

## 2. 评估指标

| 指标 | 含义 | 默认阈值 |
|------|------|----------|
| **faithfulness**（忠实度） | 回答是否忠实于检索到的上下文 | ≥ 0.85 |
| **answer_relevancy**（回答相关性） | 回答是否与用户问题相关 | ≥ 0.8 |
| **context_precision**（上下文精确度） | 检索上下文中相关片段的命中率 | ≥ 0.7 |
| **context_recall**（上下文召回率） | 是否召回了足够多的相关信息 | ≥ 0.7 |

**通过条件**：忠实度和回答相关性均达标；如果提供了标准答案，还要求上下文精确度或召回率至少有一项达标。

---

## 3. 核心数据结构

### EvaluationCase — 评估用例

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 用例名称 |
| `query` | `str` | 用户查询 |
| `ground_truth` | `str \| None` | 标准答案（可选） |

### EvaluationResult — 评估结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `case_name` | `str` | 用例名称 |
| `query` / `answer` | `str` | 查询与系统回答 |
| `contexts` | `List[str]` | 检索到的上下文 |
| `faithfulness` 等 | `float` | 四项评估指标 |
| `passed` | `bool` | 是否通过 |
| `problem_description` | `str` | 未通过时的问题描述 |

### EvaluationSummary — 评估汇总

包含总用例数、通过/失败数、各指标平均值、耗时、阈值配置等。

---

## 4. 评估流程

```
用户查询 ──▶ retrieve_kb() 获取检索上下文
                │
                ▼
        RAGAS 可用？──── 是 ──▶ _evaluate_with_ragas()
                │                    │
               否                    │
                │                    ▼
                ▼              datasets.Dataset 构建
        _evaluate_simple()     + ragas.evaluate() 执行
        (关键词匹配降级)              │
                │                    │
                ▼◄───────────────────┘
        生成 EvaluationResult（含通过/失败判定 + 问题描述）
```

**降级策略**：当 `ragas` 库未安装或版本不兼容时，自动切换为基于关键词重叠率的简化评估，保证接口始终可用。

---

## 5. API 接口

所有接口支持通过 `X-Tenant-ID` 请求头或 `?tenant=` 查询参数指定租户。

### 5.1 健康检查

```
GET /api/ragas/health
```

**响应示例**：

```json
{
  "available": true,
  "evaluator_initialized": true
}
```

### 5.2 获取默认测试用例

```
GET /api/ragas/test-cases
```

返回内置的 5 个默认测试用例（产品相关、售前咨询、售后服务、订单查询、直答闲聊）。

### 5.3 单个用例评估

```
POST /api/ragas/evaluate/single
```

**请求体**：

```json
{
  "case": {
    "name": "产品相关",
    "query": "AI会不会替代程序员？",
    "ground_truth": "不会完全替代，但会极大提升效率。"
  },
  "answer": "AI 不会完全替代程序员...",
  "route": "kb"
}
```

**响应示例**：

```json
{
  "case_name": "产品相关",
  "query": "AI会不会替代程序员？",
  "answer": "AI 不会完全替代程序员...",
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_precision": 0.78,
    "context_recall": 0.65
  },
  "passed": true,
  "problem_description": "无明显问题"
}
```

### 5.4 批量评估

```
POST /api/ragas/evaluate/batch
```

**请求体**：

```json
{
  "cases": [
    { "name": "用例1", "query": "...", "ground_truth": "..." },
    { "name": "用例2", "query": "..." }
  ],
  "faith_threshold": 0.85,
  "relev_threshold": 0.8 
}
```

批量评估为异步执行：接口立即返回 `task_id`，评估在后台运行。评估过程会自动调用 `run_graph` 获取系统回答，再逐个评估。完成后报告自动保存到 `service/logs/ragas_reports/` 目录。

**响应示例**：

```json
{
  "message": "评估任务已启动",
  "task_id": "140234567890",
  "cases_count": 5
}
```

### 5.5 查看评估报告

```
GET /api/ragas/reports
```

返回最近 20 份报告的摘要信息（文件名、时间戳、汇总指标）。

---

## 6. 报告存储

- 存储路径：`service/logs/ragas_reports/`
- 文件格式：`ragas_report_YYYYMMDD_HHMMSS.json`
- 报告结构：

```json
{
  "timestamp": "2026-03-23T10:00:00",
  "summary": {
    "total_cases": 5,
    "passed_cases": 4,
    "failed_cases": 1,
    "avg_faithfulness": 0.82,
    "avg_answer_relevancy": 0.88,
    "avg_context_precision": 0.71,
    "avg_context_recall": 0.65,
    "elapsed_seconds": 45.2,
    "thresholds": { "faithfulness": 0.85, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.7 }
  },
  "cases": [ ... ]
}
```

---

## 7. 独立运行

评估脚本支持独立执行，用于快速验证环境和默认用例：

```bash
cd service
python -m rag.ragas_evaluator
```

输出 RAGAS 是否可用以及默认测试用例数量。

---

## 8. 依赖与注意事项

- **必须依赖**：`langchain`、项目内 `config`（提供 LLM 和 Embeddings 实例）、`tools.retrieve_kb`
- **可选依赖**：`ragas`、`datasets` — 未安装时自动降级为简化评估
- 评估器采用**单例模式**（`get_evaluator()`），避免重复初始化
- 批量评估通过 `asyncio.create_task` 异步执行，不阻塞 API 响应
- `_evaluate_with_ragas` 兼容多版本 RAGAS 的 LLM/Embeddings 包装器命名差异
