# Agent 评估测试集说明（200 条）

本文档说明 `eval_agent.py` 配套的大规模测试集 **`eval_agent_cases_200.json`** 的设计、与内置用例的差异及使用方法。数据与当前 **LangGraph 意图标签**（`service/graph/graph.py` 中 `VALID_INTENTS`）对齐，并与 RAG 知识库语料紧密结合。

---

## 1. 文件位置与规模

| 项目 | 说明 |
|------|------|
| 数据文件 | `service/data/eval_agent_cases_200.json` |
| 用例条数 | **200** |
| 格式 | JSON 数组，每条为对象，字段与 `eval_agent.py` 中 `_load_cases` 约定一致 |

---

## 2. 与 `eval_agent.py` 内置 `BUILT_IN_CASES` 的关系

- **内置集**（`eval_agent.py` 内嵌列表）体量小，便于快速冒烟；其中部分 `expected_route` 使用了 `product` / `presale` / `postsale` / `direct` 等**细分标签**。
- **当前对话图**在 `intent` 节点写入的 `state.route` 为四元意图：**`faq`**（知识库）、**`order`**（订单）、**`chitchat`**（闲聊）、**`human`**（转人工）；槽位缺失或模糊意图时可能走反问，最终对外展示时若存在 `clarify_question`，`determine_answer` 仅在 `route` 为空时才会落到 **`clarify`**（见 `service/core/hander.py`）。
- **本 200 条集**将「可判定」的 `expected_route` 与上述实现对齐（例如 KB 统一标为 **`faq`**，闲聊标为 **`chitchat`**），用于 **意图路由准确率** 统计时含义清晰；避免因标签体系不一致导致指标长期偏低或不可解释。

---

## 3. 用例构成与分布

| 类别 | `case_type` | 条数 | 说明 |
|------|-------------|------|------|
| 知识库 FAQ | `kb` | **150** | 其中 **120** 条问答对解析；**30** 条为同一知识点的**换说法/前缀改写**（`KB-PARA-*`），用于鲁棒性 |
| 订单工具 | `order` | **35** | 覆盖 `get_order_detail`、`cancel_order`、`apply_refund`、`get_logistics_info`、`modify_order_address`；订单号形如 `ORD` + 8～16 位数字，符合 `validate_order_id_format` |
| 闲聊直答 | `direct` | **8** | `expected_route` 为 **`chitchat`**（与图中闲聊意图一致） |
| 转人工 | `human` | **4** | `expected_route` 为 **`human`** |
| 边界 | `edge` | **3** | 含空查询、无订单号取消、模糊说法；其中 **2** 条 `expected_route` 为 `null`，表示**不强制路由标签**（仅关注不崩溃与行为合理） |

**说明（边界用例）：**

- `EDGE-无单号取消`：缺订单号时会触发反问话术；评估脚本里最终 `route` 在存在 `clarify_question` 且原意图已为 `order` 时仍为 **`order`**，故标注为 `order`。
- `expected_route: null`：不参与「意图路由准确率」分子分母（与 `eval_agent.py` 汇总逻辑一致）。

---

## 4. 单条用例字段说明

与脚本中 `TestCase` / JSON 加载逻辑一致：

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 建议 | 用例名称，便于报告与日志定位 |
| `query` | 是 | 用户输入 |
| `case_type` | 是 | `kb` / `order` / `direct` / `human` / `edge` |
| `expected_route` | 否 | 预期路由；`null` 表示本条不统计路由准确率 |
| `expected_tool` | 否 | 仅 `order` 类：预期工具名，与脚本中 `infer_tool_from_answer` 推断结果对比 |
| `ground_truth` | 否 | KB 类建议填写：供 LLM-as-Judge 准确率使用 |
| `tenant_id` | 否 | 默认由命令行 `--tenant` 注入 |

---

## 5. 与知识库文档的对应关系

- 知识库条目的 **标准答案** 优先取自 `faq.txt` 中 **「A：」** 段落，与 RAG 检索语料一致，便于评估 **回答准确率** 与 **幻觉率**（KB 类且存在 `sources` 时）。
- 标签列（「标签：」）仅用于生成可读用例名（如 `KB-我的订单-001`），**不参与**运行时检索。
- **30 条改写**在同一 `ground_truth` 下替换用户问法，用于观察「同义问法」下的路由与回答稳定性。

---

## 6. 使用方法

在 **`service/app`** 目录下执行（与脚本内 `sys.path` 约定一致），或先 `cd` 到 `service` 根目录并保证 `app` 在 `PYTHONPATH` 中，按你项目现有方式运行：

```bash
cd service/app
python eval_agent.py --dataset ../data/eval_agent_cases_200.json --concurrency 2 --output eval_report_200
```

常用参数（与脚本 `--help` / 文件头注释一致）：

- `--concurrency`：并发数，200 条建议 **2～4**，避免 API 限流。
- `--skip-hallucination`：跳过幻觉检测以缩短耗时。
- `--skip-accuracy`：跳过 LLM 准确率评分。
- `--tenant`：租户 ID。

报告 JSON 默认写入：`service/app/logs/eval_reports/<前缀>_<时间戳>.json`。

---

## 7. 指标与限制（阅读报告前必读）

`eval_agent.py` 当前汇总的核心指标如下（与控制台报告、JSON `summary` 一致）：

1. **回答准确率（avg_accuracy）**：仅统计有 `ground_truth` 的用例，分值 0-100；并额外给出 **通过率（accuracy_pass_rate）**，阈值为 `>= 70`。
2. **幻觉率（hallucination_rate）**：仅对 `case_type == "kb"` 且能拿到 `sources` 的 KB 回答进行检测；跳过或失败时汇总中可能显示为 `-`。
3. **工具调用正确率（tool_accuracy）**：仅统计有 `expected_tool` 的订单用例；实际工具来自 `infer_tool_from_answer` 的文本信号推断，而非图内真实 tool 调用日志。
4. **意图路由准确率（route_accuracy）**：仅统计 `expected_route` 非 `null` 的用例；最终 `route` 以 `determine_answer` 解析结果为准。
5. **响应时间（Latency）**：统计全部用例的 `avg / p50 / p95 / max`（毫秒）。
6. **首字响应时间（TTFT）**：仅在未开启 `--skip-ttft`（流式模式）且成功测到首 token 时统计 `avg / p50 / p95 / max`（毫秒）；对应样本数为 `ttft_cases`。
7. **异常计数（error_count）**：统计执行异常的用例条数，便于区分「质量问题」与「运行失败」。
8. **内置集与 200 条集混用**：若需对比历史曲线，请固定使用同一数据集与同一参数，避免混用导致指标不可比。

---

## 8. 维护与扩展建议

- **知识库更新**：知识库变更后，可重新解析生成 KB 条目（或增量追加 JSON），并人工抽查 `ground_truth` 与事实一致性。
- **订单场景**：若工具列表或话术变更，请调整 `order` 用例的预期工具及 `_TOOL_SIGNALS`。
- **路由变更**：若 `VALID_INTENTS` 或反问逻辑调整，需同步修订本 JSON 中的 `expected_route` 及本说明文档。

---

## 9. 版本记录

| 日期 | 说明 |
|------|------|
| 2026-03-18 | 首版：200 条测试集与说明文档，与当前 Graph 意图对齐 |
