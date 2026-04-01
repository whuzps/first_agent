# 客服智能体系统设计文档

> 框架：LangGraph  
> 更新时间：2026-03-26

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [核心模块说明](#3-核心模块说明)
   - 3.1 [API 接入层](#31-api-接入层fastapi)
   - 3.2 [LangGraph 对话图](#32-langgraph-对话图)
   - 3.3 [意图识别引擎](#33-意图识别引擎)
   - 3.4 [RAG 知识库检索管线](#34-rag-知识库检索管线)
   - 3.5 [订单处理节点](#35-订单处理节点)
     - 3.5.1 [ReAct 推理模式](#351-react-推理模式)
   - 3.6 [记忆系统](#36-记忆系统)
   - 3.7 [语义缓存](#37-语义缓存)
   - 3.8 [消息队列与异步处理](#38-消息队列与异步处理)
   - 3.9 [熔断器与容错机制](#39-熔断器与容错机制)
   - 3.10 [可观测性](#310-可观测性)
4. [状态机与流程图](#4-状态机与流程图)
5. [数据模型](#5-数据模型)
6. [关键数据流](#6-关键数据流)
   - 6.1 [同步对话流程](#61-同步对话流程)
   - 6.2 [流式对话流程（SSE）](#62-流式对话流程sse)
   - 6.3 [知识库 RAG 检索流程](#63-知识库-rag-检索流程)
   - 6.4 [长期记忆提取与更新流程](#64-长期记忆提取与更新流程)
7. [提示词体系](#7-提示词体系)
8. [多租户设计](#8-多租户设计)
9. [安全与限流](#9-安全与限流)
10. [基础设施与部署](#10-基础设施与部署)
11. [质量评估体系](#11-质量评估体系)
12. [开发文档](#12-当前工作--todo)

---

## 1. 系统概述

本系统是一套基于大语言模型（LLM）的电商智能客服 Agent。系统采用 **LangGraph** 构建状态机式对话流，结合检索增强生成（RAG）、多阶段意图识别、短期/长期记忆管理、订单工具调用等能力，支持以下核心业务场景：

| 场景 | 说明 |
|------|------|
| **知识库问答（kb）** | 售前咨询、售后政策、账户操作、运费、物流、积分等需查阅知识库的业务问题 |
| **订单操作（order）** | 查询订单详情/状态、取消订单、申请退款、修改收货地址、查物流 |
| **闲聊（chitchat）** | 问候、感谢、情绪宣泄等非业务场景 |
| **转人工（human）** | 用户明确要求人工或情绪极度激动时转接人工客服 |

**核心技术栈：**

- **对话图框架**：LangGraph（StateGraph）
- **大语言模型**：阿里云（DashScope），qwen系列模型
- **向量数据库**：Milvus 2.5（混合检索：稠密向量 + 服务端 BM25 稀疏向量）
- **重排模型**：gte-rerank-v2、qwen3-rerank（DashScope）
- **关系数据库**：PostgreSQL（对话 Checkpoint、订单、用户、会话注册、长期记忆）
- **缓存**：Redis Stack（短期会话历史 + 语义缓存 SemanticCache）
- **消息队列**：RabbitMQ（Broker）+ Celery Worker（异步流式处理）
- **链路追踪**：LangSmith/trace_id
- **监控**：Prometheus + Grafana

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          客户端 (Web / App)                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ HTTP / SSE
┌─────────────────────────────────▼───────────────────────────────────┐
│                      FastAPI 应用层 (app/main.py)                    │
│  JWT 认证 │ 限流（Token Bucket） │ 链路追踪中间件 │ CORS │ MCP SSE    │
│   /chat (同步) │ /chat/stream (SSE) │ 知识库管理 │ 会话管理 │ 用户认证  │
└──────────┬────────────────────────────────────┬──────────────────────┘
           │ 直连                                │ MQ 投递
┌──────────▼──────────┐              ┌───────────▼──────────────┐
│  LangGraph 对话图    │              │  Celery Worker           │
│  (graph/graph.py)   │              │  (tasks/chat_task.py)    │
│                     │              │  RabbitMQ chat_queue     │
│  load_memory        │              └──────────┬───────────────┘
│  → intent           │                         │ 写入 Redis Stream
│  → [clarify /       │                         │
│     query_rewrite   │◄────────────────────────┘
│     → kb            │  SSE 消费 Redis Stream
│     order_id_check  │
│     → order         │
│     handoff         │
│     direct          │
│     fallback]       │
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────────────────┐
    │              外部服务依赖                     │
    │  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
    │  │ Milvus  │ │PostgreSQL│ │ Redis Stack  │  │
    │  │(向量库) │ │(订单/记忆│ │(会话/语义缓存│  │
    │  │         │ │/Checkpoint│ │/Stream)      │  │
    │  └─────────┘ └──────────┘ └──────────────┘  │
    │  ┌─────────────────────────────────────────┐  │
    │  │  DashScope API（LLM + Embedding + Rerank│  │
    │  └─────────────────────────────────────────┘  │
    └─────────────────────────────────────────────┘
```

---

## 3. 核心模块说明

### 3.1 API 接入层（FastAPI）

**文件**：`app/main.py`

API 层承担请求接入、认证鉴权、限流、链路追踪等职责，对外暴露如下核心接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 同步对话，返回完整 JSON 响应 |
| `/chat/stream` | POST | 流式对话（SSE），逐 token 推送 |
| `/chat/task/{task_id}` | GET | 查询聊天任务状态（轮询）|
| `/health` | GET | 健康检查（含熔断器状态、延迟指标） |
| `/metrics` | GET | Prometheus 指标拉取 |
| `/greet` | GET | 欢迎词与场景引导选项 |
| `/suggest/{thread_id}` | GET | 建议追问问题（SSE 推流）|
| `/sessions` | GET | 列出用户会话列表 |
| `/sessions/{thread_id}/messages` | GET | 获取会话历史消息 |
| `/sessions/{thread_id}` | DELETE/PATCH | 删除/重命名会话 |
| `/auth/signup` | POST | 普通用户公开注册 |
| `/auth/login` | POST | 登录获取 JWT Token |
| `/auth/register` | POST | 管理员创建用户 |
| `/auth/me` | GET | 获取当前用户信息 |
| `/api/v1/kb/upload` | POST | 批量上传知识库文档 |
| `/api/v1/kb/tasks` | GET | 查询知识库解析任务列表 |
| `/api/v1/milvus/collections` | GET | 列出 Milvus 集合 |
| `/models/list` | GET | 查看支持的 LLM 模型 |
| `/models/switch` | POST | 热切换 LLM 模型 |
| `/api/ragas/evaluate/*` | POST/GET | RAGAS 评估接口 |

**关键设计：**

- **JWT 认证**：HS256 算法，7 天有效期，权限分 `user` / `admin` 两级
- **限流**：令牌桶算法（Token Bucket），按用户/IP 双维度限流（60次/分钟、1000次/小时），通过装饰器链实现无侵入
- **链路追踪中间件**：每个请求生成唯一 `trace_id`（`X-Request-Id` 优先，否则 UUID），注入 `ContextVar` 贯穿全链路日志
- **CORS**：支持本地开发前端多端口（5173-5178）
- **MCP SSE 挂载**：挂载本地 MCP Server，支持 `/mcp` 路径 SSE 协议
- **Gradio UI**：挂载 Gradio 调试界面
- **图模板预热**：应用启动（lifespan）阶段预编译普通图与流式图模板，避免首次请求编译延迟

**流式消息投递三级降级策略：**

1. **MQ 路径（优先）**：任务投递到 Celery → RabbitMQ，Worker 执行后将 token 写入 Redis Stream，SSE 从 Stream 实时读取；Worker 心跳超时（5s）则自动触发降级
2. **FastAPI 后台 Task 降级**：MQ 不可用时，在 FastAPI 进程内启动 asyncio Task 执行，结果同样写入 Redis Stream
3. **内联兜底**：Redis 不可用时完全内联执行，直接从内存队列（asyncio.Queue）推送 token

---

### 3.2 LangGraph 对话图

**文件**：`graph/graph.py`

系统使用 LangGraph 的 `StateGraph` 构建有向状态机，节点之间通过条件路由（`add_conditional_edges`）连接。每次对话轮次共享同一 `State` 对象，由 PostgreSQL Checkpointer 持久化跨轮上下文。

**节点清单：**

| 节点名称 | 功能 | 模型/资源 |
|----------|------|-----------|
| `load_memory` | 加载短期记忆（最近 N 轮原文 + 历史摘要）+ 长期记忆（语义检索用户画像/偏好） | Redis / Milvus |
| `intent` | 多阶段意图识别（规则快速路径 + LLM 联合识别 + 意图转移检测 + 槽位提取） | 小模型（qwen-turbo） |
| `clarify` | 统一反问节点：模糊意图反问 + 槽位缺失追问 | 小模型 |
| `query_rewrite` | Query 改写：补全省略/指代/简短输入为完整清晰的标准 query | 小模型 |
| `order_id_check` | 订单号校验：格式校验（正则）+ 有效性校验（查 DB） | PostgreSQL |
| `order_id_ask` | 反问用户补充/修正订单号 | 规则 |
| `kb` | 知识库节点：语义缓存优先 → RAG 检索 → LLM 生成 | 主模型 + Milvus |
| `kb_node_stream` | 知识库节点（流式版本）：语义缓存命中推全文；否则流式 LLM | 主模型 + Milvus |
| `order` | 订单节点（ReAct）：思考→行动→观察推理循环，LLM 多步选工具 → SafeToolExecutor 安全执行 | 主模型 + PostgreSQL |
| `handoff` | 人工客服节点：记录未命中问题，返回转人工渠道信息 | PostgreSQL |
| `direct` | 直答节点：闲聊/问候场景，结合长期记忆生成个性化回复 | 主模型 |
| `direct_node_stream` | 直答节点（流式版本）| 主模型 |
| `fallback` | 兜底节点：所有节点失败后的友好降级回复 | 规则 |

**图编译策略：**

- 预编译两份全局模板（普通图 + 流式图），懒加载 + 线程安全单例（`threading.Lock`）
- 每次请求通过 `copy(update={"checkpointer": checkpointer})` 动态绑定本次 Postgres Checkpointer，无需重复编译
- 整图执行设有全局超时（`GRAPH_TIMEOUT_SECONDS`，默认 300s），防止 LLM 调用卡死

---

### 3.3 意图识别引擎

**文件**：`graph/graph.py`（`intent_node`）

意图识别采用「规则快速路径 + LLM 联合识别」双层流水线，支持意图转移检测与置信度校准。

**五阶段流水线：**

```
Stage 1: 保存上轮意图（prev_intent），初始化 intent_switched=False
Stage 2: 规则快速路径（关键词匹配 + 正则）
         ├─ 命中 → 直接确定意图 + 规则提取槽位（无 LLM 开销）
         └─ 未命中 → 进入 Stage 3
Stage 3: LLM 联合识别（UNIFIED_INTENT_PROMPT，一次调用完成）
         └─ 同时输出：intent_switched / intent / confidence / slots / ambiguity_reason
Stage 4: 置信度校准（意图转移时降权 × 0.95，增加不确定性）
Stage 5: 必需槽位校验（order 意图需要 order_id + action）
         └─ 缺失 → 记录到 missing_slots，后续路由到 clarify 节点
```

**意图分类及路由逻辑：**

| 意图 | 触发条件 | 路由目标 |
|------|----------|----------|
| `kb` | 业务知识查询（无具体订单号）| `query_rewrite → kb` |
| `order` | 含订单号（ORD\d{8,16}）或明确操作动词 | `order_id_check → order` |
| `chitchat` | 问候/闲聊/感谢 | `direct` |
| `human` | 明确要求转人工/情绪激动 | `handoff` |
| 模糊（置信度 < 0.55） | 任意意图但置信度不足 | `clarify` |
| 槽位缺失 | order 意图但缺 order_id | `clarify` |

**多轮意图继承策略：**

- 未发生意图转移时（`intent_switched=False`）：继承上轮意图，增量合并槽位
- 意图转移时（`intent_switched=True`）：清空旧槽位，重新识别
- 识别来源标记（`recognition_source`）：`keyword`（规则命中）/ `llm`（模型判断）/ `context_inherit`（上下文继承）

---

### 3.4 RAG 知识库检索管线

**文件**：`rag/retrieval.py`、`rag/ingestion.py`、`rag/milvus_store.py`

RAG 管线采用「混合召回 → Rerank 重排 → Parent Expansion 扩展」四段式架构。

**检索流程：**

```
用户 Query
    ↓ 向量化（DashScope text-embedding-v4，1024 维）
    ↓ 混合检索（Milvus 2.5 Hybrid Search）
    │   ├─ 稠密向量（COSINE）：召回 k×10 候选
    │   └─ 稀疏向量（服务端 BM25）：召回 k×10 候选
    ↓ RRF 融合排序（Reciprocal Rank Fusion）
    ↓ Rerank 重排（gte-rerank-v2 DashScope，过滤分数 < 0.1 的文档）
    ↓ Parent Expansion（Small-to-Big：将子块替换为 metadata.parent_context）
    ↓ 返回 top-k 文档（默认 k=3）+ sources 元数据
```

**Parent Expansion（小到大检索）：**  
入库时将文档切分为子块（child chunk），同时在 metadata 中存储父块内容（parent_context）。检索时以子块粒度做向量匹配（精准），展示时替换为更完整的父块内容（上下文丰富），兼顾精度和召回质量。

**知识库文档入库：**

- 支持格式：`.txt`、`.md`、`.pdf`、`.docx`
- 上传 API 接收文件 → 独立进程池（`ProcessPoolExecutor`）异步入库，不阻塞主进程
- 入库状态（pending / processing / done / failed）写入 PostgreSQL `kb_tasks` 表
- 多租户隔离：每个租户对应独立的 Milvus Collection（`{tenant_id}_{COLLECTION_NAME}`）

---

### 3.5 订单处理节点

**文件**：`graph/graph.py`（`order_node`）、`tools/service_tools.py`

订单节点采用「LLM 工具选择 + SafeToolExecutor 安全执行」模式，支持以下五种订单操作：

| 工具名称 | 操作 | 风险级别 |
|----------|------|----------|
| `get_order_detail` | 查询订单详情/状态 | 只读 |
| `get_logistics_info` | 查询物流信息 | 只读 |
| `cancel_order` | 取消订单 | 写操作 ⚠️ |
| `modify_order_address` | 修改收货地址 | 写操作 ⚠️ |
| `apply_refund` | 申请退款 | 高危 💰 |

**SafeToolExecutor 执行策略：**

1. **工具存在性验证**：检查工具名称是否注册
2. **参数 Schema 校验**：通过 Pydantic `args_schema` 在执行前校验入参
3. **强制注入关键字段**：防止 LLM 幻觉遗漏 `order_id` / `tenant_id`
4. **三级降级策略**：备用工具 → LLM 模拟回答 → 兜底错误文案

**订单号校验：**

- 格式校验：正则 `ORD\d{8,16}`
- 有效性校验：查询 PostgreSQL `orders` 表确认存在
- 最多反问 `MAX_HANDOFF_RETRY`（默认 3）次，超出后转人工

**跨轮意图透传（action_hint）：**  
首轮已确认操作意图（如「取消订单」）存入 `slots.action`，第二轮用户仅补充订单号时，系统将上轮意图以 `action_hint` 形式注入 Prompt，避免信息断链导致工具选择错误。

#### 3.5.1 ReAct 推理模式

**文件**：`graph/graph.py`（`order_node` + `_react_order_loop`）

订单节点支持 **ReAct（Reasoning + Acting）** 推理模式，通过「思考 → 行动 → 观察」循环实现多步推理和自我纠错。可通过环境变量 `REACT_ENABLED`（默认 `true`）开关，`REACT_MAX_ITERATIONS`（默认 `3`）控制最大迭代次数。

**ReAct 推理循环流程：**

```
用户提问 + 订单上下文
    ↓
┌─────────────────────────────────────┐
│  Step N（最多 REACT_MAX_ITERATIONS） │
│                                     │
│  ① Thought（思考）                  │
│     LLM 分析当前情况，输出推理过程    │
│     ↓                               │
│  ② Action（行动）                   │
│     LLM 通过 function call 调用工具  │
│     强制注入 order_id / tenant_id    │
│     ↓                               │
│  ③ Observation（观察）              │
│     工具返回结果反馈给 LLM           │
│     LLM 根据结果决定：               │
│     ├─ 继续下一轮推理 → 回到 ①      │
│     └─ 已得出结论 → 输出最终回答     │
└─────────────────────────────────────┘
    ↓
最终回答 + reasoning_trace（完整推理链路）
```

**核心设计要点：**

| 要点 | 说明 |
|------|------|
| **消息构建** | SystemMessage（ReAct Prompt）+ HumanMessage（用户问题），工具结果以 ToolMessage 形式追加，构成完整推理链 |
| **意图提示注入** | 将槽位中的 `action`（如 `cancel` → `取消订单`）转为中文 `action_hint` 注入 Prompt，辅助 LLM 选择正确工具 |
| **强制参数注入** | 每次工具调用前自动补全 `order_id` 和 `tenant_id`，防止 LLM 幻觉遗漏关键参数 |
| **终止条件** | ① LLM 未发起工具调用（已得出结论）② 达到最大迭代次数（强制收尾） |
| **强制收尾** | 达到最大步数时，解绑工具约束，注入「请直接给出最终回答」指令，强制 LLM 输出纯文本回答 |
| **熔断器保护** | LLM 调用和工具执行分别受 `llm_main` / `order_tools` 熔断器保护 |
| **流式支持** | 每个推理步骤（thought/action/observation）实时推送至 `stream_queue`，前端可展示推理过程 |

**reasoning_trace 推理链路结构：**

```json
[
  {"step": 1, "type": "thought",      "content": "用户想查询订单详情，我来调用工具获取信息。"},
  {"step": 1, "type": "action",       "content": "调用 get_order_detail({\"order_id\": \"ORD20250101001\"})"},
  {"step": 1, "type": "observation",  "content": "订单状态：已发货，物流单号：SF1234567890..."},
  {"step": 2, "type": "thought",      "content": "已获取订单信息，可以直接回复用户。"}
]
```

**ReAct vs 单次调用模式对比：**

| 维度 | ReAct 模式（默认） | 单次调用模式（降级） |
|------|-------------------|---------------------|
| 推理方式 | 多步循环，支持自我纠错 | 单次 LLM 调用 |
| 工具调用 | 可连续多次调用不同工具 | 单次选工具 + 执行 |
| 适用场景 | 复杂操作（先查详情再决策） | 简单查询 |
| 开启方式 | `REACT_ENABLED=true` | `REACT_ENABLED=false` |
| 推理链路 | 完整 `reasoning_trace` 可追溯 | 无推理链路记录 |
| 兜底策略 | 达到最大步数强制总结 | 无工具调用时查询订单详情 |

**降级兜底（双模式共享）：**

- 熔断器触发（`CircuitBreakerError`）→ 返回「订单服务暂时不可用」提示
- 异常兜底 → 自动调用 `get_order_detail` 查询订单详情，尽量给出有用信息
- 所有结果写入 `state.order_summary`，ReAct 模式额外记录 `state.reasoning_trace`

---

### 3.6 记忆系统

系统实现了两级记忆架构，短期记忆保障多轮对话连贯性，长期记忆实现跨会话用户感知。

#### 3.6.1 短期记忆

**文件**：`core/config.py`（会话管理函数）

| 存储 | 内容 | TTL |
|------|------|-----|
| Redis（优先） | 完整消息历史（最多 `SESSION_MAX_MESSAGES=200` 条） | 24 小时 |
| 进程内存（降级） | 消息历史（maxlen=200） | 进程生命周期 |

**增量摘要机制：**

- 最近 N 轮（`SESSION_WINDOW_TURNS=5`）原文直接注入 Prompt 上下文
- 窗口之外的历史以「增量摘要」形式呈现，每新增 `SUMMARY_TRIGGER_EVERY=5` 轮触发一次异步后台摘要更新
- 摘要存储于 Redis（key = `summary:{thread_id}:{user_id}`），双写本地内存兜底
- 增量合并：有存量摘要时用 `INCREMENTAL_SUMMARY_PROMPT` 将新消息合并进去，避免全量重算

**对话 Checkpoint：**

LangGraph 使用 `AsyncPostgresSaver` 将每轮节点状态序列化存入 PostgreSQL，实现跨轮状态恢复（`intent`、`slots`、`order_id` 等字段由 Checkpointer 自动恢复，每轮仅传入变更字段）。

#### 3.6.2 长期记忆

**文件**：`memory/store.py`

**存储架构：**

- **PostgreSQL**（主存储）：`memories` 表存储用户画像、偏好、关键事实、重要事件
- **Milvus**（可选向量索引）：不可用时自动降级为关键词匹配

**记忆分类：**

| 类型 | 说明 | 示例 |
|------|------|------|
| `user_profile` | 用户画像 | 姓名/城市/职业/年龄段 |
| `preference` | 用户偏好 | 沟通方式偏好/价格敏感度/品牌喜好 |
| `fact` | 关键事实 | 已购产品/订单历史/特殊需求 |
| `important_event` | 重要事件 | 投诉/纠纷/情绪激烈 |

**对话后异步更新流程（`extract_and_save_memory`）：**

```
Step 1: LLM 提取记忆（EXTRACT_MEMORY_PROMPT → ExtractedMemoryList 结构化输出）
Step 2: 对每条新记忆，LLM 决策操作类型（ADD / UPDATE / DELETE / NONE），避免冗余和冲突
Step 3: 对话计数 +1 + 时间衰减清理（轻量，每轮执行）
Step 4: 每 LLM_REVIEW_EVERY(=20) 次触发 LLM 智能审查（删除过时/空洞/冲突记忆）
```

**遗忘策略（指数衰减 + 访问加成）：**

```
有效重要度 = 基础重要度 × exp(-days / 半衰期×1.44) + min(访问次数×0.3, 2.0)
有效重要度 < FORGET_THRESHOLD(=1.5) → 软删除
```

---

### 3.7 语义缓存

**文件**：`core/semantic_cache.py`

基于 `redisvl.SemanticCache` + Redis Stack（含 RediSearch 向量搜索模块）实现 FAQ 场景的语义级缓存，大幅降低 RAG 检索和 LLM 推理的重复开销。

**核心参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 余弦距离阈值 | 0.1（相似度 ≥ 0.9） | 超过阈值的相似 query 直接命中缓存 |
| TTL | 3600s | 缓存条目有效期 |
| 向量化器 | DashScope text-embedding-v4 | 包装为 redisvl BaseVectorizer |

**缓存流程：**

```
lookup(query, tenant_id)
    ↓ 将 query 向量化（Embedding API）
    ↓ Redis 服务端 KNN 向量检索
    ├─ 命中（distance <= 0.1）→ 直接返回缓存答案，跳过 RAG + LLM（节省 ~1-2s）
    └─ 未命中 → 执行完整 RAG 流程
              → RAG 完成后异步写入缓存（不阻塞流式返回）

save(query, answer, tenant_id)
    ↓ redisvl 自动管理向量索引 + TTL
```

**多租户隔离**：每个 `tenant_id` 对应独立的 Redis 索引（`kb_semantic_cache_{tenant_id}`）。

**降级机制**：Redis 不支持 RediSearch 模块时，`_unavailable=True` 静默跳过所有缓存操作，主流程不受影响。

---

### 3.8 消息队列与异步处理

**文件**：`celery_app.py`、`tasks/chat_task.py`、`core/mq.py`

**架构设计：**

```
FastAPI /chat/stream
    │ 投递任务（task_id = trace_id）
    ▼
RabbitMQ chat_queue ──────────────────────────► Celery Worker
    │  (x-dead-letter-exchange: chat_dlx          │ task_acks_late=True
    │   x-max-length: 1000                         │ prefetch=1
    │   x-message-ttl: 300000ms)                   │ soft_timeout=120s
    │                                              ↓
chat_dlq（DLQ）                           run_graph_stream()
（超时/拒绝消息落入死信队列）                   │ 逐 token 写入
                                           ▼
                                    Redis Stream
                                    key = chat:stream:{task_id}
                                           │
                                    FastAPI SSE 消费
                                    (xread, block=5s)
```

**可靠性配置：**

- `task_acks_late=True`：任务执行完毕后再 ACK，防止 Worker 崩溃丢消息
- `task_reject_on_worker_lost=True`：Worker 异常断开自动重新入队
- `worker_prefetch_multiplier=1`：防止消息大量堆积在 Worker 内存
- 死信队列（DLQ）：`chat_dlq` 接收超时/被拒绝任务，供告警和人工复核

**对话后处理（异步批处理，不阻塞主流程）：**

每轮对话结束后，`_post_process_tasks` 批量启动以下异步 Task：

1. 写入用户消息到 Redis 会话历史
2. 写入助手消息到 Redis 会话历史
3. 更新 PostgreSQL 会话注册表（标题首次写入）
4. 触发增量摘要更新检查
5. 提取并保存长期记忆
6. 生成建议追问问题（推入 `SUGGEST_QUEUES[thread_id]`）

---

### 3.9 熔断器与容错机制

**文件**：`core/circuit_breaker.py`、`core/retry_policy.py`、`core/robust.py`

#### 熔断器（Circuit Breaker）

标准三态熔断器，独立保护每个外部资源：

| 熔断器 | 保护目标 | 失败阈值 | 恢复超时 | 成功阈值 |
|--------|----------|----------|----------|----------|
| `llm_main` | 主 LLM（kb/direct/order） | 5 次 | 60s | 2 次 |
| `llm_small` | 小 LLM（intent/rewrite/clarify） | 5 次 | 60s | 2 次 |
| `kb_retrieval` | Milvus 知识库检索 | 3 次 | 30s | 2 次 |
| `order_tools` | 订单工具调用 | 5 次 | 30s | 2 次 |

**三态转换：**

```
CLOSED（正常）
    │ 连续失败 >= failure_threshold
    ▼
OPEN（断路）────── 快速失败（CircuitBreakerError），不执行 func
    │ recovery_timeout 到期
    ▼
HALF_OPEN（半开）─── 有限放行探测
    │ 连续成功 >= success_threshold → CLOSED
    └ 失败 → OPEN
```

#### 智能重试策略（SmartRetryPolicy）

- LLM 调用：速率限制（RateLimitError）、服务不可用（503）时重试；上下文超长（ContextLengthError）不重试
- API 调用：连接超时、临时服务异常时重试；参数错误（400）不重试
- 最大重试次数：`MAX_ATTEMPTS=3`

#### 全局容灾恢复（Robust）

`Robust` 类在图执行层面提供最多 3 次的恢复重试，确保节点级重试失败后整图可重新执行。

---

### 3.10 可观测性

**文件**：`core/observability.py`

**Prometheus 指标：**

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `chat_stream_requests_total` | Counter | 流式对话请求总数（按 path/tenant_id 分） |
| `chat_stream_active_requests` | Gauge | 当前活跃流式请求数 |
| `chat_stream_messages_total` | Counter | 消息消费成功/失败统计 |
| `chat_stream_consume_duration` | Histogram | 端到端流式消费耗时 |
| `ttft_duration` | Histogram | 首 token 响应时间（TTFT） |
| 节点执行耗时 | Summary | 各 LangGraph 节点耗时统计 |
| 知识库缓存命中率 | Counter | 语义缓存 hit/miss 统计 |

**链路追踪：**

- 启用 LangSmith Tracing（`LANGCHAIN_TRACING_V2=true`），每次图执行全链路可观察
- 日志格式携带 `trace_id`：`%(asctime)s | %(levelname)s | %(name)s | trace=%(trace_id)s | %(message)s`

**熔断器状态暴露：**

`/health` 接口实时返回所有熔断器快照（state / failure_count / last_failure_time）。

---

## 4. 状态机与流程图

### LangGraph 节点流转图

```
START
  │
  ▼
load_memory（加载短期 + 长期记忆）
  │
  ▼
intent（多阶段意图识别）
  │
  ├─ status=FAILED ──────────────────────────────────► fallback → END
  │
  ├─ intent_confidence < 0.55（模糊）──────────────────► clarify → END
  │
  ├─ intent=order
  │     ├─ missing_slots 只缺 action & 有 order_id → 自动填 query_detail
  │     ├─ missing_slots 缺 order_id ───────────────────► clarify → END
  │     └─ slots 完整 ──────────────────────────────────► order_id_check
  │                                                          │
  │                                              ┌───────────┼───────────┐
  │                                              │           │           │
  │                                   order_id_valid=True  False  retry>=3
  │                                              │           │           │
  │                                              ▼           ▼           ▼
  │                                           order    order_id_ask  handoff
  │                                              │           │           │
  │                                              ▼           ▼           ▼
  │                                             END         END         END
  │
  ├─ intent=kb ────────────────────────────────► query_rewrite → kb
  │                                                              │
  │                                               ┌─────────────┤
  │                                          has_kb           no_kb
  │                                               │               │
  │                                               ▼               ▼
  │                                              END           handoff → END
  │
  ├─ intent=human ──────────────────────────────► handoff → END
  │
  └─ intent=chitchat / other ──────────────────► direct → END
```

---

## 5. 数据模型

### State（图状态对象）

`core/state.py` 定义了贯穿整个 LangGraph 执行流的状态对象：

```python
@dataclass
class State:
    # 基础字段
    thread_id: str          # 会话唯一标识
    query: str              # 用户本轮输入
    user_id: str            # 用户ID
    tenant_id: str          # 租户ID（多租户隔离）
    history: str            # 短期记忆（最近 N 轮 + 摘要）
    status: StateStatus     # 执行状态（PENDING/RUNNING/SUCCESS/FAILED/RETRYING/FALLBACK）

    # 意图识别
    intent: str             # kb / order / chitchat / human
    intent_confidence: float  # 置信度（0~1）
    intent_switched: bool   # 是否发生意图转移
    prev_intent: str        # 上轮意图
    slots: Dict             # 槽位（order_id / action / address）
    missing_slots: List     # 缺失的必需槽位

    # 节点输出
    kb_answer: str          # FAQ 知识库回答
    sources: List           # 来源文档列表
    order_summary: str      # 订单操作结果
    ask_human: str          # 反问/提示话术
    human_handoff: Dict     # 转人工渠道信息
    clarify_question: str   # 反问话术（模糊意图/槽位缺失）
    fallback: str           # 兜底回复

    # Query 改写
    rewritten_query: str    # 改写后的 query
    query_rewritten: bool   # 是否实际执行了改写

    # 订单相关
    order_id: str           # 提取的订单号
    order_id_valid: bool    # 订单号有效性
    ask_retry_times: int    # 反问重试次数

    # ReAct 推理
    reasoning_trace: List[Dict]  # ReAct 推理链路（step/type/content）

    # 记忆
    long_term_memory: str   # 长期记忆格式化文本
    quoted_message: str     # 用户引用的历史消息
```

### PostgreSQL 主要数据表

| 表名 | 用途 |
|------|------|
| `users` | 用户账号（id / username / password_sha256 / role / tenant_id） |
| `sessions` | 会话注册表（thread_id / user_id / title / message_count / timestamps） |
| `orders` | 订单数据（order_id / status / amount / create_time / update_time） |
| `memories` | 长期记忆（user_id / memory_type / content / importance / access_count）|
| `memory_meta` | 记忆元信息（user_id / conversation_count / last_llm_review_at） |
| `unanswered_questions` | 未命中问题记录（tenant_id / user_id / text / created_at） |
| `kb_tasks` | 知识库解析任务（id / filename / collection_name / status / chunk_count）|
| LangGraph Checkpoint | 图执行状态序列化（由 AsyncPostgresSaver 管理） |

---

## 6. 关键数据流

### 6.1 同步对话流程

```
客户端 POST /chat
    │ 携带 query / thread_id / user_id / tenant_id
    ▼
FastAPI 认证 + 限流 + 生成 trace_id
    ▼
构建 State（仅传入本轮变更字段）
    ▼
run_graph(state)
    │ asyncio.wait_for(timeout=300s)
    ▼
LangGraph 图执行（Postgres Checkpoint 恢复上轮状态）
    ▼
determine_answer(result)  →  route / answer / sources
    ▼
_post_process_tasks（异步后处理）
    ▼
返回 JSON：{"route": ..., "answer": ..., "sources": ..., "trace_id": ...}
```

### 6.2 流式对话流程（SSE）

```
客户端 POST /chat/stream
    │
    ▼
FastAPI 优先投递 MQ（Celery chat_queue）
    │ 同时初始化 Redis task 状态
    │
    ├──── MQ 可用 ─────────────────────────────────────────────────────┐
    │      Celery Worker 接收任务                                       │
    │         ↓ run_graph_stream(state, stream_queue)                  │
    │         ↓ kb_node_stream/direct_node_stream 逐 token 推送        │
    │         ↓ 每个 token 写入 Redis Stream (chat:stream:{task_id})    │
    │      FastAPI SSE 消费 Redis Stream → 推送给客户端                  │
    │         ↓ 收到 type=done 事件结束                                  │
    │                                                                   │
    ├──── MQ 不可用 / Worker 心跳超时（5s）─────────────────────────────┘
    │      FastAPI 进程内 asyncio Task 降级执行
    │         ↓ 同样写入 Redis Stream
    │
    └──── Redis 不可用（兜底）
           FastAPI 完全内联执行
              ↓ 直接从 asyncio.Queue 推送 token 给 SSE 生成器

SSE 事件格式：
  {"type": "token", "content": "..."}   ← 逐 token 输出
  {"type": "done", "route": "...", "answer": "...", "sources": [...]}
  {"type": "error", "message": "..."}
```

### 6.3 知识库 RAG 检索流程

```
kb_node 接收 query
    │
    ├── 语义缓存 lookup（Embedding → Redis KNN）
    │       ├─ 命中（相似度 >= 0.9）→ 直接返回缓存答案，跳过后续
    │       └─ 未命中 ↓
    │
    ├── Milvus 混合检索（稠密 + BM25）→ RRF 融合 → top k×10 候选
    │
    ├── Rerank 重排（gte-rerank-v2）→ 过滤分 < 0.1 的文档 → top k
    │
    ├── Parent Expansion（子块 → 父块扩展）
    │
    ├── 构建 Prompt（RAG_PROMPT_TEMPLATE + 检索上下文）
    │
    ├── LLM 生成回答（主模型）
    │
    └── 异步写入语义缓存（不阻塞返回）
```

### 6.4 长期记忆提取与更新流程

```
每轮对话完成后（_post_process_tasks 异步）
    │
    ▼
extract_and_save_memory(user_id, tenant_id, query, answer)
    │
    ├── Step 1: LLM 提取记忆（EXTRACT_MEMORY_PROMPT）
    │           → 结构化输出 List[ExtractedMemoryItem]
    │
    ├── Step 2: 对每条新记忆，LLM 决策（UPDATE_MEMORY_PROMPT）
    │           → ADD / UPDATE / DELETE / NONE
    │           → 执行对应的 PostgreSQL + Milvus 操作
    │
    ├── Step 3: 对话计数 +1，触发时间衰减清理（轻量）
    │           有效重要度 < 1.5 → 软删除（is_active=0）
    │
    └── Step 4: 每 20 次对话 → LLM 智能审查（MEMORY_REVIEW_PROMPT）
                → 批量清理空洞/过时/冲突记忆
```

---

## 7. 提示词体系

系统在 `core/prompts.py` 中定义了完整的提示词体系：

| 提示词 | 用途 | 输出类型 |
|--------|------|----------|
| `UNIFIED_INTENT_PROMPT` | 意图转移检测 + 意图识别 + 槽位提取 + 置信度（一次调用） | JSON |
| `INTENT_SLOT_PROMPT` | 意图+槽位联合识别（无上下文续接判断） | JSON |
| `CLARIFY_INTENT_PROMPT` | 模糊意图反问话术生成 | 自然语言 |
| `CLARIFY_SLOT_PROMPT` | 槽位缺失追问话术生成 | 自然语言 |
| `QUERY_REWRITE_PROMPT` | 多场景 query 改写（省略/指代/情绪/多轮）| 改写后 query |
| `RAG_PROMPT_TEMPLATE` | 基于参考资料的 FAQ 问答 | 自然语言 |
| `DIRECT_PROMPT_TEMPLATE` | 闲聊/问候直接回复（结合长期记忆） | 自然语言 |
| `REACT_ORDER_PROMPT_TEMPLATE` | ReAct 推理模式订单 Prompt（思考→行动→观察） | 工具调用 + 推理链 |
| `ORDER_AGENT_WITH_SKILLS_PROMPT_TEMPLATE` | 订单操作工具选择（含风险等级提示） | 工具调用 |
| `SUMMARIZATION_PROMPT` | 对话历史压缩摘要 | 摘要文本 |
| `INCREMENTAL_SUMMARY_PROMPT` | 增量合并摘要 | 更新后摘要 |
| `EXTRACT_MEMORY_PROMPT` | 从对话中提取长期记忆 | 结构化 JSON |
| `UPDATE_MEMORY_PROMPT` | 记忆更新决策（ADD/UPDATE/DELETE/NONE） | 结构化 JSON |
| `MEMORY_REVIEW_PROMPT` | 批量审查记忆质量（遗忘策略） | 结构化 JSON |
| `SUGGEST_QUESTIONS_PROMPT_TEMPLATE` | 生成建议追问问题 | 问题列表 |

---

## 8. 多租户设计

系统从数据层到模型层全面支持多租户隔离：

| 资源 | 隔离方式 |
|------|----------|
| Milvus 知识库 | Collection 名称 = `{tenant_id}_{COLLECTION_NAME}` |
| Milvus 记忆向量 | Collection 名称 = `{tenant_id}_{MEMORY_COLLECTION_NAME}` |
| PostgreSQL DSN | 环境变量 `POSTGRES_DSN_{TENANT_ID}` 可覆盖 |
| LangGraph Checkpoint | Checkpointer DSN 按租户独立配置 |
| 语义缓存 | Redis 索引名 = `kb_semantic_cache_{tenant_id}` |
| LLM 模型 | 支持按租户独立配置 `_TENANT_MODELS[tenant_id]` |
| 未命中问题记录 | `unanswered_questions.tenant_id` 字段过滤 |

租户 ID 来源（优先级从高到低）：

1. HTTP Header `X-Tenant-ID`
2. Query 参数 `tenant`
3. JWT Token payload 中的 `tenant_id`
4. 默认值 `default`

---

## 9. 安全与限流

**认证鉴权：**

- JWT Bearer Token（HS256，7 天过期）
- 用户角色：`admin`（可管理用户、知识库）/ `user`（普通使用）
- 密码存储：SHA-256 哈希
- 输入脱敏：`RedactionMiddleware` 自动隐藏 token/password/credit card 等敏感字段

**限流（Token Bucket）：**

| 接口范围 | 限流规则 | 维度 |
|----------|----------|------|
| `/chat`、`/chat/stream` | 60次/分钟 | 用户/IP |
| `/chat`、`/chat/stream` | 1000次/小时 | 租户 |
| `/auth/signup` | 20次/小时 | IP |
| `/api/v1/kb/upload` | 30次/小时 | 用户 |
| `/api/v1/milvus/collections/delete` | 10次/分钟 | 用户 |
| `/sessions` | 120次/分钟 | 用户 |

**输入预处理：**

- `core/preprocessing.py`（`clean_input`）：过滤控制字符、规范化空白
- SQL 注入防护：订单查询全程使用参数化占位符 `%s`，不拼接用户输入

**订单号格式验证：**

正则 `ORD\d{8,16}`，防止越权查询（结合 `tenant_id` 隔离）。

---

## 10. 基础设施与部署

**Docker Compose 服务清单（`docker-compose.yml`）：**

| 服务 | 镜像/说明 | 端口 |
|------|----------|------|
| `app` | FastAPI 应用（uvicorn，4 workers） | 8000 |
| `celery_worker` | Celery Worker（concurrency=4） | - |
| `redis` | Redis Stack（含 RediSearch 向量搜索）| 6379 |
| `postgres` | PostgreSQL 15 | 5432 |
| `milvus` | Milvus 2.5（Standalone 模式） | 19530 |
| `etcd` | Milvus 依赖（元数据存储） | 2379 |
| `minio` | Milvus 依赖（对象存储） | 9000 |
| `rabbitmq` | RabbitMQ 3（含管理界面） | 5672 / 15672 |
| `prometheus` | Prometheus 监控 | 9090 |
| `grafana` | Grafana 可视化 | 3000 |

**关键环境变量：**

| 变量名 | 说明 |
|--------|------|
| `DASHSCOPE_API_KEY` | 阿里云 API Key |
| `MODEL_NAME` | 默认 LLM 模型 |
| `EMBEDDING_MODEL` | 向量模型（text-embedding-v4） |
| `POSTGRES_DSN` | PostgreSQL 连接串 |
| `REDIS_URL` | Redis 连接地址 |
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus 连接信息 |
| `RABBITMQ_URL` | RabbitMQ Broker URL |
| `LANGCHAIN_TRACING_V2` | 启用 LangSmith 链路追踪 |
| `LANGCHAIN_API_KEY` | LangSmith API Key |
| `GRAPH_TIMEOUT_SECONDS` | 图执行全局超时（默认 300s） |
| `INTENT_CONFIDENCE_THRESHOLD` | 意图置信度阈值（默认 0.55） |
| `SESSION_WINDOW_TURNS` | 短期记忆窗口轮数（默认 5） |
| `SEMANTIC_CACHE_ENABLED` | 是否启用语义缓存（默认 true） |
| `REACT_ENABLED` | 订单节点 ReAct 推理模式开关（默认 true） |
| `REACT_MAX_ITERATIONS` | ReAct 最大推理迭代次数（默认 3） |

---

## 11. 质量评估体系

系统提供两套独立的评估框架，分别针对不同的评估维度和使用场景。

---

### 11.1 Agent 自动化评估（eval_agent.py）

**文件**：`scripts/eval_agent.py`

这是一套面向整个 Agent 全链路的端到端自动化评估框架，直接调用 LangGraph 图执行（不走 HTTP），支持并发执行和多维度指标统计。

#### 评估指标

| 序号 | 指标名称 | 说明 | 评分方式 |
|------|----------|------|----------|
| 1 | **回答准确率（Accuracy）** | LLM-as-Judge 对比标准答案，0-100 分，≥70 分视为通过 | qwen-turbo 评判，超时降级为关键词重叠率 |
| 2 | **幻觉率（Hallucination）** | KB 回答中编造来源文档未提及内容的比例 | LLM-as-Judge 对比 sources，输出 hallucinated: true/false |
| 3 | **工具调用正确率（Tool Accuracy）** | 订单操作是否调用预期工具 | 从回答内容推断实际调用的工具（关键词信号匹配） |
| 4 | **意图路由准确率（Route Accuracy）** | 是否路由到预期意图节点 | actual_route == expected_route 精确匹配 |
| 5 | **响应时间（Latency）** | avg / p50 / p95 / max（ms） | perf_counter 计时 |
| 6 | **首字响应时间（TTFT）** | 流式接口首个 token 到达时间（ms） | 通过 asyncio.Queue 监听流式节点第一个 token 事件 |

#### 准确率评分等级

| 分数区间 | 等级 |
|----------|------|
| 90 ~ 100 | 优秀（核心信息完全正确，无误导） |
| 70 ~ 89 | 良好（基本正确，少量细节遗漏） |
| 40 ~ 69 | 一般（部分正确，缺失重要信息） |
| 10 ~ 39 | 差（大部分错误，极少信息匹配） |
| 0 ~ 9 | 完全错误/无关/拒绝回答 |

#### 内置测试集（BUILT_IN_CASES）

测试集覆盖所有业务场景，共 18 条用例：

| 场景类型 | 用例数 | 典型用例 |
|----------|--------|----------|
| `kb`（知识库 FAQ） | 7 | 运费计算、密码找回、发货时间、退货政策、收到破损商品等 |
| `order`（订单操作） | 6 | 查询详情、取消订单、申请退款、查物流、修改地址 |
| `direct`（闲聊直答） | 3 | 问候、感谢、闲聊 |
| `human`（转人工） | 1 | 要求人工客服 |
| `edge`（边界/鲁棒性） | 2 | 空查询兜底、无订单号触发反问 |

支持通过 `--dataset` 参数加载自定义 JSON 测试集，字段格式与内置用例保持一致：`name / query / case_type / expected_route / expected_tool / ground_truth`。

#### 工具推断信号表

订单工具调用正确率通过答案内容关键词反推（按特异性优先级从高到低匹配）：

| 工具 | 触发关键词示例 |
|------|--------------|
| `cancel_order` | 已为您取消、取消成功、订单已取消 |
| `apply_refund` | 退款申请、已为您申请退款、退款金额 |
| `modify_order_address` | 地址已修改、收货地址已更新、新地址 |
| `get_logistics_info` | 物流、快递、运输、派送、已签收 |
| `get_order_detail` | 订单状态、订单详情、待付款、已发货、创建时间 |

#### 运行方式

```bash
# 内置测试集，默认并发 2，同时测量 TTFT
python eval_agent.py

# 自定义测试集
python eval_agent.py --dataset my_cases.json

# 调整并发（建议不超过 4，避免 API 限速）
python eval_agent.py --concurrency 3

# 跳过幻觉检测（节省约 600ms/条）
python eval_agent.py --skip-hallucination

# 跳过 LLM 准确率评分（仅统计路由/工具/延迟指标）
python eval_agent.py --skip-accuracy

# 跳过 TTFT 测量，使用非流式接口（速度更快）
python eval_agent.py --skip-ttft

# 指定租户
python eval_agent.py --tenant default

# 自定义报告文件名前缀
python eval_agent.py --output my_report
```

#### 报告输出

评估完成后输出：
- **控制台格式化表格**：每条用例的路由、准确率、幻觉、工具调用、耗时、TTFT 一览
- **JSON 报告文件**：保存至 `logs/eval_reports/{output_prefix}_{timestamp}.json`，包含完整的 `summary` 和逐条 `cases` 详情

`summary` 字段包含：`avg_accuracy` / `accuracy_pass_rate` / `hallucination_rate` / `tool_accuracy` / `route_accuracy` / `avg_latency_ms` / `p50/p95/max_latency_ms` / `avg_ttft_ms` / `p50/p95/max_ttft_ms`。

#### 降级策略

| 场景 | 降级方式 |
|------|----------|
| LLM 评分超时（>5s）| 降级为关键词重叠率计算 |
| 幻觉检测 LLM 调用失败 | 返回 `None`（标记为未检测），不影响其他指标 |
| 幻觉检测 JSON 解析失败 | 关键词判断（文本含 `true` → 有幻觉） |

---

### 11.2 RAG 知识库评估（RAGAS）

**文件**：`scripts/ragas_evaluator.py`

基于 RAGAS 框架，专注于知识库检索与回答质量的精细化评估。

#### 评估指标

| 指标 | 说明 | 默认通过阈值 |
|------|------|------------|
| `faithfulness`（忠实度） | 答案是否忠实于检索上下文，不产生幻觉 | ≥ 0.85 |
| `answer_relevancy`（相关性） | 答案与用户问题的相关程度 | ≥ 0.70 |
| `context_precision`（上下文精度） | 检索上下文与 ground_truth 的精准匹配度 | ≥ 0.70 |
| `context_recall`（上下文召回） | 检索上下文对 ground_truth 的覆盖率 | ≥ 0.70 |

#### 评估接口

- `POST /api/ragas/evaluate/single`：评估单个用例（同步）
- `POST /api/ragas/evaluate/batch`：批量评估（后台异步）
- `GET /api/ragas/test-cases`：使用内置测试用例快速跑一次评估并返回结果
- `GET /api/ragas/reports`：列出最近 20 份历史评估报告

---

### 11.3 两套评估框架对比

| 维度 | Agent 自动化评估（eval_agent.py） | RAGAS 评估（ragas_evaluator.py） |
|------|----------------------------------|----------------------------------|
| 评估对象 | 整个 Agent 全链路（意图→工具→回答） | 仅 RAG 知识库检索+生成质量 |
| 执行方式 | 命令行脚本，直接调 LangGraph 图 | HTTP API，后台异步任务 |
| 测试场景 | kb / order / direct / human / edge | 仅 kb（FAQ 场景） |
| 幻觉检测 | LLM-as-Judge（对比 sources） | RAGAS faithfulness 指标 |
| 工具正确率 | ✓ 支持（关键词信号推断） | ✗ 不支持 |
| 意图路由准确率 | ✓ 支持 | ✗ 不支持 |
| TTFT 测量 | ✓ 支持（流式接口）| ✗ 不支持 |
| 报告格式 | 控制台表格 + JSON 文件 | JSON 文件 |

---

### 11.4 性能压测

- **Locust 压测**：`scripts/locust_chat.py`，模拟并发用户持续发起对话请求，统计吞吐量与延迟分布
- **Embedding 基准测试**：`tests/benchmark_embedding.py`，量化向量化模型在不同 batch size 下的吞吐率和单次延迟

---

### 11.5 日志审计

- 所有写操作（知识库删除、文件入库、模型切换）写入结构化审计日志（`logs/requests.log`）
- 日志字段包含：时间戳 / trace_id / 操作类型 / 关键参数（敏感字段自动脱敏）
- 日志可部署ELK，高效收集、存储、分析和监控大量的日志数据


## 12. 当前工作 & TODO

[开发文档](/service/docs/开发文档.md)