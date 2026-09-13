# Inference Benchmark System Architecture and Methodology Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一套面向硬件选型、模型与推理栈选型、性能优化和问题定位的企业级大模型推理 Benchmark 方法论，并将其固化为可审阅、可追溯、可扩展的体系文档。

**Architecture:** 采用“3 个交付层次 × 4 个技术域 × 场景化用例 × 共享证据链”。L1 面向管理层/PM/客户输出决策结论，L2 面向应用/部署/测试/QA 输出可执行的服务评测，L3 面向 Infra/推理服务/引擎/Kernel 输出可下钻的工程证据。三层来自同一次 Benchmark 运行的共享结果，不维护互相矛盾的独立口径。

**Tech Stack:** Markdown、Mermaid、表格化指标注册表、场景卡和版本化 YAML/JSON 契约示例；MLPerf Inference 用作标准参考，EvalScope/vLLM/SGLang 等用作工具参考。本计划当前不包含 Runner、服务平台、CI 或 Dashboard 的代码实现。

---

## 1. Scope and Principles

### 1.1 In Scope

- 统一三层交付模型：L1 决策层、L2 服务层、L3 工程层。
- 统一四个技术域：算力硬件、模型架构和算法、推理软件栈、Benchmark 方法论。
- 建立场景卡、负载画像、用例分层、指标注册表、证据链和归因方法。
- 建立硬件选型、模型选型、推理栈优化和 Profiling 的方法步骤。
- 定义三类标准报告：L1 Decision Card、L2 Service Evaluation Report、L3 Engineering Profile。

### 1.2 Out of Scope

- 本阶段不实现 Benchmark Runner、部署编排、采集器、CI/CD、Prometheus 或 Grafana。
- 本阶段不承诺一个适用于所有模型、硬件和业务的统一分数。
- 本阶段不把示例阈值直接当成生产门禁；阈值必须按场景、基线和统计置信度校准。

### 1.3 Architecture Rules

1. **场景优先**：Benchmark 比较的是“模型 + 硬件 + 推理栈 + 负载 + SLA”的组合，而不是孤立的模型或芯片。
2. **证据可追溯**：每个 L1 结论都能追溯到 L2 请求级结果，再追溯到 L3 原始事件或硬件证据。
3. **质量先于性能**：任何吞吐、成本和容量结论都必须绑定质量门禁与 SLA。
4. **指标必须可行动**：每个 L3 指标必须有数据来源、统计口径、责任人和可能的优化动作。
5. **基线优先**：先建立稳定基线，再比较模型、硬件、引擎或 Kernel 的变化。
6. **不做全局排名**：输出场景内的硬约束过滤和 Pareto 候选，不输出脱离业务的“最佳模型/最佳硬件”。

---

## 2. Target Architecture

### 2.1 Three Delivery Layers

| Layer | Users | Questions | Required Outputs |
|---|---|---|---|
| **L1 Decision** | 管理层、PM、客户、业务方 | 能不能用、是否达标、多少钱、能支撑多少业务 | 业务质量、SLA、有效 Token 成本、容量、风险、候选排序和上线建议 |
| **L2 Service** | 应用开发、模型部署、测试、QA | 如何执行、如何回归、部署是否合格、哪里失败 | Golden Dataset、Schema/Tool/安全测试、部署检查、性能压测、回归门禁、标准结果 |
| **L3 Engineering** | Infra、推理服务、推理引擎、Kernel/算子开发 | 瓶颈在哪、应该改哪一层、修改是否有效 | 请求 Trace、Prefill/Decode、Queue、Batch、KV、通信、显存、GPU Counter、功耗和 Profiler 证据 |

### 2.2 Four Technical Domains

| Domain | Selection Questions | Optimization Questions |
|---|---|---|
| **算力硬件** | 算力、显存、带宽、互联、精度、功耗和供应是否满足场景 | 计算受限、访存受限、通信受限还是热受限 |
| **模型架构和算法** | Dense/MoE、Attention、上下文、量化和推理能力是否适配业务 | 质量损失来自模型、量化、采样、上下文还是算法路径 |
| **推理 Infra/服务/引擎/Kernel** | 部署拓扑、调度、Batch、KV Cache、并行和 API 是否满足 SLA | 问题发生在网关、调度、引擎、通信还是具体算子 |
| **Benchmark 方法论** | 用什么数据、负载、指标和统计方法进行比较 | 如何复现、归因、验证优化收益并形成决策 |

### 2.3 Shared Evidence Chain

```text
Scenario Card
  -> Workload Profile
  -> Model / Hardware / Stack Candidate Matrix
  -> L2 Quality and Service Evaluation
  -> L3 Engineering Profiling
  -> Root-cause Attribution
  -> Optimization Record
  -> L1 Cost, Capacity and Selection Decision
```

每次 Benchmark 运行都必须能标识以下事实：场景版本、模型版本或 digest、引擎和配置、硬件/驱动环境、数据集版本、负载分布、随机种子、预热和重复策略、SLA 版本、基线版本以及原始结果位置。

### 2.4 Agent Benchmark Semantics

Agent 场景必须同时观察模型调用和完整任务轨迹，不能用单次生成吞吐替代任务级结果。

| Concept | Definition in this system |
|---|---|
| **Sample** | 一个待完成的业务任务、Agent turn 或函数调用样本 |
| **Query** | 一次提交给 Agent 服务的任务请求，可能包含多个 turn 和 tool call |
| **SUT** | 被测完整系统，包括模型、编排器、工具适配器、推理服务、网络和后处理 |
| **Run** | 在指定场景、负载、质量门禁和延迟约束下完成的一次基准运行 |

Agent 的性能结果必须拆解为：

```text
端到端任务延迟
= 排队时间 + 模型推理时间 + 工具等待时间 + 编排/网络开销
```

Agent 质量、性能、效率和可靠性至少分别包含：

- **质量**：任务成功率、部分完成率、工具选择正确率、参数正确率、最终答案正确率、无效循环和提前终止率；
- **性能**：任务端到端延迟 P50/P95/P99、每一步 TTFT/ITL、工具延迟、排队时间、每分钟完成任务数；
- **效率**：Token/task、模型调用次数、工具调用次数、重试次数、GPU-seconds/task、成本/task；
- **可靠性**：超时率、工具失败恢复率、服务错误率、达到最大步骤数比例和轨迹方差。

Agent 结果不压成未经解释的单一总分；先应用质量硬门槛，再比较质量、延迟、吞吐和成本的 Pareto 候选。

---

## 3. Execution Workstreams

### Task 1: Freeze Architecture Vocabulary and Ownership

**Files:**
- Modify: `InferenceBenchmarkSystem/1-Master-Architecture-and-Design-Principles.md`
- Modify: `InferenceBenchmarkSystem/README.md`
- Reference: `InferenceBenchmarkSystem/2-Accuracy-Benchmark-and-Gating.md`
- Reference: `InferenceBenchmarkSystem/3-Performance-Benchmark-and-Capacity.md`
- Reference: `InferenceBenchmarkSystem/4-Tooling-Pipelines-and-Dashboards.md`

- [ ] **Step 1: Replace the current three-role framing with the three-layer delivery model.**

  主文档必须明确 L1/L2/L3 的用户、问题、输入、输出和边界；角色作为每层的用户群，而不是体系的唯一主轴。

- [ ] **Step 2: Add the four technical domains and their ownership boundaries.**

  为硬件、模型/算法、推理软件栈和 Benchmark 方法论分别定义输入、输出、责任团队及与其他域的接口。

- [ ] **Step 3: Add the shared evidence-chain rule.**

  明确 L1 结论必须能下钻到 L2 请求级结果和 L3 原始工程证据，并规定所有跨文档指标使用同一术语、单位和版本。

- [ ] **Step 4: Update README navigation and remove claims that imply an already implemented platform.**

  README 只描述当前方法论和规划交付，不把 Markdown 命令示例描述为已经存在的生产工具。

- [ ] **Step 5: Review vocabulary consistency.**

  检查 TTFT、TPOT、ITL、Goodput、SLA、质量门禁、CPM-Q、SingleStream、Server、Offline 和 Power 的定义是否全局一致。

**Acceptance:** 任何读者只看主文档和 README，都能回答“谁使用哪一层、依赖哪一个技术域、输出什么决策或工程动作”。

### Task 2: Build the Scenario Catalog and Workload Profiles

**Files:**
- Create: `InferenceBenchmarkSystem/6-Scenario-Catalog-and-Use-Cases.md`
- Modify: `InferenceBenchmarkSystem/2-Accuracy-Benchmark-and-Gating.md`
- Modify: `InferenceBenchmarkSystem/3-Performance-Benchmark-and-Capacity.md`

- [ ] **Step 1: Define the scenario-card template.**

  每张场景卡至少包含：业务目标、用户体验、输入/输出 Token 分布、请求到达模型、并发和多租户特征、质量门禁、SLA/SLO、候选模型、候选硬件、推理栈配置、Benchmark 用例、观测指标、常见瓶颈和推荐动作。

- [ ] **Step 2: Define the initial scenario catalog.**

  首批固定交互式聊天/客服、RAG/长上下文、Agent/Tool Calling、批处理/离线生成、多租户在线服务、端侧/边缘六类场景；多模态或语音作为扩展场景，必须复用同一场景卡格式。

- [ ] **Step 3: Define workload-profile dimensions.**

  规定不能只使用单一固定长度；必须描述输入和输出长度分布、长短请求比例、Burst、会话轮数、缓存命中、取消、失败和重试比例。

- [ ] **Step 4: Map each scenario to test tiers.**

  每类场景至少映射 Smoke、Quality Regression、SLA/Performance、Capacity/Stress 和 Profiling 五类用例，并说明每类用例的目的和停止条件。

- [ ] **Step 5: Add scenario-specific comparison rules.**

  明确交互场景重点看 TTFT/ITL，RAG 看检索和上下文质量，Agent 看端到端任务成功率与轨迹延迟，批处理看吞吐和成本，端侧看延迟、功耗和热稳定性。

- [ ] **Step 6: Define Agent trajectory classes and fault dimensions.**

  Agent 用例至少覆盖：单轮回答、单次工具调用、多步串行调用、可并行工具调用、长上下文/多轮记忆、真实代码/浏览器/数据库环境，以及工具超时、错误、空结果、重试和重新规划。每条轨迹标注输入/输出 Token、工具数、深度、并行宽度、工具延迟分布、工具失败率、最大步数和超时限制。

- [ ] **Step 7: Define deterministic and realistic execution modes.**

  提供可重复模式（固定工具结果、可注入延迟和失败）与真实环境模式（接入实际工具）两套方法；前者用于回归，后者用于真实性能验证。

**Acceptance:** 每个首批场景都有完整场景卡，并能从场景卡直接得到候选组合、用例集、主要指标和预期归因路径；Agent 场景另有完整轨迹分类、工具故障维度和端到端指标分解。

### Task 3: Define the Model and Hardware Selection Method

**Files:**
- Create: `InferenceBenchmarkSystem/7-Model-Hardware-Selection-Methodology.md`
- Modify: `InferenceBenchmarkSystem/1-Master-Architecture-and-Design-Principles.md`
- Modify: `InferenceBenchmarkSystem/3-Performance-Benchmark-and-Capacity.md`

- [ ] **Step 1: Define the model feature sheet.**

  记录模型版本、参数量和活跃参数量、Dense/MoE、Attention 结构、上下文长度、Tokenizer、Chat Template、KV Cache 特征、量化、推理能力、结构化输出和 Tool Calling 能力。

- [ ] **Step 2: Define the hardware capability sheet.**

  记录计算精度、算力、显存容量和带宽、互联拓扑、主机和 PCIe、功耗、温度、驱动/软件支持、供应约束和成本假设。

- [ ] **Step 3: Define hard filters before scoring.**

  先过滤质量不达标、显存放不下、上下文不足、精度不支持、无法部署或无法满足基本 SLA 的候选；不能用加权总分掩盖硬约束失败。

- [ ] **Step 4: Define the Pareto comparison.**

  在通过硬约束的候选中比较质量、SLA、容量、有效 Token 成本、功耗和运维复杂度；输出场景内 Pareto 候选及适用边界。

- [ ] **Step 5: Define capacity and cost assumptions.**

  明确时间窗口、利用率、折旧/租赁周期、输入/输出 Token、重试、缓存、故障冗余和峰值系数，禁止使用未经解释的固定 Buffer 得出采购结论。

**Acceptance:** 对任一场景，可以形成“候选过滤 -> Pareto 比较 -> 容量/成本 -> 推荐组合”的完整选型记录，并说明不确定性和适用边界。

### Task 4: Define the L2 Service Evaluation and QA Method

**Files:**
- Modify: `InferenceBenchmarkSystem/2-Accuracy-Benchmark-and-Gating.md`
- Modify: `InferenceBenchmarkSystem/4-Tooling-Pipelines-and-Dashboards.md`
- Create: `InferenceBenchmarkSystem/8-Service-Evaluation-and-QA-Methodology.md`

- [ ] **Step 1: Define the L2 test-suite tiers.**

  明确 Smoke、Quality Regression、Release Gate、SLA/Performance、Capacity/Stress 和 Deployment Acceptance 的入口、样本规模、运行条件、输出和失败处理。

- [ ] **Step 2: Define quality dimensions and task-level gates.**

  将确定性任务、开放生成、Schema、Tool Calling、RAG、长上下文、安全和引用溯源分开定义；每类任务使用自己的指标、阈值和统计策略，不使用一个未经校准的全局 99% 门槛。

- [ ] **Step 2a: Define Agent-specific quality gates.**

  将 Agent 任务成功率、工具选择、参数生成、幻觉调用、轨迹完整性、无效循环、失败恢复和最终答案分别计分；区分离线 Function Calling 准确性与在线多轮 Agent 任务完成度。

- [ ] **Step 3: Define deployment acceptance checks.**

  规定模型/镜像 digest、引擎配置、Tokenizer/Template、硬件环境、启动健康、预热、API 协议、流式行为、错误/超时/重试和回滚前置条件。

- [ ] **Step 4: Define the standard result contract.**

  结果至少分为 run metadata、case result、request result、agent trajectory event、metric summary、gate decision、failure sample 和 artifact references；所有结果都必须带版本和来源。

  Agent trajectory event 至少包含：`run_id`、`task_id`、`step_id`、时间戳、模型/工具、排队时间、调用延迟、输入/输出 Token、工具名、工具参数、错误、重试次数和 finish reason。

- [ ] **Step 5: Define QA statistical and change-control rules.**

  规定数据集版本、抽样、随机种子、重复次数、置信区间、基线更新、Flaky 分类、人工复核和门禁豁免记录。

**Acceptance:** QA 或测试人员可以根据文档独立编写测试计划，明确跑什么、何时判失败、如何复现、如何提交证据和如何归因。

### Task 5: Define the L3 Profiling and Optimization Method

**Files:**
- Create: `InferenceBenchmarkSystem/9-Engineering-Profiling-and-Optimization.md`
- Modify: `InferenceBenchmarkSystem/3-Performance-Benchmark-and-Capacity.md`
- Modify: `InferenceBenchmarkSystem/4-Tooling-Pipelines-and-Dashboards.md`

- [ ] **Step 1: Define the symptom-to-root-cause tree.**

  用“用户现象 -> L2 指标 -> 服务/调度 -> Prefill/Decode -> KV/Batch/通信 -> GPU/显存 -> Kernel”固定下钻顺序。

- [ ] **Step 2: Define the L3 metric registry.**

  每个指标必须包含名称、单位、采样粒度、聚合方式、数据源、适用场景、基线、责任人和对应优化动作；MBU/MFU 等指标不得只有公式和经验阈值。

- [ ] **Step 3: Define profiling use cases by engineering layer.**

  至少覆盖排队和调度、Continuous Batching、Prefill/Decode、KV/Prefix Cache、Tensor/Pipeline Parallel、通信、显存碎片、GEMM、Attention、MoE Kernel、功耗和热稳定性；Agent 额外覆盖编排开销、工具等待、关键路径、并行工具调用和重试放大。

- [ ] **Step 4: Define experiment isolation rules.**

  规定单变量变更、固定模型和数据、预热、重复、采样窗口、硬件状态、并发、功耗边界以及优化前后对照，避免把多个改动的总收益误归因给单一组件。

- [ ] **Step 5: Define optimization records.**

  每次优化记录问题、假设、变更、影响范围、L2 结果、L3 证据、质量回归、性能变化、成本变化和是否建议合入。

**Acceptance:** 对一个明确的性能或质量问题，工程师可以从 L2 结果定位到至少一个 L3 假设，并得到可验证的优化实验和回归结论。

### Task 6: Define the L1 Decision and Reporting Method

**Files:**
- Create: `InferenceBenchmarkSystem/10-Decision-Reports-and-Traceability.md`
- Modify: `InferenceBenchmarkSystem/1-Master-Architecture-and-Design-Principles.md`
- Modify: `InferenceBenchmarkSystem/4-Tooling-Pipelines-and-Dashboards.md`

- [ ] **Step 1: Define the L1 Decision Card.**

  固定展示业务质量、SLA、容量、有效 Token 成本、功耗、风险、候选比较、推荐组合、适用边界和结论置信度；不展示无法解释的底层指标。

- [ ] **Step 2: Define L2 and L3 report links.**

  每个 L1 结论必须引用对应的服务评测结果、失败样本、工程 Profile 和原始 artifact，形成从决策到证据的可追溯链。

- [ ] **Step 3: Define attribution language.**

  报告要区分“观测事实”“推断根因”“已验证动作”和“待验证假设”，禁止把经验性推测写成确定结论。

- [ ] **Step 4: Define decision types.**

  至少支持模型选型、硬件选型、推理栈选型、发布准入、性能回退、容量扩容和优化优先级七类决策。

- [ ] **Step 5: Define audience-specific views.**

  同一运行结果分别生成客户/PM、应用/QA 和 Infra/Kernel 视图，禁止三套视图重新计算出互相矛盾的指标。

**Acceptance:** 管理层可以只看 L1 做决策，QA 可以从 L1 进入 L2 失败证据，Infra 可以从同一结论继续下钻到 L3 原始指标和优化记录。

### Task 7: Integrate the Methodology Through a Paper Benchmark Pilot

**Files:**
- Modify: `InferenceBenchmarkSystem/README.md`
- Modify: `InferenceBenchmarkSystem/5-Execution-Plan.md`
- Reference: `InferenceBenchmarkSystem/6-Scenario-Catalog-and-Use-Cases.md`
- Reference: `InferenceBenchmarkSystem/7-Model-Hardware-Selection-Methodology.md`
- Reference: `InferenceBenchmarkSystem/8-Service-Evaluation-and-QA-Methodology.md`
- Reference: `InferenceBenchmarkSystem/9-Engineering-Profiling-and-Optimization.md`
- Reference: `InferenceBenchmarkSystem/10-Decision-Reports-and-Traceability.md`

- [ ] **Step 1: Select one representative scenario.**

  优先选择一个同时包含业务质量、在线 SLA 和明显工程瓶颈的场景，例如 RAG 在线问答或 Agent Tool Calling。

- [ ] **Step 2: Walk one candidate matrix through the complete evidence chain.**

  使用至少两个模型候选、两个硬件或部署候选和两个推理栈配置，完成场景卡、L2 评测、L3 归因和 L1 决策的纸面闭环。Agent 试点至少包含 40 个任务、4 类轨迹、并发 1/8/32、工具失败率 0%/5%/20%，每个任务重复 5 次。

- [ ] **Step 3: Run a consistency review.**

  检查术语、指标、阈值、单位、样本口径、结论和引用是否一致；特别检查质量门禁、SLA 分位数、成本、容量和 Power 的定义。

- [ ] **Step 4: Review with all four user groups.**

  由 PM/客户代表、应用开发/部署代表、测试/QA 代表和 Infra/引擎代表分别确认报告是否能支持其决策或优化动作。

- [ ] **Step 5: Record unresolved gaps as methodology debt.**

  将不能由当前证据回答的问题写入后续计划，不用增加没有明确使用场景的指标或工具。

**Acceptance:** 一个完整场景能够同时产出 L1 Decision Card、L2 Service Evaluation Report 和 L3 Engineering Profile，且三份材料的结论互相一致、可以互相追溯。

---

## 4. Cross-Cutting Governance

### 4.1 Versioned Contracts

以下对象必须版本化：

- Scenario Card；
- Workload Profile；
- Model Feature Sheet；
- Hardware Capability Sheet；
- Metric Registry；
- Quality/SLA Gate Policy；
- Benchmark Run Metadata；
- Decision Report。

MLPerf 场景语义作为外部对照时，固定使用 **SingleStream、MultiStream、Server、Offline** 四种核心负载场景；Power/能效作为跨场景的测量维度，不替代负载场景。Agent 的离线函数调用准确性和在线多轮轨迹性能必须分开报告。

### 4.2 Ownership

| Artifact | Primary Owner | Reviewers |
|---|---|---|
| 场景卡和业务 SLO | PM/业务方 | 应用、QA、Infra |
| 质量用例和 Golden Dataset | 应用/QA | PM、模型、测试 |
| 模型特征和质量基线 | 模型/应用团队 | QA、Infra |
| 硬件能力和容量基线 | Infra 厂家/Infra 团队 | PM、服务、财务 |
| 推理栈和 Profiling 证据 | 推理服务/引擎/Kernel 团队 | Infra、QA |
| 指标注册表和报告口径 | Benchmark Owner | 四类代表共同评审 |

### 4.3 Review Gates

1. **Architecture Gate**：三层、四域、场景卡和证据链无歧义。
2. **Methodology Gate**：每个场景都有负载、质量、性能、成本和归因方法。
3. **Traceability Gate**：L1 结论可追溯到 L2/L3 证据。
4. **Audience Gate**：PM、应用/QA、Infra 三类报告都能支持实际决策。
5. **Pilot Gate**：至少一个场景完成纸面闭环，并记录方法论债务。

---

## 5. Deferred Implementation Plan

实现阶段不属于本设计计划的验收范围，只有在 Task 1-7 通过后才进入。后续应按以下顺序单独立项：

1. 标准化 Benchmark Run Schema 和 Metric Registry；
2. 实现一个最小 Runner，覆盖部署检查、L2 评测和性能测试；
3. 接入 L3 Trace、GPU/DCGM、功耗和 Profiler 数据；
4. 接入 CI 门禁、结果存储和基线比较；
5. 最后构建 L1/L2/L3 Dashboard 和客户报告出口。

本计划的完成标志不是“工具已经上线”，而是“方法论能够指导选型、评测、Profiling、优化和决策，并且四类用户对同一证据链得到一致结论”。
