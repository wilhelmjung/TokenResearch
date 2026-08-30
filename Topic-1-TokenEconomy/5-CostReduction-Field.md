# Token经济学中：降低每Token成本属于哪个部分和领域

## 结论

**供给侧/生产端 - Infra效率层**，对应 `6-WAIC2026-Token经济学原文.md` 提出的核心标尺 `TCO（总拥有成本）/ 每百万Token成本`。与Web3中Token分配/激励设计无关，属于AI推理的系统工程经济学。

## 1. 在Token经济学框架中的定位

| 维度 | 归属 | 说明 |
| :--- | :--- | :--- |
| **供需划分** | 供给侧（生产端） | 负责“生产”Token的效率，而非“消耗”Token的需求 |
| **经济学属性** | 成本经济学 / 生产率经济学 | 目标：压低单位产出的边际成本 |
| **与需求侧区别** | 需求侧=节流（少用Token） | 如Prompt压缩、减少调用；供给侧=降本（更便宜地生产Token） |

> 公式：`每Token成本 = 年化TCO / 年化吞吐Tokens`。所有工作要么压低分子(TCO)，要么做大分母(吞吐)。

## 2. 在“五层蛋糕”产业链中的位置

黄仁勋五层：`能源 -> 芯片 -> Infra -> 模型 -> 应用`

*   **属于：芯片层 + Infra层** `4-HowToTakeTokenCostDown.md:第4章`
*   **不属于：模型层/应用层**（后者是Token的消费者，为成本买单方）

WAIC原文观点：现在很难以固定比例分配各层价值，因为“最后看的是每个Token的成本”，芯片企业必须向软件栈与网络协议延伸，实现深度耦合。

## 3. 细分技术领域

### 3.1 硬件层（压低TCO分子）
*   芯片架构：Chiplet、晶体管与工艺优化
*   存储与互联：HBM带宽、NVLink、PD分离（Prefill-Decode分离）
*   能源与散热：单位算力功耗

### 3.2 系统软件层（做大吞吐分母）

#### 3.2.1 存储层（Storage）
*   **PagedAttention**：显存管理，借鉴OS分页解决KV Cache显存碎片化，非调度算法，使显存利用率提升，为大Batch提供物理前提。
*   **Prefix Caching**：缓存复用，引擎内自动复用相同前缀（System Prompt/历史）的KV，避免Prefill重算，属于存储复用/计算复用，可理解为存储层之上的Cache。

#### 3.2.2 调度层（Scheduling）
*   **Continuous Batching / Iteration-level Scheduling**：决定请求何时进/出Batch的时序策略，将GPU利用率从 30% -> 90%。

#### 3.2.3 计算优化层（Compute）
*   **量化**：FP8/INT8（吞吐 x1.5-2）
*   **投机采样 Speculative Decoding**：小模型草稿+大模型验证，速度 x2-3
*   **分级存储**：HBM/DRAM/SSD协同

> PagedAttention与Prefix Caching已合并为存储层（Memory+Cache），`4-HowToTakeTokenCostDown.md:第4章` 已同步。

## 4. 与应用层优化的边界

| 层级 | 目标 | 典型手段 | 归类 |
| :--- | :--- | :--- | :--- |
| **Infra层（本篇）** | 降低“生产”成本 | 量化、批处理、硬件选型 | 供给侧降本 |
| **应用层** | 降低“消耗”数量 | 会话压缩、智能路由、缓存命中、限制上下文 | 需求侧节流 |

两者互补：Infra降本决定“每Token能多便宜”，应用节流决定“少用多少Token”。商业闭环需两者同时达标：`每Token成本 < 每Token定价`。

---
*归档：TokenResearch/Topic-1-TokenEconomy/5-CostReduction-Field.md · 对应 WAIC2026 Token经济学 TCO范式*
