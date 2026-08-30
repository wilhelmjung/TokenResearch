# 📚 LLM 推理与端侧异构计算顶会论文库 (Inference Systems & Edge Architecture)

> 本目录收录大模型推理优化、端侧异构计算（CPU/GPU/NPU/UMA）、内存子系统及调度优化领域的顶级学术会议（SOSP、OSDI、EuroSys、ASPLOS 等）前沿论文与开源系统实现。
> 重点聚焦：**如何通过系统级创新攻克内存瓶颈、降低端侧及数据中心每个 Token 的推理成本（Cost per Token）**。

---

## 🧭 论文库全景导览

| 论文标识 | 论文全称 | 发表会议 | 核心贡献 / 优化维度 | 开源 / 状态 |
| :--- | :--- | :--- | :--- | :--- |
| 🌟 **libumsh (主推)** | **[Characterizing CPU-GPU Memory Sharing over Unified Memory Architecture](#-重点关注主推论文libumsh-eurosys-27)** | **EuroSys '27** | 统一内存架构（UMA）跨平台 CPU-GPU 零冗余共享库 | 刚刚条件接收 |
| 🚀 **PowerInfer** | *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU* | **SOSP '24** | 神经元激活幂律分布 × 消费级 GPU+CPU 混合高吞吐推理 | [GitHub (18k+ ⭐)](https://github.com/SJTU-IPADS/PowerInfer) / [arXiv](https://arxiv.org/abs/2312.12456) |
| 📱 **PowerInfer-2** | *PowerInfer-2: Fast Large Language Model Inference on Smartphones* | **Preprint** | 智能手机 SoC 异构多核流水线与张量级细粒度编排 | [GitHub](https://github.com/SJTU-IPADS/PowerInfer-2) / [arXiv](https://arxiv.org/abs/2406.06282) |
| ⏱️ **XSched** | *XSched: Preemptive Scheduling for Diverse XPUs* | **OSDI '25** | 异构硬件加速器（GPU/NPU/ASIC）统一抢占式调度与 XQueue 抽象 | [USENIX Open Access](https://www.usenix.org/conference/osdi25) |
| 🧩 **HeteroInfer** | *Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference* | **SOSP '25** | 手机端 GPU 与 NPU 协同异构并行与硬件感知张量切分 | [ACM DL](https://dl.acm.org/) |
| 🛡️ **Sereno** | *Inference in the Shadows: Taming Memory Bandwidth Contention in Mobile LLM Inference with Sereno* | **OSDI '26** | 移动端大模型多处理器共享内存的带宽争用缓解与动态隔离 | [USENIX OSDI '26](https://www.usenix.org/) |

---

## 🌟 重点关注 / 主推论文：libumsh (EuroSys '27)

### 1. 基本信息
* **论文全名**：*Characterizing CPU-GPU Memory Sharing over Unified Memory Architecture*
* **作者团队**：Jinrong Yang, Zimeng Wang, Rong Chen, Haibo Chen（上海交通大学 IPADS 实验室）
* **发表录用**：The 22nd European Conference on Computer Systems (**EuroSys '27**), Rabat, Morocco.
* **官方收录索引**：[SJTU IPADS Publications](https://ipads.se.sjtu.edu.cn/publications/)

---

### 2. 为什么需要特别关注该论文？（背景与产业痛点）

随着端侧 AI（On-Device AI）爆发，大模型正加速从云端数据中心走向 PC、笔记本、手机及边缘迷你超算：
1. **UMA 成为端侧硬件事实标准**：统一内存架构（Unified Memory Architecture, UMA）通过物理共享内存，消除了传统独立显存（Discrete VRAM）的显存瓶颈与成本，支持 128GB~512GB 的大内存寻址（如 Apple Silicon M-series、NVIDIA GB10/DGX Spark、高通骁龙 X Elite）。
2. **主流大模型框架存在巨大的“架构错位”**：
   * **性能浪费**：主流推理框架（PyTorch、`llama.cpp`、`vLLM` 等）仍沿用传统分立式 GPU（PCIe 拷贝）的思维定势，在权重加载与 KV 缓存/输入输出处理中残留了大量**冗余内存拷贝**，浪费了宝贵的内存带宽与生成延迟。
   * **正确性隐患**：直接内存访问如果不进行细致的 CPU/GPU 缓存维护（Cache Maintenance / Invalidation），会导致严重的跨处理器数据不一致性（例如部分 Vulkan/MoltenVK 渲染管线的隐蔽 Bug）。
3. **平台黑盒复杂**：不同厂商的 SoC 在 CPU-GPU 互连拓扑、缓存组织层级、硬件/软件一致性协议上极度碎片化，开发者极难写出既正确又最优的零拷贝共享代码。

---

### 3. 核心技术创新与方法论

```
                     ┌──────────────────────────────────────────────┐
                     │            应用层 (Application)             │
                     │   PyTorch / llama.cpp / vLLM / EdgeNN / GPUfs │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │            libumsh 跨平台内存共享库          │
                     │  - 数据面零拷贝传输 (Zero-Copy Data Plane)    │
                     │  - 控制面轻量协同 (Optimized Control Plane)   │
                     │  - 跨硬件缓存一致性自动维护 (Auto Cache Maint)  │
                     └──────────────────────┬───────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      【Apple Silicon UMA】         【NVIDIA GB10 / Spark】       【Qualcomm / ARM SoC】
     (M4/M5/M6 Ultra 512GB)         (Grace Blackwell 128GB)         (Snapdragon X Elite)
```

1. **三维解耦分析体系**：
   * **应用维**：将共享行为划分为“数据面传输（Data Plane）”与“控制面协同（Control Plane）”；
   * **实现维**：将每次共享细拆为可独立配置的“发送（Sender）”与“接收（Receiver）”阶段；
   * **平台维**：全覆盖主流主流端侧 UMA 设备，系统性测定所有正确配置集合并定位性能帕累托前沿。
2. **`libumsh` 跨平台统一抽象库**：
   * 对上层大模型框架屏蔽复杂的驱动与底层缓存一致性细节；
   * 自动根据硬件平台与调用场景自适应匹配最优内存共享通路；
   * 提炼了适配全新 UMA 芯片的“三步扩展法”（并在 NVIDIA DGX Spark 上完成验证）。

---

### 4. 优化实测效果（9 个真实系统改造）

在采用 `libumsh` 深度改造包括 **PyTorch、`llama.cpp`、`vLLM`、EdgeNN、OpenCV、GPUfs** 等 9 个工业级与开源系统后：
* ⚡ **数据面传输耗时**：降低 **16% ~ 75%**；
* ⏱️ **控制面协同开销**：降低 **7% ~ 90%**；
* 🔒 **正确性保证**：消除潜在的 Cache Coherence 导致的偶发数据损坏，实现安全高效的零拷贝内存共享。

---

## 📚 异构计算与端侧推理系列论文精读

### 1. PowerInfer (SOSP '24)
* **题目**：*PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*
* **论文链接**：[arXiv:2312.12456](https://arxiv.org/abs/2312.12456) ｜ [GitHub 代码库](https://github.com/SJTU-IPADS/PowerInfer)
* **核心突破**：
  * 首次系统性论证了大模型推理中 MLP 神经元激活的**幂律分布（Power-law Sparsity）**：极小比例（如 10%~20%）的高频“热神经元”贡献了绝大部分激活。
  * **GPU-CPU 协同解耦**：将热神经元常驻消费级显卡（如 RTX 4090 24GB），冷神经元交由主机 CPU 动态计算，将消费级硬件上的大模型推理速度推高了 **11 倍**，达到接近 A100 服务器级别的吞吐。

### 2. XSched (OSDI '25)
* **题目**：*XSched: Preemptive Scheduling for Diverse XPUs*
* **论文链接**：[USENIX OSDI '25 论文集](https://www.usenix.org/conference/osdi25)
* **核心突破**：
  * 针对现代大模型集群与端侧设备中 GPU、NPU、ASIC 等异构算力单元（XPU）缺乏灵活抢占与调度原语的问题，提出 **XQueue** 可抢占指令队列抽象；
  * 支持多租户、多 Agent 场景下的细粒度任务抢占、优先级保证与 SLO 延迟敏感型调度。

### 3. HeteroInfer (SOSP '25)
* **题目**：*Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference*
* **论文链接**：[ACM Digital Library (SOSP '25)](https://dl.acm.org/)
* **核心突破**：
  * 针对智能手机 SoC 算力分散、单计算单元（如纯 NPU 或纯 GPU）显存和算力吞吐受限的瓶颈；
  * 设计细粒度张量混合切分算子与异步流水线，让 GPU 与 NPU 同步并发计算，显著缩短端侧大模型首字延迟（TTFT）与解码延迟（TPOT）。

### 4. Sereno (OSDI '26)
* **题目**：*Inference in the Shadows: Taming Memory Bandwidth Contention in Mobile LLM Inference with Sereno*
* **论文链接**：[USENIX OSDI '26 论文集](https://www.usenix.org/) ｜ [作者主页](https://tongxin.github.io/)
* **核心突破**：
  * 深入分析了在移动端 UMA 架构下，大模型自回归解码（Memory-bound 访存密集型）与后台多媒体/渲染任务并发时严重的**内存总线带宽争用（Bandwidth Contention）**；
  * 提出操作系统级的软硬件协同带宽监控与动态流控机制，保障大模型常驻交互式推理的确定性低延迟。

---

## 🔗 与本仓库其他专题的联动

* 🖥️ **硬件深度结合**：
  * 配合 [`hardware/1-GB10-UMA-OperatorOptimization.md`](../hardware/1-GB10-UMA-OperatorOptimization.md) 了解 NVIDIA GB10 统一内存算子优化；
  * 配合 [`hardware/4-Apple-MacStudio-M5-M6-Ultra-UMA.md`](../hardware/4-Apple-MacStudio-M5-M6-Ultra-UMA.md) 查看 512GB UMA 架构常驻 670B 模型的工程实现。
* ⚙️ **专属裸机引擎**：
  * 配合 [`Topic-1-TokenEconomy/3-ModelSpecific-Baremetal-Engines.md`](../Topic-1-TokenEconomy/3-ModelSpecific-Baremetal-Engines.md) 查看轻量化 C++/CUDA 裸机推理引擎如何通过绕过框架冗余实现极致降低 Token 成本。
