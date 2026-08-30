# TokenResearch: AI 基础设施、Token 经济学与算力芯片全景调研库

> 本代码库系统研究大模型（LLM/VLM）时代的 **Token 经济学（Tokenomics）**、**AI 全栈基础设施（Infra）降本路径** 与 **全球算力芯片（Compute Hardware & ASICs）微架构与优化实战**。
>
> 🌐 **在线交互式汇报大屏 (Live Demo)**：👉 **[https://wilhelmjung.github.io/TokenResearch/](https://wilhelmjung.github.io/TokenResearch/)**（支持深浅色切换、图文架构动效与内置 TCO 测算器）

---

## 🧭 知识库目录全景

```
TokenResearch/
├── Report/                      # 📊 决策简报与综合报告（Executive Summaries & Slides）
│   ├── README.md                             # 报告目录索引与导航
│   ├── Token-Economics-and-Inference-Infra-Presentation.html # 🎨 交互式图文分享大屏（汇报演示用）
│   └── 1-Executive-Summary-TokenEconomics-2026.md # 2026 Token 经济学与推理降本高管决策简报
│
├── Topic-1-TokenEconomy/        # 📘 主题一：Token 经济学、全栈架构与策略实操
│   ├── 1-Infra-TokenCostReduction-Report.md   # 全栈总览大白皮书（五层蛋糕、选型矩阵、TCO模型）
│   ├── 2-Qwen3.8-Flash-Next-Architecture-Analysis.md # Qwen3.8 混合注意力与 6B 极致稀疏专题
│   ├── 3-ModelSpecific-Baremetal-Engines.md  # 专属裸机推理引擎专题（DS4.c / H3.c 范式）
│   ├── 4-HowToTakeTokenCostDown.md           # Token 降本六大实操策略（Prompt压缩/路由/缓存）
│   ├── 5-CostReduction-Field.md              # Token 经济学理论归属与分层定位
│   ├── 6-WAIC2026-Token经济学原文.md         # WAIC 2026 天数智芯 × 易方达 产业深度对谈
│   ├── 7-腾讯研究院-Token经济学七问.md       # 腾讯研究院发布的新经济宏观全景地图
│   ├── 8-智谱AI-GLM-5.3-Flash原生多模态架构.md # 智谱 GLM-5.3-Flash 原生全模态与混合注意力专题
│   └── 9-月之暗面-Kimi-K3-2.8T超大规模MoE与KDA架构.md # 月之暗面 2.8T MoE 与 KDA/Mooncake 架构专题
│
├── hardware/                    # 🚀 算力硬件、自研 ASIC 与 UMA 算子优化专题
│   ├── README.md                             # 算力硬件目录索引与芯片选型速查
│   ├── 1-GB10-UMA-OperatorOptimization.md   # NVIDIA GB10 (Grace Blackwell) 算子级深度优化
│   ├── 2-OpenAI墨西哥辣椒自研芯片.md         # OpenAI Jalapeño 芯片与 InferenceX 实测（104x能效）
│   ├── 3-mu25-L20-多模态专属裸机引擎实战.md # mu25 纯 C++/CUDA 裸机引擎在 L20/GB10 上的实测突破
│   └── 4-Apple-MacStudio-M5-M6-Ultra-UMA.md  # Apple Mac Studio (512GB UMA) 桌面千亿模型常驻
│
└── TODO.md                      # 📌 学术前沿与研究待办清单（ACL 2026 / ICML 2026 AI Infra 论文）
```

---

## 🌟 核心量化发现与突破

1. **宏观爆发与 Jevons 悖论**：
   * 中国日均 Token 消耗量 2 年暴增 1400 倍突破 140 万亿；J.P. Morgan 预测 2030 年全球日均消耗达 3,900 万亿；
   * 单价降低不会减少总算力支出，反而激发 Agent 7×24 小时机器级消耗，推动总支出爆炸式增长。
2. **算力硬件重构（带宽与近存优先）**：
   * **NVIDIA H20**：以 15% 的算力保留 120% 的 HBM 带宽，成为国内自回归解码的高能效基石；
   * **OpenAI Jalapeño ASIC**：联合 Cerebras 打造 15.4 TB/s HBM4 近存互联，在 DeepSeek R1 低延迟解码下取得 **104.3x TPS/kW 每千瓦能效领先**；
   * **Apple Mac Studio (M5/M6 Ultra)**：**512GB 统一显存（UMA）+ 1.2~1.6 TB/s 带宽**，桌面级 200W 功耗实现 **DeepSeek-R1 670B 顶尖大模型单机常驻**，零网络通信税；
   * **NVIDIA GB10 (DGX Spark)**：128GB UMA + 1 PFLOPS FP4，实现企业级 Agent 零边际 Token 生产成本；
   * **小米玄戒 O100**：3D 近存晶圆堆叠（1.22 TB/s），在手机端实现 3B 模型 **330 tok/s 极速推理**。
3. **两维架构解耦（PD 分离 × AM 分离）**：
   * **PD 分离（时间维）**：消除行头阻塞，P99 延迟降低 70%，有效吞吐（Goodput）提升 2–3 倍；
   * **AM 分离（空间维 / JANUS）**：Attention 与 MoE 专家池独立弹性扩缩容，彻底消除“资源搁浅”，综合硬件利用率突破 **85%+**。
4. **算法与裸机引擎突破**：
   * **DFlash 块扩散并行投机采样**：突破自回归瓶颈，实现 **5–6x 无损加速**（接受率 > 89%）；
   * **mu25 专属裸机引擎**：在 NVIDIA L20 与 GB10 上相比通用 vLLM，首字延迟 (TTFT) 缩短 **19.8%–20.9%**，生成延迟 (TPOT) 缩短 **27.3%**，零显存泄漏。
