# AMPD：多轮 LLM 推理下的解耦服务解读（2026）

> 论文：*Efficient Multi-round LLM Inference over Disaggregated Serving*
> arXiv：[2602.14516](https://arxiv.org/abs/2602.14516) ｜ HTML：[v2](https://arxiv.org/html/2602.14516v2)
> 关键词：Agent 推理、迭代 RAG、Prefill/Decode 解耦、KV cache、SLO 调度

## 1. 一句话主旨

多轮 Agent/RAG 工作流会让 Prefill 反复发生在 Decode 过程中。AMPD 不再固定把所有 Prefill 送到专用 Prefill worker，而是根据实时 TTFT/ITL 余量、排队时间和 KV cache 传输成本，动态决定每次增量 Prefill 放在 Decode worker 本地执行，还是远程送到 Prefill worker 执行，从而在同样 GPU 资源下提高满足延迟 SLO 的请求比例。

## 2. 问题背景：多轮工作流改变了负载形态

传统单轮请求是：

```text
用户 Prompt
  ↓
Prefill：一次性处理输入，生成 KV cache
  ↓
Decode：逐 token 输出回答
```

但 Agent、工具调用和迭代 RAG 是：

```text
初始问题 + 系统提示 + 工具说明
  ↓ Prefill
Decode：生成工具调用
  ↓
搜索 / 数据库 / 代码执行
  ↓
工具结果追加进上下文
  ↓ 增量 Prefill
Decode：根据新证据继续推理
  ↓
再次调用工具 / 检索
  ↓
又一段环境结果进入上下文
  ↓ 又一次增量 Prefill
Decode：继续生成
```

因此，一次用户会话内部可能出现多轮 Prefill/Decode 交替。Prefill 不再只是请求开头的一次性任务，而会随着工具结果、检索文档、环境反馈不断插入。

这带来两个直接后果：

1. **上下文持续变长**：历史 KV cache 越来越大，新增上下文的处理成本和搬运成本都在上升。
2. **负载交错化**：系统必须在“尽快处理新增输入”和“保证已开始输出的 token 流稳定”之间做实时权衡。

## 3. 核心概念辨析

### 3.1 什么是静态 PD 解耦？

Prefill/Decode disaggregation 把推理按阶段拆到不同资源池：

- **Prefill**：处理输入上下文并计算 KV cache，通常偏 compute-bound，更吃算力。
- **Decode**：逐 token 生成并反复读取 KV cache，通常偏 memory-bound，更吃 HBM 带宽。

常见静态规则是：

```text
所有 Prefill → Prefill GPU 池
所有 Decode → Decode GPU 池
```

这里的“静态”不是指完全不能动态组 batch，而是指**阶段路由和资源划分基本固定，不根据每个任务、每个 worker 的实时 SLO 余量重新决策**。

在单轮场景中，这种解耦可以减少 Prefill 和 Decode 互相抢资源；但在多轮 Agent/RAG 中，增量 Prefill 不断出现，静态远程执行会带来：

1. **KV cache 传输成本**：远端 Prefill worker 可能需要读取历史 KV，算完后再把新增 KV 传回 Decode worker。
2. **Prefill 队列拥堵**：多个会话的工具结果同时返回时，大量增量 Prefill 集中打到 Prefill 池，推高 TTFT。
3. **资源比例失配**：不同 workload 的 Prefill/Decode 强度不同，固定 GPU 配比容易造成一侧排队、一侧闲置。

### 3.2 什么是 Decode 干扰？

如果为了避免远程 KV 传输，把增量 Prefill 放在 Decode worker 本地执行，又会打断原本的逐 token 生成：

```text
Decode worker 正在批量 Decode
  ↓
某会话的工具结果返回，需要增量 Prefill
  ↓
Prefill 在同一 GPU 上执行
  ↓
占用算力 / HBM 带宽 / SM
  ↓
其他请求的 Decode 被阻塞或变慢
  ↓
ITL 尖刺，SLO 违约
```

这就是 Decode 干扰：新增输入的处理任务打断了正在进行的 token 生成，造成 Inter-Token Latency 抖动。

## 4. AMPD 的核心机制

AMPD 的贡献是把这个两难变成实时调度问题：

- 远程 Prefill：保护 Decode，但增加 KV 传输和 Prefill 排队。
- 本地 Prefill：省网络、可能降低 TTFT，但会干扰 Decode。

### 4.1 自适应路由

每个增量 Prefill 到来时，coordinator 判断放在哪里执行：

1. 若某个 Prefill worker 的窗口化 TTFT 仍有足够余量，则远程路由到该 worker。
2. 若 Prefill worker 均接近 TTFT 阈值，而当前 Decode worker 的 ITL 仍有余量，则本地执行。
3. 否则使用性能模型比较本地与远程的预计完成时间。

远程成本包含：

```text
远程执行成本 = Prefill 计算 + KV 传输 + 队列等待
```

本地成本包含：

```text
本地执行成本 = 本地 Prefill 计算 + 本地队列等待
```

决策目标是提高 SLO attainment，即更多请求同时满足 TTFT 和 ITL 要求，而不是只优化单一平均延迟。

### 4.2 TTFT-aware Prefill 重排

AMPD 只查看 Prefill 队列头部的小窗口，枚举其中的执行顺序，选择能让最多任务满足 TTFT SLO 的顺序。

它还维护 postponement counter，限制每个任务被连续推迟的次数，避免短任务或余量大的任务长期挤压紧急任务，防止饥饿。

### 4.3 离线部署规划

AMPD 将 Prefill/Decode worker 的部署建模为资源约束下的 ILP/MILP 问题，决定：

- Prefill/Decode 各分配多少 GPU；
- 每类 worker 建多少个数据并行副本；
- 每个副本采用多大的模型并行度。

目标是在固定 GPU 容量内最小化瓶颈 worker 的估计 P95 延迟。

### 4.4 工程实现

论文实现基于 NVIDIA Dynamo：

- Redis：共享任务队列与窗口化 TTFT/ITL 统计；
- NIXL：跨 Prefill/Decode worker 的 RDMA KV cache 传输；
- lazy read：先传任务 metadata，真正调度远程 Prefill 时再读取历史 KV；
- 计算与传输重叠：用前一个任务的计算隐藏 KV 传输延迟。

## 5. 实验结论

实验环境：

- 4 台服务器，每台 8 张 NVIDIA H20 96GB GPU；
- 节点内 NVLink 900GB/s，节点间 InfiniBand 200GB/s；
- 模型：Qwen3-32B、Llama3.1-70B、Mixtral-8x7B；
- workload：ToolBench、GAIA、HotpotQA、DuReader；
- baseline：NVIDIA Dynamo、vLLM、vLLM-Continuum。

论文报告：

- 相比 Dynamo，SLO attainment 平均提升 **67.29%**，最高 **967.54%**；
- 相比 vLLM，SLO attainment 平均提升 **339.74%**，最高 **3435.1%**；
- 自适应路由将 **13.9%~31.7%** 的 Prefill 任务放在 Decode worker 本地执行；
- 路由平均开销约 **0.39ms**，重排平均开销约 **0.45ms**，相对 GPU 推理延迟可以忽略。

解读这些数字时要看基线的绝对 SLO attainment：百分比提升在低基线场景下会显得很大。工程落地时应以自己的 workload、SLO 阈值、GPU 拓扑和网络带宽为基准复测。

## 6. 对 Token 经济学的价值

AMPD 的价值不是让模型变小，而是提高同硬件下的**有效 token 供给质量**。

Agent 推理的真实成本不只来自生成 token 的 FLOPs，还包括：

- 反复处理新增上下文；
- 历史上下文的 KV cache 显存占用；
- KV cache 跨 GPU / 跨节点搬运；
- Prefill 与 Decode 的资源错配；
- 队列等待造成的 SLO 违约。

AMPD 通过实时调度减少这些隐性成本，把 GPU 从“生产 token”进一步优化为“生产满足交互延迟要求的 token”。这对 Agent 服务尤其重要：用户感知价值不仅取决于输出内容，也取决于每轮工具调用后的响应是否稳定。

## 7. 边界与后续方向

1. **依赖性能模型**：自适应路由需要较准确的 Prefill、Decode、KV 传输成本估计。
2. **离线 planner 假设较固定**：论文承认 workload 漂移、弹性扩缩容、故障恢复后需要重新规划。
3. **本地 Prefill 仍有干扰风险**：只有 Decode ITL 余量足够时才适合本地执行。
4. **可与 chunked prefill 结合**：把长增量 Prefill 切块，进一步降低本地执行对 Decode 的阻塞。

## 8. 结论

AMPD 的核心洞察是：多轮 Agent/RAG 让 Prefill 从“请求前置阶段”变成“贯穿请求生命周期的重复任务”。静态 PD 解耦虽然能隔离资源，却会被增量 Prefill 和 KV 传输拖慢；简单本地执行又会造成 Decode ITL 抖动。AMPD 用自适应路由、队列重排和部署规划，把 TTFT/ITL 权衡变成实时 SLO 调度问题，是 Agent 推理基础设施中很有代表性的方向。
