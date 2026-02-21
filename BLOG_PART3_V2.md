# vLLM 从入门到专家（2.0 版）第三部分

> **本部分内容**：DeepSeek MoE（混合专家架构）、MLA（多头潜在注意力）、PD 分离（Prefill-Decode 解耦）、vLLM V1 引擎全局架构、Mini vLLM 完整整合。
>
> **阅读前提**：已读第一、二部分，理解 PagedAttention、KV Cache、Scheduler、Chunked Prefill。

---

## 第十一章：DeepSeek MoE——专家混合架构深度解析

### 11.1 理论背景：为什么需要 MoE？

现代大语言模型遵循一个近乎铁律的规律：**参数越多，能力越强**。然而，随着模型规模扩展到百亿、千亿参数，计算量也线性增长，推理延迟和成本急剧上升。

**稠密 Transformer 的困境**：

```
标准 FFN 层（每个 token 都经历全部计算）：

  x ──→ Linear(d_model → 4d_model) ──→ GELU ──→ Linear(4d_model → d_model) ──→ y

  计算量：O(d_model² × seq_len)
  缺点：参数量 ↑ → 计算量等比 ↑
```

**MoE（Mixture of Experts，专家混合）** 打破这个困局：

> 用 N 个"专家"（每个是一个小 FFN）替代一个大 FFN。每个 token 只激活 K 个专家（K << N），模型参数量大但计算量小。

```
MoE FFN 层：

  x ──→ Router ──→ 选择 Top-K 个专家（K/N < 1/10）
                        ├─ Expert_3(x)  ─┐
                        └─ Expert_47(x) ─┤ 加权求和 → y
                                         │
                                  w3 * out3 + w47 * out47

  参数量：N × expert_size（很大）
  计算量：K × expert_size（很小）
```

**论文参考**：

- MoE 在 NLP 的奠基作：Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017
  https://arxiv.org/abs/1701.06538

- Google Switch Transformer（简化路由到 Top-1）：Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022
  https://arxiv.org/abs/2101.03961

- DeepSeek MoE 创新（共享专家 + 细粒度路由）：Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, ACL 2024
  https://arxiv.org/abs/2401.06066

- DeepSeek V2（GroupedTopK + MLA）：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024
  https://arxiv.org/abs/2405.04434

### 11.2 DeepSeek MoE 的创新设计

#### 11.2.1 共享专家 + 路由专家双轨架构

普通 MoE 中，所有专家都参与竞争。DeepSeek 发现这会导致部分专家"退化"成通用知识存储器，浪费容量。

解决方案：**显式分离共享知识与专门知识**：

```
DeepSeek MoE 架构（每层）：

                        ┌─────────────────────────┐
          x ─────────── │     Shared Expert(s)     │ ──────┐
          │             └─────────────────────────┘       │
          │                                               │ 加法合并
          │             ┌─────────────────────────┐       │
          └──→ Router ──│  Routed Expert Pool     │ ──────┘
                        │  (Top-K 竞争)           │
                        └─────────────────────────┘

共享专家：每个 token 必然经过（存储通用知识）
路由专家：每个 token 选 K 个（存储专业知识）
```

**DeepSeek V2 / V3 配置对比**：

| 参数 | DeepSeek V2 | DeepSeek V3 |
|------|------------|------------|
| 路由专家数 (`n_routed_experts`) | 160 | 256 |
| 共享专家数 (`n_shared_experts`) | 2 | 1 |
| 每 token 激活路由专家数 | 6 | 8 |
| 分组数 (`n_group`) | 8 | 8 |
| 每组选取数 (`topk_group`) | 3 | 4 |
| 总参数量 | 236B | 671B |
| 激活参数量 | 21B | 37B |

#### 11.2.2 细粒度专家（Fine-grained Experts）

DeepSeek MoE 还将每个专家的中间维度缩小（相对于 FFN），同时增加专家数量。

```
等效参数量对比：

  Standard MoE（粗粒度）：
    N=8 专家，intermediate=4d
    每个专家参数量大，但专门化程度低

  DeepSeek MoE（细粒度）：
    N=160 专家，intermediate=d (更小)
    每个专家参数量小，但专门化程度更高
    → 更精准的 token-expert 匹配
```

#### 11.2.3 GroupedTopK 路由算法

标准 TopK 路由存在**负载不均衡**问题：部分"热门"专家被大量 token 选中，其他专家几乎闲置。

DeepSeek 的 **GroupedTopK** 在分组约束下选择专家：

```
GroupedTopK 算法（n_experts=256, n_group=8, topk_group=4, top_k=8）：

Step 1: 将 256 个专家均分为 8 组
        Group 0: Expert[0..31]
        Group 1: Expert[32..63]
        ...
        Group 7: Expert[224..255]

Step 2: 每组内选 topk_group=4 个分数最高的候选专家

Step 3: 计算每组的"组分数"= 组内 top4 分数之和
        group_score[g] = sum(top4 scores in group g)

Step 4: 选出 num_selected_groups=top_k//topk_group=2 个组分数最高的组

Step 5: 从这 2 个组的候选专家中，取最终 top_k=8 个专家

效果：强制专家选择跨越至少 2 个不同组，避免所有 token 集中在某一组
```

这与 DeepSeek V3 使用的 `noaux_tc`（无辅助损失训练）技术结合，实现了极佳的负载均衡。

### 11.3 vLLM 中的 MoE 实现

#### 11.3.1 整体架构图

```
vLLM MoE 执行栈（DeepSeek V2/V3）：

┌──────────────────────────────────────────────────────────────┐
│                   DeepseekV2MoE（Python 层）                  │
│                                                              │
│  ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │  shared_experts  │    │      SharedFusedMoE             │ │
│  │  DeepseekV2MLP   │    │  ┌─────────────────────────┐   │ │
│  │  (普通 MLP)       │    │  │    ReplicatedLinear      │   │ │
│  └──────────────────┘    │  │    (gate/router)         │   │ │
│                          │  └─────────────────────────┘   │ │
│                          │  ┌─────────────────────────┐   │ │
│                          │  │    FusedMoE (Triton)     │   │ │
│                          │  │    (路由专家计算)          │   │ │
│                          │  └─────────────────────────┘   │ │
│                          └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

Expert Parallel 通信层（多节点时）：
┌──────────────────────────────────────────────────────────────┐
│  GPU0 (Expert[0..63])    GPU1 (Expert[64..127])             │
│  GPU2 (Expert[128..191]) GPU3 (Expert[192..255])            │
│                                                              │
│  All2All 分发:                                               │
│    每个 GPU 的 token → 按路由目标 → 发送到对应 Expert GPU       │
│  All2All 汇聚:                                               │
│    每个 Expert GPU 计算结果 → 发送回原始 GPU                   │
└──────────────────────────────────────────────────────────────┘
```

#### 11.3.2 关键源码路径

```
vllm/model_executor/models/deepseek_v2.py
  ├── DeepseekV2MoE                    # MoE 层入口
  │     ├── gate (ReplicatedLinear)    # 路由器（每个 GPU 都有完整副本）
  │     ├── shared_experts             # 共享专家
  │     └── experts (SharedFusedMoE)  # 路由专家（EP 分片）
  └── DeepseekV2MLAAttention           # MLA 注意力

vllm/model_executor/layers/fused_moe/
  ├── fused_moe.py         # Triton kernel 入口，grouped GEMM
  ├── all2all_utils.py     # All2All 通信工具
  └── moe_align_block_size.py  # token 按专家排序

vllm/v1/core/sched/scheduler.py
  └── EPLB (Expert Parallel Load Balancer) 支持
```

#### 11.3.3 FusedMoE Triton Kernel 的核心思路

Python for 循环（每个专家一次 kernel）非常低效：

```
低效的朴素实现：
  for expert_id in range(256):
      tokens = x[routing_ids == expert_id]
      out[routing_ids == expert_id] = expert(tokens)

  问题：256 次 kernel launch，严重的 GPU 调度开销
```

FusedMoE 用**一次** grouped GEMM 解决：

```
FusedMoE 执行流程：

Step 1: 按 expert_id 对 token 排序（moe_align_block_size）
        [tok0→E3, tok1→E0, tok2→E3, tok3→E7]
        排序后 →
        [tok1→E0, tok0→E3, tok2→E3, tok3→E7]

Step 2: 单次 Triton grouped GEMM
        每个 Triton block 负责一个 expert 的部分 token
        所有 expert 并行计算

Step 3: 按原始顺序 scatter 回结果，加权求和

优势：
  - O(1) kernel launch（vs O(N_experts) 朴素版）
  - Triton 自动处理 tensor core 对齐和 L2 缓存
```

#### 11.3.4 DeepseekV2MoE.forward 源码解析

```python
# vllm/model_executor/models/deepseek_v2.py
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config, prefix):
        # ReplicatedLinear：路由器权重在每个 GPU 上都有完整副本
        # 因为每个 GPU 需要知道所有 token 应该去哪个专家
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
        )

        # 共享专家（在所有 GPU 上完整执行，非 EP 分片）
        self.shared_experts = DeepseekV2MLP(...)

        # 路由专家（通过 EP 分片：每个 GPU 只持有 N/ep_size 个专家）
        self.experts = SharedFusedMoE(
            num_experts=config.n_routed_experts,    # 256
            top_k=config.num_experts_per_tok,       # 8
            use_grouped_topk=True,
            num_expert_group=config.n_group,         # 8
            topk_group=config.topk_group,            # 4
            ...
        )

    def forward(self, hidden_states):
        # 1. 共享专家（非路由，每个 token 都经过）
        shared_out = self.shared_experts(hidden_states)

        # 2. FusedMoE（内含路由 + All2All + 专家计算 + 汇聚）
        router_logits = self.gate(hidden_states)
        routed_out = self.experts(hidden_states, router_logits)

        # 3. 合并
        return shared_out + routed_out
```

#### 11.3.5 Expert Parallel 的 All2All 通信

```
Expert Parallel（ep_size=4, n_experts=256）执行时序：

                 GPU0          GPU1          GPU2          GPU3
Expert划分:   E[0..63]     E[64..127]   E[128..191]  E[192..255]

t=0: 每个GPU各自计算路由逻辑（gate/router 是 ReplicatedLinear）
     GPU0: tok_A→E72, tok_B→E5, tok_C→E130 ...
     GPU1: tok_D→E200, tok_E→E64 ...

t=1: All2All 分发（每个GPU把token发给对应Expert的GPU）
     GPU0 → GPU1: tok_A（因为 E72 在 GPU1）
     GPU0 → GPU2: tok_C（因为 E130 在 GPU2）
     GPU0 保留: tok_B（因为 E5 在 GPU0）

t=2: 本地专家计算（每个GPU处理分配到本地的所有token）
     GPU1: 计算 E72(tok_A), E64(tok_E) ...

t=3: All2All 汇聚（结果发回原始GPU）
     GPU1 → GPU0: E72(tok_A) 的结果

t=4: 每个GPU在本地做加权合并
```

### 11.4 从零实现：Mini MoE

**设计目标**：用纯 Python + PyTorch 复现 DeepSeek MoE 的核心机制。

#### 11.4.1 架构设计图

```
Mini MoE 组件关系：

  ┌─────────────────────────────────────────────────────┐
  │                    MoELayer                          │
  │                                                      │
  │  input_x [T, D]                                      │
  │     │                                                │
  │     ├──────────────────────────┐                    │
  │     │                          │                    │
  │     ▼                          ▼                    │
  │  ┌──────────────┐    ┌──────────────────────────┐   │
  │  │ shared_expert│    │       router             │   │
  │  │ Expert(MLP)  │    │ TopKRouter /             │   │
  │  └──────────────┘    │ GroupedTopKRouter        │   │
  │         │            └──────────────────────────┘   │
  │         │                 │                          │
  │  shared_out         topk_ids [T, K]                  │
  │         │           topk_weights [T, K]              │
  │         │                 │                          │
  │         │            ┌────┴─────────────┐           │
  │         │            │  Expert Dispatch  │           │
  │         │            │  for e in E:     │           │
  │         │            │    tokens → e    │           │
  │         │            │    e(tokens) → o │           │
  │         │            └────┬─────────────┘           │
  │         │                 │                          │
  │         │           routed_out [T, D]                │
  │         │                 │                          │
  │         └────────── + ────┘                         │
  │                          │                          │
  │                    output [T, D]                     │
  └─────────────────────────────────────────────────────┘
```

#### 11.4.2 核心实现要点

见 `03_moe/mini_moe.py`，关键设计决策：

**Expert（专家）**：使用 SwiGLU 激活（与 DeepSeek/LLaMA 对齐）：

```python
class Expert(nn.Module):
    """单个专家：SwiGLU MLP（与 DeepSeek / LLaMA 保持一致）"""
    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**TopKRouter（路由器）**：softmax 后选 TopK，并计算辅助损失：

```python
class TopKRouter(nn.Module):
    def forward(self, x):
        logits = self.gate(x)              # [T, E]
        probs = F.softmax(logits, dim=-1)  # [T, E]

        # TopK 选择（带归一化）
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Switch Transformer 辅助损失（促进负载均衡）
        # 目标：expert_usage[e] ≈ 1/E（均匀分布）
        aux_loss = num_experts * (expert_usage * mean_prob).sum()
        return topk_probs, topk_ids, aux_loss
```

**GroupedTopKRouter**：DeepSeek 风格的分组路由：

```python
class GroupedTopKRouter(nn.Module):
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)  # [T, E]

        # 分组：[T, E] → [T, G, E/G]
        scores_grouped = probs.view(T, self.num_expert_group, self.experts_per_group)

        # 每组内选 topk_group 个候选
        group_topk_vals, group_topk_local_ids = torch.topk(
            scores_grouped, k=self.topk_group, dim=-1
        )

        # 计算组分数，选最好的组
        group_scores = group_topk_vals.sum(dim=-1)  # [T, G]
        _, selected_groups = torch.topk(group_scores, k=self.num_selected_groups, dim=-1)

        # 从选中组内的候选专家里取最终 top_k 个
        # （实现细节见 03_moe/mini_moe.py）
```

**MoELayer.forward 的 token dispatch 逻辑**：

```python
# 路由专家：token dispatch
output = torch.zeros_like(x_flat)
for expert_id in range(self.num_experts):
    expert_mask = (topk_ids == expert_id)     # [T, K] bool
    token_mask = expert_mask.any(dim=-1)       # [T] bool
    if not token_mask.any():
        continue

    tokens = x_flat[token_mask]               # [M, D]
    expert_out = self.experts[expert_id](tokens)  # [M, D]

    # 取该专家对应的路由权重
    weights = topk_weights[token_mask][expert_mask[token_mask].int().argmax(dim=-1)]
    output[token_mask] += expert_out * weights.unsqueeze(-1)
```

> **注**：这是教学版实现（O(N_experts) kernel launch）。生产环境的 FusedMoE 用 Triton grouped GEMM，只需 O(1) 次 kernel。

#### 11.4.3 运行测试

```bash
docker exec vllm python3 -m pytest /mnt/esfs/master_work/vllm-from-scratch/03_moe/ -v
```

19 个测试全部通过，覆盖：Expert 前向、TopKRouter 归一化、GroupedTopK 分组正确性、负载均衡辅助损失、共享专家、Expert Parallel 模拟。

---

## 第十二章：MLA——多头潜在注意力深度解析

### 12.1 理论背景：KV Cache 成为推理瓶颈

随着上下文长度的增加，标准 MHA（Multi-Head Attention）的 KV Cache 占用急剧膨胀：

```
标准 MHA KV Cache 分析：

  每个 token，每层，存储：
    K: [num_heads, head_dim]  = 128 × 128 = 16384 float16
    V: [num_heads, head_dim]  = 128 × 128 = 16384 float16
    合计：32768 float16 = 64 KB per token per layer

  DeepSeek V2（60层，batch=8，seq=8192）：
    8 × 8192 × 60 × 64KB ≈ 251 GB 的 KV Cache！

  这是 H100 80GB 显存的 3 倍多，根本放不下。
```

**GQA（Grouped Query Attention）** 是已有的缓解方案——减少 KV 头数，但仍然无法从根本上解决问题。

**论文参考**：

- MLA 原论文：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024
  https://arxiv.org/abs/2405.04434（Section 2.1）

- GQA 对比：Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023
  https://arxiv.org/abs/2305.13245

### 12.2 MLA 的核心数学

**关键洞察**：K 和 V 矩阵在低维空间中有冗余。可以用一个低秩潜在向量 $c_{KV}$ 来代替：

$$\text{标准 MHA：} K = X W_K,\quad V = X W_V$$

$$\text{MLA：} c_{KV} = X W^{KV}_{down},\quad K = c_{KV} W^K_{up},\quad V = c_{KV} W^V_{up}$$

其中 $c_{KV} \in \mathbb{R}^{d_{KV}}$，$d_{KV} \ll d_{model}$。

**KV Cache 只需存储 $c_{KV}$**，而非完整的 K、V！

```
维度对比（DeepSeek V2 规格）：

  MHA KV 存储：num_heads × head_dim × 2 = 128 × 128 × 2 = 32768 fp16/token/layer
  MLA KV 存储：kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576 fp16/token/layer

  节省比例：32768 / 576 = 56.9x！
```

### 12.3 解耦 RoPE：位置编码的特殊处理

**难点**：RoPE（旋转位置编码）依赖 token 的绝对位置，而低秩 $c_{KV}$ 是一个没有位置信息的向量——直接对低秩向量应用 RoPE 不正确。

**解决方案：解耦 RoPE（Decoupled RoPE）**

```
K 被分成两部分：

  K = [K_nope | K_rope]
       └─────┘   └─────┘
       从 c_KV    单独投影并
       恢复，     应用 RoPE
       无位置信息  有位置信息

KV Cache 存储：
  - c_kv  [kv_lora_rank]    ← 用于恢复 K_nope 和完整 V
  - k_rope [qk_rope_head_dim] ← 带位置信息的小向量
```

Q 侧同样解耦，使用两阶段低秩投影：

$$q = \text{LayerNorm}(x W^Q_{down}) W^Q_{up}$$

$$q = [q_{nope} | q_{rope}]$$

注意：**Q 不需要存入 KV Cache**（每次推理重新计算），因此 Q 的低秩压缩不节省 KV Cache，只节省计算量。

### 12.4 vLLM 中的 MLA 实现

#### 12.4.1 整体架构图

```
vLLM MLA 执行路径：

Prefill 阶段（处理 prompt 的所有 token）：
  ┌──────────────────────────────────────────────────────────────┐
  │  hidden_states [T_prompt, D]                                 │
  │         │                                                    │
  │    ┌────┴──────────┐    ┌──────────────────────────┐         │
  │    │   Q 计算       │    │   KV 压缩                 │         │
  │    │ q_a_proj      │    │ kv_a_proj_with_mqa        │         │
  │    │ q_a_layernorm │    │  → c_kv [T, kv_lora_rank]│         │
  │    │ q_b_proj      │    │  → k_rope [T, rope_dim]  │         │
  │    └────┬──────────┘    └────────┬─────────────────┘         │
  │         │                        │                           │
  │     Q [T, H, qk_dim]         存入 KV Cache ←──────────────  │
  │         │                        │                           │
  │    ┌────┴────────────────────────┴──┐                        │
  │    │      FlashAttention            │                        │
  │    │  (kv_b_proj 展开 c_kv → K,V)  │                        │
  │    └────────────────────────────────┘                        │
  └──────────────────────────────────────────────────────────────┘

Decode 阶段（每次生成 1 个 token）：
  ┌──────────────────────────────────────────────────────────────┐
  │  hidden_states [1, D]                                        │
  │         │                                                    │
  │    Q 计算（同 Prefill）                                       │
  │         │                                                    │
  │    从 KV Cache 读取历史 c_kv, k_rope                          │
  │         │                                                    │
  │    kv_b_proj(c_kv) → K_nope, V    （实时展开）               │
  │    apply_rope(k_rope) → K_rope                               │
  │    K = [K_nope | K_rope]                                     │
  │         │                                                    │
  │    PagedAttention（使用 block table 读取历史块）               │
  └──────────────────────────────────────────────────────────────┘
```

#### 12.4.2 关键源码解析

```python
# vllm/model_executor/models/deepseek_v2.py
class DeepseekV2MLAAttention(nn.Module):
    def __init__(self, ..., q_lora_rank, kv_lora_rank, ...):

        # Q 侧：两阶段低秩投影
        self.q_a_proj = Linear(hidden_size, q_lora_rank)          # 压缩
        self.q_a_layernorm = RMSNorm(q_lora_rank)
        self.q_b_proj = ColumnParallelLinear(q_lora_rank, num_heads * qk_head_dim)  # 展开

        # KV 侧：低秩压缩（输出存入 KV Cache）
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,  # c_kv + k_rope
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)

        # KV 展开（计算时，从 c_kv 恢复完整 K_nope + V）
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),  # K_nope + V
        )

        # 输出投影
        self.o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size)
```

**KV Cache 物理布局**（MLA 与 MHA 的关键差异）：

```
MHA 的 KV Cache（每个 block）：
  k_cache: [block_size, num_heads, head_dim]
  v_cache: [block_size, num_heads, head_dim]

MLA 的 KV Cache（每个 block）：
  kv_cache: [block_size, kv_lora_rank + qk_rope_head_dim]
  ^^^^^^^^ 只有一个 tensor，维度远小于 MHA！

vllm/v1/attention/backends/mla/utils.py (MLACommonImpl)
  - 处理 MLA 特有的 prefill/decode 路径分支
  - kv_b_proj 在 attention 计算时实时展开（不提前存储完整 K,V）
```

#### 12.4.3 KV Cache 节省量化

```
参数                       MHA           MLA
─────────────────────────────────────────────────
num_heads                  128           128
head_dim                   128           128
kv_lora_rank               -             512
qk_rope_head_dim           -             64
─────────────────────────────────────────────────
每 token 每层存储量        32768 fp16    576 fp16
─────────────────────────────────────────────────
节省比例                   1x            56.9x !!!
─────────────────────────────────────────────────
8192 token, 60 层, batch=1 (GB)
  MHA：  8192 × 60 × 32768 × 2B = 32.2 GB
  MLA：  8192 × 60 ×   576 × 2B =  0.6 GB
```

### 12.5 从零实现：Mini MLA

**设计目标**：端到端实现 MLA 的 Prefill + Decode 路径，包括解耦 RoPE 和 KV Cache 存储。

#### 12.5.1 架构设计图

```
Mini MLA 数据流（Prefill）：

hidden_states [B, T, D]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
  ┌──────────────────┐                    ┌──────────────────────┐
  │    Q 计算         │                    │    KV 低秩压缩        │
  │ q_a_proj [B,T,L] │                    │ kv_a_proj_with_mqa   │
  │ q_a_layernorm    │                    │ [B, T, lora+rope]    │
  │ q_b_proj         │                    └──────────┬───────────┘
  │ [B,T,H,nope+rope]│                               │
  └──────────────────┘                    ┌──────────┴───────────┐
        │                                 │                      │
        ├──── q_nope [B,T,H,nope]        c_kv                k_rope
        └──── q_rope [B,T,H,rope]    [B,T,lora]           [B,T,rope]
                 │                        │                      │
                 │           ┌────────────┘                      │
                 │           ▼                                    │
                 │    kv_b_proj(c_kv)                           RoPE
                 │    → k_nope [B,S,H,nope]                      │
                 │    → v     [B,S,H,v_dim]      k_rope_out [B,S,H,rope]
                 │           │                        │
        RoPE     │           └──────────────┐         │
           │     │                          │         │
        q_rope_out                   k_full = [k_nope | k_rope_out]
                 │                          │
                 └─────────────────────────▶│
                                           │
                            q_full = [q_nope | q_rope_out]
                                           │
                                    ┌──────┴──────┐
                                    │  Attention   │
                                    │  Q @ K^T     │
                                    │  softmax     │
                                    │  @ V         │
                                    └──────┬──────┘
                                           │
                                    o_proj → output [B, T, D]

KV Cache 存储（只存这两个）：
  ┌─────────────┐  ┌───────────┐
  │ c_kv [B,T,L]│  │k_rope[B,T,r]│
  └─────────────┘  └───────────┘
```

#### 12.5.2 关键实现细节

见 `04_mla/mini_mla.py`，特别注意：

**RoPE 的应用方式**（修复了非连续张量的问题）：

```python
# 对 q_rope [B, T, H, rope] 应用 RoPE
# 广播: cos_q [B, T, 1, rope] → [B, T, H, rope]
cos_q = cos_q.unsqueeze(2)  # [B, T, 1, rope]
sin_q = sin_q.unsqueeze(2)
q_rope_out = q_rope * cos_q + rotate_half(q_rope) * sin_q  # [B, T, H, rope]

# 对 k_rope [B, S, rope] 扩展到多头后应用 RoPE
k_rope_expand = k_rope.unsqueeze(2).expand(-1, -1, H, -1)  # [B, S, H, rope]
cos_k = cos_k.unsqueeze(2)  # [B, S, 1, rope]
sin_k = sin_k.unsqueeze(2)
k_rope_out = k_rope_expand * cos_k + rotate_half(k_rope_expand) * sin_k
```

> **重要**：`k_rope` 是单头（k_rope per token，与头无关），展开时用 `.expand()` 而非 `.repeat()`，避免内存复制。

**Causal Mask 的正确计算**：

```python
# Prefill 时，T=prompt_len，S=total_seq_len（含历史）
# 新 token 在序列末尾，diagonal 应为 S-T+1
if T > 1:
    causal_mask = torch.triu(
        torch.full((T, S), float('-inf'), device=device),
        diagonal=S - T + 1   # 不是1，因为 S > T（decode步有历史）
    )
```

#### 12.5.3 运行测试

```bash
docker exec vllm python3 -m pytest /mnt/esfs/master_work/vllm-from-scratch/04_mla/ -v
```

13 个测试全部通过：RoPE 正交性、Prefill 形状验证、Decode 追加 KV Cache、因果掩码验证、梯度回传、KV Cache 节省量化（>5x）。

---

## 第十三章：PD 分离——Prefill-Decode 解耦架构

### 13.1 理论背景：Prefill 和 Decode 的计算特性差异

在同一个 vLLM 实例中混合处理 Prefill 和 Decode，会产生严重的干扰：

```
Prefill 的特点：
  - Compute-bound：大量矩阵乘法，GPU 算力利用率高
  - 长序列处理：一次处理 1000-8000 tokens
  - 高延迟可接受：用户在等待第一个 token

Decode 的特点：
  - Memory-bandwidth-bound：每次只生成 1 个 token，
    但需要从 GPU 内存读取完整 KV Cache
  - 延迟敏感：用户实时看到流式输出
  - 吞吐优先：需要并发服务多个对话

混合批次的问题（示例）：

  t=0: 调度 [Prefill(req_A, 4096 tokens)]
       → GPU 花 2s 做 prefill
       → req_B、req_C 的 decode 被阻塞 2s！
       → TTFT(req_B) 增加 2s，用户体验很差

  t=2: 调度 [Decode(req_B), Decode(req_C), Decode(req_D)]
       → 3个 decode 请求，每步 4ms
```

**论文参考**：

- PD 分离思路最早系统化：Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving*, OSDI 2024
  https://arxiv.org/abs/2401.09670

- Mooncake（基于 RDMA 的 KV 传输）：Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, 2024
  https://arxiv.org/abs/2407.00079

### 13.2 PD 分离的系统架构

```
PD 分离集群架构：

┌──────────────────────────────────────────────────────────────────┐
│                       生产 LLM 服务集群                            │
│                                                                  │
│  ┌───────────────────┐           ┌─────────────────────────────┐ │
│  │   Prefill 节点集群 │           │    Decode 节点集群            │ │
│  │                   │           │                             │ │
│  │  [P-Node 0]       │  KV Cache │  [D-Node 0] ← KV ←         │ │
│  │  [P-Node 1]       │ ─────────→│  [D-Node 1] ← KV ←         │ │
│  │  [P-Node 2]       │  (RDMA)   │  [D-Node 2] ← KV ←         │ │
│  │                   │           │                             │ │
│  │  特点：           │           │  特点：                      │ │
│  │  - 高算力 GPU     │           │  - 高内存带宽 GPU             │ │
│  │  - 专注 Prefill   │           │  - 持续流式输出               │ │
│  │  - Batch 大型请求 │           │  - 服务 M 个并发对话          │ │
│  └───────────────────┘           └─────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   全局元数据服务器（类 Mooncake Store）                     │   │
│  │   - 记录哪个 Prefill 节点有哪些 block                       │   │
│  │   - 路由请求到合适的节点                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 13.3 KV Cache 跨节点传输

#### 13.3.1 RDMA 单边读取（One-sided RDMA READ）

```
为什么用 RDMA 而不是 TCP/IP？

  TCP/IP：
    发送端：CPU 拷贝 → 内核 → 网卡 → 传输
    接收端：网卡 → 内核 → CPU 拷贝 → 用户空间
    延迟：10-100μs，带宽 10-100Gbps，CPU 介入

  RDMA (Remote Direct Memory Access)：
    发送端：GPU 内存直接标记为可远程访问
    接收端：直接从发送端 GPU 内存读取（无 CPU 介入）
    延迟：0.5-2μs，带宽 200-800Gbps (InfiniBand HDR/NDR)
    CPU 零介入

RDMA One-sided READ 流程：

  D-Node (读取方)                      P-Node (数据源)
       │                                    │
       │── 1. 注册本地接收 buffer ──────────│
       │── 2. 发送 RDMA READ 请求 ─────────→│
       │                              （无需 CPU 干预）
       │←─ 3. DMA 直接传输 KV blocks ───────│
       │── 4. 本地 buffer 已填充 ────────── │
       │── 5. 通知调度器，开始 Decode ───── │
```

#### 13.3.2 vLLM 的 KVConnector 接口

vLLM V1 提供标准化的 KV 传输接口：

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py
# 以及各具体实现

class KVConnectorBase_V1:
    """
    KV Cache 传输抽象基类

    Prefill 节点（send side）：
      1. 模型推理完成后，调用 send_kv_caches
      2. 将计算好的 KV blocks 发送给 Decode 节点

    Decode 节点（recv side）：
      1. 在推理前，调用 recv_kv_caches
      2. 等待 KV blocks 从 Prefill 节点到达
    """

    # Prefill 侧：推理完成后发送
    def send_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states,
    ) -> None: ...

    # Decode 侧：推理前接收
    def recv_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, bool]: ...
    # 返回 (hidden_states, bypass_model)
    # bypass_model=True 意味着直接跳过模型推理，用传来的 KV 做 decode
```

#### 13.3.3 Mooncake 实现（基于 RDMA 的高速传输）

Mooncake 实现了 KVConnectorBase_V1，将 KV Cache 以 block 为单位存入分布式 KV Store：

```python
# 简化的 Mooncake Connector 工作逻辑

class MooncakeConnector(KVConnectorBase_V1):
    """
    通过 Mooncake Store（RDMA 分布式 KV 存储）传输 KV Cache

    关键特性：
    - 按 block_hash 存储：自动支持 Prefix Cache（相同前缀不重复传输）
    - 异步传输：不阻塞模型推理
    - RDMA 零拷贝：GPU→网卡→GPU，无 CPU 介入
    """

    def send_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches, hidden_states
    ):
        # 遍历需要发送的 blocks
        for layer_id, kv_cache in enumerate(kv_caches):
            for block_hash, physical_block_id in self._blocks_to_send(model_input):
                k = kv_cache[0][physical_block_id]  # [block_size, H, head_dim]
                v = kv_cache[1][physical_block_id]

                # 按 block_hash 存入 Mooncake Store（RDMA 注册内存）
                key = f"kv/{block_hash}/layer{layer_id}"
                self.store.put(key, torch.stack([k, v]))
                # ↑ 这步触发 RDMA 注册，D-Node 可以直接远程读取

    def recv_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches
    ):
        # D-Node：从 Mooncake Store 拉取 KV（触发 RDMA READ）
        for layer_id, kv_cache in enumerate(kv_caches):
            for block_hash, local_block_id in self._blocks_to_recv(model_input):
                key = f"kv/{block_hash}/layer{layer_id}"
                kv_data = self.store.get(key)  # RDMA READ

                k, v = kv_data[0], kv_data[1]
                kv_cache[0][local_block_id] = k
                kv_cache[1][local_block_id] = v

        # bypass_model=True：KV 已就绪，直接做 decode attention，不需要重新 prefill
        return None, True
```

#### 13.3.4 端到端请求流程

```
时序图（PD 分离完整流程）：

  时刻    Prefill Node (P-Node)        Decode Node (D-Node)
  ────────────────────────────────────────────────────────────
  t=0:    接收请求 req_A（4096 tokens）
          Scheduler 分配 KV blocks

  t=0~2s: Prefill Forward Pass
          计算并填充 KV Cache blocks

  t=2s:   KVConnector.send_kv_caches()
          → 按 block_hash 存入 Mooncake  → 接收通知（block_hash 列表）
          → 异步 RDMA 传输开始           → 发起 RDMA READ

  t=2s~   P-Node 开始下一个请求          等待 RDMA 完成（~100ms）
  t=2.1s: （P-Node 已不关心 req_A）
                                         RDMA 完成，KV blocks 就绪
                                         Scheduler: req_A → Decode 队列
  t=2.1s~                                Decode Forward（逐 token）
  t=2.6s:                                生成 128 tokens 完成
  ────────────────────────────────────────────────────────────

关键优势：
  1. P-Node 在 t=2s 后立即可以服务下一个请求（吞吐 ↑）
  2. D-Node 可同时服务多个 req 的 decode（延迟 ↓）
  3. P-Node 和 D-Node 可以用不同硬件配置
```

### 13.4 从零实现：全局 KV Cache 池（Mooncake 风格）

#### 13.4.1 架构设计图

```
全局 KV Cache 池（06_global_prefix_cache/global_kv_pool.py）：

┌─────────────────────────────────────────────────────────────┐
│                   SimulatedCluster                           │
│                                                             │
│  ┌────────────────┐          ┌─────────────────────────┐   │
│  │  Prefill Nodes │          │     Decode Nodes         │   │
│  │  ┌──────────┐  │          │  ┌──────────────────┐   │   │
│  │  │MooncakeCon│ │          │  │  MooncakeConnector│   │   │
│  │  │nector    │ │          │  │  get_matched_tokens│   │   │
│  │  │publish_kv│ │          │  │  wait_for_kv      │   │   │
│  │  └──────────┘  │          │  └──────────────────┘   │   │
│  └────────────────┘          └─────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            GlobalMetadataServer                      │   │
│  │   block_hash → {node_id, token_ids}                 │   │
│  │   LRU 淘汰 + 命中率统计 + 并发安全（RLock）           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            TransferEngine                            │   │
│  │   异步 worker thread 模拟 RDMA 传输                   │   │
│  │   submit_transfer() → transfer_id                   │   │
│  │   wait(tid, timeout) → TransferResult               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 13.4.2 实现层次

**Layer 1：块哈希（链式 hash，防止不同前缀的相同块碰撞）**：

```python
def compute_block_hashes(token_ids: list[int], extra_key=None) -> list[int]:
    """
    链式哈希：每个块的哈希依赖前一个块的哈希（parent hash）

    这确保：相同内容但不同前缀的块，哈希值不同
    （防止跨请求的错误 prefix cache 命中）
    """
    prev_hash = hash(extra_key) if extra_key is not None else 0
    hashes = []
    for i in range(len(token_ids) // BLOCK_SIZE):
        block_tokens = tuple(token_ids[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
        h = hash((prev_hash, block_tokens))  # 链式！
        hashes.append(h)
        prev_hash = h
    return hashes
```

**Layer 2：GlobalMetadataServer（分布式元数据，etcd 模拟）**：

```python
class GlobalMetadataServer:
    def publish(self, block_hash, node_id, token_ids, max_blocks_per_node=10000):
        with self._lock:  # 线程安全
            self._registry[block_hash] = KVBlockMeta(block_hash, node_id, token_ids)
            self._node_blocks[node_id].add(block_hash)
            # LRU 淘汰：超出容量时驱逐最久未使用的块
            if len(self._node_blocks[node_id]) > max_blocks_per_node:
                self._evict_lru(node_id)

    def query_prefix(self, block_hashes: list[int]):
        """前缀匹配查询：遇到第一个 miss 就停止"""
        matched = []
        for h in block_hashes:
            if h not in self._registry:
                break  # 前缀必须连续
            matched.append(self._registry[h])
            self._hit_count += 1
        self._query_count += len(block_hashes)  # 按块计数，不是按次计数
        return len(matched), matched
```

**Layer 3：TransferEngine（RDMA 传输模拟）**：

```python
class TransferEngine:
    """
    用 Python threading 模拟 RDMA 传输
    - 机架内：~200μs 延迟
    - 跨机架：~1ms 延迟
    """
    def submit_transfer(self, src_node_id, block_hashes, callback=None):
        transfer_id = str(uuid.uuid4())
        # 在后台线程中模拟传输延迟
        threading.Thread(
            target=self._do_transfer,
            args=(transfer_id, src_node_id, block_hashes, callback),
            daemon=True,
        ).start()
        return transfer_id

    def _do_transfer(self, tid, src_node_id, block_hashes, callback):
        # 模拟网络延迟
        intra_rack = src_node_id // 4 == self.node_id // 4
        latency = 0.0002 if intra_rack else 0.001  # 200μs vs 1ms
        bytes_transferred = len(block_hashes) * KV_BLOCK_BYTES
        time.sleep(latency + bytes_transferred / RDMA_BANDWIDTH_BPS)

        result = TransferResult(tid, success=True, bytes_transferred=bytes_transferred)
        self._results[tid] = result
        if callback:
            callback(result)
```

**Layer 4：MooncakeConnector（vLLM worker 接口）**：

```python
class MooncakeConnector:
    def get_num_new_matched_tokens(self, request_id, block_hashes):
        """
        查询全局 KV Pool，返回命中的 token 数
        如果命中在远端节点，自动触发 RDMA 传输
        """
        num_matched, metas = self.meta_server.query_prefix(block_hashes)
        if num_matched == 0:
            return 0, False

        # 判断是本地命中还是远端命中
        src_node = metas[0].node_id
        if src_node == self.node_id:
            return num_matched * BLOCK_SIZE, False  # 本地，无需传输
        else:
            # 远端命中，触发 RDMA 传输
            matched_hashes = [m.block_hash for m in metas]
            tid = self.transfer_engine.submit_transfer(src_node, matched_hashes)
            self._pending_transfers[request_id] = tid
            return num_matched * BLOCK_SIZE, True  # 需要传输

    def wait_for_kv(self, request_id, timeout=30.0):
        """等待 RDMA 传输完成"""
        tid = self._pending_transfers.pop(request_id, None)
        if tid is None:
            return True  # 本地命中，无需等待
        result = self.transfer_engine.wait(tid, timeout=timeout)
        return result is not None and result.success
```

#### 13.4.3 运行测试

```bash
docker exec vllm python3 -m pytest /mnt/esfs/master_work/vllm-from-scratch/06_global_prefix_cache/ -v
```

34 个测试全部通过：块哈希链式依赖、元数据服务器 LRU 淘汰、并发发布线程安全、RDMA 传输模拟、跨节点缓存共享、命中率统计。

---

## 第十四章：vLLM V1 引擎——整合全局视角

### 14.1 V0 vs V1：架构演进

vLLM 在 V1 版本进行了一次根本性重构，解决 V0 的架构局限：

```
V0 架构（旧）：

  [用户请求] → AsyncLLMEngine
                    │ (blocking)
               LLMEngine
                    │
               EngineCore ─── Scheduler (Python)
                    │
               ModelExecutor ─── Worker[GPU]
                    │                  │
               (单进程，同步)      Attention/MoE/...

V0 问题：
  - Engine 和 Model 在同一进程，GIL 竞争
  - Scheduler 每步都在 Python 层做大量工作
  - 难以支持 CUDA Graph（形状变化）
  - 投机解码、PD 分离等功能难以插入
```

```
V1 架构（新）：

  [用户请求] → AsyncLLMEngine （非阻塞前端，独立进程）
                    │ ZeroMQ (IPC)
               EngineCore 进程 ─── Scheduler（完全重写）
                    │                   │
                    │              KVCacheManager（显式管理）
                    │
               ModelExecutorV1 → GPU Worker 进程群
                    │                  │
               InputBatch              │
               CUDA Graph Capture    Attention Backends
               Sampler                │
               RejectionSampler      MLA / MHA / Flash
               (spec decode)         MoE / FusedMoE
```

**V1 主要改进**：

| 方面 | V0 | V1 |
|------|----|----|
| 进程模型 | 单进程混合 | Engine + Worker 多进程分离 |
| 通信 | 函数调用 | ZeroMQ IPC（高效序列化） |
| 调度器 | 简单 FCFS | 支持 Chunked Prefill / Spec Decode / PD 分离 |
| KV 管理 | 隐式 | 显式 BlockManager，Python 层管理 |
| CUDA Graph | 有限支持 | Prefill / Decode 分离的 CUDA Graph |
| 投机解码 | 插件式 | 一等公民，Scheduler 原生支持 |

### 14.2 V1 完整请求生命周期

```
完整数据流（从请求到 token 输出）：

用户侧：
  engine.generate("Hello, world", sampling_params)
       │
       ▼
AsyncLLMEngine（前端进程）
  - 管理并发请求
  - 流式返回 token
  - 处理超时/取消
       │ ZeroMQ PUSH
       ▼
EngineCore（独立进程）
       │
  ┌────┴────────────────────────────────────────┐
  │              推理主循环（每步）               │
  │                                             │
  │  1. Scheduler.schedule()                    │
  │     - 选择本步要处理的请求                   │
  │     - 分配/释放 KV Cache blocks              │
  │     - 决定 chunked prefill 的 chunk 大小     │
  │     - 投机解码：决定 draft token 数           │
  │     → SchedulerOutput                       │
  │              │                              │
  │  2. ModelExecutor.execute_model()           │
  │     - 构建 InputBatch（token_ids, pos, ...)  │
  │     - 推理（可能 CUDA Graph 加速）           │
  │     - Sampling（greedy/top-p/top-k）         │
  │     - 如果投机解码：RejectionSampling        │
  │     → ModelRunnerOutput                     │
  │              │                              │
  │  3. Scheduler.update_from_output()          │
  │     - 更新 num_computed_tokens               │
  │     - 完成的请求：释放 KV blocks             │
  │     - 更新 prefix cache hash 表              │
  │     - 投机解码：统计接受率                   │
  │              │                              │
  │  4. AsyncLLMEngine 读取输出，流式返回         │
  └─────────────────────────────────────────────┘
```

### 14.3 关键子系统详解

#### 14.3.1 Scheduler（调度器）

```
Scheduler 状态机：

  新请求 → [waiting 队列]
               │
               │ (有足够 KV blocks)
               ▼
           [running 队列]  ←── decode 步持续在此
               │
               │ (KV blocks 不足，发生抢占)
               ▼
           [preempted 队列] → swap-out KV 到 CPU
               │
               │ (CPU 内存有空间时)
               ▼
           [swapped 队列]
               │
               │ (GPU 内存释放后，swap-in)
               ▼
           [running 队列]

关键配置路径：
  vllm/v1/core/sched/scheduler.py
    ├── Scheduler.schedule()         # 主调度逻辑
    ├── _schedule_prefills()          # Chunked Prefill 处理
    ├── _schedule_running()           # Decode 请求管理
    └── _preempt()                    # 抢占逻辑
```

#### 14.3.2 CUDA Graph 加速

```
CUDA Graph 的作用：

  普通执行：每步都要 CPU → GPU 发送 kernel 启动命令
    开销：~100μs per step（对于短 decode 步影响显著）

  CUDA Graph：把一整个前向传播录制成 graph，回放时零 CPU 开销
    开销：~10μs per step（10x 加速）

  V1 的 CUDA Graph 策略：
    - Decode 步（固定形状）：使用 CUDA Graph
    - Prefill 步（可变长度）：不使用（或用 padding 对齐）
    - 关键文件：vllm/v1/worker/gpu_model_runner.py
                → capture_model()
                → _dummy_run()
```

#### 14.3.3 Spec Decode 在 V1 中的集成

```
投机解码调度（Scheduler 原生支持）：

  ┌──────────────────────────────────────────────────┐
  │  step N：Draft 模型生成 K 个 draft token          │
  │    EAGLE / ngram / 小模型                         │
  │                                                  │
  │  step N+K：Target 模型验证 K+1 个 token          │
  │    并行处理 draft tokens（= chunked prefill）     │
  │                                                  │
  │  RejectionSampler：                              │
  │    accepted_tokens = rejection_sample(            │
  │        draft_probs, target_probs, draft_tokens   │
  │    )                                             │
  │    # 保证分布等价于纯 target 模型                 │
  └──────────────────────────────────────────────────┘

关键文件：
  vllm/v1/spec_decode/eagle.py          # EAGLE draft 模型
  vllm/v1/sample/rejection_sampler.py   # rejection sampling
  vllm/v1/core/sched/scheduler.py       # spec decode 调度逻辑
```

### 14.4 生产配置速查

```
场景1：在线聊天（低延迟优先，TTFT < 200ms）
  vllm serve model \
    --enable-chunked-prefill \      # 减少 decode 等待
    --speculative-model eagle \     # 加速单序列 decode
    --max-num-batched-tokens 2048 \ # 限制每步处理量
    --gpu-memory-utilization 0.9

场景2：离线批处理（吞吐优先）
  vllm serve model \
    --max-num-seqs 256 \            # 大批次
    --enable-chunked-prefill \      # GPU 利用率
    --disable-log-requests \        # 减少 I/O 开销
    --gpu-memory-utilization 0.95

场景3：长上下文（128K+ tokens）
  vllm serve model \
    --max-model-len 131072 \
    --enable-prefix-caching \       # 系统提示词复用
    --kv-cache-dtype fp8 \          # 压缩 KV Cache
    --tensor-parallel-size 4        # 多卡分担内存

场景4：DeepSeek V3（671B MoE）
  vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \      # Expert Parallel
    --trust-remote-code \
    --gpu-memory-utilization 0.95

关键参数文件：
  vllm/config.py
  vllm/v1/core/sched/scheduler.py（BLOCK_SIZE, token_budget）
```

---

## 第十五章：Mini vLLM——完整整合

### 15.1 Mini vLLM 架构总览

经过前14章的深入分析，我们的 **Mini vLLM**（`05_mini_vllm/mini_vllm.py`）将所有核心组件串联起来：

```
Mini vLLM 完整架构图：

  ┌────────────────────────────────────────────────────────────────┐
  │                        MiniVLLM 引擎                            │
  │                                                                │
  │  generate(prompts)                                             │
  │       │                                                        │
  │       ▼                                                        │
  │  ┌──────────────────────────────────────────────────────────┐  │
  │  │                   推理主循环                               │  │
  │  │                                                          │  │
  │  │  ┌─────────────────────────┐                             │  │
  │  │  │       Scheduler         │                             │  │
  │  │  │  - waiting / running 队列│                             │  │
  │  │  │  - BlockAllocator        │                             │  │
  │  │  │    (LRU + Prefix Cache)  │                             │  │
  │  │  │  - Chunked Prefill 支持  │                             │  │
  │  │  └─────────────┬───────────┘                             │  │
  │  │                │ SchedulerOutput                          │  │
  │  │                ▼                                          │  │
  │  │  ┌─────────────────────────┐                             │  │
  │  │  │    MiniTransformer       │                             │  │
  │  │  │  ┌───────────────────┐  │                             │  │
  │  │  │  │  Embedding Layer  │  │                             │  │
  │  │  │  └────────┬──────────┘  │                             │  │
  │  │  │           │             │                             │  │
  │  │  │  for layer in layers:   │                             │  │
  │  │  │  ┌────────▼──────────┐  │                             │  │
  │  │  │  │  Attention         │  │  ← 读写 KV Cache           │  │
  │  │  │  │  (MHA 简化版)      │  │    按 block_table 寻址      │  │
  │  │  │  └────────┬──────────┘  │                             │  │
  │  │  │  ┌────────▼──────────┐  │                             │  │
  │  │  │  │  FFN / MoE       │  │                             │  │
  │  │  │  └────────┬──────────┘  │                             │  │
  │  │  │           │             │                             │  │
  │  │  │  ┌────────▼──────────┐  │                             │  │
  │  │  │  │  LM Head (logits) │  │                             │  │
  │  │  │  └───────────────────┘  │                             │  │
  │  │  └─────────────┬───────────┘                             │  │
  │  │                │ logits                                   │  │
  │  │                ▼                                          │  │
  │  │  ┌─────────────────────────┐                             │  │
  │  │  │       Sampler            │                             │  │
  │  │  │  - Greedy / Temperature  │                             │  │
  │  │  │  - Top-P / Top-K         │                             │  │
  │  │  └─────────────┬───────────┘                             │  │
  │  │                │ next_tokens                              │  │
  │  │                ▼                                          │  │
  │  │         更新请求状态 → 循环                                │  │
  │  └──────────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────────┘

外部 KV Cache（GPU 内存）：
  kv_caches: List[(k_cache, v_cache)]  每层一个
  k_cache: [num_blocks, block_size, num_heads, head_dim]
  v_cache: [num_blocks, block_size, num_heads, head_dim]
```

### 15.2 各组件协作数据流

#### 15.2.1 Prefill 步

```
时序（Prefill：处理 prompt）：

  用户: "Tell me about..."（100 tokens）

  Scheduler:
    1. 检查 Prefix Cache：哈希前几个 block
    2. 发现 block_0 命中（节省重算）
    3. 分配 block_1..block_6（新块，100 tokens / 16 = 7块）
    4. 生成 block_table = [0 (复用), 23, 45, 67, 89, 100, 112]
    5. num_tokens_to_compute = 100 - 16 = 84（跳过已缓存的）

  compute_slot_mapping:
    [16, 17, 18, ..., 99] → 实际物理 slot 地址

  MiniTransformer.forward(input_ids[16:100], positions[16:100]):
    - 每层 attention：
        Q: 当前 token 的 Q
        K, V: 写入 kv_cache 对应物理 slot
        Attention: Q @ K^T（含历史 block）

  Sampler: 取 logits[-1]（最后一个 token 的 logit）→ next_token
```

#### 15.2.2 Decode 步

```
时序（Decode：逐 token 生成）：

  Scheduler:
    1. req 已在 running，只需处理 1 个 token
    2. 分配下一个 slot（如果当前 block 满，分配新块）
    3. num_tokens_to_compute = 1

  MiniTransformer.forward(input_ids=[next_token], positions=[seq_len]):
    - 每层 attention：
        Q: 当前 token 的 Q（1个）
        从 kv_cache 读取所有历史 K, V
        Attention: 标准 decode attention
        将当前 K, V 写入下一个空 slot

  Sampler: logits[0] → next_token

  循环，直到生成 EOS 或达到 max_new_tokens
```

#### 15.2.3 BlockAllocator 与 Prefix Cache

```
BlockAllocator 关键逻辑（mini_vllm.py）：

  allocate(request_id, token_ids):
    1. 计算链式哈希：h0, h1, h2 = compute_hashes(token_ids)

    2. 前缀复用：
       if h0 in cache:
           blocks[0] = cache[h0]  # 直接复用，ref_count += 1
       else:
           blocks[0] = free_list.pop()  # 分配新块

    3. 记录 block_table：
       block_table[request_id] = [physical_id_0, physical_id_1, ...]

  free(request_id):
    for block in block_table[request_id]:
        block.ref_count -= 1
        if block.ref_count == 0:
            # 归还到 LRU 缓存（而非立即释放）
            lru_cache.put(block.hash, block)
```

### 15.3 关键设计决策与 trade-off

| 设计决策 | Mini vLLM 的选择 | 真实 vLLM 的做法 | 原因 |
|---------|----------------|----------------|------|
| Attention 实现 | 纯 PyTorch SDPA | FlashAttention / PagedAttention kernel | 简单清晰，但无法处理 padding |
| KV Cache 布局 | `[blocks, block_size, heads, head_dim]` | 同 | 符合 PagedAttention 论文 |
| Block Table | Python list | CUDA tensor，直接传给 kernel | 避免 GPU-CPU 同步开销 |
| Decode Attention | 读取连续 slice | 真正的 block table 寻址 | 简化实现，功能等价 |
| 调度器 | FCFS + 简单 Chunked Prefill | 复杂抢占、swap-in/out | 覆盖核心概念 |

### 15.4 性能基准

在 NVIDIA GPU 上运行 Mini vLLM 的基准测试（`test_mini_vllm.py`）：

```
配置：
  模型：MiniTransformer(vocab=1000, hidden=256, layers=4, heads=4)
  序列：batch=4, prompt_len=50, max_new_tokens=20
  硬件：RTX PRO 6000（CUDA）

结果（示意，以实际运行为准）：
  Throughput：~400-500 tokens/s
  BlockAllocator：分配/释放延迟 < 1μs
  Prefix Cache：相同前缀请求节省 ~50% prefill 计算
```

### 15.5 全系列回顾：各章技术点与收益

| 章节 | 核心技术 | 关键收益 |
|------|---------|---------|
| 第1-2章 | vLLM 整体架构、设计哲学 | 理解推理系统为何如此设计 |
| 第3章 | Paged Attention、Block Table | 内存碎片 60-80% → <4% |
| 第4章 | KV Cache 读写、swap-in/out | 超过 GPU 内存的序列可以运行 |
| 第5章 | 完整推理路径 | 端到端流程打通 |
| 第6章 | 单机 Prefix Cache（LRU + 链式哈希）| 相同前缀：跳过全部重复 Prefill |
| 第7章 | 全局 KV Pool（Mooncake RDMA）| 跨节点共享 Prefix Cache |
| 第8章 | Scheduler（FCFS + 抢占 + 连续批处理）| GPU 利用率从 40% → 85%+ |
| 第9章 | 投机解码（EAGLE + Rejection Sampling）| Decode 速度 1.5-5x |
| 第10章 | Chunked Prefill（Sarathi）| TTFT 降低 20-40% |
| **第11章** | **DeepSeek MoE（GroupedTopK + EP）** | **计算量降低 5-10x（相比同参数稠密模型）** |
| **第12章** | **MLA（低秩 KV + 解耦 RoPE）** | **KV Cache 节省 7-57x** |
| **第13章** | **PD 分离（KVConnector + RDMA）** | **Decode 节点吞吐独立扩展** |
| **第14章** | **vLLM V1 引擎全局架构** | **理解生产级推理系统设计** |
| **第15章** | **Mini vLLM 完整整合** | **端到端验证所有技术点** |

---

## 结语：LLM 推理系统的设计哲学

回顾整个系列，vLLM 的每一项核心技术都在针对一个具体瓶颈做精准的工程突破：

```
瓶颈                    解决方案                  效果
────────────────────────────────────────────────────────────────
GPU 内存碎片           Paged Attention           碎片率 <4%
重复计算               Prefix Cache              相同前缀零计算
Decode 延迟             投机解码                  1.5-5x 加速
Prefill 干扰 Decode    Chunked Prefill            TTFT -20-40%
参数量 vs 计算量       MoE（稀疏激活）            计算量 ÷10
KV Cache 内存          MLA（低秩压缩）            KV ÷7~57
跨节点数据孤岛         Mooncake RDMA              全局 KV 共享
────────────────────────────────────────────────────────────────
```

这些技术共同构成了现代 LLM 推理系统的核心骨架。理解了这些机制，你就掌握了现代 LLM 推理系统设计的核心密码——也为参与 vLLM、SGLang、TensorRT-LLM 等开源项目打下了坚实基础。

---

## 配套代码运行

```bash
# 运行全套测试（113 个用例）
docker exec -w /mnt/esfs/master_work/vllm-from-scratch vllm \
  python3 -m pytest 01_paged_attention/ 02_kvcache/ 03_moe/ \
                    04_mla/ 05_mini_vllm/ 06_global_prefix_cache/ -v

# 运行各模块演示
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/03_moe/mini_moe.py
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/04_mla/mini_mla.py
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/05_mini_vllm/mini_vllm.py
```

---

*本系列博客配套代码：`/mnt/esfs/master_work/vllm-from-scratch/`*
*参考代码库：`/mnt/esfs/master_work/vllm/`（vLLM 源码）*
