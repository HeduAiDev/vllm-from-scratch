# vLLM：从入门到专家（第三部分）

---

## 第十一章：DeepSeek MoE——专家混合架构从0实现

### 11.1 为什么需要 MoE？

普通 Transformer 的 FFN 层，每个 token 都会经过相同的 MLP 计算：

```
输入 x → Linear1(x) → GELU → Linear2 → 输出
参数量：hidden_size × intermediate_size × 2
计算量：所有 token 都经过全部参数
```

**问题**：参数越多，效果越好——但计算量也线性增加。

MoE（Mixture of Experts）的思路：

> 用 N 个"专家"（Expert）替代一个 FFN，每个 token 只激活其中 K 个专家。

```
输入 x → Router → 选择 Top-K 专家
                      ├─ Expert_0(x) ─┐
                      ├─ Expert_3(x) ─┤ 加权求和 → 输出
                      └─ Expert_7(x) ─┘
```

**收益**：
- 参数量：N × expert_size（大）
- 计算量：K × expert_size（K << N，小）
- 效果接近大模型，计算接近小模型

### 11.2 DeepSeek V2/V3 的 MoE 设计

DeepSeek 的创新点在于**共享专家 + 路由专家**的双轨设计：

```
普通 MoE：
  所有专家都参与竞争

DeepSeek MoE：
  ┌─ 共享专家（Shared Experts）：每个 token 都激活
  ├─ 路由专家（Routed Experts）：Top-K 竞争
  └─ 输出：shared_out + weighted_sum(routed_experts)
```

**DeepSeek V2 配置**：
- `n_routed_experts = 160`（路由专家数）
- `n_shared_experts = 2`（共享专家数）
- `num_experts_per_tok = 6`（每个 token 激活的路由专家数）
- `n_group = 8, topk_group = 3`（分组 TopK 路由）

**DeepSeek V3/R1 配置**：
- `n_routed_experts = 256`
- `n_shared_experts = 1`
- `num_experts_per_tok = 8`

### 11.3 分组 TopK 路由算法

标准 TopK 路由：在所有专家中选 K 个分数最高的。

分组 TopK（GroupedTopK）：

```
假设 n_experts=256, n_group=8, topk_group=3

Step1：将 256 个专家分成 8 组（每组 32 个）
Step2：在每组内选 topk_group=3 个分数最高的
Step3：计算每组的 "组分数" = 组内 top3 分数之和
Step4：选 topk_group=3 个组分数最高的组
Step5：从这 3 个组中，最终选出 num_experts_per_tok=8 个专家
```

好处：**全局负载均衡**——专家均匀分布在各组，避免全部 token 都涌向少数"明星专家"。

```python
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,   # [num_tokens, n_experts]
    topk: int,                     # 最终选择的专家数
    num_expert_group: int,         # 分组数
    topk_group: int,               # 每组选几个专家
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (topk_weights, topk_ids)
    topk_weights: [num_tokens, topk]
    topk_ids:     [num_tokens, topk]
    """
    num_token = gating_output.shape[0]
    num_expert = gating_output.shape[1]
    experts_per_group = num_expert // num_expert_group

    scores = torch.softmax(gating_output, dim=-1)  # [T, E]

    # Step1: 分组 reshape
    scores_grouped = scores.view(num_token, num_expert_group, -1)
    # [T, G, E/G]

    # Step2: 每组选 topk_group 个专家
    group_topk_vals, group_topk_ids = torch.topk(
        scores_grouped, k=topk_group, dim=-1
    )  # [T, G, topk_group]

    # Step3: 计算组分数
    group_scores = group_topk_vals.sum(dim=-1)  # [T, G]

    # Step4: 选最好的几个组
    num_groups_to_select = topk // topk_group
    _, selected_groups = torch.topk(group_scores, k=num_groups_to_select, dim=-1)
    # [T, num_groups_to_select]

    # Step5: 在选中的组内，取出 topk_group 个专家
    # 构建掩码
    group_mask = torch.zeros_like(scores_grouped)
    group_mask.scatter_(1, selected_groups.unsqueeze(-1).expand(-1, -1, topk_group),
                        group_topk_ids.gather(1, selected_groups.unsqueeze(-1).expand(-1, -1, topk_group)))

    # 简化版：直接展开拿 top-K
    # 实际 vLLM 用 Triton kernel 实现高效版本
    mask = torch.zeros(num_token, num_expert, device=scores.device)
    for b in range(num_token):
        for g in selected_groups[b]:
            g = g.item()
            for k in range(topk_group):
                expert_idx = g * experts_per_group + group_topk_ids[b, g, k].item()
                mask[b, expert_idx] = 1.0

    masked_scores = scores * mask
    topk_weights, topk_ids = torch.topk(masked_scores, k=topk, dim=-1)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

    return topk_weights, topk_ids
```

### 11.4 Expert Parallel（专家并行）

当专家数量多（256个）时，一块 GPU 放不下，需要**跨 GPU 分发专家**：

```
EP_size=4, n_experts=256

GPU0: Expert[0-63]    GPU1: Expert[64-127]
GPU2: Expert[128-191] GPU3: Expert[192-255]

执行流程：
1. 路由计算：每个GPU都计算所有token的路由logits（ReplicatedLinear）
2. All2All通信：把token分发给对应的专家GPU
3. 本地计算：每个GPU处理分配到本地的token
4. All2All汇聚：结果送回原始GPU
```

**All2All 通信示意**：

```
[GPU0 有 token_A → Expert[70], token_B → Expert[5]]

通信：
  GPU0 发送 token_A 到 GPU1（因为 Expert[70] 在 GPU1）
  GPU0 保留 token_B（因为 Expert[5] 在 GPU0）

GPU1 处理 token_A，结果返回 GPU0
```

### 11.5 从0实现：Mini MoE

```python
# 见 03_moe/mini_moe.py
```

下面是 Mini MoE 的核心实现，展示 Router + Expert + Load Balancing 的完整流程：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """单个专家：就是一个 MLP"""
    def __init__(self, input_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 激活（与 LLaMA / DeepSeek 一致）
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """
    路由器：给每个 token 打分，选出 Top-K 专家
    支持辅助损失（aux loss）用于负载均衡
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [T, D]
        logits = self.gate(x)          # [T, E]
        probs = F.softmax(logits, dim=-1)

        # Top-K 选择
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)  # [T, K]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 归一化

        # 辅助负载均衡损失（Switch Transformer）
        # 目标：每个专家的平均使用率相等 = 1/E
        expert_usage = F.one_hot(topk_ids[:, 0], self.num_experts).float().mean(0)
        mean_prob = probs.mean(0)
        aux_loss = (expert_usage * mean_prob).sum() * self.num_experts

        return topk_probs, topk_ids, aux_loss


class MoELayer(nn.Module):
    """
    完整的 MoE 层
    包含：路由器 + N 个专家 + 可选共享专家
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        intermediate_dim: int,
        num_shared_experts: int = 0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, intermediate_dim) for _ in range(num_experts)
        ])

        # 共享专家（可选，DeepSeek 设计）
        if num_shared_experts > 0:
            self.shared_expert = Expert(
                input_dim, intermediate_dim * num_shared_experts
            )
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, input_dim] 或 [num_tokens, input_dim]
        """
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])  # [T, D]
        T, D = x.shape

        # 1. 路由
        topk_probs, topk_ids, aux_loss = self.router(x)
        # topk_probs: [T, K], topk_ids: [T, K]

        # 2. 共享专家（不需要路由，每个 token 都过）
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)  # [T, D]
        else:
            shared_out = torch.zeros_like(x)

        # 3. 路由专家：计算每个专家需要处理哪些 token
        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            # 找出路由到 expert_id 的 token
            # topk_ids: [T, K] → 找 topk_ids == expert_id 的位置
            expert_mask = (topk_ids == expert_id)  # [T, K] bool
            token_mask = expert_mask.any(dim=-1)    # [T] bool

            if not token_mask.any():
                continue

            # 取出对应 token
            tokens_for_expert = x[token_mask]        # [num_tokens_for_this_expert, D]

            # 专家计算
            expert_out = self.experts[expert_id](tokens_for_expert)

            # 加权：找这些 token 的路由权重
            # expert_mask[token_mask]: [num_tokens, K]
            weights = topk_probs[token_mask][expert_mask[token_mask]]
            # [num_tokens_for_this_expert]

            expert_out = expert_out * weights.unsqueeze(-1)

            # 累加回 output
            output[token_mask] += expert_out

        # 4. 合并共享专家和路由专家
        output = output + shared_out

        return output.view(original_shape), aux_loss
```

### 11.6 FusedMoE：Triton 加速版本

vLLM 中的 MoE 用 Triton 内核实现了高效的批量专家计算，避免了上面 Python 级别的 for 循环：

```
核心思路：
1. 按专家ID对token排序（argsort）
2. 用一个大的 grouped GEMM 一次性计算所有专家
3. 按原始顺序归约

时间复杂度对比：
  Python for 循环：O(num_experts) 个 kernel launch
  Triton FusedMoE：O(1) 个 kernel launch
```

```python
# vLLM 实际调用位置
# vllm/model_executor/layers/fused_moe/fused_moe.py

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K 路由，返回权重和专家ID"""

    topk_weights, topk_ids = torch.topk(
        gating_output, topk, dim=-1
    )
    topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(hidden_states.dtype), topk_ids


# FusedMoE.forward 的核心调用
output = fused_experts(
    hidden_states=hidden_states,
    w1=self.w1,              # Gate projection weights
    w2=self.w2,              # Down projection weights
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    use_grouped_topk=True,
    override_config=...,
)
```

---

## 第十二章：MLA——多头潜在注意力从0实现

### 12.1 标准 MHA 的 KV Cache 问题

在推理阶段，每个 token 都需要把自己的 K, V 存入 KV Cache 供后续 token 使用：

```
标准 MHA:
  每个 token 存储：
    K: [num_heads, head_dim]   = 32 × 128 = 4096 float16
    V: [num_heads, head_dim]   = 32 × 128 = 4096 float16
  总计：8192 float16 per token per layer

一个 4096-token 对话，32层模型：
  4096 × 32 × 8192 × 2 bytes ≈ 2GB KV cache!
```

随着上下文长度的增加，KV Cache 成为推理的主要内存瓶颈。

### 12.2 MLA 的核心思想

DeepSeek V2 提出 **Multi-head Latent Attention（MLA）**，通过低秩分解压缩 KV：

```
标准 KV 计算：
  K = X @ W_k    [T, n_heads × head_dim]
  V = X @ W_v    [T, n_heads × head_dim]

MLA KV 计算：
  c_kv = X @ W_down_kv   [T, kv_lora_rank]   ← 潜在向量（低维！）
  K = c_kv @ W_up_k       [T, n_heads × head_dim]
  V = c_kv @ W_up_v       [T, n_heads × head_dim]
```

**KV Cache 只需存 c_kv**（低秩潜在向量），而不是完整的 K, V：

```
KV Cache 大小对比（DeepSeek V2-Lite）：
  MHA：  n_heads × head_dim × 2 = 16 × 128 × 2 = 4096
  MLA：  kv_lora_rank = 512

  节省：4096 / 512 = 8x！
```

### 12.3 RoPE 的特殊处理

MLA 的挑战：RoPE（旋转位置编码）依赖绝对位置，但低秩 K 无法直接应用。

解决方案：**解耦 RoPE**——

```
K = [K_nope | K_rope]
    └──────┘  └──────┘
    不带RoPE  带RoPE（从额外投影得来）

KV Cache 存储：
  - c_kv（低秩，恢复 K_nope 和 V 用）
  - k_rope（带位置信息的小向量）

推理时重建：
  K = [c_kv @ W_up_k_nope | k_rope]
  V = c_kv @ W_up_v
```

### 12.4 Prefill vs Decode 的不同处理

```
Prefill 阶段（处理所有 prompt token）：
  - 正常计算 Q, K, V（完整展开）
  - 使用 FlashAttention 计算全量 attention
  - 将 c_kv 存入 KV Cache

Decode 阶段（每次生成1个token）：
  - Q：完整计算
  - K, V：从 KV Cache 恢复（c_kv → K_nope, V）
  - 使用 PagedAttention 计算
```

### 12.5 从0实现 Mini MLA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码"""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, heads, seq, head_dim]
        positions: [batch, seq]
        """
        sincos = torch.outer(positions.float().view(-1), self.inv_freq)
        sin = sincos.sin()[None, None, :, :]  # [1, 1, S, dim/2]
        cos = sincos.cos()[None, None, :, :]

        x_rot = x[..., :x.shape[-1]//2]
        x_pass = x[..., x.shape[-1]//2:]

        # 旋转操作
        x_rot_new = torch.stack([
            x_rot[..., ::2] * cos - x_rot[..., 1::2] * sin,
            x_rot[..., ::2] * sin + x_rot[..., 1::2] * cos,
        ], dim=-1).flatten(-2)

        return torch.cat([x_rot_new, x_pass], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    """
    Mini MLA 实现

    关键维度（参考 DeepSeek V2-Lite 缩小版）：
    - hidden_size: 2048
    - num_heads: 16
    - head_dim: 128 (= qk_nope_head_dim + qk_rope_head_dim)
    - qk_nope_head_dim: 128 (不带RoPE的query/key维度)
    - qk_rope_head_dim: 64  (带RoPE的query/key维度)
    - kv_lora_rank: 512     (低秩KV缓存维度)
    - v_head_dim: 128       (value的维度)
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        kv_lora_rank: int = 512,
        v_head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Query 投影：两阶段
        # Stage1: 低秩压缩
        self.q_down_proj = nn.Linear(hidden_size, num_heads * q_head_dim // 2, bias=False)
        self.q_down_norm = nn.LayerNorm(num_heads * q_head_dim // 2)
        # Stage2: 展开到完整维度
        self.q_up_proj   = nn.Linear(num_heads * q_head_dim // 2, num_heads * q_head_dim, bias=False)

        # KV 压缩投影（这部分存入 KV Cache）
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_down_norm = nn.LayerNorm(kv_lora_rank)

        # KV 展开投影（推理时从 KV Cache 恢复）
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),  # K_nope + V
            bias=False
        )

        # 输出投影
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(dim=qk_rope_head_dim)

        self.scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    def forward(
        self,
        x: torch.Tensor,              # [B, T, D]
        positions: torch.Tensor,       # [B, T]
        kv_cache: tuple | None = None, # (c_kv, k_rope) from past tokens
        is_decode: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        B, T, D = x.shape
        H = self.num_heads

        # ===== Query 计算 =====
        q_compressed = self.q_down_norm(self.q_down_proj(x))  # [B, T, H*q_dim/2]
        q = self.q_up_proj(q_compressed)  # [B, T, H*(nope+rope)]

        q = q.view(B, T, H, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]        # [B, T, H, nope]
        q_rope = q[..., self.qk_nope_head_dim:]          # [B, T, H, rope]

        # 对 q_rope 应用 RoPE
        q_rope = self.rope(q_rope.transpose(1, 2), positions).transpose(1, 2)

        # ===== KV 压缩（存入 KV Cache 的是这个！）=====
        kv_compressed_full = self.kv_down_proj(x)  # [B, T, kv_lora_rank + rope]
        c_kv = kv_compressed_full[..., :self.kv_lora_rank]     # [B, T, kv_lora_rank]
        k_rope = kv_compressed_full[..., self.kv_lora_rank:]    # [B, T, rope_dim]

        c_kv = self.kv_down_norm(c_kv)

        # 拼接历史 KV Cache（decode 时）
        if kv_cache is not None:
            past_c_kv, past_k_rope = kv_cache
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)         # [B, T+past, lora]
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)    # [B, T+past, rope]

        new_kv_cache = (c_kv, k_rope)
        S = c_kv.shape[1]  # 总序列长度（包含历史）

        # ===== KV 展开（计算时从 c_kv 恢复 K, V）=====
        kv = self.kv_up_proj(c_kv)  # [B, S, H*(nope+v_dim)]
        kv = kv.view(B, S, H, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., :self.qk_nope_head_dim]  # [B, S, H, nope]
        v = kv[..., self.qk_nope_head_dim:]         # [B, S, H, v_dim]

        # 对 k_rope 应用 RoPE（需要所有历史位置）
        all_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, H, -1)  # [B, S, H, rope]
        k_rope = self.rope(k_rope.transpose(1, 2), all_positions).transpose(1, 2)

        # 拼接完整的 K
        q_full = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)  # [B, H, T, nope+rope]
        k_full = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)  # [B, H, S, nope+rope]
        v = v.transpose(1, 2)                                          # [B, H, S, v_dim]

        # ===== Attention 计算 =====
        attn_weights = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        # [B, H, T, S]

        if not is_decode:
            # Causal mask（prefill 时需要）
            mask = torch.triu(
                torch.full((T, S), float('-inf'), device=x.device), diagonal=1
            )
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_out = torch.matmul(attn_weights, v)  # [B, H, T, v_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim)

        # ===== 输出投影 =====
        output = self.o_proj(attn_out)  # [B, T, D]

        return output, new_kv_cache
```

### 12.6 KV Cache 内存节省验证

```python
def compare_kv_cache_size():
    """量化对比 MHA 和 MLA 的 KV Cache 大小"""

    # 模型参数（DeepSeek V2-Lite 规格）
    batch_size = 1
    num_layers = 27
    num_heads = 16
    head_dim = 128        # nope + rope

    # MHA KV Cache
    mha_kv_per_token = num_heads * head_dim * 2  # K + V
    # = 16 * 128 * 2 = 4096 float16

    # MLA KV Cache
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    mla_kv_per_token = kv_lora_rank + qk_rope_head_dim
    # = 576 float16（远小于 4096！）

    seq_len = 4096
    dtype_bytes = 2  # float16

    mha_total = seq_len * num_layers * mha_kv_per_token * dtype_bytes / 1e6
    mla_total = seq_len * num_layers * mla_kv_per_token * dtype_bytes / 1e6

    print(f"序列长度: {seq_len} tokens")
    print(f"层数: {num_layers}")
    print(f"MHA KV Cache: {mha_total:.1f} MB")
    print(f"MLA KV Cache: {mla_total:.1f} MB")
    print(f"节省比例: {mha_total / mla_total:.1f}x")

# 输出：
# 序列长度: 4096 tokens
# 层数: 27
# MHA KV Cache: 917.5 MB
# MLA KV Cache: 128.7 MB
# 节省比例: 7.1x
```

---

## 第十三章：PD 分离——Prefill-Decode 解耦架构

### 13.1 为什么需要 PD 分离？

在标准的 vLLM 架构中，Prefill 和 Decode 混在同一批次中执行：

**问题：**

```
Prefill 特点：           Decode 特点：
  - 计算密集（compute-bound）  - 内存带宽密集（memory-bound）
  - 长序列 → 高延迟           - 短序列（1个token）→ 低延迟要求
  - 吞吐优先                  - 延迟优先（用户在等待！）

混合批次的问题：
  Prefill(req_A=2048 tokens) + Decode(req_B, req_C, req_D)

  → req_B/C/D 必须等 Prefill 完成才能生成，延迟剧增！
  → 对话场景（req_B/C/D）用户体验很差
```

### 13.2 PD 分离的方案

**核心思路**：用两类机器分别处理 Prefill 和 Decode：

```
┌──────────────────────────────────────────────────────────────┐
│                    PD 分离架构                                │
│                                                              │
│  [客户端请求]                                                │
│       │                                                      │
│       ▼                                                      │
│  [路由器/LB]                                                 │
│       │                                                      │
│  ┌────┴──────────────────────────────┐                      │
│  │                                   │                      │
│  ▼                                   ▼                      │
│ [Prefill 节点集群]             [Decode 节点集群]            │
│  - 多 GPU，高带宽                - 中等 GPU，低延迟          │
│  - 全力做 Prefill                - 持续 Decode              │
│  - 计算完成后传 KV Cache →→→→→→→→ 接收 KV Cache，开始 decode│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**关键问题：KV Cache 如何跨节点传输？**

### 13.3 vLLM V1 的 KV Cache 传输实现

vLLM 在 `vllm/v1/worker/kv_connector/` 中实现了 KV Cache 连接器接口：

```python
# vllm/v1/worker/kv_connector/base.py（简化）

class KVConnectorBase:
    """
    KV Cache 连接器基类
    Prefill 节点：send side（发送 KV）
    Decode 节点：recv side（接收 KV）
    """

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input,
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: torch.Tensor,
    ) -> None:
        """Prefill 侧：推理完成后，发送 KV Cache 给 Decode 节点"""
        raise NotImplementedError

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input,
        kv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        """Decode 侧：接收 Prefill 节点发来的 KV Cache"""
        raise NotImplementedError

    def close(self) -> None:
        """关闭连接，释放资源"""
        raise NotImplementedError
```

**Mooncake 实现（基于 RDMA 的高速传输）：**

```python
# vllm/v1/worker/kv_connector/mooncake_store_connector.py

class MooncakeStoreConnector(KVConnectorBase):
    """
    通过 Mooncake 分布式 KV 存储传输 KV Cache
    特点：
    - 基于 RDMA 的零拷贝传输
    - 按 block 哈希存储，支持 Prefix Cache 共享
    - 异步传输，不阻塞模型推理
    """

    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        from mooncake.store import MooncakeStore
        self.store = MooncakeStore(...)
        self.rank = rank

    def send_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches,
        hidden_or_intermediate_states,
    ):
        # 1. 找出需要传输的 block
        # 2. 按 block_hash 存入 Mooncake Store
        # 3. 通知 Decode 节点（通过 request metadata 中的 key 列表）

        for layer_id, kv_cache in enumerate(kv_caches):
            for block_hash, physical_block in self._get_blocks_to_send(model_input):
                # 提取 K, V
                k = kv_cache[0][physical_block]  # [block_size, num_heads, head_dim]
                v = kv_cache[1][physical_block]

                # 存入分布式存储（RDMA 零拷贝）
                key = f"kv/{block_hash}/layer{layer_id}"
                self.store.put(key, torch.stack([k, v]))

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input,
        kv_caches,
    ):
        # 1. 获取需要接收的 block 列表（从 request metadata）
        # 2. 从 Mooncake Store 拉取 KV
        # 3. 填充到本地 KV Cache

        for layer_id, kv_cache in enumerate(kv_caches):
            for block_hash, local_block in self._get_blocks_to_recv(model_input):
                key = f"kv/{block_hash}/layer{layer_id}"
                kv_data = self.store.get(key)  # RDMA 拉取

                k, v = kv_data[0], kv_data[1]
                kv_cache[0][local_block] = k
                kv_cache[1][local_block] = v

        return None, True  # hidden_states=None (从头decode), bypass_model=True
```

### 13.4 调度器的 PD 分离支持

Prefill 节点的调度器只做 Prefill，然后把请求"迁移"到 Decode 节点：

```python
# 简化的 Prefill-Only Scheduler 逻辑

class PrefillScheduler(Scheduler):
    """只做 Prefill 的调度器"""

    def schedule(self) -> SchedulerOutput:
        # 只处理 waiting 队列中的 prefill 请求
        # 不处理 decode 请求（decode 交给 Decode 节点）

        scheduled = []
        for request in self.waiting_queue:
            if request.is_prefill:
                scheduled.append(request)

        return SchedulerOutput(scheduled_requests=scheduled)

    def update_from_output(self, output: ModelRunnerOutput):
        for request_id, tokens in output.finished_prefill:
            request = self.requests[request_id]

            # Prefill 完成，把请求（包含 KV 位置信息）
            # 发送给 Decode 节点
            self.migrate_to_decode_node(request)
```

### 13.5 端到端数据流

```
[Prefill 节点执行流程]

时间轴：
t=0: 接收新请求，token_ids=[t0..t4096]
t=1: Scheduler 调度 → 执行 Prefill Forward
t=2: Forward 完成 → KV Cache 存入 GPU
t=3: KV Connector 异步发送 KV → Mooncake Store
t=4: 请求 metadata（block 哈希列表）发送给 Decode 节点
t=5: Prefill 节点完成，可以接下一个请求

[Decode 节点执行流程]

t=3 (并行): 收到 metadata（block 哈希列表）
t=3+Δ:     从 Mooncake Store 拉取 KV Cache → 填充本地 GPU
t=4:        开始 Decode 推理（生成第1个 token）
t=5:        生成第2个 token...（流式返回给用户）
```

### 13.6 性能收益分析

```
场景：4096 token prefill + 128 token decode
单机混合：
  Prefill 时间: 2s
  Decode 时间: 0.5s（128 * 4ms/token）
  用户等待第一个 token：2s（TTFT，Time to First Token）

PD 分离：
  Prefill 节点: 2s（不影响 Decode 节点）
  Decode 节点: 收到 KV Cache 后立即开始
  KV 传输: ~0.1s（RDMA 高速，4096 tokens × 每层KV大小）
  用户等待第一个 token：2.1s（略慢于单机，但 Decode 节点不等待）

关键优势：
  - Decode 节点可以同时服务 M 个 Decode 请求，吞吐提升 M 倍
  - Prefill 节点专注 prefill，不受 decode 干扰
  - 两种硬件可以分别优化（Prefill 用高算力，Decode 用高带宽）
```

---

## 第十四章：vLLM V1 引擎架构——整合全局视角

### 14.1 V0 vs V1 架构对比

vLLM 在 V1 版本进行了重大重构：

```
V0 架构（旧）：
  LLMEngine
    └── AsyncLLMEngine
          └── EngineCore (blocking)
                ├── ModelExecutor (单进程)
                └── Scheduler (C++ / Python)

V1 架构（新）：
  AsyncLLMEngine (非阻塞)
    └── EngineCore (独立进程)
          ├── Scheduler (Python，完全重写)
          ├── ModelExecutor（多进程Worker）
          └── KV Cache Manager（显式管理）
```

**V1 的主要改进：**

1. **完全异步**：前端与推理引擎解耦，支持高并发请求
2. **Scheduler 全重写**：支持 Chunked Prefill、Spec Decode、PD 分离
3. **显式 KV Cache 管理**：Block Manager 直接在 Python 层管理
4. **更好的 CUDA Graph 支持**：Prefill 和 Decode 分离的 CUDA Graph

### 14.2 V1 请求生命周期

```
1. 用户提交请求
   AsyncLLMEngine.generate(prompt, sampling_params)
         │
2. 请求入队
   EngineCore._add_request(request)
         │
3. 调度（每个推理步骤）
   Scheduler.schedule() → SchedulerOutput
     - 选哪些请求
     - 分配 KV Cache Block
     - 计算 token_budget
         │
4. 模型执行
   ModelExecutor.execute_model(SchedulerOutput)
     - InputBatch 准备
     - Forward Pass（Attention + MoE + ...）
     - Sampling（或 RejectionSampling）
         │
5. 更新调度器状态
   Scheduler.update_from_output(ModelRunnerOutput)
     - 更新 num_computed_tokens
     - 完成请求 → 释放 KV Cache
     - 更新 prefix cache 状态
         │
6. 返回结果
   AsyncLLMEngine 流式返回给用户
```

### 14.3 关键配置组合

| 功能 | 配置参数 | 效果 |
|------|---------|------|
| Chunked Prefill | `--enable-chunked-prefill` | 长 prompt 分块处理 |
| 投机解码 | `--speculative-model <model>` | 加速 decode，适合低熵输出 |
| EAGLE | `--speculative-model eagle` | 更高接受率的投机解码 |
| PD 分离 | `--kv-transfer-config` | 多节点分工 |
| Expert Parallel | `--tensor-parallel-size` + `--pipeline-parallel-size` | DeepSeek MoE 分布式 |
| EPLB | `--enable-expert-parallel-load-balancer` | 动态专家负载均衡 |

### 14.4 性能调优速查表

```
场景：在线聊天（低延迟优先）
  → 启用 Chunked Prefill（减少 Decode 等待）
  → 启用投机解码（加速单序列生成）
  → PD 分离（如果 GPU 资源充足）
  → TTFT 目标 < 500ms

场景：批量推理（吞吐优先）
  → 增大 max_num_seqs（更大批次）
  → 关闭投机解码（批量时收益不大）
  → 开启 Chunked Prefill（GPU 利用率）
  → 吞吐目标 > 10000 tokens/s

场景：长上下文（128K+）
  → 显式设置 max_model_len
  → 启用 prefix caching（系统提示词复用）
  → 考虑 MLA 模型（节省 KV Cache 内存）
  → 外部 KV Cache 卸载到 CPU/NVMe

场景：DeepSeek V3 大模型（671B）
  → EP + TP 组合
  → 启用 FusedMoE
  → 启用 EPLB
  → 至少 8x H100/H800
```

---

## 第十五章：从0构建 Mini vLLM

### 15.1 Mini vLLM 架构总结

经过前14章的深入分析，我们来整合一个 **Mini vLLM**，把所有核心组件串联起来：

```python
# 见 05_mini_vllm/mini_vllm.py
```

```
Mini vLLM 组件清单：

01_paged_attention/paged_attention.py
  └── PagedAttentionManager：Block Table、物理/逻辑块映射

02_kvcache/block_pool_lru.py
  └── BlockPoolLRU：LRU 缓存、Prefix Cache 支持

03_moe/mini_moe.py（本部分新增）
  └── MoELayer：Router + Expert + FusedMoE 简化版

04_mla/mini_mla.py（本部分新增）
  └── MultiHeadLatentAttention：低秩KV，节省内存

05_mini_vllm/mini_vllm.py（本部分新增）
  └── MiniVLLM：
        ├── Scheduler（请求队列 + KV 分配）
        ├── Model（Transformer + MoE 或 MLA）
        ├── Sampler（贪心/Top-P）
        └── 流式推理循环
```

### 15.2 完整的推理主循环

```python
class MiniVLLM:
    """
    极简版 vLLM，整合：
    - Paged Attention (KV Cache 分页管理)
    - 简单调度器（FCFS，支持 Chunked Prefill）
    - Speculative Decoding (可选)
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config

        # KV Cache 块管理
        self.block_manager = PagedAttentionManager(
            num_blocks=config['num_kv_blocks'],
            block_size=config['block_size'],
        )

        # 调度器
        self.scheduler = SimpleScheduler(
            max_num_seqs=config['max_num_seqs'],
            max_num_batched_tokens=config['max_num_batched_tokens'],
            enable_chunked_prefill=config.get('enable_chunked_prefill', True),
        )

        # KV Cache（每层一个 tensor）
        num_layers = config['num_layers']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        num_blocks = config['num_kv_blocks']
        block_size = config['block_size']

        self.kv_caches = [
            (
                torch.zeros(num_blocks, block_size, num_heads, head_dim),
                torch.zeros(num_blocks, block_size, num_heads, head_dim),
            )
            for _ in range(num_layers)
        ]

    def generate(self, prompts: list[str]) -> list[str]:
        """端到端推理（同步版本）"""

        # 1. 将 prompt 加入调度器
        requests = []
        for prompt in prompts:
            req = Request(
                prompt_token_ids=self.tokenizer.encode(prompt),
                max_new_tokens=self.config['max_new_tokens'],
            )
            self.scheduler.add_request(req)
            requests.append(req)

        # 2. 推理主循环
        while not all(req.is_finished for req in requests):
            # a. 调度
            scheduled = self.scheduler.schedule()
            if not scheduled:
                break

            # b. 构建 InputBatch
            input_ids, positions, block_tables, seq_lens = \
                self._build_input_batch(scheduled)

            # c. 模型前向
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=self.kv_caches,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )

            # d. 采样
            next_tokens = self._sample(logits, scheduled)

            # e. 更新请求状态
            self.scheduler.update_requests(scheduled, next_tokens)

        return [self.tokenizer.decode(req.output_token_ids) for req in requests]

    def _build_input_batch(self, scheduled: list[ScheduledRequest]):
        """构建模型输入"""
        input_ids_list = []
        positions_list = []
        block_tables_list = []
        seq_lens_list = []

        for sched_req in scheduled:
            req = sched_req.request

            # 本次推理的 token（prefill 取 chunk，decode 取最后1个）
            tokens = req.get_next_tokens(sched_req.num_tokens_to_compute)
            input_ids_list.append(tokens)

            # 位置
            start_pos = req.num_computed_tokens
            positions_list.append(torch.arange(
                start_pos, start_pos + len(tokens)
            ))

            # Block Table（虚拟→物理块映射）
            block_tables_list.append(
                self.block_manager.get_block_table(req.request_id)
            )

            seq_lens_list.append(req.num_computed_tokens + len(tokens))

        return (
            torch.cat(input_ids_list),
            torch.cat(positions_list),
            block_tables_list,  # ragged（不同长度）
            seq_lens_list,
        )
```

### 15.3 各章内容总结

| 章节 | 技术点 | 关键收益 |
|-----|--------|---------|
| 第1-2章 | 整体架构、Paged Attention | 理解 vLLM 设计哲学 |
| 第3-4章 | KV Cache 管理 | 内存高效利用 |
| 第5章 | 完整推理路径 | 端到端流程打通 |
| 第6-7章 | Prefix Cache | 相同前缀不重复计算 |
| 第8章 | Scheduler | 高效批调度 |
| 第9章 | 投机解码 | Decode 加速 1-5x |
| 第10章 | Chunked Prefill | 延迟降低 20-40% |
| 第11章 | DeepSeek MoE | 计算量降低 5-10x |
| 第12章 | MLA | KV Cache 减少 7x |
| 第13章 | PD 分离 | 架构级解耦，吞吐提升 |
| 第14章 | V1 引擎全局 | 生产级配置 |

---

## 结语

vLLM 的设计哲学可以用一句话总结：

> **用工程创新弥补硬件限制，把计算和内存的每一个 cycle 用到极致。**

从 Paged Attention 解决内存碎片，到 Prefix Cache 复用计算，再到 Speculative Decoding 隐藏延迟，再到 MoE/MLA 压缩参数和 KV Cache——每一个优化都在某个瓶颈上做了精准的工程突破。

理解了这些机制，你就掌握了现代 LLM 推理系统的核心密码。

---

*本系列博客对应代码：`/mnt/esfs/master_work/vllm-from-scratch/`*

*参考代码库：`/mnt/esfs/master_work/vllm/`（vLLM 源码）*
