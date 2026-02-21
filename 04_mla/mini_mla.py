"""
Mini MLA (Multi-head Latent Attention) 从零实现
对应博客第十二章

实现了：
1. RotaryEmbedding (RoPE)
2. MultiHeadLatentAttention (MLA)
   - 低秩 KV 压缩（减少 KV Cache 7x）
   - 解耦 RoPE（分离带位置和不带位置的部分）
   - Prefill / Decode 两阶段处理
3. KV Cache 大小对比演示
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# RoPE 旋转位置编码
# ============================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转向量的一半维度，用于 RoPE"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,            # [..., head_dim]
    k: torch.Tensor,            # [..., head_dim]
    cos: torch.Tensor,          # [seq_len, head_dim]
    sin: torch.Tensor,          # [seq_len, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q, K 应用旋转位置编码"""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated


class RotaryEmbedding(nn.Module):
    """RoPE：旋转位置编码"""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # 频率向量
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # [S, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, dim]（重复两次对应旋转）
        self.register_buffer('cos_cached', emb.cos())  # [S, dim]
        self.register_buffer('sin_cached', emb.sin())  # [S, dim]

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 cos, sin [seq_len, dim]"""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ============================================================
# KV Cache 数据结构
# ============================================================

class MLAKVCache:
    """
    MLA 的 KV Cache 只存储低维潜在向量

    对比标准 MHA 的 KV Cache（存完整 K, V），MLA 只存：
    - c_kv: 低秩潜在向量 [kv_lora_rank]
    - k_rope: 带位置信息的小向量 [qk_rope_head_dim]
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # 实际存储的维度（远小于完整 K/V）
        cache_dim = kv_lora_rank + qk_rope_head_dim

        # [num_layers, batch, seq, cache_dim]
        self.cache = torch.zeros(
            num_layers, max_batch_size, max_seq_len, cache_dim,
            device=device, dtype=dtype
        )
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def get(self, layer_id: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取当前缓存的 c_kv 和 k_rope"""
        seq_len = self.seq_lens[:batch_size].max().item()
        cache = self.cache[layer_id, :batch_size, :seq_len]  # [B, S, cache_dim]
        c_kv  = cache[..., :self.kv_lora_rank]               # [B, S, kv_lora_rank]
        k_rope = cache[..., self.kv_lora_rank:]               # [B, S, rope_dim]
        return c_kv, k_rope

    def update(
        self,
        layer_id: int,
        c_kv: torch.Tensor,    # [B, T, kv_lora_rank]
        k_rope: torch.Tensor,  # [B, T, rope_dim]
        seq_lens: torch.Tensor,
    ):
        """追加新的 KV 到缓存"""
        B, T, _ = c_kv.shape
        for b in range(B):
            start = self.seq_lens[b].item()
            self.cache[layer_id, b, start:start+T, :self.kv_lora_rank] = c_kv[b]
            self.cache[layer_id, b, start:start+T, self.kv_lora_rank:] = k_rope[b]
        self.seq_lens[:B] += T


# ============================================================
# 核心 MLA 实现
# ============================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA)

    关键维度（参考 DeepSeek V2 规格，适当缩小）：
    - hidden_size:      输入维度
    - num_heads:        注意力头数
    - qk_nope_head_dim: Q/K 中不带 RoPE 的维度
    - qk_rope_head_dim: Q/K 中带 RoPE 的维度
    - kv_lora_rank:     KV 低秩压缩维度（这是 KV Cache 实际存储的）
    - v_head_dim:       V 的每头维度
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        kv_lora_rank: int = 128,
        v_head_dim: int = 64,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # ── Query 侧（两阶段投影）──
        # 低秩压缩（可选，有些实现直接一步到位）
        q_lora_rank = num_heads * q_head_dim // 4  # 压缩比 1/4
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * q_head_dim, bias=False)

        # ── KV 侧（低秩压缩，关键！）──
        # 输出：kv_lora_rank（不带位置）+ qk_rope_head_dim（带位置的K rope部分）
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank)

        # 从低秩恢复完整 K (nope) + V
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )

        # ── 输出投影 ──
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(dim=qk_rope_head_dim, max_seq_len=max_seq_len)

        self.scale = q_head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, T, D]
        positions: torch.Tensor,           # [B, T]，当前 token 的位置
        past_c_kv: Optional[torch.Tensor] = None,   # [B, past_T, kv_lora_rank]
        past_k_rope: Optional[torch.Tensor] = None, # [B, past_T, rope_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            output: [B, T, D]
            new_c_kv: [B, T, kv_lora_rank]  ← 存入 KV Cache 的内容
            new_k_rope: [B, T, rope_dim]     ← 存入 KV Cache 的内容
        """
        B, T, D = hidden_states.shape
        H = self.num_heads

        # ===== Step 1: Query 计算 =====
        q = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, T, q_lora_rank]
        q = self.q_b_proj(q)  # [B, T, H*(nope+rope)]
        q = q.view(B, T, H, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]    # [B, T, H, nope]
        q_rope = q[..., self.qk_nope_head_dim:]     # [B, T, H, rope]

        # ===== Step 2: KV 低秩压缩（新 token 的 c_kv 和 k_rope）=====
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        # [B, T, kv_lora_rank + rope_dim]

        new_c_kv   = kv_compressed[..., :self.kv_lora_rank]      # [B, T, kv_lora_rank]
        new_k_rope = kv_compressed[..., self.kv_lora_rank:]        # [B, T, rope_dim]
        new_c_kv = self.kv_a_layernorm(new_c_kv)

        # ===== Step 3: 拼接历史 KV Cache =====
        if past_c_kv is not None:
            c_kv   = torch.cat([past_c_kv,   new_c_kv],   dim=1)   # [B, past+T, lora]
            k_rope = torch.cat([past_k_rope,  new_k_rope], dim=1)   # [B, past+T, rope]
        else:
            c_kv   = new_c_kv    # [B, T, lora]
            k_rope = new_k_rope  # [B, T, rope]

        S = c_kv.shape[1]  # 总序列长度

        # ===== Step 4: 从 c_kv 恢复 K (nope) + V =====
        kv = self.kv_b_proj(c_kv)  # [B, S, H*(nope+v_dim)]
        kv = kv.view(B, S, H, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., :self.qk_nope_head_dim]   # [B, S, H, nope]
        v      = kv[..., self.qk_nope_head_dim:]    # [B, S, H, v_dim]

        # ===== Step 5: 应用 RoPE =====
        # 获取 cos/sin（取所有可能用到的最大长度）
        max_pos = max(positions.max().item() + 1, S)
        cos_full, sin_full = self.rotary_emb(int(max_pos))

        # 取 Query 对应位置的 cos/sin: [B, T, rope_dim]
        cos_q = cos_full[positions.reshape(-1)].reshape(B, T, self.qk_rope_head_dim)
        sin_q = sin_full[positions.reshape(-1)].reshape(B, T, self.qk_rope_head_dim)

        # 取 Key 对应位置（0..S-1）的 cos/sin: [B, S, rope_dim]
        all_positions = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        cos_k = cos_full[all_positions.reshape(-1)].reshape(B, S, self.qk_rope_head_dim)
        sin_k = sin_full[all_positions.reshape(-1)].reshape(B, S, self.qk_rope_head_dim)

        # 对 q_rope [B, T, H, rope] 应用 RoPE
        # 广播: cos_q [B, T, 1, rope] → [B, T, H, rope]
        cos_q = cos_q.unsqueeze(2)  # [B, T, 1, rope]
        sin_q = sin_q.unsqueeze(2)
        q_rope_out = q_rope * cos_q + rotate_half(q_rope) * sin_q  # [B, T, H, rope]

        # 对 k_rope [B, S, rope] 扩展到多头后应用 RoPE
        k_rope_expand = k_rope.unsqueeze(2).expand(-1, -1, H, -1)  # [B, S, H, rope]
        cos_k = cos_k.unsqueeze(2)  # [B, S, 1, rope]
        sin_k = sin_k.unsqueeze(2)
        k_rope_out = k_rope_expand * cos_k + rotate_half(k_rope_expand) * sin_k  # [B, S, H, rope]

        # ===== Step 6: 拼接 nope + rope 得到完整 Q, K =====
        q_full = torch.cat([q_nope, q_rope_out], dim=-1)   # [B, T, H, nope+rope]
        k_full = torch.cat([k_nope, k_rope_out], dim=-1)   # [B, S, H, nope+rope]

        # Transpose for attention: [B, H, T/S, dim]
        q_full = q_full.permute(0, 2, 1, 3)   # [B, H, T, qk_head_dim]
        k_full = k_full.permute(0, 2, 1, 3)   # [B, H, S, qk_head_dim]
        v      = v.permute(0, 2, 1, 3)         # [B, H, S, v_dim]

        # ===== Step 7: Attention =====
        attn = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.scale
        # [B, H, T, S]

        # Causal mask（Prefill 时需要）
        if T > 1:
            causal_mask = torch.triu(
                torch.full((T, S), float('-inf'), device=hidden_states.device),
                diagonal=S - T + 1
            )
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)  # [B, H, T, S]

        out = torch.matmul(attn, v)  # [B, H, T, v_dim]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, T, H, v_dim]
        out = out.view(B, T, H * self.v_head_dim)

        # ===== Step 8: 输出投影 =====
        output = self.o_proj(out)  # [B, T, D]

        return output, new_c_kv, new_k_rope


# ============================================================
# 工具函数：KV Cache 大小对比
# ============================================================

def compare_kv_cache_size(
    num_heads: int = 32,
    head_dim: int = 128,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    num_layers: int = 27,
    seq_len: int = 4096,
    batch_size: int = 8,
    dtype_bytes: int = 2,  # float16
):
    """量化对比 MHA 和 MLA 的 KV Cache 内存占用"""

    # MHA KV Cache per token per layer
    # 存储完整 K: [num_heads, head_dim] + V: [num_heads, head_dim]
    mha_per_token_per_layer = num_heads * head_dim * 2

    # MLA KV Cache per token per layer
    # 只存 c_kv [kv_lora_rank] + k_rope [qk_rope_head_dim]
    mla_per_token_per_layer = kv_lora_rank + qk_rope_head_dim

    # 总大小（MB）
    mha_total_mb = (
        batch_size * seq_len * num_layers * mha_per_token_per_layer * dtype_bytes
    ) / 1e6

    mla_total_mb = (
        batch_size * seq_len * num_layers * mla_per_token_per_layer * dtype_bytes
    ) / 1e6

    print("=" * 60)
    print("KV Cache 内存对比")
    print("=" * 60)
    print(f"  模型参数:")
    print(f"    num_heads = {num_heads}")
    print(f"    head_dim  = {head_dim}")
    print(f"    num_layers = {num_layers}")
    print(f"  批次参数:")
    print(f"    batch_size = {batch_size}")
    print(f"    seq_len    = {seq_len}")
    print()
    print(f"  MHA KV per token per layer: {mha_per_token_per_layer} float16")
    print(f"  MLA KV per token per layer: {mla_per_token_per_layer} float16")
    print(f"    = kv_lora_rank({kv_lora_rank}) + k_rope({qk_rope_head_dim})")
    print()
    print(f"  MHA 总 KV Cache: {mha_total_mb:.1f} MB")
    print(f"  MLA 总 KV Cache: {mla_total_mb:.1f} MB")
    print(f"  节省比例: {mha_total_mb / mla_total_mb:.1f}x")
    print("=" * 60)

    return mha_total_mb, mla_total_mb


# ============================================================
# 测试
# ============================================================

def test_rotary_emb():
    """测试 RoPE"""
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    cos, sin = rope(128)
    assert cos.shape == (128, 64), f"Expected (128, 64), got {cos.shape}"
    # 验证正交性：cos^2 + sin^2 ≈ 1
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)
    print("✓ RotaryEmbedding: OK")


def test_mla_prefill():
    """测试 MLA 的 Prefill（处理整个序列）"""
    mla = MultiHeadLatentAttention(
        hidden_size=256,
        num_heads=4,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        kv_lora_rank=64,
        v_head_dim=32,
    )

    B, T, D = 2, 16, 256
    x = torch.randn(B, T, D)
    positions = torch.arange(T).unsqueeze(0).expand(B, -1)  # [B, T]

    # Prefill（无历史 KV）
    out, c_kv, k_rope = mla(x, positions)

    assert out.shape == (B, T, D), f"out: {out.shape}"
    assert c_kv.shape == (B, T, 64), f"c_kv: {c_kv.shape}"
    assert k_rope.shape == (B, T, 16), f"k_rope: {k_rope.shape}"

    print(f"✓ MLA Prefill: out={out.shape}, c_kv={c_kv.shape}, k_rope={k_rope.shape}")

    return c_kv, k_rope  # 返回供 decode 测试使用


def test_mla_decode(prefill_c_kv: torch.Tensor, prefill_k_rope: torch.Tensor):
    """测试 MLA 的 Decode（逐 token 生成）"""
    mla = MultiHeadLatentAttention(
        hidden_size=256,
        num_heads=4,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        kv_lora_rank=64,
        v_head_dim=32,
    )

    B = prefill_c_kv.shape[0]
    prefill_len = prefill_c_kv.shape[1]
    D = 256

    # Decode：逐 token 生成
    all_c_kv   = prefill_c_kv
    all_k_rope = prefill_k_rope

    for step in range(3):
        x = torch.randn(B, 1, D)  # 每次只处理 1 个 token
        pos = torch.tensor([[prefill_len + step]]).expand(B, -1)

        out, new_c_kv, new_k_rope = mla(
            x, pos,
            past_c_kv=all_c_kv,
            past_k_rope=all_k_rope,
        )

        assert out.shape == (B, 1, D)

        # 追加到 KV Cache
        all_c_kv   = torch.cat([all_c_kv,   new_c_kv],   dim=1)
        all_k_rope = torch.cat([all_k_rope, new_k_rope], dim=1)

        print(f"✓ MLA Decode step {step}: seq_len = {all_c_kv.shape[1]}")


def test_kv_cache_comparison():
    """验证 KV Cache 大小对比"""
    # DeepSeek V2 规格
    mha_mb, mla_mb = compare_kv_cache_size(
        num_heads=128,       # DeepSeek V2 MHA heads
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_layers=60,
        seq_len=8192,
        batch_size=1,
    )
    ratio = mha_mb / mla_mb
    print(f"\n✓ KV Cache 节省: {ratio:.1f}x（MHA vs MLA）")
    assert ratio > 5, f"节省比例应该 > 5x，实际 {ratio:.1f}x"


def test_gradient_flow():
    """验证梯度能正确回传"""
    mla = MultiHeadLatentAttention(
        hidden_size=128,
        num_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        kv_lora_rank=32,
        v_head_dim=16,
    )

    x = torch.randn(2, 8, 128, requires_grad=True)
    positions = torch.arange(8).unsqueeze(0).expand(2, -1)

    out, _, _ = mla(x, positions)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad 为 None，梯度没有回传！"
    print(f"✓ 梯度回传: x.grad norm = {x.grad.norm().item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Mini MLA 测试套件")
    print("=" * 60)

    test_rotary_emb()

    c_kv, k_rope = test_mla_prefill()
    test_mla_decode(c_kv, k_rope)

    print()
    test_kv_cache_comparison()

    print()
    test_gradient_flow()

    print("\n所有测试通过！")
