"""
Mini MLA pytest 测试套件
运行：docker exec vllm python3 -m pytest 04_mla/test_mini_mla.py -v
"""
import torch
import pytest
from mini_mla import RotaryEmbedding, rotate_half, apply_rotary_emb, MultiHeadLatentAttention, compare_kv_cache_size


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=64)
        cos, sin = rope(128)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_unit_norm(self):
        """cos^2 + sin^2 == 1（逐元素）"""
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(64)
        assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)

    def test_different_lengths(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        for length in [1, 10, 128, 512]:
            cos, sin = rope(length)
            assert cos.shape[0] == length

    def test_position_0_cos_is_1(self):
        """位置 0 的 cos 应全为 1，sin 应全为 0"""
        rope = RotaryEmbedding(dim=64)
        cos, sin = rope(1)
        assert torch.allclose(cos[0], torch.ones(64), atol=1e-5)
        assert torch.allclose(sin[0], torch.zeros(64), atol=1e-5)


class TestRotateHalf:
    def test_shape_preserved(self):
        x = torch.randn(4, 8, 64)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_double_rotation_is_negation(self):
        """两次旋转等于取负（rotate_half(rotate_half(x)) == -x）"""
        x = torch.randn(3, 16)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


class TestMLA:
    @pytest.fixture
    def small_mla(self):
        return MultiHeadLatentAttention(
            hidden_size=128,
            num_heads=4,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            kv_lora_rank=32,
            v_head_dim=16,
            max_seq_len=256,
        )

    def test_prefill_output_shape(self, small_mla):
        B, T, D = 2, 16, 128
        x = torch.randn(B, T, D)
        positions = torch.arange(T).unsqueeze(0).expand(B, -1)
        out, c_kv, k_rope = small_mla(x, positions)
        assert out.shape == (B, T, D)
        assert c_kv.shape == (B, T, 32)   # kv_lora_rank
        assert k_rope.shape == (B, T, 8)  # qk_rope_head_dim

    def test_decode_output_shape(self, small_mla):
        """Decode：每次只处理 1 个 token"""
        B, D = 2, 128
        # 先做 prefill
        T_prefill = 10
        x_p = torch.randn(B, T_prefill, D)
        pos_p = torch.arange(T_prefill).unsqueeze(0).expand(B, -1)
        _, c_kv, k_rope = small_mla(x_p, pos_p)

        # 再做 decode
        x_d = torch.randn(B, 1, D)
        pos_d = torch.tensor([[T_prefill]] * B)
        out, new_c_kv, new_k_rope = small_mla(x_d, pos_d, past_c_kv=c_kv, past_k_rope=k_rope)
        assert out.shape == (B, 1, D)
        assert new_c_kv.shape == (B, 1, 32)

    def test_kv_cache_accumulation(self, small_mla):
        """多步 decode 后 KV Cache 应该增长"""
        B, D = 1, 128
        x = torch.randn(B, 5, D)
        pos = torch.arange(5).unsqueeze(0)
        _, c_kv, k_rope = small_mla(x, pos)
        assert c_kv.shape[1] == 5

        for step in range(3):
            x_d = torch.randn(B, 1, D)
            pos_d = torch.tensor([[5 + step]])
            _, new_c, new_r = small_mla(x_d, pos_d, past_c_kv=c_kv, past_k_rope=k_rope)
            c_kv   = torch.cat([c_kv,   new_c], dim=1)
            k_rope = torch.cat([k_rope, new_r], dim=1)

        assert c_kv.shape[1] == 8  # 5 prefill + 3 decode

    def test_gradient_backprop(self, small_mla):
        x = torch.randn(2, 8, 128, requires_grad=True)
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        out, _, _ = small_mla(x, positions)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_causal_mask_applied_prefill(self, small_mla):
        """Prefill 时 token i 不能看到 token j > i"""
        small_mla.eval()
        B, T, D = 1, 4, 128

        # 第一次推理
        x = torch.randn(B, T, D)
        pos = torch.arange(T).unsqueeze(0)
        with torch.no_grad():
            out1, _, _ = small_mla(x, pos)

        # 修改 token 2 之后的内容，不应该影响 out1[0, :2]
        x2 = x.clone()
        x2[0, 2:] = torch.randn_like(x2[0, 2:])
        with torch.no_grad():
            out2, _, _ = small_mla(x2, pos)

        # 前两个 token 的输出应该相同（因为 causal mask）
        assert torch.allclose(out1[0, :2], out2[0, :2], atol=1e-4), \
            "Causal mask 不生效：后续 token 影响了前面 token 的输出"


class TestKVCacheComparison:
    def test_mla_saves_memory(self):
        mha_mb, mla_mb = compare_kv_cache_size(
            num_heads=32,
            head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_layers=32,
            seq_len=4096,
            batch_size=1,
        )
        ratio = mha_mb / mla_mb
        assert ratio > 5, f"MLA 应节省 >5x，实际 {ratio:.1f}x"

    def test_mla_kv_size_linear_in_seq(self):
        """MLA KV Cache 大小应与序列长度线性相关"""
        _, mla_1k = compare_kv_cache_size(seq_len=1024, num_heads=16,
                                           head_dim=64, kv_lora_rank=128,
                                           qk_rope_head_dim=32, num_layers=8,
                                           batch_size=1)
        _, mla_4k = compare_kv_cache_size(seq_len=4096, num_heads=16,
                                           head_dim=64, kv_lora_rank=128,
                                           qk_rope_head_dim=32, num_layers=8,
                                           batch_size=1)
        ratio = mla_4k / mla_1k
        assert abs(ratio - 4.0) < 0.01, f"应该是 4x，实际 {ratio:.2f}x"
