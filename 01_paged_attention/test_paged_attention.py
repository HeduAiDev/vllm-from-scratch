"""
Paged Attention 单元测试
运行: pytest test_paged_attention.py -v
"""
import math
import pytest
import torch
from paged_attention import (
    KVCachePool, paged_attention_decode, paged_attention_prefill,
    compute_slot_mapping, BLOCK_SIZE
)


@pytest.fixture
def small_pool():
    return KVCachePool(num_blocks=16, block_size=BLOCK_SIZE,
                       num_kv_heads=4, head_dim=32, num_layers=2,
                       dtype=torch.float32, device='cpu')


class TestKVCachePool:
    def test_alloc_free_basic(self, small_pool):
        n_free = len(small_pool.free_blocks)
        blocks = small_pool.allocate(3)
        assert len(blocks) == 3
        assert len(small_pool.free_blocks) == n_free - 3
        small_pool.free(blocks)
        assert len(small_pool.free_blocks) == n_free

    def test_oom(self, small_pool):
        with pytest.raises(RuntimeError, match="OOM"):
            small_pool.allocate(999)

    def test_write_read_kv(self, small_pool):
        blocks = small_pool.allocate(1)
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        small_pool.write_kv(0, blocks[0], 0, k, v)
        assert torch.allclose(small_pool.kv_cache[0, 0, blocks[0], :, 0, :], k)
        assert torch.allclose(small_pool.kv_cache[1, 0, blocks[0], :, 0, :], v)


class TestSlotMapping:
    def test_slot_mapping_correctness(self):
        # block_table = [3, 7], block_size = 4
        block_table = [3, 7]
        slots = compute_slot_mapping(block_table, seq_len=6, block_size=4)
        # token 0: block 3, slot 0 → 3*4+0=12
        assert slots[0] == 12
        # token 3: block 3, slot 3 → 3*4+3=15
        assert slots[3] == 15
        # token 4: block 7, slot 0 → 7*4+0=28
        assert slots[4] == 28
        # token 5: block 7, slot 1 → 7*4+1=29
        assert slots[5] == 29


class TestPagedAttentionDecode:
    def test_output_shape(self):
        num_q_heads, num_kv_heads, head_dim = 8, 4, 32
        kv_cache = torch.randn(2, 8, num_kv_heads, BLOCK_SIZE, head_dim)
        query = torch.randn(num_q_heads, head_dim)
        out = paged_attention_decode(query, kv_cache, [0, 1], 20,
                                     head_dim, num_q_heads, num_kv_heads)
        assert out.shape == (num_q_heads, head_dim)

    def test_equivalence_with_standard_attention(self):
        """核心测试：Paged Attention 与标准 Attention 数值等价"""
        torch.manual_seed(42)
        num_heads, head_dim, seq_len = 4, 32, 8
        block_size = 4
        num_blocks = 4

        K = torch.randn(seq_len, num_heads, head_dim)
        V = torch.randn(seq_len, num_heads, head_dim)
        Q = torch.randn(num_heads, head_dim)

        # 标准 Attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum('hd,shd->hs', Q, K) * scale
        attn = torch.softmax(scores, dim=-1)
        out_std = torch.einsum('hs,shd->hd', attn, V)

        # Paged Attention（使用非连续块 [2, 0]）
        kv_cache = torch.zeros(2, num_blocks, num_heads, block_size, head_dim)
        block_table = [2, 0]  # 乱序物理块
        for i in range(seq_len):
            blk = i // block_size
            slot = i % block_size
            phys = block_table[blk]
            kv_cache[0, phys, :, slot] = K[i]
            kv_cache[1, phys, :, slot] = V[i]

        out_paged = paged_attention_decode(
            Q, kv_cache, block_table, seq_len, head_dim, num_heads, num_heads,
            block_size=block_size,
        )

        assert torch.allclose(out_std, out_paged, atol=1e-5), \
            f"Max diff: {(out_std - out_paged).abs().max():.6f}"

    def test_gqa_grouping(self):
        """GQA：多个Q头共享一个KV头"""
        num_q_heads, num_kv_heads, head_dim = 8, 2, 16
        seq_len = 4
        kv_cache = torch.randn(2, 2, num_kv_heads, BLOCK_SIZE, head_dim)
        Q = torch.randn(num_q_heads, head_dim)
        out = paged_attention_decode(Q, kv_cache, [0], seq_len,
                                     head_dim, num_q_heads, num_kv_heads)
        assert out.shape == (num_q_heads, head_dim)


class TestPagedAttentionPrefill:
    def test_kv_written_to_cache(self):
        """Prefill 应写入 KV Cache"""
        seq_len, num_q_heads, num_kv_heads, head_dim = 4, 4, 4, 16
        block_size = 4
        kv_cache = torch.zeros(2, 2, num_kv_heads, block_size, head_dim)
        Q = torch.randn(seq_len, num_q_heads, head_dim)
        K = torch.randn(seq_len, num_kv_heads, head_dim)
        V = torch.randn(seq_len, num_kv_heads, head_dim)

        paged_attention_prefill(Q, K, V, kv_cache, block_table=[0], block_size=block_size)

        # 验证K被写入块0
        assert torch.allclose(kv_cache[0, 0, :, 0, :], K[0])
        assert torch.allclose(kv_cache[0, 0, :, 3, :], K[3])

    def test_output_shape(self):
        seq_len, num_heads, head_dim = 8, 4, 32
        block_size = 4
        kv_cache = torch.zeros(2, 4, num_heads, block_size, head_dim)
        Q = torch.randn(seq_len, num_heads, head_dim)
        K = torch.randn(seq_len, num_heads, head_dim)
        V = torch.randn(seq_len, num_heads, head_dim)
        out = paged_attention_prefill(Q, K, V, kv_cache, [0, 1], block_size)
        assert out.shape == (seq_len, num_heads, head_dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
