"""
Paged Attention 从0实现
对应博客第三章

核心思想：KV Cache 不需要连续存储，用 block_table 做逻辑→物理映射
"""
import math
import torch
import torch.nn.functional as F
from typing import List, Optional

BLOCK_SIZE = 16  # 每块容纳的token数


class KVCachePool:
    """KV Cache 物理块池"""
    def __init__(self, num_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int, num_layers: int,
                 dtype=torch.float32, device='cpu'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 核心：大的连续KV Cache tensor
        # [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        self.kv_cache = torch.zeros(
            2, num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )
        self.ref_counts = [0] * num_blocks
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate(self, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"OOM: need {num_blocks}, have {len(self.free_blocks)}")
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks)]
        for b in allocated:
            self.ref_counts[b] = 1
        return allocated

    def free(self, block_ids: List[int]) -> None:
        for b in block_ids:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.free_blocks.append(b)

    def write_kv(self, layer_idx: int, block_id: int, slot_in_block: int,
                 key: torch.Tensor, value: torch.Tensor):
        self.kv_cache[0, layer_idx, block_id, :, slot_in_block, :] = key
        self.kv_cache[1, layer_idx, block_id, :, slot_in_block, :] = value


def compute_slot_mapping(block_table: List[int], seq_len: int,
                         block_size: int) -> List[int]:
    """
    计算slot_mapping：每个token的物理存储位置

    slot = block_table[token_idx // block_size] * block_size + (token_idx % block_size)
    """
    return [
        block_table[i // block_size] * block_size + (i % block_size)
        for i in range(seq_len)
    ]


def paged_attention_decode(
    query: torch.Tensor,       # [num_q_heads, head_dim]
    kv_cache: torch.Tensor,    # [2, num_blocks, num_kv_heads, block_size, head_dim]
    block_table: List[int],    # [num_blocks]
    seq_len: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """
    Paged Attention 解码：从非连续块读取KV并计算Attention
    """
    scale = 1.0 / math.sqrt(head_dim)
    gqa_factor = num_q_heads // num_kv_heads
    output = torch.zeros(num_q_heads, head_dim,
                         dtype=query.dtype, device=query.device)

    for kv_h in range(num_kv_heads):
        q_h = query[kv_h * gqa_factor:(kv_h + 1) * gqa_factor]  # [gqa, head_dim]

        keys = [kv_cache[0, block_table[i // block_size], kv_h, i % block_size]
                for i in range(seq_len)]
        vals = [kv_cache[1, block_table[i // block_size], kv_h, i % block_size]
                for i in range(seq_len)]

        K = torch.stack(keys)  # [seq_len, head_dim]
        V = torch.stack(vals)

        scores = torch.einsum('gh,sh->gs', q_h, K) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('gs,sh->gh', attn, V)
        output[kv_h * gqa_factor:(kv_h + 1) * gqa_factor] = out

    return output


def paged_attention_prefill(
    query: torch.Tensor,   # [seq_len, num_q_heads, head_dim]
    key: torch.Tensor,     # [seq_len, num_kv_heads, head_dim]
    value: torch.Tensor,   # [seq_len, num_kv_heads, head_dim]
    kv_cache: torch.Tensor,
    block_table: List[int],
    block_size: int,
) -> torch.Tensor:
    """Prefill：计算全序列Attention并写入KV Cache"""
    seq_len, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    gqa_factor = num_q_heads // num_kv_heads

    # 写入KV Cache
    for i in range(seq_len):
        blk = i // block_size
        slot = i % block_size
        kv_cache[0, block_table[blk], :, slot] = key[i]
        kv_cache[1, block_table[blk], :, slot] = value[i]

    # Causal self-attention
    Q = query.permute(1, 0, 2)
    K = key.permute(1, 0, 2).repeat_interleave(gqa_factor, dim=0)
    V = value.permute(1, 0, 2).repeat_interleave(gqa_factor, dim=0)

    scores = torch.bmm(Q, K.transpose(1, 2)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device),
                      diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = torch.bmm(attn, V)
    return out.permute(1, 0, 2)
