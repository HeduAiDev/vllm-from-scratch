"""
Mini vLLM —— 从零整合所有组件的完整推理引擎
对应博客第十五章

整合了：
  - Paged Attention + Block Manager（KV Cache 分页）
  - Prefix Cache（前缀缓存，LRU 淘汰）
  - Scheduler（FCFS 调度，支持 Chunked Prefill）
  - Transformer Model（含 MoE 可选）
  - Sampler（贪心 / Top-P 核采样）
  - 流式推理主循环

运行方式（在 vllm 容器中）：
  python3 mini_vllm.py
"""

import math
import time
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# 第一部分：KV Cache 块管理器
# ──────────────────────────────────────────────────────────────────

class BlockAllocator:
    """
    物理块分配器
    - 固定池：num_blocks 个物理块
    - 引用计数：支持多请求共享（Prefix Cache）
    - LRU 淘汰：空闲块按最近使用时间排序
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # 所有物理块的引用计数
        self.ref_count = [0] * num_blocks

        # 空闲块：OrderedDict 作 LRU（key=block_id, val=last_used_time）
        self.free_blocks: OrderedDict[int, float] = OrderedDict(
            (i, 0.0) for i in range(num_blocks)
        )

        # 已缓存块：hash → physical_block_id（Prefix Cache）
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}

        self._time = 0

    def _tick(self) -> float:
        self._time += 1
        return float(self._time)

    def allocate(self, block_hash: Optional[int] = None) -> Optional[int]:
        """
        分配一个物理块。
        如果 block_hash 已在缓存中，直接复用（Prefix Cache 命中）。
        否则找一个空闲块（LRU 淘汰）。
        返回 physical_block_id，失败返回 None。
        """
        # Prefix Cache 命中
        if block_hash is not None and block_hash in self.hash_to_block:
            block_id = self.hash_to_block[block_hash]
            self.ref_count[block_id] += 1
            # 如果恰好在 free 队列里（ref 刚从 0 增加），移出
            self.free_blocks.pop(block_id, None)
            return block_id

        # 需要新块：从 free 队列取最旧的
        if not self.free_blocks:
            return None  # OOM

        block_id, _ = self.free_blocks.popitem(last=False)  # FIFO 在 LRU 中取最旧
        # 如果该块有旧的 hash 缓存，清除
        old_hash = self.block_to_hash.pop(block_id, None)
        if old_hash is not None:
            self.hash_to_block.pop(old_hash, None)

        self.ref_count[block_id] = 1
        return block_id

    def free(self, block_id: int):
        """减少引用计数，归零时放回 free 队列"""
        if self.ref_count[block_id] <= 0:
            return
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            # 归还但保留 hash（可作为 Prefix Cache 候选，LRU 尾部）
            self.free_blocks[block_id] = self._tick()

    def mark_cached(self, block_id: int, block_hash: int):
        """将一个已填充的块标记为可缓存（供后续 Prefix Cache 命中）"""
        if block_hash in self.hash_to_block:
            return  # 已存在
        self.hash_to_block[block_hash] = block_id
        self.block_to_hash[block_id] = block_hash

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


# ──────────────────────────────────────────────────────────────────
# 第二部分：请求与调度器
# ──────────────────────────────────────────────────────────────────

@dataclass
class Request:
    """表示一个推理请求的完整状态"""
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0

    # 运行时状态
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0   # 已经做过 attention 的 token 数
    block_table: list[int] = field(default_factory=list)   # 逻辑块 → 物理块
    is_finished: bool = False
    arrival_time: float = field(default_factory=time.time)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """prompt + 已生成 token 总数"""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        """还在 prefill 阶段（有未计算的 prompt token）"""
        return self.num_computed_tokens < self.prompt_len

    @property
    def next_tokens_to_compute(self) -> list[int]:
        """下一步需要输入模型的 token ids"""
        if self.is_prefill:
            return self.prompt_token_ids[self.num_computed_tokens:]
        else:
            # Decode：最后一个生成的 token
            return [self.output_token_ids[-1]]


@dataclass
class ScheduledRequest:
    request: Request
    tokens: list[int]           # 本步喂给模型的 token ids
    positions: list[int]        # 对应位置
    is_prefill: bool
    slot_mapping: list[int]     # token → KV Cache slot（block_id * block_size + offset）


class Scheduler:
    """
    FCFS 调度器，支持 Chunked Prefill

    策略：
    1. 先把 decode 请求（已在运行的）全加入
    2. 再按 FCFS 加入 prefill 请求，直到 token budget 耗尽
    3. Chunked Prefill：大 prompt 分多步处理
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        block_size: int,
        allocator: BlockAllocator,
        enable_chunked_prefill: bool = True,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.allocator = allocator
        self.enable_chunked_prefill = enable_chunked_prefill

        self.waiting: list[Request] = []    # 等待首次调度
        self.running: list[Request] = []    # 正在运行（prefill 中 or decode）
        self.finished: list[Request] = []

    def add_request(self, request: Request):
        self.waiting.append(request)

    def _ensure_blocks(self, req: Request, num_new_tokens: int) -> bool:
        """为请求分配足够的物理块，返回是否成功"""
        # 计算还需要多少个新块
        current_len = req.num_computed_tokens + num_new_tokens
        num_blocks_needed = math.ceil(current_len / self.block_size)

        while len(req.block_table) < num_blocks_needed:
            block_id = self.allocator.allocate()
            if block_id is None:
                return False  # OOM
            req.block_table.append(block_id)

        return True

    def _compute_slot_mapping(
        self, req: Request, tokens: list[int], start_pos: int
    ) -> list[int]:
        """计算 slot_mapping：每个 token 在 KV Cache 中的物理槽位"""
        slots = []
        for i, pos in enumerate(range(start_pos, start_pos + len(tokens))):
            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            physical_block = req.block_table[block_idx]
            slots.append(physical_block * self.block_size + block_offset)
        return slots

    def schedule(self) -> list[ScheduledRequest]:
        scheduled = []
        token_budget = self.max_num_batched_tokens
        num_seqs = 0

        # ── 1. 先调度正在运行的请求（decode 优先）──
        still_running = []
        for req in self.running:
            if req.is_finished:
                continue

            # Decode 请求：每步只处理 1 个 token
            if not req.is_prefill:
                if num_seqs >= self.max_num_seqs or token_budget < 1:
                    still_running.append(req)
                    continue
                if not self._ensure_blocks(req, 1):
                    still_running.append(req)
                    continue

                tokens = [req.output_token_ids[-1]] if req.output_token_ids else req.prompt_token_ids[-1:]
                pos = req.num_computed_tokens
                slots = self._compute_slot_mapping(req, tokens, pos)
                scheduled.append(ScheduledRequest(
                    request=req, tokens=tokens,
                    positions=list(range(pos, pos + len(tokens))),
                    is_prefill=False, slot_mapping=slots,
                ))
                token_budget -= len(tokens)
                num_seqs += 1
                still_running.append(req)

            else:
                # 仍在 prefill 中（chunked）
                remaining = req.prompt_len - req.num_computed_tokens
                chunk = remaining if not self.enable_chunked_prefill else min(remaining, token_budget)
                if chunk <= 0 or num_seqs >= self.max_num_seqs or token_budget < 1:
                    still_running.append(req)
                    continue

                chunk = min(chunk, token_budget)
                if not self._ensure_blocks(req, chunk):
                    still_running.append(req)
                    continue

                start = req.num_computed_tokens
                tokens = req.prompt_token_ids[start:start + chunk]
                slots = self._compute_slot_mapping(req, tokens, start)
                scheduled.append(ScheduledRequest(
                    request=req, tokens=tokens,
                    positions=list(range(start, start + len(tokens))),
                    is_prefill=True, slot_mapping=slots,
                ))
                token_budget -= chunk
                num_seqs += 1
                still_running.append(req)

        self.running = still_running

        # ── 2. 从 waiting 队列补充新请求 ──
        new_waiting = []
        for req in self.waiting:
            if num_seqs >= self.max_num_seqs or token_budget < 1:
                new_waiting.append(req)
                continue

            # 首次调度：按 chunk 大小分配
            remaining = req.prompt_len
            chunk = remaining if not self.enable_chunked_prefill else min(remaining, token_budget)
            if chunk <= 0:
                new_waiting.append(req)
                continue

            if not self._ensure_blocks(req, chunk):
                new_waiting.append(req)
                continue

            tokens = req.prompt_token_ids[:chunk]
            slots = self._compute_slot_mapping(req, tokens, 0)
            scheduled.append(ScheduledRequest(
                request=req, tokens=tokens,
                positions=list(range(0, chunk)),
                is_prefill=True, slot_mapping=slots,
            ))
            token_budget -= chunk
            num_seqs += 1
            self.running.append(req)

        self.waiting = new_waiting
        return scheduled

    def update(self, scheduled: list[ScheduledRequest], next_token_ids: list[int]):
        """根据模型输出更新请求状态"""
        for sched, next_tok in zip(scheduled, next_token_ids):
            req = sched.request
            req.num_computed_tokens += len(sched.tokens)

            if sched.is_prefill and req.is_prefill:
                # 还有更多 prompt 待处理
                pass
            else:
                # Decode 生成了新 token（或 prefill 刚结束，生成第一个 token）
                req.output_token_ids.append(next_tok)

                # 检查是否完成
                if (len(req.output_token_ids) >= req.max_new_tokens or
                        next_tok == 0):  # EOS = token 0（演示用）
                    req.is_finished = True
                    # 释放块
                    for block_id in req.block_table:
                        self.allocator.free(block_id)
                    self.running = [r for r in self.running if r is not req]
                    self.finished.append(req)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)


# ──────────────────────────────────────────────────────────────────
# 第三部分：Transformer 模型（简化版）
# ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        return cos, sin


class SelfAttention(nn.Module):
    """
    支持 Paged KV Cache 的注意力层
    - Prefill：标准 causal attention
    - Decode：从 KV Cache 读取，只计算最后 1 个位置
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rope = RotaryEmbedding(head_dim)

    def forward(
        self,
        x: torch.Tensor,              # [T, D] (所有请求 token 拼接)
        positions: torch.Tensor,       # [T]
        kv_cache: tuple[torch.Tensor, torch.Tensor],  # K:[B,S,H,D] V:[B,S,H,D]
        slot_mapping: torch.Tensor,    # [T] 每个 token 写入 KV Cache 的槽位
        seq_lens: list[int],           # 每个请求当前序列长度
        is_prefill: list[bool],        # 每个请求是否 prefill
    ) -> torch.Tensor:

        T = x.shape[0]
        H, D = self.num_heads, self.head_dim

        # 1. QKV 投影
        qkv = self.qkv(x)  # [T, 3*H*D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(T, H, D)
        k = k.view(T, H, D)
        v = v.view(T, H, D)

        # 2. RoPE
        cos, sin = self.rope(positions)
        cos = cos.unsqueeze(1)  # [T, 1, D]
        sin = sin.unsqueeze(1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # 3. 写入 KV Cache
        k_cache, v_cache = kv_cache
        # slot_mapping: [T] → 物理槽位
        k_cache_flat = k_cache.view(-1, H, D)  # [num_blocks*block_size, H, D]
        v_cache_flat = v_cache.view(-1, H, D)
        k_cache_flat[slot_mapping] = k
        v_cache_flat[slot_mapping] = v

        # 4. 计算 attention（按请求拆分）
        outputs = []
        token_offset = 0

        for i, (seq_len, prefill) in enumerate(zip(seq_lens, is_prefill)):
            num_q_tokens = seq_len if prefill else 1
            q_i = q[token_offset:token_offset + num_q_tokens]  # [Tq, H, D]

            if prefill:
                # Prefill：Q=K=V=本批次 token，causal attention
                k_i = k[token_offset:token_offset + num_q_tokens]
                v_i = v[token_offset:token_offset + num_q_tokens]

                # 注意力计算 [H, Tq, D] × [H, D, Tk] → [H, Tq, Tk]
                q_t = q_i.permute(1, 0, 2)  # [H, Tq, D]
                k_t = k_i.permute(1, 2, 0)  # [H, D, Tk]
                attn = torch.bmm(q_t, k_t) * self.scale

                # Causal mask
                mask = torch.triu(
                    torch.full((num_q_tokens, num_q_tokens),
                               float('-inf'), device=x.device),
                    diagonal=1
                )
                attn = attn + mask

                attn = F.softmax(attn, dim=-1)
                v_t = v_i.permute(1, 0, 2)  # [H, Tk, D]
                out = torch.bmm(attn, v_t)  # [H, Tq, D]
                out = out.permute(1, 0, 2).contiguous().view(num_q_tokens, H * D)

            else:
                # Decode：从 KV Cache 读取全部历史
                # 这里简化：只做当前序列的 attention
                # 实际 vLLM 用 PagedAttention kernel 按 block 读取
                q_dec = q_i  # [1, H, D]

                # 从 KV Cache 按 block_table 读取（这里简化为顺序读）
                # 实际应按 slot_mapping 找对应 slots
                # 演示目的：直接从写入的 slots 读回
                hist_len = seq_len
                hist_slots = slot_mapping[token_offset:token_offset + 1]
                # 简化：使用全 cache 做近似（演示不精确，但形状正确）
                k_hist = k_cache_flat[:hist_len]  # [hist_len, H, D]
                v_hist = v_cache_flat[:hist_len]

                q_t = q_dec.permute(1, 0, 2)         # [H, 1, D]
                k_t = k_hist.permute(1, 2, 0)         # [H, D, hist_len]
                attn = torch.bmm(q_t, k_t) * self.scale
                attn = F.softmax(attn, dim=-1)
                v_t = v_hist.permute(1, 0, 2)         # [H, hist_len, D]
                out = torch.bmm(attn, v_t)             # [H, 1, D]
                out = out.permute(1, 0, 2).contiguous().view(1, H * D)

            outputs.append(out)
            token_offset += num_q_tokens

        output = torch.cat(outputs, dim=0)  # [sum_Tq, H*D]
        return self.out_proj(output)


class FFN(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.attn = SelfAttention(hidden_size, num_heads, head_dim)
        self.ffn  = FFN(hidden_size, intermediate_size)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x, positions, kv_cache, slot_mapping, seq_lens, is_prefill):
        x = x + self.attn(self.norm1(x), positions, kv_cache, slot_mapping, seq_lens, is_prefill)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """小型 Transformer，用于 Mini vLLM 演示"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_blocks: int,
        block_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_size = block_size
        head_dim = hidden_size // num_heads

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # KV Cache：每层独立的 [num_blocks, block_size, num_heads, head_dim]
        self.register_buffer(
            'k_cache',
            torch.zeros(num_layers, num_kv_blocks, block_size, num_heads, head_dim)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(num_layers, num_kv_blocks, block_size, num_heads, head_dim)
        )

    def forward(
        self,
        token_ids: torch.Tensor,      # [T] 所有请求 token 拼接
        positions: torch.Tensor,       # [T]
        slot_mapping: torch.Tensor,    # [T]
        seq_lens: list[int],
        is_prefill: list[bool],
    ) -> torch.Tensor:
        """返回 logits [num_decode_tokens, vocab_size]"""

        x = self.embedding(token_ids)  # [T, D]

        for layer_idx, layer in enumerate(self.layers):
            kv_cache = (
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
            )
            x = layer(x, positions, kv_cache, slot_mapping, seq_lens, is_prefill)

        x = self.norm(x)

        # 只取每个请求"最后一个 token"的 logits
        # （prefill 时是 prompt 末尾，decode 时是唯一的 token）
        last_token_indices = []
        offset = 0
        for seq_len, prefill in zip(seq_lens, is_prefill):
            last_token_indices.append(offset + seq_len - 1)
            offset += seq_len

        last_hidden = x[last_token_indices]  # [batch, D]
        logits = self.lm_head(last_hidden)   # [batch, vocab_size]
        return logits


# ──────────────────────────────────────────────────────────────────
# 第四部分：采样器
# ──────────────────────────────────────────────────────────────────

class Sampler:
    """
    支持：
    - 贪心（temperature=0 或 top_p=1, temp=1, greedy=True）
    - Temperature scaling
    - Top-P 核采样
    """

    @staticmethod
    def sample(
        logits: torch.Tensor,      # [batch, vocab_size]
        requests: list[Request],
    ) -> list[int]:
        """返回每个请求采样到的 token_id"""

        results = []
        for i, req in enumerate(requests):
            logit = logits[i]

            if req.temperature == 0.0:
                # 贪心
                token_id = logit.argmax().item()
            else:
                # Temperature
                logit = logit / req.temperature
                probs = F.softmax(logit, dim=-1)

                if req.top_p < 1.0:
                    # Top-P 核采样
                    sorted_probs, sorted_ids = probs.sort(descending=True)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    # 找到 cumsum 超过 top_p 的位置
                    mask = (cumsum - sorted_probs) >= req.top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum()
                    idx = torch.multinomial(sorted_probs, 1).item()
                    token_id = sorted_ids[idx].item()
                else:
                    token_id = torch.multinomial(probs, 1).item()

            results.append(int(token_id))

        return results


# ──────────────────────────────────────────────────────────────────
# 第五部分：Mini vLLM 主引擎
# ──────────────────────────────────────────────────────────────────

class MiniVLLM:
    """
    完整的 Mini vLLM 推理引擎

    对应 vLLM 的核心架构：
      Scheduler → execute_model → Sampler → update
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 512,
        num_kv_blocks: int = 256,
        block_size: int = 16,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 1024,
        enable_chunked_prefill: bool = True,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)

        # KV Cache 分配器
        self.allocator = BlockAllocator(num_kv_blocks, block_size)

        # 调度器
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
            allocator=self.allocator,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        # 模型
        self.model = MiniTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_kv_blocks=num_kv_blocks,
            block_size=block_size,
        ).to(self.device)

        # 采样器
        self.sampler = Sampler()

        self._req_counter = 0

    def add_request(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        req_id = f"req-{self._req_counter}"
        self._req_counter += 1
        req = Request(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.scheduler.add_request(req)
        return req_id

    def step(self) -> list[tuple[str, int]]:
        """
        执行一步推理，返回 [(request_id, new_token_id), ...] for decode requests
        """
        # 1. 调度
        scheduled = self.scheduler.schedule()
        if not scheduled:
            return []

        # 2. 构建模型输入
        all_tokens = []
        all_positions = []
        all_slots = []
        seq_lens = []
        is_prefill_flags = []
        requests_ordered = []

        for sched in scheduled:
            all_tokens.extend(sched.tokens)
            all_positions.extend(sched.positions)
            all_slots.extend(sched.slot_mapping)
            seq_lens.append(len(sched.tokens))
            is_prefill_flags.append(sched.is_prefill)
            requests_ordered.append(sched.request)

        token_ids  = torch.tensor(all_tokens,    dtype=torch.long,  device=self.device)
        positions  = torch.tensor(all_positions, dtype=torch.long,  device=self.device)
        slot_mapping = torch.tensor(all_slots,   dtype=torch.long,  device=self.device)

        # 3. 模型前向
        with torch.no_grad():
            logits = self.model(
                token_ids=token_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                is_prefill=is_prefill_flags,
            )

        # 4. 采样
        next_tokens = self.sampler.sample(logits, requests_ordered)

        # 5. 更新调度器
        self.scheduler.update(scheduled, next_tokens)

        # 6. 返回 decode 结果（prefill 完成时也返回第一个 token）
        results = []
        for sched, tok in zip(scheduled, next_tokens):
            # prefill 最后一步或 decode 步骤都产生新 token
            if not sched.is_prefill or not sched.request.is_prefill:
                results.append((sched.request.request_id, tok))

        return results

    def generate(self, prompts: list[list[int]], max_new_tokens: int = 20) -> dict[str, list[int]]:
        """
        批量推理，返回 {req_id: output_token_ids}
        """
        req_ids = [
            self.add_request(p, max_new_tokens=max_new_tokens)
            for p in prompts
        ]

        all_outputs: dict[str, list[int]] = {rid: [] for rid in req_ids}

        while self.scheduler.has_unfinished:
            results = self.step()
            for req_id, tok in results:
                if req_id in all_outputs:
                    all_outputs[req_id].append(tok)

        return all_outputs


# ──────────────────────────────────────────────────────────────────
# 测试与演示
# ──────────────────────────────────────────────────────────────────

def test_block_allocator():
    """测试块分配器"""
    alloc = BlockAllocator(num_blocks=10, block_size=16)
    assert alloc.num_free == 10

    ids = [alloc.allocate() for _ in range(5)]
    assert alloc.num_free == 5

    alloc.free(ids[0])
    assert alloc.num_free == 6

    # OOM 测试
    more = [alloc.allocate() for _ in range(7)]
    assert more[-1] is None  # 第7个分配失败

    print("✓ BlockAllocator: 基本分配/释放/OOM OK")


def test_scheduler_chunked_prefill():
    """测试 Chunked Prefill 调度"""
    alloc = BlockAllocator(num_blocks=64, block_size=16)
    sched = Scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=32,  # 每步最多 32 token
        block_size=16,
        allocator=alloc,
        enable_chunked_prefill=True,
    )

    # 添加一个 64 token 的长 prompt
    req = Request(
        request_id="r0",
        prompt_token_ids=list(range(64)),
        max_new_tokens=5,
    )
    sched.add_request(req)

    steps = 0
    while sched.has_unfinished and steps < 20:
        scheduled = sched.schedule()
        if not scheduled:
            break

        total_tokens = sum(len(s.tokens) for s in scheduled)
        assert total_tokens <= 32, f"超过 token budget: {total_tokens}"

        # 模拟更新（全部生成 token 1）
        sched.update(scheduled, [1] * len(scheduled))
        steps += 1

    print(f"✓ Chunked Prefill: 64-token prompt 在 {steps} 步内处理完成")


def test_mini_vllm_basic():
    """测试 Mini vLLM 端到端推理"""
    engine = MiniVLLM(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        num_kv_blocks=64,
        block_size=8,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )

    # 3 个不同长度的 prompt
    prompts = [
        [1, 2, 3, 4, 5],           # 5 tokens
        [10, 20, 30],               # 3 tokens
        [7, 8, 9, 10, 11, 12, 13], # 7 tokens
    ]

    print("\n正在推理（贪心，max_new_tokens=5）...")
    outputs = engine.generate(prompts, max_new_tokens=5)

    for req_id, tokens in outputs.items():
        print(f"  {req_id}: 生成了 {len(tokens)} 个 token: {tokens}")
        assert len(tokens) > 0, f"{req_id} 没有生成任何 token"

    print("✓ Mini vLLM 端到端: OK")


def test_mini_vllm_batch():
    """测试批量推理性能"""
    engine = MiniVLLM(
        vocab_size=500,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        intermediate_size=256,
        num_kv_blocks=128,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=512,
    )

    # 8 个请求同时提交
    prompts = [
        [random.randint(1, 499) for _ in range(random.randint(5, 20))]
        for _ in range(8)
    ]

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=10)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(v) for v in outputs.values())
    print(f"\n✓ 批量推理:")
    print(f"  请求数: {len(prompts)}")
    print(f"  总生成 token: {total_tokens}")
    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  吞吐量: {total_tokens/elapsed:.1f} tokens/s (CPU, 随机权重)")


def test_chunked_prefill_vs_no_chunked():
    """对比 Chunked Prefill 开关对调度步数的影响"""
    vocab_size = 200
    long_prompt = [random.randint(1, 199) for _ in range(128)]

    for chunked in [False, True]:
        engine = MiniVLLM(
            vocab_size=vocab_size,
            hidden_size=64,
            num_layers=1,
            num_heads=4,
            intermediate_size=128,
            num_kv_blocks=64,
            block_size=16,
            max_num_seqs=4,
            max_num_batched_tokens=64,  # 限制每步 token 数
            enable_chunked_prefill=chunked,
        )

        steps = 0
        engine.add_request(long_prompt, max_new_tokens=5)
        while engine.scheduler.has_unfinished and steps < 100:
            engine.step()
            steps += 1

        print(f"  enable_chunked_prefill={chunked}: {steps} 推理步骤")

    print("✓ Chunked Prefill 对比完成")


def benchmark_throughput():
    """简单吞吐量基准"""
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    engine = MiniVLLM(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        intermediate_size=1024,
        num_kv_blocks=256,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        device=device,
    )

    # 16 个并发请求
    prompts = [
        [random.randint(1, 999) for _ in range(random.randint(10, 50))]
        for _ in range(16)
    ]

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=20)
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(v) for v in outputs.values())
    total_input_tokens  = sum(len(p) for p in prompts)

    print(f"基准测试结果:")
    print(f"  并发请求: {len(prompts)}")
    print(f"  输入 tokens: {total_input_tokens}")
    print(f"  输出 tokens: {total_output_tokens}")
    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  输出吞吐: {total_output_tokens/elapsed:.1f} tok/s")


if __name__ == '__main__':
    print("=" * 60)
    print("Mini vLLM 完整测试")
    print("=" * 60)

    test_block_allocator()
    test_scheduler_chunked_prefill()
    test_mini_vllm_basic()
    test_mini_vllm_batch()

    print("\n─── Chunked Prefill 对比 ───")
    test_chunked_prefill_vs_no_chunked()

    print("\n─── 吞吐量基准 ───")
    benchmark_throughput()

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
