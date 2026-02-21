"""
Mini vLLM pytest 测试套件
运行：docker exec vllm python3 -m pytest 05_mini_vllm/test_mini_vllm.py -v
"""
import math
import random
import torch
import pytest
from mini_vllm import (
    BlockAllocator, Request, Scheduler, Sampler,
    MiniTransformer, MiniVLLM
)


# ──────────────────────────────────────────────────────────────────
# BlockAllocator 测试
# ──────────────────────────────────────────────────────────────────

class TestBlockAllocator:
    def test_initial_free_count(self):
        alloc = BlockAllocator(num_blocks=10, block_size=16)
        assert alloc.num_free == 10

    def test_allocate_reduces_free(self):
        alloc = BlockAllocator(num_blocks=10, block_size=16)
        block = alloc.allocate()
        assert block is not None
        assert alloc.num_free == 9

    def test_free_restores_count(self):
        alloc = BlockAllocator(num_blocks=5, block_size=8)
        blocks = [alloc.allocate() for _ in range(5)]
        assert alloc.num_free == 0
        alloc.free(blocks[0])
        assert alloc.num_free == 1

    def test_oom_returns_none(self):
        alloc = BlockAllocator(num_blocks=3, block_size=8)
        [alloc.allocate() for _ in range(3)]
        result = alloc.allocate()
        assert result is None

    def test_prefix_cache_reuse(self):
        """带 hash 的块可以被复用（ref_count += 1）"""
        alloc = BlockAllocator(num_blocks=10, block_size=16)
        block_id = alloc.allocate()
        alloc.mark_cached(block_id, block_hash=42)
        alloc.free(block_id)
        # 现在引用计数为 0，进入 free 队列

        # 再次用相同 hash 分配，应该复用
        reused_id = alloc.allocate(block_hash=42)
        assert reused_id == block_id  # 复用了同一个块

    def test_all_blocks_allocatable(self):
        n = 20
        alloc = BlockAllocator(num_blocks=n, block_size=16)
        blocks = []
        for _ in range(n):
            b = alloc.allocate()
            assert b is not None
            blocks.append(b)
        assert len(set(blocks)) == n  # 所有块 ID 唯一

    def test_free_and_reallocate(self):
        alloc = BlockAllocator(num_blocks=4, block_size=8)
        ids = [alloc.allocate() for _ in range(4)]
        for b in ids:
            alloc.free(b)
        # 全部释放后可以再次分配
        new_ids = [alloc.allocate() for _ in range(4)]
        assert all(b is not None for b in new_ids)


# ──────────────────────────────────────────────────────────────────
# Scheduler 测试
# ──────────────────────────────────────────────────────────────────

def make_scheduler(max_batched=64, chunked=True, num_blocks=64):
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=16)
    return Scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=max_batched,
        block_size=16,
        allocator=alloc,
        enable_chunked_prefill=chunked,
    ), alloc


class TestScheduler:
    def test_empty_schedule(self):
        sched, _ = make_scheduler()
        result = sched.schedule()
        assert result == []
        assert not sched.has_unfinished

    def test_single_short_prefill(self):
        sched, _ = make_scheduler(max_batched=64)
        req = Request("r0", [1, 2, 3], max_new_tokens=5)
        sched.add_request(req)

        scheduled = sched.schedule()
        assert len(scheduled) == 1
        assert scheduled[0].tokens == [1, 2, 3]
        assert scheduled[0].is_prefill

    def test_token_budget_respected(self):
        sched, _ = make_scheduler(max_batched=32, chunked=True)
        # 64 token prompt，每步只能处理 32
        req = Request("r0", list(range(64)), max_new_tokens=5)
        sched.add_request(req)

        scheduled = sched.schedule()
        total = sum(len(s.tokens) for s in scheduled)
        assert total <= 32

    def test_chunked_prefill_progression(self):
        """Chunked Prefill：多步处理完整 prompt"""
        sched, _ = make_scheduler(max_batched=16, chunked=True)
        req = Request("r0", list(range(64)), max_new_tokens=0)
        sched.add_request(req)

        steps = 0
        while sched.has_unfinished and steps < 20:
            scheduled = sched.schedule()
            if not scheduled:
                break
            # 模拟：prefill 完成后请求结束（max_new_tokens=0 特殊处理）
            for s in scheduled:
                s.request.num_computed_tokens += len(s.tokens)
                if not s.request.is_prefill:
                    s.request.is_finished = True
            sched.running = [r for r in sched.running if not r.is_finished]
            steps += 1

        assert steps >= 4  # 64 tokens / 16 per step = 4 步

    def test_max_num_seqs_limit(self):
        alloc = BlockAllocator(num_blocks=128, block_size=16)
        sched = Scheduler(
            max_num_seqs=2,
            max_num_batched_tokens=256,
            block_size=16,
            allocator=alloc,
        )
        for i in range(5):
            sched.add_request(Request(f"r{i}", [1, 2, 3], max_new_tokens=5))

        scheduled = sched.schedule()
        assert len(scheduled) <= 2

    def test_slot_mapping_valid(self):
        """slot_mapping 中的槽位不超过 KV Cache 容量"""
        alloc = BlockAllocator(num_blocks=16, block_size=8)
        sched = Scheduler(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            block_size=8,
            allocator=alloc,
        )
        req = Request("r0", list(range(16)), max_new_tokens=5)
        sched.add_request(req)
        scheduled = sched.schedule()

        max_slot = alloc.num_blocks * alloc.block_size
        for s in scheduled:
            for slot in s.slot_mapping:
                assert 0 <= slot < max_slot, f"非法 slot: {slot}"

    def test_multiple_requests_batched(self):
        sched, _ = make_scheduler(max_batched=256)
        for i in range(4):
            sched.add_request(Request(f"r{i}", [i+1, i+2, i+3], max_new_tokens=5))

        scheduled = sched.schedule()
        assert len(scheduled) == 4


# ──────────────────────────────────────────────────────────────────
# Sampler 测试
# ──────────────────────────────────────────────────────────────────

class TestSampler:
    def test_greedy(self):
        """Temperature=0 应该贪心选最大 logit"""
        logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        reqs = [
            Request("r0", [1], max_new_tokens=5, temperature=0.0),
            Request("r1", [1], max_new_tokens=5, temperature=0.0),
        ]
        tokens = Sampler.sample(logits, reqs)
        assert tokens[0] == 1  # argmax([1,5,2]) = 1
        assert tokens[1] == 2  # argmax([3,1,4]) = 2

    def test_temperature_affects_distribution(self):
        """高温度应该使分布更均匀"""
        torch.manual_seed(42)
        vocab = 100
        logits = torch.randn(1, vocab)

        req_low = Request("r0", [1], max_new_tokens=5, temperature=0.1)
        req_high = Request("r1", [1], max_new_tokens=5, temperature=10.0)

        samples_low  = [Sampler.sample(logits, [req_low])[0]  for _ in range(200)]
        samples_high = [Sampler.sample(logits, [req_high])[0] for _ in range(200)]

        # 低温度应该集中在几个 token；高温度分散
        unique_low  = len(set(samples_low))
        unique_high = len(set(samples_high))
        assert unique_low < unique_high, \
            f"低温度 {unique_low} 种类 >= 高温度 {unique_high} 种类"

    def test_top_p_filters_low_prob(self):
        """Top-P 应该过滤掉低概率 token"""
        torch.manual_seed(0)
        # 第 0 个 token 概率极高（logit=100），其余接近 0
        logits = torch.zeros(1, 10)
        logits[0, 0] = 100.0

        req = Request("r0", [1], max_new_tokens=5, temperature=1.0, top_p=0.9)
        for _ in range(20):
            token = Sampler.sample(logits, [req])[0]
            assert token == 0, f"Top-P 没有过滤，采样到了 token {token}"

    def test_valid_token_ids(self):
        vocab_size = 500
        logits = torch.randn(3, vocab_size)
        reqs = [
            Request(f"r{i}", [1], max_new_tokens=5, temperature=1.0)
            for i in range(3)
        ]
        tokens = Sampler.sample(logits, reqs)
        for tok in tokens:
            assert 0 <= tok < vocab_size


# ──────────────────────────────────────────────────────────────────
# MiniVLLM 端到端测试
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_engine():
    return MiniVLLM(
        vocab_size=200,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        num_kv_blocks=64,
        block_size=8,
        max_num_seqs=4,
        max_num_batched_tokens=128,
    )


class TestMiniVLLM:
    def test_single_request(self, small_engine):
        outputs = small_engine.generate([[1, 2, 3, 4, 5]], max_new_tokens=5)
        assert len(outputs) == 1
        tokens = list(outputs.values())[0]
        assert len(tokens) > 0
        assert len(tokens) <= 5

    def test_multiple_requests(self, small_engine):
        prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        outputs = small_engine.generate(prompts, max_new_tokens=3)
        assert len(outputs) == 3
        for tokens in outputs.values():
            assert len(tokens) > 0

    def test_output_token_ids_valid(self, small_engine):
        outputs = small_engine.generate([[10, 20, 30]], max_new_tokens=10)
        for tokens in outputs.values():
            for tok in tokens:
                assert 0 <= tok < 200  # vocab_size

    def test_max_new_tokens_respected(self, small_engine):
        MAX = 7
        outputs = small_engine.generate([[1, 2, 3, 4, 5]], max_new_tokens=MAX)
        for tokens in outputs.values():
            assert len(tokens) <= MAX, f"生成了 {len(tokens)} 个 token，超过 max={MAX}"

    def test_different_prompt_lengths(self, small_engine):
        """不同长度 prompt 可以正常批处理"""
        prompts = [
            [1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [5, 6, 7],
        ]
        outputs = small_engine.generate(prompts, max_new_tokens=5)
        assert len(outputs) == 3

    def test_engine_reuse(self, small_engine):
        """同一个引擎可以多次推理"""
        for _ in range(3):
            outputs = small_engine.generate([[1, 2, 3]], max_new_tokens=3)
            assert len(outputs) == 1

    def test_greedy_deterministic(self):
        """贪心采样：同一引擎两次独立调用，只要 prefill 输出相同就是确定性的"""
        engine = MiniVLLM(
            vocab_size=100, hidden_size=64, num_layers=1, num_heads=4,
            intermediate_size=128, num_kv_blocks=64, block_size=8,
        )
        # 两次单独 prefill（只做 prefill，不做 decode），输出 logits 应相同
        # 验证：相同权重 + 相同输入 → 相同 prefill logits
        import torch
        prompt_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        positions  = torch.arange(4, dtype=torch.long)
        slots      = torch.arange(4, dtype=torch.long)

        with torch.no_grad():
            logits1 = engine.model(prompt_ids, positions, slots, [4], [True])
            # 重置 KV cache（仅 prefill 测试不需要 decode，直接重用相同槽位）
            engine.model.k_cache.zero_()
            engine.model.v_cache.zero_()
            logits2 = engine.model(prompt_ids, positions, slots, [4], [True])

        assert torch.allclose(logits1, logits2, atol=1e-6), \
            "相同输入下 prefill logits 应完全一致（贪心确定性）"

    def test_chunked_prefill_produces_output(self):
        """开关 Chunked Prefill，都能正常完成推理并产生输出"""
        common = dict(
            vocab_size=100, hidden_size=64, num_layers=1, num_heads=4,
            intermediate_size=128, num_kv_blocks=64, block_size=8,
            max_num_seqs=4, max_num_batched_tokens=32,
        )
        for chunked in [True, False]:
            eng = MiniVLLM(**common, enable_chunked_prefill=chunked)
            out = eng.generate([[1, 2, 3]], max_new_tokens=3)
            tokens = list(out.values())[0]
            assert len(tokens) > 0, f"chunked={chunked} 时没有生成 token"
            assert len(tokens) <= 3
