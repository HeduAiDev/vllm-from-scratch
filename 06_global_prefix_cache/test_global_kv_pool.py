"""
全局 KV Cache 池 pytest 测试套件
运行：docker exec vllm python3 -m pytest 06_global_prefix_cache/test_global_kv_pool.py -v
"""
import threading
import time
import pytest
from global_kv_pool import (
    BLOCK_SIZE, KV_BLOCK_BYTES,
    KVBlockMeta, GlobalMetadataServer, TransferEngine,
    MooncakeConnector, SimulatedCluster,
    compute_block_hash, compute_block_hashes,
)


# ──────────────────────────────────────────────────────────────────
# compute_block_hashes 测试
# ──────────────────────────────────────────────────────────────────

class TestBlockHash:
    def test_same_tokens_same_hash(self):
        tokens = list(range(BLOCK_SIZE))
        h1 = compute_block_hashes(tokens)
        h2 = compute_block_hashes(tokens)
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        h1 = compute_block_hashes(list(range(BLOCK_SIZE)))
        h2 = compute_block_hashes(list(range(BLOCK_SIZE, 2 * BLOCK_SIZE)))
        assert h1 != h2

    def test_chain_dependency(self):
        """不同前缀的相同 token 块，哈希应不同（链式依赖）"""
        tokens_a = list(range(BLOCK_SIZE * 2))
        tokens_b = list(range(1000, 1000 + BLOCK_SIZE)) + list(range(BLOCK_SIZE))

        hashes_a = compute_block_hashes(tokens_a)
        hashes_b = compute_block_hashes(tokens_b)

        # 第二个块内容相同，但前缀不同，hash 应该不同
        assert hashes_a[1] != hashes_b[1]

    def test_num_blocks(self):
        tokens = list(range(BLOCK_SIZE * 3 + 5))  # 3 完整块 + 剩余（不足一块）
        hashes = compute_block_hashes(tokens)
        assert len(hashes) == 3  # 只有完整块才计算哈希

    def test_short_sequence(self):
        tokens = list(range(BLOCK_SIZE - 1))  # 不足一个块
        hashes = compute_block_hashes(tokens)
        assert len(hashes) == 0

    def test_extra_key_changes_hash(self):
        tokens = list(range(BLOCK_SIZE))
        h1 = compute_block_hashes(tokens, extra_key=None)
        h2 = compute_block_hashes(tokens, extra_key=42)
        assert h1 != h2


# ──────────────────────────────────────────────────────────────────
# GlobalMetadataServer 测试
# ──────────────────────────────────────────────────────────────────

class TestGlobalMetadataServer:
    @pytest.fixture
    def meta(self):
        return GlobalMetadataServer()

    def test_publish_and_query(self, meta):
        tokens = list(range(BLOCK_SIZE * 3))
        hashes = compute_block_hashes(tokens)
        tids = tuple(tokens[:BLOCK_SIZE])

        meta.publish(block_hash=hashes[0], node_id=0, token_ids=tids)
        num_matched, metas = meta.query_prefix([hashes[0]])
        assert num_matched == 1
        assert metas[0].block_hash == hashes[0]

    def test_miss_returns_zero(self, meta):
        num_matched, metas = meta.query_prefix([99999])
        assert num_matched == 0
        assert metas == []

    def test_chain_miss_stops_early(self, meta):
        tokens = list(range(BLOCK_SIZE * 4))
        hashes = compute_block_hashes(tokens)

        # 只发布块 0 和块 2（跳过块 1）
        meta.publish(hashes[0], 0, tuple(tokens[:BLOCK_SIZE]))
        meta.publish(hashes[2], 0, tuple(tokens[BLOCK_SIZE*2:BLOCK_SIZE*3]))

        # 查询 [h0, h1, h2]，块1 miss → 只匹配块0
        num_matched, _ = meta.query_prefix(hashes[:3])
        assert num_matched == 1

    def test_total_blocks(self, meta):
        for i in range(5):
            meta.publish(i, 0, tuple(range(BLOCK_SIZE)))
        assert meta.total_blocks == 5

    def test_unpublish(self, meta):
        meta.publish(123, 0, tuple(range(BLOCK_SIZE)))
        assert meta.total_blocks == 1
        ok = meta.unpublish(123)
        assert ok
        assert meta.total_blocks == 0

    def test_unpublish_nonexistent(self, meta):
        ok = meta.unpublish(99999)
        assert not ok

    def test_lru_eviction(self, meta):
        """超出容量时自动 LRU 淘汰"""
        MAX = 5
        for i in range(MAX + 2):
            meta.publish(i, node_id=0, token_ids=tuple(range(BLOCK_SIZE)),
                         max_blocks_per_node=MAX)

        # 节点 0 的块数不超过 MAX
        stats = meta.node_stats()
        assert stats[0]['num_blocks'] <= MAX

    def test_hit_rate(self, meta):
        tokens = list(range(BLOCK_SIZE * 2))
        hashes = compute_block_hashes(tokens)

        meta.publish(hashes[0], 0, tuple(tokens[:BLOCK_SIZE]))

        # 查 2 次（1 命中，1 miss）
        meta.query_prefix([hashes[0]])
        meta.query_prefix([99999])

        # 命中率：1 命中 / 2 查询（按块算）
        # 1 次 query 命中 1 块，1 次 query 命中 0 块，共 2 次 query
        assert meta.total_queries == 2
        assert meta.total_hits == 1

    def test_concurrent_publish(self, meta):
        """并发发布不应有竞争条件"""
        errors = []

        def publish_batch(start):
            try:
                for i in range(start, start + 50):
                    meta.publish(i, node_id=i % 4, token_ids=tuple(range(BLOCK_SIZE)))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=publish_batch, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert meta.total_blocks == 200

    def test_node_stats(self, meta):
        meta.publish(1, node_id=0, token_ids=tuple(range(BLOCK_SIZE)))
        meta.publish(2, node_id=0, token_ids=tuple(range(BLOCK_SIZE)))
        meta.publish(3, node_id=1, token_ids=tuple(range(BLOCK_SIZE)))

        stats = meta.node_stats()
        assert stats[0]['num_blocks'] == 2
        assert stats[1]['num_blocks'] == 1


# ──────────────────────────────────────────────────────────────────
# TransferEngine 测试
# ──────────────────────────────────────────────────────────────────

class TestTransferEngine:
    @pytest.fixture
    def engine(self):
        return TransferEngine(node_id=10)

    def test_submit_returns_id(self, engine):
        tid = engine.submit_transfer(src_node_id=0, block_hashes=[1, 2, 3])
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_unique_transfer_ids(self, engine):
        ids = [engine.submit_transfer(0, [i]) for i in range(10)]
        assert len(set(ids)) == 10  # 全部唯一

    def test_transfer_completes(self, engine):
        tid = engine.submit_transfer(src_node_id=0, block_hashes=[1])
        result = engine.wait(tid, timeout=5.0)
        assert result is not None
        assert result.success

    def test_bytes_transferred(self, engine):
        num_blocks = 5
        tid = engine.submit_transfer(src_node_id=0, block_hashes=list(range(num_blocks)))
        result = engine.wait(tid, timeout=5.0)
        assert result.bytes_transferred == num_blocks * KV_BLOCK_BYTES

    def test_callback_called(self, engine):
        callback_results = []

        def cb(result):
            callback_results.append(result)

        tid = engine.submit_transfer(src_node_id=0, block_hashes=[1], callback=cb)
        engine.wait(tid, timeout=5.0)
        time.sleep(0.05)  # 等 callback 线程执行
        assert len(callback_results) == 1
        assert callback_results[0].transfer_id == tid

    def test_is_complete(self, engine):
        tid = engine.submit_transfer(src_node_id=0, block_hashes=[1, 2])
        # 刚提交，可能还未完成
        # 等待后应该完成
        engine.wait(tid, timeout=5.0)
        assert engine.is_complete(tid)

    def test_concurrent_transfers(self, engine):
        """并发提交多个传输请求"""
        tids = [
            engine.submit_transfer(src_node_id=i % 4, block_hashes=[i, i + 1])
            for i in range(10)
        ]
        results = [engine.wait(tid, timeout=10.0) for tid in tids]
        assert all(r is not None and r.success for r in results)


# ──────────────────────────────────────────────────────────────────
# MooncakeConnector 测试
# ──────────────────────────────────────────────────────────────────

class TestMooncakeConnector:
    @pytest.fixture
    def setup(self):
        meta = GlobalMetadataServer()
        engine_src = TransferEngine(node_id=0)
        engine_dst = TransferEngine(node_id=1)
        src_conn = MooncakeConnector(node_id=0, metadata_server=meta, transfer_engine=engine_src)
        dst_conn = MooncakeConnector(node_id=1, metadata_server=meta, transfer_engine=engine_dst)
        return meta, src_conn, dst_conn

    def test_miss_returns_zero(self, setup):
        _, src_conn, _ = setup
        tokens = list(range(BLOCK_SIZE * 2))
        hashes = compute_block_hashes(tokens)
        num_matched, needs_transfer = src_conn.get_num_new_matched_tokens("r0", hashes)
        assert num_matched == 0
        assert not needs_transfer

    def test_local_hit_no_transfer(self, setup):
        """同节点命中不需要传输"""
        meta, src_conn, _ = setup
        tokens = list(range(BLOCK_SIZE * 2))
        hashes = compute_block_hashes(tokens)

        # 在同节点发布
        for h in hashes:
            meta.publish(h, node_id=0, token_ids=tuple(range(BLOCK_SIZE)))

        num_matched, needs_transfer = src_conn.get_num_new_matched_tokens("r0", hashes)
        assert num_matched == len(hashes) * BLOCK_SIZE
        assert not needs_transfer  # 同节点，无需传输

    def test_remote_hit_triggers_transfer(self, setup):
        """远端命中应发起传输"""
        meta, src_conn, dst_conn = setup
        tokens = list(range(BLOCK_SIZE * 3))
        hashes = compute_block_hashes(tokens)

        # 节点 0 发布 KV
        for h in hashes:
            meta.publish(h, node_id=0, token_ids=tuple(range(BLOCK_SIZE)))

        # 节点 1 查询（应该触发 RDMA 传输）
        num_matched, needs_transfer = dst_conn.get_num_new_matched_tokens("r1", hashes)
        assert num_matched == len(hashes) * BLOCK_SIZE
        assert needs_transfer

    def test_wait_for_kv_completes(self, setup):
        meta, src_conn, dst_conn = setup
        tokens = list(range(BLOCK_SIZE * 2))
        hashes = compute_block_hashes(tokens)

        for h in hashes:
            meta.publish(h, node_id=0, token_ids=tuple(range(BLOCK_SIZE)))

        _, needs_transfer = dst_conn.get_num_new_matched_tokens("r2", hashes)

        if needs_transfer:
            ok = dst_conn.wait_for_kv("r2", timeout=10.0)
            assert ok

    def test_publish_kv(self, setup):
        meta, src_conn, _ = setup
        tokens = list(range(BLOCK_SIZE * 4))
        hashes = compute_block_hashes(tokens)
        token_ids_per_block = [
            tuple(tokens[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
            for i in range(len(hashes))
        ]
        count = src_conn.publish_kv(hashes, token_ids_per_block)
        assert count == len(hashes)
        assert meta.total_blocks == len(hashes)

    def test_stats_tracking(self, setup):
        meta, src_conn, dst_conn = setup
        tokens = list(range(BLOCK_SIZE))
        hashes = compute_block_hashes(tokens)

        # 先发布
        for h in hashes:
            meta.publish(h, node_id=0, token_ids=tuple(tokens))

        # 查询 3 次
        for i in range(3):
            dst_conn.get_num_new_matched_tokens(f"r{i}", hashes)

        assert dst_conn.stats['queries'] == 3
        assert dst_conn.stats['hits'] > 0


# ──────────────────────────────────────────────────────────────────
# SimulatedCluster 集成测试
# ──────────────────────────────────────────────────────────────────

class TestSimulatedCluster:
    @pytest.fixture
    def cluster(self):
        return SimulatedCluster(num_prefill_nodes=2, num_decode_nodes=2)

    def test_first_request_no_cache(self, cluster):
        """第一个请求没有缓存，cached=0"""
        tokens = list(range(BLOCK_SIZE * 5))
        cached, _ = cluster.simulate_prefill("req-0", tokens, prefill_node_idx=0)
        assert cached == 0

    def test_second_request_cache_hit(self, cluster):
        """相同前缀的第二个请求应该命中缓存"""
        sys_prompt = list(range(BLOCK_SIZE * 4))
        user_tokens = [9999, 10000, 10001]

        tokens_1 = sys_prompt + user_tokens[:1]
        tokens_2 = sys_prompt + user_tokens[1:3]

        # 第一个请求：无缓存
        cluster.simulate_prefill("req-0", tokens_1, prefill_node_idx=0)

        # 第二个请求：系统提示词部分应该命中
        cached, _ = cluster.simulate_prefill("req-1", tokens_2, prefill_node_idx=0)
        assert cached >= len(sys_prompt)  # 至少命中系统提示词部分

    def test_cross_node_cache_sharing(self, cluster):
        """Prefill 节点 0 的 KV 可以被 Decode 节点 0 使用"""
        tokens = list(range(BLOCK_SIZE * 6))

        # Prefill 节点 0 计算并发布
        cluster.simulate_prefill("req-prefill", tokens, prefill_node_idx=0)

        # Decode 节点 0 查询（应命中）
        from_cache, wait_ms = cluster.simulate_decode("req-decode", tokens, decode_node_idx=0)
        assert from_cache == len(tokens)  # 完全命中

    def test_global_pool_grows(self, cluster):
        """多次 prefill 后全局池应该增长"""
        initial = cluster.meta_server.total_blocks

        tokens_1 = list(range(BLOCK_SIZE * 3))
        tokens_2 = list(range(100, 100 + BLOCK_SIZE * 3))

        cluster.simulate_prefill("req-a", tokens_1, prefill_node_idx=0)
        cluster.simulate_prefill("req-b", tokens_2, prefill_node_idx=1)

        assert cluster.meta_server.total_blocks > initial

    def test_hit_rate_improves(self, cluster):
        """随着请求增加，相同前缀的命中率应该提升"""
        sys_prompt = list(range(BLOCK_SIZE * 4))

        # 第一批：无缓存
        for i in range(5):
            tokens = sys_prompt + [i * 100]
            cluster.simulate_prefill(f"warm-{i}", tokens, prefill_node_idx=0)

        # 此时命中率较低（初次）
        hr_after_warmup = cluster.meta_server.hit_rate

        # 第二批：应命中系统提示词
        for i in range(5, 15):
            tokens = sys_prompt + [i * 100]
            cluster.simulate_prefill(f"hot-{i}", tokens, prefill_node_idx=0)

        hr_after_hot = cluster.meta_server.hit_rate
        # 命中率应该提升
        assert hr_after_hot >= hr_after_warmup
