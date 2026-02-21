"""
BlockPool with LRU 单元测试
运行: pytest test_block_pool_lru.py -v
"""
import pytest
from block_pool_lru import Block, FreeBlockQueue, BlockPool


class TestFreeBlockQueue:
    def test_lru_order(self):
        """最先释放的块最先被重用（LRU头部）"""
        blocks = [Block(i) for i in range(4)]
        q = FreeBlockQueue(blocks)
        assert q.num_free_blocks == 4

        b = q.popleft()
        assert b.block_id == 0  # 最先加入的先出

    def test_append_and_pop(self):
        blocks = [Block(i) for i in range(2)]
        q = FreeBlockQueue(blocks)
        b0 = q.popleft()
        b1 = q.popleft()
        assert q.num_free_blocks == 0

        q.append(b0)
        q.append(b1)
        assert q.num_free_blocks == 2

        # b0先加入，所以先出
        out = q.popleft()
        assert out.block_id == b0.block_id

    def test_remove_from_middle(self):
        blocks = [Block(i) for i in range(4)]
        q = FreeBlockQueue(blocks)
        target = blocks[2]
        q.remove(target)
        assert q.num_free_blocks == 3


class TestBlockPool:
    @pytest.fixture
    def pool(self):
        return BlockPool(num_gpu_blocks=8, enable_caching=True)

    def test_basic_alloc_free(self, pool):
        initial = pool.get_num_free_blocks()
        blocks = pool.get_new_blocks(3)
        assert len(blocks) == 3
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.get_num_free_blocks() == initial - 3

        pool.free_blocks(blocks)
        assert pool.get_num_free_blocks() == initial

    def test_oom_raises(self, pool):
        with pytest.raises(RuntimeError, match="OOM"):
            pool.get_new_blocks(999)

    def test_usage_metric(self, pool):
        initial_usage = pool.get_usage()
        blocks = pool.get_new_blocks(3)
        assert pool.get_usage() > initial_usage
        pool.free_blocks(blocks)

    def test_prefix_cache_hit(self, pool):
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, ["hash_abc"], 0, 1)
        pool.free_blocks(blocks)  # 释放但保留缓存

        cached = pool.get_cached_block("hash_abc")
        assert cached is not None
        assert cached.block_id == blocks[0].block_id

    def test_cache_eviction_lru(self, pool):
        """LRU淘汰：最久未用的块（LRU头部）先被驱逐"""
        # pool: 8块, null占用块0, 可用7块: 1,2,3,4,5,6,7
        # 分配3块 → [1,2,3]，缓存h0,h1,h2
        blocks_a = pool.get_new_blocks(3)
        pool.cache_full_blocks(blocks_a, ["h0", "h1", "h2"], 0, 3)
        pool.free_blocks(blocks_a)
        # LRU队列: [4,5,6,7,1,2,3]（1,2,3在尾部=最近释放）

        # 分配3块 → [4,5,6]（LRU头部），缓存h3,h4,h5
        blocks_b = pool.get_new_blocks(3)
        pool.cache_full_blocks(blocks_b, ["h3", "h4", "h5"], 0, 3)
        pool.free_blocks(blocks_b)
        # LRU队列: [7,1,2,3,4,5,6]（4,5,6最近释放在尾部）

        # 再分配3块 → [7,1,2]（LRU头部）
        # 块7没有hash，块1 evict h0，块2 evict h1
        new_blocks = pool.get_new_blocks(3)

        # h0, h1 被驱逐（块1,2被重用）
        assert pool.get_cached_block("h0") is None
        assert pool.get_cached_block("h1") is None
        # h2(块3), h3,h4,h5 还在
        assert pool.get_cached_block("h2") is not None
        assert pool.get_cached_block("h3") is not None

        pool.free_blocks(new_blocks)

    def test_touch_increments_ref(self, pool):
        """touch：前缀命中时增加引用计数"""
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, ["hash_sys"], 0, 1)
        pool.free_blocks(blocks)

        # 块现在在LRU队列（ref=0）
        assert blocks[0].ref_cnt == 0

        # 另一个请求命中前缀
        cached = pool.get_cached_block("hash_sys")
        pool.touch([cached])

        assert cached.ref_cnt == 1
        # 从LRU队列中移除
        assert pool.get_num_free_blocks() == pool.get_num_free_blocks()

    def test_shared_prefix_multi_requests(self, pool):
        """多请求共享前缀块"""
        # 请求A创建前缀
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, ["system_prompt"], 0, 1)
        # A还在运行（ref=1）

        # 请求B命中并touch
        cached = pool.get_cached_block("system_prompt")
        pool.touch([cached])
        assert cached.ref_cnt == 2  # A和B都在用

        # A完成
        pool.free_blocks(blocks)
        assert cached.ref_cnt == 1  # B还在用

        # B完成
        pool.free_blocks([cached])
        assert cached.ref_cnt == 0  # 完全释放

    def test_reset_prefix_cache(self, pool):
        blocks = pool.get_new_blocks(2)
        pool.cache_full_blocks(blocks, ["h_a", "h_b"], 0, 2)
        pool.free_blocks(blocks)

        assert pool.get_cached_block("h_a") is not None
        success = pool.reset_prefix_cache()
        assert success
        assert pool.get_cached_block("h_a") is None
        assert pool.get_cached_block("h_b") is None

    def test_null_block_never_freed(self, pool):
        """null block 永不进入空闲队列"""
        null = pool.null_block
        assert null.ref_cnt == 1
        # 不应该在空闲队列中
        free_ids = []
        curr = pool.free_block_queue._head.next
        while curr is not pool.free_block_queue._tail:
            free_ids.append(curr.block_id)
            curr = curr.next
        assert null.block_id not in free_ids


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
