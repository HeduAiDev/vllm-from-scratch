"""
KV Cache BlockPool with LRU Eviction
对应 vLLM 的 vllm/v1/core/block_pool.py

实现了:
- 双向链表 FreeBlockQueue（LRU顺序）
- 前缀缓存 hash 表
- LRU 淘汰策略
"""
from typing import Optional, Dict, List


class Block:
    """KV Cache 物理块"""
    __slots__ = ['block_id', 'ref_cnt', 'block_hash', 'prev', 'next']

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: Optional[str] = None
        self.prev: Optional['Block'] = None
        self.next: Optional['Block'] = None

    def reset_hash(self):
        self.block_hash = None

    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_cnt}, hash={self.block_hash})"


class FreeBlockQueue:
    """
    空闲块双向链表（LRU顺序）

    最久未用 → ... → 最近释放
    head                    tail

    popleft(): 取最久未用的块（LRU驱逐目标）
    append():  释放时加到尾部（最新）
    remove():  被touch时从中间移除
    """
    def __init__(self, blocks: List[Block]):
        self._head = Block(-1)  # 哨兵
        self._tail = Block(-1)  # 哨兵
        self._head.next = self._tail
        self._tail.prev = self._head
        self.num_free_blocks = 0

        for b in blocks:
            self._link_before_tail(b)

    def _link_before_tail(self, node: Block):
        prev = self._tail.prev
        prev.next = node
        node.prev = prev
        node.next = self._tail
        self._tail.prev = node
        self.num_free_blocks += 1

    def _unlink(self, node: Block):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self.num_free_blocks -= 1

    def popleft(self) -> Block:
        """取出最久未用的块"""
        if self._head.next is self._tail:
            raise RuntimeError("No free blocks available (OOM)")
        block = self._head.next
        self._unlink(block)
        return block

    def popleft_n(self, n: int) -> List[Block]:
        return [self.popleft() for _ in range(n)]

    def append(self, node: Block):
        """释放，加到尾部"""
        self._link_before_tail(node)

    def append_n(self, nodes: List[Block]):
        for n in nodes:
            self.append(n)

    def remove(self, node: Block):
        """从中间移除（touch时）"""
        self._unlink(node)


class BlockPool:
    """
    KV Cache 块池（对应 vLLM BlockPool）

    功能:
    1. 分配块 (get_new_blocks)
    2. 释放块 (free_blocks)
    3. 前缀缓存 (cache_full_blocks / get_cached_block)
    4. Touch（增加引用计数）
    5. LRU 淘汰
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool = True):
        assert num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # 所有块
        self.blocks: List[Block] = [Block(i) for i in range(num_gpu_blocks)]

        # 空闲队列（包含可缓存的块，按LRU顺序）
        self.free_block_queue = FreeBlockQueue(self.blocks)

        # 前缀缓存：hash → Block
        # 当多个块有相同hash时存 dict[block_id, Block]
        self.cached_block_hash_to_block: Dict[str, Block | Dict[int, Block]] = {}

        # null block（block_id=0，用于padding）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.ref_cnt = 1  # 永不释放

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """返回0.0~1.0的KV Cache使用率"""
        total = self.num_gpu_blocks - 1  # 减去null block
        if not total:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total)

    def get_cached_block(self, block_hash: str) -> Optional[Block]:
        """查找前缀缓存，返回任意匹配的块"""
        entry = self.cached_block_hash_to_block.get(block_hash)
        if entry is None:
            return None
        if isinstance(entry, Block):
            return entry
        return next(iter(entry.values()))

    def get_new_blocks(self, num_blocks: int) -> List[Block]:
        """
        从空闲队列分配块（可能触发LRU淘汰）
        """
        if num_blocks > self.get_num_free_blocks():
            raise RuntimeError(
                f"OOM: need {num_blocks} blocks, "
                f"only {self.get_num_free_blocks()} free"
            )
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt = 1
        return blocks

    def _maybe_evict_cached_block(self, block: Block) -> bool:
        """如果块有缓存hash，从hash表中移除（LRU淘汰）"""
        if block.block_hash is None:
            return False

        h = block.block_hash
        entry = self.cached_block_hash_to_block.get(h)
        if entry is None:
            block.reset_hash()
            return False

        if isinstance(entry, Block):
            if entry.block_id == block.block_id:
                del self.cached_block_hash_to_block[h]
        elif isinstance(entry, dict):
            entry.pop(block.block_id, None)
            if len(entry) == 0:
                del self.cached_block_hash_to_block[h]
            else:
                self.cached_block_hash_to_block[h] = entry

        block.reset_hash()
        return True

    def cache_full_blocks(self, blocks: List[Block],
                          block_hashes: List[str],
                          num_already_cached: int,
                          num_full: int) -> None:
        """
        缓存已满的块（用于前缀缓存）

        blocks: 请求的所有块
        block_hashes: 对应的哈希值
        num_already_cached: 已经缓存的块数
        num_full: 当前满块数
        """
        if not self.enable_caching:
            return
        if num_already_cached >= num_full:
            return

        for i in range(num_already_cached, num_full):
            block = blocks[i]
            block_hash = block_hashes[i]

            if block.block_hash is not None:
                continue  # 已缓存

            block.block_hash = block_hash
            entry = self.cached_block_hash_to_block.get(block_hash)
            if entry is None:
                self.cached_block_hash_to_block[block_hash] = block
            elif isinstance(entry, Block):
                self.cached_block_hash_to_block[block_hash] = {
                    entry.block_id: entry,
                    block.block_id: block,
                }
            elif isinstance(entry, dict):
                entry[block.block_id] = block

    def touch(self, blocks: List[Block]) -> None:
        """
        增加引用计数（前缀命中时复用块）
        如果块在空闲队列中（ref_cnt==0），先从队列移除
        """
        for block in blocks:
            if block.ref_cnt == 0 and block is not self.null_block:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: List[Block]) -> None:
        """
        释放块：ref_cnt--，归0则入LRU队列尾部
        """
        for block in blocks:
            block.ref_cnt -= 1

        to_free = [b for b in blocks
                   if b.ref_cnt == 0 and b is not self.null_block]
        self.free_block_queue.append_n(to_free)

    def reset_prefix_cache(self) -> bool:
        """重置所有前缀缓存（用于RLHF后权重更新）"""
        if self.get_num_free_blocks() < self.num_gpu_blocks - 1:
            return False  # 还有请求在运行
        self.cached_block_hash_to_block.clear()
        for block in self.blocks:
            block.reset_hash()
        return True
