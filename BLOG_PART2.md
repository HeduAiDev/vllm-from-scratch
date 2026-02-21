# vLLM：从入门到专家（第二部分）

---

## 第六章：Prefix Cache——单机版从0实现

### 6.1 核心思想

Prefix Cache（前缀缓存）解决的是这样一个问题：

> 如果 1000 个请求都以相同的"系统提示词"开头，为什么要重复计算这 1000 次？

传统方式：每个请求独立计算所有 token 的 KV Cache。
Prefix Cache：第一个请求计算完后，相同前缀的 KV Cache 可以复用。

```
请求1: [SYS] + [用户A的问题]
         ↓
     计算所有KV, 缓存 [SYS] 对应的块

请求2: [SYS] + [用户B的问题]
              ↑ 直接复用！跳过重新计算
```

**节省**：跳过 N 个 cached token 的计算，省去 O(N) 的 prefill 时间。

### 6.2 块哈希：怎么判断"相同前缀"？

vLLM 的做法：对每个满块（16 tokens）计算哈希值。

```python
# vllm/v1/core/kv_cache_utils.py
def hash_block_tokens(
    parent_block_hash: int,   # 上一块的哈希（保证前缀链式依赖）
    curr_block_token_ids: tuple[int, ...],  # 本块的 token IDs
    extra_keys: tuple | None = None,  # 额外键（LoRA ID等）
) -> BlockHash:
    return hash((parent_block_hash, curr_block_token_ids, extra_keys))
```

**链式哈希设计**：每个块的哈希值依赖于前一个块，这保证了：
- `[SYS_BLOCK_1, SYS_BLOCK_2]` 的哈希是唯一的
- 即使 token 内容相同，不同前缀链的块哈希不同

```
块0: hash(seed=0, tokens=[t0..t15])        = H0
块1: hash(parent=H0, tokens=[t16..t31])    = H1
块2: hash(parent=H1, tokens=[t32..t47])    = H2

如果另一请求前32个token和上面相同:
    块0命中H0 ✓
    块1命中H1 ✓ (因为parent H0也相同)
    块2不同token, 哈希不同 ✗
```

### 6.3 从0实现单机 Prefix Cache

**文件：`03_prefix_cache/prefix_cache_local.py`**

```python
# vllm-from-scratch/03_prefix_cache/prefix_cache_local.py
"""
单机 Prefix Cache 从0实现
对应 vLLM 的 BlockPool + KVCacheManager 前缀命中逻辑
"""
from typing import List, Optional, Tuple
from collections import OrderedDict
import hashlib


BLOCK_SIZE = 16  # 每块 token 数


def compute_block_hash(parent_hash: int, token_ids: Tuple[int, ...],
                       extra_keys: Optional[Tuple] = None) -> int:
    """
    链式块哈希：hash(parent_hash, token_ids, extra_keys)
    对应 vLLM 的 hash_block_tokens()
    """
    return hash((parent_hash, token_ids, extra_keys))


class CachedBlock:
    """前缀缓存中的一个块"""
    def __init__(self, block_id: int, block_hash: int, token_ids: Tuple[int, ...]):
        self.block_id = block_id
        self.block_hash = block_hash
        self.token_ids = token_ids
        self.ref_cnt = 0  # 当前使用此块的请求数


class PrefixCacheLocal:
    """
    单机版前缀缓存（LRU 淘汰）

    核心操作：
    1. compute_hash_chain: 计算一段 tokens 的哈希链
    2. match_prefix: 找出命中的前缀块
    3. cache_blocks: 请求完成后缓存已计算的块
    4. evict_lru: LRU 淘汰最久未用的块

    对应 vLLM 的 KVCacheManager.get_computed_blocks() 逻辑
    """

    def __init__(self, max_cached_blocks: int):
        self.max_cached_blocks = max_cached_blocks
        self.next_block_id = 0

        # LRU cache: OrderedDict 维护 LRU 顺序
        # key: block_hash, value: CachedBlock
        self._cache: OrderedDict[int, CachedBlock] = OrderedDict()

        # 所有正在使用的块（ref_cnt > 0）
        self._active_blocks: dict[int, CachedBlock] = {}

    def _allocate_block_id(self) -> int:
        bid = self.next_block_id
        self.next_block_id += 1
        return bid

    def compute_hash_chain(self, token_ids: List[int],
                           extra_keys: Optional[Tuple] = None
                           ) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        计算 token_ids 对应的块哈希链

        返回: [(block_hash, block_tokens), ...]
        只返回完整块（丢弃不满块）
        """
        num_full_blocks = len(token_ids) // BLOCK_SIZE
        result = []
        parent_hash = 0  # 初始哈希种子

        for i in range(num_full_blocks):
            start = i * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block_tokens = tuple(token_ids[start:end])
            block_hash = compute_block_hash(parent_hash, block_tokens, extra_keys)
            result.append((block_hash, block_tokens))
            parent_hash = block_hash

        return result

    def match_prefix(self, token_ids: List[int],
                     extra_keys: Optional[Tuple] = None
                     ) -> Tuple[List[CachedBlock], int]:
        """
        查找命中的前缀缓存

        返回:
            (matched_blocks, num_cached_tokens)
            matched_blocks: 命中的块列表
            num_cached_tokens: 命中的 token 数
        """
        hash_chain = self.compute_hash_chain(token_ids, extra_keys)
        matched_blocks = []

        for block_hash, block_tokens in hash_chain:
            cached = self._cache.get(block_hash)
            if cached is None:
                break  # 链断开，停止查找
            # LRU: 访问了，移到末尾
            self._cache.move_to_end(block_hash)
            matched_blocks.append(cached)

        num_cached_tokens = len(matched_blocks) * BLOCK_SIZE
        return matched_blocks, num_cached_tokens

    def touch(self, blocks: List[CachedBlock]) -> None:
        """
        增加引用计数（请求开始使用这些前缀块）
        """
        for block in blocks:
            block.ref_cnt += 1
            self._active_blocks[block.block_id] = block

    def cache_computed_blocks(self, token_ids: List[int],
                              extra_keys: Optional[Tuple] = None
                              ) -> List[CachedBlock]:
        """
        将新计算的块加入缓存

        调用时机：请求完成 prefill 后（新块已满）
        返回：新创建/更新的块列表
        """
        hash_chain = self.compute_hash_chain(token_ids, extra_keys)
        new_blocks = []

        for block_hash, block_tokens in hash_chain:
            if block_hash in self._cache:
                # 已经缓存，复用
                existing = self._cache[block_hash]
                self._cache.move_to_end(block_hash)
                new_blocks.append(existing)
                continue

            # 淘汰 LRU（如果缓存满了）
            while len(self._cache) >= self.max_cached_blocks:
                if not self._evict_lru():
                    break  # 无法淘汰（所有块都在用）

            # 创建新缓存块
            block = CachedBlock(
                block_id=self._allocate_block_id(),
                block_hash=block_hash,
                token_ids=block_tokens,
            )
            self._cache[block_hash] = block
            new_blocks.append(block)

        return new_blocks

    def _evict_lru(self) -> bool:
        """
        淘汰最久未用的块（LRU 头部）
        只淘汰 ref_cnt == 0 的块（未被使用的）
        """
        for block_hash, block in self._cache.items():
            if block.ref_cnt == 0:
                del self._cache[block_hash]
                return True
        return False  # 所有块都在用，无法淘汰

    def release(self, blocks: List[CachedBlock]) -> None:
        """
        请求完成，释放对块的引用（ref_cnt--）
        """
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._active_blocks.pop(block.block_id, None)

    @property
    def num_cached_blocks(self) -> int:
        return len(self._cache)

    @property
    def cache_hit_rate(self) -> float:
        """简化的缓存命中率统计"""
        return self._hit_count / max(1, self._total_count)
```

### 6.4 Prefix Cache 测试

**文件：`03_prefix_cache/test_prefix_cache_local.py`**

```python
# vllm-from-scratch/03_prefix_cache/test_prefix_cache_local.py
import pytest
from prefix_cache_local import PrefixCacheLocal, BLOCK_SIZE, compute_block_hash


SYS_PROMPT = list(range(32))  # 2块的系统提示词


class TestHashChain:
    def test_full_blocks_only(self):
        """只有满块才被缓存"""
        cache = PrefixCacheLocal(max_cached_blocks=10)
        # 17 tokens: 1完整块 + 1剩余
        tokens = list(range(17))
        chain = cache.compute_hash_chain(tokens)
        assert len(chain) == 1  # 只有1个完整块

    def test_chain_dependency(self):
        """链式哈希：相同内容但不同前缀，哈希不同"""
        cache = PrefixCacheLocal(max_cached_blocks=10)
        tokens1 = list(range(32))    # [0..31]
        tokens2 = list(range(16, 48))  # [16..47]

        chain1 = cache.compute_hash_chain(tokens1)
        chain2 = cache.compute_hash_chain(tokens2)

        # block1 内容相同: tokens[0:16] vs tokens[16:32]，不同
        # 但即使内容相同，parent_hash 也不同
        assert chain1[0][0] != chain2[0][0]  # 哈希不同（内容不同）

    def test_deterministic(self):
        """相同输入必须产生相同哈希"""
        cache = PrefixCacheLocal(max_cached_blocks=10)
        tokens = list(range(16))
        h1 = cache.compute_hash_chain(tokens)
        h2 = cache.compute_hash_chain(tokens)
        assert h1[0][0] == h2[0][0]


class TestPrefixCache:
    @pytest.fixture
    def cache(self):
        return PrefixCacheLocal(max_cached_blocks=20)

    def test_cold_miss(self, cache):
        """冷启动：没有缓存，miss"""
        tokens = SYS_PROMPT + list(range(100, 116))
        blocks, num_cached = cache.match_prefix(tokens)
        assert len(blocks) == 0
        assert num_cached == 0

    def test_cache_then_hit(self, cache):
        """缓存后能命中"""
        tokens = SYS_PROMPT + list(range(100, 116))

        # 第一次：miss，但缓存
        blocks, num_cached = cache.match_prefix(tokens)
        assert num_cached == 0
        cache.cache_computed_blocks(tokens)

        # 第二次：命中
        blocks2, num_cached2 = cache.match_prefix(tokens)
        assert num_cached2 == len(tokens) // BLOCK_SIZE * BLOCK_SIZE
        assert len(blocks2) > 0

    def test_partial_prefix_hit(self, cache):
        """部分前缀命中"""
        tokens_a = SYS_PROMPT + list(range(100, 148))  # 4块
        cache.cache_computed_blocks(tokens_a)

        # 新请求只共享前缀（SYS_PROMPT 部分）
        tokens_b = SYS_PROMPT + list(range(200, 248))  # 不同后缀
        blocks, num_cached = cache.match_prefix(tokens_b)

        # 只有 SYS_PROMPT 的2块命中
        assert num_cached == 32  # BLOCK_SIZE * 2

    def test_lru_eviction(self):
        """LRU淘汰：最久未用的块先被驱逐"""
        cache = PrefixCacheLocal(max_cached_blocks=3)

        # 缓存3块（填满）
        cache.cache_computed_blocks(list(range(16)))   # 块A
        cache.cache_computed_blocks(list(range(16, 32)))  # 块B（不同前缀链，独立）

        # 让cache再学习一个不同的3-token链
        tokens_c = list(range(200, 248))  # 3块新内容
        cache.cache_computed_blocks(tokens_c)
        # 现在缓存满了（3块）

        # 再缓存一个新块，触发LRU淘汰
        tokens_new = list(range(300, 316))
        cache.cache_computed_blocks(tokens_new)

        # 缓存大小不超过max
        assert cache.num_cached_blocks <= 3

    def test_shared_prefix_multi_requests(self, cache):
        """多请求共享前缀，ref_cnt 正确增减"""
        # 缓存系统提示词
        cache.cache_computed_blocks(SYS_PROMPT)
        blocks, _ = cache.match_prefix(SYS_PROMPT)

        # 请求A和B都使用这些块
        cache.touch(blocks)  # 请求A
        cache.touch(blocks)  # 请求B
        assert all(b.ref_cnt == 2 for b in blocks)

        # A完成
        cache.release(blocks)
        assert all(b.ref_cnt == 1 for b in blocks)

        # B完成
        cache.release(blocks)
        assert all(b.ref_cnt == 0 for b in blocks)

    def test_ref_blocks_not_evicted(self):
        """正在使用的块不应该被LRU淘汰"""
        cache = PrefixCacheLocal(max_cached_blocks=2)
        tokens = list(range(32))  # 2块
        cache.cache_computed_blocks(tokens)
        blocks, _ = cache.match_prefix(tokens)
        cache.touch(blocks)  # 标记为使用中

        # 尝试触发淘汰
        for i in range(10):
            new_tokens = list(range(1000 + i * 16, 1016 + i * 16))
            cache.cache_computed_blocks(new_tokens)  # 触发淘汰

        # 使用中的块应该还在
        blocks2, num_cached = cache.match_prefix(tokens)
        assert num_cached == 32  # 仍然命中

        cache.release(blocks)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 第七章：Prefix Cache 全局池化版本（Mooncake 风格）

### 7.1 单机 vs 全局池化的区别

| 特性 | 单机版 | 全局池化（Mooncake） |
|------|--------|---------------------|
| KV Cache 位置 | 每个 Worker 本地 GPU | 分布式 DRAM/SSD |
| 缓存容量 | 受单卡显存限制 | 可达 TB 级别 |
| 命中后操作 | 直接读取本地 | 需要网络传输 |
| 适用场景 | 单机推理 | 多节点集群 |
| vLLM 实现 | BlockPool | MooncakeConnector |

Mooncake 是月之暗面（Kimi）开源的 KV Cache 传输引擎，vLLM 已集成：

```
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/
├── mooncake_connector.py   # vLLM 侧的 KVConnector 实现
└── mooncake_utils.py       # 工具函数（bootstrap server等）
```

### 7.2 Mooncake 架构

```
┌─────────────────────────────────────────────────┐
│              Global KV Pool (Mooncake)           │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Node 0   │  │ Node 1   │  │ Node 2   │      │
│  │ GPU VRAM │  │ CPU DRAM │  │ NVMe SSD │      │
│  │   80GB   │  │  512GB   │  │    2TB   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └─────────────┴─────────────┘             │
│              RDMA/RoCE 高速网络                  │
└─────────────────────────────────────────────────┘
           ↕  TransferEngine API
┌─────────────────────────────────────────────────┐
│          vLLM Worker（Prefill/Decode 实例）      │
│  MooncakeConnector                              │
│   ├── 查找全局缓存                              │
│   ├── 发起 RDMA 传输                           │
│   └── 等待接收完成                              │
└─────────────────────────────────────────────────┘
```

### 7.3 MooncakeConnector 关键实现

```python
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py

class MooncakeConnector(KVConnectorBase_V1):
    """
    vLLM V1 的 Mooncake KV 传输连接器

    在 PD 分离场景下:
    - Prefill 节点：计算 KV → 发送给 Decode 节点
    - Decode 节点：接收 KV → 直接解码，跳过 prefill
    """

    def __init__(self, rank, local_rank, config):
        # 初始化 Mooncake TransferEngine（基于 RDMA）
        from mooncake.engine import TransferEngine
        self.engine = TransferEngine()
        self.engine.initialize(local_hostname, metadata_server_addr)

        # 注册本机的 KV Cache 内存区域（RDMA 需要预注册）
        self.kv_cache_base_addr = self._register_kv_cache_memory()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        查询全局缓存命中情况

        返回:
            (num_matched_tokens, load_kv_async)
            - num_matched_tokens: 命中的 token 数
            - load_kv_async: 是否异步加载（True=后台传输）
        """
        # 向 Mooncake metadata server 查询前缀哈希
        matched_tokens = self._query_global_cache(request.block_hashes)

        if matched_tokens > 0:
            # 异步发起 RDMA 传输（不阻塞调度器）
            self._start_async_transfer(request, matched_tokens)
            return matched_tokens, True  # load_kv_async=True

        return 0, False

    def _start_async_transfer(self, request, num_tokens):
        """
        发起异步 KV 传输
        对应 vLLM 的 WAITING_FOR_REMOTE_KVS 状态
        """
        # 1. 确定源节点（谁有这些KV？）
        source_node = self._locate_kv_source(request.block_hashes)

        # 2. 发起 RDMA 读（从远端内存读到本地GPU）
        transfer_id = self.engine.transfer_sync(
            remote_hostname=source_node.hostname,
            remote_port=source_node.port,
            remote_kv_addrs=source_node.kv_addrs,
            local_kv_addrs=self.local_kv_addrs,
        )
        self._pending_transfers[request.request_id] = transfer_id

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """
        KV 块分配完成后，更新 Mooncake 状态
        告诉 Mooncake "这些块已经准备好接收KV了"
        """
        block_ids = [b.block_id for b in blocks]
        self.engine.notify_ready_to_receive(
            request_id=request.request_id,
            block_ids=block_ids,
        )

    def request_finished(self, request, blocks):
        """
        请求完成，决定是否将 KV 保存到全局池
        """
        if request.num_computed_tokens >= MIN_TOKENS_TO_CACHE:
            # 发布到全局缓存（其他节点可以查到）
            self.engine.publish_kv(
                block_hashes=request.block_hashes,
                block_addrs=self._get_kv_addrs(blocks),
            )
```

### 7.4 从0实现简化版全局 KV Cache 池

**文件：`03_prefix_cache/global_kv_pool.py`**

```python
# vllm-from-scratch/03_prefix_cache/global_kv_pool.py
"""
简化版全局 KV Cache 池（模拟 Mooncake 功能）
不使用真实的 RDMA，用 Python 队列模拟网络传输
"""
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVBlock:
    """全局池中的一个KV块"""
    block_hash: int
    node_id: int          # 存储在哪个节点
    token_ids: Tuple[int, ...]
    kv_data: bytes        # 序列化的KV数据（真实场景是GPU内存地址）
    access_time: float = 0.0


class GlobalKVPool:
    """
    全局 KV Cache 池（单进程内的模拟实现）

    真实的 Mooncake 通过：
    - RDMA 传输 KV 数据
    - etcd/Redis 存储元数据
    - Bootstrap server 注册节点

    这里用 Python dict + threading.Queue 模拟
    """

    def __init__(self, max_blocks_per_node: int = 100):
        self.max_blocks_per_node = max_blocks_per_node
        self._lock = threading.Lock()

        # 元数据：block_hash → KVBlock
        self._metadata: Dict[int, KVBlock] = {}

        # 节点容量追踪
        self._node_block_count: Dict[int, int] = {}

        # 异步传输队列
        self._transfer_queue: queue.Queue = queue.Queue()
        self._transfer_results: Dict[str, bool] = {}

        # 启动后台传输线程（模拟 RDMA）
        self._transfer_thread = threading.Thread(
            target=self._transfer_worker, daemon=True
        )
        self._transfer_thread.start()

    def query(self, block_hashes: List[int]) -> Tuple[int, List[int]]:
        """
        查询全局缓存命中情况

        返回:
            (num_matched_tokens, matched_block_ids)
        """
        with self._lock:
            matched = 0
            matched_node_ids = []
            for h in block_hashes:
                if h in self._metadata:
                    matched += 16  # BLOCK_SIZE
                    matched_node_ids.append(self._metadata[h].node_id)
                    self._metadata[h].access_time = time.time()  # LRU更新
                else:
                    break  # 链断开
            return matched, matched_node_ids

    def async_transfer(self, request_id: str, block_hashes: List[int],
                       target_node_id: int, callback=None):
        """
        发起异步KV传输（模拟RDMA）

        真实场景：Mooncake TransferEngine 通过 RDMA 直接写入目标 GPU 内存
        """
        self._transfer_queue.put({
            'request_id': request_id,
            'block_hashes': block_hashes,
            'target_node_id': target_node_id,
            'callback': callback,
        })

    def _transfer_worker(self):
        """后台传输线程（模拟网络传输延迟）"""
        while True:
            try:
                task = self._transfer_queue.get(timeout=1)
                time.sleep(0.001)  # 模拟 1ms 网络延迟（真实RDMA约2-5μs）

                # 模拟传输成功
                self._transfer_results[task['request_id']] = True
                if task['callback']:
                    task['callback'](task['request_id'], success=True)
            except queue.Empty:
                continue

    def is_transfer_complete(self, request_id: str) -> bool:
        """检查传输是否完成"""
        return self._transfer_results.get(request_id, False)

    def publish(self, block_hash: int, node_id: int,
                token_ids: Tuple[int, ...], kv_data: bytes) -> bool:
        """
        将 KV 块发布到全局池

        调用时机：请求完成 prefill 后
        """
        with self._lock:
            if block_hash in self._metadata:
                return True  # 已存在

            node_count = self._node_block_count.get(node_id, 0)
            if node_count >= self.max_blocks_per_node:
                # 触发 LRU 淘汰
                self._evict_from_node(node_id)

            block = KVBlock(
                block_hash=block_hash,
                node_id=node_id,
                token_ids=token_ids,
                kv_data=kv_data,
                access_time=time.time(),
            )
            self._metadata[block_hash] = block
            self._node_block_count[node_id] = node_count + 1
            return True

    def _evict_from_node(self, node_id: int):
        """LRU 淘汰：移除该节点最久未访问的块"""
        candidates = [
            (h, b) for h, b in self._metadata.items()
            if b.node_id == node_id
        ]
        if candidates:
            oldest_hash = min(candidates, key=lambda x: x[1].access_time)[0]
            del self._metadata[oldest_hash]
            self._node_block_count[node_id] -= 1

    @property
    def total_cached_blocks(self) -> int:
        with self._lock:
            return len(self._metadata)
```

---

## 第八章：Scheduler 调度器——从0实现

### 8.1 调度器的核心职责

```
输入：waiting队列 + running列表 + KV Cache状态
输出：这一步执行哪些请求，各执行多少token
约束：
  1. 总token数 ≤ max_num_batched_tokens (e.g. 32768)
  2. 并发请求数 ≤ max_num_seqs (e.g. 256)
  3. KV Cache 块数够用
```

### 8.2 vLLM Scheduler 的调度优先级

```python
# Phase 1: 先调度 RUNNING 中的请求（它们已经有KV Cache了）
for req in self.running:
    tokens = min(req.remaining_tokens, token_budget)
    if allocate_kv(req, tokens):
        schedule(req, tokens)
        token_budget -= tokens
    else:
        preempt(req)  # KV Cache不够 → 抢占

# Phase 2: 再从 WAITING 调度新请求
while waiting and token_budget > 0 and len(running) < max_seqs:
    req = waiting.peek()
    check_prefix_cache(req)  # 前缀命中？
    if allocate_kv(req, min(req.num_prompt_tokens, token_budget)):
        schedule(req)
        token_budget -= ...
```

### 8.3 从0实现调度器

**文件：`04_scheduler/scheduler.py`**

```python
# vllm-from-scratch/04_scheduler/scheduler.py
"""
简化版 vLLM Scheduler 从0实现
"""
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math


BLOCK_SIZE = 16


class RequestStatus(IntEnum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED_STOPPED = 4
    FINISHED_LENGTH_CAPPED = 5
    FINISHED_ABORTED = 6


@dataclass
class Request:
    request_id: str
    token_ids: List[int]         # prompt token IDs
    max_new_tokens: int
    priority: int = 0
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING

    output_token_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0  # 已处理的token数
    block_table: List[int] = field(default_factory=list)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def seq_len(self) -> int:
        return len(self.token_ids) + len(self.output_token_ids)

    @property
    def remaining_tokens(self) -> int:
        """当前步还需要处理的token数"""
        return self.seq_len - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status >= RequestStatus.FINISHED_STOPPED

    @property
    def num_blocks_needed(self) -> int:
        return math.ceil(self.seq_len / BLOCK_SIZE)


@dataclass
class SchedulerOutput:
    """调度器输出（对应 vLLM 的 SchedulerOutput）"""
    scheduled_requests: List[Request]
    num_scheduled_tokens: Dict[str, int]  # {req_id: num_tokens}
    preempted_requests: List[Request]
    total_tokens: int = 0

    def __post_init__(self):
        self.total_tokens = sum(self.num_scheduled_tokens.values())


class SimpleBlockManager:
    """简化版块管理器"""
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.request_blocks: Dict[str, List[int]] = {}

    def allocate(self, req_id: str, num_new_blocks: int) -> bool:
        """为请求分配新块，返回是否成功"""
        if len(self.free_blocks) < num_new_blocks:
            return False
        new_blocks = [self.free_blocks.pop(0) for _ in range(num_new_blocks)]
        if req_id not in self.request_blocks:
            self.request_blocks[req_id] = []
        self.request_blocks[req_id].extend(new_blocks)
        return True

    def free(self, req_id: str):
        """释放请求的所有块"""
        blocks = self.request_blocks.pop(req_id, [])
        self.free_blocks.extend(blocks)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def blocks_needed_for_new_tokens(self, req: Request, num_new_tokens: int) -> int:
        """计算处理num_new_tokens还需要分配多少新块"""
        current_blocks = len(self.request_blocks.get(req.request_id, []))
        new_total_tokens = req.num_computed_tokens + num_new_tokens
        needed_blocks = math.ceil(new_total_tokens / BLOCK_SIZE)
        return max(0, needed_blocks - current_blocks)


class Scheduler:
    """
    简化版 vLLM Scheduler

    对应 vllm/v1/core/sched/scheduler.py 的核心逻辑
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        num_gpu_blocks: int = 1000,
        policy: str = "fcfs",  # "fcfs" 或 "priority"
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.policy = policy

        self.block_manager = SimpleBlockManager(num_gpu_blocks)

        # 请求队列
        self.waiting: List[Request] = []   # 等待队列
        self.running: List[Request] = []   # 运行列表
        self.requests: Dict[str, Request] = {}

        # 统计
        self.stats = {
            'total_scheduled': 0,
            'total_preempted': 0,
            'total_finished': 0,
        }

    def add_request(self, request: Request):
        """添加新请求"""
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def _sort_waiting(self):
        """排序等待队列"""
        if self.policy == "fcfs":
            self.waiting.sort(key=lambda r: r.arrival_time)
        elif self.policy == "priority":
            self.waiting.sort(key=lambda r: (-r.priority, r.arrival_time))

    def schedule(self) -> SchedulerOutput:
        """
        调度一步：决定执行哪些请求和多少token

        两阶段算法：
        Phase 1: RUNNING 请求继续执行
        Phase 2: WAITING 请求新启动
        """
        scheduled: List[Request] = []
        num_tokens: Dict[str, int] = {}
        preempted: List[Request] = []
        token_budget = self.max_num_batched_tokens

        # ──────────────────────────────────────
        # Phase 1: 调度 RUNNING 请求
        # ──────────────────────────────────────
        still_running = []
        for req in self.running:
            if token_budget <= 0:
                # 没有token预算了，剩余running请求暂不调度
                still_running.append(req)
                continue

            # 这一步要处理多少token（decode: 1, prefill续: 剩余）
            tokens_this_step = min(req.remaining_tokens, token_budget)

            # 检查KV Cache块是否够用
            new_blocks_needed = self.block_manager.blocks_needed_for_new_tokens(
                req, tokens_this_step
            )
            if new_blocks_needed > 0:
                success = self.block_manager.allocate(req.request_id, new_blocks_needed)
                if not success:
                    # KV Cache不足 → 抢占此请求（或优先级最低的）
                    victim = self._select_preemption_victim(req, still_running)
                    self._preempt(victim)
                    preempted.append(victim)
                    if victim is not req:
                        # 重试分配
                        success = self.block_manager.allocate(
                            req.request_id, new_blocks_needed
                        )
                    if not success:
                        still_running.append(req)
                        continue

            scheduled.append(req)
            num_tokens[req.request_id] = tokens_this_step
            token_budget -= tokens_this_step
            still_running.append(req)

        self.running = still_running

        # ──────────────────────────────────────
        # Phase 2: 调度 WAITING 请求
        # ──────────────────────────────────────
        self._sort_waiting()
        new_waiting = []

        for req in self.waiting:
            if token_budget <= 0:
                new_waiting.append(req)
                continue
            if len(self.running) >= self.max_num_seqs:
                new_waiting.append(req)
                continue

            # 新请求：需要分配初始块
            tokens_this_step = min(req.num_prompt_tokens, token_budget)
            blocks_needed = math.ceil(tokens_this_step / BLOCK_SIZE)

            success = self.block_manager.allocate(req.request_id, blocks_needed)
            if not success:
                new_waiting.append(req)
                continue

            # 成功启动
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            scheduled.append(req)
            num_tokens[req.request_id] = tokens_this_step
            token_budget -= tokens_this_step

        self.waiting = new_waiting

        self.stats['total_scheduled'] += len(scheduled)
        return SchedulerOutput(
            scheduled_requests=scheduled,
            num_scheduled_tokens=num_tokens,
            preempted_requests=preempted,
        )

    def update_from_output(self, output: SchedulerOutput,
                           generated_tokens: Dict[str, List[int]]):
        """
        用模型输出更新调度器状态

        generated_tokens: {req_id: [new_token_id, ...]}
        """
        finished_reqs = []

        for req in output.scheduled_requests:
            new_tokens = generated_tokens.get(req.request_id, [])
            req.output_token_ids.extend(new_tokens)
            req.num_computed_tokens += output.num_scheduled_tokens[req.request_id]

            # 检查停止条件
            if new_tokens and new_tokens[-1] == 2:  # EOS token_id=2
                req.status = RequestStatus.FINISHED_STOPPED
                finished_reqs.append(req)
            elif len(req.output_token_ids) >= req.max_new_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                finished_reqs.append(req)

        # 清理完成的请求
        for req in finished_reqs:
            self.running.remove(req)
            self.block_manager.free(req.request_id)
            self.stats['total_finished'] += 1

    def abort_request(self, request_id: str):
        """中止请求"""
        req = self.requests.get(request_id)
        if req is None:
            return
        req.status = RequestStatus.FINISHED_ABORTED
        if req in self.running:
            self.running.remove(req)
            self.block_manager.free(request_id)
        elif req in self.waiting:
            self.waiting.remove(req)

    def _preempt(self, req: Request):
        """抢占请求：释放KV Cache，回到waiting队列"""
        if req in self.running:
            self.running.remove(req)
        self.block_manager.free(req.request_id)
        req.status = RequestStatus.PREEMPTED
        # 重置到前缀缓存命中位置（简化：重置到0）
        req.num_computed_tokens = 0
        self.waiting.append(req)
        self.stats['total_preempted'] += 1

    def _select_preemption_victim(self, current_req: Request,
                                  candidates: List[Request]) -> Request:
        """
        选择被抢占的受害者

        策略：
        - FCFS: 抢占最后到达的（当前请求自己）
        - PRIORITY: 抢占优先级最低的
        """
        if self.policy == "priority":
            all_candidates = [current_req] + candidates
            return min(all_candidates, key=lambda r: (-r.arrival_time, r.priority))
        return current_req  # 默认抢占自己（等待更多资源）

    def has_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def get_stats(self) -> dict:
        return {
            **self.stats,
            'num_waiting': len(self.waiting),
            'num_running': len(self.running),
            'kv_cache_usage': 1.0 - (
                self.block_manager.get_num_free_blocks() /
                self.block_manager.num_blocks
            ),
        }
```

---

## 第九章：投机解码（Speculative Decoding）——从0实现

### 9.1 核心思想

LLM 解码的瓶颈在于：每次只能生成 1 个 token，而每个 token 都要做完整的 forward pass。

投机解码的思路：

```
用小模型（Draft）快速猜 K 个 token
用大模型（Verifier）同时验证这 K 个 token
```

```
Draft 模型（1B参数，极快）：
  [t1, t2, t3] → 猜测 [t4, t5, t6, t7, t8]（5个draft token）

Verifier 模型（70B参数，慢但准）：
  [t1, t2, t3, t4, t5, t6, t7, t8] → 验证所有token
  → 输出每个位置的logits

拒绝采样：
  - t4 正确 ✓ 接受
  - t5 正确 ✓ 接受
  - t6 错误 ✗ 拒绝，从verifier的t6分布采样一个新token
  - 停止（忽略t7, t8）

结果：一次 forward pass 得到 3 个 token！
```

**加速比**：如果 draft 准确率 80%，期望每次得到 3-4 个 token，速度快 3-4 倍。

### 9.2 vLLM 中的实现架构

```
Scheduler.schedule():
    scheduled_spec_decode_tokens[req_id] = [draft_t1, draft_t2, ...]

ModelRunner.execute_model():
    # 将 draft tokens 拼接到 prompt 后面
    input = [prompt_tokens..., draft_t1, draft_t2, ...]
    output = verifier_model(input)  # 一次forward pass验证所有

    # 拒绝采样（Triton kernel）
    rejection_sample(target_sampled, input_ids, cu_num_logits)

Scheduler.update_from_output():
    num_accepted = len(accepted_tokens)
    num_rejected = len(draft_tokens) - num_accepted
    req.num_computed_tokens -= num_rejected  # 回退
```

### 9.3 拒绝采样 Triton Kernel

vLLM 用 Triton 实现了高效的拒绝采样（`rejection_sample.py`）：

```python
@triton.jit
def _rejection_sample_kernel(
    sampled_ptr,       # [num_reqs, num_spec_steps + 1]
    num_sampled_ptr,   # [num_reqs]  每个请求接受了多少token
    target_sampled_ptr,  # Verifier采样的token
    input_ids_ptr,       # Draft采样的token（移位后）
    cu_num_logits_ptr,   # 每个请求的offset
):
    req_idx = tl.program_id(0)  # 一个kernel处理一个请求
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_tok = tl.load(target_sampled_ptr + start_idx + i)
            draft_tok = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_tok)
            num_sampled += 1
            if target_tok != draft_tok:  # 拒绝！
                rejected = True
    # 如果全部接受，还有一个额外token
    if not rejected:
        tl.store(sampled_ptr + ..., tl.load(target_sampled_ptr + ...))
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)
```

### 9.4 从0实现投机解码

**文件：`05_spec_decode/spec_decode.py`**

```python
# vllm-from-scratch/05_spec_decode/spec_decode.py
"""
投机解码从0实现
包括：Draft生成 + 拒绝采样验证
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ─────────────────────────────────────────────
# 简化的模型接口（真实场景是transformer模型）
# ─────────────────────────────────────────────

class SimpleLM:
    """
    极简语言模型（用于测试）
    真实场景替换为 vLLM 中的 ModelRunner
    """
    def __init__(self, vocab_size: int, hidden_size: int = 64):
        self.vocab_size = vocab_size
        torch.manual_seed(42)
        # 简单的 embedding + linear head
        self.embedding = torch.randn(vocab_size, hidden_size) * 0.1
        self.lm_head = torch.randn(hidden_size, vocab_size) * 0.1

    def get_logits(self, token_ids: List[int]) -> torch.Tensor:
        """
        给定 token 序列，返回每个位置的 logits
        [seq_len, vocab_size]
        """
        seq_len = len(token_ids)
        # 简化：每个位置的表示是 embedding 的累积和
        hidden = torch.zeros(self.lm_head.shape[0])
        logits_list = []
        for t in token_ids:
            hidden = hidden + self.embedding[t]
            logits = hidden @ self.lm_head  # [vocab_size]
            logits_list.append(logits)
        return torch.stack(logits_list)  # [seq_len, vocab_size]


# ─────────────────────────────────────────────
# 核心算法
# ─────────────────────────────────────────────

def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 greedy: bool = True) -> int:
    """采样一个token"""
    if greedy or temperature == 0:
        return logits.argmax().item()
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def draft_generate(
    draft_model: SimpleLM,
    context_ids: List[int],
    num_speculative_tokens: int,
    temperature: float = 0.0,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Draft 模型生成 K 个候选 token

    返回:
        (draft_token_ids, draft_logits)
    """
    draft_tokens = []
    draft_logits = []
    current_ids = list(context_ids)

    for _ in range(num_speculative_tokens):
        logits = draft_model.get_logits(current_ids)
        next_logit = logits[-1]  # 最后一个位置的logits
        next_token = sample_token(next_logit, temperature)
        draft_tokens.append(next_token)
        draft_logits.append(next_logit)
        current_ids.append(next_token)

    return draft_tokens, draft_logits


def verify_and_sample(
    target_model: SimpleLM,
    context_ids: List[int],
    draft_tokens: List[int],
    temperature: float = 0.0,
) -> Tuple[List[int], int]:
    """
    验证 Draft token 并通过拒绝采样确定接受哪些

    算法（简化版）：
    对于每个 draft token t_i:
        如果 target_model.argmax(context + t_0..t_{i-1}) == t_i:
            接受 t_i
        否则:
            从 target 分布采样替换 token
            停止

    返回:
        (accepted_tokens, num_accepted)
    """
    # 一次 forward pass 验证所有 draft tokens
    all_ids = context_ids + draft_tokens
    logits = target_model.get_logits(all_ids)
    # logits[i] 是基于 all_ids[:i+1] 的预测

    # 从 context 最后一个位置开始（位置 len(context_ids)-1）
    accepted_tokens = []
    num_accepted = 0

    for i, draft_token in enumerate(draft_tokens):
        target_pos = len(context_ids) - 1 + i  # target 预测的位置
        target_logit = logits[target_pos]
        target_token = sample_token(target_logit, temperature)

        if target_token == draft_token:
            # 接受
            accepted_tokens.append(draft_token)
            num_accepted += 1
        else:
            # 拒绝：使用 target 采样的 token
            accepted_tokens.append(target_token)
            break  # 拒绝后停止

    return accepted_tokens, num_accepted


def speculative_decode_step(
    target_model: SimpleLM,
    draft_model: SimpleLM,
    context_ids: List[int],
    num_speculative_tokens: int = 5,
    temperature: float = 0.0,
) -> Tuple[List[int], dict]:
    """
    一步投机解码：生成多个 token

    返回:
        (accepted_token_ids, stats)
    """
    # 1. Draft 模型生成候选
    draft_tokens, draft_logits = draft_generate(
        draft_model, context_ids, num_speculative_tokens, temperature
    )

    # 2. Target 模型验证
    accepted_tokens, num_accepted = verify_and_sample(
        target_model, context_ids, draft_tokens, temperature
    )

    stats = {
        'num_draft': len(draft_tokens),
        'num_accepted': num_accepted,
        'acceptance_rate': num_accepted / len(draft_tokens),
        'speedup': len(accepted_tokens),  # 每步得到的token数
    }

    return accepted_tokens, stats


def rejection_sampling_batched(
    target_sampled: torch.Tensor,  # [total_draft_tokens + num_reqs]
    draft_tokens: torch.Tensor,    # [total_draft_tokens + num_reqs]  (shifted)
    cu_num_logits: torch.Tensor,   # [num_reqs + 1]  cumulative token counts
    num_speculative_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量拒绝采样（Python 版本，对应 vLLM 的 Triton kernel）

    这是 vllm/v1/worker/gpu/spec_decode/rejection_sample.py 的 Python 等价实现
    """
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.full(
        (num_reqs, num_speculative_steps + 1),
        fill_value=-1, dtype=target_sampled.dtype
    )
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32)

    for req_idx in range(num_reqs):
        start = cu_num_logits[req_idx].item()
        end = cu_num_logits[req_idx + 1].item()
        num_tokens = end - start

        n = 0
        rejected = False
        for i in range(num_tokens - 1):
            if not rejected:
                target_tok = target_sampled[start + i].item()
                draft_tok = draft_tokens[start + i + 1].item()
                sampled[req_idx, i] = target_tok
                n += 1
                if target_tok != draft_tok:
                    rejected = True

        if not rejected:
            # 全部接受，追加最后一个
            sampled[req_idx, num_tokens - 1] = target_sampled[start + num_tokens - 1]
            n += 1

        num_sampled[req_idx] = n

    return sampled, num_sampled
```

---

## 第十章：Chunked Prefill——从0实现

### 10.1 问题背景

当一个请求的 prompt 很长（如 8192 tokens），整个 prefill 会：

1. 占用大量 GPU 时间（可能 >1 秒）
2. 导致其他请求的 decode 被延迟（P99 时延飙升）

Chunked Prefill 的解决方案：**将长 prefill 分块处理**，每次只处理一部分，与其他请求的 decode 混合执行。

```
传统方式（no chunked prefill）：
  Step 1: [长prefill 8192 tokens]  ← 独占GPU，耗时长
  Step 2: [decode req1] [decode req2]

Chunked Prefill：
  Step 1: [prefill_chunk1 2048] + [decode req1] + [decode req2]
  Step 2: [prefill_chunk2 2048] + [decode req1] + [decode req2]
  ...
  Step 4: [prefill_chunk4 2048] + [decode req1] + [decode req2]
```

**好处**：
- P99 时延降低（prefill 不再独占）
- 吞吐量提升（decode 和 prefill 混合批处理，GPU 利用率更高）
- 长文档推理友好

### 10.2 vLLM 实现要点

在 `scheduler.py` 中，Chunked Prefill 的核心逻辑：

```python
# 在 schedule() 中处理新请求时
long_prefill_threshold = scheduler_config.long_prefill_token_threshold

if num_new_tokens > long_prefill_threshold > 0:
    # 超过阈值，分块处理
    num_new_tokens = long_prefill_threshold

# 每次只处理 chunk 大小的 tokens
# 请求会停留在 running 中，下次继续处理剩余部分
```

**关键数据：每个请求的 `num_computed_tokens`**

```
请求状态：
  num_prompt_tokens = 8192
  num_computed_tokens = 0      ← 第1步 schedule 时
  num_tokens_scheduled = 2048  ← 本步处理2048 tokens

  下一步：
  num_computed_tokens = 2048   ← 更新
  num_tokens_scheduled = 2048  ← 继续处理2048

  最终：
  num_computed_tokens = 8192   ← prefill完成
  开始 decode
```

### 10.3 混合批处理（Mixed Batching）的注意力计算

Chunked Prefill 最复杂的地方是**注意力计算**：在同一批次中，有的请求在做 Prefill（causal attention），有的在做 Decode（cross attention to KV cache）。

```
批次示例：
  [prefill_req_A_chunk: tokens 0-2047] + [decode_req_B] + [decode_req_C]

Attention 计算：
  - req_A（prefill部分）：全量 causal attention，Q=K=V=chunk tokens
  - req_B（decode）：cross attention to KV cache，Q=last_token, K,V=all history
  - req_C（decode）：同上

vLLM 用 FlashAttention-2 的 variable-length batch 支持实现这种混合批次
```

---

完整的代码实现请见对应子目录。博客第三部分将覆盖 DeepSeek MoE 和 PD 分离。
