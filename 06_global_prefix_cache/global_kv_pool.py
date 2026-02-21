"""
Mooncake 风格全局 KV Cache 池 —— 从零实现
对应博客第七章：Prefix Cache 全局池化版本

实现了：
1. GlobalKVPool        —— 中心化 KV Cache 元数据存储（etcd/Redis 的模拟）
2. TransferEngine      —— 异步 KV 传输（RDMA 的 Python 模拟）
3. MooncakeConnector   —— vLLM Worker 侧连接器接口
4. SimulatedWorker     —— 模拟多节点 prefill/decode 协同场景

架构对比：
  单机 Prefix Cache (02_kvcache/)：
    每个 Worker 自己维护本地 LRU 池
    命中 → 直接读本地 GPU 内存

  全局池化 (本文件)：
    全局元数据服务（Mooncake metadata server）
    命中 → RDMA 从远端节点拉取 KV 数据
    容量：TB 级（不受单卡限制）
"""

import hashlib
import heapq
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────

BLOCK_SIZE = 16        # tokens per KV block
HEAD_DIM   = 128       # 每个 attention head 的维度
NUM_LAYERS = 32        # 模型层数
NUM_HEADS  = 32        # attention head 数
DTYPE_BYTES = 2        # float16

# 每个块的 KV 数据大小（字节）
KV_BLOCK_BYTES = BLOCK_SIZE * NUM_LAYERS * NUM_HEADS * HEAD_DIM * 2 * DTYPE_BYTES


# ──────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────

@dataclass
class KVBlockMeta:
    """
    全局池中一个 KV 块的元数据
    真实 Mooncake 存入 etcd/Redis；这里用 Python dict 模拟
    """
    block_hash: int
    node_id: int                    # 存储在哪个节点（RDMA 源端）
    token_ids: Tuple[int, ...]      # 对应的 token 序列
    size_bytes: int = KV_BLOCK_BYTES
    access_count: int = 0
    last_access: float = field(default_factory=time.monotonic)
    create_time: float = field(default_factory=time.monotonic)

    def touch(self):
        self.access_count += 1
        self.last_access = time.monotonic()


@dataclass
class TransferRequest:
    """一次 KV 传输请求"""
    transfer_id: str
    src_node_id: int
    dst_node_id: int
    block_hashes: List[int]
    callback: Optional[Callable] = None
    submit_time: float = field(default_factory=time.monotonic)


@dataclass
class TransferResult:
    """传输完成结果"""
    transfer_id: str
    success: bool
    latency_ms: float
    bytes_transferred: int


# ──────────────────────────────────────────────────────────────────
# 第一层：全局元数据服务（模拟 Mooncake metadata server）
# ──────────────────────────────────────────────────────────────────

class GlobalMetadataServer:
    """
    全局 KV Cache 元数据服务

    真实实现：etcd 或专用 metadata server
    存储：block_hash → (node_id, 内存地址, 大小)

    这里用带锁的 Python dict 模拟
    """

    def __init__(self):
        self._lock = threading.RLock()
        # block_hash → KVBlockMeta
        self._blocks: Dict[int, KVBlockMeta] = {}
        # node_id → set of block_hashes（用于节点级 LRU）
        self._node_blocks: Dict[int, List[int]] = {}
        # 统计
        self.total_queries = 0
        self.total_hits = 0

    def query_prefix(self, block_hashes: List[int]) -> Tuple[int, List[KVBlockMeta]]:
        """
        查询前缀匹配情况（链式匹配，遇到 miss 即停）

        返回：
            (num_matched_blocks, matched_metas)
        """
        with self._lock:
            self.total_queries += len(block_hashes)  # 统计单个 block lookup 次数
            matched = []
            for h in block_hashes:
                if h in self._blocks:
                    meta = self._blocks[h]
                    meta.touch()
                    matched.append(meta)
                else:
                    break  # 链断开，后续 hash 依赖前缀，无法命中

            self.total_hits += len(matched)
            return len(matched), matched

    def publish(
        self,
        block_hash: int,
        node_id: int,
        token_ids: Tuple[int, ...],
        max_blocks_per_node: int = 1000,
    ) -> bool:
        """
        将一个 KV 块注册到全局元数据服务

        调用时机：Prefill 节点完成计算后
        """
        with self._lock:
            if block_hash in self._blocks:
                self._blocks[block_hash].touch()
                return True  # 已存在，更新访问时间

            # 节点容量检查
            node_hashes = self._node_blocks.setdefault(node_id, [])
            if len(node_hashes) >= max_blocks_per_node:
                self._evict_lru(node_id)

            meta = KVBlockMeta(
                block_hash=block_hash,
                node_id=node_id,
                token_ids=token_ids,
            )
            self._blocks[block_hash] = meta
            node_hashes.append(block_hash)
            return True

    def unpublish(self, block_hash: int) -> bool:
        """从全局池中移除（节点下线或主动驱逐）"""
        with self._lock:
            if block_hash not in self._blocks:
                return False
            meta = self._blocks.pop(block_hash)
            node_hashes = self._node_blocks.get(meta.node_id, [])
            if block_hash in node_hashes:
                node_hashes.remove(block_hash)
            return True

    def _evict_lru(self, node_id: int):
        """LRU 驱逐：移除该节点最久未访问的块"""
        node_hashes = self._node_blocks.get(node_id, [])
        if not node_hashes:
            return

        oldest_hash = min(
            node_hashes,
            key=lambda h: self._blocks[h].last_access if h in self._blocks else 0
        )
        self.unpublish(oldest_hash)

    def node_stats(self) -> Dict[int, Dict]:
        """返回各节点的缓存统计"""
        with self._lock:
            stats = {}
            for node_id, hashes in self._node_blocks.items():
                valid = [h for h in hashes if h in self._blocks]
                stats[node_id] = {
                    'num_blocks': len(valid),
                    'total_bytes': len(valid) * KV_BLOCK_BYTES,
                    'total_gb': len(valid) * KV_BLOCK_BYTES / 1e9,
                }
            return stats

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    @property
    def total_blocks(self) -> int:
        with self._lock:
            return len(self._blocks)


# ──────────────────────────────────────────────────────────────────
# 第二层：RDMA 传输引擎模拟
# ──────────────────────────────────────────────────────────────────

class TransferEngine:
    """
    KV Cache 传输引擎（RDMA 的 Python 模拟）

    真实 Mooncake 使用：
    - RDMA（InfiniBand / RoCEv2）：2-5μs 延迟，100Gbps 带宽
    - GPUDirect RDMA：绕过 CPU 直接 GPU-to-GPU 传输
    - Zero-copy：避免 CPU 内存拷贝

    这里模拟：
    - 网络延迟：intra-rack 200μs，cross-rack 1ms
    - 带宽：100Gbps（模拟，速度按比例缩放）
    - 异步：后台线程处理传输队列
    """

    # 延迟配置（秒）
    INTRA_RACK_LATENCY  = 0.0002   # 200μs
    CROSS_RACK_LATENCY  = 0.001    # 1ms
    BANDWIDTH_BPS       = 100e9    # 100 Gbps

    def __init__(self, node_id: int, num_workers: int = 4):
        self.node_id = node_id
        self._queue: queue.Queue = queue.Queue()
        self._results: Dict[str, TransferResult] = {}
        self._result_lock = threading.Lock()
        self._callbacks: Dict[str, Callable] = {}
        self._transfer_counter = 0
        self._counter_lock = threading.Lock()

        # 启动工作线程（模拟 RDMA 多队列对）
        self._workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def submit_transfer(
        self,
        src_node_id: int,
        block_hashes: List[int],
        same_rack: bool = True,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        提交一次 KV 传输请求（异步）

        返回 transfer_id，可用于轮询完成状态
        """
        with self._counter_lock:
            self._transfer_counter += 1
            transfer_id = f"transfer-{self.node_id}-{self._transfer_counter}"

        req = TransferRequest(
            transfer_id=transfer_id,
            src_node_id=src_node_id,
            dst_node_id=self.node_id,
            block_hashes=block_hashes,
            callback=callback,
        )

        if callback:
            self._callbacks[transfer_id] = callback

        self._queue.put(req)
        return transfer_id

    def is_complete(self, transfer_id: str) -> bool:
        """轮询传输完成状态"""
        with self._result_lock:
            return transfer_id in self._results

    def wait(self, transfer_id: str, timeout: float = 5.0) -> Optional[TransferResult]:
        """同步等待传输完成"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._result_lock:
                if transfer_id in self._results:
                    return self._results[transfer_id]
            time.sleep(0.0001)
        return None  # Timeout

    def _worker_loop(self):
        """后台工作线程：模拟 RDMA 传输"""
        while True:
            try:
                req: TransferRequest = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 模拟传输时间
            num_bytes = len(req.block_hashes) * KV_BLOCK_BYTES
            transfer_time_s = num_bytes / self.BANDWIDTH_BPS

            # 选取延迟（同机架 or 跨机架，用节点 ID 奇偶判断）
            base_latency = (
                self.INTRA_RACK_LATENCY
                if (req.src_node_id // 4) == (req.dst_node_id // 4)  # 同机架（4 节点/rack）
                else self.CROSS_RACK_LATENCY
            )
            total_latency = base_latency + transfer_time_s
            time.sleep(total_latency)

            result = TransferResult(
                transfer_id=req.transfer_id,
                success=True,
                latency_ms=total_latency * 1000,
                bytes_transferred=num_bytes,
            )

            with self._result_lock:
                self._results[req.transfer_id] = result

            cb = self._callbacks.pop(req.transfer_id, None)
            if cb:
                cb(result)


# ──────────────────────────────────────────────────────────────────
# 第三层：MooncakeConnector（vLLM Worker 侧接口）
# ──────────────────────────────────────────────────────────────────

class MooncakeConnector:
    """
    vLLM Worker 侧的 Mooncake 连接器

    每个 vLLM Worker（Prefill 或 Decode 节点）持有一个实例

    核心 API：
    1. query_prefix_cache(block_hashes) → 命中数，发起异步传输
    2. wait_for_kv(request_id)          → 等待 KV 到达本节点
    3. publish_kv(block_hashes, node_id) → 将本节点 KV 发布到全局池
    """

    def __init__(
        self,
        node_id: int,
        metadata_server: GlobalMetadataServer,
        transfer_engine: TransferEngine,
        max_pending_transfers: int = 32,
    ):
        self.node_id = node_id
        self._meta = metadata_server
        self._engine = transfer_engine
        self._max_pending = max_pending_transfers

        # request_id → transfer_id（追踪进行中的传输）
        self._pending: Dict[str, str] = {}
        self._lock = threading.Lock()

        # 统计
        self.stats = {
            'queries': 0,
            'hits': 0,
            'transfers_started': 0,
            'transfers_completed': 0,
            'bytes_received': 0,
        }

    def get_num_new_matched_tokens(
        self,
        request_id: str,
        block_hashes: List[int],
    ) -> Tuple[int, bool]:
        """
        对应 vLLM 的 KVConnectorBase.get_num_new_matched_tokens()

        返回：
            (num_matched_tokens, load_kv_async)
            - num_matched_tokens: 命中的 token 数
            - load_kv_async: True 表示已发起异步传输
        """
        self.stats['queries'] += 1

        num_matched, matched_metas = self._meta.query_prefix(block_hashes)

        if num_matched == 0:
            return 0, False

        self.stats['hits'] += num_matched

        # 按源节点分组，批量发起传输（同节点的块一次 RDMA 请求）
        by_node: Dict[int, List[int]] = {}
        for meta in matched_metas:
            by_node.setdefault(meta.node_id, []).append(meta.block_hash)

        transfer_ids = []
        for src_node_id, hashes in by_node.items():
            if src_node_id == self.node_id:
                continue  # 本节点的块，无需传输

            tid = self._engine.submit_transfer(
                src_node_id=src_node_id,
                block_hashes=hashes,
            )
            transfer_ids.append(tid)
            self.stats['transfers_started'] += 1

        if transfer_ids:
            with self._lock:
                self._pending[request_id] = transfer_ids

        matched_tokens = num_matched * BLOCK_SIZE
        return matched_tokens, bool(transfer_ids)

    def wait_for_kv(self, request_id: str, timeout: float = 5.0) -> bool:
        """
        等待该请求所有 KV 传输完成

        对应 vLLM 的 WAITING_FOR_REMOTE_KVS 状态
        """
        with self._lock:
            transfer_ids = self._pending.pop(request_id, [])

        if not transfer_ids:
            return True  # 无需等待（本地命中）

        for tid in transfer_ids:
            result = self._engine.wait(tid, timeout=timeout)
            if result is None or not result.success:
                return False
            self.stats['transfers_completed'] += 1
            self.stats['bytes_received'] += result.bytes_transferred

        return True

    def publish_kv(
        self,
        block_hashes: List[int],
        token_ids_per_block: List[Tuple[int, ...]],
        max_blocks_per_node: int = 1000,
    ) -> int:
        """
        Prefill 完成后将 KV 发布到全局池

        对应 vLLM 的 request_finished() 回调
        返回：成功发布的块数
        """
        count = 0
        for h, tids in zip(block_hashes, token_ids_per_block):
            ok = self._meta.publish(
                block_hash=h,
                node_id=self.node_id,
                token_ids=tids,
                max_blocks_per_node=max_blocks_per_node,
            )
            if ok:
                count += 1
        return count


# ──────────────────────────────────────────────────────────────────
# 工具函数：块哈希计算
# ──────────────────────────────────────────────────────────────────

def compute_block_hash(
    parent_hash: int,
    token_ids: Tuple[int, ...],
    extra_key: Optional[int] = None,
) -> int:
    """
    计算块哈希（链式依赖，保证前缀唯一性）

    与 vLLM 的 hash_block_tokens 一致：
      hash = hash((parent_hash, token_ids, extra_key))
    """
    return hash((parent_hash, token_ids, extra_key))


def compute_block_hashes(
    token_ids: List[int],
    block_size: int = BLOCK_SIZE,
    extra_key: Optional[int] = None,
) -> List[int]:
    """计算整个序列的块哈希链"""
    hashes = []
    parent_hash = 0  # 初始 seed

    for i in range(0, len(token_ids) - block_size + 1, block_size):
        block_tokens = tuple(token_ids[i:i + block_size])
        h = compute_block_hash(parent_hash, block_tokens, extra_key)
        hashes.append(h)
        parent_hash = h

    return hashes


# ──────────────────────────────────────────────────────────────────
# 模拟场景：多节点 Prefill + Decode 协同
# ──────────────────────────────────────────────────────────────────

class SimulatedCluster:
    """
    模拟一个多节点 vLLM 集群

    拓扑：
    - N 个 Prefill 节点（计算密集）
    - M 个 Decode 节点（内存带宽密集）
    - 1 个全局 Metadata Server（etcd 模拟）
    - 每个节点有独立的 TransferEngine

    场景：同一个系统提示词被多个用户共享
    """

    def __init__(self, num_prefill_nodes: int = 2, num_decode_nodes: int = 2):
        self.meta_server = GlobalMetadataServer()

        # Prefill 节点
        self.prefill_nodes: List[MooncakeConnector] = []
        for i in range(num_prefill_nodes):
            engine = TransferEngine(node_id=i)
            conn = MooncakeConnector(
                node_id=i,
                metadata_server=self.meta_server,
                transfer_engine=engine,
            )
            self.prefill_nodes.append(conn)

        # Decode 节点（节点 ID 从 num_prefill_nodes 开始）
        self.decode_nodes: List[MooncakeConnector] = []
        for j in range(num_decode_nodes):
            node_id = num_prefill_nodes + j
            engine = TransferEngine(node_id=node_id)
            conn = MooncakeConnector(
                node_id=node_id,
                metadata_server=self.meta_server,
                transfer_engine=engine,
            )
            self.decode_nodes.append(conn)

    def simulate_prefill(
        self,
        request_id: str,
        prompt_tokens: List[int],
        prefill_node_idx: int = 0,
    ) -> Tuple[int, float]:
        """
        模拟 Prefill 节点处理一个请求

        流程：
        1. 检查全局 Prefix Cache
        2. 计算未命中部分（实际 prefill）
        3. 将结果发布到全局池

        返回：(num_cached_tokens, prefill_time_ms)
        """
        conn = self.prefill_nodes[prefill_node_idx]
        block_hashes = compute_block_hashes(prompt_tokens)

        # Step1: 查询全局缓存
        num_cached, needs_transfer = conn.get_num_new_matched_tokens(
            request_id, block_hashes
        )

        if needs_transfer:
            conn.wait_for_kv(request_id)

        # Step2: 计算未命中部分
        num_to_compute = max(0, len(prompt_tokens) - num_cached)
        # 模拟 prefill 计算时间：0.5ms per token (A100 估算)
        prefill_time_ms = num_to_compute * 0.5

        # Step3: 发布到全局池
        token_ids_per_block = []
        for i in range(0, len(prompt_tokens) - BLOCK_SIZE + 1, BLOCK_SIZE):
            token_ids_per_block.append(tuple(prompt_tokens[i:i + BLOCK_SIZE]))

        conn.publish_kv(block_hashes, token_ids_per_block)

        return num_cached, prefill_time_ms

    def simulate_decode(
        self,
        request_id: str,
        prompt_tokens: List[int],
        decode_node_idx: int = 0,
    ) -> Tuple[int, float]:
        """
        模拟 Decode 节点处理一个请求
        从全局池拉取 KV，跳过 prefill

        返回：(num_from_cache, transfer_wait_ms)
        """
        conn = self.decode_nodes[decode_node_idx]
        block_hashes = compute_block_hashes(prompt_tokens)

        t0 = time.monotonic()
        num_cached, needs_transfer = conn.get_num_new_matched_tokens(
            request_id, block_hashes
        )

        if needs_transfer:
            conn.wait_for_kv(request_id)

        wait_ms = (time.monotonic() - t0) * 1000
        return num_cached, wait_ms


# ──────────────────────────────────────────────────────────────────
# 演示与测试
# ──────────────────────────────────────────────────────────────────

def demo_basic_flow():
    """演示基本的发布-查询流程"""
    print("=" * 60)
    print("演示：基本发布-查询流程")
    print("=" * 60)

    meta = GlobalMetadataServer()

    # 模拟 2000 token 的系统提示词
    sys_prompt = list(range(2000))
    block_hashes = compute_block_hashes(sys_prompt)
    print(f"系统提示词 {len(sys_prompt)} tokens → {len(block_hashes)} 个块")

    # Prefill 节点 0 完成计算，发布 KV 到全局池
    for i, (h, block_hashes_elem) in enumerate(zip(block_hashes, block_hashes)):
        tids = tuple(sys_prompt[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        meta.publish(block_hash=h, node_id=0, token_ids=tids)
    print(f"发布完成：全局池现有 {meta.total_blocks} 个块")

    # 第二个请求查询（完全命中）
    num_matched, matched_metas = meta.query_prefix(block_hashes)
    print(f"第二个请求查询：命中 {num_matched} 个块 = {num_matched * BLOCK_SIZE} tokens")
    print(f"节省 prefill 计算：{num_matched * BLOCK_SIZE * 0.5:.0f}ms（@0.5ms/token）")


def demo_multi_node():
    """演示多节点 Prefix Cache 共享"""
    print("\n" + "=" * 60)
    print("演示：多节点 KV Cache 共享（Mooncake 风格）")
    print("=" * 60)

    cluster = SimulatedCluster(num_prefill_nodes=2, num_decode_nodes=2)

    # 公共系统提示词（所有用户共享）
    sys_prompt = list(range(512))  # 512 token 系统提示词
    user_queries = [
        [1001, 1002, 1003, 1004],    # 用户A
        [2001, 2002, 2003],           # 用户B
        [3001, 3002, 3003, 3004, 3005],  # 用户C
    ]

    print(f"\n系统提示词：{len(sys_prompt)} tokens")
    print(f"用户数：{len(user_queries)}")
    print()

    # 第一个请求：无缓存，完整 prefill
    req0_tokens = sys_prompt + user_queries[0]
    cached0, time0 = cluster.simulate_prefill("req-0", req0_tokens, prefill_node_idx=0)
    print(f"请求 0（首次）：缓存命中 {cached0} tokens，计算耗时 ~{time0:.0f}ms")

    # 第二、三个请求：系统提示词命中
    req1_tokens = sys_prompt + user_queries[1]
    cached1, time1 = cluster.simulate_prefill("req-1", req1_tokens, prefill_node_idx=1)
    print(f"请求 1（复用系统提示词）：缓存命中 {cached1} tokens，计算耗时 ~{time1:.0f}ms")

    req2_tokens = sys_prompt + user_queries[2]
    cached2, time2 = cluster.simulate_prefill("req-2", req2_tokens, prefill_node_idx=0)
    print(f"请求 2（复用系统提示词）：缓存命中 {cached2} tokens，计算耗时 ~{time2:.0f}ms")

    # 打印全局池统计
    print(f"\n全局 KV 池统计：")
    print(f"  总块数：{cluster.meta_server.total_blocks}")
    print(f"  命中率：{cluster.meta_server.hit_rate * 100:.1f}%")
    for node_id, stats in cluster.meta_server.node_stats().items():
        print(f"  节点 {node_id}：{stats['num_blocks']} 块，{stats['total_gb']:.2f} GB")


def demo_cross_node_transfer():
    """演示跨节点 KV 传输（RDMA 模拟）"""
    print("\n" + "=" * 60)
    print("演示：跨节点 KV Cache 传输")
    print("=" * 60)

    cluster = SimulatedCluster(num_prefill_nodes=1, num_decode_nodes=1)

    # 系统提示词先在 Prefill 节点完成计算
    sys_prompt = list(range(320))  # 320 tokens = 20 块
    t_start = time.monotonic()
    cached, prefill_ms = cluster.simulate_prefill("prefill-req", sys_prompt, prefill_node_idx=0)
    t_prefill = (time.monotonic() - t_start) * 1000

    print(f"Prefill 节点计算完成：{len(sys_prompt)} tokens，发布 {cluster.meta_server.total_blocks} 块到全局池")
    print(f"耗时：{t_prefill:.1f}ms（含模拟计算延迟 {prefill_ms:.0f}ms）")

    # Decode 节点从全局池拉取
    t_start = time.monotonic()
    from_cache, transfer_ms = cluster.simulate_decode("decode-req", sys_prompt, decode_node_idx=0)
    t_decode = (time.monotonic() - t_start) * 1000

    # from_cache 是 token 数（get_num_new_matched_tokens 返回 tokens）
    num_blocks = from_cache // BLOCK_SIZE
    num_bytes = num_blocks * KV_BLOCK_BYTES
    bandwidth_gbps = (num_bytes / (transfer_ms / 1000)) / 1e9 if transfer_ms > 0 else float('inf')

    print(f"\nDecode 节点通过 RDMA 拉取：")
    print(f"  命中：{num_blocks} 块 = {from_cache} tokens")
    print(f"  传输数据：{num_bytes / 1e6:.1f} MB")
    print(f"  等待时间：{transfer_ms:.1f}ms")
    print(f"  等效带宽：{bandwidth_gbps:.1f} Gbps（模拟值，真实 RDMA 约 100-400 Gbps）")

    # 对比：如果 Decode 节点自己做 Prefill
    standalone_prefill_ms = len(sys_prompt) * 0.5
    saving_ms = standalone_prefill_ms - transfer_ms
    print(f"\n  若不用全局缓存，Decode 节点需要自行 Prefill：{standalone_prefill_ms:.0f}ms")
    print(f"  节省时间（TTFT）：{saving_ms:.0f}ms（{saving_ms/standalone_prefill_ms*100:.0f}%）")


def demo_cache_capacity():
    """演示全局缓存容量 vs 单机缓存"""
    print("\n" + "=" * 60)
    print("演示：全局缓存容量分析")
    print("=" * 60)

    gpu_vram_gb = 80        # 单张 A100 80GB
    kv_ratio    = 0.3       # KV Cache 占用比例
    kv_gb = gpu_vram_gb * kv_ratio

    dram_per_node_gb = 512  # 每个 CPU 节点 512GB DRAM
    num_cpu_nodes    = 8    # 8 个 CPU 节点

    per_token_kv_bytes = NUM_LAYERS * NUM_HEADS * HEAD_DIM * 2 * DTYPE_BYTES

    gpu_max_tokens  = int(kv_gb * 1e9 / per_token_kv_bytes)
    dram_max_tokens = int(dram_per_node_gb * num_cpu_nodes * 1e9 / per_token_kv_bytes)

    print(f"单机 GPU KV Cache：")
    print(f"  容量：{kv_gb:.0f} GB")
    print(f"  可缓存：{gpu_max_tokens:,} tokens")

    print(f"\n全局池（Mooncake DRAM）：")
    print(f"  容量：{dram_per_node_gb * num_cpu_nodes:.0f} GB（{num_cpu_nodes} 节点 × {dram_per_node_gb} GB）")
    print(f"  可缓存：{dram_max_tokens:,} tokens")
    print(f"  容量倍数：{dram_max_tokens / gpu_max_tokens:.0f}x")

    # 系统提示词场景
    sys_prompt_tokens = 32768  # 32K token 系统提示词
    num_concurrent_users = 1000
    total_tokens_needed  = sys_prompt_tokens * num_concurrent_users

    print(f"\n场景：{num_concurrent_users} 个并发用户，{sys_prompt_tokens} token 系统提示词")
    print(f"  如无缓存，每轮推理需要：{total_tokens_needed:,} 个 token 的 Prefill 计算")
    print(f"  全局缓存命中后：只需计算 {sys_prompt_tokens:,} tokens（一次），节省 {num_concurrent_users-1}x")


if __name__ == "__main__":
    demo_basic_flow()
    demo_multi_node()
    demo_cross_node_transfer()
    demo_cache_capacity()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
