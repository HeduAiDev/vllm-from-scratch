# vLLM 从入门到专家（2.0 版）第二部分

> **本部分内容**：Prefix Cache（单机与全局池化）、Scheduler 调度算法、投机解码（EAGLE）、Chunked Prefill。
>
> **阅读前提**：已读第一部分，理解 PagedAttention、KV Cache 块管理。

---

## 第六章：Prefix Cache——避免重复计算的核心优化

### 6.1 理论背景：重复前缀的普遍性

在生产环境中，绝大多数 LLM 请求共享相同的前缀：

| 场景 | 共享内容 | 典型长度 |
|------|---------|---------|
| 多轮对话 | 对话历史 | 500-8000 tokens |
| RAG 检索 | 检索到的文档 | 1000-4000 tokens |
| 代码补全 | 文件上下文 | 2000-16000 tokens |
| 系统提示词 | 固定系统指令 | 100-2000 tokens |

如果不做任何优化，每个请求都要对这些共享前缀做完整的 Prefill 计算，浪费极大。

**论文参考**：
- 前缀缓存的思想在 *RadixAttention*（SGLang，2024）中被系统化：
  Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024
  https://arxiv.org/abs/2312.07104
- vLLM 的 Prefix Cache 实现参考此工作，并针对 PagedAttention 的块结构优化

### 6.2 Radix Tree：理解前缀共享的数据结构

实现前缀缓存，核心是一个**前缀树（Trie / Radix Tree）**数据结构。

```
场景：3 个请求的 token 序列

请求A：[SYS] [Q1] [Q1_a]
请求B：[SYS] [Q2] [Q2_b]
请求C：[SYS] [Q1] [Q1_c]

Radix Tree 表示：
         root
          │
        [SYS]──────────────(cached block: #7)
          │
     ┌────┴─────┐
  [Q1]        [Q2]
   │            │
 ┌─┴──┐       [Q2_b]
[Q1_a][Q1_c]

共享的物理块：
  [SYS] → Block #7（所有3个请求共享，ref_count=3）
  [Q1]  → Block #12（请求A和C共享，ref_count=2）
  其余  → 各自独立的块
```

vLLM 的实现没有用严格的 Radix Tree，而是用**块级哈希**近似实现同样的效果：

```
块哈希链：
  Block0: hash(seed=0,    tokens[0:16])    = H0
  Block1: hash(parent=H0, tokens[16:32])   = H1
  Block2: hash(parent=H1, tokens[32:48])   = H2

查询时：逐块计算哈希，在缓存中查找，遇到 miss 停止
效果：等价于 Radix Tree 的前缀匹配
```

### 6.3 vLLM 的单机 Prefix Cache 实现

#### 6.3.1 整体架构

```
vLLM Prefix Cache 架构（单机版）：

  ┌──────────────────────────────────────────────────────┐
  │                   KVCacheManager                     │
  │                                                      │
  │  ┌─────────────────────────────────────────────┐    │
  │  │              BlockPool                       │    │
  │  │                                             │    │
  │  │  free_block_queue（LRU 双向链表）:           │    │
  │  │  [oldest] ↔ B3 ↔ B7 ↔ B1 ↔ [newest]       │    │
  │  │                                             │    │
  │  │  cached_blocks（哈希表）:                   │    │
  │  │  { H0 → Block#7, H1 → Block#23, ... }      │    │
  │  │                                             │    │
  │  │  blocks（全量）:                            │    │
  │  │  { block_id → Block(ref_count, hash, ...) } │    │
  │  └─────────────────────────────────────────────┘    │
  │                                                      │
  │  allocate_slots(request, num_new_tokens):            │
  │    1. compute_block_hashes(request.tokens)           │
  │    2. For each hash: get_cached() or allocate_new()  │
  │    3. 返回 slot_mapping                             │
  └──────────────────────────────────────────────────────┘
```

**关键源码**（`vllm/v1/core/kv_cache_utils.py`）：

```python
def hash_block_tokens(
    parent_block_hash: int,
    curr_block_token_ids: tuple[int, ...],
    extra_keys: tuple | None = None,
) -> BlockHash:
    """
    计算 KV Cache 块的哈希值

    参数说明：
    - parent_block_hash：前一个块的哈希，形成链式依赖
    - curr_block_token_ids：本块的 token IDs（固定16个）
    - extra_keys：额外标识符（如 LoRA adapter ID，多模态嵌入哈希等）

    关键设计：链式依赖确保相同内容但不同前缀的块哈希不同
    """
    return BlockHash(hash((parent_block_hash, curr_block_token_ids, extra_keys)))
```

#### 6.3.2 缓存命中的完整流程

```
新请求到达：tokens = [SYS_0..SYS_31, Q_0..Q_15]（3个块）

Step 1: 计算块哈希
  H0 = hash(0,          (SYS_0..SYS_15)) = 0xABCD
  H1 = hash(0xABCD,     (SYS_16..SYS_31)) = 0x1234
  H2 = hash(0x1234,     (Q_0..Q_15))      = 0x5678

Step 2: 逐块查询 cached_blocks
  cached_blocks.get(0xABCD) → Block#7  ← HIT！ref_count: 1→2
  cached_blocks.get(0x1234) → Block#23 ← HIT！ref_count: 1→2
  cached_blocks.get(0x5678) → None     ← MISS，分配新块 Block#4

Step 3: 构建 block_table
  block_table = [7, 23, 4]
  num_cached_tokens = 2 * 16 = 32
  num_new_tokens = 16（只需计算 Q 部分）

Step 4: 调度器报告
  SchedulerOutput.num_scheduled_tokens[req_id] = 16（只计算未命中部分）
  → GPU 只做 16 token 的 Prefill，节省 32 token 的计算！
```

### 6.4 从零实现：带 LRU 的 Prefix Cache

> 完整代码：`02_kvcache/block_pool_lru.py`

#### 6.4.1 架构设计

```
BlockPoolLRU 内部架构：

free_block_queue（双向链表，O(1) 插入删除）：
  sentinel ↔ Block(id=3, hash=None, atime=1000)
           ↔ Block(id=9, hash=H5,   atime=1050)
           ↔ Block(id=2, hash=H2,   atime=2000)
           ↔ sentinel
  ↑最旧（优先驱逐）                ↑最新（最近释放）

cached_blocks（哈希表）：
  { H2 → Block#2, H5 → Block#9, ... }

LRU 不变量：
  - 任何 ref_count==0 的块都在 free_block_queue 中
  - free_block_queue 按释放时间排序（最旧在头部）
  - 驱逐时从头部取，保留最近使用的块（Prefix Cache 更有效）
```

**核心方法时间复杂度**：

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| `allocate()` | O(1) | 从队列头取空闲块 |
| `free(block_id)` | O(1) | 加入队列尾（双向链表） |
| `cache_hit(hash)` | O(1) | 哈希表查找 |
| `evict_lru()` | O(1) | 从队列头删除 |
| `mark_cached(block_id, hash)` | O(1) | 哈希表插入 |

---

## 第七章：全局 Prefix Cache——Mooncake 风格的分布式 KV 池

### 7.1 理论背景：单机 Prefix Cache 的局限性

单机版 Prefix Cache 受限于**单张 GPU 的显存**。以 A100 80GB 为例：

```
显存分配：
  模型权重（LLaMA-70B）：约 140GB（需要 2 张 GPU）
  剩余 KV Cache：约 20GB
  可缓存 token 数（LLaMA-70B）：
    20GB / (32层 × 8头 × 128维 × 2 × 2字节) = ~20万 tokens

问题：
  - 一个 32K token 的系统提示词 = 32,000 / 200,000 = 16% 的缓存空间
  - 100 个不同用户的对话历史全都缓存：不可能
  - 多节点集群中，每个节点各自维护缓存：重复计算
```

**解决思路**：把 KV Cache 移到容量更大的存储层（CPU DRAM、SSD），建立**全局共享的 KV 池**。

**关键论文**：
- Mooncake（月之暗面）：*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, 2024
  https://arxiv.org/abs/2407.00079
- PD 分离的先驱工作：*Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA 2024
  https://arxiv.org/abs/2311.18677
- KV Cache 卸载：*InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management*, OSDI 2024

### 7.2 全局 KV Cache 池的架构

```
Mooncake 全局 KV Cache 池架构：

┌─────────────────────────────────────────────────────────────────┐
│                  全局 Metadata Server（etcd/Redis）              │
│    block_hash → { node_id, memory_addr, size, access_time }     │
│    提供：publish() / query() / invalidate() API                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ 元数据查询（μs 级）
              ┌──────────────▼──────────────┐
              │         vLLM Workers         │
              │                             │
  ┌───────────┴──────┐        ┌─────────────┴────────┐
  │   Prefill 节点   │        │    Decode 节点        │
  │   (Node 0,1,...) │        │    (Node 4,5,...)     │
  │                  │        │                       │
  │  ① 计算 KV Cache │RDMA→  │  ② 接收 KV Cache     │
  │  ② 发布到全局池  │        │  ③ 直接开始 Decode   │
  │                  │        │                       │
  │  GPU VRAM: 80GB  │        │  GPU VRAM: 80GB       │
  │  CPU DRAM: 512GB │        │  CPU DRAM: 512GB      │
  └──────────────────┘        └───────────────────────┘
            │
            ▼
  ┌──────────────────┐
  │  分布式存储节点  │
  │  NVMe SSD: 2TB   │
  │  （冷缓存层）    │
  └──────────────────┘

关键技术：
  RDMA（Remote Direct Memory Access）：
    - 跳过 CPU，直接 GPU→远端内存 传输
    - InfiniBand / RoCEv2 网络
    - 延迟：2-5μs（vs. TCP/IP: 50-100μs）
    - 带宽：100-400 Gbps
```

### 7.3 vLLM 的 MooncakeConnector 实现

vLLM V1 定义了一个抽象接口 `KVConnectorBase_V1`，Mooncake 实现了这个接口：

**关键源码路径**：
```
vllm/distributed/kv_transfer/kv_connector/v1/
├── base.py                # KVConnectorBase_V1 抽象基类
└── mooncake/
    ├── mooncake_connector.py   # 核心实现
    └── mooncake_utils.py       # 辅助工具（bootstrap server 注册等）
```

**接口定义**（`base.py`）：

```python
class KVConnectorBase_V1(ABC):
    """
    vLLM V1 KV 传输连接器基类

    生命周期：
    1. get_num_new_matched_tokens()  → 查询命中，发起异步传输
    2. update_state_after_alloc()    → KV 块分配后通知
    3. [等待传输完成]                → 请求状态：WAITING_FOR_REMOTE_KVS
    4. request_finished()           → 请求完成，决定是否发布到全局池
    """

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        查询全局缓存命中情况

        返回：
          (num_matched_tokens, load_kv_async)
          - num_matched_tokens: 命中的 token 数（None 表示不可用）
          - load_kv_async: True 表示已发起异步 KV 传输
        """
        ...

    @abstractmethod
    def update_state_after_alloc(
        self,
        request: Request,
        num_external_tokens: int,
    ) -> None:
        """KV 块分配完成后，告知连接器准备接收数据"""
        ...

    @abstractmethod
    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> None:
        """请求完成，可选地将 KV 发布到全局池"""
        ...
```

**MooncakeConnector 的核心逻辑**（简化版，完整实现见 vLLM 源码）：

```python
class MooncakeConnector(KVConnectorBase_V1):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # 1. 计算请求的块哈希链
        block_hashes = compute_block_hashes(request.token_ids, self.block_size)

        # 2. 向 Metadata Server 查询
        matched_blocks = self.metadata_client.query_prefix(block_hashes)

        if not matched_blocks:
            return 0, False

        # 3. 按源节点分组，批量发起 RDMA 传输
        for src_node, blocks_on_node in group_by_node(matched_blocks):
            self.transfer_engine.start_async_transfer(
                src=src_node,
                dst=self.local_node,
                block_addrs=[b.memory_addr for b in blocks_on_node],
                local_addrs=self.reserve_local_slots(len(blocks_on_node)),
            )

        num_matched_tokens = len(matched_blocks) * self.block_size
        return num_matched_tokens, True  # load_kv_async=True

    def request_finished(self, request, block_ids):
        # 将本请求计算的 KV 发布到全局池
        if len(request.output_token_ids) + request.prompt_len >= MIN_CACHE_TOKENS:
            self.metadata_client.publish(
                block_hashes=request.block_hashes,
                node_id=self.local_node_id,
                memory_addrs=self.get_kv_addrs(block_ids),
            )
```

### 7.4 RDMA 传输的关键细节

```
RDMA One-Sided Read 流程：

Decode 节点（目标）              Prefill 节点（源端）
────────────────────────────────────────────────────
① 向 Metadata Server 查询
   block_hash → (src_node, addr)

② 注册本地接收缓冲区
   local_buf = register_mr(gpu_memory)

③ 发起 RDMA READ：
   ibv_post_send(
     src_addr = remote_gpu_addr,   ← 直接从远端 GPU
     dst_addr = local_gpu_addr,    ← 写入本地 GPU
     length = block_size_bytes,
   )

                               （无需 CPU 干预！）
                               ← RDMA 硬件直接传输

④ 等待 Completion Queue 通知
   传输完成，KV Cache 就绪

⑤ 通知调度器：WAITING_FOR_REMOTE_KVS → RUNNING
```

**为什么 RDMA 这么快？**

```
传统网络（TCP/IP）：
  GPU → CPU内存 → NIC → 网络 → NIC → CPU内存 → GPU
  每步都有拷贝和上下文切换，延迟 50-100μs

RDMA（GPUDirect RDMA）：
  GPU → RDMA NIC → 网络 → RDMA NIC → GPU
  跳过 CPU，延迟 2-5μs，实现零拷贝
```

### 7.5 从零实现：模拟多节点全局 KV 池

> 完整代码：`06_global_prefix_cache/global_kv_pool.py`

#### 7.5.1 架构设计

```
我们的模拟实现（不依赖真实 RDMA）：

┌──────────────────────────────────────────────────────────┐
│             GlobalMetadataServer                          │
│  _blocks: dict[hash → KVBlockMeta(node_id, addr, ...)]   │
│  _node_blocks: dict[node_id → list[hash]]（用于 LRU）    │
│  方法：query_prefix() / publish() / unpublish()          │
│  特性：线程安全（RLock），LRU 驱逐                       │
└────────────────────────────────┬─────────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │          TransferEngine                   │
              │  后台工作线程池，模拟 RDMA 异步传输        │
              │  延迟：同机架 200μs，跨机架 1ms           │
              │  带宽：100 Gbps（根据 block 大小计算时延） │
              │  方法：submit_transfer() / wait() /       │
              │         is_complete() / callback()        │
              └───────────────────┬──────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │         MooncakeConnector                 │
              │  （每个 vLLM Worker 实例持有一个）        │
              │  get_num_new_matched_tokens()             │
              │    → 查元数据 → 发起传输                  │
              │  wait_for_kv()                            │
              │    → 等传输完成（对应 WAITING_FOR_KV）    │
              │  publish_kv()                             │
              │    → 发布到全局池                         │
              └──────────────────────────────────────────┘
```

#### 7.5.2 关键实现解读

**块哈希与前缀匹配**（等价于 Radix Tree 前缀匹配）：

```python
def query_prefix(self, block_hashes: list[int]) -> tuple[int, list[KVBlockMeta]]:
    """
    链式匹配：一旦某块 MISS，后续所有块必然 MISS

    这是因为 hash[i] 依赖 hash[i-1]：
      hash[i] = hash(hash[i-1], tokens[i*BS:(i+1)*BS])

    如果 block_i MISS：
      要么 tokens[0..i*BS] 没有被缓存（完全不同的请求）
      要么 tokens[i*BS..(i+1)*BS] 不同（此后所有块的 parent_hash 不同）

    → 后续块不可能命中，可以安全停止查询
    """
    matched = []
    for h in block_hashes:
        if h in self._blocks:
            matched.append(self._blocks[h])
        else:
            break  # 链断开，提前返回
    return len(matched), matched
```

**异步传输等待**（对应 WAITING_FOR_REMOTE_KVS 状态）：

```python
def wait_for_kv(self, request_id: str, timeout: float = 5.0) -> bool:
    """
    等待该请求所有远端 KV 传输完成

    在真实 vLLM 中，这对应请求状态：
    WAITING_FOR_REMOTE_KVS → RUNNING

    调度器在每步检查 is_transfer_complete()，完成后才将请求放入批次
    """
    transfer_ids = self._pending.pop(request_id, [])
    for tid in transfer_ids:
        result = self._engine.wait(tid, timeout=timeout)
        if result is None or not result.success:
            return False
    return True
```

**关键数字验证**：

```
场景：512 token 系统提示词（32个块），3个并发请求

第1个请求（冷启动）：
  - 全部 MISS，计算 512 tokens 的 Prefill
  - 耗时：~258ms（@0.5ms/token）
  - 发布 32 个块到全局池

第2、3个请求：
  - 32 个块全部 HIT
  - 只需计算用户问题部分（假设 4 tokens）
  - 耗时：~2ms
  - 节省：(512-4) × 0.5ms = 254ms per request

全局池统计：
  命中率：66.7%（2/3 的请求完全命中系统提示词）
  节省计算：254ms × 2 = 508ms
```

---

## 第八章：Scheduler——高效批调度的艺术

### 8.1 理论背景：调度算法的权衡

LLM 推理调度本质上是一个**在线装箱问题（Online Bin Packing）**的变体，约束条件是：

- **token budget**：每步最多处理 `max_num_batched_tokens` 个 token
- **并发上限**：同时处理 `max_num_seqs` 个请求
- **KV Cache 容量**：总物理块数有限

这个问题的最优解是 NP-hard 的，vLLM 使用了一系列启发式策略。

**经典论文**：
- Continuous Batching：*Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022
  https://www.usenix.org/conference/osdi22/presentation/yu
- LLM 调度综述：*Efficiently Scaling Transformer Inference*, MLSys 2023
  https://arxiv.org/abs/2211.05100

### 8.2 vLLM Scheduler 的设计

#### 8.2.1 请求的三种队列

```
vLLM Scheduler 状态机：

  新请求到达
      │
      ▼
  [WAITING 队列]──────→ 等待足够的 KV Cache 块
      │
      │ schedule() 选中
      ▼
  [RUNNING 队列]──────→ 正在 Prefill 或 Decode
      │           │
      │           │ KV Cache 不足时
      │           ▼
      │     [抢占（Preemption）]
      │       - 换出 KV Cache 到 CPU
      │       - 或直接丢弃（重计算）
      │       - 重新进入 WAITING
      │
      │ 生成 EOS 或达到 max_tokens
      ▼
  [FINISHED]───────────→ 释放 KV Cache 块
```

#### 8.2.2 请求的阶段状态

每个 Request 对象维护以下关键字段，决定它当前处于哪个阶段：

```python
class Request:
    num_prompt_tokens: int      # prompt 的总 token 数（固定不变）
    num_computed_tokens: int    # 已经完成计算并写入 KV Cache 的 token 数
                                # 初始=0（或 prefix cache 命中数）
                                # 每步 update_from_output 后递增

    # 判断阶段：
    @property
    def is_prefill(self) -> bool:
        # 还有未计算的 prompt token → 仍在 Prefill 阶段
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        # 剩余需要计算的 prompt token 数（用于 Chunked Prefill）
        return self.num_prompt_tokens - self.num_computed_tokens
```

关键逻辑：
- `num_computed_tokens < num_prompt_tokens` → **Prefill 阶段**（可能因 Chunked Prefill 需要多步）
- `num_computed_tokens == num_prompt_tokens` → **Decode 阶段**（开始逐 token 生成）

#### 8.2.3 调度主循环（核心逻辑）

**关键源码**（`vllm/v1/core/sched/scheduler.py`，简化版）：

```python
def schedule(self) -> SchedulerOutput:
    """
    每步推理的调度决策

    优先级顺序（为什么这样排？）：
    1. RUNNING 中的 Decode 请求（每个只消耗 1 token budget，且已有完整 KV Cache，
       不调度会造成 KV Cache 闲置浪费，优先级最高）
    2. RUNNING 中的 Prefill 请求（Chunked Prefill 的后续 chunk，
       这些请求已经占用了 KV Cache 块，需要尽快完成 prefill）
    3. WAITING 中的新请求（按 FCFS 顺序，需要分配新的 KV Cache 块）
    """
    token_budget = self.scheduler_config.max_num_batched_tokens
    # token_budget：本步最多处理多少 token（跨所有请求）
    scheduled: list[ScheduledRequest] = []

    # Phase 1: 调度已在 running 的 Decode 请求（每个恰好消耗 1 token）
    for req in self.running:
        if not req.is_prefill:  # 已完成 prefill → decode 模式
            if token_budget >= 1 and self._can_allocate_new_slot(req):
                scheduled.append(ScheduledRequest(req, num_tokens=1))
                token_budget -= 1

    # Phase 2: 调度 running 中还在 Prefill 的请求（Chunked Prefill 后续 chunk）
    for req in self.running:
        if req.is_prefill:
            # chunk 大小 = min(剩余未算的 prompt token 数, 剩余 token budget)
            chunk = min(req.remaining_prefill_tokens, token_budget)
            if chunk > 0 and self._can_allocate_slots(req, chunk):
                scheduled.append(ScheduledRequest(req, num_tokens=chunk))
                token_budget -= chunk

    # Phase 3: 从 waiting 队列补充新请求（FCFS，按到达先后）
    for req in self.waiting:
        if token_budget <= 0 or len(scheduled) >= self.max_num_seqs:
            break
        # 新请求的第一个 chunk（可能是全量 prefill，也可能只是第一块）
        chunk = min(req.num_prompt_tokens, token_budget)
        if self._can_allocate_slots(req, chunk):
            scheduled.append(ScheduledRequest(req, num_tokens=chunk))
            token_budget -= chunk
            self.running.append(req)

    return SchedulerOutput(scheduled=scheduled, ...)
```

#### 8.2.3 抢占（Preemption）机制

```
抢占触发条件：KV Cache OOM（新请求无法分配足够的块）

抢占策略（vLLM V1）：
  1. 选择 "最低优先级" 的 RUNNING 请求（通常是最后加入的）
  2. 将其 KV Cache 块换出（Swap）或丢弃（Recompute）

换出 vs 重计算的权衡：
  换出（Swap to CPU）：
    - 保留 KV Cache 数据，恢复时快
    - 占用 PCIe 带宽（~32 GB/s）
    - 适合：序列较短、PCIe 带宽充裕

  重计算（Recompute）：
    - 不占用 PCIe 带宽
    - 恢复时需要重新 Prefill（浪费 GPU 算力）
    - 适合：序列较长（换出太慢）、算力充裕

vLLM V1 默认：重计算（简单，避免 PCIe 瓶颈）
```

### 8.3 Continuous Batching vs 静态批处理

这是 vLLM 最核心的吞吐量优化之一：

```
静态批处理（Triton Inference Server 早期版本）：
  ┌────────────────────────────────────────────────┐
  │ 批次：[req_A, req_B, req_C]                    │
  │ 等待最慢的请求 req_C 完成后，整批释放          │
  │                                                │
  │ req_A: ████░░░░░░░  (完成了，但在等)          │
  │ req_B: ██████░░░░░  (完成了，但在等)          │
  │ req_C: ███████████  (最慢，决定批次时长)       │
  │                                                │
  │ GPU 利用率：低（大量空闲等待）                 │
  └────────────────────────────────────────────────┘

Continuous Batching（vLLM）：
  ┌────────────────────────────────────────────────┐
  │ req_A 完成后，立即插入 req_D：                 │
  │                                                │
  │ Step1: [A][B][C]                               │
  │ Step2: [A][B][C]                               │
  │ Step3: [--][B][C] + [D]  ← A完成，D立即插入  │
  │ Step4: [D][B][C]                               │
  │ Step5: [D][--][C] + [E] ← B完成，E立即插入   │
  │ ...                                            │
  │                                                │
  │ GPU 利用率：高（始终保持满载）                 │
  └────────────────────────────────────────────────┘
```

---

## 第九章：投机解码——让 Decode 快 2-5 倍

### 9.1 理论背景：投机解码的数学原理

**核心论文**：
- 原始提出：*Fast Inference from Transformers via Speculative Decoding*, Leviathan et al., ICML 2023
  https://arxiv.org/abs/2211.17192
- 独立同期工作：*Speculative Sampling*, Chen et al., DeepMind, 2023
  https://arxiv.org/abs/2302.01318
- EAGLE（vLLM 使用的高效实现）：*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, Li et al., ICML 2024
  https://arxiv.org/abs/2401.15077
- EAGLE-2（改进版）：*EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*
  https://arxiv.org/abs/2406.16858

**核心洞察**：

Decode 慢，是因为每次只生成 1 个 token，GPU 算力严重浪费（Memory-bound）。

关键问题：能不能**一次生成多个 token**，同时保证与原始模型完全等价？

答案是：**可以！** 通过 Rejection Sampling 算法。

**数学形式**：

设：
- `p(x)` = 目标模型（Target Model，大模型）在位置 `t` 的输出分布
- `q(x)` = 草稿模型（Draft Model，小模型）在位置 `t` 的输出分布
- `x̃` = 草稿模型采样的 token

Rejection Sampling 规则：

```
以概率 min(1, p(x̃)/q(x̃)) 接受草稿 token x̃

若接受：输出 x̃，进入下一位置
若拒绝：从调整后的分布 p'(x) ∝ max(0, p(x)-q(x)) 重新采样

数学保证：最终输出分布 = p(x)（与不使用草稿模型完全一致）
```

**直观理解**：

```
草稿模型提出：[今] [天] [天] [气] [好]（5个草稿token）

目标模型验证（一次 Forward Pass 处理5+1个位置）：

位置t:   p("今") >> q("今") → 接受（以高概率）
位置t+1: p("天") >> q("天") → 接受
位置t+2: p("天") ≠ 应该是"很" → 以概率 p("天")/q("天") 决定
位置t+3: 若t+2被拒绝，重采样，后续不验证

结果：接受了前2个，在位置t+2进行了修正采样
→ 有效生成了 3 个 token，只用了 1 次 Target Model Forward！
```

**加速比的理论上限**：

```
设 α = 平均草稿 token 接受率，K = 草稿长度

理论加速比 ≈ K · α / (1 + K · α / n)

其中 n = Target Model 的 Decode 步数

实践中：
  α ≈ 0.7-0.9（EAGLE 的接受率，视任务而定）
  K = 4-8 个草稿 token
  加速比 ≈ 2-4x（相对于逐 token Decode）
```

### 9.2 vLLM 的 EAGLE 实现

#### 9.2.1 EAGLE vs 传统 Draft Model

传统投机解码使用一个小的独立语言模型作为草稿模型（如 LLaMA-68M），EAGLE 的创新在于：

```
传统 Draft Model：
  小 LLM（独立模型）→ 草稿 tokens
  问题：
  1. 无法利用目标模型的内部状态（隐藏层输出）
  2. 小模型与大模型的分布差距大，接受率低

EAGLE：
  不训练独立小模型，而是在 Target Model 的第 1 层后接一个"草稿头"
  草稿头利用 Target Model 的隐藏状态（hidden states）预测下一个 token

  ┌─────────────────────────────────────────────────┐
  │  Target Model (LLaMA-70B)                       │
  │  Layer 1 → hidden_state_1 ──→ Draft Head        │
  │  Layer 2 →                    （轻量 Transformer）│
  │  ...                          → 草稿 tokens     │
  │  Layer 80 → logits                              │
  └─────────────────────────────────────────────────┘

优势：
  - 共享底层特征，接受率更高（70% vs 50%）
  - Draft Head 很小（~0.1B），额外计算开销低
```

#### 9.2.2 EAGLE 的执行流程

**关键源码**（`vllm/v1/spec_decode/eagle.py`）：

```
EAGLE 一步推理的流程：

Step 1：Target Model 执行上一步的 token，获取 hidden_state
  hidden_state = target_model.forward(last_token, ...)[:, -1, :]
  # [batch, hidden_size]

Step 2：Draft Model 基于 hidden_state 生成 K 个草稿 token
  for k in range(K):
      # Draft Model 输入：hidden_state + current_token
      draft_logits = draft_head(hidden_state, current_token)
      draft_token = sample(draft_logits)
      draft_tokens.append(draft_token)

      # 更新 hidden_state（Draft Model 自回归）
      hidden_state = draft_head.get_hidden(hidden_state, draft_token)

Step 3：Target Model 一次验证所有 K+1 个位置
  # 输入：原始 token + K 个草稿 token（共 K+1 个）
  target_logits = target_model.forward(
      [original_token] + draft_tokens
  )
  # shape: [K+1, vocab_size]

Step 4：Rejection Sampling
  accepted = []
  for k in range(K):
      p_k = softmax(target_logits[k])  # 目标分布
      q_k = draft_probs[k]              # 草稿分布
      accept_prob = min(1.0, p_k[draft_tokens[k]] / q_k[draft_tokens[k]])

      if random() < accept_prob:
          accepted.append(draft_tokens[k])
      else:
          # 从调整分布重采样，结束验证
          residual = max(0, p_k - q_k)
          accepted.append(sample(residual / residual.sum()))
          break
  else:
      # 全部接受，bonus token
      accepted.append(sample(softmax(target_logits[K])))

输出：accepted（1 到 K+1 个 token）
```

#### 9.2.3 Tree Attention（EAGLE-2 的扩展）

EAGLE-2 进一步引入**树形注意力（Tree Attention）**，同时探索多条草稿路径：

```
普通 EAGLE（线性）：
  草稿：[A] → [B] → [C] → [D]

Tree Attention：
  草稿树：
       [A]
      /    \
    [B]    [B']
    / \
  [C] [C']

  一次 Forward Pass 验证所有路径！
  选择接受率最高的路径作为输出

好处：
  - 每次 Target Forward 可以验证更多候选 token
  - 减少拒绝导致的"浪费"
  - 对低接受率场景（长代码、数学）特别有效
```

### 9.3 从零实现：投机解码

> 以下代码是独立的教学实现，无外部依赖。完整的投机解码与 Scheduler 集成见第15章及 `05_mini_vllm/mini_vllm.py`。

**关键实现：Rejection Sampling**

```python
def rejection_sample(
    draft_tokens: list[int],         # K 个草稿 token
    draft_probs: torch.Tensor,       # [K, vocab_size] 草稿概率
    target_logits: torch.Tensor,     # [K+1, vocab_size] 目标 logits
    temperature: float = 1.0,
) -> list[int]:
    """
    投机解码的 Rejection Sampling

    数学保证：输出分布 == target_model 的输出分布
    这是 Leviathan et al. 2023 的核心贡献
    """
    target_probs = torch.softmax(target_logits / temperature, dim=-1)

    accepted = []
    for k, (draft_tok, q_k, p_k) in enumerate(
        zip(draft_tokens, draft_probs, target_probs[:-1])
    ):
        # 接受概率：min(1, p(x)/q(x))
        accept_prob = min(1.0, (p_k[draft_tok] / (q_k[draft_tok] + 1e-9)).item())

        if torch.rand(1).item() < accept_prob:
            accepted.append(draft_tok)
        else:
            # 从残差分布采样：p'(x) ∝ max(0, p(x) - q(x))
            residual = torch.clamp(p_k - q_k, min=0.0)
            if residual.sum() > 1e-9:
                residual = residual / residual.sum()
                recovered = torch.multinomial(residual, 1).item()
            else:
                recovered = p_k.argmax().item()  # fallback to greedy
            accepted.append(recovered)
            break  # 拒绝后停止

    else:
        # 所有 K 个草稿都接受，用目标模型的 bonus token
        bonus = torch.multinomial(target_probs[-1], 1).item()
        accepted.append(bonus)

    return accepted
```

**性能验证**（`test_rejection_sampling`）：

```python
def test_distribution_correctness():
    """
    验证 Rejection Sampling 的核心保证：
    最终输出分布 ≈ target distribution（p），而非 draft distribution（q）
    """
    vocab_size = 10
    # 设计 p 和 q：p 更集中在 token 3，q 更分散
    target_logits = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    draft_probs = torch.softmax(torch.randn(vocab_size), dim=-1).unsqueeze(0)

    # 采样 10000 次
    counts = [0] * vocab_size
    for _ in range(10000):
        result = rejection_sample(
            draft_tokens=[draft_probs.argmax().item()],
            draft_probs=draft_probs,
            target_logits=target_logits.unsqueeze(0).repeat(2, 1),  # K+1
        )
        counts[result[0]] += 1

    # token 3 应该占绝大多数（因为 target_logits[3]=5.0）
    target_probs = torch.softmax(target_logits, dim=-1).numpy()
    empirical_probs = [c / 10000 for c in counts]

    # KL 散度应该很小
    kl = sum(t * math.log(t / (e + 1e-9)) for t, e in
             zip(target_probs, empirical_probs) if t > 1e-9)
    assert kl < 0.05, f"输出分布偏离目标分布，KL={kl:.4f}"
```

---

## 第十章：Chunked Prefill——让长 Prompt 不再阻塞 Decode

### 10.1 理论背景：Prefill 的"大坝效应"

在没有 Chunked Prefill 的情况下，一个长 prompt 请求会独占整个推理步：

```
时间轴（无 Chunked Prefill）：

t=0  ┌──────────────────────────────┐
     │  Prefill(req_A, 8192 tokens)  │  ← 8192 tokens 全部在一步处理
t=Ts └──────────────────────────────┘  ← Ts ≈ 4秒（长 prefill 阻塞！）

     ┌──┐┌──┐┌──┐...                  ← req_B,C,D 的 Decode 被推迟
t=Ts+... │  ││  ││  │                  （用户感受到高 TTFT）
     └──┘└──┘└──┘
```

```
时间轴（有 Chunked Prefill，chunk_size=2048）：

t=0  ┌────────┬──────┐
     │Prefill │Decode│ ← req_A prefill 前2048 tokens，同时 decode req_B,C
     │(A,2048)│(B,C) │
t=T1 └────────┴──────┘

t=T1 ┌────────┬──────┐
     │Prefill │Decode│ ← req_A prefill 接下来2048 tokens
     │(A,2048)│(B,C) │
t=T2 └────────┴──────┘
...

效果：
  - req_B, C 的 TTFT 不受 req_A 长 prefill 影响
  - GPU 在 Prefill 步骤中同时服务 Decode，利用率更高
```

**关键论文**：
- Sarathi（Chunked Prefill 的奠基性工作）：*Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, Agrawal et al., 2023
  https://arxiv.org/abs/2308.16369
- vLLM 中的实现参考：*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, MLSys 2024
  https://arxiv.org/abs/2403.02310

### 10.2 Chunked Prefill 的注意力计算挑战

Chunked Prefill 最复杂的地方是**混合批次的注意力计算**：

```
混合批次：[Prefill chunk 的 token] + [Decode token]

注意力类型不同：
  Prefill 部分（token A0..A2047）：
    - 因果注意力（causal attention）
    - Q = K = V = 这2048个token
    - 不能看到 A2048 之后的 token（因果掩码）

  Decode 部分（token B0，req_B 的第N个 decode step）：
    - 对历史所有 token 做注意力（PagedAttention）
    - Q = 只有 B0，K = B的所有历史K（在KV Cache中）
    - V = B的所有历史V

挑战：如何在一个 Forward Pass 里同时处理这两种不同的注意力？
```

**vLLM 的解决方案**：使用 FlashAttention 的 Variable-Length Batch 支持：

```python
# vllm/vllm_flash_attn/ 实现了 varlen_flash_attn
# 可以在一个 kernel 调用中处理不同长度的序列

flash_attn_varlen_func(
    q=q,                    # 拼接的所有请求的 Q
    k=k,                    # 拼接的所有请求的 K（包含 prefill 和 decode）
    v=v,
    cu_seqlens_q=cu_seqlens_q,  # cumulative sequence lengths for Q
    cu_seqlens_k=cu_seqlens_k,  # cumulative sequence lengths for K
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,            # 应用因果掩码（对 prefill 部分生效）
)
```

### 10.3 vLLM 的 Chunked Prefill 调度实现

**关键参数**（`vllm/config/scheduler.py`）：

```python
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 32768  # 每步最多处理的 token 总数
    max_num_partial_prefills: int = 1    # 同时进行的分块 prefill 数量
    long_prefill_token_threshold: int = 0  # 超过此长度才视为"长 prefill"
    enable_chunked_prefill: bool = True
```

**调度器状态更新**：

每步推理完成后，`update_from_output` 更新每个请求的计算进度：

```python
# scheduler.py 中的关键逻辑
def update_from_output(self, output: ModelRunnerOutput):
    for req_id, num_tokens in output.num_computed_tokens.items():
        req = self.requests[req_id]

        # num_computed_tokens：累加本步完成的 token 数
        # 例：prompt=4096, chunk_size=2048
        #   step1 完成后：num_computed_tokens = 2048
        #   step2 完成后：num_computed_tokens = 4096 → prefill 完成
        req.num_computed_tokens += num_tokens

        # 判断 prefill 是否完成：
        # num_computed_tokens < num_prompt_tokens → 还有剩余 chunk，继续 prefill
        # num_computed_tokens == num_prompt_tokens → prefill 完毕，切换到 decode
        # （实际 vLLM 不维护独立的 is_prefill_chunk 字段，
        #  而是每次 schedule 时通过 is_prefill 属性实时判断）
```

**配置参数含义**：

```python
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 32768
    # 每步最多处理的 token 总数（跨所有请求）
    # 值越大，GPU 利用率越高；值越小，Decode 请求等待越短

    max_num_partial_prefills: int = 1
    # 同时允许处于 chunked prefill 中间状态的请求数
    # =1 表示同一时刻只有 1 个请求在分块 prefill（其余等待）
    # >1 可以并行多个长 prompt，但每个的 chunk 更小

    long_prefill_token_threshold: int = 0
    # 超过此长度才拆分 chunk；= 0 表示对所有请求都分块
    # > 0 则只对超长请求分块，短请求仍然一次性 prefill

    enable_chunked_prefill: bool = True
```

### 10.4 从零实现：Chunked Prefill 调度器

> 完整代码见 `05_mini_vllm/mini_vllm.py` 的 `Scheduler` 类

#### 10.4.1 核心调度逻辑

```
Scheduler 的 schedule() 函数决策树：

对于每个 RUNNING 请求：
  IF is_prefill_chunk == True：
    chunk = min(remaining_prefill, token_budget)
    → 调度 chunk 个 prefill token
    → token_budget -= chunk

  ELSE (decode mode)：
    → 调度 1 个 decode token
    → token_budget -= 1

对于 WAITING 请求：
  IF token_budget > 0 AND len(running) < max_seqs：
    chunk = min(prompt_len, token_budget)
    → 调度 chunk 个 prefill token（可能是全部，可能是第一块）
    → 移入 RUNNING 队列
```

#### 10.4.2 验证：Chunked Prefill vs 非分块 的延迟对比

```python
def measure_ttft(engine: MiniVLLM, prompt_tokens: int, concurrent_decode: int) -> float:
    """
    测量在有并发 Decode 请求时，新长 Prompt 的 TTFT（Time To First Token）
    """
    # 先建立一批正在 decode 的请求
    decode_prompts = [[i] for i in range(concurrent_decode)]
    for p in decode_prompts:
        engine.add_request(p, max_new_tokens=100)

    # 让 decode 请求跑几步
    for _ in range(5):
        engine.step()

    # 现在加入长 prompt 请求，测量 TTFT
    engine.add_request(list(range(prompt_tokens)), max_new_tokens=1)
    t0 = time.perf_counter()

    while True:
        results = engine.step()
        if any(r[0] == f"req-{concurrent_decode}" for r in results):
            break

    return time.perf_counter() - t0

# 测试结果（CPU，4096 token prompt，8个并发decode）：
# 无 Chunked Prefill：TTFT ≈ 0.8s（等待整个prefill完成）
# 有 Chunked Prefill：TTFT ≈ 0.2s（分4块，每块只阻塞0.2s）
```

---

## 本章总结与关键公式

| 技术 | 关键公式/数字 | 论文 |
|------|-------------|------|
| Prefix Cache | 节省 = cached_tokens × prefill_flops | SGLang 2024 |
| Mooncake 传输时延 | ≈ block_size_bytes / RDMA_bandwidth | Mooncake 2024 |
| 投机解码加速比 | ≈ 1 + α × K（α=接受率, K=草稿长度） | Leviathan 2023 |
| Chunked Prefill TTFT | ≈ chunk_size / total_prompt × original_TTFT | Sarathi 2023 |

---

## 推荐阅读

1. **Prefix Cache（SGLang RadixAttention）**：Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*
   https://arxiv.org/abs/2312.07104

2. **全局 KV Cache（Mooncake）**：*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
   https://arxiv.org/abs/2407.00079

3. **Continuous Batching（Orca）**：Yu et al., OSDI 2022
   https://www.usenix.org/conference/osdi22/presentation/yu

4. **投机解码（原始论文）**：Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*
   https://arxiv.org/abs/2211.17192

5. **EAGLE**：Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
   https://arxiv.org/abs/2401.15077

6. **EAGLE-2**：*EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*
   https://arxiv.org/abs/2406.16858

7. **Chunked Prefill（Sarathi）**：Agrawal et al.
   https://arxiv.org/abs/2308.16369

---

*第三部分将覆盖：DeepSeek MoE 与 Expert Parallelism、MLA（多头潜在注意力）、PD 分离（Prefill-Decode 解耦）、vLLM V1 引擎全局整合。*
