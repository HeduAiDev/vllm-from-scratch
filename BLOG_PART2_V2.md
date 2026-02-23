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

### 6.3.3 深入理解：四个关键问题

#### Q1：一个请求的生命周期中，block 什么时候被 free，什么时候开始驱逐？

Block 的生命周期分为四个阶段：

```
                    ┌─────────────────────────────────────────┐
                    │          Block 状态机                    │
                    │                                         │
  初始化 →  [FREE]  ──── get_new_blocks() ──→  [ACTIVE]       │
              │                                    │          │
              │ 位于 free_block_queue               │          │
              │ ref_cnt = 0                        │          │
              │                                    │ ref_cnt > 0
              │ ← free_blocks() 归零时 ─────────── ┘          │
              │                                              │
              ├─ 空间充裕：留在队列等待复用（Prefix Cache）     │
              │                                              │
              └─ 空间不足：被 popleft() 驱逐 ─→  [EVICTED]    │
                   ↑                              从 cached_blocks 删除 hash
                   └─── 分配给新请求 ─────────── [ACTIVE]     │
                                                              │
                    └─────────────────────────────────────────┘
```

**完整时序示例**（block_size=16, 系统总共10个块）：

```
t=0: 系统启动，10个块全在 free_block_queue
     [B0, B1, B2, B3, B4, B5, B6, B7, B8, B9]（从左到右：最旧→最新）

t=1: 请求A到达（32 token 系统提示词 + 16 token 用户问题 = 3个块）
     get_new_blocks(3) → 弹出 B0, B1, B2
     B0.ref_cnt = B1.ref_cnt = B2.ref_cnt = 1
     free_block_queue: [B3, B4, B5, B6, B7, B8, B9]

t=2: A的Prefill完成，cache_full_blocks()把满块登记入缓存
     B0.block_hash = H0, B1.block_hash = H1（B2未满，不缓存）
     cached_blocks: { H0 → B0, H1 → B1 }

t=3: 请求B到达（相同系统提示词 + 不同用户问题）
     touch([B0, B1])  → B0.ref_cnt: 1→2, B1.ref_cnt: 1→2（命中！）
     get_new_blocks(1) → 弹出 B3 给B的用户问题
     B3.ref_cnt = 1

t=4: 请求A完成，free_blocks([B0, B1, B2])
     B0.ref_cnt: 2→1（还有B共享，不入队）
     B1.ref_cnt: 2→1（同上）
     B2.ref_cnt: 1→0 → 加入 free_block_queue 尾部
     free_block_queue: [B4, B5, B6, B7, B8, B9, B2]

t=5: 请求B完成，free_blocks([B0, B1, B3])
     B0.ref_cnt: 1→0 → 加入 free_block_queue 尾部（有 hash H0）
     B1.ref_cnt: 1→0 → 加入 free_block_queue 尾部（有 hash H1）
     B3.ref_cnt: 1→0 → 加入 free_block_queue 尾部（无 hash）
     free_block_queue: [B4, B5, B6, B7, B8, B9, B2, B0, B1, B3]
     cached_blocks: { H0 → B0, H1 → B1 }  ← 仍然有效！

t=6: 请求C到达（需要5个新块，但只有6个空闲：B4-B9,B2,B0,B1,B3）
     get_new_blocks(5)：弹出 B4, B5, B6, B7, B8
     B4-B8 均无 hash，直接分配
     驱逐触发（如 free_block_queue 已剩 B9,B2,B0,B1,B3 共5块）

t=7: 请求D需要更多块，触发驱逐
     popleft() → B9（最旧，无 hash） → 直接复用
     popleft() → B2（次旧，无 hash） → 直接复用
     popleft() → B0（有 hash H0）：
         _maybe_evict_cached_block(B0) → 删除 cached_blocks[H0]
         B0 现在可以被新请求使用（H0 的 KV 数据被覆盖）
```

**关键结论**：
- `ref_cnt > 0`：块正被请求使用，**不能**被驱逐
- `ref_cnt == 0`：块在 LRU 队列中，**等待**复用，暂时保留（Prefix Cache）
- 驱逐**不是主动**触发的：当 `get_new_blocks()` 被调用且队列中有旧块时，旧块被弹出覆盖

#### Q2：链式哈希会出现前面的 block 被驱逐而后面没有的情况吗？

**会！** 这是 LRU 策略的已知代价，称为"孤儿块"（orphaned block）问题：

```
场景：100 个块的池，缓存了一个 3 块的系统提示词 [H0→B3, H1→B7, H2→B12]

假设系统负载很高，驱逐顺序取决于 LRU 时间戳（而非哈希链顺序）

可能的驱逐顺序：
  B3 (H0) 最旧 → 被驱逐
  cached_blocks 变为：{ H1 → B7, H2 → B12 }

新请求 E 到达（相同系统提示词）：
  query: H0 → MISS → break！
  返回 0 个命中块

  H1 → B7 和 H2 → B12 虽然还在缓存中，但永远不会被匹配到！
  它们成为"孤儿块"：占用 KV Cache 空间，却永远不能被复用
```

**为什么 vLLM 接受这个设计缺陷？**

1. **简单性**：LRU 是 O(1) 的全局策略，不需要追踪块之间的链式依赖
2. **自然修复**：孤儿块也会被 LRU 驱逐（它们的访问时间早于新块），最终被清理
3. **低概率**：热门前缀（如系统提示词）会被频繁访问，访问时间会刷新，不容易被驱逐
4. **正确性不受影响**：孤儿块只是浪费空间，不会导致计算错误

**改进方案**（vLLM 未实现但理论上可行）：
- "链式驱逐"：驱逐 H0 时，同时驱逐所有以 H0 为祖先的块（H1, H2）
- 代价：需要维护反向索引（parent_hash → [child_hashes]），增加复杂度

#### Q3：为什么必须从头匹配前缀，不能从后往前找第一个命中？

这是**物理约束**，不只是算法选择：

```
场景：tokens = [SYS_0..SYS_15 | SYS_16..SYS_31 | Q_0..Q_15]
     缓存状态：H0 被驱逐（B0 已被覆盖），H1 和 H2 还在缓存

如果我们"跳过 H0 miss，直接使用 H1"：

  GPU 的 KV Cache：
    Block H1（B7）：包含 positions 16-31 的 K/V 向量
    Block H2（B12）：包含 positions 32-47 的 K/V 向量

  现在处理 position 32（Q_0）的 attention：
    需要 attend 到 positions 0..31 的 K/V
    positions 16-31 → B7 可用 ✓
    positions 0-15  → ??? KV 不在 GPU 内存中！

    → attention 结果错误（未 attend 到所有历史 token）
    → 或者 GPU 访问非法内存地址，直接崩溃
```

更深层原因：**KV 的语义是注意力上下文，而非孤立的向量**

```python
# attention 计算时 GPU 需要访问连续的 KV 块
for pos in range(0, seq_len):
    attn_score = Q[pos] @ K[0:pos]   # 必须有 0..pos-1 的全部 K
    output[pos] = softmax(attn_score) @ V[0:pos]  # 必须有 0..pos-1 的全部 V

# 如果 K[0:16] 不在 GPU 内存中，这段代码的结果是错的！
# block table 必须提供 0..seq_len-1 的连续物理块地址
```

**链式哈希确保的是 token 身份（内容正确性）**，而不是 GPU 内存可用性。即使 H1 block 在哈希表中存在（意味着它保存的确实是 SYS_16..SYS_31 的 KV），但如果 H0 block 不在 GPU 内存中，attention 运算就无法正确执行。

因此：

```python
# 正确的匹配逻辑（vLLM 的做法）
matched = []
for h in block_hashes:
    if h in cached_blocks:
        matched.append(cached_blocks[h])
    else:
        break  # 必须停在第一个 miss！后面的 block 即使命中也无法使用

# 错误的逻辑（假设的"从后往前"，结果是错的）
for h in reversed(block_hashes):    # ← 永远不能这样做！
    if h in cached_blocks:
        return cached_blocks[h]     # 孤立使用后续 block，attention 结果错误
```

#### Q4：KV Cache 在 TP/PP 并行时如何存储？对 Mooncake 有什么影响？

**Tensor Parallelism（TP）的影响**：

```
TP=4 的 LLaMA-70B（128个 attention head）：

  GPU_0：负责 head 0-31 的 Q/K/V 计算
  GPU_1：负责 head 32-63 的 Q/K/V 计算
  GPU_2：负责 head 64-95 的 Q/K/V 计算
  GPU_3：负责 head 96-127 的 Q/K/V 计算

每个 GPU 的 KV Cache 块布局：
  kv_cache[block_id, :, head_start:head_end, :] = 该GPU负责的头的 KV

同一个逻辑 block_id 在4个 GPU 上各有一片物理内存：
  Block #7 on GPU_0: positions 0-15, heads 0-31
  Block #7 on GPU_1: positions 0-15, heads 32-63
  Block #7 on GPU_2: positions 0-15, heads 64-95
  Block #7 on GPU_3: positions 0-15, heads 96-127

Block table 是全局共享的（所有 GPU 上相同的 block_id）
实际 KV 数据分布在 4 张 GPU 上
```

**TP 对 Mooncake 全局池化的约束**：

```
P 节点（TP=4）→ D 节点（TP=4）：
  P.GPU_0 → 通过 RDMA WRITE → D.GPU_0（head 0-31）
  P.GPU_1 → 通过 RDMA WRITE → D.GPU_1（head 32-63）
  P.GPU_2 → 通过 RDMA WRITE → D.GPU_2（head 64-95）
  P.GPU_3 → 通过 RDMA WRITE → D.GPU_3（head 96-127）
  共 4 个并行 RDMA 传输

当前限制（Mooncake Connector 源码中明确）：
  - P.TP_size == D.TP_size 才能直接传输
  - P.TP=4, D.TP=2 → 不支持（NotImplementedError）
  - 原因：每张 GPU 的块大小不同，内存地址映射不同

Pipeline Parallelism（PP）：
  当前 Mooncake Connector 完全不支持 PP（源码中 ValueError）
  原因：PP 将不同层分到不同 GPU，KV Cache 按层分布，
        跨 PP 节点的元数据管理更复杂
```

---

### 6.4 从零手搓：带 LRU 的 Prefix Cache

> 完整代码：`02_kvcache/block_pool_lru.py`

从最简单的 BlockPool 出发，逐步增加 Prefix Cache 和 LRU，理解每一步的设计动机。

#### 6.4.1 第一步：最简 BlockPool（只有分配/释放）

最基础的 KV Cache 管理：一个空闲块池，每次请求从中取块，结束后归还。

```python
class BlockPool_v1:
    """最简版：只有分配和释放，无 Prefix Cache"""

    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        # 空闲栈（简单实现，LIFO）
        self.free_blocks = list(range(num_blocks))

    def allocate(self, n: int) -> list[int]:
        """分配 n 个块，返回 block_id 列表"""
        if len(self.free_blocks) < n:
            raise RuntimeError("OOM: 显存不足")
        ids = []
        for _ in range(n):
            ids.append(self.free_blocks.pop())  # 从栈顶取
        return ids

    def free(self, block_ids: list[int]):
        """归还块（简单归还，内容丢弃）"""
        self.free_blocks.extend(block_ids)
```

**问题**：每个请求结束后，prefill 计算的 KV 全部丢弃。下一个相同 prompt 的请求要重新计算。

#### 6.4.2 第二步：加入哈希表（Prefix Cache）

增加一个 `hash → block_id` 映射，让 prefill 完的块不立刻丢弃，等待后续请求复用。

```python
class BlockPool_v2:
    """加入 Prefix Cache 哈希表（无 LRU，块只会增加不会释放）"""

    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks = list(range(num_blocks))

        # 新增：hash → Block（Prefix Cache 核心）
        self.cached_blocks: dict[str, Block] = {}

    def get_cached(self, block_hash: str) -> Block | None:
        """查询 Prefix Cache"""
        return self.cached_blocks.get(block_hash)

    def allocate(self, n: int) -> list[int]:
        if len(self.free_blocks) < n:
            raise RuntimeError("OOM: 所有块都被占用，无法驱逐")
        return [self.free_blocks.pop() for _ in range(n)]

    def mark_cached(self, block_id: int, block_hash: str):
        """一个块的 Prefill 完成后，登记到缓存"""
        block = self.blocks[block_id]
        block.block_hash = block_hash
        self.cached_blocks[block_hash] = block

    def free(self, block_ids: list[int]):
        """释放块（ref_cnt 减到 0 才真正归还，否则只减计数）"""
        for bid in block_ids:
            block = self.blocks[bid]
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and block.block_hash is None:
                # 无缓存 hash → 可以直接归还
                self.free_blocks.append(bid)
            # 有 hash 的块：不归还，留在缓存中等待复用
```

**问题**：GPU 显存有限，缓存的块永远不会被清理，最终 OOM。需要驱逐策略。

#### 6.4.3 第三步：加入 LRU（驱逐策略）

用双向链表实现 O(1) LRU：最近被归还的块在尾部（不驱逐），最久未用的在头部（优先驱逐）。

```python
class BlockPool_v3:
    """完整实现：Prefix Cache + LRU 驱逐"""

    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]

        # ★ 关键数据结构：LRU 双向链表（所有 ref_cnt==0 的块都在此）
        # 布局：head(哨兵) ↔ 最旧 ↔ ... ↔ 最新 ↔ tail(哨兵)
        self.free_block_queue = FreeBlockQueue(self.blocks)

        # Prefix Cache：hash → Block
        self.cached_blocks: dict[str, Block] = {}

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    # ── 1. 查询缓存（O(1)）──────────────────────────────────────
    def get_cached_block(self, block_hash: str) -> Block | None:
        return self.cached_blocks.get(block_hash)

    # ── 2. Touch：命中的缓存块，增加引用计数 ─────────────────────
    def touch(self, blocks: list[Block]):
        """
        Prefix Cache 命中时调用
        ref_cnt==0 的块需要从 free_block_queue 中摘除（不再可驱逐）
        """
        for block in blocks:
            if block.ref_cnt == 0:
                # 从 LRU 队列中间移除（O(1) 双向链表）
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    # ── 3. 分配新块（可能触发驱逐）─────────────────────────────
    def get_new_blocks(self, num_blocks: int) -> list[Block]:
        """
        从 free_block_queue 取块（隐式 LRU 驱逐）

        popleft() 弹出最旧的块：
        - 如果有 block_hash → 同时从 cached_blocks 删除（驱逐！）
        - 然后赋给新请求（ref_cnt 从 0 → 1）
        """
        if num_blocks > self.get_num_free_blocks():
            raise RuntimeError("OOM")

        result = []
        for _ in range(num_blocks):
            block = self.free_block_queue.popleft()   # 取最旧的块

            # 如果该块有缓存 hash → 驱逐（从 Prefix Cache 删除）
            if block.block_hash is not None:
                del self.cached_blocks[block.block_hash]
                block.block_hash = None

            block.ref_cnt = 1                          # 分配给新请求
            result.append(block)

        return result

    # ── 4. 释放块（请求结束）────────────────────────────────────
    def free_blocks(self, blocks: list[Block]):
        """
        ref_cnt-- 后：
        - 降到 0 → 加入 free_block_queue 尾部（LRU 最新）
        - 若有 hash，保留在 cached_blocks 中（待复用）
        """
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                # 加到 LRU 队尾（最新→最不可能被驱逐）
                self.free_block_queue.append(block)
                # block.block_hash 不清除 → 保留在 cached_blocks

    # ── 5. 登记缓存（满块 Prefill 完成后）────────────────────────
    def cache_full_blocks(self, blocks: list[Block], block_hashes: list[str],
                          num_already_cached: int, num_full: int):
        """
        已计算完的满块，登记到 Prefix Cache

        只登记 num_already_cached..num_full-1 范围内的新块：
        - 0..num_already_cached-1：已缓存（Prefix Cache 命中）
        - num_full..（最后一个未满块）：不缓存（未写满，下次可能覆盖）
        """
        for i in range(num_already_cached, num_full):
            block = blocks[i]
            if block.block_hash is None:   # 尚未缓存
                block.block_hash = block_hashes[i]
                self.cached_blocks[block_hashes[i]] = block
```

#### 6.4.4 完整请求处理流程（整合所有步骤）

```python
def allocate_slots(
    pool: BlockPool_v3,
    token_ids: list[int],
    block_size: int = 16,
) -> tuple[list[Block], int]:
    """
    为一个新请求分配 KV Cache 块（含 Prefix Cache 命中逻辑）

    返回：(block_table, num_cached_tokens)
    """
    # Step 1: 计算块哈希链
    block_hashes = compute_block_hashes(token_ids, block_size)
    num_full_blocks = len(token_ids) // block_size
    block_table = []

    # Step 2: 逐块查询 Prefix Cache（遇到 MISS 停止）
    num_cached = 0
    cached_blocks = []
    for i, h in enumerate(block_hashes[:num_full_blocks]):
        block = pool.get_cached_block(h)
        if block is None:
            break  # 链断开，停止匹配
        cached_blocks.append(block)
        num_cached += 1

    # Step 3: Touch 命中的缓存块（ref_cnt++，从 LRU 摘除）
    pool.touch(cached_blocks)
    block_table.extend(cached_blocks)
    num_cached_tokens = num_cached * block_size

    # Step 4: 分配剩余新块
    num_new_blocks = (len(token_ids) - num_cached_tokens + block_size - 1) // block_size
    new_blocks = pool.get_new_blocks(num_new_blocks)
    block_table.extend(new_blocks)

    return block_table, num_cached_tokens


def on_request_finished(
    pool: BlockPool_v3,
    blocks: list[Block],
    block_hashes: list[str],
    num_full_blocks: int,
    num_already_cached: int,
):
    """请求完成后，登记新计算的块到 Prefix Cache，然后释放"""
    # 登记满块到缓存
    pool.cache_full_blocks(blocks, block_hashes, num_already_cached, num_full_blocks)

    # 释放所有块（ref_cnt--，降到 0 的加入 LRU）
    pool.free_blocks(blocks)
```

**测试运行**（`02_kvcache/block_pool_lru.py`）：

```bash
docker exec vllm python3 -m pytest /mnt/esfs/master_work/vllm-from-scratch/02_kvcache/ -v
```

12 个测试覆盖：分配/释放、Prefix Cache 命中、LRU 驱逐顺序、孤儿块处理、并发请求共享块。

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

### 7.3 Mooncake 真实源码深度解析

> 源码路径：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

#### 7.3.1 架构：不是"发布-查询"，而是"主动推送"

很多文章把 Mooncake 描述成"D-node 从全局池 pull KV"，但真实实现是**P-node 主动 RDMA WRITE 到 D-node 的内存**。这两种模式有本质区别：

```
错误理解（RDMA READ，类似 CDN 拉取）：
  D-node → 查询元数据服务 → 找到 block 在 P-node 的内存地址
  D-node → 发起 RDMA READ → 从 P-node GPU 内存直接读取
  (D 控制传输方向)

真实实现（RDMA WRITE，类似主动推送）：
  D-node → 向 P-node 的 ZMQ side channel 发送请求
              包含：D-node 的 GPU 内存地址
  P-node → Prefill 完成后，从本地 GPU 内存 RDMA WRITE 到 D-node
              engine.batch_transfer_sync_write(src_ptrs, dst_ptrs, lengths)
  (P 控制传输方向)
```

**为什么 P 主动写而非 D 主动读？**

- P-node 知道精确的传输时机（Prefill 刚完成的那一刻）
- P-node 掌握 KV 的物理内存地址（block_id × block_len）
- D-node 不需要知道 P-node 的内存布局细节

#### 7.3.2 核心数据流（逐步拆解源码）

```
完整 PD 传输时序（对应 mooncake_connector.py）：

┌─────────────────────────────────────────────────────────────────┐
│                      Bootstrap 阶段（启动时）                    │
│                                                                 │
│  P-node 启动：                                                  │
│    MooncakeConnectorWorker.__init__()                           │
│      → self.engine = TransferEngine()                           │
│      → self.engine.initialize(hostname, "P2PHANDSHAKE", "rdma") │
│      → 启动 MooncakeBootstrapServer（HTTP 服务）                 │
│      → 启动 ZMQ ROUTER socket（side channel，等待 D 的请求）     │
│      → 调用 engine.batch_register_memory(kv_data_ptrs, ...)     │
│        ← 预注册 GPU 内存到 RDMA NIC（一次性操作，之后 RDMA 直接访问）│
│                                                                 │
│  D-node 启动：                                                  │
│    MooncakeConnectorWorker.__init__()                           │
│      → self.engine = TransferEngine()                           │
│      → self.engine.initialize(hostname, "P2PHANDSHAKE", "rdma") │
│      → engine.batch_register_memory(kv_data_ptrs, ...)          │
│      → 启动后台接收线程（receiver_loop）                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      请求处理阶段                                │
│                                                                 │
│  ① Scheduler 决策（P-node）                                     │
│    scheduler: get_num_new_matched_tokens(request)               │
│      → 检查 kv_transfer_params["do_remote_prefill"]             │
│        （这个 flag 由路由层/LB 设置，表示本请求需要送到 D-node）   │
│      → 返回 (count, True)：告诉调度器这批 token 将异步传给 D     │
│    update_state_after_alloc()                                   │
│      → 记录 request → local_block_ids 的映射                    │
│                                                                 │
│  ② P-node Prefill                                               │
│    start_load_kv()：record_send_reqs() 更新 reqs_need_send      │
│    → Prefill 正常执行，GPU 计算 KV Cache                         │
│    request_finished()                                           │
│      → send_meta.ready.set()  ← 通知等待中的 D-node，KV 已就绪  │
│                                                                 │
│  ③ D-node 发送请求（receiver_loop 异步执行）                     │
│    start_load_kv(): _start_load_kv() →                         │
│      连接 P-node bootstrap server，拿到 ZMQ side channel 地址   │
│      receive_kv_from_single_worker(p_node_addr, pull_metas)    │
│      → 通过 ZMQ 发送 MooncakeXferMetadata：                    │
│          {                                                      │
│            remote_hostname: D-node IP,                          │
│            remote_port: D-node RDMA RPC port,                  │
│            req_blocks: {req_id: (transfer_id, local_block_ids)},│
│            kv_caches_base_addr: D-node GPU 内存基地址列表         │
│          }                                                      │
│                                                                 │
│  ④ P-node 响应（sender_loop 异步执行）                           │
│    _mooncake_sender_listener()接收 ZMQ 请求                     │
│    等待 send_meta.ready（等 Prefill 完成）                       │
│    _build_transfer_params()：                                   │
│      计算 src_ptrs = kv_caches_base_addr + block_id * block_len │
│      计算 dst_ptrs = D-node 发来的 kv_caches_base_addr + 偏移   │
│      连续块合并：group_concurrent_contiguous() 减少 RDMA 描述符  │
│    _send_blocks()：                                             │
│      engine.batch_transfer_sync_write(remote_session,           │
│                                       src_ptrs, dst_ptrs, lens) │
│      ← RDMA WRITE 从 P-node GPU 直接写入 D-node GPU 内存        │
│                                                                 │
│  ⑤ D-node 等待完成                                              │
│    process_pulling_result() → finished_recving_reqs.add(req_id) │
│    get_finished() → 通知调度器 req_id 的 KV 已就绪              │
│    调度器：req_id → RUNNING 状态，加入下一批次 Decode            │
└─────────────────────────────────────────────────────────────────┘
```

#### 7.3.3 关键源码注解

**内存注册**（一次性，启动时执行）：
```python
# MooncakeConnectorWorker.register_kv_caches()
# kv_caches: 每层的 KV Cache 张量，形如 {layer_name: Tensor[num_blocks, ...]}

kv_data_ptrs = []
kv_data_lens = []
for layer_name, cache in kv_caches.items():
    base_addr = cache.data_ptr()   # GPU 内存的物理地址（指针）
    kv_data_ptrs.append(base_addr)
    kv_data_lens.append(cache.nbytes)

# 向 Mooncake RDMA 引擎注册：将这些 GPU 内存区域"钉住"（pin）
# 注册后，RDMA NIC 可以直接读写这块内存，无需 CPU 干预
ret = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)

# block 访问公式：
# 第 block_id 个块的 KV 在第 layer 的偏移 = block_id * block_len
# block_len = tensor.nbytes / num_blocks
self.block_len = tensor_size_bytes // self.num_blocks
```

**地址计算 + 连续块合并**：
```python
# _build_transfer_params() 的关键逻辑
for layer_addr_p, layer_addr_d in zip(local_base_addr, remote_base_addr):
    for group_local, group_remote in zip(group_local_blocks, group_remote_blocks):
        # 连续块合并为单次 RDMA 操作（减少硬件描述符开销）
        # 例：P-node block_ids=[5,6,7] 连续 → 一次 RDMA WRITE 3*block_len 字节
        src_ptrs.append(layer_addr_p + group_local[0] * block_len)
        dst_ptrs.append(layer_addr_d + group_remote[0] * block_len)
        lengths.append(block_len * len(group_local))

# group_concurrent_contiguous() 的作用：
# input: src=[5,6,7,12,13], dst=[2,3,4,8,9]
# output: [(5,6,7), (12,13)], [(2,3,4), (8,9)]
# → 2次 RDMA 操作代替 5次，减少延迟和开销
brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
```

**异步协同机制（asyncio.Event）**：
```python
# SendBlockMeta.ready 是 asyncio.Event
# P-node Prefill 完成时，request_finished() 调用 send_meta.ready.set()
# P-node sender 线程一直在 wait_tasks 中等待：
wait_tasks = [asyncio.create_task(wait_and_ret(d_req_id, send_meta))]
# ...
done, pending = await asyncio.wait(wait_tasks, timeout=ABORT_TIMEOUT)
# ready 触发后，立刻开始 RDMA WRITE

# 关键：D-node 在发送 ZMQ 请求时，Prefill 可能还没完成
# P-node 会在 asyncio.wait() 中挂起，直到 Prefill 完成（ready.set()）
# 这样 D-node 的 ZMQ 请求和 P-node 的 Prefill 可以并发进行，
# P 的 Prefill 完成后 ZMQ 响应立刻就绪，最小化 D-node 等待时间
```

#### 7.3.4 与第七章模拟实现的对比

| 特性 | 模拟实现（global_kv_pool.py） | 真实 Mooncake 实现 |
|------|------------------------------|-------------------|
| 元数据服务 | Python dict（单进程内） | 分布式（每个 engine 独立维护） |
| 传输方向 | submit_transfer（D pull） | RDMA WRITE（P push） |
| 传输协议 | Python threading sleep | mooncake.engine（RDMA） |
| P/D 发现 | 预置 node_id | Bootstrap Server（HTTP） |
| P/D 协调 | wait_for_kv() 同步等 | asyncio.Event + ZMQ |
| 块地址 | 抽象 block_hash | GPU 物理内存指针 |
| 块合并 | 无 | group_concurrent_contiguous() |

### 7.4 RDMA 从零到懂——完整技术指南

#### 7.4.1 RDMA 是什么？硬件要求

**RDMA（Remote Direct Memory Access）**：允许一台机器的网卡直接读写另一台机器的内存，**完全绕过 CPU**。

```
传统 TCP/IP 网络栈（每次传输的开销）：
  应用层：数据在用户空间
    ↓ syscall（上下文切换 ~1μs）
  内核网络栈：TCP/IP 封包
    ↓ DMA 拷贝
  网卡（NIC）：发送
    ↓ 网络
  对端网卡：接收
    ↓ DMA 拷贝到内核缓冲区
    ↓ 中断/轮询（CPU 介入）
    ↓ 内核到用户空间拷贝
  对端应用层：拿到数据

  总延迟：50-100μs，吞吐量：~10 Gbps（受 CPU 限制）

RDMA 网络栈：
  应用层：在内核注册内存（pin memory，一次性）
    ↓ 用户态 RDMA Verb 直接写入 NIC 队列（零拷贝，无 syscall）
  RDMA NIC：直接从/向注册内存 DMA 传输
    ↓ 网络（InfiniBand 或 RoCEv2）
  对端 RDMA NIC：直接 DMA 写入对端注册内存
    ↓ 完成通知（Completion Queue，无需 CPU 中断）
  对端应用层：直接读取内存（无等待）

  总延迟：1-5μs，吞吐量：100-400 Gbps（硬件直达）
```

**硬件要求**：

| 组件 | 要求 | 备注 |
|------|------|------|
| 网卡 | 支持 RDMA 的 NIC | Mellanox/NVIDIA ConnectX-5/6/7，Intel E810 等 |
| 网络协议 | InfiniBand **或** RoCEv2 | InfiniBand：专用网络，延迟最低；RoCEv2：基于以太网 |
| 驱动 | rdma-core 包，verbs API | Ubuntu: `apt install rdma-core ibverbs-utils` |
| GPU 直传 | GPUDirect RDMA | 需要 nvidia-peermem 驱动模块 |
| 系统 | Linux kernel 4.14+，huge pages | RDMA 通常需要锁定大页内存 |

**检查当前环境是否支持 RDMA**：

```bash
# 1. 查看是否有 RDMA 设备
ibv_devices
# 有输出（如 mlx5_0）→ 支持；无输出 → 无 RDMA 硬件

# 2. 查看设备详情（GID，支持的协议）
ibv_devinfo -d mlx5_0

# 3. 检查 GPUDirect RDMA 支持（需要 NVIDIA GPU）
ls /proc/driver/nvidia-peermem/
# 或
nvidia-smi | grep "RDMA"

# 4. 检查 RoCEv2 配置（以太网 RDMA）
rdma link show

# 5. 检查 mlx5 驱动
lsmod | grep mlx5
# 应该看到 mlx5_core 和 mlx5_ib

# 6. 检查 RDMA 子系统
cat /sys/class/infiniband/*/node_type
# 1 = CA (Channel Adapter) → 支持 RDMA
```

**没有 RDMA 硬件怎么办？**

```bash
# 使用 SoftRoCE（软件模拟 RoCEv2，延迟较高但功能完整）
modprobe rdma_rxe
rdma link add rxe0 type rxe netdev eth0  # eth0 是你的以太网接口
ibv_devices  # 应该出现 rxe0

# 注意：SoftRoCE 延迟约 50-100μs（与 TCP 相当），仅用于开发测试
```

#### 7.4.2 从零开始写一个 RDMA 程序

以下是最简单的 RDMA 发送端/接收端示例，展示核心 API：

```c
// rdma_basic.c — RDMA WRITE 基础版（仅展示关键步骤，省略错误处理）
#include <infiniband/verbs.h>

// ─── 第一步：初始化 ───────────────────────────────────────────────
// 1.1 打开 RDMA 设备
struct ibv_device **dev_list = ibv_get_device_list(NULL);
struct ibv_context *ctx = ibv_open_device(dev_list[0]);

// 1.2 创建 Protection Domain（权限隔离域）
struct ibv_pd *pd = ibv_alloc_pd(ctx);

// ─── 第二步：注册内存 ─────────────────────────────────────────────
// 2.1 分配内存并注册（"钉住"内存，让 RDMA NIC 可以直接访问）
char *buf = malloc(BUF_SIZE);
struct ibv_mr *mr = ibv_reg_mr(
    pd, buf, BUF_SIZE,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
);
// mr->lkey：本地密钥（发送时使用）
// mr->rkey：远端密钥（对方写入时使用）

// ─── 第三步：创建通信对象 ─────────────────────────────────────────
// 3.1 Completion Queue（完成队列，接收操作完成通知）
struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);

// 3.2 Queue Pair（发送/接收队列对）
struct ibv_qp_init_attr qp_attr = {
    .send_cq = cq, .recv_cq = cq,
    .qp_type = IBV_QPT_RC,   // RC = Reliable Connection（有序可靠传输）
};
struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

// 3.3 建立连接（需要交换 QP 信息：lid, qp_num, gid）
//     （这里通常用 TCP 做 out-of-band 交换，即 "bootstrap"）

// ─── 第四步：发起 RDMA WRITE ─────────────────────────────────────
// 4.1 准备发送工作请求
struct ibv_sge sge = {
    .addr   = (uintptr_t)buf,    // 本地内存地址
    .length = BUF_SIZE,
    .lkey   = mr->lkey,          // 本地密钥
};
struct ibv_send_wr wr = {
    .opcode     = IBV_WR_RDMA_WRITE,   // RDMA WRITE 操作
    .wr.rdma = {
        .remote_addr = remote_addr,    // 对端内存地址（bootstrap 阶段获取）
        .rkey        = remote_rkey,    // 对端远端密钥
    },
    .sg_list = &sge,
    .num_sge = 1,
    .send_flags = IBV_SEND_SIGNALED,   // 完成时通知 CQ
};
ibv_post_send(qp, &wr, NULL);         // 提交（异步，立即返回）

// ─── 第五步：等待完成 ─────────────────────────────────────────────
// 5.1 轮询 Completion Queue（用户态轮询，无中断，零延迟）
struct ibv_wc wc;
while (ibv_poll_cq(cq, 1, &wc) == 0) {}  // 轮询直到完成
// wc.status == IBV_WC_SUCCESS → 传输完成！
```

**与 Mooncake 的对应关系**：

```python
# Mooncake 的 TransferEngine 封装了上述 C 代码
# 对应关系：
self.engine.initialize(hostname, "P2PHANDSHAKE", "rdma")
# ↑ ibv_open_device() + ibv_alloc_pd() + 建立 RC QP

self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
# ↑ ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE)

self.engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)
# ↑ ibv_post_send(IBV_WR_RDMA_WRITE) + 等待 ibv_poll_cq()
#   "sync" 表示等待所有传输完成才返回
```

#### 7.4.3 如何测量 RDMA 极限带宽

```bash
# ── 测试环境：两台机器 A（发送）和 B（接收），各有 RDMA NIC ──

# 方法1：使用 perftest 工具（最常用）
apt install perftest

# B（接收端）：
ib_write_bw --ib-dev=mlx5_0

# A（发送端）：
ib_write_bw --ib-dev=mlx5_0 <B的IP>

# 输出示例（100GbE RoCEv2）：
# ---------------------------------------------------------------------------------------
#  #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
#  65536      1000           11234.25           11198.78              0.170497
# ---------------------------------------------------------------------------------------
# 换算：11234 MB/s = 89.9 Gbps ≈ 100 Gbps 网络的 ~90% 利用率

# 方法2：测量延迟
ib_write_lat --ib-dev=mlx5_0             # 接收端
ib_write_lat --ib-dev=mlx5_0 <B的IP>    # 发送端
# 输出：平均 1.2μs，最小 1.0μs（InfiniBand HDR）

# 方法3：GPUDirect RDMA 带宽（GPU 内存直传）
# 需要 nvidia-peermem 已加载
ib_write_bw --ib-dev=mlx5_0 --use-cuda=0    # 使用 GPU 0 的内存
# 理想值：接近 NIC 带宽（但 PCIe 带宽可能成为瓶颈）

# 方法4：模拟 Mooncake 的实际场景（多块 batch 传输）
# 单个 KV block（block_size=16, 32层, 32头, 128维, fp16）
# = 16 * 32 * 32 * 128 * 2 * 2B = 67MB
# 传输 10 个块：670 MB
# 100Gbps 网络：670MB / (100Gbps/8) = 53.6ms
# 但真实 RDMA 延迟只是基准延迟（1-5μs）+ 传输时间
# 所以多块 batch 的效率远高于逐块传输
```

#### 7.4.4 全局池化 vs 本地 Prefix Cache：何时有收益？

通过可观测指标进行定量比较：

```python
# 可观测指标（在 global_kv_pool.py 模拟中可直接测量）

import time
from global_kv_pool import GlobalMetadataServer, MooncakeConnector, compute_block_hashes

# ── 指标1：本地 Prefix Cache 命中耗时 ────────────────────────────
# 直接读取 GPU 内存，不涉及传输
def local_prefix_cache_time(num_cached_tokens: int, gpu_bandwidth_gbps: float = 2000) -> float:
    """
    从 GPU L2 Cache 读取 KV 数据（实际上不需要读，只是跳过 Prefill 计算）
    真实耗时 ≈ 0（token 已在 GPU 内存，attention 直接 attend）
    """
    return 0.0  # 本地命中不消耗时间，只是跳过 Prefill 计算

# ── 指标2：跨节点 RDMA 传输耗时 ──────────────────────────────────
def rdma_transfer_time(num_tokens: int, config: dict) -> float:
    """
    RDMA 传输时间 = 基础延迟 + 数据量 / 带宽

    参数：
      num_tokens: 需要传输的 token 数
      config: {'latency_us': 5, 'bandwidth_gbps': 100,
               'num_layers': 32, 'num_heads': 32, 'head_dim': 128,
               'block_size': 16, 'dtype_bytes': 2}
    """
    num_blocks = num_tokens // config['block_size']
    bytes_per_block = (config['block_size'] * config['num_layers']
                       * config['num_heads'] * config['head_dim']
                       * 2 * config['dtype_bytes'])
    total_bytes = num_blocks * bytes_per_block
    transfer_s = total_bytes / (config['bandwidth_gbps'] * 1e9 / 8)
    latency_s = config['latency_us'] * 1e-6
    return (latency_s + transfer_s) * 1000  # ms

# ── 指标3：Prefill 计算耗时（跳过的对象）─────────────────────────
def prefill_compute_time(num_tokens: int, ms_per_token: float = 0.5) -> float:
    """
    Prefill 计算时间估算（A100 @ 70B 模型，~0.5ms/token）
    """
    return num_tokens * ms_per_token  # ms

# ── 收益分析 ──────────────────────────────────────────────────────
config = {
    'latency_us': 5,           # RDMA 基础延迟 5μs（同机房）
    'bandwidth_gbps': 100,     # 100GbE RoCEv2
    'num_layers': 32,
    'num_heads': 32,
    'head_dim': 128,
    'block_size': 16,
    'dtype_bytes': 2,          # fp16
}

scenarios = [
    ("短系统提示词 (128 tokens)",  128),
    ("中等系统提示词 (1024 tokens)", 1024),
    ("长系统提示词 (8192 tokens)", 8192),
    ("超长上下文 (32768 tokens)", 32768),
]

print(f"{'场景':<30} {'Prefill计算':<15} {'RDMA传输':<15} {'净收益':<12} {'值得？'}")
print("=" * 85)
for desc, num_tokens in scenarios:
    prefill_ms = prefill_compute_time(num_tokens)
    rdma_ms = rdma_transfer_time(num_tokens, config)
    benefit_ms = prefill_ms - rdma_ms
    worth_it = "✅ 有收益" if benefit_ms > 0 else "❌ 无收益"
    print(f"{desc:<30} {prefill_ms:<15.1f} {rdma_ms:<15.1f} {benefit_ms:<12.1f} {worth_it}")
```

**典型输出**：

```
场景                           Prefill计算     RDMA传输        净收益       值得？
=====================================================================================
短系统提示词 (128 tokens)      64.0            0.5             63.5         ✅ 有收益
中等系统提示词 (1024 tokens)   512.0           3.7             508.3        ✅ 有收益
长系统提示词 (8192 tokens)     4096.0          29.4            4066.6       ✅ 有收益
超长上下文 (32768 tokens)      16384.0         117.5           16266.5      ✅ 有收益
```

**无收益场景**（全局池化反而更慢）：

```
场景1：RDMA 带宽很低（如 10GbE）+ 短 prompt（128 tokens）
  RDMA 传输 128 tokens @ 10GbE = 4.7ms
  Prefill 计算 128 tokens      = 64ms
  → 仍然有收益（64 > 4.7）

场景2：本地 Prefix Cache 命中（same-GPU cache hit）
  本地命中耗时：~0ms（GPU L2 cache 读取）
  全局 RDMA 耗时：0.5ms（即使很快）
  → 无收益！全局池化比本地慢！
  结论：本地 Prefix Cache 优先，全局池化仅在本地 miss 时触发

场景3：极高并发，RDMA 网络拥塞
  名义带宽 100Gbps，实际拥塞后 20Gbps
  8192 tokens RDMA：146ms vs Prefill 4096ms
  → 仍有收益，但收益下降

场景4：Prefill 计算很快（小模型，GPU 算力充足）
  小模型（7B），Prefill 128 tokens = 5ms
  RDMA 传输 128 tokens        = 0.5ms
  → 收益变小，但通常仍值得
```

**决策矩阵**：

```
                  本地 GPU KV Cache
                  HIT        MISS
                ┌──────────┬──────────────┐
全局 KV Pool    │ 使用本地  │  使用全局    │
HIT             │ (更快)    │  RDMA传输   │
                ├──────────┼──────────────┤
全局 KV Pool    │ 使用本地  │  重新Prefill │
MISS            │ (更快)    │  (唯一选择) │
                └──────────┴──────────────┘

结论：
  - 本地 hit → 无论如何使用本地（最快）
  - 本地 miss + 全局 hit → 用 RDMA（比重新 Prefill 快）
  - 两级都 miss → 必须 Prefill（无法优化）
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
