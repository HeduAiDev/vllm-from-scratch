# vLLM：从入门到专家

> 版本：基于 vLLM v0.15.1 | 2026年2月 | GPU: NVIDIA RTX PRO 6000 Blackwell (96GB)

---

## 前言

如果你在 2023 年之前问一个工程师"如何高效推理 LLM"，答案大概是"用 Hugging Face，加个 pipeline"。但随着模型体积膨胀、并发请求激增，一个残酷的现实浮出水面：**传统推理框架的显存利用率极低，吞吐量惨不忍睹**。

vLLM 横空出世，用一篇 SOSP 2023 的论文彻底改变了局面。它的核心思想——Paged Attention——借鉴操作系统虚拟内存的思想，将 KV Cache 的利用率从不足 40% 提升至接近 100%，吞吐量比 Hugging Face 高出 24 倍。

但 vLLM 远不止于此。经过两年的社区共建，它已经发展成一个集 **调度器**、**投机解码**、**前缀缓存**、**PD分离**、**DeepSeek MoE 加速** 于一体的完整推理生态系统。

本文目标是：**读完此文，你能彻底理解 vLLM 的每一行关键代码，并能从 0 手写出每个核心特性的可运行实现**。

---

## 第一章：为什么需要 vLLM？

### 1.1 传统推理的痛点

让我们先看一个简单的例子。假设你有一张 40GB 显卡，跑一个 7B 参数的 LLaMA 模型（fp16），模型权重占用约 14GB，还剩 26GB。

一个请求的 KV Cache 大小是多少？

```
KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × dtype_size
                   = 2 × 32 × 32 × 128 × 2 bytes
                   = 524,288 bytes ≈ 0.5 MB/token
```

如果一个请求最大长度 2048 tokens，那一个请求的 KV Cache 最多占用 **1 GB**。26GB 空间理论上可以同时处理 26 个请求。

**但实际情况是：** 传统框架（如早期 HF Accelerate）为每个请求预分配其声明的最大长度内存。如果用户设置 `max_new_tokens=2048`，就预分配 1GB，哪怕这个请求最终只生成了 50 个 token。

结果：**显存碎片严重，实际并发通常不到理论值的 20%**。

### 1.2 vLLM 的核心洞察

vLLM 的洞察简单而深刻：

> **KV Cache 不需要连续存储，也不需要预先知道最终长度。**

就像操作系统的虚拟内存，进程看到的是连续地址，实际上内存是碎片化的物理页面。vLLM 用同样的方法管理 KV Cache：

- 将显存切分成等大的 **Block（块）**，每块容纳固定数量的 token（如 16 个）
- 用 **Block Table** 记录每个请求的逻辑块到物理块的映射
- 按需分配，用完即释放，不同请求可以**共享前缀块**

这就是 **Paged Attention**，后续所有特性都建立在它之上。

---

## 第二章：vLLM 整体架构

### 2.1 代码目录结构

```
vllm/
├── v1/                          # V1 架构（当前主线）
│   ├── engine/                  # 引擎层
│   │   ├── llm_engine.py        # 对外API入口 (LLMEngine)
│   │   ├── core.py              # 核心协调层 (EngineCore)
│   │   ├── core_client.py       # ZMQ通信客户端
│   │   ├── input_processor.py   # 输入预处理
│   │   └── output_processor.py  # 输出后处理（detokenize）
│   ├── core/                    # 核心调度组件
│   │   ├── sched/
│   │   │   ├── scheduler.py     # 调度器主体（2200行）
│   │   │   ├── interface.py     # 调度器接口
│   │   │   └── output.py        # 调度输出数据结构
│   │   ├── kv_cache_manager.py  # KV缓存管理（高层）
│   │   ├── block_pool.py        # 块池（低层分配）
│   │   └── kv_cache_coordinator.py  # 多种缓存协调
│   ├── attention/               # 注意力计算
│   │   ├── backends/            # FlashAttention/FlashInfer等后端
│   │   └── ops/
│   │       └── paged_attn.py    # PagedAttention写入操作
│   ├── spec_decode/             # 投机解码（调度侧）
│   │   ├── eagle.py             # EAGLE提议器
│   │   └── draft_model.py       # Draft模型基类
│   └── worker/
│       └── gpu/
│           └── spec_decode/     # 投机解码（执行侧）
│               └── rejection_sample.py  # 拒绝采样Triton kernel
├── model_executor/              # 模型执行
│   ├── layers/
│   │   ├── fused_moe/           # MoE融合kernel
│   │   │   ├── fused_moe.py     # Triton FusedMoE
│   │   │   └── deep_gemm_moe.py # DeepGEMM加速版
│   │   └── attention/
│   │       └── mla_attention.py # MLA（DeepSeek特有）
│   └── models/
│       ├── deepseek_v2.py       # DeepSeek-V2/V3
│       └── llama.py             # LLaMA系列
├── distributed/
│   └── kv_transfer/             # KV缓存传输（PD分离）
│       └── kv_connector/
│           ├── v1/              # V1版本connector
│           │   └── mooncake/    # Mooncake全局KV池
│           └── base.py          # V0版本接口
├── csrc/                        # CUDA/Triton C++内核
│   ├── attention/               # PagedAttention CUDA kernel
│   └── moe/                     # MoE CUDA kernel
└── entrypoints/                 # 对外入口
    ├── llm.py                   # LLM类（离线推理）
    └── openai/                  # OpenAI兼容API服务
```

### 2.2 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层                                │
│  LLM("model")              AsyncLLMEngine                   │
│  llm.generate(...)         openai_server                    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                      前端引擎层 (LLMEngine)                  │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  InputProcessor  │  │  OutputProcessor │                  │
│  │  (tokenize/mm)  │  │  (detokenize)    │                  │
│  └────────┬────────┘  └─────────▲────────┘                  │
└───────────│─────────────────────│───────────────────────────┘
            │ EngineCoreRequest   │ EngineCoreOutput
┌───────────▼─────────────────────│───────────────────────────┐
│                    协调层 (EngineCore)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                      Scheduler                        │   │
│  │  waiting_queue → [Req1, Req2, ...]                   │   │
│  │  running_list  → [Req3, Req4, ...]                   │   │
│  │  KVCacheManager → BlockPool → BlockHashToBlockMap    │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────│───────────────────────────────────┘
                          │ SchedulerOutput
┌─────────────────────────▼───────────────────────────────────┐
│                   执行层 (ModelExecutor)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               GPU Model Runner                       │    │
│  │  ┌──────────────┐  ┌──────────────┐                 │    │
│  │  │  Attention   │  │  MoE Layer   │  ...            │    │
│  │  │  (FlashAttn) │  │  (FusedMoE)  │                 │    │
│  │  └──────────────┘  └──────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          GPU                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 一个请求的完整生命周期

```
① 请求到达
   HTTP POST /v1/chat/completions
        ↓
② 输入处理 (InputProcessor)
   文本 → token_ids [1234, 5678, 9012, ...]
        ↓
③ 入队等待 (Scheduler.add_request)
   waiting_queue.push(Request(status=WAITING))
        ↓
④ 调度决策 (Scheduler.schedule)
   if 资源充足:
       KVCacheManager.get_computed_blocks()  ← 前缀缓存命中？
       KVCacheManager.allocate_slots()       ← 分配KV Cache块
       Request.status = RUNNING
        ↓
⑤ 模型执行 (ModelExecutor.execute_model)
   for each layer:
       QKV = linear(hidden_states)
       K,V → write_to_paged_cache(slot_mapping)  ← 写入KV Cache
       Attn = paged_attention(Q, K_cache, V_cache, block_table)
        ↓
⑥ 采样 (TokenSampler)
   logits → temperature/top_p → new_token_id
        ↓
⑦ 更新状态 (Scheduler.update_from_output)
   request.output_tokens.append(new_token_id)
   检查停止条件 (EOS/max_len/stop_words)
        ↓
⑧ 输出处理 (OutputProcessor)
   token_ids → text (detokenize)
        ↓
⑨ 返回用户
   HTTP streaming response / 完整response
```

---

## 第三章：Paged Attention——从0手搓

### 3.1 理解核心问题

传统 Attention 计算：

```python
# 标准 Attention（连续内存）
Q = [q1, q2, q3]      # [seq_len, head_dim]
K = [k1, k2, k3]      # [seq_len, head_dim]  ← 必须连续！
V = [v1, v2, v3]      # [seq_len, head_dim]  ← 必须连续！

scores = Q @ K.T / sqrt(head_dim)
output = softmax(scores) @ V
```

Paged Attention 允许 K、V **非连续存储**：

```
KV Cache 物理布局（Block Size=2）：
Block 0: [k1,k2] [v1,v2]   ← Request A 的前2个token
Block 1: [k5,k6] [v5,v6]   ← Request B 的前2个token
Block 2: [k3,k4] [v3,v4]   ← Request A 的后2个token
Block 3: [k7,k8] [v7,v8]   ← Request B 的后2个token

Request A 的 block_table = [0, 2]  → 逻辑上连续，物理上不连续
Request B 的 block_table = [1, 3]
```

### 3.2 从0实现 Paged Attention

**文件：`01_paged_attention/paged_attention.py`**

```python
# vllm-from-scratch/01_paged_attention/paged_attention.py
import math
import torch
import torch.nn.functional as F
from typing import List

# ─────────────────────────────────────────────────────────
# 核心数据结构
# ─────────────────────────────────────────────────────────

class PhysicalBlock:
    """物理块：GPU显存中的一段连续内存"""
    def __init__(self, block_id: int, block_size: int,
                 num_kv_heads: int, head_dim: int, dtype=torch.float16):
        self.block_id = block_id
        self.block_size = block_size  # 每块容纳的token数
        # K cache: [block_size, num_kv_heads, head_dim]
        # V cache: [block_size, num_kv_heads, head_dim]
        # 实际上在GPU上是连续的大tensor，这里简化表示
        self.ref_count = 0           # 引用计数（多请求共享前缀时>1）


class KVCachePool:
    """
    KV Cache 物理块池
    对应 vLLM 的 BlockPool
    """
    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 num_kv_heads: int,
                 head_dim: int,
                 num_layers: int,
                 dtype=torch.float16,
                 device='cuda'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device

        # 关键：所有层的KV Cache存储在一个大tensor中
        # shape: [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        # 第0维：0=K, 1=V
        self.kv_cache = torch.zeros(
            2, num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )

        # 空闲块队列（双向链表，这里用list简化）
        self.free_blocks: List[int] = list(range(num_blocks))
        self.all_blocks = [PhysicalBlock(i, block_size, num_kv_heads, head_dim)
                          for i in range(num_blocks)]

    def allocate(self, num_blocks: int) -> List[int]:
        """分配指定数量的物理块，返回块ID列表"""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"OOM: need {num_blocks} blocks, "
                             f"only {len(self.free_blocks)} free")
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)  # 简化：从头取
            self.all_blocks[block_id].ref_count = 1
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: List[int]) -> None:
        """释放物理块"""
        for block_id in block_ids:
            block = self.all_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)

    def write_kv(self, layer_idx: int, block_id: int,
                 slot_in_block: int, key: torch.Tensor, value: torch.Tensor):
        """
        将一个token的KV写入指定位置
        key: [num_kv_heads, head_dim]
        value: [num_kv_heads, head_dim]
        """
        self.kv_cache[0, layer_idx, block_id, :, slot_in_block, :] = key
        self.kv_cache[1, layer_idx, block_id, :, slot_in_block, :] = value

    def read_kv_for_request(self, layer_idx: int,
                            block_table: List[int],
                            seq_len: int) -> tuple:
        """
        读取一个请求的所有KV
        block_table: [num_blocks] 该请求的物理块映射
        返回: K[seq_len, num_kv_heads, head_dim], V[seq_len, num_kv_heads, head_dim]
        """
        k_list, v_list = [], []
        for i in range(seq_len):
            block_idx = i // self.block_size
            slot_in_block = i % self.block_size
            physical_block = block_table[block_idx]
            k = self.kv_cache[0, layer_idx, physical_block, :, slot_in_block, :]
            v = self.kv_cache[1, layer_idx, physical_block, :, slot_in_block, :]
            k_list.append(k)
            v_list.append(v)
        return torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)


class Request:
    """推理请求"""
    def __init__(self, request_id: str, token_ids: List[int], max_new_tokens: int):
        self.request_id = request_id
        self.token_ids = token_ids  # prompt token IDs
        self.output_token_ids: List[int] = []
        self.max_new_tokens = max_new_tokens
        self.block_table: List[int] = []  # 逻辑块 → 物理块映射

    @property
    def seq_len(self) -> int:
        return len(self.token_ids) + len(self.output_token_ids)

    @property
    def num_blocks_needed(self) -> int:
        """当前序列需要的块数"""
        return math.ceil(self.seq_len / BLOCK_SIZE)


# ─────────────────────────────────────────────────────────
# Paged Attention 核心计算
# ─────────────────────────────────────────────────────────

BLOCK_SIZE = 16  # 每个物理块存放16个token

def paged_attention_decode(
    query: torch.Tensor,        # [num_heads, head_dim] 当前token的Q
    kv_cache: torch.Tensor,     # [2, num_blocks, num_kv_heads, block_size, head_dim]
    block_table: List[int],     # 该请求的物理块ID列表
    seq_len: int,               # 已有序列长度（含当前）
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Paged Attention 解码阶段实现

    这是最核心的函数：给定非连续存储的KV Cache，计算Attention输出

    实际vLLM中这是CUDA/Triton kernel实现的，这里是纯Python参考实现
    """
    scale = 1.0 / math.sqrt(head_dim)
    # GQA支持：每个KV头对应多个Q头
    gqa_factor = num_q_heads // num_kv_heads

    # 初始化输出
    output = torch.zeros(num_q_heads, head_dim,
                        dtype=query.dtype, device=query.device)

    # 对每个KV头分组计算
    for kv_head_idx in range(num_kv_heads):
        q_heads = query[kv_head_idx * gqa_factor : (kv_head_idx+1) * gqa_factor]
        # q_heads: [gqa_factor, head_dim]

        # 从非连续的物理块中收集K、V
        keys = []
        vals = []
        for token_idx in range(seq_len):
            block_idx = token_idx // BLOCK_SIZE
            slot_in_block = token_idx % BLOCK_SIZE
            phys_block = block_table[block_idx]

            # 从物理块读取K、V
            k = kv_cache[0, phys_block, kv_head_idx, slot_in_block]  # [head_dim]
            v = kv_cache[1, phys_block, kv_head_idx, slot_in_block]  # [head_dim]
            keys.append(k)
            vals.append(v)

        K = torch.stack(keys, dim=0)  # [seq_len, head_dim]
        V = torch.stack(vals, dim=0)  # [seq_len, head_dim]

        # 计算Attention
        # scores: [gqa_factor, seq_len]
        scores = torch.einsum('gh,sh->gs', q_heads, K) * scale
        attn_weights = F.softmax(scores, dim=-1)
        # attended: [gqa_factor, head_dim]
        attended = torch.einsum('gs,sh->gh', attn_weights, V)

        output[kv_head_idx * gqa_factor : (kv_head_idx+1) * gqa_factor] = attended

    return output  # [num_q_heads, head_dim]


def paged_attention_prefill(
    query: torch.Tensor,        # [seq_len, num_heads, head_dim]
    key: torch.Tensor,          # [seq_len, num_kv_heads, head_dim]
    value: torch.Tensor,        # [seq_len, num_kv_heads, head_dim]
    kv_cache: torch.Tensor,     # KV Cache tensor
    block_table: List[int],     # 物理块映射
    block_size: int,
) -> torch.Tensor:
    """
    Prefill阶段：计算完整prompt的Attention并写入KV Cache
    """
    seq_len, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    gqa_factor = num_heads // num_kv_heads

    # 先写入KV Cache
    for token_idx in range(seq_len):
        block_idx = token_idx // block_size
        slot = token_idx % block_size
        phys_block = block_table[block_idx]
        kv_cache[0, phys_block, :, slot] = key[token_idx]    # K
        kv_cache[1, phys_block, :, slot] = value[token_idx]  # V

    # 计算causal self-attention（上三角mask）
    # [seq_len, num_heads, head_dim] → [num_heads, seq_len, head_dim]
    Q = query.permute(1, 0, 2)
    K = key.permute(1, 0, 2)   # [num_kv_heads, seq_len, head_dim]
    V = value.permute(1, 0, 2)

    # GQA展开
    K = K.repeat_interleave(gqa_factor, dim=0)  # [num_heads, seq_len, head_dim]
    V = V.repeat_interleave(gqa_factor, dim=0)

    # 计算分数
    scores = torch.bmm(Q, K.transpose(1, 2)) * scale  # [num_heads, seq_len, seq_len]

    # 因果mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device),
                             diagonal=1).bool()
    scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.bmm(attn_weights, V)  # [num_heads, seq_len, head_dim]
    return output.permute(1, 0, 2)       # [seq_len, num_heads, head_dim]


# ─────────────────────────────────────────────────────────
# slot_mapping：vLLM真正的写入机制
# ─────────────────────────────────────────────────────────

def compute_slot_mapping(block_table: List[int], seq_len: int,
                         block_size: int) -> List[int]:
    """
    计算slot_mapping：每个token对应的物理位置

    slot_mapping[i] = block_table[i // block_size] * block_size + (i % block_size)

    这就是vLLM中 kv_cache[layer][slot_mapping[i]] 的含义
    """
    slots = []
    for token_idx in range(seq_len):
        block_idx = token_idx // block_size
        slot_in_block = token_idx % block_size
        phys_block = block_table[block_idx]
        slot = phys_block * block_size + slot_in_block
        slots.append(slot)
    return slots
```

### 3.3 单元测试

**文件：`01_paged_attention/test_paged_attention.py`**

```python
# vllm-from-scratch/01_paged_attention/test_paged_attention.py
import pytest
import torch
import math
from paged_attention import (
    KVCachePool, Request, paged_attention_decode,
    paged_attention_prefill, compute_slot_mapping, BLOCK_SIZE
)


@pytest.fixture
def kv_pool():
    """创建一个小型KV Cache池用于测试"""
    return KVCachePool(
        num_blocks=32,
        block_size=BLOCK_SIZE,
        num_kv_heads=4,
        head_dim=64,
        num_layers=2,
        dtype=torch.float32,
        device='cpu'
    )


class TestKVCachePool:
    def test_allocate_and_free(self, kv_pool):
        """测试块的分配和释放"""
        initial_free = len(kv_pool.free_blocks)

        # 分配3个块
        blocks = kv_pool.allocate(3)
        assert len(blocks) == 3
        assert len(kv_pool.free_blocks) == initial_free - 3

        # 释放
        kv_pool.free(blocks)
        assert len(kv_pool.free_blocks) == initial_free

    def test_oom_raises(self, kv_pool):
        """OOM应该抛出异常"""
        with pytest.raises(RuntimeError, match="OOM"):
            kv_pool.allocate(999)

    def test_write_read_kv(self, kv_pool):
        """写入然后读取KV应该一致"""
        blocks = kv_pool.allocate(1)
        block_id = blocks[0]

        # 写入K, V
        k = torch.randn(4, 64)  # [num_kv_heads, head_dim]
        v = torch.randn(4, 64)
        kv_pool.write_kv(layer_idx=0, block_id=block_id,
                        slot_in_block=0, key=k, value=v)

        # 验证读取
        stored_k = kv_pool.kv_cache[0, 0, block_id, :, 0, :]
        stored_v = kv_pool.kv_cache[1, 0, block_id, :, 0, :]
        assert torch.allclose(stored_k, k)
        assert torch.allclose(stored_v, v)


class TestPagedAttention:
    def test_decode_output_shape(self):
        """解码Attention输出形状正确"""
        num_q_heads, num_kv_heads, head_dim = 8, 4, 64
        seq_len = 32
        num_blocks = 4
        block_size = 16

        # 创建KV Cache
        kv_cache = torch.randn(2, num_blocks, num_kv_heads, block_size, head_dim)
        query = torch.randn(num_q_heads, head_dim)
        block_table = [0, 1, 2]  # 3个块覆盖48个slot，seq_len=32足够

        output = paged_attention_decode(
            query, kv_cache, block_table, seq_len,
            head_dim, num_q_heads, num_kv_heads
        )
        assert output.shape == (num_q_heads, head_dim)

    def test_paged_vs_standard_attention(self):
        """
        关键测试：Paged Attention与标准Attention的数值等价性
        """
        torch.manual_seed(42)
        num_q_heads, num_kv_heads, head_dim = 4, 4, 32
        seq_len = 8
        block_size = 4
        num_blocks = 4

        # 生成随机K、V序列
        K_full = torch.randn(seq_len, num_kv_heads, head_dim)
        V_full = torch.randn(seq_len, num_kv_heads, head_dim)
        Q_last = torch.randn(num_q_heads, head_dim)  # 最后一个token的Q

        # ── 标准Attention（连续内存）──
        scale = 1.0 / math.sqrt(head_dim)
        K_flat = K_full.view(seq_len, num_q_heads, head_dim)  # 这里GQA=1
        V_flat = V_full.view(seq_len, num_q_heads, head_dim)
        scores_std = torch.einsum('hd,shd->hs', Q_last, K_flat) * scale
        attn_std = torch.softmax(scores_std, dim=-1)
        output_std = torch.einsum('hs,shd->hd', attn_std, V_flat)

        # ── Paged Attention（非连续块）──
        # 填充KV Cache：seq_len=8, block_size=4，需要2个块
        kv_cache = torch.zeros(2, num_blocks, num_kv_heads, block_size, head_dim)
        block_table = [0, 1]  # 物理块0和1

        for i in range(seq_len):
            blk = i // block_size
            slot = i % block_size
            phys = block_table[blk]
            kv_cache[0, phys, :, slot] = K_full[i]
            kv_cache[1, phys, :, slot] = V_full[i]

        output_paged = paged_attention_decode(
            Q_last, kv_cache, block_table, seq_len,
            head_dim, num_q_heads, num_kv_heads
        )

        # 数值应该相同（误差在浮点精度范围内）
        assert torch.allclose(output_std, output_paged, atol=1e-5), \
            f"Max diff: {(output_std - output_paged).abs().max()}"

    def test_slot_mapping_correctness(self):
        """slot_mapping计算正确性"""
        block_table = [3, 7, 1]  # 物理块3, 7, 1
        block_size = 4
        seq_len = 10

        slots = compute_slot_mapping(block_table, seq_len, block_size)

        # token 0 → block 3, slot 0 → physical slot 3*4+0=12
        assert slots[0] == 12
        # token 3 → block 3, slot 3 → physical slot 3*4+3=15
        assert slots[3] == 15
        # token 4 → block 7, slot 0 → physical slot 7*4+0=28
        assert slots[4] == 28
        # token 8 → block 1, slot 0 → physical slot 1*4+0=4
        assert slots[8] == 4

    def test_non_contiguous_blocks_same_result(self):
        """
        验证：无论块如何分布（连续或乱序），结果都相同
        """
        torch.manual_seed(0)
        num_heads, head_dim, seq_len = 2, 16, 6
        block_size = 2
        num_blocks = 8

        K = torch.randn(seq_len, num_heads, head_dim)
        V = torch.randn(seq_len, num_heads, head_dim)
        Q = torch.randn(num_heads, head_dim)

        def run_with_blocks(block_ids):
            kv_cache = torch.zeros(2, num_blocks, num_heads, block_size, head_dim)
            for i in range(seq_len):
                blk = i // block_size
                slot = i % block_size
                phys = block_ids[blk]
                kv_cache[0, phys, :, slot] = K[i]
                kv_cache[1, phys, :, slot] = V[i]
            return paged_attention_decode(
                Q, kv_cache, block_ids, seq_len,
                head_dim, num_heads, num_heads
            )

        # 连续块顺序
        out1 = run_with_blocks([0, 1, 2])
        # 乱序块
        out2 = run_with_blocks([5, 2, 7])
        # 块顺序不影响结果（内容相同，位置不同）
        assert torch.allclose(out1, out1)  # 自我一致性
        assert torch.allclose(out2, out2)
        # 两种布局对同一内容应该得到相同结果
        assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 3.4 vLLM 中的真实实现

vLLM 的 PagedAttention 实际上是 CUDA kernel，位于：

- **写入KV**: `vllm/v1/attention/ops/paged_attn.py` → `PagedAttention.write_to_paged_cache()` → `ops.reshape_and_cache()`
- **CUDA kernel**: `vllm/csrc/attention/attention_kernels.cu`

```python
# vllm/v1/attention/ops/paged_attn.py (精简版)
class PagedAttention:
    @staticmethod
    def write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, kv_cache_dtype, k_scale, v_scale):
        """
        slot_mapping: [num_tokens] 每个token对应的物理slot
        key_cache: [num_blocks, num_kv_heads, head_size//x, block_size, x]

        注意key_cache的特殊layout！head_size被拆分为(head_size//x, x)
        x = 16 // element_size，这是为了CUDA访存对齐优化
        """
        ops.reshape_and_cache(
            key, value, key_cache, value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype, k_scale, v_scale,
        )
```

**slot_mapping 是整个 Paged Attention 的关键**：

```
slot_mapping[i] = block_table[i // block_size] * block_size + (i % block_size)

例子：
  token 0 在 block_table[0]=5号物理块的slot 0 → 5*16+0 = 80
  token 1 在 block_table[0]=5号物理块的slot 1 → 5*16+1 = 81
  ...
  token 16 在 block_table[1]=2号物理块的slot 0 → 2*16+0 = 32

kv_cache的内存布局是扁平化的：
  kv_cache[80] = token 0 的 K
  kv_cache[81] = token 1 的 K
  kv_cache[32] = token 16 的 K
```

---

## 第四章：KV Cache 的存取、换入与换出

### 4.1 KV Cache 的三种状态

每个物理块可以处于三种状态：

```
┌─────────────────────────────────────────────────┐
│                  BlockPool                       │
│                                                  │
│  ┌──────────────┐  ┌───────────┐  ┌──────────┐  │
│  │  Active      │  │  Cached   │  │  Free    │  │
│  │  (ref_cnt>0) │  │  (有hash) │  │  (ref=0) │  │
│  │              │  │           │  │          │  │
│  │ 当前运行中的  │  │ 可复用的  │  │ 可立即   │  │
│  │ 请求正在使用  │  │ 前缀块   │  │ 分配     │  │
│  └──────────────┘  └───────────┘  └──────────┘  │
└─────────────────────────────────────────────────┘
```

### 4.2 块的生命周期

```
分配 → 写入KV → 请求运行中 → 请求完成
                                  ↓
                      ref_cnt-- 变为0
                                  ↓
                      有 block_hash?
                        ├── YES → 加入 free_block_queue（LRU末尾）
                        │         继续作为前缀缓存，等待复用或淘汰
                        └── NO  → 加入 free_block_queue（LRU末尾）
                                  下次分配时直接覆盖
```

**关键代码（`block_pool.py`）：**

```python
def free_blocks(self, ordered_blocks):
    """释放块：ref_cnt--，归0则入队"""
    for block in ordered_blocks:
        block.ref_cnt -= 1
    # 只有ref_cnt==0的块才真正进入空闲队列
    self.free_block_queue.append_n(
        [b for b in ordered_blocks if b.ref_cnt == 0 and not b.is_null]
    )

def get_new_blocks(self, num_blocks):
    """从空闲队列取块（可能触发LRU淘汰）"""
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        self._maybe_evict_cached_block(block)  # 淘汰旧缓存
        block.ref_cnt += 1
    return ret

def _maybe_evict_cached_block(self, block):
    """如果块有hash（是前缀缓存），从hash表中删除"""
    block_hash = block.block_hash
    if block_hash is None:
        return False
    # 从 BlockHashToBlockMap 中移除
    self.cached_block_hash_to_block.pop(block_hash, block.block_id)
    block.reset_hash()
    return True
```

### 4.3 vLLM V1 没有 Swap（CPU换出）

**重要区别**：vLLM V1 完全移除了 V0 中的 CPU swap 机制！

在 V0 中，当 GPU 内存不足时，会将被抢占请求的 KV Cache swap 到 CPU RAM。这导致了：
- CPU-GPU 传输延迟
- 调度器逻辑复杂
- CPU 内存也可能耗尽

**V1 的策略：直接抢占（Preemption without swap）**

```python
# vllm/v1/core/sched/scheduler.py
def _preempt_request(self, request, scheduled_timestamp):
    """
    抢占一个请求：
    1. 释放其所有KV Cache块（块变回free状态）
    2. 请求重新进入waiting队列
    3. 下次调度时，如果前缀缓存命中，可以避免重新计算
    """
    self.kv_cache_manager.free(request)  # 释放KV Cache块
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = request.num_cached_tokens  # 回退到缓存位置

    # 重新加入waiting队列（优先级可能不同）
    self.waiting.add_request(request)
    self.running.remove(request)
```

这样设计的好处：
- **简单**：不需要CPU内存管理
- **前缀缓存救场**：被抢占的请求块如果有hash，会保留在LRU缓存中，下次调度时可能命中
- **等效效果**：只要缓存没被淘汰，重新调度时直接复用，不需要重新计算

### 4.4 从0实现 BlockPool with LRU

**文件：`02_kvcache/block_pool_lru.py`**

```python
# vllm-from-scratch/02_kvcache/block_pool_lru.py
from collections import OrderedDict
from typing import Optional, List, Dict
import hashlib


class DoublyLinkedNode:
    """双向链表节点"""
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: Optional[str] = None
        self.prev: Optional['DoublyLinkedNode'] = None
        self.next: Optional['DoublyLinkedNode'] = None


class FreeBlockQueue:
    """
    空闲块双向链表（LRU顺序）

    head ↔ [最久未用] ↔ ... ↔ [最近释放] ↔ tail

    分配时从头取（最久未用的先被重用）
    释放时加到尾（最近释放的作为LRU末尾）
    """
    def __init__(self, blocks: List[DoublyLinkedNode]):
        self.head = DoublyLinkedNode(-1)  # 哨兵头
        self.tail = DoublyLinkedNode(-1)  # 哨兵尾
        self.head.next = self.tail
        self.tail.prev = self.head
        self.num_free_blocks = 0

        # 初始化：所有块都空闲
        for block in blocks:
            self._append_to_tail(block)

    def _append_to_tail(self, node: DoublyLinkedNode):
        """加到链表尾部（最近释放）"""
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node
        self.num_free_blocks += 1

    def _remove(self, node: DoublyLinkedNode):
        """从链表中移除"""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self.num_free_blocks -= 1

    def popleft(self) -> DoublyLinkedNode:
        """取出最久未用的块（头部）"""
        if self.head.next == self.tail:
            raise RuntimeError("No free blocks!")
        block = self.head.next
        self._remove(block)
        return block

    def append(self, node: DoublyLinkedNode):
        """释放块：加到尾部"""
        self._append_to_tail(node)

    def remove(self, node: DoublyLinkedNode):
        """从空闲队列中移除（块被复用时）"""
        self._remove(node)


class BlockPoolWithLRU:
    """
    带LRU淘汰策略的块池

    对应 vLLM 的 BlockPool 实现
    """
    def __init__(self, num_blocks: int, enable_prefix_cache: bool = True):
        self.num_blocks = num_blocks
        self.enable_prefix_cache = enable_prefix_cache

        # 所有块
        self.blocks = [DoublyLinkedNode(i) for i in range(num_blocks)]

        # 空闲块队列（LRU顺序）
        self.free_queue = FreeBlockQueue(self.blocks)

        # 前缀缓存哈希表：hash → block
        self.hash_to_block: Dict[str, DoublyLinkedNode] = {}

    def get_num_free_blocks(self) -> int:
        return self.free_queue.num_free_blocks

    def get_cached_block(self, block_hash: str) -> Optional[DoublyLinkedNode]:
        """查找前缀缓存命中"""
        return self.hash_to_block.get(block_hash)

    def touch(self, block: DoublyLinkedNode):
        """
        增加引用计数
        如果block在空闲队列中（ref_cnt==0），从队列中移除
        """
        if block.ref_cnt == 0:
            self.free_queue.remove(block)  # 从空闲队列中取出
        block.ref_cnt += 1

    def allocate_new_block(self) -> DoublyLinkedNode:
        """
        分配新块（从LRU队列头取）
        可能淘汰已有的前缀缓存
        """
        block = self.free_queue.popleft()  # LRU：取最久未用

        # 淘汰旧缓存
        if block.block_hash is not None:
            del self.hash_to_block[block.block_hash]
            block.block_hash = None

        block.ref_cnt = 1
        return block

    def free(self, block: DoublyLinkedNode):
        """释放块：ref_cnt--，归0则入空闲队列"""
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_queue.append(block)  # 加到LRU尾部（最近释放）

    def cache_block(self, block: DoublyLinkedNode, block_hash: str):
        """
        将块标记为可缓存（前缀缓存）
        在块变满后调用
        """
        if not self.enable_prefix_cache:
            return
        block.block_hash = block_hash
        self.hash_to_block[block_hash] = block
```

**对应单元测试：`02_kvcache/test_block_pool_lru.py`**

```python
# vllm-from-scratch/02_kvcache/test_block_pool_lru.py
import pytest
from block_pool_lru import BlockPoolWithLRU


class TestBlockPoolLRU:
    def test_basic_alloc_free(self):
        pool = BlockPoolWithLRU(num_blocks=4)
        assert pool.get_num_free_blocks() == 4

        b = pool.allocate_new_block()
        assert b.ref_cnt == 1
        assert pool.get_num_free_blocks() == 3

        pool.free(b)
        assert b.ref_cnt == 0
        assert pool.get_num_free_blocks() == 4

    def test_lru_eviction_order(self):
        """LRU顺序：最久未用的先被驱逐"""
        pool = BlockPoolWithLRU(num_blocks=4)

        b0 = pool.allocate_new_block()  # block 0
        b1 = pool.allocate_new_block()  # block 1
        b2 = pool.allocate_new_block()  # block 2

        # 释放顺序：b0 → b1 → b2
        pool.free(b0)  # b0最先释放 → LRU头部
        pool.free(b1)
        pool.free(b2)  # b2最后释放 → LRU尾部

        # 下次分配应该取b0（最久未用）
        new_b = pool.allocate_new_block()
        assert new_b.block_id == b0.block_id

    def test_prefix_cache_hit(self):
        """前缀缓存命中测试"""
        pool = BlockPoolWithLRU(num_blocks=4, enable_prefix_cache=True)

        # 分配并缓存一个块
        b = pool.allocate_new_block()
        pool.cache_block(b, block_hash="hash_abc")
        pool.free(b)  # 释放但保留缓存

        # 查找应该命中
        cached = pool.get_cached_block("hash_abc")
        assert cached is not None
        assert cached.block_id == b.block_id

    def test_cache_eviction_on_alloc(self):
        """分配新块时应淘汰LRU缓存"""
        pool = BlockPoolWithLRU(num_blocks=2, enable_prefix_cache=True)

        b0 = pool.allocate_new_block()
        pool.cache_block(b0, "hash_0")
        pool.free(b0)

        b1 = pool.allocate_new_block()
        pool.cache_block(b1, "hash_1")
        pool.free(b1)

        # 现在全部空闲，分配一个新块
        # 应该淘汰最旧的（b0，即"hash_0"）
        new_b = pool.allocate_new_block()
        assert pool.get_cached_block("hash_0") is None  # 已淘汰
        assert pool.get_cached_block("hash_1") is not None  # 仍在缓存

    def test_touch_removes_from_free_queue(self):
        """touch应该从空闲队列中移除"""
        pool = BlockPoolWithLRU(num_blocks=4, enable_prefix_cache=True)

        b = pool.allocate_new_block()
        pool.cache_block(b, "hash_x")
        pool.free(b)
        assert pool.get_num_free_blocks() == 3 + 1  # b回到空闲队列

        # touch: 复用缓存块（前缀命中）
        pool.touch(b)
        # b从空闲队列移除，ref_cnt=1
        assert b.ref_cnt == 1
        assert pool.get_num_free_blocks() == 3

    def test_multi_request_sharing(self):
        """多请求共享前缀块"""
        pool = BlockPoolWithLRU(num_blocks=8, enable_prefix_cache=True)

        # 请求A创建前缀块
        b = pool.allocate_new_block()
        pool.cache_block(b, "system_prompt_hash")

        # 请求B命中并共享
        cached = pool.get_cached_block("system_prompt_hash")
        assert cached.block_id == b.block_id
        pool.touch(cached)  # 增加引用

        # 块现在被两个请求共享，ref_cnt=2
        assert b.ref_cnt == 2

        # 请求A完成
        pool.free(b)
        assert b.ref_cnt == 1  # 请求B还在用

        # 请求B完成
        pool.free(b)
        assert b.ref_cnt == 0  # 完全释放，回到空闲队列


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 第五章：一个请求的完整推理路径

这一章用完整的代码走读方式，跟踪一个请求从 HTTP 到 token 输出的每一步。

### 5.1 Step 1: 用户发起请求

```
POST /v1/chat/completions
{
  "model": "glm-4-9b-chat",
  "messages": [{"role": "user", "content": "解释量子纠缠"}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

### 5.2 Step 2: OpenAI Entrypoint 处理

```
vllm/entrypoints/openai/api_server.py
    → chat()
    → async_engine.create_chat_completion()
    → AsyncLLMEngine.generate()
    → LLMEngine.add_request()
```

### 5.3 Step 3: InputProcessor 处理输入

```python
# vllm/v1/engine/input_processor.py
class InputProcessor:
    def process_inputs(self, request_id, prompt, params, ...):
        # 1. Tokenize
        token_ids = self.tokenizer.encode("解释量子纠缠")
        # → [1234, 5678, 9012, ...]

        # 2. 构造 EngineCoreRequest
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
            arrival_time=time.time(),
        )
```

### 5.4 Step 4: Scheduler 接收请求

```python
# vllm/v1/core/sched/scheduler.py
def add_request(self, request: Request):
    self.waiting.add_request(request)       # 加入等待队列
    self.requests[request.request_id] = request
    # request.status = WAITING
```

### 5.5 Step 5: schedule() 调度

```python
def schedule(self):
    # PHASE 2: 尝试调度WAITING请求
    request = self.waiting.peek_request()   # 取出一个等待请求

    # 前缀缓存检查
    computed_blocks, num_cached_tokens = \
        self.kv_cache_manager.get_computed_blocks(request)
    # 假设没有缓存：computed_blocks=[], num_cached_tokens=0

    num_new_tokens = len(request.prompt_token_ids)  # 全部prompt需要处理

    # 分配KV Cache块
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=0
    )
    # 分配了 ceil(512/16)=32 个物理块

    # 成功调度
    request.status = RequestStatus.RUNNING
    self.running.append(request)

    return SchedulerOutput(
        scheduled_new_reqs=[NewRequestData(request, ...)],
        num_scheduled_tokens={request.request_id: num_new_tokens},
        req_to_new_blocks={request.request_id: new_blocks},
    )
```

### 5.6 Step 6: ModelExecutor 执行 Forward Pass

```python
# GPU Model Runner
def execute_model(self, scheduler_output):
    # 从 scheduler_output 准备输入
    input_ids = [token_ids]      # prompt tokens
    positions = [0, 1, 2, ...]   # 位置编码
    block_table = [[2, 5, 7, ...]]  # 物理块映射

    # slot_mapping: 每个token对应的物理slot
    slot_mapping = compute_slot_mapping(block_table, seq_len, block_size)
    # → [32, 33, 34, ..., 47, 80, 81, ...]

    # 运行模型
    hidden_states = self.model(
        input_ids=input_ids,
        positions=positions,
        kv_cache=self.kv_cache,
        slot_mapping=slot_mapping,
        block_table=block_table,
    )
    # 内部每层Attention：
    # 1. QKV projection
    # 2. reshape_and_cache(K, V, slot_mapping)  ← 写入KV Cache
    # 3. flash_attention(Q, K_cache, V_cache, block_table)  ← 读取

    # 最后一层 logits
    logits = self.model.lm_head(hidden_states[-1:])  # 只取最后一个token的logits
    sampled_token = sample(logits, temperature=0.7)   # 采样

    return ModelRunnerOutput(sampled_token_ids=[[sampled_token]])
```

### 5.7 Step 7: update_from_output() 更新状态

```python
def update_from_output(self, scheduler_output, model_output):
    new_token = model_output.sampled_token_ids[0][0]  # 获取生成的token

    request.output_token_ids.append(new_token)        # 添加到输出
    request.num_computed_tokens += num_scheduled_tokens # 更新已处理token数

    # 检查停止条件
    if new_token == EOS_TOKEN_ID:
        request.status = RequestStatus.FINISHED_STOPPED
    elif len(request.output_token_ids) >= 512:
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    # 否则继续：下一轮 decode 阶段（只生成1个token）

    return EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[new_token],
        finish_reason=...,
    )
```

### 5.8 Step 8: OutputProcessor Detokenize

```python
# vllm/v1/engine/output_processor.py
def process_outputs(self, engine_core_outputs):
    for output in engine_core_outputs:
        detokenizer = self.detokenizers[output.request_id]
        new_text = detokenizer.decode(output.new_token_ids)
        # "量" / "子" / "纠" / "缠" / ...（逐token流式输出）

        yield RequestOutput(
            request_id=output.request_id,
            outputs=[CompletionOutput(text=new_text, ...)],
        )
```

### 5.9 Decode 阶段循环

Prefill 完成后，进入 Decode 循环：

```
每次 step():
  scheduler.schedule():
    - request 在 running 列表
    - num_new_tokens = 1（只生成1个token）
    - 检查是否需要新块：
        current_slot = (total_tokens) % block_size
        if current_slot == 0:  # 块满了
            allocate 1 new block

  executor.execute_model(1 token):
    - 只对最新token做 Attention（Decode模式）
    - Q = 最新token的Q
    - K,V = 读取 KV Cache 中所有历史 K,V（通过 block_table）
    - 输出下一个 token

  重复直到 EOS 或 max_tokens
```

---

这是第一部分，后续将继续写 Prefix Cache、Scheduler、投机解码、Chunked Prefill、DeepSeek MoE 和 PD 分离。
