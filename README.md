# vLLM 从入门到专家 —— 配套代码仓库

本仓库是《vLLM：从入门到专家》三篇博客的配套代码，所有实现均可在 vllm 容器中运行。

## 目录结构

```
vllm-from-scratch/
├── BLOG_PART1.md          # 第1-5章：架构 + Paged Attention + KV Cache
├── BLOG_PART2.md          # 第6-10章：Prefix Cache + Scheduler + 投机解码 + Chunked Prefill
├── BLOG_PART3.md          # 第11-15章：DeepSeek MoE + MLA + PD分离 + 整合
│
├── 01_paged_attention/
│   ├── paged_attention.py       # Paged Attention + Block Table 完整实现
│   └── test_paged_attention.py  # 9 个单元测试
│
├── 02_kvcache/
│   ├── block_pool_lru.py        # LRU Prefix Cache 实现
│   └── test_block_pool_lru.py   # 12 个单元测试
│
├── 03_moe/
│   ├── mini_moe.py              # Mini MoE：TopK + GroupedTopK + 共享专家
│   └── test_mini_moe.py         # 19 个单元测试
│
├── 04_mla/
│   ├── mini_mla.py              # Mini MLA：低秩KV压缩 + 解耦RoPE
│   └── test_mini_mla.py         # 13 个单元测试
│
├── 05_mini_vllm/
│   ├── mini_vllm.py             # 完整 Mini vLLM 引擎（整合所有组件）
│   └── test_mini_vllm.py        # 26 个单元测试
│
└── 06_global_prefix_cache/
    ├── global_kv_pool.py        # Mooncake 风格全局 KV Cache 池（含 RDMA 传输模拟）
    └── test_global_kv_pool.py   # 34 个单元测试
```

## 快速运行（在 vllm 容器中）

```bash
# 运行全套测试（113 个用例，约 1.5 秒）
docker exec -w /mnt/esfs/master_work/vllm-from-scratch vllm \
  python3 -m pytest 01_paged_attention/ 02_kvcache/ 03_moe/ 04_mla/ 05_mini_vllm/ 06_global_prefix_cache/ -v

# 运行单个模块演示
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/03_moe/mini_moe.py
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/04_mla/mini_mla.py
docker exec vllm python3 /mnt/esfs/master_work/vllm-from-scratch/05_mini_vllm/mini_vllm.py
```

## 各章核心内容速查

| 章节 | 关键技术 | 代码位置 |
|-----|---------|---------|
| 第1-2章 | vLLM 整体架构 | BLOG_PART1.md |
| 第3章 | Paged Attention，Block Table | `01_paged_attention/paged_attention.py` |
| 第4章 | KV Cache 存取、swap-in/swap-out | `01_paged_attention/paged_attention.py` |
| 第5章 | 完整推理路径 | BLOG_PART1.md |
| 第6章 | Prefix Cache（单机），链式哈希，LRU | `02_kvcache/block_pool_lru.py` |
| 第7章 | Prefix Cache（全局池化，Mooncake 风格） | `06_global_prefix_cache/global_kv_pool.py` |
| 第8章 | Scheduler，FCFS，抢占 | BLOG_PART2.md |
| 第9章 | 投机解码，EAGLE，rejection sampling | BLOG_PART2.md |
| 第10章 | Chunked Prefill，混合批处理 | BLOG_PART2.md + `05_mini_vllm/mini_vllm.py` |
| 第11章 | DeepSeek MoE，GroupedTopK路由 | `03_moe/mini_moe.py` |
| 第12章 | MLA，低秩KV，解耦RoPE | `04_mla/mini_mla.py` |
| 第13章 | PD 分离，KV Transfer | BLOG_PART3.md |
| 第14章 | V1 引擎架构，全局视角 | BLOG_PART3.md |
| 第15章 | Mini vLLM 完整整合 | `05_mini_vllm/mini_vllm.py` |

## 关键数字

| 优化 | 收益 |
|-----|------|
| Paged Attention | 内存碎片从 60-80% → <4% |
| Prefix Cache | 相同系统提示词：跳过所有重复 prefill |
| 投机解码 | Decode 速度 1.5-5x（视接受率） |
| Chunked Prefill | TTFT 降低 20-40% |
| MoE (K/N 激活) | 等效参数量下计算量降低 5-10x |
| MLA | KV Cache 内存节省 7-57x（vs MHA） |
| PD 分离 | Decode 节点吞吐可独立扩展 |

## 参考源码

vLLM 源码位于：`/mnt/esfs/master_work/vllm/`

关键文件：
- `vllm/v1/core/sched/scheduler.py` —— V1 调度器
- `vllm/v1/spec_decode/eagle.py` —— EAGLE 投机解码
- `vllm/v1/sample/rejection_sampler.py` —— Rejection Sampling
- `vllm/model_executor/models/deepseek_v2.py` —— DeepSeek MoE
- `vllm/model_executor/layers/fused_moe/` —— FusedMoE Triton 内核
- `vllm/model_executor/layers/mla.py` —— MLA 实现
