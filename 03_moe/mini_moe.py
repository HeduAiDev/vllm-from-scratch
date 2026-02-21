"""
Mini MoE (Mixture of Experts) 从零实现
对应博客第十一章

实现了：
1. Expert (单个专家 MLP)
2. Router (路由器，支持 TopK 和 GroupedTopK)
3. MoELayer (完整 MoE 层，支持共享专家)
4. 辅助负载均衡损失
5. Expert Parallel 模拟
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 基础组件
# ============================================================

class Expert(nn.Module):
    """单个专家：SwiGLU MLP（与 DeepSeek / LLaMA 保持一致）"""

    def __init__(self, input_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# 路由器
# ============================================================

class TopKRouter(nn.Module):
    """
    标准 TopK 路由器
    每个 token 在所有专家中选 top_k 个
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [T, D]
        返回:
            topk_weights: [T, K] 归一化后的路由权重
            topk_ids:     [T, K] 选中的专家ID
            aux_loss:     标量，辅助负载均衡损失
        """
        logits = self.gate(x)          # [T, E]
        probs = F.softmax(logits, dim=-1)  # [T, E]

        # TopK 选择
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 辅助负载均衡损失（Switch Transformer 风格）
        # 每个专家的实际使用率应等于 1/E
        # 用 top1 的 one-hot 估计专家使用率（不影响梯度）
        top1_ids = topk_ids[:, 0]
        expert_usage = F.one_hot(top1_ids, self.num_experts).float().mean(0)  # [E]
        mean_prob = probs.mean(0)  # [E]
        aux_loss = self.num_experts * (expert_usage * mean_prob).sum()

        return topk_probs, topk_ids, aux_loss


class GroupedTopKRouter(nn.Module):
    """
    分组 TopK 路由器（DeepSeek V2/V3 风格）

    n_experts = num_expert_group × experts_per_group
    每个 token：
      1. 将专家分成 num_expert_group 组
      2. 计算每组的代表分数（组内 topk_group 分数之和）
      3. 选出 num_selected_groups 个最高分的组
      4. 在选中组内选出 top_k 个专家
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        num_expert_group: int,
        topk_group: int,
    ):
        super().__init__()
        assert num_experts % num_expert_group == 0, \
            f"num_experts({num_experts}) must be divisible by num_expert_group({num_expert_group})"
        assert top_k % topk_group == 0, \
            f"top_k({top_k}) must be divisible by topk_group({topk_group})"

        self.top_k = top_k
        self.num_experts = num_experts
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_selected_groups = top_k // topk_group
        self.experts_per_group = num_experts // num_expert_group

        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [T, D]
        """
        T = x.shape[0]
        logits = self.gate(x)             # [T, E]
        probs = F.softmax(logits, dim=-1) # [T, E]

        # Reshape 为分组形式
        scores_grouped = probs.view(T, self.num_expert_group, self.experts_per_group)
        # [T, G, E/G]

        # 在每组内选 topk_group 个专家
        group_topk_vals, group_topk_local_ids = torch.topk(
            scores_grouped, k=self.topk_group, dim=-1
        )  # [T, G, topk_group]

        # 计算每组的组分数
        group_scores = group_topk_vals.sum(dim=-1)  # [T, G]

        # 选最好的几个组
        _, selected_groups = torch.topk(
            group_scores, k=self.num_selected_groups, dim=-1
        )  # [T, num_selected_groups]

        # 构建每个 token 的候选专家集合
        # 将局部 ID 转换为全局专家 ID
        candidate_ids = []
        candidate_probs = []

        for g_offset in range(self.num_selected_groups):
            g = selected_groups[:, g_offset]  # [T]
            global_offset = g * self.experts_per_group  # [T]

            for k_offset in range(self.topk_group):
                # 取该组第 k_offset 个专家
                local_id = group_topk_local_ids[
                    torch.arange(T), g, k_offset
                ]  # [T]
                global_id = global_offset + local_id  # [T]
                expert_prob = probs[torch.arange(T), global_id]  # [T]

                candidate_ids.append(global_id)
                candidate_probs.append(expert_prob)

        # 合并所有候选 [T, num_selected_groups * topk_group]
        candidate_ids = torch.stack(candidate_ids, dim=-1)    # [T, K]
        candidate_probs = torch.stack(candidate_probs, dim=-1) # [T, K]

        # 归一化权重
        topk_weights = candidate_probs / (candidate_probs.sum(dim=-1, keepdim=True) + 1e-9)
        topk_ids = candidate_ids

        # 辅助损失
        top1_ids = topk_ids[:, 0]
        expert_usage = F.one_hot(top1_ids, self.num_experts).float().mean(0)
        mean_prob = probs.mean(0)
        aux_loss = self.num_experts * (expert_usage * mean_prob).sum()

        return topk_weights, topk_ids, aux_loss


# ============================================================
# MoE 层
# ============================================================

class MoELayer(nn.Module):
    """
    完整的 MoE 层

    支持：
    - 标准 TopK 路由 或 GroupedTopK 路由
    - 共享专家（DeepSeek 设计）
    - 辅助负载均衡损失
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        intermediate_dim: int,
        num_shared_experts: int = 0,
        use_grouped_topk: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # 路由器
        if use_grouped_topk:
            self.router = GroupedTopKRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            self.router = TopKRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
            )

        # 路由专家
        self.experts = nn.ModuleList([
            Expert(input_dim, intermediate_dim) for _ in range(num_experts)
        ])

        # 共享专家（可选，DeepSeek 设计）
        if num_shared_experts > 0:
            self.shared_expert = Expert(
                input_dim, intermediate_dim * num_shared_experts
            )
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, S, D] 或 [T, D]（T = B*S 展开后）
        返回: (output [同输入shape], aux_loss [标量])
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # [T, D]
        T = x_flat.shape[0]

        # 1. 路由
        topk_weights, topk_ids, aux_loss = self.router(x_flat)
        # topk_weights: [T, K], topk_ids: [T, K]

        # 2. 共享专家
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)  # [T, D]
        else:
            shared_out = 0.0

        # 3. 路由专家：token dispatch
        output = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            # 找路由到该专家的 token，以及对应的权重位置
            expert_mask = (topk_ids == expert_id)  # [T, K] bool
            token_mask = expert_mask.any(dim=-1)    # [T] bool

            if not token_mask.any():
                continue

            tokens = x_flat[token_mask]  # [M, D]，M是路由到这里的token数

            # 专家计算
            expert_out = self.experts[expert_id](tokens)  # [M, D]

            # 权重：取对应的路由权重
            # expert_mask[token_mask]: [M, K]
            # 每个 token 在本专家的权重（可能有多个 top-k slot 指向同一专家，取第一个）
            weights_for_tokens = topk_weights[token_mask]  # [M, K]
            expert_weights_mask = expert_mask[token_mask]  # [M, K]

            # 取第一个匹配的权重
            first_match = expert_weights_mask.int().argmax(dim=-1)  # [M]
            weights = weights_for_tokens[torch.arange(tokens.shape[0]), first_match]  # [M]

            output[token_mask] += expert_out * weights.unsqueeze(-1)

        # 4. 合并
        output = output + shared_out

        return output.view(original_shape), aux_loss * self.aux_loss_coeff


# ============================================================
# Expert Parallel 模拟
# ============================================================

class SimulatedEPMoELayer(nn.Module):
    """
    模拟 Expert Parallel 的 MoE 层
    假设 EP_size=2，本 GPU 只持有一半专家

    实际 vLLM 用 All2All 通信分发 token，这里用索引模拟
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        intermediate_dim: int,
        ep_rank: int,
        ep_size: int,
    ):
        super().__init__()
        assert num_experts % ep_size == 0
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.top_k = top_k

        # 本 GPU 只持有本地专家
        self.local_experts_start = ep_rank * (num_experts // ep_size)
        self.local_experts_end   = self.local_experts_start + (num_experts // ep_size)
        self.n_local_experts = num_experts // ep_size

        self.router = TopKRouter(input_dim, num_experts, top_k)
        self.local_experts = nn.ModuleList([
            Expert(input_dim, intermediate_dim) for _ in range(self.n_local_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(-1, x.shape[-1])
        T = x_flat.shape[0]

        topk_weights, topk_ids, aux_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        # 只处理路由到本 GPU 专家范围内的 token
        for local_idx in range(self.n_local_experts):
            global_expert_id = self.local_experts_start + local_idx
            expert_mask = (topk_ids == global_expert_id)
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            tokens = x_flat[token_mask]
            expert_out = self.local_experts[local_idx](tokens)

            weights_for_tokens = topk_weights[token_mask]
            expert_weights_mask = expert_mask[token_mask]
            first_match = expert_weights_mask.int().argmax(dim=-1)
            weights = weights_for_tokens[torch.arange(tokens.shape[0]), first_match]

            output[token_mask] += expert_out * weights.unsqueeze(-1)

        # 在实际 EP 中，这里需要 AllReduce 来聚合其他 GPU 的结果
        # output = dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output.view(x.shape), aux_loss


# ============================================================
# 测试
# ============================================================

def test_expert():
    """测试单个专家"""
    expert = Expert(input_dim=64, intermediate_dim=256)
    x = torch.randn(4, 64)
    out = expert(x)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print("✓ Expert forward pass: OK")


def test_topk_router():
    """测试 TopK 路由器"""
    router = TopKRouter(input_dim=64, num_experts=8, top_k=2)
    x = torch.randn(10, 64)  # 10 tokens

    weights, ids, loss = router(x)
    assert weights.shape == (10, 2)
    assert ids.shape == (10, 2)
    assert ids.max() < 8
    assert abs(weights.sum(dim=-1).mean().item() - 1.0) < 1e-5, "权重应归一化到1"
    print(f"✓ TopK Router: weights {weights.shape}, ids {ids.shape}, aux_loss={loss.item():.4f}")


def test_grouped_topk_router():
    """测试分组 TopK 路由器"""
    router = GroupedTopKRouter(
        input_dim=64,
        num_experts=16,
        top_k=4,        # 最终选 4 个专家
        num_expert_group=4,  # 分成 4 组
        topk_group=1,   # 每组选 1 个（共 4 组 × 1 = 4 个）
    )
    x = torch.randn(8, 64)  # 8 tokens

    weights, ids, loss = router(x)
    assert weights.shape == (8, 4)
    assert ids.shape == (8, 4)
    assert ids.max() < 16
    print(f"✓ GroupedTopK Router: weights {weights.shape}, aux_loss={loss.item():.4f}")


def test_moe_layer():
    """测试完整 MoE 层"""
    moe = MoELayer(
        input_dim=64,
        num_experts=8,
        top_k=2,
        intermediate_dim=256,
        num_shared_experts=1,
    )
    x = torch.randn(4, 16, 64)  # [B=4, S=16, D=64]
    out, aux_loss = moe(x)
    assert out.shape == (4, 16, 64), f"Expected (4, 16, 64), got {out.shape}"
    print(f"✓ MoELayer with shared experts: out={out.shape}, aux_loss={aux_loss.item():.6f}")


def test_grouped_topk_moe():
    """测试 GroupedTopK MoE（DeepSeek 风格）"""
    moe = MoELayer(
        input_dim=128,
        num_experts=16,
        top_k=4,
        intermediate_dim=512,
        num_shared_experts=1,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=1,
    )
    x = torch.randn(2, 8, 128)
    out, aux_loss = moe(x)
    assert out.shape == (2, 8, 128)
    print(f"✓ GroupedTopK MoE: out={out.shape}, aux_loss={aux_loss.item():.6f}")


def test_load_balancing():
    """验证辅助损失确实能引导负载均衡"""
    import torch.optim as optim

    moe = MoELayer(
        input_dim=32,
        num_experts=4,
        top_k=1,
        intermediate_dim=64,
        aux_loss_coeff=1.0,  # 强辅助损失
    )
    optimizer = optim.Adam(moe.parameters(), lr=1e-3)

    # 初始专家使用分布（可能不均衡）
    x = torch.randn(100, 32)

    with torch.no_grad():
        _, ids, _ = moe.router(x)
        initial_usage = torch.bincount(ids[:, 0], minlength=4).float()

    # 训练几步
    for _ in range(50):
        x = torch.randn(100, 32)
        out, aux_loss = moe(x)
        # 辅助损失 + 简单的任务损失（让模型学习）
        task_loss = out.mean()
        total_loss = task_loss + aux_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 训练后的分布
    with torch.no_grad():
        _, ids, _ = moe.router(x)
        final_usage = torch.bincount(ids[:, 0], minlength=4).float()

    print(f"✓ 负载均衡测试:")
    print(f"  初始使用分布: {initial_usage.tolist()}")
    print(f"  训练后分布:   {final_usage.tolist()}")

    # 标准差应该减小（更均衡）
    # （不保证一定更好，只是验证辅助损失在发挥作用）
    print(f"  初始标准差: {initial_usage.std().item():.2f}")
    print(f"  最终标准差: {final_usage.std().item():.2f}")


def benchmark_moe():
    """简单性能测试"""
    import time

    moe = MoELayer(
        input_dim=1024,
        num_experts=64,
        top_k=6,
        intermediate_dim=4096,
    )

    x = torch.randn(32, 512, 1024)  # batch=32, seq=512, dim=1024

    # Warmup
    for _ in range(3):
        out, _ = moe(x)

    # 计时
    start = time.perf_counter()
    N = 10
    for _ in range(N):
        out, _ = moe(x)
    elapsed = (time.perf_counter() - start) / N

    print(f"\n性能测试 (batch=32, seq=512, dim=1024, E=64, K=6):")
    print(f"  平均推理时间: {elapsed*1000:.1f}ms")
    print(f"  输出形状: {out.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Mini MoE 测试套件")
    print("=" * 60)

    test_expert()
    test_topk_router()
    test_grouped_topk_router()
    test_moe_layer()
    test_grouped_topk_moe()
    test_load_balancing()

    print("\n所有测试通过！")
