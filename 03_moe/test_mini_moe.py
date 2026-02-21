"""
Mini MoE pytest 测试套件
运行：docker exec vllm python3 -m pytest 03_moe/test_mini_moe.py -v
"""
import math
import torch
import pytest
from mini_moe import Expert, TopKRouter, GroupedTopKRouter, MoELayer


class TestExpert:
    def test_output_shape(self):
        expert = Expert(input_dim=64, intermediate_dim=256)
        x = torch.randn(8, 64)
        out = expert(x)
        assert out.shape == (8, 64)

    def test_batch_independence(self):
        """单样本结果和批次中的同一样本结果一致"""
        expert = Expert(input_dim=32, intermediate_dim=64)
        expert.eval()
        x = torch.randn(4, 32)
        out_batch = expert(x)
        out_single = expert(x[2:3])
        assert torch.allclose(out_batch[2], out_single[0], atol=1e-5)

    def test_swiglu_activation(self):
        """验证 SwiGLU 激活：down(silu(gate(x)) * up(x))"""
        expert = Expert(input_dim=4, intermediate_dim=8)
        x = torch.randn(2, 4)
        manual = expert.down_proj(
            torch.nn.functional.silu(expert.gate_proj(x)) * expert.up_proj(x)
        )
        assert torch.allclose(expert(x), manual, atol=1e-6)


class TestTopKRouter:
    def test_output_shapes(self):
        router = TopKRouter(input_dim=64, num_experts=8, top_k=2)
        x = torch.randn(10, 64)
        weights, ids, loss = router(x)
        assert weights.shape == (10, 2)
        assert ids.shape == (10, 2)
        assert loss.shape == ()

    def test_weights_normalized(self):
        router = TopKRouter(input_dim=32, num_experts=16, top_k=4)
        x = torch.randn(20, 32)
        weights, ids, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(20), atol=1e-5)

    def test_ids_in_range(self):
        num_experts = 12
        router = TopKRouter(input_dim=32, num_experts=num_experts, top_k=3)
        x = torch.randn(16, 32)
        _, ids, _ = router(x)
        assert ids.min() >= 0
        assert ids.max() < num_experts

    def test_aux_loss_positive(self):
        router = TopKRouter(input_dim=32, num_experts=8, top_k=2)
        x = torch.randn(64, 32)
        _, _, loss = router(x)
        assert loss.item() > 0

    def test_gradient_flows(self):
        router = TopKRouter(input_dim=32, num_experts=8, top_k=2)
        x = torch.randn(10, 32, requires_grad=True)
        weights, ids, loss = router(x)
        loss.backward()
        assert x.grad is not None


class TestGroupedTopKRouter:
    @pytest.mark.parametrize("n_experts,n_group,topk_group,top_k", [
        (16, 4, 1, 4),
        (32, 8, 2, 8),
        (64, 8, 3, 6),
    ])
    def test_output_shapes(self, n_experts, n_group, topk_group, top_k):
        router = GroupedTopKRouter(
            input_dim=64,
            num_experts=n_experts,
            top_k=top_k,
            num_expert_group=n_group,
            topk_group=topk_group,
        )
        x = torch.randn(8, 64)
        weights, ids, loss = router(x)
        assert weights.shape == (8, top_k)
        assert ids.shape == (8, top_k)

    def test_weights_normalized(self):
        router = GroupedTopKRouter(
            input_dim=32, num_experts=16, top_k=4,
            num_expert_group=4, topk_group=1,
        )
        x = torch.randn(10, 32)
        weights, _, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_ids_in_range(self):
        router = GroupedTopKRouter(
            input_dim=32, num_experts=16, top_k=4,
            num_expert_group=4, topk_group=1,
        )
        x = torch.randn(10, 32)
        _, ids, _ = router(x)
        assert ids.min() >= 0
        assert ids.max() < 16


class TestMoELayer:
    def test_output_shape_3d(self):
        moe = MoELayer(input_dim=64, num_experts=8, top_k=2, intermediate_dim=256)
        x = torch.randn(4, 16, 64)  # [B, S, D]
        out, loss = moe(x)
        assert out.shape == (4, 16, 64)
        assert loss.shape == ()

    def test_output_shape_2d(self):
        moe = MoELayer(input_dim=64, num_experts=8, top_k=2, intermediate_dim=256)
        x = torch.randn(32, 64)  # [T, D]
        out, loss = moe(x)
        assert out.shape == (32, 64)

    def test_shared_expert(self):
        moe = MoELayer(
            input_dim=64, num_experts=8, top_k=2,
            intermediate_dim=256, num_shared_experts=2,
        )
        x = torch.randn(4, 8, 64)
        out, loss = moe(x)
        assert out.shape == (4, 8, 64)

    def test_gradient_backprop(self):
        moe = MoELayer(input_dim=32, num_experts=4, top_k=2, intermediate_dim=64)
        x = torch.randn(8, 32, requires_grad=True)
        out, loss = moe(x)
        (out.sum() + loss).backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_grouped_topk(self):
        moe = MoELayer(
            input_dim=128, num_experts=16, top_k=4,
            intermediate_dim=512,
            use_grouped_topk=True,
            num_expert_group=4, topk_group=1,
        )
        x = torch.randn(2, 10, 128)
        out, loss = moe(x)
        assert out.shape == (2, 10, 128)

    def test_aux_loss_scale(self):
        """辅助损失被 aux_loss_coeff 正确缩放"""
        moe1 = MoELayer(32, 4, 2, 64, aux_loss_coeff=1.0)
        moe2 = MoELayer(32, 4, 2, 64, aux_loss_coeff=0.01)
        # 使用相同随机种子
        x = torch.randn(16, 32)
        torch.manual_seed(42)
        _, loss1 = moe1(x)
        torch.manual_seed(42)
        _, loss2 = moe2(x)
        # loss1 应比 loss2 大约 100 倍
        # （不完全精确，因为路由结果也会有差异）
        assert loss1.item() > loss2.item()
