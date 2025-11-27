"""
Spike-Timing-Dependent Plasticity (STDP) with Reward Modulation
用于SNN意图分类头的本地无监督学习规则和reward调制机制
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class STDPTrace:
    """管理pre和post-synaptic traces"""

    def __init__(self, tau_pre: float = 10.0, tau_post: float = 10.0):
        """
        Args:
            tau_pre: pre-synaptic trace时间常数
            tau_post: post-synaptic trace时间常数
        """
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # 衰减因子（CPU上创建，后续动态移动到正确设备）
        self.decay_pre = torch.exp(torch.tensor(-1.0 / tau_pre))
        self.decay_post = torch.exp(torch.tensor(-1.0 / tau_post))

        # traces初始化
        self.x_pre = None  # pre-synaptic trace
        self.x_post = None  # post-synaptic trace

    def reset(self, batch_size: int, pre_size: int, post_size: int, device: torch.device):
        """重置traces

        Args:
            batch_size: 批量大小
            pre_size: 突触前神经元数量
            post_size: 突触后神经元数量
            device: 目标设备
        """
        self.x_pre = torch.zeros((batch_size, pre_size), device=device)
        self.x_post = torch.zeros((batch_size, post_size), device=device)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        更新pre和post-synaptic traces

        Args:
            pre_spikes: [B, pre_size] 突触前脉冲
            post_spikes: [B, post_size] 突触后脉冲
        """
        # 确保衰减因子在正确的设备上
        device = pre_spikes.device
        decay_pre = self.decay_pre.to(device)
        decay_post = self.decay_post.to(device)

        # 衰减并添加新spike
        self.x_pre = decay_pre * self.x_pre + pre_spikes
        self.x_post = decay_post * self.x_post + post_spikes

    def get_traces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前traces"""
        return self.x_pre, self.x_post


class STDPLearner:
    """标准STDP学习规则"""

    def __init__(
        self,
        A_pre: float = 0.01,  # LTP幅度
        A_post: float = -0.01,  # LTD幅度
        tau_pre: float = 10.0,
        tau_post: float = 10.0,
    ):
        """
        Args:
            A_pre: LTP幅度（正）
            A_post: LTD幅度（负）
            tau_pre: pre-synaptic trace时间常数
            tau_post: post-synaptic trace时间常数
        """
        self.A_pre = A_pre
        self.A_post = A_post
        self.trace = STDPTrace(tau_pre=tau_pre, tau_post=tau_post)

    def compute_weight_change(
        self,
        pre_spikes: torch.Tensor,  # [B, in_features]
        post_spikes: torch.Tensor,  # [B, out_features]
    ) -> torch.Tensor:
        """
        计算STDP权重变化矩阵

        标准STDP规则:
        Δw = A_pre * pre_spike * post_trace + A_post * post_spike * pre_trace

        Args:
            pre_spikes: [B, in_features]
            post_spikes: [B, out_features]

        Returns:
            dw: [in_features, out_features] 权重变化矩阵
        """
        B = pre_spikes.shape[0]

        # 更新traces
        self.trace.update(pre_spikes, post_spikes)
        x_pre, x_post = self.trace.get_traces()

        # 计算权重变化 (Hebbian + anti-Hebbian)
        # dw_{ij} = A_pre * x_pre_i * post_spikes_j + A_post * pre_spikes_i * x_post_j
        dw = self.A_pre * torch.outer(x_pre.mean(dim=0), post_spikes.mean(dim=0)) + \
             self.A_post * torch.outer(pre_spikes.mean(dim=0), x_post.mean(dim=0))

        return dw

    def reset(self, batch_size: int, pre_size: int, post_size: int, device: torch.device):
        """重置traces

        Args:
            batch_size: 批量大小
            pre_size: 突触前神经元数量
            post_size: 突触后神经元数量
            device: 目标设备
        """
        self.trace.reset(batch_size, pre_size, post_size, device)


class RewardModulatedSTDPUpdater:
    """Reward调制的STDP更新器"""

    def __init__(
        self,
        learning_rate: float = 0.001,
        A_pre: float = 0.01,
        A_post: float = -0.01,
        tau_pre: float = 10.0,
        tau_post: float = 10.0,
    ):
        """
        Args:
            learning_rate: STDP学习率
            A_pre: LTP幅度
            A_post: LTD幅度
            tau_pre: pre-synaptic trace时间常数
            tau_post: post-synaptic trace时间常数
        """
        self.learning_rate = learning_rate
        self.stdp_learner = STDPLearner(
            A_pre=A_pre, A_post=A_post, tau_pre=tau_pre, tau_post=tau_post
        )

        # 统计信息
        self.weight_change_history = []
        self.reward_stats = {"positive": 0, "negative": 0}

    def compute_reward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算reward向量

        Args:
            logits: [B, num_classes] 网络输出logits
            labels: [B] 真实标签

        Returns:
            reward: [B, num_classes] reward向量
                   正确类对应+1，错误类对应-1
        """
        B, num_classes = logits.shape
        device = logits.device
        reward = torch.ones((B, num_classes), device=device) * -1.0

        # 设置正确类的reward为+1
        batch_idx = torch.arange(B, device=device)
        reward[batch_idx, labels] = 1.0

        return reward

    def compute_softmax_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算softmax概率

        Args:
            logits: [B, num_classes]

        Returns:
            prob: [B, num_classes]
        """
        return torch.softmax(logits, dim=-1)

    def update_weight(
        self,
        weight: torch.Tensor,  # [out_features, in_features]
        pre_spikes_sequence: torch.Tensor,  # [T, B, in_features]
        post_spikes_sequence: torch.Tensor,  # [T, B, out_features]
        logits: torch.Tensor,  # [B, num_classes]
        labels: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        使用Reward调制的STDP更新权重

        Δw_{ij} = E_{ij} × R[i] × softmax_prob[i]

        其中:
        - E_{ij}: STDP计算的权重变化
        - R[i]: 输出神经元i的reward (+1正确/-1错误)
        - softmax_prob[i]: 输出神经元i的softmax概率

        Args:
            weight: [out_features, in_features] 权重矩阵
            pre_spikes_sequence: [T, B, in_features] 突触前脉冲序列
            post_spikes_sequence: [T, B, out_features] 突触后脉冲序列
            logits: [B, num_classes] 网络输出
            labels: [B] 真实标签

        Returns:
            weight_delta: [out_features, in_features] 权重变化
        """
        T, B, pre_size = pre_spikes_sequence.shape
        out_features = post_spikes_sequence.shape[-1]
        device = weight.device

        # 重置traces - 传递目标设备
        self.stdp_learner.reset(B, pre_size, out_features, device)

        # 在时间维度上累积STDP权重变化
        total_weight_change = torch.zeros_like(weight)

        for t in range(T):
            pre_spikes = pre_spikes_sequence[t]  # [B, in_features]
            post_spikes = post_spikes_sequence[t]  # [B, out_features]

            # 计算该时间步的STDP权重变化 [in_features, out_features]
            dw_t = self.stdp_learner.compute_weight_change(pre_spikes, post_spikes)
            # 转置为 [out_features, in_features] 以匹配权重矩阵形状
            total_weight_change += dw_t.T

        # 计算reward和softmax概率
        reward = self.compute_reward(logits, labels)  # [B, num_classes]
        softmax_prob = self.compute_softmax_probability(logits)  # [B, num_classes]

        # Reward调制: 按输出神经元索引加权
        # reward_weight: [out_features] = mean_B(reward[b, i] * softmax_prob[b, i])
        reward_weight = (reward * softmax_prob).mean(dim=0)  # [out_features]

        # 应用reward调制到权重变化
        # total_weight_change: [out_features, in_features]
        # reward_weight: [out_features]
        # unsqueeze(1) -> [out_features, 1]
        # broadcasting: [out_features, in_features] * [out_features, 1] -> [out_features, in_features]
        weight_delta = total_weight_change * reward_weight.unsqueeze(1)

        # 记录统计信息
        self.weight_change_history.append(weight_delta.detach().clone())
        self.reward_stats["positive"] = (reward == 1.0).sum().item()
        self.reward_stats["negative"] = (reward == -1.0).sum().item()

        return weight_delta

    def apply_update(self, weight: torch.Tensor, weight_delta: torch.Tensor) -> torch.Tensor:
        """
        应用权重更新

        Args:
            weight: [out_features, in_features]
            weight_delta: [out_features, in_features]

        Returns:
            updated_weight: 更新后的权重
        """
        return weight + self.learning_rate * weight_delta

    def get_metrics(self) -> Dict:
        """获取STDP训练指标"""
        if not self.weight_change_history:
            return {"weight_change_mean": 0.0, "weight_change_std": 0.0}

        weight_changes = torch.stack(self.weight_change_history)
        return {
            "weight_change_mean": weight_changes.abs().mean().item(),
            "weight_change_std": weight_changes.std().item(),
            "positive_rewards": self.reward_stats["positive"],
            "negative_rewards": self.reward_stats["negative"],
        }

    def reset_metrics(self):
        """重置指标"""
        self.weight_change_history = []
        self.reward_stats = {"positive": 0, "negative": 0}
