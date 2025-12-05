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
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        spike_trains: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算reward向量（支持活动加权的群体编码）

        Args:
            logits: [B, num_classes] 网络输出logits
            labels: [B] 真实标签
            spike_trains: [T, B, num_classes, N] 或 [T, B, num_classes]
                         可选，用于活动加权群体编码

        Returns:
            reward: [B, num_classes] 如果N=1（向后兼容）
                   [B, num_classes, N] 如果N>1（活动加权的每个神经元）
        """
        B = logits.shape[0]
        device = logits.device

        # 向后兼容：无spike_trains或spike_trains是3D
        if spike_trains is None or spike_trains.ndim == 3:
            # 旧行为：统一reward
            num_classes = logits.shape[1]
            reward = torch.ones((B, num_classes), device=device) * -1.0
            batch_idx = torch.arange(B, device=device)
            reward[batch_idx, labels] = 1.0
            return reward

        # 群体编码模式：spike_trains是 [T, B, num_classes, N]
        T, B, num_classes, N = spike_trains.shape

        # 计算每个神经元的脉冲率：对时间维度求和
        spike_rates = spike_trains.sum(dim=0)  # [B, num_classes, N]

        # 在每个群体内归一化以获得活动权重
        # 确保每个群体的权重和为N（保持总reward幅度）
        population_sum = spike_rates.sum(dim=2, keepdim=True) + 1e-8  # [B, num_classes, 1]
        activity_weights = spike_rates / population_sum * N  # [B, num_classes, N]

        # 基础reward：正确类+1，错误类-1
        base_reward = torch.ones((B, num_classes, 1), device=device) * -1.0
        batch_idx = torch.arange(B, device=device)
        base_reward[batch_idx, labels, :] = 1.0  # [B, num_classes, 1]

        # 应用活动加权
        reward = base_reward * activity_weights  # [B, num_classes, N]

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
        post_spikes_sequence: torch.Tensor,  # [T, B, out_features] 或 [T, B, num_classes, N]
        logits: torch.Tensor,  # [B, num_classes]
        labels: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        使用Reward调制的STDP更新权重（支持群体编码）

        Δw_{ij} = E_{ij} × R[i] × softmax_prob[i]

        群体编码时:
        - post_spikes_sequence: [T, B, num_classes, N]
        - reward: [B, num_classes, N] (活动加权的每个神经元)

        其中:
        - E_{ij}: STDP计算的权重变化
        - R[i]: 输出神经元i的reward (活动加权)
        - softmax_prob[i]: 输出神经元i对应类别的softmax概率（扩展到N个神经元）

        Args:
            weight: [out_features, in_features] 权重矩阵
            pre_spikes_sequence: [T, B, in_features] 突触前脉冲序列
            post_spikes_sequence: [T, B, out_features] 或 [T, B, num_classes, N] 突触后脉冲序列
            logits: [B, num_classes] 网络输出
            labels: [B] 真实标签

        Returns:
            weight_delta: [out_features, in_features] 权重变化
        """
        T, B, pre_size = pre_spikes_sequence.shape
        device = weight.device

        # 处理群体编码：展平后两个维度
        if post_spikes_sequence.ndim == 4:
            # 群体编码模式: [T, B, num_classes, N]
            T_check, B_check, num_classes, N = post_spikes_sequence.shape
            post_spikes_flat = post_spikes_sequence.reshape(T, B, num_classes * N)
            out_features = num_classes * N

            # 计算活动加权reward
            reward = self.compute_reward(logits, labels, spike_trains=post_spikes_sequence)
            # reward: [B, num_classes, N], 展平为 [B, out_features]
            reward_flat = reward.reshape(B, out_features)
        else:
            # 向后兼容: [T, B, out_features]
            post_spikes_flat = post_spikes_sequence
            out_features = post_spikes_sequence.shape[2]

            # 统一reward（无活动加权）
            reward = self.compute_reward(logits, labels, spike_trains=None)
            reward_flat = reward  # [B, num_classes]

        # 重置traces - 传递目标设备
        self.stdp_learner.reset(B, pre_size, out_features, device)

        # 在时间维度上累积STDP权重变化
        total_weight_change = torch.zeros_like(weight)

        for t in range(T):
            pre_spikes = pre_spikes_sequence[t]  # [B, in_features]
            post_spikes = post_spikes_flat[t]  # [B, out_features]

            # 计算该时间步的STDP权重变化 [in_features, out_features]
            dw_t = self.stdp_learner.compute_weight_change(pre_spikes, post_spikes)
            # 转置为 [out_features, in_features] 以匹配权重矩阵形状
            total_weight_change += dw_t.T

        # 计算softmax概率并扩展到每个神经元
        softmax_prob = self.compute_softmax_probability(logits)  # [B, num_classes]

        # 扩展softmax_prob以匹配reward_flat的形状
        if post_spikes_sequence.ndim == 4:
            # 群体编码: 每个类的概率复制N次
            # [B, num_classes] -> [B, num_classes, N] -> [B, out_features]
            softmax_prob_expanded = softmax_prob.unsqueeze(2).expand(B, num_classes, N).reshape(B, out_features)
        else:
            softmax_prob_expanded = softmax_prob

        # Reward调制: 按输出神经元索引加权（每个神经元独立）
        # reward_weight: [out_features] = mean_B(reward_flat[b, i] * softmax_prob_expanded[b, i])
        reward_weight = (reward_flat * softmax_prob_expanded).mean(dim=0)  # [out_features]

        # 应用reward调制到权重变化
        # total_weight_change: [out_features, in_features]
        # reward_weight: [out_features] -> unsqueeze(1) -> [out_features, 1]
        # broadcasting: [out_features, in_features] * [out_features, 1] -> [out_features, in_features]
        weight_delta = total_weight_change * reward_weight.unsqueeze(1)

        # 记录统计信息
        self.weight_change_history.append(weight_delta.detach().clone())
        self.reward_stats["positive"] = (reward_flat > 0).sum().item()
        self.reward_stats["negative"] = (reward_flat < 0).sum().item()

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
