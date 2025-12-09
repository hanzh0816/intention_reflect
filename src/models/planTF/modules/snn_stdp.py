"""
Spike-Timing-Dependent Plasticity (STDP) with Reward Modulation
基于SpikingJelly STDPLearner的实现,去除群体编码,简化为MLP网络STDP
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable


class SpikingJellySTDPWrapper:
    """
    基于SpikingJelly STDP算法的包装器，添加reward调制

    特点:
    - 使用SpikingJelly的STDP trace更新公式
    - 手动管理spike数据（不使用monitors，更简单）
    - 支持reward调制用于监督学习
    - 去除群体编码逻辑，简化为1对1神经元映射
    """

    def __init__(
        self,
        layer: nn.Linear,
        neuron: nn.Module,
        learning_rate: float = 0.001,
        tau_pre: float = 10.0,
        tau_post: float = 10.0,
        f_pre: Optional[Callable] = None,
        f_post: Optional[Callable] = None,
    ):
        """
        Args:
            layer: 要训练的Linear层 (output_linear)
            neuron: 突触后神经元层 (output_lif) - 保留用于接口一致性
            learning_rate: STDP学习率
            tau_pre: pre-synaptic trace时间常数
            tau_post: post-synaptic trace时间常数
            f_pre: LTD权重边界函数 (默认为恒等函数)
            f_post: LTP权重边界函数 (默认为恒等函数)
        """
        self.learning_rate = learning_rate
        self.layer = layer
        self.neuron = neuron
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # 默认权重边界函数（无边界）
        if f_pre is None:
            self.f_pre = lambda x: torch.ones_like(x)
        else:
            self.f_pre = f_pre
        if f_post is None:
            self.f_post = lambda x: torch.ones_like(x)
        else:
            self.f_post = f_post

        # Traces
        self.trace_pre = None
        self.trace_post = None

        # 统计信息
        self.weight_change_history = []
        self.reward_stats = {"positive": 0, "negative": 0}

    def enable_monitors(self, pre_layer: nn.Module):
        """
        启用spike监控 (兼容接口，实际上我们手动传递spikes)

        Args:
            pre_layer: 突触前层 - 保留用于接口一致性
        """
        pass  # 不需要monitors，手动传递spikes

    def reset_traces(self, batch_size: int, pre_size: int, post_size: int, device: torch.device):
        """重置traces"""
        self.trace_pre = torch.zeros((batch_size, pre_size), device=device)
        self.trace_post = torch.zeros((batch_size, post_size), device=device)

    def update_traces(self, pre_spike: torch.Tensor, post_spike: torch.Tensor):
        """
        使用SpikingJelly的trace更新公式
        trace = trace - trace/tau + spike
        """
        # SpikingJelly trace更新: trace = trace - trace/tau + spike
        self.trace_pre = self.trace_pre - self.trace_pre / self.tau_pre + pre_spike
        self.trace_post = self.trace_post - self.trace_post / self.tau_post + post_spike

    def compute_stdp_weight_change(
        self,
        pre_spike: torch.Tensor,
        post_spike: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算单个时间步的STDP权重变化

        SpikingJelly公式:
        delta_w_post = f_post(w) * (trace_pre.T @ post_spike)
        delta_w_pre = -f_pre(w) * (trace_post.T @ pre_spike)
        delta_w = delta_w_post + delta_w_pre

        Args:
            pre_spike: [B, in_features]
            post_spike: [B, out_features]
            weight: [out_features, in_features]

        Returns:
            dw: [out_features, in_features]
        """
        # LTP: pre发生在post之前 -> 权重增加
        # f_post(w) * trace_pre^T @ post_spike
        # trace_pre: [B, in_features], post_spike: [B, out_features]
        # Result: [in_features, out_features] -> transpose -> [out_features, in_features]
        delta_w_post = self.f_post(weight) * torch.mm(post_spike.T, self.trace_pre)

        # LTD: post发生在pre之前 -> 权重减少
        # -f_pre(w) * trace_post^T @ pre_spike
        # trace_post: [B, out_features], pre_spike: [B, in_features]
        # Result: [out_features, in_features]
        delta_w_pre = -self.f_pre(weight) * torch.mm(self.trace_post.T, pre_spike)

        # 总的权重变化
        dw = delta_w_post + delta_w_pre

        return dw

    def compute_reward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算reward向量 (简化版，无群体编码)

        Args:
            logits: [B, num_classes] 网络输出logits
            labels: [B] 真实标签

        Returns:
            reward: [B, num_classes] 每个类的reward
        """
        B, num_classes = logits.shape
        device = logits.device

        # Binary reward: 正确类+1, 错误类-1
        reward = torch.ones((B, num_classes), device=device) * -1.0
        batch_idx = torch.arange(B, device=device)
        reward[batch_idx, labels] = 1.0

        return reward

    def update_weight(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pre_spikes_sequence: torch.Tensor,
        post_spikes_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用reward调制的STDP更新权重

        Args:
            logits: [B, num_classes]
            labels: [B]
            pre_spikes_sequence: [T, B, in_features] 突触前脉冲序列
            post_spikes_sequence: [T, B, out_features] 突触后脉冲序列

        Returns:
            weight_delta: [out_features, in_features] 权重变化
        """
        T, B, in_features = pre_spikes_sequence.shape
        _, _, out_features = post_spikes_sequence.shape
        device = self.layer.weight.device

        # 重置traces
        self.reset_traces(B, in_features, out_features, device)

        # 累积所有时间步的STDP权重变化
        total_weight_change = torch.zeros_like(self.layer.weight.data)

        for t in range(T):
            pre_spike = pre_spikes_sequence[t]  # [B, in_features]
            post_spike = post_spikes_sequence[t]  # [B, out_features]

            # 计算该时间步的STDP权重变化
            dw_t = self.compute_stdp_weight_change(
                pre_spike, post_spike, self.layer.weight.data
            )

            # 累积
            total_weight_change += dw_t

            # 更新traces
            self.update_traces(pre_spike, post_spike)

        # 计算reward信号
        reward = self.compute_reward(logits, labels)  # [B, num_classes]

        # 计算softmax概率用于调制
        softmax_prob = torch.softmax(logits, dim=-1)  # [B, num_classes]

        # Reward调制: 按输出神经元索引加权
        reward_weight = (reward * softmax_prob).mean(dim=0)  # [num_classes]

        # 应用reward调制到权重变化
        weight_delta = total_weight_change * reward_weight.unsqueeze(1)

        # 记录统计信息
        self.weight_change_history.append(weight_delta.detach().clone())
        self.reward_stats["positive"] = (reward > 0).sum().item()
        self.reward_stats["negative"] = (reward < 0).sum().item()

        return weight_delta

    def apply_update(self, weight_delta: torch.Tensor):
        """
        应用权重更新

        Args:
            weight_delta: [out_features, in_features]
        """
        with torch.no_grad():
            self.layer.weight.data += self.learning_rate * weight_delta

    def reset(self):
        """重置traces"""
        self.trace_pre = None
        self.trace_post = None

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

