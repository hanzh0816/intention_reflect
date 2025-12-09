import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven.neuron import (
    MultiStepIFNode,
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
    LIFNode,
)
from spikingjelly.clock_driven import functional
import torch.nn.functional as F
from tqdm import tqdm


# ==================== 模型定义 ====================
class SimpleSNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, output_size=10, time_steps=60, tau=2.0):
        super().__init__()
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        # 标准 Linear 层（权重可学习）
        self.fc_input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

        # spikingjelly 的 LIFNode（版本0.0.0.0.14兼容，使用step_mode='s'）
        self.hidden = LIFNode(
            tau=tau,
            v_threshold=0.5,
            v_reset=None,
        )
        self.output = LIFNode(
            tau=tau,
            v_threshold=0.5,
            v_reset=None,
        )

        # ---------------- R-STDP 参数 ----------------
        self.eta = 0.0005  # 学习率（对STDP要小很多）
        self.A_plus = 0.01
        self.A_minus = 0.015  # LTD 通常略大于 LTP
        self.tau_plus = 20.0
        self.tau_minus = 20.0

        # 输入到隐藏层保持随机初始化不训练（经典做法）
        for p in self.fc_input_to_hidden.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, 784] ∈ [0,1]
        B = x.shape[0]
        device = x.device

        # 重置神经元状态（重要，对于step_mode='s'）
        functional.reset_net(self.hidden)
        functional.reset_net(self.output)

        # 记录首次放电时间（inf 表示未放电）
        hid_first_spike = torch.full((B, self.hidden_size), float("inf"), device=device)
        out_first_spike = torch.full((B, self.output_size), float("inf"), device=device)

        # 将输入扩展成 Poisson 脉冲序列 [T, B, 784]
        poisson_input = torch.rand(
            self.time_steps, B, self.input_size, device=device
        ) < x.unsqueeze(0)
        poisson_input = poisson_input.float()  # [T, B, 784]

        for t in range(self.time_steps):
            inp_t = poisson_input[t]  # [B, 784]

            # ---- 隐藏层 ----
            hid_curr = self.fc_input_to_hidden(inp_t)  # [B, hidden]
            hid_spk = self.hidden(hid_curr)  # [B, hidden]

            # 记录首次放电
            mask_first = (hid_spk > 0) & (hid_first_spike == float("inf"))
            hid_first_spike[mask_first] = t

            # ---- 输出层 ----
            out_curr = self.fc_hidden_to_output(hid_spk)  # [B, output]
            out_spk = self.output(out_curr)  # [B, output]

            mask_first_out = (out_spk > 0) & (out_first_spike == float("inf"))
            out_first_spike[mask_first_out] = t

        # 预测：最早放电的输出神经元
        pred = out_first_spike.argmin(dim=1)

        return pred, hid_first_spike, out_first_spike

    # ---------------- 奖励调制的 STDP 更新 ----------------
    def reward_modulated_stdp(self, hid_times, out_times, targets):
        """
        hid_times : [B, hidden_size]   首次放电时间
        out_times : [B, output_size]
        targets   : [B] 长整型标签
        """
        B = targets.shape[0]
        device = self.fc_hidden_to_output.weight.device

        for b in range(B):
            true_class = targets[b].item()

            for o in range(self.output_size):
                reward = 1.0 if o == true_class else 0  # 正确 +1，错误轻惩罚（经验值）

                t_post = out_times[b, o]
                if not torch.isfinite(t_post):
                    continue

                for h in range(self.hidden_size):
                    t_pre = hid_times[b, h]
                    if not torch.isfinite(t_pre):
                        continue

                    delta_t = t_post - t_pre

                    if delta_t > 0:  # pre → post : LTP
                        dw = reward * self.A_plus * torch.exp(-delta_t / self.tau_plus)
                    elif delta_t < 0:  # post → pre : LTD
                        dw = reward * (-self.A_minus) * torch.exp(delta_t / self.tau_minus)
                    else:
                        dw = 0.0

                    # 注意：weight[h, o] 对应 pre=h → post=o
                    self.fc_hidden_to_output.weight.data[h, o] += self.eta * dw

        # 防止权重爆炸
        with torch.no_grad():
            self.fc_hidden_to_output.weight.data.clamp_(-2.0, 2.0)


# ==================== 数据 & 训练循环 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

model = SimpleSNN(hidden_size=512, time_steps=60, tau=2.0).to(device)

print("开始训练（纯 R-STDP 有监督学习）...")
epochs = 15
for epoch in range(1, epochs + 1):
    model.train()
    correct = total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
    for inputs, labels in progress_bar:
        labels = labels.to(device)
        inputs = inputs.view(inputs.size(0), -1).to(device)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)  # [0,1]

        pred, hid_t, out_t = model(inputs)
        model.reward_modulated_stdp(hid_t, out_t, labels)

        correct += (pred == labels).sum().item()
        total += labels.size(0)
        if total > 0:
            running_acc = correct / total * 100
            progress_bar.set_postfix(train_acc=f"{running_acc:5.2f}%")

    train_acc = correct / total * 100

    # 测试
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
            pred, _, _ = model(inputs)
            test_correct += (pred == labels.to(device)).sum().item()
            test_total += labels.size(0)

    print(
        f"Epoch {epoch:2d} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_correct/test_total*100:5.2f}%"
    )
