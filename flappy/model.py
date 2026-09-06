"""网络：Nature-DQN 卷积栈 + Dueling 双头。

刻意 **不含** BatchNorm 和 Dropout。这不是简化，是正确性要求：
目标网络若长期处于 train 模式，BatchNorm 会用当前 minibatch 的统计量算 TD 目标，
于是 Bellman 算子随批次组成而变 —— 一个不固定的算子没有不动点，训练不可能收敛。
Dropout 则让"贪婪"动作变成随机动作。两者删掉之后，train()/eval() 成为空操作，
再也不会因为漏调用 eval() 而出错。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out(size, kernel, stride, padding=0):
    """卷积输出边长。改结构时用它算，别手算 —— 算错了要到第一次前向才报错。"""
    return (size + 2 * padding - kernel) // stride + 1


class DuelingDQN(nn.Module):
    """Nature-DQN 卷积栈 + Dueling 双头。

    输入形状和 fc 宽度都可配置：

    - ``in_h`` / ``in_w`` 是观测数组的两个轴。注意观测是 **转置** 的
      （pygame surfarray 是 width-major），所以 in_h 对应屏幕宽、
      in_w 对应屏幕高。卷积网络不在乎哪边是"上"，但形状必须对。
    - ``fc_hidden`` 默认 256。原来是 512，实测那 512 维里有 14-25% 的单元
      从不激活、95% 的方差只活在约 100 维子空间里，是明确的冗余。
      见 docs/learn/12-network-sizing.md。

    卷积栈本身不动：它只占 6.2% 的参数，而实测几乎零死单元、
    有效维度 600-800，是真的在干活。
    """

    def __init__(self, n_actions=2, in_channels=4, in_h=80, in_w=128,
                 fc_hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 展平维度按输入形状算出来，不写死 —— 换分辨率时最容易漏掉的就是这里
        h = conv_out(conv_out(conv_out(in_h, 8, 4), 4, 2), 3, 1)
        w = conv_out(conv_out(conv_out(in_w, 8, 4), 4, 2), 3, 1)
        self.feat_shape = (64, h, w)
        self.flat_dim = 64 * h * w

        self.fc = nn.Linear(self.flat_dim, fc_hidden)
        self.value = nn.Linear(fc_hidden, 1)
        self.advantage = nn.Linear(fc_hidden, n_actions)

        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        # 输出层用极小增益，使初始 Q ≈ 0 —— 冒烟测试要读这个信号
        for layer in (self.value, self.advantage):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x: (N,4,H,W) uint8 或 float。归一化放在 GPU 上做，省 4 倍 PCIe 流量
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.flatten(1)))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


def build_net(cfg, device):
    """按配置造网络。所有入口都走这里，免得某条路径用了不同的形状。"""
    return DuelingDQN(in_h=cfg['obs_w'], in_w=cfg['obs_h'],
                      in_channels=cfg['frame_stack'],
                      fc_hidden=cfg['fc_hidden']).to(device)


def assert_deterministic(net, cfg, device):
    """一行自检：网络在推理时必须是确定的。

    当初就是这一行能抓到 Dropout(0.3) 的 bug —— 那个 bug 让"贪婪"评测
    实际上一直在掷骰子。训练、评测、可视化三条路径都在启动时跑一次。
    """
    dummy = torch.zeros(1, cfg['frame_stack'], cfg['obs_w'], cfg['obs_h'],
                        dtype=torch.uint8, device=device)
    with torch.no_grad():
        q1 = net(dummy)
        q2 = net(dummy)
    assert torch.allclose(q1, q2), \
        "network is stochastic at act time (Dropout / BatchNorm in train mode)"
