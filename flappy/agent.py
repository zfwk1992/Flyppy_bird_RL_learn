"""Double-DQN 的学习步与目标网络同步。"""

import random

import torch
import torch.nn.functional as F

from .model import assert_deterministic, build_net
from .rollout import sample_random_action


class Agent:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.online = build_net(cfg, device)
        self.target = build_net(cfg, device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg['lr'], eps=cfg['adam_eps'])
        # 无 weight_decay：对价值函数做 L2 等于把 Q 往 0 拉，方向是错的。
        # 无 LR 衰减：最好那轮的证据是 LR 衰减 36 倍而 loss 上升 53% ——
        # 那是"跟不上自己移动的目标"，是 LR 太小的特征，不是太大。

        self.grad_steps = 0
        self.syncs = 0
        assert_deterministic(self.online, cfg, device)

    # ------------------------------------------------------------------
    # 行动
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, stack_uint8):
        t = torch.from_numpy(stack_uint8).unsqueeze(0).to(self.device)
        return int(self.online(t).argmax(1).item())

    def act_epsilon_greedy(self, stack_uint8, epsilon):
        """训练时的行为策略。

        没有任何按训练进度短路的分支 —— warmup 期间靠 epsilon_at 返回
        eps_start=1.0 来实现"全随机"，而不是靠一条 if。这样 epsilon 曲线
        永远等于实际的随机比例，日志可信。
        """
        if random.random() < epsilon:
            return sample_random_action(self.cfg)
        return self.act(stack_uint8)

    @torch.no_grad()
    def q_stats(self, stack_uint8):
        t = torch.from_numpy(stack_uint8).unsqueeze(0).to(self.device)
        q = self.online(t)[0]
        return float(q.max()), float(q[1] - q[0])

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------
    def learn(self, buffer):
        cfg = self.cfg
        s, a, r, s1, d = buffer.sample(cfg['batch_size'])

        s = torch.from_numpy(s).to(self.device, non_blocking=True)
        s1 = torch.from_numpy(s1).to(self.device, non_blocking=True)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        q_sa = self.online(s).gather(1, a.unsqueeze(1))

        # Double DQN：在线网络选动作，目标网络打分
        with torch.no_grad():
            a_star = self.online(s1).argmax(1, keepdim=True)
            q_next = self.target(s1).gather(1, a_star)
            y = r.unsqueeze(1) + cfg['gamma'] * q_next * (~d).unsqueeze(1)

        loss = F.smooth_l1_loss(q_sa, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), cfg['grad_clip'])
        self.optimizer.step()
        self.grad_steps += 1

        # 同步计数器是 **梯度步** 而不是环境步。旧管线按环境步每 125,000 步
        # 才同步一次，12.5 小时只做了 11 次 Bellman 回传，而管道从生成到得分
        # 需要 12.5 次决策 —— 价值根本传不回去。
        synced = False
        if self.grad_steps % cfg['target_sync_grad_steps'] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.syncs += 1
            synced = True

        with torch.no_grad():
            td = (q_sa - y).abs()
            return dict(
                loss=float(loss),
                td_abs_mean=float(td.mean()),
                td_abs_max=float(td.max()),
                q_mean=float(q_sa.mean()),
                q_std=float(q_sa.std()) if q_sa.numel() > 1 else 0.0,
                q_min=float(q_sa.min()),
                q_max=float(q_sa.max()),
                target_q_mean=float(y.mean()),
                grad_norm=float(grad_norm),
                synced=synced,
            )

    # 学习步统计里需要平均的字段（CSV 记录时按窗口求均值）
    STAT_KEYS = ('loss', 'td_abs_mean', 'td_abs_max', 'q_mean', 'q_std',
                 'q_min', 'q_max', 'target_q_mean', 'grad_norm')
