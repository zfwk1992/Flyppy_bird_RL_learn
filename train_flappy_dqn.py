"""
Flappy Bird Dueling Double-DQN —— 干净重写版。

为什么重写而不是原地修补
------------------------
旧管线（deep_Q_dueling_DQN.py / continue_training.py / deep_Q_oneStep.py）
跑了 4 轮、最长 10 万局 / 13 小时，近百局均分始终停在 ~24（约 1.2 根管道）。
诊断确认了 12 个互相掩盖的缺陷，其中 5 个是结构性的、无法靠调参绕开：

  1. 从不调用 eval()：Dropout(0.3) 让"贪婪"动作变随机；更致命的是目标网络
     长期处于 train 模式，BatchNorm 用当前 minibatch 统计量，
     导致 TD 目标随批次组成而变 —— Bellman 算子不是固定算子，不存在不动点。
     => 本文件彻底删除 BatchNorm 和 Dropout，让 train()/eval() 变成空操作。

  2. 目标网络每 125,000 步才同步，12.5 小时只做了 11 次 Bellman 回传；
     而管道从生成到得分需 50 帧（skip=4 时 12.5 次决策）。
     => 改为每 1000 次 **梯度步** 硬同步，计数器是 grad_steps 而非环境步。

  3. `step += 1` 夹在两个 `% k` 判断之间，存进经验池的是
     (s_{t+3}, a_t, r_{t+3}, s_{t+4}) —— 状态/动作/奖励三者全错位，
     且丢弃 75% 的奖励与 terminal。
     => 帧跳过封装进 skip_step()，主循环永不出现 `% k`。

  4. 崩溃时环境先 reset 再绘制，terminal 观测是下一局首帧；训练循环又
     无条件 s_t = s_t1 且不重置帧栈 => 每局 3 个跨局拼接样本（约 23%）。
     => 用 game/flappy_env.FlappyEnv（step 绝不内部 reset）+ 显式重置帧栈。

  5. PER 的 α 被施加两次、max_priority 单调不降 => ~276× 偏向最近样本，
     经验回放的去相关作用被完全抵消。
     => 先用均匀采样。管道样本稀疏的问题由 skip=4 和 1000× 的回传次数解决。

其余修正见各处注释。
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.flappy_env import FlappyEnv

# ======================================================================
# 配置 —— 所有量的单位都写在名字里，杜绝"帧 vs 决策步"混用
# ======================================================================
CONFIG = dict(
    # 环境
    frame_skip=4,
    frame_stack=4,
    obs_size=80,
    pipe_reward=1.0,
    death_reward=-1.0,
    alive_reward=0.0,
    shaping_coef=0.05,

    # 难度：管道上下间隙（像素）。小鸟高 24px。
    #   150 = 本仓库此前放宽过的值（第一版训练用的）
    #   100 = 原版 Flappy Bird 的难度
    # 间隙越小，可通行的竖直窗口越窄，容错越低。
    # 改这个值会让新旧模型的成绩不可直接比较。
    pipe_gap=100,

    # 学习
    gamma=0.99,                      # 决策级折扣
    batch_size=128,                  # RTX 3060 实测：batch 128 = 4.67ms/步，
                                     # batch 32 = 5.04ms/步 —— 小网络下 GPU 被
                                     # kernel 启动延迟主导，梯度步速率恒为
                                     # ~200 步/秒与 batch 无关。所以 batch 白拿。
    lr=1.5e-4,                       # batch 32->128 后按 sqrt 缩放
    adam_eps=1.5e-4,                 # Rainbow 的取值；1e-8 会让 Adam 退化成
                                     # lr*sign(g)，分不清"+1 管道"和数值噪声
    grad_clip=10.0,                  # Huber 已限幅；1.0 会让几乎每批都被重缩放

    # 节奏
    warmup_decisions=20_000,
    train_every_decisions=4,         # 配 batch 128 = 每次决策 32 个样本的学习量
                                     # （标准回放比），同时把梯度步需求压到
                                     # GPU 的 200 步/秒上限之内
    target_sync_grad_steps=1_000,

    # 经验回放
    buffer_capacity=400_000,         # 环境提速 11 倍后缓冲区要跟上；
                                     # uint8 存储下 400k ~= 12.8GB（本机 34GB）

    # 探索
    eps_start=1.0,
    eps_mid=0.05,
    eps_final=0.01,
    eps_anneal1_decisions=250_000,
    eps_anneal2_decisions=500_000,
    eps_random_flap_prob=0.2,        # 见 sample_random_action 的说明

    # 回合长度上限。一旦策略学好，回合可以无限长（实测冒烟测试的 checkpoint
    # 能连过 400+ 根管道不死），不设上限会让 episode 级的日志/存档/评测全部卡住。
    # 注意：截断 **不是** 终止 —— 截断处 done 保持 False，价值仍然自举，
    # 否则等于告诉网络"飞太久会凭空死掉"。
    max_episode_decisions=4_000,

    # 记录
    max_episodes=200_000,
    seed=0,
    eval_every_episodes=500,         # 长跑需要更密的评测曲线
    eval_episodes=20,                # 回合变长后评测本身也耗时
    eval_epsilon=0.01,
    resume_every_minutes=30,
    log_train_every_grad_steps=100,
)

SMOKE_OVERRIDES = dict(
    warmup_decisions=5_000,
    buffer_capacity=50_000,
    target_sync_grad_steps=300,
    eps_anneal1_decisions=40_000,
    eps_anneal2_decisions=40_000,
    max_episodes=20_000,
    eval_every_episodes=2_000,
)


# ======================================================================
# 预处理
# ======================================================================
# 预处理已下沉到 game/flappy_env.FlappyEnv._observe()：
#   env.step() / env.reset() 直接返回 (80,80) uint8 的 {0,255} 二值图。
# 那里用 pixels3d 视图代替 array3d 拷贝，整条取图路径 977us -> 234us，
# 且输出与旧路径逐像素等价（600 帧核对，不一致率 0.0033%）。


# ======================================================================
# 网络：Nature-DQN 卷积栈 + Dueling 双头，无 BatchNorm 无 Dropout
# ======================================================================
class DuelingDQN(nn.Module):
    """约 1.26M 参数 / 5.0MB fp32。

    旧网络是 3.62M 参数 / 14.5MB，其中 90.5% 集中在一个 Linear(6400,512) 上 ——
    因为它的 conv3 用了 stride=1,padding=1，特征图停在 10x10。
    这里用未 padding 的 Nature 栈：80 -> 19 -> 8 -> 6，得 64*6*6 = 2304。
    """

    def __init__(self, n_actions=2, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 6 * 6, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        # 输出层用极小增益，使初始 Q ≈ 0 —— 冒烟测试的判据 1 要读这个信号
        for layer in (self.value, self.advantage):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x: (N,4,80,80) uint8 或 float。归一化放在 GPU 上做，省 4 倍 PCIe 流量
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.flatten(1)))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


# ======================================================================
# 经验回放：均匀采样，uint8 存储
# ======================================================================
class ReplayBuffer:
    """只存 state(4帧) + next_frame(1帧)，采样时拼出 next_state。

    next_state = concat(next_frame, state[:3]) 正是主循环推进帧栈用的同一个
    恒等式，可直接单测（见 test/test_env_and_buffer.py 测试 4）。

    内存：cap * 5 * 6400 字节。cap=200k -> 6.4GB。
    旧实现存 float32 的双份完整帧栈 = 200KB/条，cap=100k 时要 20GB，
    而帧已被二值化成 {0,255} —— 32 倍浪费。
    """

    def __init__(self, capacity, stack=4, size=80):
        self.capacity = capacity
        self.stack = stack
        self.states = np.zeros((capacity, stack, size, size), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, 1, size, size), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_frame, done):
        i = self.pos
        self.states[i] = state
        self.next_frames[i, 0] = next_frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        # 有放回采样：标准 DQN 的做法，O(n)。
        # 旧实现用 np.random.choice(replace=False, p=probs) 在 10 万条上采样，
        # 走的是 NumPy 的顺序选择慢路径，约 20ms/步，是整个训练的吞吐瓶颈。
        idx = np.random.randint(0, self.size, size=n)
        s = self.states[idx]
        s1 = np.concatenate([self.next_frames[idx], s[:, :self.stack - 1]], axis=1)
        return s, self.actions[idx], self.rewards[idx], s1, self.dones[idx]

    def __len__(self):
        return self.size

    def nbytes(self):
        return (self.states.nbytes + self.next_frames.nbytes + self.actions.nbytes
                + self.rewards.nbytes + self.dones.nbytes)


# ======================================================================
# 帧跳过：一次决策 = 一条经验，奖励全额累加
# ======================================================================
def skip_step(env, action, phi_prev, cfg):
    """把 frame_skip 帧压成一次决策。

    这是旧管线 #3 的结构性修复：动作在整个窗口内重复，窗口内每一帧的奖励
    都被累加，terminal 提前 break。主循环里不存在任何 `% frame_skip`，
    因此不可能再出现"选动作用一个计数器、存经验用另一个"的错位。

    势能塑形在 **决策级** 施加。若按帧施加，skip 窗口内的求和不等于
    γ_dec·Φ(s_{t+k}) − Φ(s_t)，Ng-Harada-Russell 的策略不变性就不精确成立。
    """
    r_raw = 0.0
    done = False
    obs = None
    info = {}
    k = cfg['frame_skip']
    for i in range(k):
        # 只有窗口的最后一帧（或提前终止的那一帧）才绘制+取图。
        # 前面几帧的画面根本不会进入帧栈，渲染它们纯属浪费 ——
        # 实测物理 5.8us vs 绘制+取图 970us，这一行是 11 倍提速的来源。
        obs, r, done, info = env.step(action, render=(i == k - 1))
        r_raw += r
        if done:
            if obs is None:                 # 提前终止：补画崩溃帧
                obs = env.observe()
            break
    phi_next = 0.0 if done else info['potential']
    r_dec = r_raw + cfg['shaping_coef'] * (cfg['gamma'] * phi_next - phi_prev)
    return obs, r_dec, done, info, phi_next


# ======================================================================
# 探索
# ======================================================================
def epsilon_at(decision_step, cfg):
    """单调不增，只依赖一个计数器 decision_step。

    旧实现把衰减挂在 decision_step 上、却把下限（0.5/0.2/0.1）挂在
    episode_count 上，两个计数器混用；还有一个占空比 0.08% 的"探索脉冲"，
    实际什么也没做。这里全部删掉。
    """
    if decision_step < cfg['warmup_decisions']:
        return cfg['eps_start']
    t = decision_step - cfg['warmup_decisions']
    a1 = cfg['eps_anneal1_decisions']
    a2 = cfg['eps_anneal2_decisions']
    if t < a1:
        return cfg['eps_start'] + (cfg['eps_mid'] - cfg['eps_start']) * (t / a1)
    if t < a1 + a2:
        return cfg['eps_mid'] + (cfg['eps_final'] - cfg['eps_mid']) * ((t - a1) / a2)
    return cfg['eps_final']


def sample_random_action(cfg):
    """随机动作按 p_flap≈0.2 采样，而非均匀的 0.5。

    实测物理：一次扇翅令 velY=-5，重力 +0.5/帧，单次扇翅后净位移在
    T≈19 帧（约 5 次决策）归零 —— 所以"悬停"对应的扇翅率就是 1/5。
    均匀随机（p=0.5）会把鸟顶死在天花板上，这正是旧 warmup 数据毫无价值的原因。
    只改行为策略，off-policy 的 Q 学习不受影响。
    """
    return 1 if random.random() < cfg['eps_random_flap_prob'] else 0


# ======================================================================
# 智能体
# ======================================================================
class Agent:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.online = DuelingDQN().to(device)
        self.target = DuelingDQN().to(device)
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
        self.assert_deterministic_act()

    def assert_deterministic_act(self):
        """一行自检，当初就能抓到 Dropout(0.3) 的 bug。"""
        dummy = torch.zeros(1, self.cfg['frame_stack'],
                            self.cfg['obs_size'], self.cfg['obs_size'],
                            dtype=torch.uint8, device=self.device)
        with torch.no_grad():
            q1 = self.online(dummy)
            q2 = self.online(dummy)
        assert torch.allclose(q1, q2), \
            "network is stochastic at act time (Dropout / BatchNorm in train mode)"

    @torch.no_grad()
    def act(self, stack_uint8):
        t = torch.from_numpy(stack_uint8).unsqueeze(0).to(self.device)
        return int(self.online(t).argmax(1).item())

    @torch.no_grad()
    def q_stats(self, stack_uint8):
        t = torch.from_numpy(stack_uint8).unsqueeze(0).to(self.device)
        q = self.online(t)[0]
        return float(q.max()), float(q[1] - q[0])

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


# ======================================================================
# 评测：独立路径，绝不经过任何带 warmup 短路的动作选择
# ======================================================================
@torch.no_grad()
def evaluate(agent, cfg, n_episodes, epsilon):
    """旧管线的 test/test_ai_gameplay.py 调用 agent.select_action()，而后者在
    decision_step < 20000 时直接返回随机动作，且 load_model 不恢复 decision_step
    —— 于是历史上所有"评测"跑的都是纯随机策略，113 个 checkpoint 没有一个
    有可信的评测数据。这里直接用网络，不存在任何短路分支。
    """
    env = FlappyEnv(pipe_reward=cfg['pipe_reward'],
                    death_reward=cfg['death_reward'],
                    alive_reward=cfg['alive_reward'],
                    pipe_gap=cfg['pipe_gap'])
    cap = cfg['max_episode_decisions']
    pipes, lengths, n_truncated = [], [], 0
    for _ in range(n_episodes):
        frame = env.reset()          # 已是 (80,80) uint8
        stack = np.repeat(frame[None], cfg['frame_stack'], axis=0)
        phi = env.current_potential()
        n_dec = 0
        while True:
            if random.random() < epsilon:
                action = sample_random_action(cfg)
            else:
                action = agent.act(stack)
            nf, r, done, info, phi = skip_step(env, action, phi, cfg)
            n_dec += 1
            # 上限是必需的：学好之后策略可以永远不死，评测会挂住
            if done or n_dec >= cap:
                pipes.append(info['score'])
                lengths.append(n_dec)
                n_truncated += int(not done)
                break
            stack = np.concatenate([nf[None], stack[:cfg['frame_stack'] - 1]], axis=0)
    return dict(
        pipes_mean=float(np.mean(pipes)),
        pipes_std=float(np.std(pipes)),
        pipes_max=int(np.max(pipes)),
        len_mean=float(np.mean(lengths)),
        truncated=n_truncated,
    )


# ======================================================================
# CSV 日志
# ======================================================================
class CsvLogger:
    """旧管线用正则去抓一行 emoji 日志（plot_training_progress.py:21），
    只能拿到 (episode, score, max, avg100)，看不到 loss / Q / epsilon /
    目标网络同步 —— 这正是所有问题在 13 小时里都没被发现的原因。
    """

    def __init__(self, path, fields, flush_every=50):
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.f.flush()
        self.n = 0
        self.flush_every = flush_every

    def write(self, **row):
        self.w.writerow(row)
        self.n += 1
        if self.n % self.flush_every == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()


EPISODE_FIELDS = ['episode', 'decision_step', 'frames', 'pipes', 'ep_return',
                  'ep_len_decisions', 'epsilon', 'recent100_pipes',
                  'recent100_return', 'buffer_size', 'wall_s', 'terminated']
TRAIN_FIELDS = ['grad_step', 'decision_step', 'loss', 'td_abs_mean', 'td_abs_max',
                'q_mean', 'q_std', 'q_min', 'q_max', 'target_q_mean', 'grad_norm',
                'lr', 'epsilon', 'buffer_size', 'syncs', 'dec_per_s', 'grad_per_s']
EVAL_FIELDS = ['episode', 'decision_step', 'n_eval_eps', 'eval_pipes_mean',
               'eval_pipes_std', 'eval_pipes_max', 'eval_len_mean']


# ======================================================================
# 主训练循环
# ======================================================================
def train(cfg, args):
    # ---- 设备 ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dev_name = torch.cuda.get_device_name(0)
    elif args.allow_cpu:
        device = torch.device('cpu')
        dev_name = 'cpu'
    else:
        raise SystemExit(
            "CUDA not available (torch=%s). The old code silently fell back to CPU\n"
            "and hid this. Restore a CUDA build (requirements_cuda.txt / Dockerfile),\n"
            "or pass --allow-cpu to accept a ~20x slowdown." % torch.__version__)

    # ---- 随机种子 ----
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark = True

    # ---- 运行目录 ----
    run_dir = args.run_dir or os.path.join(
        'runs', datetime.now().strftime('%Y%m%d_%H%M%S') + ('_smoke' if args.smoke else ''))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    ep_log = CsvLogger(os.path.join(run_dir, 'episodes.csv'), EPISODE_FIELDS)
    tr_log = CsvLogger(os.path.join(run_dir, 'train.csv'), TRAIN_FIELDS)
    # 评测行很稀疏（每 500 局才一条），每行即刷，免得进程意外中断时全丢
    ev_log = CsvLogger(os.path.join(run_dir, 'eval.csv'), EVAL_FIELDS, flush_every=1)

    # 纯 ASCII 日志：旧环境在热循环里 print emoji，在 Windows cp1252 stdout 上
    # 会抛 UnicodeEncodeError
    def say(msg):
        line = "[%s] %s" % (datetime.now().strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        with open(os.path.join(run_dir, 'run.log'), 'a', encoding='utf-8') as fh:
            fh.write(line + "\n")

    say("run_dir=%s" % run_dir)
    say("device=%s (%s), torch=%s" % (device, dev_name, torch.__version__))

    # ---- 组件 ----
    agent = Agent(cfg, device)
    n_params = sum(p.numel() for p in agent.online.parameters())
    say("network params=%d (%.2f MB fp32)" % (n_params, n_params * 4 / 1e6))

    buffer = ReplayBuffer(cfg['buffer_capacity'], cfg['frame_stack'], cfg['obs_size'])
    say("replay capacity=%d (%.2f GB allocated)" % (cfg['buffer_capacity'],
                                                    buffer.nbytes() / 1e9))

    env = FlappyEnv(pipe_reward=cfg['pipe_reward'],
                    death_reward=cfg['death_reward'],
                    alive_reward=cfg['alive_reward'],
                    pipe_gap=cfg['pipe_gap'])

    # ---- 状态 ----
    decision_step = 0
    episode = 0
    best_recent100 = -float('inf')
    last_best_episode = -10 ** 9
    recent_pipes = deque(maxlen=100)
    recent_returns = deque(maxlen=100)
    resume_slot = 0
    t_start = time.time()
    last_resume_t = t_start
    last_log_grad = 0
    last_log_dec = 0
    last_log_t = t_start
    acc = {k: 0.0 for k in ('loss', 'td_abs_mean', 'td_abs_max', 'q_mean', 'q_std',
                            'q_min', 'q_max', 'target_q_mean', 'grad_norm')}
    acc_n = 0

    # ---- 首个回合 ----
    frame = env.reset()          # 已是 (80,80) uint8
    stack = np.repeat(frame[None], cfg['frame_stack'], axis=0)
    phi = env.current_potential()
    ep_return = 0.0
    ep_dec = 0

    say("start training: warmup=%d decisions, learning starts after that"
        % cfg['warmup_decisions'])

    max_seconds = args.max_hours * 3600 if args.max_hours else None
    stop_reason = 'max_episodes'

    while episode < cfg['max_episodes']:
        if max_seconds and time.time() - t_start > max_seconds:
            stop_reason = 'max_hours'
            break
        eps = epsilon_at(decision_step, cfg)

        if random.random() < eps:
            action = sample_random_action(cfg)
        else:
            action = agent.act(stack)

        next_frame, r_dec, done, info, phi = skip_step(env, action, phi, cfg)
        buffer.add(stack, action, r_dec, next_frame, done)

        decision_step += 1
        ep_dec += 1
        ep_return += r_dec

        # ---- 学习 ----
        if (decision_step > cfg['warmup_decisions']
                and len(buffer) >= cfg['batch_size']
                and decision_step % cfg['train_every_decisions'] == 0):
            stats = agent.learn(buffer)
            for k in acc:
                acc[k] += stats[k]
            acc_n += 1

            if agent.grad_steps % cfg['log_train_every_grad_steps'] == 0:
                now = time.time()
                dt = max(now - last_log_t, 1e-9)
                tr_log.write(
                    grad_step=agent.grad_steps,
                    decision_step=decision_step,
                    **{k: round(acc[k] / max(acc_n, 1), 6) for k in acc},
                    lr=cfg['lr'],
                    epsilon=round(eps, 5),
                    buffer_size=len(buffer),
                    syncs=agent.syncs,
                    dec_per_s=round((decision_step - last_log_dec) / dt, 1),
                    grad_per_s=round((agent.grad_steps - last_log_grad) / dt, 1),
                )
                acc = {k: 0.0 for k in acc}
                acc_n = 0
                last_log_t, last_log_dec, last_log_grad = now, decision_step, agent.grad_steps

        # ---- 回合结束（撞死，或达到长度上限被截断） ----
        truncated = (not done) and ep_dec >= cfg['max_episode_decisions']
        if done or truncated:
            episode += 1
            pipes = info['score']
            recent_pipes.append(pipes)
            recent_returns.append(ep_return)
            r100 = float(np.mean(recent_pipes))

            ep_log.write(
                episode=episode, decision_step=decision_step, frames=info['frames'],
                pipes=pipes, ep_return=round(ep_return, 4), ep_len_decisions=ep_dec,
                epsilon=round(eps, 5), recent100_pipes=round(r100, 4),
                recent100_return=round(float(np.mean(recent_returns)), 4),
                buffer_size=len(buffer), wall_s=round(time.time() - t_start, 1),
                terminated=int(done),
            )

            if episode % 500 == 0:
                say("ep=%d dec=%d grad=%d syncs=%d eps=%.3f recent100_pipes=%.2f "
                    "buf=%d wall=%.0fs"
                    % (episode, decision_step, agent.grad_steps, agent.syncs,
                       eps, r100, len(buffer), time.time() - t_start))

            # ---- 最佳模型：门控在平滑指标上，且带冷却 ----
            # 旧代码把 max_score 放在每局都执行的分支里重新赋值
            # (deep_Q_dueling_DQN.py:793-796)，20000 局后阈值恒为 0，
            # 于是每个正分局都是"新纪录" —— 触发了 31182 次 58MB 存盘，
            # 却从未保存过任何真正意义上的最佳模型。
            if (episode >= 500 and len(recent_pipes) == 100
                    and r100 > best_recent100 * 1.02
                    and episode - last_best_episode >= 100):
                best_recent100 = r100
                last_best_episode = episode
                torch.save({'model': agent.online.state_dict(),
                            'recent100_pipes': r100, 'episode': episode,
                            'decision_step': decision_step, 'config': cfg},
                           os.path.join(run_dir, 'best.pt'))
                say("new best: recent100_pipes=%.3f at ep=%d -> best.pt" % (r100, episode))

            # ---- 定期评测 ----
            if cfg['eval_every_episodes'] and episode % cfg['eval_every_episodes'] == 0:
                res = evaluate(agent, cfg, cfg['eval_episodes'], cfg['eval_epsilon'])
                ev_log.write(episode=episode, decision_step=decision_step,
                             n_eval_eps=cfg['eval_episodes'],
                             eval_pipes_mean=round(res['pipes_mean'], 3),
                             eval_pipes_std=round(res['pipes_std'], 3),
                             eval_pipes_max=res['pipes_max'],
                             eval_len_mean=round(res['len_mean'], 2))
                say("eval @ep=%d: pipes=%.2f+-%.2f max=%d len=%.1f truncated=%d/%d"
                    % (episode, res['pipes_mean'], res['pipes_std'],
                       res['pipes_max'], res['len_mean'], res['truncated'],
                       cfg['eval_episodes']))

            # ---- 断点续训：双槽轮转 ----
            if time.time() - last_resume_t > cfg['resume_every_minutes'] * 60:
                torch.save({'model': agent.online.state_dict(),
                            'target': agent.target.state_dict(),
                            'optim': agent.optimizer.state_dict(),
                            'episode': episode, 'decision_step': decision_step,
                            'grad_steps': agent.grad_steps, 'syncs': agent.syncs,
                            'epsilon': eps, 'config': cfg},
                           os.path.join(run_dir, 'resume_%d.pt' % resume_slot))
                resume_slot ^= 1
                last_resume_t = time.time()

            # ---- 重置帧栈：杜绝跨局拼接 ----
            # 旧代码在 terminal 那一轮也执行无条件的 s_t = s_t1
            # (deep_Q_dueling_DQN.py:785)，于是新回合的前 3 次决策都拿着
            # 混有上一局画面的帧栈，且以 done=False 存进经验池。
            frame = env.reset()          # 已是 (80,80) uint8
            stack = np.repeat(frame[None], cfg['frame_stack'], axis=0)
            phi = env.current_potential()
            ep_return = 0.0
            ep_dec = 0
        else:
            stack = np.concatenate([next_frame[None], stack[:cfg['frame_stack'] - 1]],
                                   axis=0)

    # 收尾：跑一次更大规模的贪婪评测，并保存最终模型
    say("stopping (%s). running final evaluation..." % stop_reason)
    final = evaluate(agent, cfg, 30, cfg['eval_epsilon'])
    say("FINAL eval: pipes=%.2f+-%.2f max=%d len=%.1f truncated=%d/30"
        % (final['pipes_mean'], final['pipes_std'], final['pipes_max'],
           final['len_mean'], final['truncated']))
    torch.save({'model': agent.online.state_dict(),
                'episode': episode, 'decision_step': decision_step,
                'grad_steps': agent.grad_steps, 'final_eval': final,
                'config': cfg}, os.path.join(run_dir, 'final.pt'))

    say("done: episodes=%d decisions=%d grad_steps=%d syncs=%d wall=%.2fh"
        % (episode, decision_step, agent.grad_steps, agent.syncs,
           (time.time() - t_start) / 3600))
    ep_log.close()
    tr_log.close()
    ev_log.close()


def main():
    p = argparse.ArgumentParser(description="Flappy Bird Dueling Double-DQN (clean rewrite)")
    p.add_argument('--smoke', action='store_true', help='short smoke-test config')
    p.add_argument('--allow-cpu', action='store_true',
                   help='allow running without CUDA (~20x slower)')
    p.add_argument('--episodes', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--run-dir', type=str, default=None)
    p.add_argument('--buffer', type=int, default=None)
    p.add_argument('--max-hours', type=float, default=None,
                   help='stop cleanly after N hours (final eval + final.pt)')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='difficulty: vertical pipe gap in px '
                        '(150 = easy/first run, 100 = original game, lower = harder)')
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
    if args.episodes is not None:
        cfg['max_episodes'] = args.episodes
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.buffer is not None:
        cfg['buffer_capacity'] = args.buffer
    if args.pipe_gap is not None:
        cfg['pipe_gap'] = args.pipe_gap

    train(cfg, args)


if __name__ == '__main__':
    main()
