"""与环境交互的一切：帧跳过、帧栈、探索策略、贪婪评测。

训练、评测、可视化三条路径都走这里的同一份代码。旧管线的教训正在于此 ——
评测脚本自己复制了一份动作选择逻辑，而那份副本里带着一条
"decision_step < 20000 就返回随机动作"的短路，且加载存档从不恢复
decision_step。于是历史上所有"评测"跑的都是纯随机策略，113 个存档
没有一个有可信数据。共用一份实现，这类偏差就不可能再发生。
"""

import random

import numpy as np
import torch

from game.flappy_env import FlappyEnv


# ======================================================================
# 帧栈
# ======================================================================
class FrameStack:
    """把最近 n 帧叠成 (n,80,80) uint8 的网络输入。

    ``push`` 的恒等式 —— 新帧在前、丢掉最旧的一帧 —— 与
    ReplayBuffer.sample 里重建 next_state 的方式必须逐位一致，
    否则网络在训练时看到的时间顺序和在游玩时看到的相反。
    这条一致性有单测保证（test/test_env_and_buffer.py 测试 4）。
    """

    def __init__(self, n):
        self.n = n
        self.array = None

    def reset(self, frame):
        """新回合：用首帧填满整个栈。

        这一步是必需的。旧代码在 terminal 那一轮也执行无条件的 s_t = s_t1，
        于是新回合的前 3 次决策都拿着混有上一局画面的帧栈，
        并且以 done=False 存进经验池 —— 每局约 3 个跨局拼接的假样本。
        """
        self.array = np.repeat(frame[None], self.n, axis=0)
        return self.array

    def push(self, frame):
        self.array = np.concatenate([frame[None], self.array[:self.n - 1]], axis=0)
        return self.array


# ======================================================================
# 帧跳过：一次决策 = 一条经验，奖励全额累加
# ======================================================================
def skip_step(env, action, phi_prev, cfg, render=True):
    """把 frame_skip 帧压成一次决策。

    ``render=False`` 时整个窗口一帧都不画，返回的 obs 是 None。
    只有完全不看画面的调用方才能这么用（例如 experiments/state_mlp.py
    直接读真实状态向量）—— 渲染是 970us 而物理只要 5.8us，
    省掉之后环境快约两个数量级。

    这是旧管线最关键的结构性修复：动作在整个窗口内重复，窗口内每一帧的奖励
    都被累加，terminal 提前 break。调用方不存在任何 `% frame_skip`，
    因此不可能再出现"选动作用一个计数器、存经验用另一个"的错位 ——
    旧代码的 `step += 1` 夹在两个 `% k` 判断之间，存进经验池的是
    (s_{t+3}, a_t, r_{t+3}, s_{t+4})，状态/动作/奖励三者全错位，
    还丢弃了 75% 的奖励与 terminal。

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
        obs, r, done, info = env.step(action, render=(render and i == k - 1))
        r_raw += r
        if done:
            if render and obs is None:      # 提前终止：补画崩溃帧
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
# 环境构造：所有入口都用这一个，避免奖励尺度/难度在某条路径上悄悄不同
# ======================================================================
def make_env(cfg, throttle_fps=None):
    return FlappyEnv(pipe_reward=cfg['pipe_reward'],
                     death_reward=cfg['death_reward'],
                     alive_reward=cfg['alive_reward'],
                     pipe_gap=cfg['pipe_gap'],
                     randomize=cfg['randomize_pipes'],
                     gap_range=cfg['pipe_gap_range'],
                     spacing_range=cfg['pipe_spacing_range'],
                     edge_margin=cfg['pipe_edge_margin'],
                     max_delta_frac=cfg['pipe_max_delta_frac'],
                     obs_w=cfg['obs_w'], obs_h=cfg['obs_h'],
                     throttle_fps=throttle_fps)


# ======================================================================
# 贪婪评测
# ======================================================================
@torch.no_grad()
def evaluate(net, cfg, device, n_episodes, epsilon=0.0, max_decisions=None,
             q_stats=False, env=None):
    """跑 n_episodes 局并汇总。训练中的定期评测和 eval.py 用的是同一份实现。

    ``max_decisions`` 的上限是必需的，不是保险丝：一个学好的策略可以永远不死，
    没有上限评测会直接挂住。截断的局按"还活着"计入 truncated，不计为死亡。

    ``env`` 可以复用一个专用的评测环境；**绝不能**传训练用的那个环境实例，
    否则训练回合的状态会被评测冲掉。
    """
    if max_decisions is None:
        max_decisions = cfg['max_episode_decisions']
    own_env = env is None
    if own_env:
        env = make_env(cfg)

    stacker = FrameStack(cfg['frame_stack'])
    pipes, lengths, returns = [], [], []
    q_max_all, q_gap_all = [], []
    n_truncated = 0

    for _ in range(n_episodes):
        stack = stacker.reset(env.reset())    # env.reset() 已是 (80,80) uint8
        phi = env.current_potential()
        n_dec, ep_ret = 0, 0.0
        while True:
            if epsilon > 0 and random.random() < epsilon:
                action = sample_random_action(cfg)
            else:
                q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                action = int(q.argmax().item())
                if q_stats:
                    q_max_all.append(float(q.max()))
                    q_gap_all.append(float(q[1] - q[0]))
            frame, r, done, info, phi = skip_step(env, action, phi, cfg)
            n_dec += 1
            ep_ret += r
            if done or n_dec >= max_decisions:
                pipes.append(info['score'])
                lengths.append(n_dec)
                returns.append(ep_ret)
                n_truncated += int(not done)
                break
            stack = stacker.push(frame)

    out = dict(
        episodes=n_episodes,
        pipes_mean=float(np.mean(pipes)),
        pipes_std=float(np.std(pipes)),
        pipes_max=int(np.max(pipes)),
        pipes_median=float(np.median(pipes)),
        len_mean=float(np.mean(lengths)),
        return_mean=float(np.mean(returns)),
        truncated=n_truncated,
        max_decisions=max_decisions,
    )
    if q_stats and q_max_all:
        out['q_max_mean'] = float(np.mean(q_max_all))
        out['q_gap_abs_mean'] = float(np.mean(np.abs(q_gap_all)))
    return out
