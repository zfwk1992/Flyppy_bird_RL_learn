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


def sample_random_action(cfg, rng=None):
    """随机动作按 p_flap≈0.2 采样，而非均匀的 0.5。

    实测物理：一次扇翅令 velY=-5，重力 +0.5/帧，单次扇翅后净位移在
    T≈19 帧（约 5 次决策）归零 —— 所以"悬停"对应的扇翅率就是 1/5。
    均匀随机（p=0.5）会把鸟顶死在天花板上，这正是旧 warmup 数据毫无价值的原因。
    只改行为策略，off-policy 的 Q 学习不受影响。

    ``rng`` 传入一个独立的 ``random.Random`` 时，探索噪声就不再消耗全局随机流。
    固定评测集需要这一点：管道生成也走全局 ``random``，如果动作噪声和管道
    共用一条流，不同模型抽到的随机动作次数不同，同一局的管道就会错位。
    """
    r = rng if rng is not None else random
    return 1 if r.random() < cfg['eps_random_flap_prob'] else 0


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
def iqm(values):
    """Interquartile mean：排序后掐掉最低和最高各 25%，对中间 50% 取均值。

    > **注意：在本项目里 IQM 不是首选统计量，均值才是。** 这个函数保留下来
    > 只作为对离群值稳健的交叉验证（比如想确认某次高分不是一两局撑起来的）。
    >
    > IQM 出自 Agarwal et al., *Deep RL at the Edge of the Statistical
    > Precipice* (NeurIPS 2021)，那篇解决的是**跨任务聚合**时少数任务尺度
    > 悬殊的问题。本项目的场景不同：单任务、分数近似**几何分布**
    > （每根管道死亡风险大致恒定，实测 1.457%/根）。对几何分布来说
    > **样本均值就是充分统计量、也是最小方差估计**，掐掉一半数据只会更差。
    >
    > 实测（`final.pt` 40 局，重采样 20 局算标准误）：
    >     均值 95.1 相对SE **20%** ← 最稳
    >     IQM  67.2 相对SE   33%
    >     中位 53.0 相对SE   35%
    >
    > 所以真正降噪的办法只有**加评测局数**（SE ∝ 1/√n）和**配对比较**
    > （固定关卡消掉难度差异），不是换统计量。详见
    > docs/research/WHERE_IS_THE_PROBLEM.md。
    """
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    if n == 0:
        return float('nan')
    k = n // 4                       # 每端掐掉 25%
    core = a[k:n - k] if n - 2 * k > 0 else a
    return float(core.mean())


def censored_mean(pipes, truncated):
    """把撞上决策上限的局当成**右删失**来估平均存活长度。

    为什么需要它
    ------------
    ``max_episode_decisions`` 会把还活着的局强行截断。这些局的真实分数是
    未知的（只知道 >= 当前分），把它们按当前分计入均值就是**系统性低估**，
    而且**策略越强截断越多、低估越严重** —— 会让"变强了"看起来像"没变化"。

    实测（`final.pt` 40 局，5 局被截断）：普通均值 95.1，删失校正后 108.7，
    低估约 14%。

    做法是生存分析里最基本的那一个：风险 = 死亡次数 / 总暴露量，
    平均存活 = 1/风险。截断的局只贡献暴露量、不贡献死亡。
    这里假设每根管道的死亡风险恒定（实测条件概率在 68-93% 之间无趋势，
    近似成立）。

    代价：相对标准误比普通均值高（实测 28% vs 20%），因为死亡次数变少了。
    **所以它不适合当门控指标，适合当"报告真实水平"的那个数。**
    """
    p = np.asarray(pipes, dtype=float)
    deaths = int(len(p) - int(np.sum(truncated)))
    if deaths <= 0:                  # 全部被截断：只能给一个下界
        return float(p.sum() / max(len(p), 1))
    return float(p.sum() / deaths)



@torch.no_grad()
def evaluate(net, cfg, device, n_episodes, epsilon=0.0, max_decisions=None,
             q_stats=False, env=None, seed_base=None):
    """跑 n_episodes 局并汇总。训练中的定期评测和 eval.py 用的是同一份实现。

    ``max_decisions`` 的上限是必需的，不是保险丝：一个学好的策略可以永远不死，
    没有上限评测会直接挂住。截断的局按"还活着"计入 truncated，不计为死亡。

    ``env`` 可以复用一个专用的评测环境；**绝不能**传训练用的那个环境实例，
    否则训练回合的状态会被评测冲掉。

    ``seed_base`` 打开**固定评测集**：第 i 局在 ``env.reset()`` 之前用
    ``seed_base + i`` 给全局 ``random`` 播种，于是第 i 局的管道序列只由 i 决定，
    与"这个模型前面几局活了多久"无关。

    为什么必须逐局播种
    ------------------
    管道生成用的是全局 ``random`` 模块。只在评测开始时播一次种的话，两个模型
    只要存活局长不同（几乎总是不同），从第 2 局起消耗掉的随机数数量就不同，
    之后每一局的管道都**错位**—— "固定种子"实际上只保证了第 1 局可比。
    这个缺陷让此前所有跨存档比较都不可信，是 docs/IMPROVEMENT_PLAN.md 里的 D0。

    两个配套的细节，缺一条都会让"固定"失效：

    1. **探索噪声必须走独立的随机源**。epsilon>0 时如果动作噪声也从全局流里抽，
       不同模型在同一局里抽到的次数不同，管道照样错位。
    2. **退出前恢复全局随机状态**。训练中每 500 局就会评测一次，如果评测把
       全局 ``random`` 重新播种了还不还原，训练本身的随机流就被评测劫持了。
    """
    if max_decisions is None:
        max_decisions = cfg['max_episode_decisions']
    own_env = env is None
    if own_env:
        env = make_env(cfg)

    stacker = FrameStack(cfg['frame_stack'])
    pipes, lengths, returns = [], [], []
    was_trunc = []
    q_max_all, q_gap_all = [], []
    n_truncated = 0

    # 固定评测集：借用全局随机流，用完必须原样还回去（见 docstring 第 2 点）
    saved_random_state = random.getstate() if seed_base is not None else None
    # 探索噪声用独立随机源，不去动全局流（第 1 点）。异或一个常数只是为了
    # 让它和管道的种子序列错开，没有别的含义。
    explore_rng = random.Random(seed_base ^ 0x5EED) if seed_base is not None else None

    try:
        for i in range(n_episodes):
            if seed_base is not None:
                random.seed(seed_base + i)
            stack = stacker.reset(env.reset())  # env.reset() 已是 (80,80) uint8
            phi = env.current_potential()
            n_dec, ep_ret = 0, 0.0
            while True:
                noise = explore_rng if explore_rng is not None else random
                if epsilon > 0 and noise.random() < epsilon:
                    action = sample_random_action(cfg, explore_rng)
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
                    was_trunc.append(not done)
                    n_truncated += int(not done)
                    break
                stack = stacker.push(frame)
    finally:
        # 无论正常结束还是抛异常，都把全局随机流还回去
        if saved_random_state is not None:
            random.setstate(saved_random_state)

    out = dict(
        episodes=n_episodes,
        pipes_mean=float(np.mean(pipes)),
        pipes_std=float(np.std(pipes)),
        pipes_max=int(np.max(pipes)),
        pipes_median=float(np.median(pipes)),
        pipes_iqm=iqm(pipes),
        # 截断局按右删失处理的平均存活。策略越强截断越多，普通均值
        # 的低估就越严重 —— 报告真实水平时看这个。
        pipes_censored_mean=censored_mean(pipes, was_trunc),
        pipes_q25=float(np.percentile(pipes, 25)),
        pipes_q75=float(np.percentile(pipes, 75)),
        len_mean=float(np.mean(lengths)),
        return_mean=float(np.mean(returns)),
        truncated=n_truncated,
        max_decisions=max_decisions,
    )
    if q_stats and q_max_all:
        out['q_max_mean'] = float(np.mean(q_max_all))
        out['q_gap_abs_mean'] = float(np.mean(np.abs(q_gap_all)))
    return out
