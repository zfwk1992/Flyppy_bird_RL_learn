"""训练期诊断：专门测 loss / Q 曲线**看不见**的那些东西。

为什么需要这个模块
------------------
`docs/IMPROVEMENT_PLAN.md` 第 0 节 A 实测到：训练后期成绩在 26 根和 105 根之间
来回震荡 3-4 倍，而同一时间窗口里 `loss`、`q_mean`、`td_abs` **全都平稳**。
ep17000（26 根）和 ep16500（92 根）的训练侧标量完全无法区分。

也就是说：**现有的训练日志对 3.5 倍的真实性能下降是瞎的。**

`docs/research/UPGRADE_ANALYSIS.md` 把病因查到了可塑性丢失这条链上：

    单元休眠 -> 有效秩塌缩 -> 动作差距塌缩 -> 极小扰动翻转 argmax -> 成绩震荡

这个模块把这条链上的每一环都变成训练时能看到的一列数：

  dorm_conv3 / dorm_fc  休眠单元比例（ReDo 的归一化判据）
  eff95_fc              fc 激活的有效秩（95% 方差需要几个主成分）
  q_gap_median          动作差距的**中位数**
  q_ceil_ratio          max-Q 相对理论上限的比值，逼近 1 就是在高估
  argmax_flip           上次诊断以来贪婪动作翻转的状态比例  <- churn 的直接测量

**注意 `q_gap_median` 必须看中位数，不能看均值。** 实测中位从 1.099 掉到 0.322
（塌缩）的同时，均值从 4.41 涨到 8.02（上升）—— 分布变双峰了，看均值会得出
完全相反的结论。这个坑已经踩过一次。

探针集为什么用启发式策略生成
----------------------------
`argmax_flip` 必须在**同一批状态**上比较，否则比的是状态不是策略。所以探针集
一旦生成就永不改变。用当前策略采集会让探针集随训练漂移，上面几个指标全部
失去纵向可比性。

用一个不依赖网络的确定性启发式（追缝隙中心）在固定 seed 上跑，
既与被测网络无关，又能覆盖真实的中后段游戏状态。
"""

import random

import numpy as np
import torch

from game.flappy_env import PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT

DIAG_FIELDS = ['episode', 'decision_step', 'n_probe',
               'dorm_conv3', 'dorm_fc', 'never_fc', 'eff95_fc',
               'q_gap_median', 'q_gap_mean', 'q_gap_tiny_frac',
               'q_max_mean', 'q_ceil_ratio', 'argmax_flip', 'flap_frac']

# ReDo 论文的休眠阈值：某单元平均激活 / 该层平均激活 <= tau 即为 tau-休眠
DORMANT_TAU = 0.025


def _chase_gap(env):
    """朴素 bang-bang：把鸟往下一个缝隙中心赶。

    和 web/tools/dump_python_trace.py 里的那份是同一个策略。不追求成绩，
    只求覆盖足够多样、足够深入的真实游戏状态。
    """
    nxt = None
    for u in env.upperPipes:
        if u['x'] + PIPE_WIDTH > env.playerx:
            nxt = u
            break
    if nxt is None:
        return 0
    center = nxt['y'] + PIPE_HEIGHT + nxt['gap'] / 2.0
    return 1 if env.playery + PLAYER_HEIGHT / 2.0 > center else 0


def build_probe_set(cfg, env, n_states=512, stride=3, seed_base=None):
    """采集一批**永不改变**的探针状态。

    借用全局随机流播种，用完原样还回去 —— 和 `rollout.evaluate` 同一个道理，
    否则会劫持训练自己的随机序列。
    """
    from .rollout import FrameStack, skip_step

    if seed_base is None:
        seed_base = cfg['eval_seed_base'] + 900_000   # 和评测集错开，别复用同一批关卡
    stacker = FrameStack(cfg['frame_stack'])
    saved = random.getstate()
    out = []
    # 每局最多取这么多，逼着探针集横跨至少 8 个不同的关卡 —— 追缝隙的启发式
    # 能活很久，不设上限的话 512 个状态可能全来自同一局的同一段。
    per_ep = max(1, n_states // 8)
    try:
        ep = 0
        while len(out) < n_states and ep < 200:
            random.seed(seed_base + ep)
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
            taken = 0
            for t in range(cfg['max_episode_decisions']):
                if t % stride == 0:
                    out.append(stack.copy())
                    taken += 1
                    if len(out) >= n_states or taken >= per_ep:
                        break
                frame, _, done, _, phi = skip_step(env, _chase_gap(env), phi, cfg)
                if done:
                    break
                stack = stacker.push(frame)
            ep += 1
    finally:
        random.setstate(saved)
    return np.stack(out)


def q_ceiling(cfg):
    """一个**永不死亡**的策略的 Q 值上限。

    每过一根管道 +1，管道之间相隔 `d` 次决策，从生成到得分约 `lead` 次决策：

        Q_max ~= gamma^lead / (1 - gamma^d)

    两个常数都由环境几何算出来（管道以 5px/帧移动，一次决策 = frame_skip 帧）。
    随机化间距下取区间中点，所以这是个**近似值** —— 但 `q_max / ceiling` 是个
    相对指标，上限差几个百分点不影响"是不是贴着天花板"这个判断。
    """
    gamma = cfg['gamma']
    px_per_decision = 5.0 * cfg['frame_skip']
    spacing = (sum(cfg['pipe_spacing_range']) / 2.0 if cfg['randomize_pipes']
               else 144.0)
    d = spacing / px_per_decision                      # 管道之间的决策数
    lead = (283.0 - 57.0) / px_per_decision            # 生成处到得分处的决策数
    denom = 1.0 - gamma ** d
    return (gamma ** lead) / denom if denom > 1e-9 else float('inf')


def _describe_layer(x, tau=DORMANT_TAU, want_rank=False):
    mean_act = x.mean(0)
    layer_mean = float(mean_act.mean())
    dormant = float((mean_act <= tau * layer_mean).mean()) if layer_mean > 0 else 1.0
    never = float((x.max(0) <= 1e-8).mean())
    eff95 = -1
    if want_rank:
        s = np.linalg.svd(x - x.mean(0), compute_uv=False)
        ratio = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-12)
        eff95 = int(np.searchsorted(ratio, 0.95) + 1)
    return dormant, never, eff95


@torch.no_grad()
def compute(net, probe, cfg, device, prev_argmax=None, batch=256):
    """在探针集上算一轮诊断。返回 (指标字典, 这次的 argmax 向量)。

    有效秩只在 fc 上算：conv3 是 4608 维，SVD 明显更贵，而 fc 才是实测塌缩
    最严重的那一层（47% -> 74% 休眠，秩 44 -> 22）。
    """
    acts = {}
    handles = []

    def hook(name):
        def fn(_m, _i, out):
            a = torch.relu(out).detach()
            acts.setdefault(name, []).append(
                (a.flatten(1) if a.dim() > 2 else a).cpu().numpy())
        return fn

    handles.append(net.conv3.register_forward_hook(hook('conv3')))
    handles.append(net.fc.register_forward_hook(hook('fc')))
    qs = []
    try:
        for i in range(0, len(probe), batch):
            qs.append(net(torch.from_numpy(probe[i:i + batch]).to(device)).cpu())
    finally:
        for h in handles:
            h.remove()

    q = torch.cat(qs)
    gap = (q[:, 1] - q[:, 0]).abs()
    argmax = q.argmax(1).numpy()
    conv3 = np.concatenate(acts['conv3'], 0)
    fc = np.concatenate(acts['fc'], 0)

    dorm_conv3, _, _ = _describe_layer(conv3)
    dorm_fc, never_fc, eff95_fc = _describe_layer(fc, want_rank=True)
    ceiling = q_ceiling(cfg)
    q_max_mean = float(q.max(1).values.mean())

    out = dict(
        n_probe=len(probe),
        dorm_conv3=round(dorm_conv3, 4),
        dorm_fc=round(dorm_fc, 4),
        never_fc=round(never_fc, 4),
        eff95_fc=eff95_fc,
        q_gap_median=round(float(gap.median()), 4),
        q_gap_mean=round(float(gap.mean()), 4),
        # 动作差距接近 0 的状态比例：这些状态上一点点数值扰动就能翻转贪婪动作
        q_gap_tiny_frac=round(float((gap < 0.01).float().mean()), 4),
        q_max_mean=round(q_max_mean, 4),
        q_ceil_ratio=round(q_max_mean / ceiling, 4),
        # 上一次诊断以来贪婪动作变了的状态比例 —— churn 的直接测量
        argmax_flip=(round(float((argmax != prev_argmax).mean()), 4)
                     if prev_argmax is not None else ''),
        flap_frac=round(float(argmax.mean()), 4),
    )
    return out, argmax
