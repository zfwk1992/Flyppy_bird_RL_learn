"""诊断：波峰存档和波谷存档在**网络内部**有什么区别？

背景
----
`docs/IMPROVEMENT_PLAN.md` 第 0 节 A 实测到：训练后期成绩在 26 根和 105 根之间
来回震荡 3-4 倍，而 loss / q_mean / td_abs **全程平稳、看不出任何异常**。
也就是说训练侧现有的全部标量指标对真实性能是瞎的。

那就换一类指标看。文献里两个专门用来描述"网络还剩多少可塑性"的量：

- **dormant neurons（休眠单元）**：训练过程中逐渐不再激活的单元。
  Sokar et al. 的 ReDo 方法就是周期性地检测并重置它们。
  判据用论文的归一化定义：某单元的平均激活除以该层平均激活 <= tau 即为 tau-dormant。
- **effective rank（有效秩）**：激活矩阵 95%/99% 方差需要几个主成分。
  网络表达力退化时它会下降。

本仓库**自己就测到过**这个现象：`docs/learn/12-network-sizing.md` 里
fc512 有 14-25% 的单元从不激活，而且**更难的任务死单元更多**（25.4% vs 14.1%）。
当时的处理是把 fc 砍到 256 —— 但砍宽度并不修复可塑性丢失的机制，
休眠单元照样会重新长出来。

这个脚本回答的问题
------------------
把**同一批观测**喂给不同的存档，比较它们的休眠率和有效秩。
如果波谷存档（resume_0）的休眠率明显高于波峰存档（best），
那么"可塑性丢失/休眠单元"就是现象 A 的一个有实据的候选机制，
ReDo / 周期性重置这类手段才值得做。
如果两者没区别，这条就该排除，别浪费一次重训。

**关键**：所有存档必须吃**完全相同**的输入，否则比的是输入分布不是网络。
所以先用第一个存档跑出一批观测存下来，再喂给所有存档。

用法：
    python experiments/netdiag_dormant.py
    python experiments/netdiag_dormant.py --episodes 6 --tau 0.025
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy import checkpoint                                    # noqa: E402
from flappy.rollout import FrameStack, make_env, skip_step       # noqa: E402

CKPTS = [
    ('best     (ep15493, 波峰)', 'runs/final_v1/best.pt'),
    ('resume_1 (ep~19.3k)     ', 'runs/final_v1/resume_1.pt'),
    ('resume_0 (ep19831, 波谷)', 'runs/final_v1/resume_0.pt'),
]


def collect_observations(net, cfg, device, n_episodes, seed_base, stride):
    """用一个策略跑几局，把途中的帧栈存下来当成公共测试输入。

    只用来产生"有代表性的观测"，用哪个策略产生的都行 —— 重点是**所有存档
    看到的是同一批**。
    """
    import random
    env = make_env(cfg)
    stacker = FrameStack(cfg['frame_stack'])
    saved = random.getstate()
    out = []
    try:
        for i in range(n_episodes):
            random.seed(seed_base + i)
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
            for t in range(cfg['max_episode_decisions']):
                if t % stride == 0:
                    out.append(stack.copy())
                with torch.no_grad():
                    q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                frame, _, done, _, phi = skip_step(env, int(q.argmax()), phi, cfg)
                if done:
                    break
                stack = stacker.push(frame)
    finally:
        random.setstate(saved)
    return np.stack(out)


def activations_of(net, obs, device, batch=256):
    """抓 conv3 和 fc 的激活（都取 ReLU 之后）。"""
    acts = {}
    handles = []

    def hook(name):
        def fn(_m, _inp, out):
            a = torch.relu(out).detach()
            a = a.flatten(1) if a.dim() > 2 else a
            acts.setdefault(name, []).append(a.cpu().numpy())
        return fn

    handles.append(net.conv3.register_forward_hook(hook('conv3')))
    handles.append(net.fc.register_forward_hook(hook('fc')))
    try:
        with torch.no_grad():
            for i in range(0, len(obs), batch):
                net(torch.from_numpy(obs[i:i + batch]).to(device))
    finally:
        for h in handles:
            h.remove()
    return {k: np.concatenate(v, 0) for k, v in acts.items()}


def describe(x, tau):
    """x: (N, dim) 的激活矩阵。"""
    mean_act = x.mean(0)                       # 每个单元的平均激活
    layer_mean = mean_act.mean()
    # ReDo 的归一化 tau-dormant 判据
    score = mean_act / (layer_mean + 1e-12)
    dormant = int((score <= tau).sum())
    # 本仓库 12-network-sizing.md 用的"从不激活"判据，便于和历史数字对照
    never = int((x.max(0) <= 1e-8).sum())
    centred = x - x.mean(0)
    s = np.linalg.svd(centred, compute_uv=False)
    ratio = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    return dict(dim=x.shape[1], dormant=dormant, never=never,
                eff95=int(np.searchsorted(ratio, 0.95) + 1),
                eff99=int(np.searchsorted(ratio, 0.99) + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--stride', type=int, default=3, help='每隔几次决策采一个观测')
    p.add_argument('--tau', type=float, default=0.025, help='ReDo 的休眠阈值')
    p.add_argument('--own-data', action='store_true',
                   help='每个存档用**自己**跑出来的观测测休眠率，而不是共用一批。'
                        '共用一批的问题是后面的存档在别人的状态分布上被测，'
                        '休眠率可能被这个分布错配抬高；这个开关用来排除那种可能。')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)

    have = [(lbl, path) for lbl, path in CKPTS if os.path.exists(path)]
    if not have:
        raise SystemExit('runs/final_v1/ 里一个存档都没有')

    obs = None
    if a.own_data:
        print('模式：每个存档用**自己**的观测（排除状态分布错配这个混淆因素）\n')
    else:
        # 用第一个存档产生公共观测集
        net0, cfg0, _ = checkpoint.load_for_inference(have[0][1], device)
        obs = collect_observations(net0, cfg0, device, a.episodes,
                                   cfg0['eval_seed_base'], a.stride)
        print('公共观测集: %d 帧栈，形状 %s（由 %s 产生，所有存档吃同一批）\n'
              % (len(obs), obs.shape[1:], have[0][0].strip()))

    print('%-26s %-6s %5s %8s %8s %7s %7s' %
          ('存档', '层', '维度', 'dormant', '从不激活', 'eff95', 'eff99'))
    print('-' * 78)
    for lbl, path in have:
        net, cfg, _ = checkpoint.load_for_inference(path, device)
        cur = obs
        if cur is None:                    # --own-data：这个存档自己跑一批
            cur = collect_observations(net, cfg, device, a.episodes,
                                       cfg['eval_seed_base'], a.stride)
            print('  (%s 自己的观测 %d 帧)' % (lbl.strip(), len(cur)))
        acts = activations_of(net, cur, device)
        for layer in ('conv3', 'fc'):
            d = describe(acts[layer], a.tau)
            print('%-26s %-6s %5d %5d(%4.1f%%) %4d(%4.1f%%) %7d %7d'
                  % (lbl, layer, d['dim'],
                     d['dormant'], 100.0 * d['dormant'] / d['dim'],
                     d['never'], 100.0 * d['never'] / d['dim'],
                     d['eff95'], d['eff99']))
        # 动作差距：churn 假说预测波谷存档的 |Q1-Q0| 更小（动作更难区分）
        with torch.no_grad():
            q = net(torch.from_numpy(cur).to(device))
        gap = (q[:, 1] - q[:, 0]).abs()
        print('%-26s %-6s  mean|Q1-Q0|=%.4f  median=%.4f  <0.01 的比例=%.1f%%'
              % ('', 'gap', float(gap.mean()), float(gap.median()),
                 100.0 * float((gap < 0.01).float().mean())))
        print()

    print('怎么读：')
    print('  - 波谷存档 dormant 明显更高 -> 可塑性丢失是候选机制，ReDo/重置值得试')
    print('  - 三者 dormant 差不多      -> 排除这条，别为此重训')
    print('  - 波谷存档 |Q1-Q0| 明显更小 -> 动作差距塌缩，指向 policy churn')


if __name__ == '__main__':
    main()
