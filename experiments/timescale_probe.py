# 未在本机验证 —— 写完直接跑，输出见 docs/research/
"""震荡发生在什么时间尺度上？—— 决定 EMA/Averaged-DQN 值不值得做。

问题
----
`docs/research/WHERE_IS_THE_PROBLEM.md` 确认成绩震荡真实存在（4.8 倍峰谷，
零假设只能解释 2.1 倍，p=0.0%）。候选解法 T1 是 EMA 软更新目标网络 /
Averaged-DQN，它们针对的是**目标值的方差**，也就是每次硬同步带来的阶跃。

但有一个时间尺度上的反证：`runs/base_s0` 里相邻两个评测点之间隔着
**41-47 次目标网络同步**。如果震荡是同步抖动，跨 45 次同步早就平均掉了，
根本不该在 500 局的采样分辨率上看得见。看得见就说明它更慢。

这个脚本用比同步更细的分辨率直接量策略变化，把两种可能分开：

  **A 同步锁定的抖动**：每次同步后 argmax 大量翻转，同步之间几乎不动
     -> 目标值方差是主因，EMA/Averaged-DQN 对症，T1 值得做
  **B 慢漂移**：翻转均匀摊在整个过程里，与同步边界无关
     -> EMA 只平滑单次阶跃，对慢漂移无能为力，T1 大概率无效

做法
----
从一个**训练后期**的存档（`resume_*.pt`，带目标网络和优化器）继续训练，
epsilon 钉在 `eps_final`（复现后期的利用阶段，而不是探索阶段），
每 `--every-grad` 个梯度步在**固定探针集**上测一次贪婪动作：

  flip_vs_prev : 与上一次测量相比翻转的状态比例  -> 瞬时抖动
  flip_vs_ref  : 与起点相比翻转的状态比例        -> 累积漂移

再按"距离上次同步多少梯度步"分箱平均 flip_vs_prev。
A 会在第一个箱子上出现尖峰，B 是平的。

用法：
    python experiments/timescale_probe.py --minutes 20
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy import checkpoint, diagnostics                       # noqa: E402
from flappy.agent import Agent                                   # noqa: E402
from flappy.config import resolve_config                         # noqa: E402
from flappy.replay import NStepAccumulator, ReplayBuffer         # noqa: E402
from flappy.rollout import FrameStack, make_env, skip_step       # noqa: E402


@torch.no_grad()
def greedy_actions(net, probe, device, batch=256):
    out = []
    for i in range(0, len(probe), batch):
        q = net(torch.from_numpy(probe[i:i + batch]).to(device))
        out.append(q.argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', default='runs/base_s0/resume_0.pt')
    p.add_argument('--minutes', type=float, default=20.0)
    p.add_argument('--every-grad', type=int, default=100,
                   help='每多少梯度步测一次策略。必须远小于 target_sync_grad_steps '
                        '(1000)，否则分辨不出同步周期内的结构')
    p.add_argument('--warmup', type=int, default=30000,
                   help='开始学习前先用当前策略填多少决策进缓冲区')
    p.add_argument('--buffer', type=int, default=60000)
    p.add_argument('--out', default='docs/research/timescale_probe.csv')
    a = p.parse_args()

    cfg = resolve_config(buffer_capacity=a.buffer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sync_every = cfg['target_sync_grad_steps']
    assert a.every_grad < sync_every, '采样间隔必须小于同步间隔，否则看不出周期内结构'

    agent = Agent(cfg, device)
    print(checkpoint.load_for_training(a.resume, agent, device))
    buffer = ReplayBuffer(cfg['buffer_capacity'], cfg['frame_stack'],
                          cfg['obs_w'], cfg['obs_h'], n_step=cfg['n_step'])
    nstep = NStepAccumulator(cfg['n_step'], cfg['gamma'])
    env = make_env(cfg)
    eval_env = make_env(cfg)

    probe = diagnostics.build_probe_set(cfg, eval_env,
                                        n_states=cfg['diag_probe_states'])
    print('探针集 %d 状态；同步间隔 %d 梯度步；采样间隔 %d 梯度步'
          % (len(probe), sync_every, a.every_grad))

    # epsilon 钉死在后期值：要复现的是"利用阶段的震荡"，不是探索阶段
    eps = cfg['eps_final']
    stacker = FrameStack(cfg['frame_stack'])
    stack = stacker.reset(env.reset())
    phi = env.current_potential()
    ep_pipes, recent = 0, []

    print('填充缓冲区 %d 决策（eps=%.3f）...' % (a.warmup, eps))
    t0 = time.time()
    rows = []
    ref_act = greedy_actions(agent.online, probe, device)
    prev_act = ref_act.copy()
    last_measure_grad = 0
    deadline = t0 + a.minutes * 60

    while time.time() < deadline:
        action = agent.act_epsilon_greedy(stack, eps)
        nxt, r, done, info, phi = skip_step(env, action, phi, cfg)
        for tr in nstep.push(stack, action, r, nxt, done):
            buffer.add(*tr)
        if done:
            recent.append(info['score'])
            if len(recent) > 100:
                recent.pop(0)
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
        else:
            stack = stacker.push(nxt)

        if len(buffer) >= max(cfg['batch_size'], a.warmup // cfg['frame_skip']):
            agent.learn(buffer)
            if agent.grad_steps - last_measure_grad >= a.every_grad:
                last_measure_grad = agent.grad_steps
                act = greedy_actions(agent.online, probe, device)
                rows.append(dict(
                    grad_step=agent.grad_steps,
                    syncs=agent.syncs,
                    # 距离上一次同步过了多少梯度步（0 = 刚同步完）
                    since_sync=agent.grad_steps % sync_every,
                    flip_vs_prev=float((act != prev_act).mean()),
                    flip_vs_ref=float((act != ref_act).mean()),
                    recent100=float(np.mean(recent)) if recent else float('nan'),
                ))
                prev_act = act
                if len(rows) % 25 == 0:
                    print('  grad=%d syncs=%d flip_prev=%.3f flip_ref=%.3f '
                          'recent100=%.1f  (%.0fs)'
                          % (agent.grad_steps, agent.syncs,
                             rows[-1]['flip_vs_prev'], rows[-1]['flip_vs_ref'],
                             rows[-1]['recent100'], time.time() - t0))

    if not rows:
        raise SystemExit('一条测量都没有：缓冲区可能没填满，把 --minutes 调大')

    import csv
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n写入 %s（%d 条测量，%d 次同步）'
          % (a.out, len(rows), rows[-1]['syncs'] - rows[0]['syncs']))

    # ---- 判据：按同步周期内的相位分箱 ----
    print('\n按"距上次同步的梯度步"分箱，看瞬时翻转率：')
    print('%14s %8s %12s' % ('距上次同步', '样本数', '平均翻转率'))
    bins = {}
    for r in rows:
        bins.setdefault(r['since_sync'], []).append(r['flip_vs_prev'])
    for k in sorted(bins):
        v = bins[k]
        print('%14d %8d %11.4f' % (k, len(v), float(np.mean(v))))

    first = bins.get(0, [])
    rest = [x for k, v in bins.items() if k != 0 for x in v]
    if first and rest:
        f0, fr = float(np.mean(first)), float(np.mean(rest))
        print('\n刚同步完那一箱: %.4f     其余: %.4f     比值 %.2f' % (f0, fr, f0 / fr))
        print('=> ' + ('**A 同步锁定**：同步后翻转显著更多，EMA/Averaged-DQN 对症'
                       if f0 > 1.5 * fr else
                       '**B 慢漂移**：翻转与同步边界无关，EMA 只平滑单次阶跃，'
                       '对这种漂移无能为力'))

    fr_all = [r['flip_vs_ref'] for r in rows]
    print('\n累积漂移 flip_vs_ref: 起点 %.3f -> 终点 %.3f（%d 梯度步内）'
          % (fr_all[0], fr_all[-1], rows[-1]['grad_step'] - rows[0]['grad_step']))
    print('瞬时翻转率中位 %.4f/次测量 -> 若各次独立累加，%d 次测量应达 %.2f；'
          % (float(np.median([r['flip_vs_prev'] for r in rows])), len(rows),
             min(1.0, float(np.median([r['flip_vs_prev'] for r in rows])) * len(rows))))
    print('实际只有 %.3f -> 差得越多说明翻转越是"来回摆"而不是"单向漂"'
          % fr_all[-1])


if __name__ == '__main__':
    main()
