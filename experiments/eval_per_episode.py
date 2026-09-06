# 未在本机验证 —— 写完就跑，输出见 docs/research/
"""把固定评测集上**每一局**的分数打出来，而不是只给均值。

为什么需要
----------
`runs/*/eval.csv` 只记 mean/std/max。但这个任务的分数分布是**重尾**的
（同一份权重 40 局跑出 5 到 321 根），而"存活时间没有上限"意味着
20 局的均值可能被一两局的超长存活主导。

于是有一个一直没排除的可能：`docs/research/UPGRADE_ANALYSIS.md` 里那个
"成绩在 23.6 ~ 114.4 之间来回震荡"，**有多少是策略真的在抖，
有多少只是均值这个统计量在重尾分布上本来就不稳？**

这两件事的修法完全不同：
- 策略在抖   -> 要改算法
- 均值不稳   -> 要改**指标**（用中位数、或者加大评测局数），算法可能没病

判据：把两个存档在**同一批关卡**上逐局比。
- 如果差异摊在所有关卡上   -> 策略整体变差，是真的退化
- 如果差异集中在少数几关   -> 是重尾 + 少数关卡翻转，均值被带偏

用法：
    python experiments/eval_per_episode.py runs/base_s0/best.pt runs/base_s0/final.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy import checkpoint                                   # noqa: E402
from flappy.rollout import FrameStack, make_env, skip_step      # noqa: E402


@torch.no_grad()
def per_episode(net, cfg, device, n_episodes, seed_base, max_decisions):
    """逐局分数。播种方式和 rollout.evaluate 的固定集完全一致。"""
    env = make_env(cfg)
    stacker = FrameStack(cfg['frame_stack'])
    saved = random.getstate()
    out = []
    try:
        for i in range(n_episodes):
            random.seed(seed_base + i)
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
            n_dec = 0
            while True:
                q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                frame, _, done, info, phi = skip_step(env, int(q.argmax()), phi, cfg)
                n_dec += 1
                if done or n_dec >= max_decisions:
                    out.append((info['score'], n_dec, not done))
                    break
                stack = stacker.push(frame)
    finally:
        random.setstate(saved)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--max-decisions', type=int, default=2000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)

    cols = {}
    for path in a.checkpoints:
        net, cfg, _ = checkpoint.load_for_inference(path, device)
        r = per_episode(net, cfg, device, a.episodes,
                        cfg['eval_seed_base'], a.max_decisions)
        cols[path] = r
        s = np.array([x[0] for x in r], dtype=float)
        print('%-34s 均值 %6.1f  中位 %5.1f  最小 %3d  最大 %4d  截断 %d/%d'
              % (os.path.relpath(path), s.mean(), np.median(s),
                 int(s.min()), int(s.max()), sum(x[2] for x in r), len(r)))

    if len(cols) < 2:
        return
    names = list(cols)
    a0 = np.array([x[0] for x in cols[names[0]]], dtype=float)
    a1 = np.array([x[0] for x in cols[names[1]]], dtype=float)
    d = a1 - a0
    print('\n逐关卡对比（关卡 i 用 seed_base+i，两个存档看到的是同一批管道）')
    print('%5s %9s %9s %9s' % ('关卡', names[0].split('/')[-1],
                               names[1].split('/')[-1], '差'))
    order = np.argsort(-np.abs(d))
    for i in order[:12]:
        print('%5d %9.0f %9.0f %+9.0f' % (i, a0[i], a1[i], d[i]))

    tot = abs(d).sum()
    top3 = abs(d)[order[:3]].sum()
    print('\n均值差 %.1f 根' % (a1.mean() - a0.mean()))
    print('绝对差最大的 3 关贡献了总绝对差的 %.0f%%（%d 关里的 3 关）'
          % (100 * top3 / tot if tot else 0, len(d)))
    print('中位数差 %.1f 根  <- 如果中位差远小于均值差，说明均值被少数关卡带偏'
          % (np.median(a1) - np.median(a0)))
    # 有多少关卡的分数方向一致
    same = int(((a0 > np.median(a0)) == (a1 > np.median(a1))).sum())
    print('两个存档"高于各自中位"的关卡重合 %d/%d 关 -> 关卡难度是不是共同因素'
          % (same, len(d)))


if __name__ == '__main__':
    main()
