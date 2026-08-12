"""
从 runs/<timestamp>/*.csv 画训练诊断图。

相对 plot_training_progress.py 的改进
------------------------------------
旧脚本用正则去抓一行 emoji 日志（plot_training_progress.py:21），
只能拿到 (episode, score, max, avg100) 四个量：
  - 看不到 loss，所以"loss 只有 0.002 而一根管道值 +20"这个铁证从未被发现
  - 看不到 Q 值，所以价值函数根本没在学也看不出来
  - 看不到目标网络同步时刻，所以"每次同步后近百局均分掉 26 分"看不出来
  - 它的"最高分"子图在 stable 那两轮里恒为 99999（max_score 每局被重置的 bug）

本脚本读结构化 CSV，关键面板是第 2、3 张：
Q 值曲线上叠加目标网络同步的竖线，以及对数轴的 loss —— 这两张图能直接
读出"目标是不是在动""价值有没有在传播"。

用法：
    python plot_training_v2.py runs/20260811_221138
    python plot_training_v2.py runs/xxx --out mydiag.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(y, w):
    if len(y) < w or w < 2:
        return y
    return pd.Series(y).rolling(w, min_periods=1).mean().values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--smooth', type=int, default=100, help='episode smoothing window')
    a = p.parse_args()

    ep_path = os.path.join(a.run_dir, 'episodes.csv')
    tr_path = os.path.join(a.run_dir, 'train.csv')
    ev_path = os.path.join(a.run_dir, 'eval.csv')

    ep = pd.read_csv(ep_path) if os.path.exists(ep_path) else None
    tr = pd.read_csv(tr_path) if os.path.exists(tr_path) else None
    ev = pd.read_csv(ev_path) if os.path.exists(ev_path) else None

    if ep is None or len(ep) == 0:
        raise SystemExit("no episodes.csv in %s" % a.run_dir)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle("Flappy Bird DQN diagnostics - %s" % os.path.basename(a.run_dir),
                 fontsize=13)

    # 目标网络同步发生的梯度步（train.csv 的 syncs 是累计计数）
    sync_grads = []
    if tr is not None and len(tr) > 1 and 'syncs' in tr:
        inc = tr['syncs'].diff().fillna(0) > 0
        sync_grads = tr.loc[inc, 'grad_step'].tolist()

    # ---- 1. 过管道数（主指标，与奖励尺度无关，跨版本可比） ----
    ax = axes[0][0]
    ax.plot(ep.episode, ep.pipes, lw=0.4, alpha=0.25, color='steelblue', label='per episode')
    ax.plot(ep.episode, ep.recent100_pipes, lw=1.8, color='navy', label='recent 100')
    ax.axhline(0.63, ls=':', color='gray', lw=1.2,
               label='random baseline (p_flap=0.2) = 0.63')
    ax.axhline(1.3, ls='--', color='firebrick', lw=1.2,
               label='old pipeline after 100k eps = ~1.3')
    ax.set_xlabel('episode'); ax.set_ylabel('pipes passed')
    ax.set_title('1. Pipes passed (primary metric)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 2. Q 值 + 目标网络同步竖线（旧管线完全没有这张图） ----
    ax = axes[0][1]
    if tr is not None and len(tr):
        ax.plot(tr.grad_step, tr.q_mean, color='darkgreen', lw=1.5, label='q_mean')
        ax.fill_between(tr.grad_step, tr.q_mean - tr.q_std, tr.q_mean + tr.q_std,
                        color='darkgreen', alpha=0.15, label='+-q_std')
        ax.plot(tr.grad_step, tr.target_q_mean, color='orange', lw=1.0, ls='--',
                label='target_q_mean')
        for i, g in enumerate(sync_grads):
            ax.axvline(g, color='crimson', alpha=0.25, lw=0.7,
                       label='target sync' if i == 0 else None)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('gradient step'); ax.set_ylabel('Q')
        ax.set_title('2. Q values + target syncs (%d syncs)' % len(sync_grads))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 3. loss 对数轴 + 同步竖线（应看到同步后的锯齿） ----
    ax = axes[1][0]
    if tr is not None and len(tr):
        ax.plot(tr.grad_step, tr.loss, color='purple', lw=0.8, alpha=0.6, label='loss')
        ax.plot(tr.grad_step, smooth(tr.loss.values, 10), color='purple', lw=1.8,
                label='loss (smoothed)')
        for g in sync_grads:
            ax.axvline(g, color='crimson', alpha=0.2, lw=0.7)
        ax.set_yscale('log')
        ax.set_xlabel('gradient step'); ax.set_ylabel('Huber loss (log)')
        ax.set_title('3. Loss - expect a sawtooth at each sync')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # ---- 4. epsilon + 每局决策数 ----
    ax = axes[1][1]
    ax.plot(ep.decision_step, ep.epsilon, color='teal', lw=1.5)
    ax.set_xlabel('decision step'); ax.set_ylabel('epsilon', color='teal')
    ax.set_title('4. Exploration schedule & episode length')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ep.decision_step, smooth(ep.ep_len_decisions.values, a.smooth),
             color='sienna', lw=1.5)
    ax2.set_ylabel('decisions / episode', color='sienna')

    # ---- 5. TD 误差与梯度范数（诊断裁剪是否过紧） ----
    ax = axes[2][0]
    if tr is not None and len(tr):
        ax.plot(tr.grad_step, tr.td_abs_mean, color='darkblue', lw=1.2,
                label='|TD| mean')
        ax.plot(tr.grad_step, tr.td_abs_max, color='lightblue', lw=0.8, alpha=0.7,
                label='|TD| max')
        ax.set_xlabel('gradient step'); ax.set_ylabel('|TD error|')
        ax.set_title('5. TD error & grad norm')
        ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
        ax3 = ax.twinx()
        ax3.plot(tr.grad_step, tr.grad_norm, color='red', lw=1.0, alpha=0.6)
        ax3.set_ylabel('grad norm (pre-clip)', color='red')

    # ---- 6. 贪婪评测 vs 训练时表现 ----
    ax = axes[2][1]
    ax.plot(ep.episode, ep.recent100_pipes, color='navy', lw=1.2,
            label='train recent100 (with eps)')
    if ev is not None and len(ev):
        ax.errorbar(ev.episode, ev.eval_pipes_mean, yerr=ev.eval_pipes_std,
                    fmt='o-', color='darkorange', capsize=3, lw=1.5,
                    label='greedy eval')
    ax.set_xlabel('episode'); ax.set_ylabel('pipes')
    ax.set_title('6. Greedy eval vs behaviour policy')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = a.out or os.path.join(a.run_dir, 'diagnostics.png')
    plt.savefig(out, dpi=110)
    print("saved %s" % out)

    # ---- 文字摘要 ----
    print("\nsummary")
    print("  episodes            : %d" % len(ep))
    print("  decisions           : %d" % ep.decision_step.iloc[-1])
    print("  recent100_pipes end : %.3f  (max seen %.3f)"
          % (ep.recent100_pipes.iloc[-1], ep.recent100_pipes.max()))
    print("  best single episode : %d pipes" % ep.pipes.max())
    if tr is not None and len(tr):
        print("  gradient steps      : %d" % tr.grad_step.iloc[-1])
        print("  target syncs        : %d" % len(sync_grads))
        print("  q_mean  first/last  : %.3f / %.3f" % (tr.q_mean.iloc[0], tr.q_mean.iloc[-1]))
        print("  loss    first/last  : %.5f / %.5f" % (tr.loss.iloc[0], tr.loss.iloc[-1]))


if __name__ == '__main__':
    main()
