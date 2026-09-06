"""从 runs/<时间戳>/*.csv 画训练诊断图。

    python plot.py runs/20260812_101530
    python plot.py runs/xxx --out mydiag.png

关键面板是第 2、3 张：Q 值曲线上叠加目标网络同步的竖线，以及对数轴的 loss。
这两张图能直接读出"目标是不是在动""价值有没有在传播" —— 旧管线的画图脚本
用正则去抓一行 emoji 日志，只拿得到 (episode, score, max, avg100)，
所以 loss 只有 0.002 而一根管道值 +20 这种铁证，13 小时里从未被看见。
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import pandas as pd                  # noqa: E402


def smooth(y, w):
    if len(y) < w or w < 2:
        return y
    return pd.Series(y).rolling(w, min_periods=1).mean().values


def read_csv(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def main():
    p = argparse.ArgumentParser(description="Plot diagnostics for a training run")
    p.add_argument('run_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--smooth', type=int, default=100, help='episode smoothing window')
    a = p.parse_args()

    ep = read_csv(a.run_dir, 'episodes.csv')
    tr = read_csv(a.run_dir, 'train.csv')
    ev = read_csv(a.run_dir, 'eval.csv')
    if ep is None:
        raise SystemExit("no (non-empty) episodes.csv in %s" % a.run_dir)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle("Flappy Bird DQN diagnostics - %s" % os.path.basename(a.run_dir),
                 fontsize=13)

    # 目标网络同步发生的梯度步（train.csv 的 syncs 是累计计数）
    sync_grads = []
    if tr is not None and len(tr) > 1 and 'syncs' in tr:
        inc = tr['syncs'].diff().fillna(0) > 0
        sync_grads = tr.loc[inc, 'grad_step'].tolist()

    # 长跑会有几百次同步，全画出来会糊成一片色块，把 loss 和 Q 曲线盖住。
    # 超过 60 条就等距抽稀 —— 竖线的作用是标出"同步大致发生在哪"，
    # 不是逐次点名。n_syncs 仍用完整计数报告。
    n_syncs = len(sync_grads)
    if n_syncs > 60:
        step = n_syncs / 60.0
        sync_marks = [sync_grads[int(i * step)] for i in range(60)]
    else:
        sync_marks = sync_grads

    # ---- 1. 过管道数（主指标，与奖励尺度无关，跨版本可比） ----
    ax = axes[0][0]
    ax.plot(ep.episode, ep.pipes, lw=0.4, alpha=0.25, color='steelblue',
            label='per episode')
    ax.plot(ep.episode, ep.recent100_pipes, lw=1.8, color='navy', label='recent 100')
    ax.axhline(0.63, ls=':', color='gray', lw=1.2,
               label='random baseline (p_flap=0.2) = 0.63')
    ax.axhline(1.3, ls='--', color='firebrick', lw=1.2,
               label='old pipeline after 100k eps = ~1.3')
    ax.set_xlabel('episode'); ax.set_ylabel('pipes passed')
    ax.set_title('1. Pipes passed (primary metric)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 2. Q 值 + 目标网络同步竖线 ----
    ax = axes[0][1]
    if tr is not None:
        ax.plot(tr.grad_step, tr.q_mean, color='darkgreen', lw=1.5, label='q_mean')
        ax.fill_between(tr.grad_step, tr.q_mean - tr.q_std, tr.q_mean + tr.q_std,
                        color='darkgreen', alpha=0.15, label='+-q_std')
        ax.plot(tr.grad_step, tr.target_q_mean, color='orange', lw=1.0, ls='--',
                label='target_q_mean')
        for i, g in enumerate(sync_marks):
            ax.axvline(g, color='crimson', alpha=0.25, lw=0.7,
                       label='target sync' if i == 0 else None)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('gradient step'); ax.set_ylabel('Q')
        ax.set_title('2. Q values + target syncs (%d syncs%s)'
                     % (n_syncs, ', subsampled' if n_syncs > 60 else ''))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- 3. loss 对数轴 + 同步竖线（应看到同步后的锯齿） ----
    ax = axes[1][0]
    if tr is not None:
        ax.plot(tr.grad_step, tr.loss, color='purple', lw=0.8, alpha=0.6, label='loss')
        ax.plot(tr.grad_step, smooth(tr.loss.values, 10), color='purple', lw=1.8,
                label='loss (smoothed)')
        for g in sync_marks:
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
    if tr is not None:
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
    if ev is not None:
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
    if tr is not None:
        print("  gradient steps      : %d" % tr.grad_step.iloc[-1])
        print("  target syncs        : %d" % n_syncs)
        print("  q_mean  first/last  : %.3f / %.3f"
              % (tr.q_mean.iloc[0], tr.q_mean.iloc[-1]))
        print("  loss    first/last  : %.5f / %.5f"
              % (tr.loss.iloc[0], tr.loss.iloc[-1]))


if __name__ == '__main__':
    main()
