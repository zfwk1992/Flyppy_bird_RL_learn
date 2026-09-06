"""训练实时监控：每隔几秒刷新一屏训练状态。

    python monitor.py                    # 自动盯最新的 runs/ 目录
    python monitor.py runs/<时间戳>       # 指定某次训练
    python monitor.py --interval 10      # 刷新间隔（秒），默认 5
    python monitor.py --once             # 打印一次就退出（适合脚本 / 日志）

刻意**不** import flappy/ 和 game/：只用标准库读 CSV。
这样训练在 Docker / 另一台机器上跑、监控对着拷贝过来的目录跑，都没问题；
也不会因为 game/resources.py 一被导入就初始化 pygame 而浪费资源。

run.log 每 500 局才一行，长跑时几乎看不出状态；这里直接读结构化 CSV，
`episodes.csv` 和 `train.csv` 一进一出都是 append，读到半行就丢掉那一行，
不会和正在写文件的训练进程打架。
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from datetime import timedelta

# 随机策略（p_flap=0.2）的基线成绩，来自 README"关键设计"第 5 条
RANDOM_BASELINE = 0.63

BLOCKS = " ▁▂▃▄▅▆▇█"


# ======================================================================
# 读取
# ======================================================================
def latest_run(runs_dir='runs'):
    if not os.path.isdir(runs_dir):
        return None
    cands = [d for d in glob.glob(os.path.join(runs_dir, '*')) if os.path.isdir(d)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def read_rows(path, tail=None):
    """读 CSV。丢掉最后一行不完整的记录 —— 训练进程可能正写到一半。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    if rows and any(v is None or v == '' for v in rows[-1].values()):
        rows.pop()
    return rows[-tail:] if tail else rows


def num(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


# ======================================================================
# 展示
# ======================================================================
def sparkline(values, width=48):
    """把一列数字压成一行方块图。空数据返回空串。"""
    if not values:
        return ''
    if len(values) > width:                 # 等距分桶取均值
        step = len(values) / width
        values = [sum(values[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
                  / max(len(values[int(i * step):max(int((i + 1) * step),
                                                     int(i * step) + 1)]), 1)
                  for i in range(width)]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return BLOCKS[1] * len(values)
    return ''.join(BLOCKS[min(int((v - lo) / (hi - lo) * 8) + 1, 8)] for v in values)


def bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def fmt_dur(seconds):
    return str(timedelta(seconds=int(max(seconds, 0))))


def phase_of(decision_step, cfg):
    """当前处于哪个训练阶段 —— 与 flappy/rollout.epsilon_at 的分段一致。"""
    warm = cfg.get('warmup_decisions', 0)
    a1 = cfg.get('eps_anneal1_decisions', 0)
    a2 = cfg.get('eps_anneal2_decisions', 0)
    if decision_step < warm:
        return 'warmup  (纯随机填缓冲区，不学习)', decision_step / max(warm, 1)
    t = decision_step - warm
    if t < a1:
        return 'anneal 1 (eps 1.0 -> 0.05)', t / max(a1, 1)
    if t < a1 + a2:
        return 'anneal 2 (eps 0.05 -> 0.01)', (t - a1) / max(a2, 1)
    return 'exploit (eps = 0.01)', 1.0


def axis_labels(values, width):
    """走势图下方的 min / max 标注，左右对齐到图的两端。"""
    lo, hi = '%.2f' % min(values), '%.2f' % max(values)
    pad = max(width - len(lo) - len(hi), 1)
    return '   ' + lo + ' ' * pad + hi


def render(run_dir, cfg, ep_rows, tr_rows, ev_rows):
    W = min(shutil.get_terminal_size((100, 30)).columns, 100)
    spark_w = W - 20
    out = []
    line = lambda s='': out.append(s)
    rule = lambda: line('─' * W)

    # 路径可能很长，截断成 .../父目录/运行目录，保证不撑破边框
    shown = run_dir
    if len(shown) > W - 26:
        shown = '...' + os.sep + os.sep.join(
            run_dir.replace('/', os.sep).split(os.sep)[-2:])

    line('╭' + '─' * (W - 2) + '╮')
    line(' Flappy Bird DQN  ·  %s' % shown)
    line('╰' + '─' * (W - 2) + '╯')

    if not ep_rows:
        line('  等待第一局结束…  (episodes.csv 还是空的)')
        return '\n'.join(out)

    last = ep_rows[-1]
    episode = int(num(last, 'episode'))
    dstep = int(num(last, 'decision_step'))
    wall = num(last, 'wall_s')
    eps = num(last, 'epsilon')
    r100 = num(last, 'recent100_pipes')
    buf = int(num(last, 'buffer_size'))

    # ---- 进度 ----
    phase, frac = phase_of(dstep, cfg)
    max_ep = int(cfg.get('max_episodes', 0) or 0)
    line(' 阶段  %s' % phase)
    line('       %s %5.1f%%' % (bar(frac), frac * 100))
    if max_ep:
        line(' 总进度 %s %d / %d 局' % (bar(episode / max_ep), episode, max_ep))
    rule()

    # ---- 核心数字 ----
    all_pipes = [num(r, 'pipes') for r in ep_rows]
    best_ep = max(all_pipes) if all_pipes else 0
    best_r100 = max(num(r, 'recent100_pipes') for r in ep_rows)
    verdict = '✓ 高于随机基线' if r100 > RANDOM_BASELINE else '… 仍在随机基线附近'
    line(' 近百局均分   %8.3f 根管道   (随机基线 %.2f)  %s'
         % (r100, RANDOM_BASELINE, verdict))
    line(' 历史最好均分 %8.3f          单局最高 %d 根' % (best_r100, best_ep))
    line(' 局数 %-8d 决策步 %-10d epsilon %.4f' % (episode, dstep, eps))
    line(' 缓冲区 %d / %d (%.0f%%)'
         % (buf, cfg.get('buffer_capacity', 0),
            100.0 * buf / max(cfg.get('buffer_capacity', 1), 1)))
    rule()

    # ---- 学习侧 ----
    if tr_rows:
        t = tr_rows[-1]
        line(' 梯度步 %-10d 目标网络同步 %-6d loss %.5f'
             % (num(t, 'grad_step'), num(t, 'syncs'), num(t, 'loss')))
        line(' Q 均值 %+8.3f  目标 Q %+8.3f  |TD| %.4f  梯度范数 %.3f'
             % (num(t, 'q_mean'), num(t, 'target_q_mean'),
                num(t, 'td_abs_mean'), num(t, 'grad_norm')))
        line(' 吞吐   %.0f 决策/秒   %.0f 梯度步/秒'
             % (num(t, 'dec_per_s'), num(t, 'grad_per_s')))
    else:
        need = cfg.get('warmup_decisions', 0)
        if dstep < need:
            line(' 尚未开始学习：warmup 还差 %d 次决策' % (need - dstep))
        else:
            # warmup 已过但 train.csv 还没有行 —— 是缓冲写入的延迟，不是没在学
            line(' 已在学习，等待第一批训练日志落盘（每 100 梯度步写一行）')
    rule()

    # ---- 评测 ----
    if ev_rows:
        e = ev_rows[-1]
        line(' 最近贪婪评测 @%d 局：%.2f ± %.2f 根   最高 %d   局长 %.0f'
             % (num(e, 'episode'), num(e, 'eval_pipes_mean'),
                num(e, 'eval_pipes_std'), num(e, 'eval_pipes_max'),
                num(e, 'eval_len_mean')))
    else:
        every = cfg.get('eval_every_episodes', 0)
        if every:
            line(' 首次贪婪评测在第 %d 局（还差 %d 局）'
                 % (every, every - episode % every))
    rule()

    # ---- 曲线 ----
    r100_series = [num(r, 'recent100_pipes') for r in ep_rows]
    line(' 近百局均分走势  %s' % sparkline(r100_series, spark_w))
    line(axis_labels(r100_series, spark_w + 13))
    if tr_rows:
        q_series = [num(r, 'q_mean') for r in tr_rows]
        line(' Q 均值走势      %s' % sparkline(q_series, spark_w))
        line(axis_labels(q_series, spark_w + 13))
    rule()

    # ---- 时间 ----
    speed = episode / max(wall, 1e-9)
    eta = (max_ep - episode) / speed if (max_ep and speed > 0) else None
    line(' 已运行 %s   %.1f 局/秒%s'
         % (fmt_dur(wall), speed,
            ('   预计剩余 %s' % fmt_dur(eta)) if eta else ''))
    # best.pt 的门槛：≥500 局 且 近百局均分创新高（2% 门槛 + 100 局冷却）
    # 注意不能拿 best_r100 当"存档时的分数"—— 那是历史最大值，而 best.pt 可能
    # 存于更早的某个高点。真实数值在存档文件里，而本脚本刻意不 import torch。
    if os.path.exists(os.path.join(run_dir, 'best.pt')):
        saved = ' 存档   best.pt ✓  (用 eval.py 看它存档时的真实成绩)'
    elif episode < 500:
        saved = ' 存档   best.pt 尚未产生（需 ≥500 局，还差 %d 局）' % (500 - episode)
    else:
        saved = ' 存档   best.pt 尚未产生（均分还没创过新高）'
    line(saved)
    line(' 刷新于 %s   Ctrl-C 退出监控（不影响训练）' % time.strftime('%H:%M:%S'))
    return '\n'.join(out)


# ======================================================================
def main():
    p = argparse.ArgumentParser(description="Live monitor for a training run")
    p.add_argument('run_dir', nargs='?', default=None,
                   help='默认自动选 runs/ 下最新修改的目录')
    p.add_argument('--interval', type=float, default=5.0, help='刷新间隔（秒）')
    p.add_argument('--once', action='store_true', help='打印一次就退出')
    a = p.parse_args()

    run_dir = a.run_dir or latest_run()
    if run_dir is None:
        raise SystemExit("runs/ 下没有找到任何训练目录。先跑 python train.py")
    if not os.path.isdir(run_dir):
        raise SystemExit("目录不存在: %s" % run_dir)

    cfg = {}
    cfg_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)

    while True:
        ep = read_rows(os.path.join(run_dir, 'episodes.csv'))
        tr = read_rows(os.path.join(run_dir, 'train.csv'))
        ev = read_rows(os.path.join(run_dir, 'eval.csv'))
        screen = render(run_dir, cfg, ep, tr, ev)

        if a.once:
            print(screen)
            return
        # 清屏后整屏重画：比逐行滚动好读，也不依赖 ANSI 光标控制
        os.system('cls' if os.name == 'nt' else 'clear')
        print(screen, flush=True)
        try:
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print("\n监控已退出（训练不受影响）")
            return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
