# 未在本机验证 —— 云端只跑过下面的自测（合成数据），从未见过真实的 diag.csv/eval.csv
"""诊断量（diag.csv）和贪婪评测分数（eval.csv）之间的纵向相关性分析。

背景：UPGRADE_ANALYSIS.md 的因果链里，"动作差距塌缩 -> argmax 翻转 -> 成绩震荡"
这一环从未被直接验证过（网络内部诊断和分数震荡目前只是"同时发生"，不是证明了
因果关系）。flappy/diagnostics.py 已经把 dorm_fc / eff95_fc / q_gap_median /
q_ceil_ratio / argmax_flip 这些量接进了训练循环的 diag.csv，但截至写这个脚本时，
本机还没有一次跑完整个训练、产出过真实的 diag.csv —— 只有 --smoke 到 ep2000 的
自检记录。这个脚本就是等那份数据跑出来之后，直接拿来验证因果链最后一环用的。

用法（在本机，diag.csv/eval.csv 都在 run_dir 下之后）：
    python3 experiments/analyze_diag_vs_score.py runs/<run_dir>

不带参数运行 = 自测模式：构造合成数据验证 spearman 实现和 csv 合并逻辑本身没写错，
不依赖真实训练数据，可以在任何装了 numpy 的机器上跑（不需要 torch/pygame）。

看什么算数：
- 每个诊断量都配了一个"预期方向"（比如 dorm_fc 越高、分数应该越低）。
- 只依赖 5-40 个 episode 点（eval_every_episodes=500 时一次几小时的训练也就
  这个量级），spearman 相关系数本身噪声很大，|r|>0.99 这种自测里的数字不代表
  真实数据也会这么干净 —— 报告结果时必须说清样本点数，并且只把这个当**方向性**
  证据，不要拿单次相关系数当因果证明。
"""
import csv
import sys

import numpy as np


def _spearman(x, y):
    """Spearman 秩相关，手写实现（不依赖 scipy，只用 numpy）。"""
    def rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = ranks[order[i:j + 1]].mean()
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def merge_by_episode(eval_rows, diag_rows):
    """按 episode 对齐两份 csv。两者都是 eval_every_episodes 一条，理论上逐行对齐，
    但不假设顺序/行数完全一致（比如训练中途改过 eval_every_episodes），
    用字典按 episode 精确匹配更稳。"""
    diag_by_ep = {int(float(r['episode'])): r for r in diag_rows}
    merged = []
    for r in eval_rows:
        ep = int(float(r['episode']))
        if ep in diag_by_ep:
            merged.append((r, diag_by_ep[ep]))
    return merged


# (诊断量列名, 预期符号) —— 符号来自 UPGRADE_ANALYSIS.md 的因果链假设：
#   休眠越多 / 翻转越频繁 / 越贴近"永不死"上限 -> 分数应该越低（负相关）
#   有效秩越高 / 动作差距越大 -> 分数应该越高（正相关）
DIAG_METRICS_AND_EXPECTED_SIGN = [
    ('dorm_fc', -1),
    ('eff95_fc', +1),
    ('q_gap_median', +1),
    ('q_ceil_ratio', -1),
    ('argmax_flip', -1),
]


def analyze(run_dir):
    eval_rows = load_csv('%s/eval.csv' % run_dir)
    diag_rows = load_csv('%s/diag.csv' % run_dir)
    merged = merge_by_episode(eval_rows, diag_rows)
    print('%s: %d 个配对点（按 episode 对齐，eval.csv 共 %d 行，diag.csv 共 %d 行）'
          % (run_dir, len(merged), len(eval_rows), len(diag_rows)))
    if len(merged) < 8:
        print('警告：配对点数 < 8，spearman 相关系数在这个样本量下基本不可信，'
              '只能定性看符号，不要报具体数值。')
    score = np.array([float(e['eval_pipes_mean']) for e, d in merged])
    print('%-16s %12s  %10s' % ('指标', 'spearman r', '符合预期'))
    results = {}
    for name, expected_sign in DIAG_METRICS_AND_EXPECTED_SIGN:
        vals = np.array([float(d[name]) for e, d in merged])
        r = _spearman(score, vals)
        matches = (r * expected_sign > 0)
        results[name] = r
        print('%-16s %12.3f  %10s' % (name, r, '是' if matches else '否'))
    return results


def _self_test():
    print('=== 自测 1/3：spearman 实现在已知数据上的表现 ===')
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y_pos = x * 2 + rng.normal(scale=0.1, size=n)
    y_neg = -x * 2 + rng.normal(scale=0.1, size=n)
    y_none = rng.normal(size=n)
    r_pos, r_neg, r_none = _spearman(x, y_pos), _spearman(x, y_neg), _spearman(x, y_none)
    print('强正相关样本 -> r=%.3f（预期接近 +1）' % r_pos)
    print('强负相关样本 -> r=%.3f（预期接近 -1）' % r_neg)
    print('无关样本     -> r=%.3f（预期接近 0）' % r_none)
    assert r_pos > 0.9, 'spearman 正相关检测失败'
    assert r_neg < -0.9, 'spearman 负相关检测失败'
    assert abs(r_none) < 0.3, 'spearman 应该检测出接近零的相关，实际 %.3f' % r_none

    print('=== 自测 2/3：并列值（ties）处理 ===')
    x_ties = np.array([1, 2, 2, 3, 4], dtype=float)
    y_ties = np.array([1, 2, 2, 3, 4], dtype=float)
    r_ties = _spearman(x_ties, y_ties)
    print('完全相同（含并列）的两列 -> r=%.3f（预期接近 +1）' % r_ties)
    assert r_ties > 0.99, '并列值应该仍判定为完全正相关，实际 %.3f' % r_ties

    print('=== 自测 3/3：csv 读取 + episode 对齐 + 端到端相关性方向 ===')
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'eval.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'decision_step', 'n_eval_eps', 'eval_pipes_mean',
                        'eval_pipes_std', 'eval_pipes_max', 'eval_len_mean'])
            for ep, score in [(500, 10.0), (1000, 50.0), (1500, 20.0), (2000, 80.0),
                               (2500, 15.0), (3000, 90.0), (3500, 25.0), (4000, 70.0)]:
                w.writerow([ep, ep * 100, 20, score, 5, 100, 300])
        with open(os.path.join(d, 'diag.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'decision_step', 'n_probe', 'dorm_conv3', 'dorm_fc',
                        'never_fc', 'eff95_fc', 'q_gap_median', 'q_gap_mean',
                        'q_gap_tiny_frac', 'q_max_mean', 'q_ceil_ratio', 'argmax_flip',
                        'flap_frac'])
            # 手工构造：分数越高 -> dorm_fc 越低（负相关）、eff95_fc 越高（正相关），
            # 其余量随手填一个和分数不相关的常数附近的抖动，验证脚本不会瞎报相关。
            for ep, score, dorm, eff95 in [(500, 10.0, 0.75, 18), (1000, 50.0, 0.20, 45),
                                            (1500, 20.0, 0.60, 28), (2000, 80.0, 0.10, 50),
                                            (2500, 15.0, 0.68, 22), (3000, 90.0, 0.08, 52),
                                            (3500, 25.0, 0.55, 30), (4000, 70.0, 0.12, 48)]:
                w.writerow([ep, ep * 100, 512, 0.1, dorm, 0.1, eff95,
                            1.0, 2.0, 0.1, 10.0, 0.5, 0.05, 0.5])
        results = analyze(d)
        assert results['dorm_fc'] < -0.9, (
            '构造的强负相关没有被正确检测到: %.3f' % results['dorm_fc'])
        assert results['eff95_fc'] > 0.9, (
            '构造的强正相关没有被正确检测到: %.3f' % results['eff95_fc'])
    print('全部自测通过。')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        _self_test()
    else:
        analyze(sys.argv[1])
