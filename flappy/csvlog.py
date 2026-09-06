"""结构化 CSV 日志。

旧管线只往 stdout 打一行带 emoji 的文本，画图脚本再用正则去抓，
只能拿到 (episode, score, max, avg100) 四个量：
  - 看不到 loss，所以"loss 只有 0.002 而一根管道值 +20"这个铁证从未被发现
  - 看不到 Q 值，所以价值函数根本没在学也看不出来
  - 看不到目标网络同步时刻，所以"每次同步后近百局均分掉 26 分"看不出来
这正是所有问题能在 13 小时里都不被发现的直接原因。
"""

import csv
import os
from datetime import datetime

EPISODE_FIELDS = ['episode', 'decision_step', 'frames', 'pipes', 'ep_return',
                  'ep_len_decisions', 'epsilon', 'recent100_pipes',
                  'recent100_return', 'buffer_size', 'wall_s', 'terminated']
TRAIN_FIELDS = ['grad_step', 'decision_step', 'loss', 'td_abs_mean', 'td_abs_max',
                'q_mean', 'q_std', 'q_min', 'q_max', 'target_q_mean', 'grad_norm',
                'lr', 'epsilon', 'buffer_size', 'syncs', 'dec_per_s', 'grad_per_s']
EVAL_FIELDS = ['episode', 'decision_step', 'n_eval_eps', 'eval_pipes_mean',
               'eval_pipes_std', 'eval_pipes_max', 'eval_len_mean',
               # 下面几列是 2026-09-06 加的。
               # **判据仍然是 eval_pipes_mean** —— 分数近似几何分布，样本均值
               # 是它的充分统计量、最小方差估计（实测相对 SE：均值 20% <
               # 几何均值 25% < iqm 33% < 中位 35%）。iqm/median 只作为
               # 对离群值稳健的交叉验证，q25/q75 给区间估计。
               # truncated 必须单独记 —— 撞上决策上限的局真实分数是未知的
               # （>=上限），当成上限计入会**低估**，而且策略越强截断越多，
               # 会让"变强"看起来像"没变化"。censored 是按右删失校正过的
               # 平均存活（实测低估约 14%：95.1 -> 108.7）。
               'eval_pipes_iqm', 'eval_pipes_median',
               'eval_pipes_q25', 'eval_pipes_q75', 'eval_truncated',
               'eval_pipes_censored']
# 诊断列单独一个文件，见 flappy/diagnostics.py。
# 不并进 eval.csv 是因为它们的语义不同：eval.csv 是"考了多少分"，
# diag.csv 是"网络内部什么状态" —— 后者在成绩崩掉时才是唯一还有信号的东西。
from .diagnostics import DIAG_FIELDS  # noqa: E402  (放这里是为了让字段定义只有一处)


class CsvLogger:
    def __init__(self, path, fields, flush_every=50):
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.f.flush()
        self.n = 0
        self.flush_every = flush_every

    def write(self, **row):
        self.w.writerow(row)
        self.n += 1
        if self.n % self.flush_every == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()


class RunLogger:
    """一次训练的全部输出：三个 CSV + 一份纯文本 run.log。

    文本日志刻意只用 ASCII —— 旧代码在热循环里 print emoji，
    在 Windows cp1252 的 stdout 上会抛 UnicodeEncodeError 直接终止训练。
    """

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, 'run.log')
        self.episodes = CsvLogger(os.path.join(run_dir, 'episodes.csv'),
                                  EPISODE_FIELDS)
        # 训练行本身就稀疏（每 100 梯度步才一行），再攒 50 行才落盘的话，
        # monitor.py 要等到第 5000 个梯度步才能看到任何学习侧的数字。
        self.train = CsvLogger(os.path.join(run_dir, 'train.csv'), TRAIN_FIELDS,
                               flush_every=2)
        # 评测行很稀疏（每 500 局才一条），每行即刷，免得进程意外中断时全丢
        self.eval = CsvLogger(os.path.join(run_dir, 'eval.csv'), EVAL_FIELDS,
                              flush_every=1)
        self.diag = CsvLogger(os.path.join(run_dir, 'diag.csv'), DIAG_FIELDS,
                              flush_every=1)

    def say(self, msg):
        line = "[%s] %s" % (datetime.now().strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as fh:
            fh.write(line + "\n")

    def close(self):
        self.episodes.close()
        self.train.close()
        self.eval.close()
        self.diag.close()
