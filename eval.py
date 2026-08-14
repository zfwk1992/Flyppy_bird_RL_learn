"""评测入口：加载一个存档，跑 N 局贪婪策略，报告过管道数。

    python eval.py runs/xxx/best.pt --episodes 100
    python eval.py runs/xxx/best.pt --episodes 30 --q-stats
    python eval.py runs/xxx/best.pt --pipe-gap 150      # 钉死难度再比

评测走的是 flappy.rollout.evaluate —— 与训练中的定期评测完全同一份代码，
不存在任何按训练进度短路的分支。
"""

import argparse
import random

import numpy as np
import torch

from flappy import checkpoint
from flappy.rollout import evaluate


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument('checkpoint')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--epsilon', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--q-stats', action='store_true')
    p.add_argument('--max-decisions', type=int, default=2000,
                   help='cap on episode length; a good policy never dies')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='fixed gap in px. Only takes effect with --no-randomize.')
    # ---- 泛化测试：把环境换成模型训练时没见过的分布 ----
    p.add_argument('--no-randomize', dest='randomize', action='store_false',
                   default=None,
                   help='evaluate on the old fixed layout (gap 100, spacing 144, '
                        '8 heights) instead of the randomised one')
    p.add_argument('--gap-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=None,
                   help='override the randomised gap-size range, e.g. --gap-range 60 80 '
                        'to test on gaps narrower than anything seen in training')
    p.add_argument('--spacing-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=None, help='override the randomised pipe spacing range')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    device = torch.device(a.device)
    net, cfg, ckpt = checkpoint.load_for_inference(
        a.checkpoint, device,
        pipe_gap=a.pipe_gap,
        randomize_pipes=a.randomize,
        pipe_gap_range=tuple(a.gap_range) if a.gap_range else None,
        pipe_spacing_range=tuple(a.spacing_range) if a.spacing_range else None)

    random.seed(a.seed)
    np.random.seed(a.seed)

    res = evaluate(net, cfg, device, a.episodes, epsilon=a.epsilon,
                   max_decisions=a.max_decisions, q_stats=a.q_stats)

    print(checkpoint.describe(ckpt, cfg, a.checkpoint))
    # 环境必须显式打出来 —— 泛化测试的全部意义就在于"用哪个分布评的"
    trained_rand = ckpt.get('config', {}).get('randomize_pipes', False)
    if cfg['randomize_pipes']:
        env_desc = ("RANDOMISED gap %.0f-%.0f, spacing %.0f-%.0f"
                    % (tuple(cfg['pipe_gap_range']) + tuple(cfg['pipe_spacing_range'])))
    else:
        env_desc = "FIXED gap %d, spacing 144" % cfg['pipe_gap']
    same = (cfg['randomize_pipes'] == trained_rand
            and (not cfg['randomize_pipes']
                 or tuple(cfg['pipe_gap_range'])
                 == tuple(ckpt.get('config', {}).get('pipe_gap_range', ()))))
    print("  environment   : %s   [%s]"
          % (env_desc, "training distribution" if same else "SHIFTED - generalisation test"))
    print("  eval episodes : %d  (epsilon=%.3f)" % (res['episodes'], a.epsilon))
    print("  pipes mean    : %.3f +- %.3f" % (res['pipes_mean'], res['pipes_std']))
    print("  pipes median  : %.1f    max: %d" % (res['pipes_median'], res['pipes_max']))
    print("  ep len (dec)  : %.1f  (cap %d)" % (res['len_mean'], res['max_decisions']))
    print("  hit the cap   : %d / %d episodes (still alive when stopped)"
          % (res['truncated'], res['episodes']))
    print("  ep return     : %.4f" % res['return_mean'])
    if 'q_max_mean' in res:
        # Q 有饱和上限：gamma^12.5 / (1 - gamma^7.2) ~= 12.6 (gamma=0.99,
        # 管道间隔 7.2 次决策，生成到得分 12.5 次决策)。
        # 所以 "Q ~ 0.9 x pipes" 这条经验规律只在低分段成立；
        # 一个不死的策略 Q 会趋向 12.6，而不是趋向 pipes。
        print("  mean max-Q    : %.3f   (low-score rule: ~0.9 x pipes; "
              "ceiling for an immortal policy: ~12.6)" % res['q_max_mean'])
        print("  mean |Q1-Q0|  : %.3f   (action preference strength; "
              "near 0 means the net cannot tell the actions apart)"
              % res['q_gap_abs_mean'])


if __name__ == '__main__':
    main()
