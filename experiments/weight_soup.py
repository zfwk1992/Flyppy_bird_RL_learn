"""权重平均（soup）+ 线性模式连通性检验。

动机
----
同一次训练里相隔约 30 分钟的两个存档，在**同一批固定关卡**上成绩差 3.7 倍
（`resume_1` 79.4 根 vs `resume_0` 21.5 根，见 docs/research/UPGRADE_ANALYSIS.md
第 1.1 节）。而 `docs/research/WHERE_IS_THE_PROBLEM.md` 又证明了这里面
大约一半是测量噪声、剩下一半是真实的策略变化。

如果这些存档只是在**同一个盆地里绕圈**，那么把权重平均一下就可能白捡一个
更稳的模型 —— 不用重训，零成本。

**但这个前提必须先验，不能假设。** 权重平均只在两个解**线性连通**
（中间没有损失壁垒）时才有意义。所以这个脚本做两件事：

1. **线性插值曲线**：沿 θ(α) = (1−α)·A + α·B 取若干点评测。
   曲线平滑/单调 -> 同一盆地，soup 可行；中间塌陷 -> 不同盆地，soup 无意义。
2. **soup 本身**：所有存档权重的均匀平均，与每个原料做配对比较。

**这不是已排除的第 ③ 条。** 那条是 EMA 软更新，作用在**TD 目标侧**、
每次同步平滑一次；这里是**行动网络侧**的事后权重平均，两回事。

主指标用 hazard（死亡数 / 总通过管道数），不是均分 —— 分数近似几何分布，
40 局均值的相对标准误就有 20%，而 hazard 的相对标准误是 1/sqrt(死亡数)。
截断的局按右删失处理（只进分母不进分子）。

用法：
    python experiments/weight_soup.py                      # 默认 base_s0 的三个存档
    python experiments/weight_soup.py --episodes 60 --interp
"""
import argparse
import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy import checkpoint                       # noqa: E402
from flappy.model import build_net                  # noqa: E402
from flappy.rollout import evaluate, make_env       # noqa: E402

DEFAULT_CKPTS = [
    'runs/base_s0/resume_1.pt',
    'runs/base_s0/resume_0.pt',
    'runs/base_s0/final.pt',
]


def hazard_of(res):
    """hazard = 死亡数 / 总通过管道数。

    evaluate() 已经把截断局按右删失算进了 pipes_censored_mean
    （= 总管道数 / 死亡数），所以取倒数即可。
    相对标准误 = 1/sqrt(死亡数)。
    """
    deaths = res['episodes'] - res['truncated']
    if deaths <= 0 or res['pipes_censored_mean'] <= 0:
        return float('nan'), float('nan'), deaths
    h = 1.0 / res['pipes_censored_mean']
    return h, h / math.sqrt(deaths), deaths


def report(label, res):
    h, se, deaths = hazard_of(res)
    print('%-30s hazard %.3f%% +- %.3f%%  (死亡 %2d/%d)   均值 %6.1f  中位 %5.1f  截断 %d'
          % (label, 100 * h, 100 * se, deaths, res['episodes'],
             res['pipes_mean'], res['pipes_median'], res['truncated']))
    return h, se


def blend(sds, weights):
    """按权重线性组合若干 state_dict。"""
    out = copy.deepcopy(sds[0])
    for k in out:
        if out[k].is_floating_point():
            acc = torch.zeros_like(out[k], dtype=torch.float64)
            for sd, w in zip(sds, weights):
                acc += sd[k].to(torch.float64) * w
            out[k] = acc.to(out[k].dtype)
        else:
            # 非浮点张量（如果有）不能平均，取第一个
            out[k] = sds[0][k].clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', default=DEFAULT_CKPTS)
    p.add_argument('--episodes', type=int, default=60)
    p.add_argument('--max-decisions', type=int, default=2000)
    p.add_argument('--interp', action='store_true', default=True,
                   help='沿第一个和最后一个存档之间做线性插值扫描')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)

    have = [c for c in a.ckpts if os.path.exists(c)]
    if len(have) < 2:
        raise SystemExit('至少需要两个存档，找到 %d 个' % len(have))

    sds, cfg = [], None
    for c in have:
        net, cfg, _ = checkpoint.load_for_inference(c, device)
        sds.append({k: v.detach().clone() for k, v in net.state_dict().items()})

    env = make_env(cfg)
    scratch = build_net(cfg, device)

    def ev(sd, label):
        scratch.load_state_dict(sd)
        scratch.eval()
        res = evaluate(scratch, cfg, device, a.episodes, epsilon=0.0,
                       max_decisions=a.max_decisions, env=env,
                       seed_base=cfg['eval_seed_base'])
        return report(label, res)

    print('固定评测集 seed_base=%d，每个模型 %d 关，上限 %d 决策\n'
          % (cfg['eval_seed_base'], a.episodes, a.max_decisions))

    print('--- 原料 ---')
    base = [ev(sd, os.path.relpath(c)) for sd, c in zip(sds, have)]

    print('\n--- soup（%d 个存档均匀平均）---' % len(sds))
    soup = blend(sds, [1.0 / len(sds)] * len(sds))
    h_soup, se_soup = ev(soup, 'SOUP(%d)' % len(sds))

    best_h = min(h for h, _ in base)
    best_i = [h for h, _ in base].index(best_h)
    d = best_h - h_soup                       # 正数 = soup 的 hazard 更低 = 更好
    comb = math.sqrt(base[best_i][1] ** 2 + se_soup ** 2)
    print('\nsoup vs 最好的原料（%s）：hazard 差 %+.3f%%，合并标准误 %.3f%% -> %s'
          % (os.path.relpath(have[best_i]), 100 * d, 100 * comb,
             '**soup 更好**' if d > 2 * comb else
             ('**soup 更差**' if d < -2 * comb else '没有区别（<2 个标准误）')))

    if a.interp and len(sds) >= 2:
        print('\n--- 线性模式连通性：θ(α) = (1−α)·%s + α·%s ---'
              % (os.path.relpath(have[0]), os.path.relpath(have[-1])))
        print('（中间塌陷 = 两个解不在同一盆地，权重平均就没有意义）')
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            ev(blend([sds[0], sds[-1]], [1 - alpha, alpha]), '  alpha=%.2f' % alpha)


if __name__ == '__main__':
    main()
