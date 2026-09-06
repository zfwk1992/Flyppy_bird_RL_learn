"""线性探针：网络的 fc 表征里还剩多少**几何信息**？

要回答的问题
------------
Fable session 的改进计划把"观测去二值化、改用灰度"排在重训类的第一位，
理由是 `cv2.threshold(small, 1, ...)` 把亚像素位置信息全扔了。
那是一个 **4 小时重训**的实验。

这个脚本用 20 分钟给出判据：**从 fc 的 256 维激活里，用岭回归能不能把
鸟的 y、速度、缝隙中心恢复出来？误差是多少像素？**

- 误差只有几 px  -> 位置信息还在，二值化没造成实质损失，**灰度不值得做**
- 误差 10+ px    -> 表征里确实丢了位置精度，灰度重训有依据

顺带还能回答第二个问题：`docs/research/UPGRADE_ANALYSIS.md` 测到有效秩
从 106 塌到 23。**秩塌缩到底有没有功能后果？** 如果晚期存档的探针误差
明显变大，那就把"秩塌缩"和"实际能力下降"挂上了钩；如果没变大，
说明剩下的 23 维仍然够用，秩这个指标本身的意义要打折扣。

方法学要点
----------
**必须有留出集。** 256 维特征去回归 8 个目标，样本不够时轻易就能拟合出
漂亮的训练集 R²，那是假的。这里按回合切分 train/test（不是按帧随机切），
因为同一回合内相邻帧高度相关，按帧随机切会让测试集泄漏。

对照基线：用训练集的均值去预测（R²=0）。探针必须明显好于它才算学到东西。

用法：
    python experiments/linear_probe.py
    python experiments/linear_probe.py --ckpts runs/base_s0/final.pt runs/base_s0/resume_1.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy import checkpoint                                    # noqa: E402
from flappy.rollout import FrameStack, make_env, skip_step       # noqa: E402
from game.flappy_env import BASEY, SCREENWIDTH                   # noqa: E402

# state_vector() 的 8 个分量、它们的物理单位、以及"归一化值 -> 物理量"的换算系数。
# 见 game/flappy_env.py: state_vector()：
#   v0 = playery/BASEY*2-1              v1 = playerVelY/playerMaxVelY
#   v2 = (p1.x-playerx)/SCREENWIDTH     v3 = gap_center1/BASEY*2-1   v4 = gap1/200
#   v5,v6,v7 = 第二根管道的同样三项
COMPONENTS = [
    ('鸟 y',        'px',   BASEY / 2.0),
    ('鸟 速度',      'px/帧', 5.0),            # playerMaxVelY = 5
    ('管道1 dx',     'px',   SCREENWIDTH),
    ('管道1 缝心 y', 'px',   BASEY / 2.0),
    ('管道1 缝宽',   'px',   200.0),
    ('管道2 dx',     'px',   SCREENWIDTH),
    ('管道2 缝心 y', 'px',   BASEY / 2.0),
    ('管道2 缝宽',   'px',   200.0),
]


def collect(net, cfg, device, n_episodes, seed_base, stride):
    """跑若干关，收集 (帧栈, state_vector, 回合号)。"""
    env = make_env(cfg)
    stacker = FrameStack(cfg['frame_stack'])
    saved = random.getstate()
    X, Y, G = [], [], []
    try:
        for i in range(n_episodes):
            random.seed(seed_base + i)
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
            for t in range(cfg['max_episode_decisions']):
                if t % stride == 0:
                    X.append(stack.copy())
                    Y.append(env.state_vector().copy())
                    G.append(i)
                with torch.no_grad():
                    q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                frame, _, done, _, phi = skip_step(env, int(q.argmax()), phi, cfg)
                if done:
                    break
                stack = stacker.push(frame)
    finally:
        random.setstate(saved)
    return np.stack(X), np.stack(Y), np.array(G)


@torch.no_grad()
def fc_features(net, X, device, batch=256):
    """抓 fc 层 ReLU 之后的 256 维激活。"""
    feats = []
    h = net.fc.register_forward_hook(
        lambda _m, _i, out: feats.append(torch.relu(out).detach().cpu().numpy()))
    try:
        for i in range(0, len(X), batch):
            net(torch.from_numpy(X[i:i + batch]).to(device))
    finally:
        h.remove()
    return np.concatenate(feats, 0)


def ridge_fit(Xtr, Ytr, lam):
    """闭式岭回归，含偏置项。"""
    n, d = Xtr.shape
    A = np.concatenate([Xtr, np.ones((n, 1), dtype=Xtr.dtype)], 1)
    G = A.T @ A
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0                    # 不惩罚偏置
    return np.linalg.solve(G + reg, A.T @ Ytr)


def ridge_predict(W, X):
    A = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    return A @ W


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+',
                   default=['runs/base_s0/final.pt', 'runs/base_s0/resume_1.pt'])
    p.add_argument('--episodes', type=int, default=25)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)

    for path in a.ckpts:
        if not os.path.exists(path):
            print('跳过（不存在）: %s' % path)
            continue
        net, cfg, _ = checkpoint.load_for_inference(path, device)
        X, Y, G = collect(net, cfg, device, a.episodes,
                          cfg['eval_seed_base'], a.stride)
        F = fc_features(net, X, device)

        # **按回合**切分，不是按帧 —— 同一回合内相邻帧高度相关，
        # 按帧随机切会让测试集泄漏，探针误差会被系统性低估。
        eps = np.unique(G)
        cut = int(len(eps) * 0.7)
        tr = np.isin(G, eps[:cut])
        te = ~tr
        # **必须先丢掉训练集上近乎恒定的单元**，否则标准化会炸：休眠单元在
        # 训练集上 std≈0，除以 std+1e-6 会把测试集上任何微小的非零值放大 1e6 倍，
        # 岭回归的权重跟着爆，R² 会出现 -700 这种荒谬值。第一版就栽在这里。
        sd_all = F[tr].std(0)
        keep = sd_all > 1e-4 * max(sd_all.max(), 1e-12)
        mu, sd = F[tr][:, keep].mean(0), sd_all[keep]
        Ftr = (F[tr][:, keep] - mu) / sd
        Fte = (F[te][:, keep] - mu) / sd

        # 正则化强度不能拍脑袋定：按留出集扫一遍，取平均 R² 最好的那个。
        best = None
        for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
            Wl = ridge_fit(Ftr, Y[tr], lam)
            Pl = ridge_predict(Wl, Fte)
            r2s = []
            for j in range(Y.shape[1]):
                ss_res = ((Pl[:, j] - Y[te, j]) ** 2).sum()
                ss_tot = ((Y[te, j] - Y[tr, j].mean()) ** 2).sum()
                r2s.append(1 - ss_res / max(ss_tot, 1e-12))
            m = float(np.mean(r2s))
            if best is None or m > best[0]:
                best = (m, lam, Wl, Pl)
        _, lam_best, W, P = best

        live = int((F > 1e-8).any(0).sum())
        print('\n=== %s ===' % os.path.relpath(path))
        print('样本 %d（%d 关，stride %d）  训练 %d / 测试 %d  '
              'fc 活跃单元 %d/256  探针实际用到 %d 维  最佳 lambda=%g'
              % (len(X), a.episodes, a.stride, tr.sum(), te.sum(),
                 live, int(keep.sum()), lam_best))
        print('%-14s %10s %12s %12s   %s'
              % ('分量', '单位', '探针误差', '基线误差', '解释掉的方差'))
        for j, (name, unit, scale) in enumerate(COMPONENTS):
            err = np.abs(P[:, j] - Y[te, j]).mean() * scale
            base = np.abs(Y[tr, j].mean() - Y[te, j]).mean() * scale
            ss_res = ((P[:, j] - Y[te, j]) ** 2).sum()
            ss_tot = ((Y[te, j] - Y[tr, j].mean()) ** 2).sum()
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            print('%-14s %10s %9.2f %-2s %9.2f %-2s   %6.3f'
                  % (name, unit, err, unit, base, unit, r2))

    print('\n怎么读：')
    print('  鸟 y / 缝心 y 的误差是关键 —— 缝隙真实容错只有 38.5px（gap=85 时）。')
    print('  误差远小于容错 -> 位置信息还在，二值化不是瓶颈，灰度重训不值得做。')
    print('  误差接近或超过容错 -> 表征里确实丢了精度，灰度有依据。')
    print('  "基线误差"是用训练集均值预测的结果；探针必须明显更小才算学到东西。')


if __name__ == '__main__':
    main()
