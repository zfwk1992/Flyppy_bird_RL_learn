"""
独立的贪婪评测：加载一个 checkpoint，跑 N 局，报告过管道数。

为什么需要单独一个脚本
----------------------
旧的 test/test_ai_gameplay.py:114 调用 agent.select_action()，而后者在
deep_Q_dueling_DQN.py:324-325 有一条短路：

    if self.decision_step < self.training_start:   # 20000
        return random.randrange(self.actions)

而 load_model() 只恢复两个 state_dict，从不恢复 decision_step。新建的 agent
其 decision_step = 0，于是 **每次评测的前 20000 次决策都是纯随机的**，
--epsilon 0 也不起作用。saved_networks/ 里 113 个 checkpoint 因此
没有一个有可信的评测数据。

本脚本直接构造网络前向，不存在任何按训练进度短路的分支。

用法：
    python eval_flappy.py runs/xxx/best.pt --episodes 100
    python eval_flappy.py runs/xxx/best.pt --episodes 30 --epsilon 0.0 --q-stats
"""

import argparse
import random

import numpy as np
import torch

from game.flappy_env import FlappyEnv
from train_flappy_dqn import (CONFIG, DuelingDQN, sample_random_action,
                              skip_step)


@torch.no_grad()
def run_eval(ckpt_path, n_episodes, epsilon, device, seed=0, q_stats=False,
             max_decisions=2000, pipe_gap=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(CONFIG)
    cfg.update(ckpt.get('config', {}))
    # 旧 checkpoint 的 config 里没有 pipe_gap，会回退到当前默认值 ——
    # 那等于用不同难度评测一个旧模型。显式指定可以钉死难度。
    if pipe_gap is not None:
        cfg['pipe_gap'] = pipe_gap

    net = DuelingDQN().to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()

    # 确定性自检：网络在推理时必须是确定的
    dummy = torch.zeros(1, cfg['frame_stack'], cfg['obs_size'], cfg['obs_size'],
                        dtype=torch.uint8, device=device)
    assert torch.allclose(net(dummy), net(dummy)), \
        "network is stochastic at act time"

    random.seed(seed)
    np.random.seed(seed)

    env = FlappyEnv(pipe_reward=cfg['pipe_reward'],
                    death_reward=cfg['death_reward'],
                    alive_reward=cfg['alive_reward'],
                    pipe_gap=cfg['pipe_gap'])

    pipes, lengths, returns, q_max_all, q_gap_all = [], [], [], [], []
    n_truncated = 0

    for _ in range(n_episodes):
        frame = env.reset()          # 已是 (80,80) uint8
        stack = np.repeat(frame[None], cfg['frame_stack'], axis=0)
        phi = env.current_potential()
        n_dec, ep_ret = 0, 0.0
        while True:
            if epsilon > 0 and random.random() < epsilon:
                action = sample_random_action(cfg)
            else:
                t = torch.from_numpy(stack).unsqueeze(0).to(device)
                q = net(t)[0]
                action = int(q.argmax().item())
                if q_stats:
                    q_max_all.append(float(q.max()))
                    q_gap_all.append(float(q[1] - q[0]))
            nf, r, done, info, phi = skip_step(env, action, phi, cfg)
            n_dec += 1
            ep_ret += r
            # 上限是必需的：一个学好的策略可以永远不死
            if done or n_dec >= max_decisions:
                pipes.append(info['score'])
                lengths.append(n_dec)
                returns.append(ep_ret)
                n_truncated += int(not done)
                break
            stack = np.concatenate([nf[None], stack[:cfg['frame_stack'] - 1]], axis=0)

    out = dict(
        episodes=n_episodes,
        pipes_mean=float(np.mean(pipes)), pipes_std=float(np.std(pipes)),
        pipes_max=int(np.max(pipes)), pipes_median=float(np.median(pipes)),
        len_mean=float(np.mean(lengths)), return_mean=float(np.mean(returns)),
        truncated=n_truncated, max_decisions=max_decisions,
        pipe_gap=cfg['pipe_gap'],
        ckpt_episode=ckpt.get('episode'),
        ckpt_recent100=ckpt.get('recent100_pipes'),
    )
    if q_stats and q_max_all:
        out['q_max_mean'] = float(np.mean(q_max_all))
        out['q_gap_abs_mean'] = float(np.mean(np.abs(q_gap_all)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--epsilon', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--q-stats', action='store_true')
    p.add_argument('--max-decisions', type=int, default=2000,
                   help='cap on episode length; a good policy never dies')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='override difficulty (px). Old checkpoints do not record it.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    res = run_eval(a.checkpoint, a.episodes, a.epsilon,
                   torch.device(a.device), a.seed, a.q_stats, a.max_decisions,
                   a.pipe_gap)

    print("checkpoint      : %s" % a.checkpoint)
    print("  saved at ep   : %s (recent100_pipes=%s)"
          % (res['ckpt_episode'], res['ckpt_recent100']))
    print("  eval episodes : %d  (epsilon=%.3f, pipe_gap=%d)"
          % (res['episodes'], a.epsilon, res['pipe_gap']))
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
