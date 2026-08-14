"""诊断实验：拿**完美状态**训一个小 MLP，测出感知与控制的分界线。

问题
----
CNN 在 gap 60-165 上稳定停在约 17 根管道，而在 gap 80-165 上能到 56.8。
两种解释：

  (A) **感知瓶颈** —— 80x80 的观测里，60px 的缝隙只剩 3 个像素行，
      容错空间约 2 个像素，比 conv1 的 stride(4 像素) 还小。网络看不清。
  (B) **控制瓶颈** —— 扇翅是 velY 瞬间设为 -5 的冲量，配 0.5/帧 的重力，
      小鸟能维持的最小竖直振荡是 22.5px，而 60px 缝隙扣掉鸟高只剩 36px。
      物理上就很难，跟看不看得清无关。

这个脚本把 (A) 拿掉：直接喂真实状态向量（小鸟 y/速度 + 下两根管道的
距离/缝隙中心/缝隙大小，共 8 维），网络换成一个几万参数的 MLP。

  - 如果它**也**停在 17 根左右  -> 瓶颈是 (B)，改网络和分辨率都没用
  - 如果它能到 100+ 根          -> 瓶颈是 (A)，提高分辨率是正确方向

刻意复用主管线的 skip_step / epsilon_at / sample_random_action ——
帧跳过对齐、奖励累加、terminal 处理这些容易写错的语义只此一份。
只有网络、缓冲区和主循环是这里独有的（都很短）。

    python experiments/state_mlp.py --episodes 20000
    python experiments/state_mlp.py --gap-range 80 165     # 对照另一个分布
"""

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy.config import resolve_config
from flappy.csvlog import EPISODE_FIELDS, EVAL_FIELDS, TRAIN_FIELDS, RunLogger
from flappy.rollout import epsilon_at, make_env, sample_random_action, skip_step

STATE_DIM = 8


# ======================================================================
class StateMLP(nn.Module):
    """8 -> 128 -> 128 -> Dueling 双头。约 2.2 万参数，是 CNN 的 1/57。

    保留 Dueling 结构是为了让对照只差在"输入是什么"这一个变量上。
    """

    def __init__(self, n_actions=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        for layer in (self.value, self.advantage):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class VectorBuffer:
    """float32 向量版的经验回放。

    一条经验只有 8*2+3 个 float = 76 字节，所以 40 万条才 30MB ——
    像素版同样容量要 12.8GB。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.s1 = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.uint8)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s1, d):
        i = self.pos
        self.s[i], self.s1[i], self.a[i], self.r[i], self.d[i] = s, s1, a, r, d
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        i = np.random.randint(0, self.size, size=n)
        return self.s[i], self.a[i], self.r[i], self.s1[i], self.d[i]

    def __len__(self):
        return self.size


# ======================================================================
@torch.no_grad()
def evaluate(net, cfg, device, env, n_episodes, max_decisions):
    pipes, lengths, trunc = [], [], 0
    for _ in range(n_episodes):
        env.reset()
        s = env.state_vector()
        phi = env.current_potential()
        n = 0
        while True:
            q = net(torch.from_numpy(s).unsqueeze(0).to(device))[0]
            _, _, done, info, phi = skip_step(env, int(q.argmax()), phi, cfg, render=False)
            s = env.state_vector()
            n += 1
            if done or n >= max_decisions:
                pipes.append(info['score'])
                lengths.append(n)
                trunc += int(not done)
                break
    return dict(pipes_mean=float(np.mean(pipes)), pipes_std=float(np.std(pipes)),
                pipes_max=int(np.max(pipes)), pipes_median=float(np.median(pipes)),
                len_mean=float(np.mean(lengths)), truncated=trunc)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--gap-range', type=float, nargs=2, default=[60.0, 165.0])
    p.add_argument('--buffer', type=int, default=400000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    cfg = resolve_config(max_episodes=a.episodes, seed=a.seed,
                         buffer_capacity=a.buffer,
                         pipe_gap_range=tuple(a.gap_range))
    device = torch.device(a.device)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    run_dir = a.run_dir or os.path.join(
        'runs', 'mlp_%s_%.0f-%.0f' % (datetime.now().strftime('%H%M%S'),
                                      a.gap_range[0], a.gap_range[1]))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(cfg, _experiment='state_mlp', _state_dim=STATE_DIM), f, indent=2)
    log = RunLogger(run_dir)

    online = StateMLP().to(device)
    target = StateMLP().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    for q in target.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg['lr'], eps=cfg['adam_eps'])

    n_par = sum(q.numel() for q in online.parameters())
    log.say("state-vector MLP diagnostic: params=%d (CNN has 1259683, ratio 1/%.0f)"
            % (n_par, 1259683 / n_par))
    log.say("gap_range=%.0f-%.0f  buffer=%d (%.0f MB)"
            % (a.gap_range[0], a.gap_range[1], a.buffer,
               a.buffer * (STATE_DIM * 2 + 3) * 4 / 1e6))

    buf = VectorBuffer(a.buffer)
    env = make_env(cfg)
    eval_env = make_env(cfg)

    dstep = episode = grad = syncs = 0
    best_r100 = -1e9
    last_best_ep = -10 ** 9
    recent = deque(maxlen=100)
    t0 = time.time()
    acc = {k: 0.0 for k in ('loss', 'q_mean', 'td')}
    acc_n = 0

    env.reset()
    s = env.state_vector()
    phi = env.current_potential()
    ep_ret = 0.0
    ep_dec = 0

    while episode < cfg['max_episodes']:
        eps = epsilon_at(dstep, cfg)
        if random.random() < eps:
            act = sample_random_action(cfg)
        else:
            with torch.no_grad():
                act = int(online(torch.from_numpy(s).unsqueeze(0).to(device))[0].argmax())

        _, r, done, info, phi = skip_step(env, act, phi, cfg, render=False)
        s1 = env.state_vector()
        buf.add(s, act, r, s1, done)
        dstep += 1
        ep_dec += 1
        ep_ret += r

        if (dstep > cfg['warmup_decisions'] and len(buf) >= cfg['batch_size']
                and dstep % cfg['train_every_decisions'] == 0):
            bs, ba, br, bs1, bd = buf.sample(cfg['batch_size'])
            bs = torch.from_numpy(bs).to(device)
            bs1 = torch.from_numpy(bs1).to(device)
            ba = torch.from_numpy(ba).long().to(device)
            br = torch.from_numpy(br).to(device)
            bd = torch.from_numpy(bd).to(device)

            q_sa = online(bs).gather(1, ba.unsqueeze(1))
            with torch.no_grad():
                a_star = online(bs1).argmax(1, keepdim=True)
                y = br.unsqueeze(1) + cfg['gamma'] * target(bs1).gather(1, a_star) * (~bd).unsqueeze(1)
            loss = F.smooth_l1_loss(q_sa, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(online.parameters(), cfg['grad_clip'])
            opt.step()
            grad += 1

            acc['loss'] += float(loss)
            acc['q_mean'] += float(q_sa.mean())
            acc['td'] += float((q_sa - y).abs().mean())
            acc_n += 1

            if grad % cfg['target_sync_grad_steps'] == 0:
                target.load_state_dict(online.state_dict())
                syncs += 1
            if grad % cfg['log_train_every_grad_steps'] == 0:
                log.train.write(
                    grad_step=grad, decision_step=dstep,
                    loss=round(acc['loss'] / acc_n, 6),
                    td_abs_mean=round(acc['td'] / acc_n, 6), td_abs_max=0,
                    q_mean=round(acc['q_mean'] / acc_n, 4), q_std=0, q_min=0, q_max=0,
                    target_q_mean=0, grad_norm=round(float(gn), 4),
                    lr=cfg['lr'], epsilon=round(eps, 5), buffer_size=len(buf),
                    syncs=syncs, dec_per_s=round(dstep / (time.time() - t0), 1),
                    grad_per_s=round(grad / (time.time() - t0), 1))
                acc = {k: 0.0 for k in acc}
                acc_n = 0

        truncated = (not done) and ep_dec >= cfg['max_episode_decisions']
        if done or truncated:
            episode += 1
            recent.append(info['score'])
            r100 = float(np.mean(recent))
            log.episodes.write(
                episode=episode, decision_step=dstep, frames=info['frames'],
                pipes=info['score'], ep_return=round(ep_ret, 4),
                ep_len_decisions=ep_dec, epsilon=round(eps, 5),
                recent100_pipes=round(r100, 4), recent100_return=0,
                buffer_size=len(buf), wall_s=round(time.time() - t0, 1),
                terminated=int(done))

            if episode % 2000 == 0:
                log.say("ep=%d dec=%d grad=%d eps=%.3f recent100=%.2f wall=%.0fs"
                        % (episode, dstep, grad, eps, r100, time.time() - t0))
            if (episode >= 500 and len(recent) == 100 and r100 > best_r100 * 1.02
                    and episode - last_best_ep >= 100):
                best_r100, last_best_ep = r100, episode
                torch.save({'model': online.state_dict(), 'recent100_pipes': r100,
                            'episode': episode, 'config': cfg},
                           os.path.join(run_dir, 'best.pt'))
            if episode % cfg['eval_every_episodes'] == 0:
                res = evaluate(online, cfg, device, eval_env, cfg['eval_episodes'],
                               cfg['max_episode_decisions'])
                log.eval.write(episode=episode, decision_step=dstep,
                               n_eval_eps=cfg['eval_episodes'],
                               eval_pipes_mean=round(res['pipes_mean'], 3),
                               eval_pipes_std=round(res['pipes_std'], 3),
                               eval_pipes_max=res['pipes_max'],
                               eval_len_mean=round(res['len_mean'], 2))
                log.say("eval @ep=%d: pipes=%.2f+-%.2f max=%d len=%.0f trunc=%d/%d"
                        % (episode, res['pipes_mean'], res['pipes_std'],
                           res['pipes_max'], res['len_mean'], res['truncated'],
                           cfg['eval_episodes']))

            env.reset()
            s = env.state_vector()
            phi = env.current_potential()
            ep_ret = 0.0
            ep_dec = 0
        else:
            s = s1

    final = evaluate(online, cfg, device, eval_env, 100, cfg['max_episode_decisions'])
    log.say("FINAL (100 eps): pipes=%.2f+-%.2f median=%.1f max=%d trunc=%d/100"
            % (final['pipes_mean'], final['pipes_std'], final['pipes_median'],
               final['pipes_max'], final['truncated']))
    torch.save({'model': online.state_dict(), 'episode': episode,
                'final_eval': final, 'config': cfg},
               os.path.join(run_dir, 'final.pt'))
    log.close()


if __name__ == '__main__':
    main()
