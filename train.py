"""训练入口：Flappy Bird Dueling Double-DQN。

    python train.py                      # 完整训练
    python train.py --smoke              # 几分钟的管线自检
    python train.py --max-hours 12       # 到点干净收尾（跑完收尾评测再退出）
    python train.py --pipe-gap 150       # 放宽难度

产物全部落在 runs/<时间戳>/ 下：
    config.json  episodes.csv  train.csv  eval.csv  run.log
    best.pt  resume_0.pt  resume_1.pt  final.pt

算法与工程细节写在各个模块的文档字符串里；重写这条管线的来龙去脉见 README。
"""

import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch

from flappy import checkpoint, diagnostics
from flappy.agent import Agent
from flappy.config import resolve_config
from flappy.csvlog import RunLogger
from flappy.replay import NStepAccumulator, ReplayBuffer
from flappy.rollout import (FrameStack, epsilon_at, evaluate, make_env,
                            skip_step)


def setup_device(allow_cpu):
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.get_device_name(0)
    if allow_cpu:
        return torch.device('cpu'), 'cpu'
    # 旧代码在这里静默回退到 CPU，于是"训练很慢"被归因成了别的东西
    raise SystemExit(
        "CUDA not available (torch=%s). The old code silently fell back to CPU\n"
        "and hid this.\n"
        "  Fix:  .\\setup.ps1            (or  .\\setup.ps1 -Mirror  in CN)\n"
        "  Note: on Windows the plain PyPI 'torch' is a CPU build - the CUDA one\n"
        "        only comes from the pytorch.org index.\n"
        "  Or pass --allow-cpu to accept a ~20x slowdown." % torch.__version__)


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def train(cfg, args):
    device, dev_name = setup_device(args.allow_cpu)
    seed_everything(cfg['seed'], device)

    run_dir = args.run_dir or os.path.join(
        'runs', datetime.now().strftime('%Y%m%d_%H%M%S')
        + ('_smoke' if args.smoke else ''))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    log = RunLogger(run_dir)
    say = log.say
    say("run_dir=%s" % run_dir)
    say("device=%s (%s), torch=%s" % (device, dev_name, torch.__version__))

    # ---- 组件 ----
    agent = Agent(cfg, device)
    n_params = sum(p.numel() for p in agent.online.parameters())
    say("network params=%d (%.2f MB fp32)" % (n_params, n_params * 4 / 1e6))

    if args.resume:
        say(checkpoint.load_for_training(args.resume, agent, device))
        say("NOTE: counters restart at 0 and the replay buffer starts empty; "
            "the exploration schedule is whatever this run's config says.")

    buffer = ReplayBuffer(cfg['buffer_capacity'], cfg['frame_stack'],
                          cfg['obs_w'], cfg['obs_h'], n_step=cfg['n_step'])
    nstep = NStepAccumulator(cfg['n_step'], cfg['gamma'])
    say("replay capacity=%d (%.2f GB allocated), n_step=%d"
        % (cfg['buffer_capacity'], buffer.nbytes() / 1e9, cfg['n_step']))

    env = make_env(cfg)
    # 评测用独立的环境实例：共用会让训练回合的状态被评测冲掉
    eval_env = make_env(cfg)

    # ---- 状态 ----
    decision_step = 0
    episode = 0
    # 门控状态：best_screen 是便宜的筛选阈值，best_confirmed 是真正决定
    # 落盘的确认分数（在另一批关卡上测的）。见定期评测那一段。
    best_screen = -float('inf')
    best_confirmed = -float('inf')
    last_best_episode = -10 ** 9
    recent_pipes = deque(maxlen=100)
    recent_returns = deque(maxlen=100)
    resume_slot = 0
    t_start = time.time()
    last_resume_t = t_start
    last_log_grad = 0
    last_log_dec = 0
    last_log_t = t_start
    acc = {k: 0.0 for k in Agent.STAT_KEYS}
    acc_n = 0

    # ---- 诊断探针集：一次生成，之后永不改变 ----
    # 必须固定，否则 argmax_flip 比的是状态不是策略。用不依赖网络的启发式
    # 在固定 seed 上采集，见 flappy/diagnostics.py。
    probe = diagnostics.build_probe_set(cfg, eval_env,
                                        n_states=cfg['diag_probe_states'])
    prev_argmax = None
    log.say("diagnostics probe set: %d states (fixed for the whole run), "
            "q ceiling ~= %.2f" % (len(probe), diagnostics.q_ceiling(cfg)))

    # ---- 首个回合 ----
    stacker = FrameStack(cfg['frame_stack'])
    stack = stacker.reset(env.reset())      # env.reset() 已是 (80,80) uint8
    phi = env.current_potential()
    ep_return = 0.0
    ep_dec = 0

    say("start training: warmup=%d decisions, learning starts after that"
        % cfg['warmup_decisions'])

    max_seconds = args.max_hours * 3600 if args.max_hours else None
    stop_reason = 'max_episodes'

    while episode < cfg['max_episodes']:
        if max_seconds and time.time() - t_start > max_seconds:
            stop_reason = 'max_hours'
            break

        eps = epsilon_at(decision_step, cfg)
        action = agent.act_epsilon_greedy(stack, eps)

        next_frame, r_dec, done, info, phi = skip_step(env, action, phi, cfg)
        # n-step：攒够 n 步（或回合结束）才产出经验。n_step=1 时等价于直接 add。
        for tr in nstep.push(stack, action, r_dec, next_frame, done):
            buffer.add(*tr)

        decision_step += 1
        ep_dec += 1
        ep_return += r_dec

        # ---- 学习 ----
        if (decision_step > cfg['warmup_decisions']
                and len(buffer) >= cfg['batch_size']
                and decision_step % cfg['train_every_decisions'] == 0):
            stats = agent.learn(buffer)
            for k in acc:
                acc[k] += stats[k]
            acc_n += 1

            if agent.grad_steps % cfg['log_train_every_grad_steps'] == 0:
                now = time.time()
                dt = max(now - last_log_t, 1e-9)
                log.train.write(
                    grad_step=agent.grad_steps,
                    decision_step=decision_step,
                    **{k: round(acc[k] / max(acc_n, 1), 6) for k in acc},
                    lr=cfg['lr'],
                    epsilon=round(eps, 5),
                    buffer_size=len(buffer),
                    syncs=agent.syncs,
                    dec_per_s=round((decision_step - last_log_dec) / dt, 1),
                    grad_per_s=round((agent.grad_steps - last_log_grad) / dt, 1),
                )
                acc = {k: 0.0 for k in acc}
                acc_n = 0
                last_log_t = now
                last_log_dec = decision_step
                last_log_grad = agent.grad_steps

        # ---- 回合结束（撞死，或达到长度上限被截断） ----
        truncated = (not done) and ep_dec >= cfg['max_episode_decisions']
        if done or truncated:
            episode += 1
            pipes = info['score']
            recent_pipes.append(pipes)
            recent_returns.append(ep_return)
            r100 = float(np.mean(recent_pipes))

            log.episodes.write(
                episode=episode, decision_step=decision_step, frames=info['frames'],
                pipes=pipes, ep_return=round(ep_return, 4), ep_len_decisions=ep_dec,
                epsilon=round(eps, 5), recent100_pipes=round(r100, 4),
                recent100_return=round(float(np.mean(recent_returns)), 4),
                buffer_size=len(buffer), wall_s=round(time.time() - t_start, 1),
                terminated=int(done),
            )

            if episode % 500 == 0:
                say("ep=%d dec=%d grad=%d syncs=%d eps=%.3f recent100_pipes=%.2f "
                    "buf=%d wall=%.0fs"
                    % (episode, decision_step, agent.grad_steps, agent.syncs,
                       eps, r100, len(buffer), time.time() - t_start))

            # ---- 定期评测 ----
            if cfg['eval_every_episodes'] and episode % cfg['eval_every_episodes'] == 0:
                # seed_base 固定 => 每次评测跑的是**同一组关卡**，于是 eval.csv
                # 变成一条可纵向比较的曲线，而不是每次换一批关卡的噪声。
                # evaluate 内部会保存/恢复全局随机流，不会劫持训练自己的随机序列。
                res = evaluate(agent.online, cfg, device, cfg['eval_episodes'],
                               epsilon=cfg['eval_epsilon'], env=eval_env,
                               seed_base=cfg['eval_seed_base'])
                log.eval.write(episode=episode, decision_step=decision_step,
                               n_eval_eps=cfg['eval_episodes'],
                               eval_pipes_mean=round(res['pipes_mean'], 3),
                               eval_pipes_std=round(res['pipes_std'], 3),
                               eval_pipes_max=res['pipes_max'],
                               eval_len_mean=round(res['len_mean'], 2),
                               eval_pipes_iqm=round(res['pipes_iqm'], 3),
                               eval_pipes_median=round(res['pipes_median'], 2),
                               eval_pipes_q25=round(res['pipes_q25'], 2),
                               eval_pipes_q75=round(res['pipes_q75'], 2),
                               eval_truncated=res['truncated'],
                               eval_pipes_censored=round(
                                   res['pipes_censored_mean'], 3))
                # 网络内部诊断：成绩崩掉时这几列是唯一还有信号的东西
                d, prev_argmax = diagnostics.compute(
                    agent.online, probe, cfg, device, prev_argmax)
                log.diag.write(episode=episode, decision_step=decision_step, **d)
                say("diag @ep=%d: dorm_fc=%.1f%% eff95_fc=%d gap_med=%.3f "
                    "tiny=%.1f%% q/ceil=%.2f flip=%s"
                    % (episode, 100 * d['dorm_fc'], d['eff95_fc'],
                       d['q_gap_median'], 100 * d['q_gap_tiny_frac'],
                       d['q_ceil_ratio'],
                       ('%.1f%%' % (100 * d['argmax_flip']))
                       if d['argmax_flip'] != '' else 'n/a'))
                say("eval @ep=%d: mean=%.2f (q25-q75 %.0f-%.0f) median=%.1f "
                    "iqm=%.1f max=%d truncated=%d/%d censored_mean=%.1f"
                    % (episode, res['pipes_mean'], res['pipes_q25'],
                       res['pipes_q75'], res['pipes_median'], res['pipes_iqm'],
                       res['pipes_max'], res['truncated'], cfg['eval_episodes'],
                       res['pipes_censored_mean']))

                # ---- 最佳模型：两级门控 ----
                # 旧做法挂在 recent100_pipes 上（带 epsilon 的训练分数，SE 约 22%），
                # 实测在选噪声：被它选中的 best.pt 在固定集上只有 74.6 根，
                # 而它没选的 final.pt 有 95.1 根。见 WHERE_IS_THE_PROBLEM.md 第二节。
                #
                # 现在：先用定期评测的 IQM 筛（便宜），出现候选再跑一次
                # **换一批关卡**的确认评测（贵，但只在候选出现时跑）。
                # 换关卡是必须的 —— 在筛选用的那 20 关上比较，等于对着筛选集过拟合。
                gate = res['pipes_iqm'] if cfg['best_gate_metric'] == 'iqm' \
                    else res['pipes_mean']
                if (episode >= cfg['best_min_episodes']
                        and gate > best_screen * cfg['best_improve_frac']
                        and episode - last_best_episode
                        >= cfg['best_cooldown_episodes']):
                    best_screen = gate
                    conf = evaluate(agent.online, cfg, device,
                                    cfg['best_confirm_episodes'],
                                    epsilon=cfg['eval_epsilon'], env=eval_env,
                                    seed_base=cfg['eval_seed_base']
                                    + cfg['best_confirm_seed_offset'])
                    c = conf['pipes_iqm'] if cfg['best_gate_metric'] == 'iqm' \
                        else conf['pipes_mean']
                    if c > best_confirmed:
                        say("new best: confirm %s=%.2f (was %.2f) over %d eps "
                            "at ep=%d -> best.pt"
                            % (cfg['best_gate_metric'], c, best_confirmed,
                               cfg['best_confirm_episodes'], episode))
                        best_confirmed = c
                        last_best_episode = episode
                        checkpoint.save_best(
                            os.path.join(run_dir, 'best.pt'),
                            agent, cfg, c, episode, decision_step)
                    else:
                        say("candidate @ep=%d rejected: confirm %s=%.2f "
                            "<= best %.2f (screening was noise)"
                            % (episode, cfg['best_gate_metric'], c, best_confirmed))

            # ---- 断点续训：双槽轮转 ----
            if time.time() - last_resume_t > cfg['resume_every_minutes'] * 60:
                checkpoint.save_resume(
                    os.path.join(run_dir, 'resume_%d.pt' % resume_slot),
                    agent, cfg, episode, decision_step, eps)
                resume_slot ^= 1
                last_resume_t = time.time()

            # ---- 重置帧栈和 n-step 累加器：杜绝跨局拼接 ----
            nstep.reset()
            stack = stacker.reset(env.reset())
            phi = env.current_potential()
            ep_return = 0.0
            ep_dec = 0
        else:
            stack = stacker.push(next_frame)

    # 收尾：跑一次更大规模的贪婪评测，并保存最终模型
    say("stopping (%s). running final evaluation..." % stop_reason)
    final = evaluate(agent.online, cfg, device, 30,
                     epsilon=cfg['eval_epsilon'], env=eval_env,
                     seed_base=cfg['eval_seed_base'])
    say("FINAL eval: pipes=%.2f+-%.2f max=%d len=%.1f truncated=%d/30"
        % (final['pipes_mean'], final['pipes_std'], final['pipes_max'],
           final['len_mean'], final['truncated']))
    checkpoint.save_final(os.path.join(run_dir, 'final.pt'),
                          agent, cfg, episode, decision_step, final)

    say("done: episodes=%d decisions=%d grad_steps=%d syncs=%d wall=%.2fh"
        % (episode, decision_step, agent.grad_steps, agent.syncs,
           (time.time() - t_start) / 3600))
    log.close()
    return run_dir


def main():
    p = argparse.ArgumentParser(
        description="Flappy Bird Dueling Double-DQN")
    p.add_argument('--smoke', action='store_true', help='short smoke-test config')
    p.add_argument('--allow-cpu', action='store_true',
                   help='allow running without CUDA (~20x slower)')
    p.add_argument('--episodes', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--run-dir', type=str, default=None)
    p.add_argument('--buffer', type=int, default=None)
    p.add_argument('--max-hours', type=float, default=None,
                   help='stop cleanly after N hours (final eval + final.pt)')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='difficulty: vertical pipe gap in px '
                        '(150 = easy/first run, 100 = original game, lower = harder); '
                        'only used when pipe randomisation is off')
    p.add_argument('--no-randomize', dest='randomize', action='store_false',
                   default=None,
                   help='disable pipe domain randomisation (fixed gap/spacing, '
                        '8 discrete heights) - for ablation against the old setup')
    p.add_argument('--gap-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=None,
                   help='randomised gap-size range, e.g. --gap-range 60 165 to '
                        'widen the curriculum towards narrower gaps')
    p.add_argument('--spacing-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=None, help='randomised pipe spacing range')
    # ---- 继续训练 ----
    p.add_argument('--resume', type=str, default=None,
                   help='start from an existing checkpoint (best.pt or resume_N.pt). '
                        'Weights are loaded; counters and the replay buffer are NOT - '
                        'set the exploration schedule explicitly with --eps-start etc.')
    p.add_argument('--eps-start', type=float, default=None,
                   help='initial epsilon. Default 1.0 (from scratch); when resuming a '
                        'trained model something like 0.6 keeps some of the learned '
                        'policy while still exploring the newly added difficulty')
    # ---- 网络 / 观测（做架构消融用） ----
    p.add_argument('--fc-hidden', type=int, default=None,
                   help='width of the fully-connected layer (default 256). '
                        'The old default was 512; measurements showed 14-25%% of '
                        'those units never fire - see docs/learn/12-network-sizing.md')
    p.add_argument('--obs-h', type=int, default=None,
                   help='observation pixels along the screen HEIGHT (default 128)')
    p.add_argument('--obs-w', type=int, default=None,
                   help='observation pixels along the screen WIDTH (default 80)')
    p.add_argument('--n-step', type=int, default=None,
                   help='n-step returns (default 3, 1 = plain one-step TD). '
                        'Capped at frame_stack: beyond that s_t and s_t+n share '
                        'no frames and the compressed replay layout breaks.')
    p.add_argument('--warmup', type=int, default=None,
                   help='decisions to collect before learning starts. The default '
                        '20000 fills the buffer from scratch; when resuming you can '
                        'shorten it since the policy already produces useful data')
    p.add_argument('--anneal1', type=int, default=None,
                   help='decisions to anneal epsilon from eps_start to 0.05')
    p.add_argument('--anneal2', type=int, default=None,
                   help='decisions to anneal epsilon from 0.05 to 0.01')
    args = p.parse_args()

    cfg = resolve_config(smoke=args.smoke,
                         max_episodes=args.episodes,
                         seed=args.seed,
                         buffer_capacity=args.buffer,
                         pipe_gap=args.pipe_gap,
                         randomize_pipes=args.randomize,
                         pipe_gap_range=tuple(args.gap_range) if args.gap_range else None,
                         pipe_spacing_range=(tuple(args.spacing_range)
                                             if args.spacing_range else None),
                         eps_start=args.eps_start,
                         fc_hidden=args.fc_hidden,
                         obs_h=args.obs_h,
                         obs_w=args.obs_w,
                         n_step=args.n_step,
                         warmup_decisions=args.warmup,
                         eps_anneal1_decisions=args.anneal1,
                         eps_anneal2_decisions=args.anneal2)
    train(cfg, args)


if __name__ == '__main__':
    main()
