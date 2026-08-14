"""存档的读写。

三种存档，用途不同：
  best.pt     —— 只有权重 + 当时的成绩，用来评测/游玩
  resume_N.pt —— 权重 + 目标网络 + 优化器 + 全部计数器，用来续训（双槽轮转）
  final.pt    —— 权重 + 收尾评测结果

所有存档都带 ``config``。这不是冗余：pipe_gap 这种难度旋钮若不随存档记录，
拿今天的默认值去评测昨天的模型，分数就没有可比性。
"""

import os

import torch

from .config import config_from_checkpoint
from .model import assert_deterministic, build_net


def save_best(path, agent, cfg, recent100_pipes, episode, decision_step):
    torch.save({'model': agent.online.state_dict(),
                'recent100_pipes': recent100_pipes,
                'episode': episode,
                'decision_step': decision_step,
                'config': cfg}, path)


def save_resume(path, agent, cfg, episode, decision_step, epsilon):
    torch.save({'model': agent.online.state_dict(),
                'target': agent.target.state_dict(),
                'optim': agent.optimizer.state_dict(),
                'episode': episode,
                'decision_step': decision_step,
                'grad_steps': agent.grad_steps,
                'syncs': agent.syncs,
                'epsilon': epsilon,
                'config': cfg}, path)


def save_final(path, agent, cfg, episode, decision_step, final_eval):
    torch.save({'model': agent.online.state_dict(),
                'episode': episode,
                'decision_step': decision_step,
                'grad_steps': agent.grad_steps,
                'final_eval': final_eval,
                'config': cfg}, path)


def load_for_training(path, agent, device):
    """把存档灌进一个已建好的 Agent，用于继续训练。返回一行说明。

    权重永远加载；``target`` 和 ``optim`` 只有存档里有才加载 ——
    ``best.pt`` 只存权重，``resume_N.pt`` 三样都有。

    **不恢复任何计数器。** 继续训练时环境/难度往往已经变了，沿用旧的
    decision_step 会让 epsilon 直接停在 0.01、并且跳过 warmup 在几乎空的
    缓冲区上就开始梯度更新。新的探索节奏由命令行显式指定，
    这样"这一轮到底怎么跑的"永远写在 config.json 里，而不是藏在存档里。
    """
    if not os.path.exists(path):
        raise SystemExit("checkpoint not found: %s" % path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    agent.online.load_state_dict(ckpt['model'])
    loaded = ['model']
    if 'target' in ckpt:
        agent.target.load_state_dict(ckpt['target'])
        loaded.append('target')
    else:
        # 没有目标网络就从在线网络复制一份，别让它停在随机初始化上
        agent.target.load_state_dict(agent.online.state_dict())
    if 'optim' in ckpt:
        agent.optimizer.load_state_dict(ckpt['optim'])
        loaded.append('optim')
    agent.target.eval()

    return ("resumed from %s (%s; trained to episode %s, recent100_pipes=%s)"
            % (path, '+'.join(loaded), ckpt.get('episode'),
               ckpt.get('recent100_pipes')))


def load_for_inference(path, device, **cfg_overrides):
    """读出 (net, cfg, ckpt)，网络已 eval() 并通过确定性自检。

    eval.py 和 play.py 共用这一个入口 —— 两边各写一遍加载逻辑，
    就是当年"评测跑的其实是随机策略"那类偏差的温床。
    """
    if not os.path.exists(path):
        raise SystemExit("checkpoint not found: %s" % path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt, **cfg_overrides)

    net = build_net(cfg, device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    assert_deterministic(net, cfg, device)
    return net, cfg, ckpt


def describe(ckpt, cfg, path):
    """存档的一行摘要，评测和游玩都会先打这几行。"""
    lines = ["checkpoint : %s" % path,
             "  saved at  : episode %s, decision_step %s"
             % (ckpt.get('episode'), ckpt.get('decision_step'))]
    if ckpt.get('recent100_pipes') is not None:
        lines.append("  recent100_pipes at save time: %.2f" % ckpt['recent100_pipes'])
    lines.append("  frame_skip=%d frame_stack=%d pipe_gap=%d"
                 % (cfg['frame_skip'], cfg['frame_stack'], cfg['pipe_gap']))
    return "\n".join(lines)
