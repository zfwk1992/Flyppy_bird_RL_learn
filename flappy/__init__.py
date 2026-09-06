"""Flappy Bird Dueling Double-DQN.

包结构
------
    config.py      超参数（唯一数据源）
    model.py       DuelingDQN 网络
    replay.py      均匀采样经验回放
    rollout.py     帧跳过、探索策略、帧栈、评测
    agent.py       Double-DQN 学习步与目标网络同步
    csvlog.py      结构化 CSV 日志
    checkpoint.py  存档的读写

入口脚本在仓库根目录：train.py / eval.py / play.py / plot.py
"""

from .agent import Agent
from .config import CONFIG, SMOKE_OVERRIDES, resolve_config
from .model import DuelingDQN, assert_deterministic
from .replay import ReplayBuffer
from .rollout import (FrameStack, epsilon_at, evaluate, sample_random_action,
                      skip_step)

__all__ = [
    'Agent', 'CONFIG', 'SMOKE_OVERRIDES', 'resolve_config',
    'DuelingDQN', 'assert_deterministic', 'ReplayBuffer',
    'FrameStack', 'epsilon_at', 'evaluate', 'sample_random_action', 'skip_step',
]
