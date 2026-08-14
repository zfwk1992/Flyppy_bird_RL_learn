"""经验回放：均匀采样，uint8 存储。

为什么不用 PER
--------------
旧管线的 PER 把 α 施加了两次，且 max_priority 单调不降，实测导致约 276 倍地
偏向最近样本 —— 经验回放的去相关作用被完全抵消，等于在做在线学习。
管道样本稀疏的问题这里靠 frame_skip=4（每条经验都携带完整的窗口奖励）和
高出三个数量级的回传次数解决，不需要 PER。
"""

import numpy as np


class ReplayBuffer:
    """只存 state(4帧) + next_frame(1帧)，采样时拼出 next_state。

    next_state = concat(next_frame, state[:3]) 正是主循环推进帧栈用的同一个
    恒等式（见 rollout.FrameStack.push），可直接单测
    （test/test_env_and_buffer.py 测试 4）。

    内存：cap * (stack+1) * h * w 字节。
    80x128 的观测下 cap=100k 约 5.1GB（80x80 时是 3.2GB）——
    换各向异性分辨率时别忘了按比例调小 buffer_capacity。
    旧实现存 float32 的双份完整帧栈 = 200KB/条，cap=100k 时要 20GB，
    而帧已被二值化成 {0,255} —— 32 倍浪费。
    """

    def __init__(self, capacity, stack=4, h=80, w=128):
        self.capacity = capacity
        self.stack = stack
        self.states = np.zeros((capacity, stack, h, w), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, 1, h, w), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_frame, done):
        i = self.pos
        self.states[i] = state
        self.next_frames[i, 0] = next_frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        # 有放回采样：标准 DQN 的做法，O(n)。
        # 旧实现用 np.random.choice(replace=False, p=probs) 在 10 万条上采样，
        # 走的是 NumPy 的顺序选择慢路径，约 20ms/步，是整个训练的吞吐瓶颈。
        idx = np.random.randint(0, self.size, size=n)
        s = self.states[idx]
        s1 = np.concatenate([self.next_frames[idx], s[:, :self.stack - 1]], axis=1)
        return s, self.actions[idx], self.rewards[idx], s1, self.dones[idx]

    def __len__(self):
        return self.size

    def nbytes(self):
        return (self.states.nbytes + self.next_frames.nbytes + self.actions.nbytes
                + self.rewards.nbytes + self.dones.nbytes)
