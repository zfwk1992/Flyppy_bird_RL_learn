"""经验回放：均匀采样，uint8 存储，支持 n-step 回报。

为什么不用 PER
--------------
旧管线的 PER 把 α 施加了两次，且 max_priority 单调不降，实测导致约 276 倍地
偏向最近样本 —— 经验回放的去相关作用被完全抵消，等于在做在线学习。
管道样本稀疏的问题这里靠 frame_skip=4（每条经验都携带完整的窗口奖励）和
高出三个数量级的回传次数解决，不需要 PER。
"""

from collections import deque

import numpy as np


class ReplayBuffer:
    """只存 state(stack 帧) + 之后的 n 帧，采样时拼出 next_state。

    1-step 时 next_state = concat(next_frame, state[:stack-1])，正是主循环
    推进帧栈用的同一个恒等式（见 rollout.FrameStack.push）。

    n-step 时这个恒等式**不再成立** —— 这是最容易写错的地方：
    s_{t+n} 与 s_t 只重叠 stack-n 帧，所以必须存 n 帧而不是 1 帧。
    以 stack=4, n=3 为例（帧栈里新帧在前）：

        s_t     = [f_t,   f_t-1, f_t-2, f_t-3]
        s_{t+3} = [f_t+3, f_t+2, f_t+1, f_t  ]
                   \_____ 存下来的 3 帧 ____/  \_ s_t[0] _/

    重建式：s1 = concat(next_frames[::-1], state[:stack-n])
    若照搬 1-step 的写法，网络会拿 s_{t+1} 当 s_{t+n} 用，
    而奖励却是 n 步累加的 —— 目标和状态错位，训练会静默地学歪。
    单测 11 专门验证这个重建。

    内存：cap * (stack+n) * h * w 字节。
    80x128、stack=4、n=3 时 cap=50k 约 3.6GB（n=1 时 2.6GB）。
    """

    def __init__(self, capacity, stack=4, h=80, w=128, n_step=1):
        if not 1 <= n_step <= stack:
            raise ValueError(
                "n_step=%d 必须在 1..stack(%d) 之间：n_step > stack 时 "
                "s_t 和 s_{t+n} 不再有重叠帧，就没法用这种压缩存储了"
                % (n_step, stack))
        self.capacity = capacity
        self.stack = stack
        self.n_step = n_step
        self.states = np.zeros((capacity, stack, h, w), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, n_step, h, w), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_frames, done):
        """next_frames: n_step 帧，按时间正序（最早的在前）。

        1-step 时可以直接传单帧，这里统一成 (n_step, h, w)。
        """
        nf = np.asarray(next_frames)
        if nf.ndim == 2:
            nf = nf[None]
        assert nf.shape[0] == self.n_step, \
            "期望 %d 帧，收到 %d" % (self.n_step, nf.shape[0])
        i = self.pos
        self.states[i] = state
        self.next_frames[i] = nf
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
        # next_frames 是时间正序，帧栈要新帧在前 -> 倒过来
        nf_rev = self.next_frames[idx][:, ::-1]
        keep = self.stack - self.n_step
        s1 = (np.concatenate([nf_rev, s[:, :keep]], axis=1) if keep > 0
              else nf_rev)
        return s, self.actions[idx], self.rewards[idx], s1, self.dones[idx]

    def __len__(self):
        return self.size

    def nbytes(self):
        return (self.states.nbytes + self.next_frames.nbytes + self.actions.nbytes
                + self.rewards.nbytes + self.dones.nbytes)


class NStepAccumulator:
    """把连续 n 次决策攒成一条 n-step 经验。

    n-step 回报把奖励往回传得更快：1-step 时"过管道 +1"这个信号每次目标网络
    同步只能往回走一步，n-step 一次走 n 步。稀疏奖励下这是标准且低风险的改进。

        R_n = r_t + γ r_{t+1} + ... + γ^{n-1} r_{t+n-1}
        y   = R_n + γ^n · max_a Q_target(s_{t+n}, a) · (1 - done)

    回合中途结束时截断：把队列里剩下的所有前缀都吐出来，各自带正确的
    折扣长度和 done 标记。否则回合末尾的 n-1 条经验会丢失，
    而那些恰恰是**包含死亡信息**的样本。
    """

    def __init__(self, n_step, gamma):
        self.n = n_step
        self.gamma = gamma
        self.buf = deque()

    def reset(self):
        """新回合开始必须调用，否则会把上一局的奖励算进这一局。"""
        self.buf.clear()

    def push(self, state, action, reward, next_frame, done):
        """返回这一步产生的 0 条或多条 (state, action, R_n, frames, done)。"""
        self.buf.append([state, action, float(reward), next_frame, bool(done)])
        out = []
        if done:
            # 回合结束：所有还在队列里的起点都要吐出来（截断的 n-step）
            while self.buf:
                out.append(self._make())
                self.buf.popleft()
        elif len(self.buf) == self.n:
            out.append(self._make())
            self.buf.popleft()
        return out

    def _make(self):
        """以 buf[0] 为起点做一条经验。"""
        R = 0.0
        end = len(self.buf)
        for i, item in enumerate(self.buf):
            R += (self.gamma ** i) * item[2]
            if item[4]:                       # 中途终止，后面的不算
                end = i + 1
                break
        frames = [self.buf[i][3] for i in range(end)]
        # 截断时用最后一帧补齐，反正 done=True 会让 next_state 被 (~d) 屏蔽
        while len(frames) < self.n:
            frames.append(frames[-1])
        done = self.buf[end - 1][4]
        return (self.buf[0][0], self.buf[0][1], R,
                np.asarray(frames, dtype=np.uint8), done)
