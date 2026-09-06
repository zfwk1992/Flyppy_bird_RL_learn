# 02 · 从 Q-learning 到 DQN

表格式 Q-learning 在 Flappy Bird 上直接死路一条：状态是 `2^(4×80×80)` 种
可能的二值图，表格存不下，也永远遇不到重复状态。

所以要用**函数近似**：拿一个神经网络来表示 `Q(s,a)`。
但这一步引入了新的不稳定性，DQN 的三个核心技巧就是为了对付它们。

---

## 1. 致命三合一（deadly triad）

函数近似 + 自举 + off-policy，这三样凑齐时 Q-learning 可能发散。本项目全占了：

| 要素 | 本项目 | 为什么危险 |
|------|--------|-----------|
| 函数近似 | 卷积网络 | 更新一个状态的 Q 会**同时改变**其他所有状态的 Q |
| 自举 | `y = r + γ·max Q(s')` | 目标里含有网络自己的输出 —— 追自己的尾巴 |
| off-policy | 经验回放 | 训练分布 ≠ 当前策略的分布 |

具体的失控方式：网络提高了 `Q(s,a)`，由于泛化，`Q(s',a')` 也被顺带提高了；
而 `Q(s',a')` 又出现在 `Q(s,a)` 的目标里，于是 `Q(s,a)` 被推得更高……
正反馈，Q 值指数发散。

DQN 用三个技巧稳住它。

---

## 2. 技巧一：经验回放

### 问题

如果按时间顺序拿刚产生的经验去训练：

1. **样本高度相关**。连续几帧几乎一样，一个 minibatch 里全是近乎相同的样本，
   梯度估计的方差极大，等价于 batch size ≈ 1。
2. **分布随策略漂移**。学会往上飞之后，数据里就只剩"在上面"的状态，
   网络会**灾难性遗忘**低处该怎么办。
3. **数据只用一次**就扔了，样本效率极低。

### 做法

把经验存进一个大缓冲区，训练时**均匀随机**采一批：

```python
# flappy/replay.py
def sample(self, n):
    idx = np.random.randint(0, self.size, size=n)
    s = self.states[idx]
    s1 = np.concatenate([self.next_frames[idx], s[:, :self.stack - 1]], axis=1)
    return s, self.actions[idx], self.rewards[idx], s1, self.dones[idx]
```

随机采样打破了时间相关性，一个 batch 里混着几十万条不同时期、
不同策略产生的经验。

### 本项目的实现细节

**只存 4 帧 state + 1 帧 next_frame，采样时才拼出 next_state。**

因为 `s` 和 `s'` 重叠 3 帧，存两份完整帧栈是 2 倍浪费：

```python
self.states      = np.zeros((capacity, 4, 80, 80), dtype=np.uint8)
self.next_frames = np.zeros((capacity, 1, 80, 80), dtype=np.uint8)
```

重建的恒等式和主循环推进帧栈用的**必须是同一个**：

```python
next_state = concat(next_frame, state[:3])       # 新帧在前
```

否则网络训练时看到的时间顺序和游玩时相反。单测 4 专门验证这一点。

**用 uint8 存储。** 观测已经是 `{0,255}` 的二值图，存 float32 是 32 倍浪费。
归一化推迟到 GPU 上做：

```python
# flappy/model.py: forward()
if x.dtype == torch.uint8:
    x = x.float().div_(255.0)
```

这样 CPU→GPU 的 PCIe 流量也省了 4 倍。

内存账：`capacity × 5 × 6400` 字节。
- 40 万条 → 12.8 GB
- 10 万条 → 3.2 GB（16GB 内存的机器用这个）

旧实现存 float32 的双份完整帧栈 = 200KB/条，10 万条要 20GB。

### 为什么本项目不用 PER

优先经验回放（PER）按 TD 误差给样本加权，理论上更高效。但：

- 它引入 `α`、`β`、重要性采样权重、优先级更新等一堆容易写错的东西。
  旧管线的 PER 就把 `α` 施加了两次，且 `max_priority` 单调不降，
  实测约 **276 倍**偏向最近样本 —— 经验回放的去相关作用被完全抵消。
- 本项目的稀疏奖励问题已经由别的手段解决了：`frame_skip=4` 让每条经验
  携带完整的窗口奖励（不再丢 75%），目标网络同步频率提高了三个数量级。

**先把均匀采样跑对，再考虑 PER。** 这是本项目最重要的一条工程经验。

---

## 3. 技巧二：目标网络

### 问题

TD 目标里含有网络自己的输出：

```
y = r + γ · max_{a'} Q_θ(s', a')
```

对 `θ` 做梯度下降时，`y` 也跟着 `θ` 变 —— **目标在你脚下移动**。
这是发散的直接来源。

### 做法

再拿一份网络的拷贝 `Q_θ⁻` 专门算目标，它**冻结**不训练，隔一段时间才同步：

```python
# flappy/agent.py
self.target = DuelingDQN().to(device)
self.target.load_state_dict(self.online.state_dict())
self.target.eval()
for p in self.target.parameters():
    p.requires_grad_(False)          # 明确不参与梯度
...
if self.grad_steps % cfg['target_sync_grad_steps'] == 0:
    self.target.load_state_dict(self.online.state_dict())
```

在两次同步之间，目标是**固定的**，这一段内的优化就退化成了普通的监督回归 ——
有确定的拟合目标，稳定得多。

### 同步频率是个真正的权衡

- **太快** → 退化成没有目标网络，追自己的尾巴，容易发散。
- **太慢** → 价值传播极慢。每次同步，价值信息只能沿时间轴往回传**一步**。

本项目：`target_sync_grad_steps = 1000`，数的是**梯度步**。

> ### 旧管线在这里错得离谱
>
> `target_update_freq = 125000`，而且数的是**环境决策步**。
> 13 小时总共只同步了十几次。
>
> 而这个游戏里，一根管道从生成到被通过要约 12.5 次决策 ——
> "过管道 +20"这个信号要传回到"该扇翅膀了"那个决策，
> **至少需要 12 次同步**。十几次同步意味着价值总共只往回走了十几步。
>
> 这就是为什么智能体永远学不会提前规划。

### 怎么在日志里看它

`train.csv` 有 `syncs` 列（累计同步次数）。`plot.py` 会把每次同步画成竖线，
叠加在 loss 和 Q 曲线上。**健康的表现是 loss 在每次同步后出现锯齿** ——
目标跳变了，loss 应声上升，然后被重新拟合下去。

看不到锯齿，说明目标网络和在线网络已经几乎一样了（学到头了），
或者同步根本没发生。

---

## 4. 技巧三：Double DQN

### 问题：最大化偏差

标准 DQN 的目标是：

```
y = r + γ · max_{a'} Q_θ⁻(s', a')
```

`max` 这个操作会**系统性高估**。直觉：Q 的估计带噪声，
`max` 总是挑出那个"碰巧被高估"的动作。数学上，对任意随机变量
`E[max(X₁,X₂)] ≥ max(E[X₁], E[X₂])`。

这个偏差会随自举一路累积放大。

### 做法：解耦"选动作"和"打分"

用**在线网络选**动作，用**目标网络打**分：

```python
# flappy/agent.py: learn()
with torch.no_grad():
    a_star = self.online(s1).argmax(1, keepdim=True)   # 在线网络：选
    q_next = self.target(s1).gather(1, a_star)         # 目标网络：打分
    y = r.unsqueeze(1) + cfg['gamma'] * q_next * (~d).unsqueeze(1)
```

两个网络的噪声不完全相关，在线网络碰巧高估的那个动作，
目标网络未必也高估它。偏差被显著削弱。

改动只有一行，几乎零成本，所以基本没有理由不用。

### 注意 `(~d)` 这一项

```python
y = r + gamma * q_next * (~d)
```

`d` 是 done 标记。终止状态没有未来，`y` 必须等于 `r`。

**这一项依赖 done 标记的正确性。** 旧管线丢弃了 75% 的 terminal，
于是网络以为撞死之后还能继续拿奖励 —— 死亡的代价被严重低估，
自然学不会躲避。

---

## 5. 完整的 DQN 学习步

把三个技巧拼起来，就是 [`flappy/agent.py: Agent.learn()`](../../flappy/agent.py)：

```python
def learn(self, buffer):
    # 1. 经验回放：均匀采一批
    s, a, r, s1, d = buffer.sample(cfg['batch_size'])
    s  = torch.from_numpy(s).to(self.device, non_blocking=True)
    s1 = torch.from_numpy(s1).to(self.device, non_blocking=True)
    ...

    # 2. 当前估计 Q(s,a)
    q_sa = self.online(s).gather(1, a.unsqueeze(1))

    # 3. Double DQN 目标（目标网络冻结）
    with torch.no_grad():
        a_star = self.online(s1).argmax(1, keepdim=True)
        q_next = self.target(s1).gather(1, a_star)
        y = r.unsqueeze(1) + cfg['gamma'] * q_next * (~d).unsqueeze(1)

    # 4. Huber 损失 + 梯度裁剪
    loss = F.smooth_l1_loss(q_sa, y)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.online.parameters(), cfg['grad_clip'])
    self.optimizer.step()
    self.grad_steps += 1

    # 5. 定期同步目标网络
    if self.grad_steps % cfg['target_sync_grad_steps'] == 0:
        self.target.load_state_dict(self.online.state_dict())
```

### `gather` 在做什么

`self.online(s)` 输出形状 `(batch, 2)` —— 每个样本两个动作的 Q 值。
但我们只关心**实际执行过的那个动作**的 Q 值。

```python
q_sa = self.online(s).gather(1, a.unsqueeze(1))     # (batch, 1)
```

`gather(1, idx)` 沿第 1 维按索引取值。没被执行的那个动作不产生梯度 ——
这是对的，我们没有关于它的信息。

---

## 6. 优化器的几个选择

```python
self.optimizer = torch.optim.Adam(
    self.online.parameters(), lr=1.5e-4, eps=1.5e-4)
```

### `eps=1.5e-4`（不是默认的 1e-8）

Adam 的更新量约为 `lr · g / (√v + eps)`。当梯度很小时，
`eps` 太小会让分母趋近于 0，更新退化成 `lr · sign(g)` ——
**数值噪声和真实信号被同等放大**。

RL 的梯度本来就小且噪声大，所以 Rainbow 论文建议把 `eps` 调大到 1.5e-4。
本项目沿用。

### 没有 weight decay

L2 正则会把权重往 0 拉，对价值函数而言就是把 **Q 往 0 拉** ——
方向是错的。Q 应该收敛到真实回报，不该被拽向零。

### 没有学习率衰减

反直觉，但有证据：旧管线最好那轮把 LR 衰减了 36 倍，
结果 **loss 反而上升 53%**。那是"跟不上自己移动的目标"的特征 ——
LR 太小，不是太大。

RL 的目标一直在动（每次目标同步都会跳），所以不像监督学习那样
适合退火到很小。

### `grad_clip = 10.0`

Huber 已经限了单样本的梯度幅度，这里的全局范数裁剪只是防极端情况。
设成 1.0 会让几乎每一批都被重缩放，等于偷偷降低了学习率。

`train.csv` 的 `grad_norm` 记录的是**裁剪前**的范数，
可以用来判断阈值设得合不合适。

---

## 小结

| 技巧 | 解决什么 | 代码 | 参数 |
|------|---------|------|------|
| 经验回放 | 样本相关、分布漂移、样本效率 | `flappy/replay.py` | `buffer_capacity` |
| 目标网络 | 目标随参数移动 | `Agent.learn()` | `target_sync_grad_steps=1000` |
| Double DQN | max 的高估偏差 | `Agent.learn()` | 无 |
| Huber | TD 离群值 | `smooth_l1_loss` | 拐点 δ=1 |
| Adam eps 调大 | 小梯度下的噪声放大 | `Agent.__init__` | `1.5e-4` |

---

上一篇：[01 · 强化学习基础](01-rl-basics.md)
下一篇：[03 · Dueling 网络架构](03-dueling.md)
