# 00 · 为什么之前不收敛

> 本文引用的旧代码都已从工作区删除，但完整保留在 git 历史里。
> 想自己核对：`git show 9b67f03:continue_training.py`

## 结论先行

旧管线**不是超参数没调好，是数据管线在往网络里灌错误的训练样本**。

最严重的一个缺陷让每条经验的 `(状态, 动作, 奖励, 终止)` 四元组**同时错位**，
并且丢弃 75% 的奖励和 75% 的死亡信号。在那种数据上，无论学习率、
网络大小、训练时长怎么调，都学不出东西 —— 因为要拟合的目标本身是错的。

对照：

同一台机器（RTX 3070 Laptop）、同一个难度（`pipe_gap=100`）：

| | 旧管线 | 新管线 |
|---|---|---|
| 训练量 | 4 轮，最长 10 万局 / 13 小时 | 19000 局 / **100 分钟** |
| 100 局纯贪婪评测 | 约 1.3 根管道 | **389.6 根**（中位数 521） |
| 随机基线 | 0.63 根 | 0.63 根 |

旧管线跑了 13 小时，成绩只有随机策略的两倍。
新管线 4.4 分钟就越过了 50 根。

---

## 一、帧跳过错位（最致命）

### 旧代码

`continue_training.py` 主循环，`training_freq = 4`：

```python
while episode_count < MAX_EPISODES:
    # 动作选择
    if agent.step % agent.training_freq == 0:
        action_index = agent.select_action(s_t)
        a_t = np.zeros([2]); a_t[action_index] = 1
    else:
        # 非决策帧强制不跳跃
        a_t = np.array([1, 0])

    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    x_t1 = agent.preprocess_state(x_t1_colored)
    s_t1 = np.concatenate([x_t1, s_t[:, :, :3]], axis=2)

    episode_reward += r_t
    agent.step += 1                      # ← 夹在两个 % 判断之间

    if agent.step % agent.training_freq == 0:
        agent.decision_step += 1
        agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
```

### 逐帧推演

`agent.step` 从 0 开始，`training_freq = 4`：

| 进入循环时 step | 选动作？ | 实际执行的动作 | `step+=1` 后 | 存经验？ | 存进去的是 |
|---|---|---|---|---|---|
| 0 | 是，得 `a₀` | `a₀` | 1 | 否 | |
| 1 | 否 | **硬编码不跳** | 2 | 否 | |
| 2 | 否 | **硬编码不跳** | 3 | 否 | |
| 3 | 否 | **硬编码不跳** | 4 | **是** | `(s₃, a₀, r₃, s₄, terminal₃)` |

选动作的条件是 `step % 4 == 0`，存经验的条件是 `(step+1) % 4 == 0` ——
**这两个条件永远不会同时成立**。

### 四重后果

1. **动作标签是错的。** 存进去的 `a₀` 是 3 帧前、从状态 `s₀` 选出来的；
   而 `s₃` 那一帧实际执行的是硬编码的"不跳"。除非 `a₀` 恰好也是"不跳"，
   这条经验就在教网络"在 s₃ 做 a₀ 会得到 r₃"，而这件事从来没发生过。

2. **75% 的奖励被丢弃。** 只有 `r₃` 被存下，`r₀ r₁ r₂` 全部蒸发。
   过管道是发生在**某一帧**上的事件，落在那三帧里的 `+20` 就永远不会
   进入训练数据。稀疏奖励本来就难学，这里又砍掉四分之三。

3. **75% 的死亡信号被丢弃 —— 而且更糟。** 假设小鸟死在 `step≡1` 那一帧：
   `terminal` 在那次调用里是 True，但那次不存经验。旧环境
   （`wrapped_flappy_bird_fast.GameState.frame_step`）撞死时会在内部调用
   `self.__init__()` 直接开新局，循环毫无察觉地继续。等到 `step≡3` 存经验时，
   `terminal₃ = False`。**这条命就这么没了，网络永远不知道那里会死。**

4. **帧栈跨局污染。** 接上一条：`s₃` 这个 4 帧栈里，前面几帧属于上一局，
   后面几帧属于新一局。这是一个物理上不可能存在的状态，却被当作
   `done=False` 的正常样本存了进去。

> **"帧跳过"这个名字在旧代码里是名不副实的。** 标准做法是
> *action repeat*：选一个动作，在窗口内**重复执行**，**累加**窗口内所有奖励。
> 旧代码是"选一次动作，然后硬编码地滑行 3 帧"，两者完全不是一回事。

### 新代码怎么做

[`flappy/rollout.py: skip_step()`](../../flappy/rollout.py)：

```python
for i in range(k):
    obs, r, done, info = env.step(action, render=(i == k - 1))
    r_raw += r                  # 每一帧的奖励都累加
    if done:
        if obs is None:
            obs = env.observe()  # 补画真实的崩溃帧
        break                    # terminal 立刻上报
```

一次决策 = 一条经验。主循环里**不存在任何 `% frame_skip`**，
所以不可能再出现"选动作用一个计数器、存经验用另一个"的错位。

单测 5 (`test_frame_skip_alignment`) 逐帧核对动作重复与奖励累加，
旧实现必然失败。

---

## 二、目标网络几乎从不同步

```python
self.target_update_freq = 125000
...
if self.decision_step % self.target_update_freq == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

125000 个**决策步**才同步一次。按旧管线的实测速度，13 小时总共只同步了
十几次。

### 为什么这是致命的

Q-learning 靠 Bellman 方程把价值**一步一步往回传**：

```
Q(s_t, a) ← r + γ · max_a' Q_target(s_{t+1}, a')
```

`Q_target` 是冻结的。价值信息每次同步只能往回传播**一步**。

在这个游戏里，一根管道从生成到被通过，中间隔约 12.5 次决策。
也就是说，"过管道 +20"这个信号要传回到"应该在这里扇一下翅膀"那个决策，
**至少需要 12 次目标网络同步**。

十几次同步，意味着价值信息总共只往回走了十几步 —— 刚好够传播一根管道的距离。
网络根本没有机会学会"提前规划"。

### 新代码

```python
target_sync_grad_steps = 1_000      # 数的是【梯度步】不是环境步
```

按梯度步计数，一次典型训练有几千次同步，比旧管线高三个数量级。

**这里有个容易踩的坑**：`decision_step` 和 `grad_step` 是不同的单位。
新配置里所有计数量都把单位写进了名字（`*_decisions` / `*_grad_steps`），
就是为了让这类错误一眼可见。

---

## 三、BatchNorm + Dropout，而且从不切换 eval 模式

### 网络定义

```python
self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
self.bn3 = nn.BatchNorm2d(64, momentum=0.01)
...
nn.Dropout(0.3)
```

### 模式切换

全文件搜索 `.eval()` 只有两处，都在 `load_model()` 里：

```python
self.q_network.load_state_dict(checkpoint)
self.target_network.load_state_dict(checkpoint)
self.q_network.eval()
self.target_network.eval()
```

而 `select_action()` 只用了 `torch.no_grad()`：

```python
def select_action(self, state):
    if random.random() < self.epsilon:
        return random.randrange(self.actions)
    with torch.no_grad():                    # ← 只关梯度，不切模式
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
```

`train()` 里也没有任何模式切换。PyTorch 模块**默认就是 train 模式**。

### 三个后果，一个比一个严重

**(a) "贪婪"动作其实是随机的。**
Dropout(0.3) 在 train 模式下会随机丢弃 30% 的激活。所以
`q_values.argmax()` 每次调用都可能给出不同的动作。ε=0 也没用。

**(b) 单样本 BatchNorm 是另一个函数。**
选动作时 batch size = 1。BatchNorm 在 train 模式下用**当前 batch 的统计量**
做归一化 —— 一个样本时，等于把每个通道的空间均值减掉、除以自己的空间标准差。
这和训练时（batch=128 的统计量）完全是两个不同的变换。
**网络在"行动"时用的函数，和在"学习"时被优化的函数，不是同一个。**
顺带一提，`momentum=0.01` 还让每次单样本前向都去污染 running stats。

**(c) 目标网络在 train 模式 → Bellman 算子没有不动点。**
这是最深的一条。TD 目标是

```
y = r + γ · Q_target(s', a*)
```

如果 `Q_target` 处于 train 模式，它的 BatchNorm 会用**当前 minibatch 的统计量**。
那么 `y` 就不只依赖 `s'`，还依赖"这一批里碰巧还有哪些别的样本"。

Bellman 算子 T 之所以能收敛，前提是它是一个**固定的**压缩映射，
存在唯一不动点 Q\*。一个随批次组成而变化的算子不是固定算子，
**不动点根本不存在**。这不是"收敛慢"，是"数学上就没有收敛目标"。

### 新代码

网络里**彻底不含** BatchNorm 和 Dropout，见 [`flappy/model.py`](../../flappy/model.py)。
这样 `train()` / `eval()` 变成空操作，再也不可能因为漏调用而出错。

而且加了一行自检，训练、评测、可视化三条路径启动时都会跑：

```python
def assert_deterministic(net, cfg, device):
    with torch.no_grad():
        q1, q2 = net(dummy), net(dummy)
    assert torch.allclose(q1, q2), "network is stochastic at act time"
```

**这一行当初就能抓到这个 bug。** 单测 6 也覆盖了它。

---

## 四、terminal 观测是下一局的第一帧

旧环境 `GameState.frame_step()`：

```python
if isCrash:
    terminal = True
    self.__init__()          # ← 先重置
    reward = -2

# draw sprites
SCREEN.blit(IMAGES['background'], (0,0))    # ← 后绘制
...
image_data = pygame.surfarray.array3d(...)
return image_data, reward, terminal
```

崩溃时先调 `self.__init__()` 重置整局，**然后**才绘制画面。
所以 `terminal=True` 那一帧返回的图像，是**新一局的第一帧**。

网络看到的是："小鸟在屏幕中央、管道在最右边 → 这个状态价值 -2"。
而真正撞上管道的那一帧画面，**从来没有进入过训练数据**。

### 新代码

`FlappyEnv.step()` **绝不内部 reset**，崩溃帧被如实绘制并返回；
回合结束后必须由调用方显式 `reset()`，否则 `step()` 抛 `RuntimeError`。
单测 3 验证 terminal 观测 ≠ reset 观测。

---

## 五、优先经验回放的 α 被施加了两次

```python
# 存的时候
priority = (abs(float(td_error)) + 1e-6) ** self.alpha
self.max_priority = max(self.max_priority, priority)

# 采样的时候
probs = valid_priorities ** self.alpha
```

优先级在**存入时**已经取过一次 `** α`，采样时又取一次 —— 实际生效的是 `α²`
对应的指数，即 `p ∝ |TD|^(2α)`。分布被极度尖锐化。

雪上加霜的是 `self.max_priority = max(self.max_priority, priority)`：
它**单调不降**。新样本一律以 `max_priority` 入库，而这个值只涨不跌。
于是最近的样本永远拿着历史最高优先级。

两者叠加，实测约 **276 倍**地偏向最近的样本 ——
经验回放存在的意义（打破样本间的时间相关性）被完全抵消，
等于退化成了在线学习，而在线学习配神经网络本来就是发散的经典配方。

### 新代码

先用**均匀采样**。管道样本稀疏的问题由两件事解决：
`frame_skip=4` 让每条经验都携带完整的窗口奖励（不再丢 75%），
以及高出三个数量级的回传次数。PER 是后面可以再加的优化，不是必需品。

---

## 其余放大伤害的问题

### 奖励用赋值而非累加

```python
if pipeMidPos <= playerMidPos < pipeMidPos + 4:
    self.score += 1
    reward = 20              # ← 赋值，覆盖掉存活奖励和势能项
...
if isCrash:
    terminal = True
    reward = -2              # ← 又一次赋值，把上面的 +20 整个吃掉
```

同一帧既过管道又撞死时，网络收到的信号是 `-2` —— "过管道是坏事"。
新代码一律 `+=` 累加（单测 2）。

### 奖励尺度 +20 / −2

Huber 损失的 δ=1 拐点、梯度裁剪阈值，都是按"奖励量级约为 1"设计的。
`+20` 会让几乎每个含管道的样本都落在 Huber 的线性区，
梯度被常数化，携带的信息量骤减。新代码归一化到 ±1。

### 势能塑形被截断，且算的是上一步

```python
potential_reward = gamma * current_potential - self.previous_potential
return max(-0.01, min(0.01, potential_reward))     # ← 截断
```

Ng-Harada-Russell 1999 保证"势能差分形式的塑形不改变最优策略"，
**前提是它精确等于 `γΦ(s') − Φ(s)`**。一截断，保证就失效了，
塑形项变成了会扭曲最优策略的普通奖励。

而且 `_calculate_potential_reward()` 在动作施加**之前**被调用，
算出来的是 `γΦ(s_t) − Φ(s_{t-1})`，与当前动作完全无关。

新代码只暴露 `Φ`，差分由调用方在**决策级**计算，不截断（单测 1，
望远镜求和误差 3e-17）。

### 静默回退到 CPU

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

机器上有 RTX 3060，torch 装成了 CPU 版，于是训练慢了 20 倍 ——
而日志里什么都看不出来。新代码在没有 CUDA 时**直接报错退出**，
要用 CPU 必须显式 `--allow-cpu`。

### 评测跑的其实是随机策略

旧评测脚本调 `agent.select_action()`，而它里面有一条

```python
if self.decision_step < self.training_start:    # 20000
    return random.randrange(self.actions)
```

`load_model()` 只恢复两个 `state_dict`，**从不恢复 `decision_step`**。
新建的 agent 其 `decision_step = 0`，于是每次评测的前 20000 次决策都是纯随机的。
`saved_networks/` 里 113 个存档，**没有一个有过可信的评测数据**。

新代码里训练、评测、回放共用 `flappy/rollout.py` 的同一份动作选择，
不存在任何按训练进度短路的分支。

### "最佳模型"从来没保存过最佳模型

`max_score` 在每局都执行的分支里被重新赋值，2 万局之后阈值恒为 0，
于是每个正分局都算"新纪录" —— 触发了 31182 次 58MB 存盘，
却从未保存过任何真正意义上的最佳模型。

新代码把门控放在**平滑指标**（近百局均分）上，加 2% 提升门槛和 100 局冷却。

---

## 为什么这些 bug 这么难发现

这是本文最值得记住的一点。

**监督学习的 bug 会喊，RL 的 bug 只会沉默。**

图像分类里数据管线错了，准确率会卡在 10%，你立刻知道出事了。
而 RL 里：

- 损失函数**照样在下降**。旧管线的 loss 降到 0.002，看起来很"健康" ——
  但那恰恰是**最坏**的信号：网络完美拟合了一个长期不动的、错误的目标。
  一根管道值 +20，而 loss 只有 0.002，这两个数放在一起本身就荒谬。
- 分数**照样有波动**，偶尔还会冲高，很容易解读成"在学，只是慢"。
- 没有任何异常抛出。每一行代码都"能跑"。

而旧管线的日志只往 stdout 打一行带 emoji 的文本，画图脚本再用正则去抓，
**只能拿到 `(episode, score, max, avg100)` 四个量**：

- 看不到 loss → "loss 0.002 vs 管道 +20"这个铁证没人看见
- 看不到 Q 值 → 价值函数根本没在学，也看不出来
- 看不到目标网络同步时刻 → 每次同步后近百局均分掉 26 分，看不出来

**13 小时里所有问题都没被发现，直接原因就是没有结构化日志。**

## 新管线怎么防止复发

不是靠"以后小心点"，是靠结构：

1. **10 个单测**，每一个都对应上面一个真实存在过的缺陷。
   改坏了会红，不需要跑 13 小时才发现。
2. **结构化 CSV 日志**（`episodes.csv` / `train.csv` / `eval.csv`），
   loss、Q 分布、TD 误差、梯度范数、同步时刻全部落盘。
3. **架构约束**：三条路径共用一份动作选择；网络里禁止 BN/Dropout；
   计数量的单位写进变量名；没有 CUDA 直接报错。
4. **启动自检**：`assert_deterministic` 一行拦住 (c) 类问题。

> 一个耐人寻味的细节：新管线的**第一次**训练就收敛了。
> 不是调了很多轮参数调出来的 —— 把数据管线修正确之后，
> 标准的 Double-DQN 超参数直接就能用。
>
> **在 RL 里，"调参"往往是在给数据管线的 bug 打掩护。**

---

下一篇：[01 · 强化学习基础](01-rl-basics.md)
