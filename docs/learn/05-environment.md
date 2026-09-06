# 05 · 环境与观测

网络能学到什么，上限由**喂给它的观测**决定。这一篇讲从游戏画面到
`(4,80,80)` 张量之间发生的每一步，以及每一步的理由。

代码：[`game/flappy_env.py`](../../game/flappy_env.py)、
[`flappy/rollout.py`](../../flappy/rollout.py)

---

## 1. 游戏本身

```
屏幕 288 × 512，地面在 y = 404 (BASEY)

物理（每帧）：
  重力         playerVelY += 0.5      （不超过 playerMaxVelY = 5）
  扇翅         playerVelY  = −5       （瞬间赋值，不是叠加）
  位移         playery    += playerVelY
  管道左移     pipe.x     += −5

管道：
  间隙 pipe_gap = 100 px（原版难度；150 = 放宽）
  间隙上沿从 {30,40,...,100} + 0.2·BASEY 随机取
  小鸟高 24 px
```

一次扇翅令 `velY = −5`，重力每帧 `+0.5`，所以扇翅后约 **19 帧**净位移归零。
这个数字后面会反复用到。

---

## 2. 观测流水线

```
pygame Surface (288×512×3 RGB)
   │  ① pixels3d 取视图（零拷贝），只取一个通道
   ▼
(288, 512) 单通道
   │  ② cv2.resize INTER_AREA → 80×80
   ▼
(80, 80) 灰度
   │  ③ threshold(1, 255, BINARY) 二值化
   ▼
(80, 80) uint8 ∈ {0, 255}
   │  ④ FrameStack：叠最近 4 次决策的末帧
   ▼
(4, 80, 80) uint8   ← 网络输入
```

### ① 为什么不用 `array3d`

```python
view = pygame.surfarray.pixels3d(SCREEN)      # 视图，零拷贝
try:
    small = cv2.resize(view[:, :, 0], (80, 80), interpolation=cv2.INTER_AREA)
finally:
    del view                                   # 必须解锁 Surface
```

`pygame.surfarray.array3d()` 会做一次 `288×512×3` 的转置**拷贝**，
实测 762 µs —— 占整个 `env.step` 的 78%。

`pixels3d` 返回的是**视图**，零拷贝。再让 cv2 直接在视图的单通道上做缩放，
整条路径从 **977 µs 降到 234 µs**，且输出逐像素等价
（600 帧核对：384 万像素里仅 128 个不一致 = 0.0033%，单测 8）。

> **`pixels3d` 会锁定 Surface**，必须在返回前 `del` 掉，
> 否则下一次 `blit` 会失败。所以用了 `try/finally`。

**只取单通道**是安全的，因为二值化的阈值是 1（"非纯黑"）——
在这个游戏的调色板下，判断 `R>0` 与判断加权灰度 `>0` 等价。

### ② 为什么用 `INTER_AREA`

从 288×512 缩到 80×80 是大幅**下采样**。`INTER_AREA` 做的是区域平均，
不会漏掉细结构；而 `INTER_NEAREST` / `INTER_LINEAR` 在这个缩放比下
可能整根管道边缘都采样不到。

注意这里**不保持长宽比**，288×512 被压成方形。这没关系 ——
网络学的是相对位置，形变是一致的。

### ③ 为什么二值化

游戏画面本来就是纯色块（黑背景、绿管道、红鸟）。二值化：

- 去掉所有无关的颜色/纹理信息，网络不用浪费容量去忽略它们
- 让观测变成 `{0,255}`，**用 uint8 存经验回放毫无精度损失**
- 消除背景细节带来的干扰

代价：小鸟和管道在二值图里都是白色，靠**形状和位置**区分。
实践证明卷积网络完全能分开。

---

## 3. 帧栈：为什么必须叠 4 帧

### 单帧不满足马尔可夫性

看一张静止的画面，你**不知道小鸟在上升还是下落**。同一个位置，
上升中和下落中的最优动作完全相反。

`P(s_{t+1} | s_t, a_t)` 要成立，`s_t` 就必须包含速度信息。

### 叠帧提供了什么

| 帧数 | 能推断出 |
|---|---|
| 1 | 位置 |
| 2 | 位置 + 速度（一阶差分） |
| 3 | + 加速度（二阶差分） |
| 4 | + 冗余，抗噪 |

本项目物理只需要位置和速度（加速度是常数 0.5），2 帧理论上够。
用 4 帧是 DQN 的惯例，多一点冗余不吃亏，而且和 `frame_skip=4` 对齐得漂亮。

### 实现

```python
# flappy/rollout.py
class FrameStack:
    def reset(self, frame):
        """新回合：用首帧填满整个栈。"""
        self.array = np.repeat(frame[None], self.n, axis=0)
        return self.array

    def push(self, frame):
        self.array = np.concatenate([frame[None], self.array[:self.n - 1]], axis=0)
        return self.array
```

**新帧在前，丢掉最旧的一帧。** 这个恒等式必须和经验回放里重建
`next_state` 的方式**逐位一致**：

```python
# flappy/replay.py: sample()
s1 = np.concatenate([self.next_frames[idx], s[:, :self.stack - 1]], axis=1)
```

否则网络训练时看到的时间顺序和游玩时相反 —— 一个极难发现的 bug。
单测 4 专门验证这条一致性。

### `reset()` 为什么必须重填

旧代码在 terminal 那一轮也执行无条件的 `s_t = s_t1`，
于是新回合的**前 3 次决策**都拿着混有上一局画面的帧栈，
并以 `done=False` 存进缓冲区 —— 约 23% 的经验是物理上不可能的状态。

单测 4 会检查采样出的帧栈里不含跨局帧。

---

## 4. 帧跳过：一次决策管 4 帧

### 为什么要跳

游戏 60 FPS 意义上每帧都可以决策，但：

1. **相邻帧信息高度冗余**。连续两帧几乎一样，逐帧决策浪费算力。
2. **时间视野被拉长**。γ=0.99 覆盖约 100 次决策 —— 逐帧决策只有 100 帧
   （1.7 秒），4 帧一决策就是 400 帧（6.7 秒），足以看到下一根管道。
3. **探索更有效**。随机策略下，逐帧的随机动作会互相抵消成"原地抖动"；
   动作保持 4 帧才能产生有意义的位移。

### 实现

```python
# flappy/rollout.py
def skip_step(env, action, phi_prev, cfg):
    r_raw = 0.0
    k = cfg['frame_skip']
    for i in range(k):
        obs, r, done, info = env.step(action, render=(i == k - 1))
        r_raw += r                      # 每一帧的奖励都累加
        if done:
            if obs is None:
                obs = env.observe()      # 补画真实的崩溃帧
            break
    phi_next = 0.0 if done else info['potential']
    r_dec = r_raw + cfg['shaping_coef'] * (cfg['gamma'] * phi_next - phi_prev)
    return obs, r_dec, done, info, phi_next
```

三个关键点：

1. **动作在窗口内重复**（action repeat），不是"选一次然后滑行"。
2. **奖励全额累加**。过管道发生在某一帧，落在窗口任意位置都不会丢。
3. **terminal 提前 break**，并补画真实的崩溃帧。

> 旧管线这三条**全错**：非决策帧硬编码"不跳"、只保留最后一帧的奖励、
> 死在窗口中间时 terminal 直接丢失。详见
> [00 · 为什么之前不收敛](00-why-it-failed.md) 第一节。

### `render=(i == k-1)` —— 本项目最大的单项提速

窗口内只有**最后一帧**需要绘制取图，前面 3 帧的画面根本不会进帧栈。

实测：物理 5.8 µs，绘制+取图 970 µs。跳过 3 次渲染，
端到端吞吐从 **204 决策/秒涨到 2328 决策/秒**（11 倍）。

安全性：碰撞检测走 hitmask 和坐标，**不依赖渲染结果**。
单测 7 逐帧核对了"渲染与否物理完全一致"。

---

## 5. 环境的两条硬约定

### `step()` 绝不内部 reset

```python
def step(self, action, render=True):
    if self._done:
        raise RuntimeError(
            "step() called on a finished episode; call reset() first")
```

旧环境在崩溃时先调 `self.__init__()` 再绘制，于是 terminal 那一帧
返回的是**下一局的第一帧**。网络从来没见过真正的碰撞画面。

现在回合结束必须由调用方显式 `reset()`，忘了会立刻抛异常而不是静默出错。
单测 3 验证 terminal 观测 ≠ reset 观测。

### 奖励一律 `+=` 累加

```python
reward = self.alive_reward            # 累加起点
...
if scored:
    reward += self.pipe_reward * scored
...
if checkCrash(...):
    self._done = True
    reward += self.death_reward       # += 而非 =
```

同一帧可能既过管道又撞死。旧代码两处都用赋值，`-2` 把 `+20` 整个吃掉，
网络收到的信号是"过管道是坏事"。单测 2 用不对称的奖励值验证这一点。

---

## 6. 得分判定：显式跨越测试

```python
player_mid = self.playerx + PLAYER_WIDTH / 2.0
for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
    prev_mid = uPipe['x'] + PIPE_WIDTH / 2.0
    uPipe['x'] += self.pipeVelX
    lPipe['x'] += self.pipeVelX
    new_mid = uPipe['x'] + PIPE_WIDTH / 2.0
    if new_mid <= player_mid < prev_mid:      # 本帧恰好跨越
        scored += 1
```

旧代码用固定的 4 像素窗口 `pipeMid <= playerMid < pipeMid + 4`，
而管道每帧移动 5 px —— **窗口比步长还小**，只是靠 x 坐标奇偶性的
算术巧合才没漏判。改个速度或初始位置就会开始漏分。

显式跨越测试与位移严格对齐，不依赖任何巧合。

---

## 7. 难度旋钮：`pipe_gap`

```python
FlappyEnv(pipe_gap=100)     # 100 = 原版，150 = 放宽，越小越难
```

只有这一个旋钮，间隙上沿的候选高度固定不变 ——
这样难度变化可以**精确归因**到间隙本身。

```python
if self.pipe_gap < PLAYER_HEIGHT + 8:
    raise ValueError("pipe_gap=%d 对高度 %d 的小鸟来说无法通过" % ...)
```

> **`pipe_gap` 会被写进存档的 `config`。** 这不是冗余：
> 拿今天的默认难度去评测昨天用另一个难度训练的模型，分数没有可比性。
> `eval.py` 因此自动采用训练时的难度，除非你显式 `--pipe-gap` 覆盖。

单测 9 验证间隙真的生效，且不可通过的间隙被拒绝。

---

## 8. 无头运行

```python
# game/resources.py，在 pygame.init() 之前
os.environ['SDL_VIDEODRIVER'] = 'dummy'
```

训练不需要真实窗口，SDL 用 dummy 驱动渲染到内存 Surface。
这样能在 Docker、SSH、无显示器的服务器上跑。

**副作用**：这是**导入时**的无条件设置，外部无法覆盖。
所以 `play.py` 用 **cv2** 显示而不是 pygame ——
`pygame.display.quit()` 之后原 SCREEN 直接作废，
重建显示会让所有已 `convert()` 的精灵失效。

热循环里也移除了 `display.update()` 和 `FPSCLOCK.tick()`：
无头模式下前者没有意义，后者会人为限速。
实测吞吐 439 fps → 1464 fps。

---

## 小结

| 环节 | 做法 | 理由 |
|---|---|---|
| 取图 | `pixels3d` 视图 + 单通道 | 977µs → 234µs |
| 缩放 | `INTER_AREA` → 80×80 | 大幅下采样不漏细结构 |
| 二值化 | 阈值 1 | uint8 存储无精度损失 |
| 帧栈 | 最近 4 次决策的末帧 | 单帧不满足马尔可夫性 |
| 帧跳过 | 动作重复 4 帧，奖励累加 | 视野 ×4，探索更有效 |
| 窗口内渲染 | 只渲染最后一帧 | 吞吐 ×11 |
| reset | 显式，`step` 绝不内部重置 | terminal 观测必须真实 |
| 奖励 | 一律 `+=` | 同帧可能既得分又撞死 |
| 得分 | 显式跨越测试 | 固定像素窗口会漏判 |

---

上一篇：[04 · 神经网络详解](04-network.md)
下一篇：[06 · 奖励设计与势能塑形](06-reward.md)
