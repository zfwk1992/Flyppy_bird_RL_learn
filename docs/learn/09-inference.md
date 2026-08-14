# 09 · 推理过程

训练完之后，**怎么用这个网络玩游戏**？这一篇追踪一次前向传播的
完整数据流，以及评测和可视化两条路径。

代码：[`eval.py`](../../eval.py)、[`play.py`](../../play.py)、
[`flappy/checkpoint.py`](../../flappy/checkpoint.py)

---

## 1. 推理 vs 训练：区别在哪

| | 训练 | 推理 |
|---|---|---|
| 动作选择 | ε-贪婪（ε 随进度退火） | 贪婪（ε=0 或 0.01） |
| 梯度 | 需要 | `torch.no_grad()` |
| 经验存储 | 存进缓冲区 | 不存 |
| 目标网络 | 需要 | 不需要 |
| batch | 128 | 1 |
| 帧跳过 | 只渲染窗口最后一帧 | 评测同训练；回放全渲染 |

**唯一实质区别是不再探索、不再学习。** 网络的前向计算完全一样。

---

## 2. 一次推理的完整数据流

```
游戏画面 (288×512×3 RGB Surface)
   │
   │ ① pixels3d 视图 → 单通道 → INTER_AREA 缩放 → 二值化
   ▼
(80, 80) uint8 ∈ {0,255}
   │
   │ ② FrameStack.push：新帧在前，丢最旧一帧
   ▼
(4, 80, 80) uint8                        ← numpy
   │
   │ ③ torch.from_numpy(...).unsqueeze(0).to(device)
   ▼
(1, 4, 80, 80) uint8                     ← GPU 张量
   │
   │ ④ 网络内部 .float().div_(255.0)
   ▼
(1, 4, 80, 80) float32 ∈ [0,1]
   │
   │ ⑤ conv1 → conv2 → conv3 → flatten → fc
   ▼
(1, 512) 特征
   │
   │ ⑥ 双头
   ├──→ value     → (1, 1)   V(s)
   └──→ advantage → (1, 2)   A(s,a)
   │
   │ ⑦ Q = V + A − mean(A)
   ▼
(1, 2)  例如 [6.21, 6.83]
   │
   │ ⑧ argmax
   ▼
action = 1 (扇翅)
   │
   │ ⑨ skip_step：在 4 帧窗口内重复这个动作
   ▼
回到游戏
```

对应代码 [`flappy/agent.py`](../../flappy/agent.py)：

```python
@torch.no_grad()
def act(self, stack_uint8):
    t = torch.from_numpy(stack_uint8).unsqueeze(0).to(self.device)
    return int(self.online(t).argmax(1).item())
```

三个细节：

- **`@torch.no_grad()`** —— 不建计算图，省显存也更快
- **`unsqueeze(0)`** —— 加 batch 维，`(4,80,80)` → `(1,4,80,80)`
- **传的是 uint8** —— 归一化在网络内部的 GPU 上做，PCIe 流量省 4 倍

---

## 3. 加载存档

```python
# flappy/checkpoint.py
def load_for_inference(path, device, **cfg_overrides):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt, **cfg_overrides)

    net = DuelingDQN().to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    assert_deterministic(net, cfg, device)
    return net, cfg, ckpt
```

### `config_from_checkpoint`

配置从**存档里**读，不用今天的默认值：

```python
def config_from_checkpoint(ckpt, **overrides):
    cfg = dict(CONFIG)
    cfg.update(ckpt.get('config', {}))     # 存档里的优先
    ...
```

为什么重要：`pipe_gap` 是难度旋钮。用今天的默认难度去评测
昨天用另一个难度训练的模型，分数**没有可比性**。

要跨难度比较就显式覆盖：

```bash
python eval.py runs/A/best.pt --pipe-gap 100
python eval.py runs/B/best.pt --pipe-gap 100      # 钉死同一难度
```

### `.eval()` 和确定性自检

本项目网络里没有 BatchNorm/Dropout，所以 `.eval()` 是空操作 ——
但还是调用它，作为习惯和防御。

真正起作用的是下一行：

```python
def assert_deterministic(net, cfg, device):
    with torch.no_grad():
        q1, q2 = net(dummy), net(dummy)
    assert torch.allclose(q1, q2), "network is stochastic at act time"
```

同一个输入前向两次，结果必须一样。

> **这一行当初就能抓到旧管线的 bug。** 旧网络有 `Dropout(0.3)`，
> 而 `select_action()` 只用了 `torch.no_grad()` 没调 `.eval()`，
> 于是"贪婪"动作每次调用都可能不同。ε=0 也没用。

---

## 4. 评测路径

```bash
python eval.py runs/<时间戳>/best.pt --episodes 100 --q-stats
```

核心在 [`flappy/rollout.py: evaluate()`](../../flappy/rollout.py)：

```python
@torch.no_grad()
def evaluate(net, cfg, device, n_episodes, epsilon=0.0, max_decisions=None,
             q_stats=False, env=None):
    for _ in range(n_episodes):
        stack = stacker.reset(env.reset())
        phi = env.current_potential()
        n_dec, ep_ret = 0, 0.0
        while True:
            if epsilon > 0 and random.random() < epsilon:
                action = sample_random_action(cfg)
            else:
                q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                action = int(q.argmax().item())
            frame, r, done, info, phi = skip_step(env, action, phi, cfg)
            n_dec += 1
            if done or n_dec >= max_decisions:
                ...
                break
            stack = stacker.push(frame)
```

**训练中的定期评测和 `eval.py` 用的是这同一个函数。**
这是刻意的架构约束 —— 旧管线的评测脚本自己复制了一份动作选择逻辑，
副本里带着按训练进度短路的分支，导致所有评测跑的都是随机策略。

### `max_decisions` 上限是必需的

不是保险丝：**一个学好的策略可以永远不死**。
实测已出现连过 387 根管道的局。没有上限，评测会直接挂住。

截断的局计入 `truncated`（"还活着"），不计为死亡：

```
hit the cap   : 7 / 100 episodes (still alive when stopped)
```

看到这一行有数字，说明策略已经强到跑满上限了 ——
这时候 `pipes_mean` 反而低估了真实水平。

### 输出解读

```
checkpoint : runs/gpu_run1/best.pt
  saved at  : episode 15470, decision_step 1284531
  recent100_pipes at save time: 13.34
  frame_skip=4 frame_stack=4 pipe_gap=100
  eval episodes : 100  (epsilon=0.000, pipe_gap=100)
  pipes mean    : 89.050 +- 97.070
  pipes median  : 61.0    max: 387
  ep len (dec)  : 665.6  (cap 2000)
  hit the cap   : 0 / 100 episodes
  ep return     : 88.2140
  mean max-Q    : 10.234   (low-score rule: ~0.9 x pipes; ceiling ~12.6)
  mean |Q1-Q0|  : 0.412    (action preference strength)
```

| 指标 | 怎么读 |
|---|---|
| `pipes mean ± std` | **标准差通常很大**，RL 策略天然高方差 |
| `pipes median` | 比均值更能反映"典型一局"，均值容易被超长局拉高 |
| `hit the cap` | 非 0 说明被上限截断，真实水平更高 |
| `mean max-Q` | 低分段 ≈ 0.9×管道数，饱和上限 12.6 |
| `mean \|Q1-Q0\|` | 动作偏好强度；接近 0 说明网络分不清两个动作 |

> **`recent100_pipes at save time` 和 `pipes mean` 通常差很多**
> （这里 13.34 vs 89.05）。前者是**训练时带 ε 的行为策略**成绩，
> 后者是**纯贪婪**成绩。别拿它们直接比。

---

## 5. 可视化路径

```bash
python play.py runs/<时间戳>/best.pt --fps 30 --scale 2
python play.py runs/<时间戳>/best.pt --record demo.mp4 --no-window
python play.py --human                       # 自己玩
```

### 为什么用 cv2 显示而不是 pygame

这不是偏好，是硬约束：

1. `game/flappy_bird_utils.py` 用 `.convert()` / `.convert_alpha()` 加载精灵，
   这些 Surface **绑定当前显示格式**
2. `game/resources.py` 在**导入时**无条件执行
   `os.environ['SDL_VIDEODRIVER'] = 'dummy'`，外部无法抢先覆盖
3. `pygame.display.quit()` 之后原 SCREEN 直接作废，
   重建显示会让所有已 convert 的精灵失效

cv2 直接显示 numpy 帧，完全绕开 SDL，对环境零改动。

### HUD 只画在拷贝出来的图上

```python
def draw_hud(img, pipes, n_dec, q, action, fps, eps, paused, human):
```

**绝不能**往 SCREEN 上 blit 分数 —— 那会污染下一帧的观测，
网络就看见分数数字了。HUD 用 `cv2.putText` 画在拷贝上。

### 回放时 4 帧全渲染

```python
for _ in range(k):
    obs_last, _, done, info = env.step(action, render=True)
    pending.append(env.raw_obs())
```

训练时窗口内只渲染最后一帧（为了速度）；给人看时全渲染，
否则画面是 4 倍快进且发跳。**动作在窗口内保持不变，
与训练时的语义完全一致。**

渲染开销约 1ms/帧，在 30fps 下完全无所谓。

### 推理速度根本不是瓶颈

GPU 单样本前向 **0.62 ms** = 1611 次/秒，
而 30 FPS 实时游玩只需要 7.5 次决策/秒 —— 快了 200 倍。

所以 `play.py` 里没有任何针对推理的优化，只有**节流**
（不然画面快到看不清）。

---

## 6. Q 值可视化：看网络在想什么

`play.py` 的 HUD 会画出两个动作的 Q 值和它们的差值条：

```
PIPES 23                              STAY  +6.21
step 187                              FLAP  +6.83   ← 绿色 = 被选中
42 fps                            ├────────█████┤   差值条
```

观察要点：

- **大部分时间差值条很短**。这印证了 Dueling 的前提 ——
  多数状态下动作无所谓，`V(s)` 才是主要成分。
- **小鸟接近管道边缘时差值条突然拉长**。优势函数在起作用，
  网络明确知道"这一帧必须扇"。
- **Q 的绝对值随局面变好而上升**，接近 12.6 时说明网络认为
  自己基本不会死了。

差值条一直很短、或者一直很长且不变，都是有问题的信号 ——
前者说明网络分不清动作，后者说明它对某个动作有恒定偏好（可能塌缩了）。

---

## 7. 把模型用到别处

网络本身就是一个普通的 PyTorch 模块，可以脱离本项目使用：

```python
import torch
from flappy.model import DuelingDQN

ckpt = torch.load('runs/gpu_run1/best.pt', map_location='cpu',
                  weights_only=False)
net = DuelingDQN()
net.load_state_dict(ckpt['model'])
net.eval()

# 输入 (N,4,80,80) uint8 或 float32∈[0,1]，输出 (N,2) Q 值
q = net(torch.zeros(1, 4, 80, 80, dtype=torch.uint8))
action = int(q.argmax())
```

要注意的只有：**输入必须和训练时同一条预处理管线**
（80×80、二值化、4 帧、新帧在前）。喂进去别的东西，
网络会给出毫无意义的输出，而且不会报错。

导出 ONNX 也是直接的：

```python
torch.onnx.export(net, torch.zeros(1,4,80,80), 'flappy.onnx',
                  input_names=['state'], output_names=['q'])
```

---

## 小结

| 环节 | 关键点 |
|---|---|
| 预处理 | 必须与训练完全一致，否则静默失效 |
| `@torch.no_grad()` | 不建图，省显存 |
| 输入 uint8 | 归一化在 GPU 内部做 |
| `.eval()` + 确定性自检 | 本项目是空操作，但自检能抓住未来的回归 |
| 存档带 config | 自动用训练时的难度评测 |
| `max_decisions` 上限 | 学好的策略可以永不死 |
| 评测共用 `evaluate()` | 与训练中的评测同一份代码 |
| HUD 画在拷贝上 | 否则污染观测 |
| 回放全渲染 | 否则画面 4 倍快进 |

---

上一篇：[08 · 训练流程](08-training.md)
下一篇：[10 · 怎么诊断训练](10-diagnostics.md)
