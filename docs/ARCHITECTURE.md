# 架构说明

本文讲**代码怎么组织、数据怎么流**。算法上"为什么这么选"的理由在
[README.md](../README.md) 的"关键设计"一节，两边不重复。

---

## 1. 三层结构

```
┌──────────────────────────────────────────────────────────────┐
│  入口层   train.py    eval.py    play.py    plot.py           │
│           只做：解析参数 → 组装 → 循环 → 打印。不含算法逻辑。  │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  算法层   flappy/                                             │
│    config     超参数（唯一数据源）                             │
│    model      DuelingDQN + 确定性自检                         │
│    replay     ReplayBuffer                                    │
│    rollout    skip_step / FrameStack / epsilon / evaluate     │
│    agent      Agent：act / act_epsilon_greedy / learn         │
│    csvlog     CsvLogger / RunLogger                           │
│    checkpoint 存档读写                                        │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  环境层   game/                                               │
│    flappy_env    FlappyEnv：物理 / 奖励 / 得分 / 观测          │
│    resources     pygame 无头初始化、精灵、像素级碰撞           │
│    flappy_bird_utils  资源加载                                │
└──────────────────────────────────────────────────────────────┘
```

依赖只能自上而下。`game/` 不认识 torch，`flappy/` 不认识 argparse，
入口层不含任何算法常量。

---

## 2. 一次决策的完整数据流

这是整个项目最该看懂的一张图。训练主循环每转一圈就走一遍：

```
                    stack: (4,80,80) uint8
                          │
     ┌────────────────────▼─────────────────────┐
     │ eps = epsilon_at(decision_step, cfg)     │  rollout.py
     │ action = agent.act_epsilon_greedy(...)   │  agent.py
     └────────────────────┬─────────────────────┘
                          │ action ∈ {0,1}
     ┌────────────────────▼─────────────────────┐
     │ skip_step(env, action, phi, cfg)         │  rollout.py
     │   for i in range(4):                     │
     │       env.step(action, render=(i==3))    │  ← 只有最后一帧绘制
     │       r_raw += r                         │  ← 每帧奖励都累加
     │       if done: break                     │
     │   r_dec = r_raw + 0.05*(γ·Φ' − Φ)        │  ← 势能塑形在决策级
     └────────────────────┬─────────────────────┘
                          │ (next_frame, r_dec, done, info, phi')
     ┌────────────────────▼─────────────────────┐
     │ buffer.add(stack, action, r_dec,         │  replay.py
     │            next_frame, done)             │
     │   存 4 帧 state + 1 帧 next_frame         │  ← 不存两份完整帧栈
     └────────────────────┬─────────────────────┘
                          │
              每 4 次决策 ├──────────────┐
                          │              ▼
                          │   ┌──────────────────────────────┐
                          │   │ agent.learn(buffer)          │  agent.py
                          │   │  s,a,r,s1,d = buffer.sample  │
                          │   │  s1 = concat(next_frame,     │  ← 采样时拼出
                          │   │              s[:3])          │     next_state
                          │   │  a* = online(s1).argmax()    │  ← Double DQN
                          │   │  y  = r + γ·target(s1)[a*]·(~d)│
                          │   │  loss = Huber(online(s)[a], y)│
                          │   │  每 1000 梯度步 → 同步目标网络 │
                          │   └──────────────────────────────┘
                          │
     ┌────────────────────▼─────────────────────┐
     │ if done or truncated:                    │
     │     stack = stacker.reset(env.reset())   │  ← 帧栈重填，杜绝跨局
     │ else:                                    │
     │     stack = stacker.push(next_frame)     │
     └──────────────────────────────────────────┘
```

**关键恒等式**：`FrameStack.push` 和 `ReplayBuffer.sample` 里重建
`next_state` 用的必须是同一个式子 ——

```python
next_state = concat(next_frame, state[:3])      # 新帧在前，丢最旧一帧
```

否则网络训练时看到的时间顺序和游玩时相反。这条由
`test/test_env_and_buffer.py` 测试 4 保证。

---

## 3. 三个计数器

单位混用是旧管线最难查的一类 bug，所以名字里一律带单位：

| 计数器 | 谁在数 | 关系 |
|--------|--------|------|
| `frames` | `FlappyEnv.frames` | 游戏帧，环境内部 |
| `decision_step` | `train.py` 主循环 | `= frames / 4`（frame_skip） |
| `grad_steps` | `Agent.grad_steps` | `= (decision_step − warmup) / 4`（train_every_decisions） |
| `episode` | `train.py` 主循环 | 每局 +1，局长随策略变好而增长 |

配置项里凡是计数量都以 `_decisions` / `_grad_steps` / `_episodes` 结尾，
`epsilon_at()` 只吃 `decision_step`，目标网络同步只吃 `grad_steps`。
不存在任何跨单位比较。

**换算**：默认配置下 1 局 ≈ 10 次决策（随机策略）到 4000 次决策（学好之后，
到上限被截断）；1 次决策 = 4 帧；每 4 次决策做 1 个梯度步。

---

## 4. 谁被谁用

```
train.py ──┬─→ flappy.config.resolve_config
           ├─→ flappy.agent.Agent ──→ flappy.model.DuelingDQN
           │                      └─→ flappy.rollout.sample_random_action
           ├─→ flappy.replay.ReplayBuffer
           ├─→ flappy.rollout {FrameStack, epsilon_at, skip_step,
           │                   make_env, evaluate}
           ├─→ flappy.csvlog.RunLogger
           └─→ flappy.checkpoint {save_best, save_resume, save_final}

eval.py  ──┬─→ flappy.checkpoint.load_for_inference
           └─→ flappy.rollout.evaluate          ← 与训练中的评测同一份代码

play.py  ──┬─→ flappy.checkpoint.load_for_inference
           └─→ flappy.rollout {FrameStack, make_env, sample_random_action}

plot.py  ──→ 只读 CSV，不 import flappy/ 和 game/

monitor.py ─→ 只读 CSV，不 import flappy/ 和 game/
```

`plot.py` 和 `monitor.py` 刻意不 import 项目代码：它们要能在训练进行中、
甚至在另一台机器上对着拷贝过来的 `runs/` 目录跑，不该被 torch 和 pygame 的
导入副作用拖累（`game/resources.py` 一被导入就会初始化 pygame）。

---

## 5. 三条路径共用同一份动作选择

```
                  ┌─────────────────────────┐
   训练（带 ε）───▶│                         │
                  │  flappy/rollout.py      │
   评测（贪婪）───▶│    skip_step()          │──▶ game/flappy_env.py
                  │    sample_random_action │
   回放（贪婪）───▶│    FrameStack           │
                  └─────────────────────────┘
```

这是刻意的架构约束，不是省代码。旧管线的评测脚本自己复制了一份动作选择，
副本里带着"训练步数不足就返回随机动作"的短路，而加载存档从不恢复步数计数器
—— 于是所有"评测"跑的都是纯随机策略，113 个存档没有一个有可信数据。

同理，**动作选择里不允许出现按训练进度短路的分支**。warmup 期间的全随机
由 `epsilon_at()` 返回 `1.0` 实现，这样 `epsilon` 这一列永远等于实际的
随机动作比例，日志才可信。

---

## 6. 环境的契约

`FlappyEnv` 是 Gym 风格，但有两条硬约定：

1. **`step()` 绝不内部 reset。** 回合结束后必须显式 `reset()`，否则
   `step()` 抛 `RuntimeError`。旧实现在崩溃时先 `__init__()` 再绘制，
   于是 terminal 观测其实是下一局的第一帧。
2. **`step(action, render=False)` 只跑物理，不绘制不取图。** 碰撞检测走
   hitmask 和坐标，不依赖渲染结果，所以跳过是安全的（测试 7 逐帧核对）。
   帧跳过窗口靠这个提速 11 倍。

返回的 `info` 含 `score`（已过管道数）、`potential`（Φ，终止态为 0）、
`frames`、`scored`（本帧新过几根）。

**难度**只由 `pipe_gap` 一个旋钮控制（构造参数），间隙上沿的候选高度固定，
这样难度变化可以精确归因。

---

## 7. 产物与存档

```
runs/<时间戳>/
├── config.json     本次训练的完整配置（复现的唯一依据）
├── episodes.csv    每局一行  ← monitor.py / plot.py 的主数据源
├── train.csv       每 100 梯度步一行（窗口内均值）
├── eval.csv        每 500 局一行
├── run.log         纯 ASCII 文本
├── best.pt         近百局均分创新高（2% 门槛 + 100 局冷却）
├── resume_0.pt     ┐ 每 30 分钟双槽轮转
├── resume_1.pt     ┘ 含 target + optimizer，可续训
└── final.pt        收尾模型 + 30 局评测结果
```

| 存档 | model | target | optim | 计数器 | 用途 |
|------|:-----:|:------:|:-----:|:------:|------|
| `best.pt` | ✓ | | | ✓ | 评测 / 回放 |
| `resume_N.pt` | ✓ | ✓ | ✓ | ✓ | 续训 |
| `final.pt` | ✓ | | | ✓ | 评测 / 回放 |

三种存档都带 `config`。这不是冗余：`pipe_gap` 若不随存档记录，
拿今天的默认值去评测昨天的模型，分数就没有可比性 ——
`eval.py` 因此会自动采用**训练时的难度**，除非显式 `--pipe-gap` 覆盖。

双槽轮转是为了防"存档写到一半进程被杀"：任何时刻至少有一个完整的 resume 槽。

---

## 8. 实时监控

训练是长跑，`run.log` 每 500 局才一行。`monitor.py` 直接读 `episodes.csv`
和 `train.csv`，每隔几秒刷新一屏：

```bash
python monitor.py                    # 自动盯最新的 runs/ 目录
python monitor.py runs/<时间戳>       # 指定
python monitor.py --interval 10      # 刷新间隔（秒）
python monitor.py --once             # 打印一次就退出（适合脚本）
```

它不 import 项目代码，所以训练在 Docker 里跑、监控在宿主机跑也没问题。
要看曲线随时 `python plot.py runs/<时间戳>`，训练进行中也能画。

**该盯什么**（详见 README"怎么读诊断图"）：

| 指标 | 健康的样子 | 出问题的样子 |
|------|-----------|-------------|
| `recent100_pipes` | 缓慢上升，越过随机基线 0.63 | 长期贴在 0.63 |
| `q_mean` | 跟着 `target_q_mean` 上升 | 贴在 0 不动 / 发散到几百 |
| `loss` | 每次同步后锯齿 | 平坦且极小（拟合了不动的错目标） |
| `syncs` | 稳定增长 | 长期为 0（梯度步没跑起来） |
| `grad_per_s` | 稳定 | 掉到 0（缓冲区没填够或卡住） |
| `eval_pipes_mean` | 与训练曲线同向 | 显著低于训练曲线（贪婪策略有问题） |

---

## 9. 加东西该往哪放

| 想做的事 | 改哪 |
|----------|------|
| 调超参数 | `flappy/config.py`，别在入口脚本写死 |
| 换网络结构 | `flappy/model.py`（**不许加 BatchNorm / Dropout**） |
| 加 PER / n-step | `flappy/replay.py` + `flappy/agent.py:learn` |
| 改奖励或物理 | `game/flappy_env.py`，并补一个单测 |
| 加新的探索策略 | `flappy/rollout.py`，三条路径自动共享 |
| 加日志字段 | `flappy/csvlog.py` 的 `*_FIELDS` + 写入处 |
| 新的分析脚本 | 根目录，只读 CSV，不 import 项目代码 |

改完必跑：

```bash
python test/test_env_and_buffer.py
python train.py --smoke --allow-cpu
```
