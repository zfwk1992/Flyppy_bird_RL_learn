# 08 · 训练流程

前面几篇讲的是各个零件，这一篇把它们拼起来：一次训练从启动到收尾，
数据怎么流、什么时候学、什么时候存。

代码：[`train.py`](../../train.py)

---

## 1. 主循环

去掉日志之后，训练主循环就这么点东西：

```python
while episode < cfg['max_episodes']:
    # ① 选动作
    eps = epsilon_at(decision_step, cfg)
    action = agent.act_epsilon_greedy(stack, eps)

    # ② 与环境交互（4 帧一动，奖励累加）
    next_frame, r_dec, done, info, phi = skip_step(env, action, phi, cfg)

    # ③ 存经验
    buffer.add(stack, action, r_dec, next_frame, done)
    decision_step += 1

    # ④ 学习（warmup 之后，每 4 次决策一个梯度步）
    if (decision_step > cfg['warmup_decisions']
            and len(buffer) >= cfg['batch_size']
            and decision_step % cfg['train_every_decisions'] == 0):
        stats = agent.learn(buffer)

    # ⑤ 回合结束处理
    truncated = (not done) and ep_dec >= cfg['max_episode_decisions']
    if done or truncated:
        episode += 1
        # 记日志 / 存 best / 定期评测 / 存 resume
        stack = stacker.reset(env.reset())     # 重填帧栈，杜绝跨局
        phi = env.current_potential()
    else:
        stack = stacker.push(next_frame)
```

**注意主循环里没有任何 `% frame_skip`。** 帧跳过完全封装在 `skip_step()` 内部，
所以不可能出现"选动作用一个计数器、存经验用另一个"的错位 ——
那正是旧管线最致命的 bug。

---

## 2. 三个计数器的换算

| 计数器 | 谁在数 | 关系 |
|--------|--------|------|
| `frames` | `FlappyEnv.frames` | 游戏帧 |
| `decision_step` | 主循环 | `= frames / 4`（`frame_skip`） |
| `grad_steps` | `Agent.grad_steps` | `= (decision_step − warmup) / 4`（`train_every_decisions`） |
| `episode` | 主循环 | 每局 +1 |

典型一轮训练（默认配置）：

```
20,000 决策 warmup（不学习）
      ↓
每 4 次决策 → 1 个梯度步
每 1000 个梯度步 → 1 次目标网络同步
每 100 个梯度步 → train.csv 写一行
每 500 局 → 一次贪婪评测
每 30 分钟 → 一次 resume 存档
```

### `train_every_decisions = 4` 是怎么定的

这叫**回放比**（replay ratio）：每收集 1 条新经验，做多少次梯度更新。

```
batch_size / train_every_decisions = 128 / 4 = 32 样本/决策
```

即每次决策，网络平均"看" 32 个样本。这是 DQN 的标准区间。

同时它把梯度步的需求压到 GPU 能力之内：
RTX 3070 上梯度步速率约 200 步/秒，环境约 800 决策/秒，
`800 / 4 = 200` —— 刚好匹配，两边都不闲着。

> 回放比太高（比如每次决策做 1 个梯度步）→ 网络在同一批旧数据上
> 反复训练，过拟合到缓冲区；太低 → 数据收集了却没充分利用。

---

## 3. 阶段划分

```
决策步    0 ─────── 20k ──────────── 270k ──────────── 770k ────→
         │  warmup  │    anneal 1    │    anneal 2     │ exploit
ε        │   1.0    │  1.0 → 0.05    │  0.05 → 0.01    │  0.01
学习      │    ✗     │       ✓        │        ✓        │    ✓
```

`monitor.py` 会直接显示当前阶段和进度条。

实测（RTX 3070 Laptop，`--buffer 100000`，`pipe_gap=100`）。
下表用的是 `recent100_pipes`（100 局平滑），比 20 局评测可信得多：

| 局数 | 累计用时 | 近百局均分 | ε |
|---|---|---|---|
| 1,000 | 7 s | 0.38 | 1.00 |
| 5,000 | 3 min | 0.42 | 0.88 |
| 10,000 | 9 min | 0.89 | 0.58 |
| 15,000 | 20 min | 4.19 | 0.15 |
| 17,000 | 41 min | 22.37 | 0.03 |
| 19,000 | 95 min | **64.19** | 0.01 |

训练结束后对 `best.pt` 做 **100 局纯贪婪评测**：

```
pipes mean    : 389.600 +- 177.028
pipes median  : 521.0    max: 541
hit the cap   : 46 / 100 episodes (still alive when stopped)
```

对照：旧管线 10 万局 / 13 小时后停在 1.3 根。

> **注意这两组数字为什么差这么多**（64 vs 390）：
> 上表是**带 ε=0.01 的行为策略**成绩，下面是**纯贪婪**。
> 学好之后一局约 2900 次决策，1% 随机率 = 每局约 29 次随机动作，
> 每一次都可能致命。详见 [07 · 探索策略](07-exploration.md) 的实测对照。

---

## 4. 回合结束：终止 vs 截断

```python
truncated = (not done) and ep_dec >= cfg['max_episode_decisions']
if done or truncated:
    ...
```

两种结束方式，处理**必须不同**：

| | `done`（撞死） | `truncated`（到长度上限） |
|---|---|---|
| 存经验时的 done 标记 | `True` | `False` |
| TD 目标 | `y = r` | `y = r + γ·Q(s')` |
| 含义 | 真的没有未来了 | 只是我们不看了，价值继续自举 |

**如果把截断当终止，等于告诉网络"飞太久会凭空死掉"** ——
一个学好的策略会被自己的成功惩罚。

上限本身是必需的：策略学好之后回合可以无限长
（实测已出现 4000 次决策仍不死的局），
没有上限的话日志、存档、评测会全部卡住。

注意代码里 `buffer.add(...)` 用的是 `done` 而不是 `truncated`，
截断的那条经验以 `done=False` 入库 —— 这正是我们要的。

---

## 5. 三种存档

```
runs/<时间戳>/
├── best.pt         近百局均分创新高时
├── resume_0.pt     ┐ 每 30 分钟双槽轮转
├── resume_1.pt     ┘
└── final.pt        收尾
```

| 存档 | model | target | optim | 计数器 | 用途 |
|------|:-----:|:------:|:-----:|:------:|------|
| `best.pt` | ✓ | | | ✓ | 评测 / 回放 |
| `resume_N.pt` | ✓ | ✓ | ✓ | ✓ | 续训 |
| `final.pt` | ✓ | | | ✓ | 评测 / 回放 |

### best 的门控

```python
if (episode >= 500 and len(recent_pipes) == 100
        and r100 > best_recent100 * 1.02
        and episode - last_best_episode >= 100):
```

四个条件缺一不可：

1. `episode >= 500` —— 早期噪声太大
2. `len(recent_pipes) == 100` —— 窗口填满才有意义
3. `r100 > best * 1.02` —— **2% 提升门槛**，避免噪声触发
4. `episode - last_best >= 100` —— **冷却**，避免连续存盘

门控放在**平滑指标**（近百局均分）上，不是单局分数。

> 旧代码把 `max_score` 放在每局都执行的分支里重新赋值，
> 2 万局之后阈值恒为 0，于是每个正分局都算"新纪录" ——
> 触发了 **31182 次 58MB 存盘**，却从未保存过任何真正意义上的最佳模型。

### 双槽轮转

```python
checkpoint.save_resume(os.path.join(run_dir, 'resume_%d.pt' % resume_slot), ...)
resume_slot ^= 1
```

交替写 `resume_0` / `resume_1`。这样即使进程在写存档的一瞬间被杀，
**至少还有一个完整的槽**。单槽的话，写到一半崩溃就全没了。

### 所有存档都带 config

```python
torch.save({'model': ..., 'config': cfg}, path)
```

不是冗余：`pipe_gap` 这种难度旋钮如果不随存档记录，
拿今天的默认值去评测昨天的模型，分数就没有可比性。
`eval.py` / `play.py` 会自动采用存档里的配置。

---

## 6. 评测用独立的环境实例

```python
env = make_env(cfg)
eval_env = make_env(cfg)     # 评测专用，绝不能共用
```

如果评测和训练共用一个 env 实例，评测会把训练回合的状态冲掉 ——
训练局跑到一半，突然被 reset 到另一局。

评测本身走的是和 `eval.py` **完全同一份代码**
（`flappy/rollout.py: evaluate()`），不存在任何按训练进度短路的分支。

---

## 7. 日志

三个 CSV，各记各的：

```python
EPISODE_FIELDS = ['episode', 'decision_step', 'frames', 'pipes', 'ep_return',
                  'ep_len_decisions', 'epsilon', 'recent100_pipes',
                  'recent100_return', 'buffer_size', 'wall_s', 'terminated']
TRAIN_FIELDS   = ['grad_step', 'decision_step', 'loss', 'td_abs_mean', 'td_abs_max',
                  'q_mean', 'q_std', 'q_min', 'q_max', 'target_q_mean', 'grad_norm',
                  'lr', 'epsilon', 'buffer_size', 'syncs', 'dec_per_s', 'grad_per_s']
EVAL_FIELDS    = ['episode', 'decision_step', 'n_eval_eps', 'eval_pipes_mean',
                  'eval_pipes_std', 'eval_pipes_max', 'eval_len_mean']
```

`train.csv` 里的统计量是**窗口内均值**（每 100 个梯度步聚合一次），
不是瞬时值 —— 单步的 loss 噪声太大，看不出趋势。

> ### 结构化日志不是锦上添花
>
> 旧管线只往 stdout 打一行带 emoji 的文本，画图脚本用正则去抓，
> 只能拿到 `(episode, score, max, avg100)` 四个量：
> 看不到 loss、看不到 Q 值、看不到目标网络同步时刻。
>
> **13 小时里所有问题都没被发现，直接原因就是这个。**
> "loss 只有 0.002 而一根管道值 +20"这个铁证，
> 只要有 loss 那一列就能一眼看出来。

日志还有一条纪律：**热循环里不 print，尤其不 print emoji**。
Windows cp1252 的 stdout 遇到 emoji 会抛 `UnicodeEncodeError`
**直接终止训练**。所有文本日志走 `RunLogger.say()`，纯 ASCII。

---

## 8. 常用命令

```bash
# 完整训练
python train.py

# 内存小的机器（默认 buffer 要 11.9GB）
python train.py --buffer 100000        # 约 3.2GB

# 几分钟的管线自检
python train.py --smoke

# 到点干净收尾（跑完收尾评测再退出，不是硬杀）
python train.py --max-hours 12

# 放宽难度先学会飞
python train.py --pipe-gap 150

# 另开终端实时监控
python monitor.py
```

### 没有 CUDA 会直接报错

```python
raise SystemExit("CUDA not available (torch=%s). The old code silently fell back
                  to CPU and hid this. ... or pass --allow-cpu" % torch.__version__)
```

旧代码 `device = "cuda" if available else "cpu"` 静默回退，
于是"机器上有显卡却在用 CPU 训练、慢了 20 倍"这件事很久没被发现。

现在要用 CPU 必须显式 `--allow-cpu`。

---

## 9. 一次训练的完整时间线

```
[启动]
  ├─ 检查 CUDA（没有就退出）
  ├─ 设随机种子
  ├─ 建 runs/<时间戳>/，写 config.json
  ├─ 建 Agent（含确定性自检 assert_deterministic）
  ├─ 分配 ReplayBuffer（一次性 np.zeros）
  └─ 建 env 和 eval_env

[主循环]
  ├─ 0 – 20k 决策：warmup，ε=1.0，只收集不学习
  ├─ 20k+：每 4 次决策一个梯度步
  │    ├─ 每 1000 梯度步：同步目标网络
  │    └─ 每 100 梯度步：写 train.csv
  ├─ 每局结束：写 episodes.csv，判断是否存 best
  ├─ 每 500 局：贪婪评测，写 eval.csv
  └─ 每 30 分钟：写 resume_N.pt

[收尾]（正常结束或 --max-hours 到点）
  ├─ 跑 30 局贪婪评测
  ├─ 存 final.pt
  └─ 关闭所有 CSV
```

Ctrl-C 中断不会走收尾流程，但 `best.pt` 和 `resume_N.pt` 已经在磁盘上了，
不会丢失进度。

---

## 小结

| 环节 | 频率 | 配置项 |
|---|---|---|
| 决策 | 每 4 帧 | `frame_skip` |
| 梯度步 | 每 4 次决策 | `train_every_decisions` |
| 目标同步 | 每 1000 梯度步 | `target_sync_grad_steps` |
| train.csv | 每 100 梯度步 | `log_train_every_grad_steps` |
| 贪婪评测 | 每 500 局 | `eval_every_episodes` |
| resume 存档 | 每 30 分钟 | `resume_every_minutes` |
| best 存档 | 均分创新高（2% 门槛 + 100 局冷却） | 硬编码 |

---

上一篇：[07 · 探索策略](07-exploration.md)
下一篇：[09 · 推理过程](09-inference.md)
