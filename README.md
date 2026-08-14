# Flappy Bird — Dueling Double-DQN

用像素画面训练一个会玩 Flappy Bird 的智能体。输入是 4 帧 80×80 的二值画面，
输出是两个动作（扇翅 / 不动）的 Q 值，没有任何手工特征。

**这个仓库的主要目的是学 RL 和神经网络** —— 代码里每个设计选择都写了理由，
配套 [12 篇教学文档](docs/learn/README.md) 和 11 个"每个都对应一个真实 bug"的单测。

实测（RTX 3070 Laptop，训练约 70 分钟到 ε=0.01，100 局纯贪婪评测）：

| 评测环境 | 过管道数 |
|---|---|
| 随机化布局（训练分布，缝隙 80–165px 逐根变） | **56.8**（中位 39，最高 286）|
| 固定布局 gap 100（**从没训练过**） | **239.6**（29% 跑满 3000 决策上限）|
| 人类玩家（同一随机化环境，20 局） | 6.8 |
| 随机策略 | 0.63 |

作为对照，重写前的管线跑了 10 万局 / 13 小时，停在 **1.3 根** ——
差距不在超参数，在[十来个结构性错误](docs/learn/00-why-it-failed.md)。

而只在固定布局上训练的模型，虽然在自己的分布上能过 275 根，
换到随机布局只剩 2.4 根 ——
见[两模型 × 五环境的完整泛化矩阵](docs/learn/11-generalization.md)。

```
                 ┌──────────────┐   80×80 二值帧    ┌──────────────┐
   4 帧一动 ───▶ │  FlappyEnv   │ ─────────────────▶│  DuelingDQN  │
                 │  (pygame)    │◀───── 动作 ───────│  1.26M 参数  │
                 └──────────────┘                   └──────────────┘
                        │                                   ▲
                        │ (s, a, r, s', done)               │ Double-DQN
                        ▼                                   │ + Huber
                 ┌──────────────┐    均匀采样 128    ┌───────────────┐
                 │ ReplayBuffer │ ─────────────────▶│    目标网络   │
                 │ uint8 · 40 万│                   │ 每 1000 梯度步│
                 └──────────────┘                   └───────────────┘
```

---

## 快速开始

### Windows：一条命令装好

```powershell
.\setup.ps1
```

它会自动检测 GPU、建 `.venv`、装对版本的 PyTorch、装依赖，最后跑 11 个单测确认能用。

国内网络建议 `.\setup.ps1 -Mirror`（阿里云 + 清华镜像），省得先失败一轮。
只想跑单测/看回放/画图用 `.\setup.ps1 -Cpu`。

> 脚本处理了装 PyTorch 的三个坑，都是实际踩过的：
> **① PyPI 上 Windows 版 `torch` 是 CPU 构建**（只有 116MB，装完
> `cuda.is_available()` 是 False），CUDA 版必须走 pytorch.org 的独立索引；
> **② 官方索引在国内经常断**，而 pip 的重试会丢弃已下载字节从头再来，
> 所以脚本改用阿里云镜像 + `curl -C -` 真断点续传；
> **③ 磁盘空间** —— 2.6GB 的 wheel 加解压要约 6GB 临时空间，
> 盘满时报的错是 `connection interrupted`，看起来像网络问题。

### 其他平台 / 手动安装

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU
```

### 然后

```bash
# 先确认管线是好的（11 个单测，约 30 秒）
python test/test_env_and_buffer.py

# 3. 训练（需要 CUDA；资源需求见下方"硬件与资源"）
python train.py                     # 完整训练，产物落在 runs/<时间戳>/
python train.py --smoke             # 几分钟的管线自检
python train.py --max-hours 12      # 到点干净收尾（跑完收尾评测再退出）

# 4. 训练进行中：另开一个终端实时监控
python monitor.py                   # 自动盯最新的 runs/ 目录，每 5 秒刷新

# 5. 评测 / 观看 / 画图
python eval.py runs/<时间戳>/best.pt --episodes 100
python play.py runs/<时间戳>/best.pt --fps 30 --scale 2
python plot.py runs/<时间戳>
```

> 内存不够就调小缓冲区：默认 `buffer_capacity=400000` 需要约 12.8GB 常驻内存，
> 16GB 的机器上要 `python train.py --buffer 120000`（约 3.8GB）。

自己上手玩一局（不需要存档，按住空格扇翅）：

```bash
python play.py --human
```

---

## 项目结构

```
train.py            训练入口
eval.py             评测入口：加载存档跑 N 局贪婪策略
play.py             可视化入口：看 AI 玩 / 录像 / --human 自己玩
plot.py             从 runs/*.csv 画 6 联诊断图
monitor.py          训练实时监控（只读 CSV，不 import 项目代码）

flappy/             算法侧
  config.py           超参数（唯一数据源，计数单位写在名字里）
  model.py            DuelingDQN：Nature 卷积栈 + 双头，无 BN 无 Dropout
  replay.py           经验回放：uint8 存储，均匀采样
  rollout.py          帧跳过、帧栈、探索策略、贪婪评测
  agent.py            Double-DQN 学习步与目标网络同步
  csvlog.py           结构化 CSV 日志
  checkpoint.py       存档读写

game/               环境侧
  resources.py        pygame 无头初始化、精灵、像素级碰撞检测
  flappy_env.py       FlappyEnv：物理、奖励、得分、观测
  flappy_bird_utils.py  资源加载

test/
  test_env_and_buffer.py  10 个单测，每个对应一个真实存在过的 bug

docs/learn/           教学文档：11 篇，从 RL 基础讲到诊断
docs/ARCHITECTURE.md  架构说明：数据流、模块依赖、计数器、加东西该往哪放
docs/legacy/          旧管线的文档存档（内容已过时，见文末）
```

- **想学 RL 和神经网络** → [docs/learn/](docs/learn/README.md)（11 篇教学文档）
- **想看代码怎么组织** → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

下面这一节讲的是"为什么这么选"。

训练、评测、可视化三条路径共用 `flappy/rollout.py` 里的同一份动作选择和帧跳过代码。
这一点是刻意的：旧管线的评测脚本自己复制了一份动作选择逻辑，那份副本里带着
"训练步数不足就返回随机动作"的短路，而加载存档从不恢复步数计数器 ——
于是历史上所有"评测"跑的都是纯随机策略，113 个存档没有一个有可信数据。

---

## 关键设计

### 1. 一次决策 = 一条经验

`frame_skip=4`：动作在 4 帧窗口内重复，窗口内每一帧的奖励都累加，
提前撞死则 break。整条代码里不存在任何 `% frame_skip`。

旧实现把 `step += 1` 夹在两个 `% k` 判断之间，动作在 `step≡0` 时选、
经验在 `step≡3` 时存，两者永不同时成立 —— 存进缓冲区的是
`(s_{t+3}, a_t, r_{t+3}, s_{t+4})`，状态/动作/奖励三者全错位，
还丢弃了 75% 的奖励与 terminal 标记。（单测 5）

窗口内只渲染最后一帧：物理只要 5.8µs，而绘制+取图要 970µs。
这一条让端到端吞吐从 204 决策/秒涨到 2328 决策/秒。碰撞检测走 hitmask 和坐标，
不依赖渲染结果，所以跳过渲染是安全的。（单测 7 逐帧核对了这一点）

### 2. 网络里没有 BatchNorm 和 Dropout

不是为了简化，是正确性要求：目标网络若处于 train 模式，BatchNorm 会用
当前 minibatch 的统计量算 TD 目标，Bellman 算子随批次组成而变 ——
一个不固定的算子没有不动点，训练不可能收敛。Dropout 则让"贪婪"动作变成随机动作。

两者删掉之后 `train()` / `eval()` 成为空操作，再也不会因为漏调用 `eval()` 而出错。
训练、评测、回放三条路径启动时都会跑一次确定性自检
（`flappy/model.py: assert_deterministic`）。（单测 6）

### 3. 目标网络按梯度步同步

`target_sync_grad_steps=1000` 数的是**梯度步**，不是环境步。
旧实现按环境步每 125,000 步才同步一次，12.5 小时只做了 11 次 Bellman 回传，
而管道从生成到得分需要 12.5 次决策 —— 价值根本传不回去。

### 4. 奖励归一化到 ±1

| 事件 | 奖励 |
|------|------|
| 过一根管道 | `+1.0` |
| 撞死 | `-1.0` |
| 存活每帧 | `0.0` |
| 势能塑形 | `0.05 × (γΦ(s') − Φ(s))`，Φ ∈ [−1, 0] |

存活奖励保持 0：任何非势能的存活项都会诱导"原地磨时间"。
势能塑形在**决策级**施加 —— 若按帧施加，skip 窗口内的求和不等于
`γ_dec·Φ(s_{t+k}) − Φ(s_t)`，Ng-Harada-Russell 的策略不变性就不再精确成立。
（单测 1 用望远镜求和验证，误差 7.7e-16）

旧实现用 `+20 / −2`，让 Huber 的 δ=1 拐点和梯度裁剪阈值都落在了错误的位置；
而且奖励用赋值而非累加，同一帧既过管道又撞死时 `+20` 会被 `−2` 整个覆盖。（单测 2）

### 5. 随机动作的扇翅率是 0.2，不是 0.5

一次扇翅令 `velY=-5`，重力 `+0.5/帧`，单次扇翅后净位移在约 19 帧（约 5 次决策）归零 ——
"悬停"对应的扇翅率就是 1/5。均匀随机会把鸟顶死在天花板上，
这正是旧 warmup 数据毫无价值的原因。只改行为策略，off-policy 的 Q 学习不受影响。

随机基线（p_flap=0.2）能过约 0.63 根管道，这是 `plot.py` 第 1 张图里的灰色虚线。

### 6. 帧栈跨局不拼接

每局结束后帧栈用首帧重新填满（`FrameStack.reset`）。旧代码在 terminal 那一轮
也执行无条件的 `s_t = s_t1`，于是新回合的前 3 次决策都拿着混有上一局画面的帧栈，
并以 `done=False` 存入 —— 约 23% 的经验是物理不可能的状态。（单测 4）

### 7. 截断不是终止

回合长度上限 `max_episode_decisions=4000`。到达上限时 `done` 保持 `False`，
价值继续自举 —— 否则等于告诉网络"飞太久会凭空死掉"。
上限本身是必需的：策略学好之后回合可以无限长，没有上限日志/存档/评测会全部卡住。

---

## 配置速查

改 `flappy/config.py`，或用命令行覆盖常用项。

| 项 | 默认 | 说明 |
|----|------|------|
| `randomize_pipes` | True | 域随机化：缝隙大小/位置/间距**逐根**采样，同一局内地图也在变。关掉会让网络只学到一套窄策略（[实测差 104 倍](docs/learn/11-generalization.md)）|
| `pipe_gap_range` | (80, 165) | 随机化时的缝隙大小 |
| `pipe_spacing_range` | (115, 200) | 随机化时的水平间距（原来恒为 144）|
| `pipe_max_delta_frac` | 0.6 | 相邻缝隙落差上限 / 间距，保证物理可达 |
| `pipe_gap` | 100 | `randomize_pipes=False` 时的固定缝隙。100 = 原版，150 = 放宽 |
| `frame_skip` / `frame_stack` | 4 / 4 | 一次决策 4 帧，状态是最近 4 次决策的末帧 |
| `gamma` | 0.99 | 决策级折扣 |
| `batch_size` | 128 | 小网络下 GPU 被 kernel 启动延迟主导，128 与 32 的每步耗时几乎一样 —— batch 是白拿的 |
| `lr` | 1.5e-4 | Adam，无 weight_decay、无 LR 衰减 |
| `adam_eps` | 1.5e-4 | Rainbow 取值。1e-8 会让 Adam 退化成 `lr*sign(g)`，分不清"+1 管道"和数值噪声 |
| `warmup_decisions` | 20,000 | 纯随机填缓冲区，期间不学习 |
| `train_every_decisions` | 4 | 配 batch 128 = 每次决策 32 个样本的学习量（标准回放比） |
| `target_sync_grad_steps` | 1,000 | 数梯度步 |
| `buffer_capacity` | 400,000 | uint8 存储约 12.8GB 常驻内存 |
| `eps` | 1.0 → 0.05 → 0.01 | 两段线性退火，只依赖 `decision_step` 一个计数器 |

命令行：`--pipe-gap` `--no-randomize` `--buffer` `--episodes` `--seed` `--max-hours`
`--run-dir` `--smoke` `--allow-cpu`

> **域随机化是本项目最重要的一个开关。** 关掉它训练出来的模型，
> 在训练分布上能过 218 根管道，换个随机地图只剩 2.1 根 —— 相差 104 倍。
> 它学的不是"怎么玩这个游戏"，而是"怎么应付那个 70px 的窄带"。
> 完整实验见 [11 · 泛化与域随机化](docs/learn/11-generalization.md)。

---

## 训练产物

每次训练在 `runs/<时间戳>/` 下生成：

| 文件 | 内容 |
|------|------|
| `config.json` | 本次训练的完整配置 |
| `episodes.csv` | 每局：过管道数、回报、局长、epsilon、近百局均值 |
| `train.csv` | 每 100 梯度步：loss、Q 分布、TD 误差、梯度范数、吞吐 |
| `eval.csv` | 每 500 局一次贪婪评测 |
| `run.log` | 纯 ASCII 文本日志 |
| `best.pt` | 近百局均分创新高时保存（带 2% 提升门槛 + 100 局冷却） |
| `resume_0.pt` / `resume_1.pt` | 每 30 分钟双槽轮转，含目标网络和优化器状态 |
| `final.pt` | 收尾模型 + 30 局评测结果 |

存档一律带 `config`，所以 `eval.py` 会自动用**训练时的难度**来评测；
要跨难度比较就显式加 `--pipe-gap`。

### 实时监控

训练是长跑，`run.log` 每 500 局才写一行。另开一个终端：

```bash
python monitor.py                    # 自动盯最新的 runs/ 目录
python monitor.py runs/<时间戳>       # 指定某次训练
python monitor.py --interval 10      # 刷新间隔（秒），默认 5
python monitor.py --once             # 打印一次就退出（适合脚本）
```

一屏显示：当前阶段与进度条、近百局均分（对比随机基线 0.63）、
epsilon、缓冲区占用、loss / Q / TD 误差 / 梯度范数、目标网络同步次数、
吞吐、最近一次贪婪评测、两条走势方块图、预计剩余时间。

`monitor.py` 只读 CSV，不 import `flappy/` 和 `game/` —— 所以它能对着从别的机器
拷回来的 `runs/` 目录跑，也不会因为导入 pygame 而白白初始化一遍 SDL。
Ctrl-C 退出监控不影响训练。

### 怎么读诊断图

`python plot.py runs/<时间戳>` 出 6 张图，其中第 2、3 张是关键：

- **图 2（Q 值 + 目标网络同步竖线）**：`q_mean` 应随训练缓慢上升并跟着
  `target_q_mean` 走。低分段有个经验规律 `Q ≈ 0.9 × 已过管道数`；
  但 Q 有饱和上限 `γ^12.5 / (1 − γ^7.2) ≈ 12.6`，一个不死的策略 Q 会趋向 12.6
  而不是趋向管道数。两条曲线贴在 0 附近不动 = 价值没在传播。
- **图 3（对数轴 loss）**：每次目标网络同步后应看到锯齿。
  loss 平坦且极小 **不是** 好事 —— 它意味着网络已经完美拟合了一个不动的错误目标。
- **图 1（过管道数）**：主指标，与奖励尺度无关，跨版本可比。
  灰色虚线是随机基线 0.63。
- **图 6（贪婪评测 vs 行为策略）**：两条线应大体同向；
  贪婪评测显著低于带 ε 的训练曲线，说明贪婪策略本身有问题。

---

## 测试

```bash
python test/test_env_and_buffer.py       # 独立运行，带逐条说明
pytest test/test_env_and_buffer.py -v    # 或用 pytest
```

10 个测试，每个都对应一个在旧管线里真实存在过的缺陷：
奖励累加与望远镜求和、同帧得分+撞死、terminal 观测是真实崩溃帧、
帧栈对齐无跨局拼接、帧跳过的动作/奖励对齐、网络确定性、
跳过渲染不改变物理、观测路径的像素级等价、难度旋钮生效。

---

## 硬件与资源

`train.py` 在没有 CUDA 时**直接报错退出**，不静默回退到 CPU ——
旧代码就是这样把"训练慢了 20 倍"藏了很久。要在 CPU 上跑得显式加 `--allow-cpu`。

| 资源 | 需求 |
|---|---|
| 显存 | 约 800 MB（网络才 5 MB，其余是 CUDA context）|
| **内存** | 默认 `buffer_capacity=400000` 要 **11.9 GB 常驻**。16 GB 的机器用 `--buffer 100000`（约 3.2 GB）|
| 磁盘 | 安装约 6 GB；每轮训练的 `runs/` 目录约 50 MB |
| 实测速度 | RTX 3070 Laptop：约 220 决策/秒、55 梯度步/秒，GPU 利用率 30% |

GPU 利用率只有 30% 是正常的 —— 网络太小，瓶颈在 CPU 侧的 pygame 渲染，
不在显卡。这也是 `batch_size=128` 比 32 几乎不多花时间的原因（见
[04 · 神经网络详解](docs/learn/04-network.md) 第 8 节）。

> 本项目**不提供 Docker 环境**。之前有一份 Dockerfile，但从未被实际运行过
> （所有训练都是 Windows 原生 venv 跑的），留着一个没验证过的构建文件
> 比没有更糟。要容器化的话，装依赖的部分照 `setup.ps1` 的逻辑翻译即可。

---

## docs/legacy/

`docs/legacy/` 是旧管线的文档存档。**内容已经过时**：它们描述的是
`deep_Q_oneStep.py` / `deep_Q_dueling_DQN.py` / `continue_training.py`
那套实现，以及"智能目标网络选择""预训练模型继续训练"等已废弃的机制。
那套管线跑了 4 轮、最长 10 万局 / 13 小时，近百局均分始终停在约 1.3 根管道。

保留它们是因为其中的推导和调试记录仍有参考价值，但**不要**照着它们改代码。
旧代码本身已删除，可在 git 历史中查阅（`git log --diff-filter=D --name-only`）。
