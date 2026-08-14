# 04 · 神经网络详解

网络定义在 [`flappy/model.py`](../../flappy/model.py)，一共 40 行。
本文逐层拆开讲：形状怎么算、参数量在哪、为什么这么设计。

---

## 1. 完整结构与逐层维度

下面的数字是用真实代码打出来的，不是手算的：

| 层 | 输出形状 | 参数量 | 占比 |
|---|---|---|---|
| input | `(N, 4, 80, 80)` | 0 | |
| conv1 | `(N, 32, 19, 19)` | 8,224 | 0.7% |
| conv2 | `(N, 64, 8, 8)` | 32,832 | 2.6% |
| conv3 | `(N, 64, 6, 6)` | 36,928 | 2.9% |
| flatten | `(N, 2304)` | 0 | |
| fc | `(N, 512)` | 1,180,160 | **93.7%** |
| value | `(N, 1)` | 513 | 0.04% |
| advantage | `(N, 2)` | 1,026 | 0.08% |

**总计 1,259,683 参数 = 5.04 MB (fp32)**

```python
class DuelingDQN(nn.Module):
    def __init__(self, n_actions=2, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 6 * 6, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)
```

这是 DeepMind 2015 年 Nature 那篇 DQN 的卷积栈（"Nature 栈"），
只把最后换成了 Dueling 双头。用一个久经考验的骨干，
省得在"网络结构"这个维度上浪费调试预算。

---

## 2. 卷积输出尺寸怎么算

公式：

```
out = floor( (in + 2·padding − kernel) / stride ) + 1
```

本项目**三个卷积层都没有 padding**，逐层代入：

```
conv1:  (80 − 8) / 4 + 1 = 72/4 + 1 = 18 + 1 = 19
conv2:  (19 − 4) / 2 + 1 = 15/2 + 1 =  7 + 1 =  8      ← 注意向下取整
conv3:  (8  − 3) / 1 + 1 =  5/1 + 1 =  5 + 1 =  6
```

所以特征图是 `80 → 19 → 8 → 6`，展平后 `64 × 6 × 6 = 2304`。

> **conv2 那一步有个坑**：`15/2 = 7.5` 向下取整成 7，
> 意味着最右/最下那一列像素被丢弃了。这在 Nature 栈里是既定行为，
> 无伤大雅 —— 但如果你改了输入尺寸或 stride，一定要重新算，
> 不然 `nn.Linear` 的输入维度对不上，会在第一次前向时报形状错误。

### 一个实用建议

不要手算完就写死。改结构时用这段代码验证：

```python
import torch
from flappy.model import DuelingDQN
net = DuelingDQN()
print(net(torch.zeros(1, 4, 80, 80, dtype=torch.uint8)).shape)   # 应为 (1, 2)
```

单测 6 也会检查网络能正常前向且参数量符合预期。

---

## 3. 参数量为什么集中在 fc

93.7% 的参数在一个 `Linear(2304, 512)` 上。这是卷积网络的典型形态：
卷积层靠**权值共享**，参数极少；全连接层参数量 = 输入维 × 输出维，
一下就上去了。

### 这里有个真实的教训

旧网络是 **3.62M 参数 / 14.5MB**，其中 90.5% 集中在一个
`Linear(6400, 512)` 上 —— 因为它的 conv3 用了 `stride=1, padding=1`，
特征图停在 `10×10`：

```
64 × 10 × 10 = 6400   →   Linear(6400, 512) = 3.28M 参数
```

而本项目不加 padding，特征图缩到 `6×6`：

```
64 × 6 × 6 = 2304     →   Linear(2304, 512) = 1.18M 参数
```

**同样的表达能力，参数量少了 65%。** 一个 padding 参数的差别而已。

更大的网络在这里没有任何好处：Flappy Bird 的状态空间结构很简单
（小鸟 y 坐标、速度、下一根管道的位置），1.26M 参数绰绰有余。
参数多了只是让每个梯度步更慢、更容易过拟合到回放缓冲区里的旧数据。

---

## 4. 感受野：网络"看得见"什么

感受野是输出特征图上一个点对应原图上多大的区域。逐层往回推：

```
conv3 的一个点 ← 3×3 (conv3)
              ← 3×3 经 conv2(k=4,s=2) 展开为 (3−1)·2 + 4 = 8
              ← 8×8 经 conv1(k=8,s=4) 展开为 (8−1)·4 + 8 = 36
```

所以 conv3 每个点的感受野是原图 **36×36 像素**，而原图是 80×80。
`6×6` 的特征图铺满整张图，且相邻点的感受野大幅重叠。

含义：网络在最后一层能同时看到小鸟和它附近的管道，
足以判断"我相对缝隙偏高还是偏低"。这正是这个任务需要的信息。

---

## 5. 前向传播

```python
def forward(self, x):
    # x: (N,4,80,80) uint8 或 float
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc(x.flatten(1)))
    v = self.value(x)
    a = self.advantage(x)
    return v + a - a.mean(dim=1, keepdim=True)
```

### 归一化放在 GPU 上做

输入是 uint8，`div_(255.0)` 在网络内部执行。这样：

- 经验回放用 uint8 存储，**内存省 4 倍**
- CPU→GPU 传输量**省 4 倍**（PCIe 常是瓶颈）
- 除法在 GPU 上并行，几乎免费

`div_` 带下划线是 in-place，省一次分配 —— 注意它作用在 `.float()`
新建的张量上，不会污染原始的 uint8 输入。

### 输出层没有激活函数

Q 值是**回报的估计**，可正可负、量纲任意，绝不能套 ReLU 或 sigmoid。
本项目的 Q 范围约 `[−1, 12.6]`。

> 常见错误：给输出层加 ReLU。那样 Q 永远非负，
> "撞死 = −1" 这个信息就永远表达不出来。

---

## 6. 初始化

```python
for layer in (self.conv1, self.conv2, self.conv3, self.fc):
    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    nn.init.constant_(layer.bias, 0.0)
# 输出层用极小增益，使初始 Q ≈ 0
for layer in (self.value, self.advantage):
    nn.init.orthogonal_(layer.weight, gain=0.01)
    nn.init.constant_(layer.bias, 0.0)
```

### 正交初始化 + `gain=√2`

正交矩阵保持向量范数，能让信号在深度方向上稳定传播，不爆炸也不消失。

`√2` 是 ReLU 的补偿系数（He 初始化的思路）：ReLU 把大约一半的激活置零，
方差减半，所以权重方差乘 2 补回来。

### 输出层 `gain=0.01`

这样初始的 `V` 和 `A` 都接近 0，于是初始 `Q ≈ 0`（实测精确为 `[0., 0.]`）。

**为什么要这样**：训练刚开始时，我们希望网络说"我不知道"，
而不是随机地对某个动作有强烈偏好。初始 Q 接近 0 让 TD 目标
`y = r + γ·Q(s')` 主要由**真实奖励** r 驱动，学习信号更干净。

这也是一个**可观测的健康信号**：训练启动时 `train.csv` 的
`q_mean` 应该从接近 0 开始慢慢上升。一开始就是几十，说明初始化写错了。

---

## 7. 为什么没有 BatchNorm 和 Dropout

**这不是简化，是正确性要求。** 详细推导见
[00 · 为什么之前不收敛](00-why-it-failed.md) 第三节，这里只讲结论。

### BatchNorm 会毁掉 Bellman 算子的不动点

TD 目标 `y = r + γ·Q_target(s')`。如果 `Q_target` 带 BatchNorm 且处于
train 模式，它用**当前 minibatch 的统计量**归一化 —— 于是 `y` 不只依赖 `s'`，
还依赖"这一批里碰巧还有哪些别的样本"。

Bellman 算子 `T` 能收敛的前提是它是**固定的**压缩映射。
一个随批次变化的算子**没有不动点**。这不是"收敛慢"，是"没有收敛目标"。

即使正确调用 `.eval()`，BatchNorm 在 RL 里也麻烦：
训练分布随策略一直漂移，running statistics 一直在追一个移动的分布。

### Dropout 会让"贪婪"变成随机

train 模式下 Dropout 随机置零 30% 的激活，`argmax` 每次调用都可能给出
不同动作。ε=0 也没用。

### 删掉之后的好处

`train()` / `eval()` 变成**空操作**，再也不可能因为漏调用而出错。
并且加了一行启动自检，三条路径（训练/评测/回放）都会跑：

```python
def assert_deterministic(net, cfg, device):
    with torch.no_grad():
        q1, q2 = net(dummy), net(dummy)
    assert torch.allclose(q1, q2), "network is stochastic at act time"
```

**这一行当初就能抓到旧管线的 Dropout bug。** 单测 6 覆盖了它。

### 那怎么稳定训练？

- 正交初始化 + 合适的 gain（信号尺度从一开始就正常）
- 输入归一化到 `[0,1]`
- 奖励归一化到 `±1`
- 梯度裁剪 `grad_clip=10`
- Huber 损失

这些手段不引入"训练/推理行为不一致"，在 RL 里比 BN 可靠得多。

---

## 8. 网络在 GPU 上的实际开销

RTX 3070 Laptop 实测：

| 项 | 数值 |
|---|---|
| 参数量 | 1.26M (5.04 MB) |
| 显存占用 | 约 800 MiB（含 CUDA context） |
| GPU 利用率 | 30–40% |
| 单样本前向 | 约 0.6 ms |
| batch 128 训练步 | 约 4.7 ms |
| batch 32 训练步 | 约 5.0 ms |

**注意最后两行**：batch 从 32 涨到 128，每步耗时几乎不变。
因为网络太小，GPU 被 **kernel 启动延迟**主导，而不是算力。
梯度步速率恒为约 200 步/秒，与 batch 无关。

**所以大 batch 是白拿的** —— 同样的时间处理 4 倍的样本，
梯度估计的方差降到 1/2（`σ/√n`）。这就是 `batch_size=128` 的理由，
不是"调"出来的，是测出来的。

反过来说，在这个规模下继续加大 batch 到 512 不会更快，
但会让每个梯度步覆盖更多样本，改变有效学习率 —— 得不偿失。

---

## 小结

```
(N,4,80,80) uint8
   │  ÷255 在 GPU 上做
   ├─ Conv2d(4→32, k8, s4)  + ReLU  →  (N,32,19,19)
   ├─ Conv2d(32→64, k4, s2) + ReLU  →  (N,64,8,8)
   ├─ Conv2d(64→64, k3, s1) + ReLU  →  (N,64,6,6)
   ├─ flatten                        →  (N,2304)
   ├─ Linear(2304→512) + ReLU        →  (N,512)          ← 93.7% 的参数
   ├─ Linear(512→1)     = V(s)
   └─ Linear(512→2)     = A(s,a)
        Q = V + A − mean(A)          →  (N,2)
```

| 设计 | 理由 |
|---|---|
| 不加 padding | 特征图缩到 6×6，fc 参数量减 65% |
| uint8 输入，GPU 内归一化 | 内存和 PCIe 各省 4 倍 |
| 输出层无激活 | Q 必须能取负值 |
| 输出层 gain=0.01 | 初始 Q≈0，学习信号由真实奖励主导 |
| 无 BatchNorm | 否则 Bellman 算子没有不动点 |
| 无 Dropout | 否则"贪婪"动作是随机的 |

---

上一篇：[03 · Dueling 网络架构](03-dueling.md)
下一篇：[05 · 环境与观测](05-environment.md)
