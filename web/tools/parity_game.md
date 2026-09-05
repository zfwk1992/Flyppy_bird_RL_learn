# 阶段 1 验收：JS 移植与 Python 参考实现的一致性

**结论：通过。** 1200 帧、144 次随机抽样、5 个回合，物理与管道生成逐位一致。

```
比对 1200 / 1200 帧，消费随机数 144 / 144
PARITY OK  物理与管道生成逐位一致，随机数消费次数也一致
```

## 为什么不是"两边用同一个 seed"

Python 的 `random` 是 Mersenne Twister，JS 里没有等价实现。硬要对齐得把
MT19937 移植到 JS —— 但那验证的是 PRNG 移植得对不对，**不是我们真正关心的
那件事**（游戏逻辑移植得对不对）。

所以改成：把 Python 消费的每一个 `random()` **原始抽样**记录下来，
让 JS 按同样顺序重放。这成立的前提是两边的取值公式等价：

```python
random.uniform(a, b)  ==  a + (b - a) * random()     # CPython 源码
```
```javascript
_uniform(lo, hi)      =>  lo + rng() * (hi - lo)     // web/game.js
```

乘法在 IEEE754 下可交换，所以同一个 `r` 进去，两边结果**逐位相同**。
于是比对拆成两件互不干扰的事：

| 验证对象 | 依赖随机数？ | 判据 |
|---|---|---|
| **物理**（`playery` / `playerVelY` / `basex`） | 否，只由动作决定 | 逐帧精确相等 |
| **管道生成**（`x` / `y` / `gap`） | 是，重放同一串抽样 | 逐位相等 |
| **计分 / done / reset** | 间接 | 逐帧一致 |

额外的一道保险：**随机数消费次数也必须相等**。JS 若多抽或少抽一次，
`replayRng` 会直接抛错或在末尾对不上 —— 这能抓到"逻辑等价但抽样顺序不同"
这类最难发现的偏差。

## 怎么跑

```bash
python web/tools/dump_python_trace.py     # 生成 web/tools/trace.json
node    web/tools/parity_check.mjs        # 逐帧比对
```

`trace.json`（约 484 KB）已提交，所以**没有 Python 环境也能跑比对**。
改了 `web/game.js` 之后重跑第二条命令即可。
只有改了 `game/flappy_env.py` 才需要重新生成 trace。

## 覆盖范围

- 1200 帧 / 5 个回合（撞死后立刻重开，因此 `reset()` 的一致性也被覆盖）
- 144 次随机抽样 ≈ 48 根管道，全部走域随机化路径（`randomize=True`, gap 85–165）
- 动作由一个朴素 bang-bang 启发式产生并**记录下来**，JS 重放而不是重算 ——
  这样即使有分歧，两条轨迹也不会发散成完全不同的东西，第一处不一致能直接定位

## 过程中发现的问题（已修）

第一次跑报了 11 处不一致，全部落在 `done` 那几帧上，PY 侧记的是
`playery=224 / velY=0 / basex=0`（重开后的值）。查下来是**采样器写反了顺序**：
先 `reset()` 后快照，于是 done 帧记的是重开后状态，而 JS 比的是撞死那一帧。
移植本身没问题 —— 其余 1189 帧当时就已经全对。

修法是把快照移到 `reset()` 之前。同时要保证两边的抽样顺序一致：
`reset()` 本身也消费随机数，所以 JS 必须在**比对完这一帧之后**再 reset。

## 已知的不覆盖项

- **`randomize=False` 的固定分布路径没有被覆盖。** 那条路径用
  `random.choice(GAP_Y_CHOICES)`，而 CPython 的 `random.choice` 走
  `_randbelow()`（getrandbits），不是单次 `random()` 调用 —— 抽样重放对不上。
  JS 侧用的是 `floor(rng()*len)`。demo 只走随机化路径，所以不影响；
  但如果以后要用固定分布，这里得单独处理。
- **碰撞只覆盖到包围盒层面。** `game/resources.py` 的 `checkCrash` 用像素级
  hitmask，JS 侧目前在未提供 hitmask 时退化为包围盒。本次比对里两边的
  `done` 逐帧一致，说明这条轨迹上没有踩到差异，但不等于所有情况都等价。
  接入 hitmask 后应当重跑本比对。
