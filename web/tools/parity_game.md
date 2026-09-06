# 阶段 1 验收：JS 移植与 Python 参考实现的一致性

**结论：通过。** 两条轨迹各 1200 帧，物理、管道生成、像素级碰撞全部逐位一致。

```
[trace.json]     比对 1200/1200 帧，消费随机数 144/144   PARITY OK
[trace_ai.json]  比对 1200/1200 帧，消费随机数 127/127   PARITY OK
```

| 轨迹 | 驱动策略 | 覆盖重点 |
|---|---|---|
| `trace.json` | bang-bang 启发式 | 管道生成、`reset()`（5 个回合） |
| `trace_ai.json` | **训练好的 CNN** | **擦边飞行**（1200 帧一次没死） |

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

## 像素级碰撞：这条测试有鉴别力

光"通过"不够，还得证明测试**测得出错**。同一条 AI 轨迹分别用两种碰撞判定跑：

| 碰撞判定 | 结果 |
|---|---|
| 像素掩码（`hitmasks.js`） | PARITY OK，1200 帧全对 |
| 包围盒（`--bbox`） | **JS 提前撞死**，Python 还在飞 |

包围盒把 AI 擦边飞过的地方误判成撞击。掩码密度解释了原因：

| 精灵 | 尺寸 | 不透明像素占比 |
|---|---|---|
| 小鸟 | 34×24 | **72.1%** |
| 管道 | 52×320 | 92.9% |

小鸟的包围盒比它实际占的地方大出 28%。真实模型有约 4.6% 的通过是擦边的，
正好压在两种判定会分歧的地方 —— **不接掩码，浏览器里的 AI 会死在 Python 里
不会死的地方**，而且阶段 2 验收对不上时会被误诊成"推理错了"。

复现：

```bash
node web/tools/parity_check.mjs trace_ai.json          # OK
node web/tools/parity_check.mjs trace_ai.json --bbox   # 提前撞死
```

## 顺带修掉的一个真 bug

接掩码时 `mask[166.5]` 抛了 undefined。根因是 **`pygame.Rect` 会把浮点坐标
向零截断成整数**（注意是 `Math.trunc` 不是 `Math.floor` —— 负数上两者不同，
`-34.976` 在 pygame 里是 `-34`），而 JS 侧直接用了浮点。

这个偏差在纯包围盒模式下不会暴露（不索引掩码就不会报错），但判定边界本来就
已经和 Python 差半个像素了。已在 `_checkCrash` 里对齐。

## 已知的不覆盖项

- **`randomize=False` 的固定分布路径没有被覆盖。** 那条路径用
  `random.choice(GAP_Y_CHOICES)`，而 CPython 的 `random.choice` 走
  `_randbelow()`（getrandbits），不是单次 `random()` 调用 —— 抽样重放对不上。
  JS 侧用的是 `floor(rng()*len)`。demo 只走随机化路径，所以不影响；
  但如果以后要用固定分布，这里得单独处理。
- **掩码是从 Python 导出的，不是 JS 从 PNG 现算的。** 这是刻意的：Node 里
  没有 canvas，现算就跑不了本比对；而且直接导出能保证和 `checkCrash` 用的是
  同一份数据。代价是改了精灵必须重新生成 `web/assets/hitmasks.js`。
