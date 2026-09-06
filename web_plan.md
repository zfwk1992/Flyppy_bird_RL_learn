# Web demo 加固计划

分支：`demo/web`。目标是让这个页面在**真实的手机和真实的网络**上稳定可玩，
而不是"在我的桌面 Chrome 上看起来没问题"。

原始需求 7 条，下面按"已完成 / 已定位 / 待复现 / 待建设"重排，
每条都写清楚**判据**（什么算做完）和**交付物**。

---

## 0. 现状

已经做完、不要重做：

| 原需求 | 状态 | 证据 |
|---|---|---|
| ① demo 用最好的模型 | **已完成** | commit `d4e2817`。`runs/base_s0/final.pt`，固定集 400 关均分 96.91 ± 3.93，hazard 0.939%；上一代 74.53 / 1.291%。四项 parity 已对新权重重验 |
| ④ 合进 main | **PR 已开** | [#1](https://github.com/zfwk1992/Flyppy_bird_RL_learn/pull/1)，`demo/web` 落后 `main` 0 个 commit，可快进 |
| ⑦ 文案面向小白 + 流程图 | **大部分完成** | 讲解区 858 词 → 约 330 词，五张内联 SVG。剩余部分见 §4 |

还开着的：**② AI 卡住**、**③ 管道不显示**、**⑤ 跨平台测试**、**⑥ GUI 可操作性提示**。

---

## 1. 【P0】玩家死后 AI 会永久卡住 —— 根因已定位

### 现象

玩家撞管后，AI 继续飞到 30 多根就**停在半空不动**，计数器不再增加，
而且永远不会显示 "The AI crashed"。

### 根因（两处叠加，都已定位到行）

**(a) `web/index.html:718`** —— 续命请求被玩家的存活状态挡住：

```js
if (worker && !planEnded && youAlive && plan.length - aiFrame < LOOKAHEAD / 2) {
  worker.postMessage({ type: 'want', seed: planSeed, want: aiFrame + LOOKAHEAD });
}
```

`youAlive` 一旦为 false，主线程就再也不向 worker 要后续动作。

**(b) `web/index.html:653-656`** —— `endRound()` 里立刻为**下一局**发 `start`：

```js
prefetch = { seed: randomSeed(), plan: [], ended: false };
worker.postMessage({ type: 'start', seed: prefetch.seed, want: LOOKAHEAD });
```

而 `web/ai-worker.js: startJob()` 第一件事就是 `gen++`。`produce()` 的循环条件是
`job.gen === myGen`，所以当前这一局的生产循环**当场退出**。

### 后果

玩家一死，当前局的 AI 计划**立刻停止生产**。AI 只能回放缓冲区里剩下的
≤ `LOOKAHEAD`(600) 帧，放完就掉进 `stepBoth()` 的

```js
} else { stalls++; return false; }        // web/index.html:705-706
```

——**永久**返回 false，整个 `stepBoth` 不再推进。

600 帧 ÷ 每根管子约 33 帧 ≈ 多飞 18 根，叠加玩家死亡时 AI 已有的进度，
正好落在用户观察到的"30 多根"。

这还和页面自己显示的文案直接矛盾：
`the AI is still flying — watch its counter, then tap`。

### 要求

修好"玩家死后 AI 继续飞到它自己撞死（或到 `MAX_FRAMES`）"。

**约束（不许违反）**：
- **不许改变 AI 的行为**。这是调度问题，不是策略问题。不许加反应延迟、
  不许降难度、不许改 `frame_skip`、不许动 `nn.js` / `obs.js` / `game.js`。
- 预取下一局的收益（换局不卡）不能丢。现在的实现是拿"当前局停产"换来的，
  这个交换不成立。

**方向**（自己权衡，不必照抄）：
1. 当前局结束前不预取，`planEnded` 或 `aiAlive === false` 之后再发 `start`；
2. 或者让 worker 支持两个并行 job 槽（当前局 + 预取），`gen` 改成按槽计；
3. 或者开第二个 worker 专门做预取。

方案 1 最小，但换局那一下的卡顿会回来 —— 先量一下那个卡顿到底有多久
（`window.__stats()` 里有 `stalls`），再决定值不值得上方案 2/3。

### 判据

- 自动化：脚本驱动，玩家在第 5 根管子主动撞死，AI 必须继续飞到自己死亡，
  且 `aiGame.score` 明显大于 30；`stalls` 在玩家死后不得单调增长到不再变化。
- `web/tools/` 下留一个可重复运行的回归脚本，退出码 0/1 可判定。

---

## 2. 【P0】部分管道不显示 —— 未复现，先复现再修

### 现象（用户报告）

"玩了几次后有一些 pipe 没有正常显示"。

### 已排除

- `reset()` 会重建 `upperPipes` / `lowerPipes` 并重新 `_planNext()`
  （`web/game.js:228-243`），不存在跨局残留的管道数组。

### 不要猜，先复现

写一个确定性复现脚本（headless CDP，`mobile_check.mjs` 里有现成的连接代码）：

1. 固定 seed 连开 10 局，每局跑满，**逐帧**把 `game.stateForRender()` 里的
   `upperPipes/lowerPipes` 数量与坐标记下来；
2. 同时每 N 帧抓一次 canvas 像素，统计管道颜色（`#3d5a34` 一族）的连通块数量；
3. **判据**：状态里有几根在可视区内，画面上就必须有几根。对不上的那一帧
   dump 出来（状态 JSON + PNG）。

这样能把问题一分为二：
- 状态里就少了 → 逻辑问题，在 `game.js` 的生成/回收（`web/game.js:311-320`）；
- 状态对但画面少了 → 渲染问题，在 `render.js:92-97` 或精灵加载
  （`SPRITE_DATA_URIS` 有没有解码失败）。

**在复现之前不要改任何渲染代码。** 如果 10 局 × 满局都复现不出来，
就如实写"未能复现"，并记录你试过的条件（设备像素比、视口、局数、
是否后台切换过标签页），不要为了交差改一段看起来相关的代码。

---

## 3. 【P1】跨平台与健壮性测试

Cloudflare 上"十分不稳定"这句话目前**没有可判定的证据**。这一节的首要产出
不是"修好了"，而是**一套能重复运行、能给出退出码的测试**。

### 3.1 设备矩阵

用 CDP 的 `Emulation.setDeviceMetricsOverride`（**必须**带 `mobile: true` 和真实
`deviceScaleFactor`；`--window-size` 不是移动端视口，这个坑踩过一次，
见 `web/tools/mobile_check.mjs` 顶部注释）。

| 类别 | 视口 | DPR | 备注 |
|---|---|---|---|
| 小屏手机 | 320×568 | 2 | iPhone SE 一代，最窄的主流机 |
| 主流 iPhone | 390×844 | 3 | |
| 大屏 iPhone | 414×896 | 2 | |
| 主流 Android | 360×800 | 3 | |
| 平板竖屏 | 768×1024 | 2 | |
| 平板横屏 | 1024×768 | 2 | **横屏时两个画布 + 讲解区会不会挤爆** |
| 桌面 | 1440×900 | 1 | |

每个视口都要断言：
- `documentElement.scrollWidth <= innerWidth`（无横向溢出）
- 两个 canvas 都在首屏可见（`getBoundingClientRect().bottom <= innerHeight`），
  或者至少"操作提示"可见
- `window.__pageErrors` 为空

**关于真机**：CDP 模拟不等于真机。iOS 上是 WebKit 不是 Blink，
`OffscreenCanvas`、Worker 里的 `fetch` 流式读取、`image-rendering: pixelated`
的行为都可能不同。**你在云端跑不了真的 Safari。** 所以：
- 能自动化的部分按上表做；
- 不能自动化的部分（真 iOS Safari、真 Android WebView）写成一份
  **人工验收清单**放进 `web/TESTING.md`，列出具体要点几下、看什么，
  **不要声称你测过真机**。

### 3.2 误操作与生命周期

每一条都要有自动化用例：

1. **加载期间就操作** —— 权重还没下完（2.5 MB）就狂点/狂按空格
2. **刷新** —— 游戏进行中 F5，重来一次必须能正常开局
3. **快速连点"下一局"** —— 连点 10 次，`gen` 竞态下不能出现两局叠加
4. **标签页切到后台再切回** —— `requestAnimationFrame` 会停，回来时
   时间累积量必须被钳制，不能一次性补几百帧（会表现为"瞬移"）
5. **worker 加载失败** —— 拦截 `./model/*.bin` 返回 500，必须退化到主线程
   同步推理（`aiPlayer` 那条路径）而不是白屏
6. **慢网络** —— CDP `Network.emulateNetworkConditions` 模拟 3G，
   进度条要动，不能卡在 0%
7. **双击缩放 / 长按选中** —— 移动端误触不能把页面缩放或选中文字

### 3.3 交付物

- `web/tools/e2e_check.mjs`：跑完上面所有用例，全过退出 0，任一失败退出 1
  并打印失败用例名 + 现场（截图路径 + 控制台错误）
- `web/TESTING.md`：自动化覆盖了什么、**没**覆盖什么、人工清单

---

## 4. 【P1】GUI 与文案

### 4.1 操作提示

现在页面上只有一行小字 `space flap · R restart`，**手机用户看不到"点屏幕"**。

- 首屏遮罩要按输入方式区分：有粗指针（`matchMedia('(pointer: coarse)')`）
  显示 "Tap to fly"，否则显示 "Press SPACE to fly"
- 提示要在**玩家那一侧的画布上**，不是在页脚
- 重开的提示同理（手机上没有 R 键）

### 4.2 讲解区剩余工作

第 ⑦ 条要求"面向 AI 小白、尽量用流程图"。已经改成五张 SVG，还差：

- **训练循环本身没有图**。现在只有一句话说 Double DQN。补一张：
  经验回放池 → 采样 → 计算 TD 目标（用目标网络）→ 梯度 → 定期同步目标网络。
  这是小白最容易卡住的地方，也是这个项目真正的教学内容。
- 检查每一处数字是否与仓库现状一致（见 §5）。

### 4.3 判据

- 上述设备矩阵里，每个视口首屏都能看到与该设备输入方式匹配的操作提示
- 讲解区新增的训练循环图在 320px 上字号不小于 8px（viewBox 宽度 ≤ 280）

---

## 5. 【P1】页面内容与项目现状一致性核对

页面上的每一个数字都要能在仓库里找到出处。逐条核对并在
`web/PROGRESS.md` 里留一张对照表：

| 页面上的说法 | 应该是 | 出处 |
|---|---|---|
| PyTorch 96.9 / 400 局 | 96.91 ± 3.93 | `eval.py runs/base_s0/final.pt --episodes 400` |
| 浏览器 100.4 / 40 局 | 100.42 ± 14.17 | `node web/tools/ai_eval.mjs 40` |
| 人类 6.8 | 小样本 | 页面已标注"small sample"，保持 |
| 1,258,659 参数 | 逐层核对 | conv 8224+32832+36928 + fc 1179904 + 头 257+514 |
| Q 最大偏差 6×10⁻³ | 6.13e-3 | `node web/tools/nn_check.mjs` |
| 1,200 帧观测一致 | 1200 | `node web/tools/obs_check.mjs` |
| 300 次决策一致 | 300/300 | `node web/tools/nn_check.mjs` |
| 曾停在 1.3 根 | 见 docs | `docs/learn/00-why-it-failed.md` |

**规矩**：每个数字带样本量。差距小于 2 个标准误一律写"没有区别"。
对不上的就改页面，不要改数字去迁就页面。

---

## 6. 硬性约束

- **不许改变 AI 的行为**。这个 demo 的卖点就是它比人强得多。
  不加反应延迟、不降难度、不改推理逻辑、不动 `nn.js` / `obs.js` / `game.js`
  的算法部分。第 1 节是调度 bug，不是策略问题。
- **不许改** `flappy/`、`game/`、`train.py`、`eval.py`、`play.py`、`plot.py`、
  `monitor.py`、`models/`、`test/`。这些是训练侧，和网页无关。
- **只在 `demo/web` 分支上工作**。`research/stability` 上有另一条线在跑诊断，
  不要碰。不要 force push。push 前先 `git pull --rebase`。
- 提交信息和代码注释用中文，注释解释**为什么**。
- 改完必须重跑四项 parity（`obs_check` / `nn_check` / `parity_check` /
  `worker_check`），全过才能提交。

---

## 7. 分批执行：每轮只做一块

这份计划由**定时 routine 分多轮完成**，柏林时间凌晨 3:00 起、每 90 分钟一轮，
每轮是一个全新的冷启动会话，**没有上一轮的记忆**。

所以每一轮的工作方式是固定的三步：

1. **读 `web/PROGRESS.md` 末尾的"批次进度"表**，找到第一个状态不是"已完成"的批次；
2. **先提交一行认领**（把该批次状态改成 `进行中 <UTC 时间戳>`，commit + push），
   这样万一两轮撞上，后一轮能看见前一轮已经在做；
3. **只做那一个批次**，做完把状态改成"已完成"或"部分完成 + 卡在哪"，
   commit + push。**不要顺手多做下一批** —— 宁可早交，也不要一轮里塞两块
   然后中途超时、两块都只做了一半。

如果所有批次都已完成，就做 §7.2 的巡检，然后如实报告"没有待办"。

### 7.1 批次划分（按工作量切，每块争取一轮能收尾）

| 批次 | 内容 | 对应 | 量级 | 交付物 |
|---|---|---|---|---|
| **B1** | 修玩家死后 AI 卡住 | §1 | 中（根因已定位，改动面小） | 代码修复 + `web/tools/stall_check.mjs` 回归脚本 |
| **B1b** | demo 缝隙下界 85 -> 100 | §8 | 小 | `index.html` + `ai-worker.js` 三处传 `gapRange`；**不许改 `DEFAULT_GAP_RANGE`** |
| **B2** | 管道不显示：**只写复现脚本，不修** | §2 | 中 | `web/tools/pipe_render_check.mjs` + 复现结论（复现到了 / 没复现到，附条件） |
| **B3** | 管道不显示：按 B2 的结论修 | §2 | 视 B2 而定 | 修复 + B2 脚本转成回归用例；**B2 没复现就跳过 B3，直接把状态写成"跳过（未复现）"** |
| **B4** | 设备矩阵 7 个视口 | §3.1 | 中 | `web/tools/viewport_check.mjs`，7 个视口全过退出 0 |
| **B5** | 误操作用例 1–4：加载中操作 / 刷新 / 连点下一局 / 切后台再回来 | §3.2 | 大 | 用例并入 `web/tools/e2e_check.mjs` |
| **B6** | 误操作用例 5–7：worker 加载失败退化 / 慢网络 / 双击缩放长按 | §3.2 | 中 | 同上，并补 `web/TESTING.md`（含**真机人工清单**） |
| **B7** | 操作提示按输入方式区分 | §4.1 | 小 | `pointer: coarse` 分支 + 在 B4 的矩阵里加断言 |
| **B8** | 训练循环图（回放池→采样→TD 目标→目标网络同步） | §4.2 | 中 | 一张内联 SVG，viewBox ≤ 280 宽，320px 上字号 ≥ 8px |
| **B9** | 页面数字与仓库现状核对 | §5 | 小 | `web/PROGRESS.md` 里的对照表 |
| **B10** | 全量巡检 + 总结 | §7.2 | 小 | 见下 |

**B3 依赖 B2，B7 依赖 B4，其余批次互相独立**，可以按表顺序做。
B1b 独立，但它会改到页面文案，所以 B9 的数字核对要覆盖它。

### 7.2 B10 巡检做什么

1. 四项 parity 全跑一遍（`obs_check` / `nn_check` / `parity_check` / `worker_check`）
2. 新增的所有 `*_check.mjs` 全跑一遍，记录退出码
3. 把 B1–B9 的状态汇总成一张表放进 `web/PROGRESS.md` 顶部，
   让人一眼看到**哪些是"已修复并验证"、哪些只是"改了但没验证"、
   哪些"没能复现"、哪些"没做"**
4. 明确列出**依赖真机（iOS Safari / 真 Android）而云端无法验证**的结论

### 7.3 每轮都适用的规矩

- 每做完一块就 commit + push，不要攒。中途超时的话，已 push 的才算数。
- 改完任何东西都要重跑四项 parity，全过才能提交。
- **宁可交空结论，不要编数字，不要假装跑过测试。**
- 分清三种状态：**已修复并验证** / **已改但未验证** / **未能复现**。

---

## 8. 【P1】把 demo 的缝隙下界从 85 提到 100

### 要做什么

网页 demo 里管道缝隙的取值范围从 `[85, 165]` 改成 **`[100, 165]`**，上界不变。
目的是让人类玩家别一上来就被 85px 的窄缝劝退 —— 那是接近物理极限的难度
（`docs/EXPERIMENTS.md`：gap 70–85 只占 10.5% 的管道，却贡献 53.5% 的死亡）。

### ⚠️ 千万不要改 `DEFAULT_GAP_RANGE`

`web/game.js:40` 的 `DEFAULT_GAP_RANGE = [85.0, 165.0]` 是**四项 parity 的基准**。
`parity_check.mjs` / `obs_check.mjs` / `nn_check.mjs` / `worker_check.mjs`
都用默认值构造 `FlappyGame`，并和 Python 侧录的 `trace.json` / `trace_ai.json` /
`obs_ref.json` / `nn_ref.json` 逐位比对。

**改了默认值，四项 parity 立刻全挂，而重新生成参考文件需要 torch + pygame ——
你（云端 routine）没有，会把自己卡死在一个改不回去的状态。**

所以：默认值原样不动，只在 demo 页面这一侧覆盖。

### 正确做法

`FlappyGame` 的构造函数已经接受 `opts.gapRange`（`web/game.js:142,162,186`），
不用改 `game.js`。在 `web/index.html` 里定义一个常量，例如

```js
// demo 专用难度：下界抬到 100，别让新玩家第一局就撞在 85px 的窄缝上。
// **不要去改 game.js 的 DEFAULT_GAP_RANGE** —— 那是 parity 的基准。
const DEMO_GAP_RANGE = [100.0, 165.0];
```

然后**三处**构造都要传：

| 位置 | 谁 |
|---|---|
| `web/index.html:603` | 玩家那一局 `you` |
| `web/index.html:605` | 主线程用来显示的 `aiGame` |
| `web/ai-worker.js:121` | worker 里真正跑推理的那一局 |

第三处**必须通过 `start` 消息把 gapRange 传进 worker**，不能在 worker 里写死 ——
写死的话 `worker_check.mjs`（用默认值构造）就会挂。

### 三处必须完全一致，否则锁步失效

人机对战的前提是"同一个 seed 生成同一串管道"。三处里只要有一处 gapRange 不同，
同一个 seed 就会产生**不同的管道序列**，左右两块画面不再是同一关，
整个对照失去意义 —— 而且这种 bug 在画面上表现得很隐蔽（两边都能正常玩，
只是管道长得不一样）。

### 这不算"改变 AI 的行为"

模型是在 gap 85–165 上训练的，`[100, 165]` 是它见过的分布的**子集**，
不存在分布外的问题。AI 只会表现得更好一点。策略、推理逻辑、frame_skip
一个字都没动，所以这条不违反"不许改变 AI 行为"的约束。

### 判据

1. **四项 parity 仍然全过**（它们用默认的 85，不受影响）—— 这是硬门槛
2. 新增一个检查（或并入 §3 的 e2e 脚本）：同一 seed 下，`you`、`aiGame`、
   worker 三方的前 600 帧 `stateForRender()` 里的管道 x/y/gap **完全一致**
3. 采样至少 200 根管道，实测 gap 最小值 **≥ 100**，最大值仍 ≤ 165
4. 页面上凡是写了"gap 85"或"85–165"的文案跟着改；§5 的数字核对要覆盖这条

### 顺带

改完之后 AI 的分数会比 96.9 高一些（更容易的分布）。**不要**因此去更新页面上
"PyTorch · 400 episodes 96.9"那个数字 —— 那是在**训练分布 85–165** 上测的，
是模型能力的基线，和 demo 的难度设置是两回事。如果想在页面上体现 demo 难度，
另加一行说明，不要覆盖原数字。
