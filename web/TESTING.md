# 这个 demo 是怎么测的，以及**哪些没测**

先说最重要的一句：**这里所有自动化跑的都是 headless Chromium（Blink）。
iOS 上是 WebKit，不是 Blink。** 凡是结论依赖浏览器引擎行为的，
自动化说"过"不等于 iPhone 上是好的。真机部分在最后一节，是**人工清单**，
没人跑过就是没跑过。

---

## 1. 一条命令跑完全部

```bash
python3 -m http.server 8123 --directory web &

node web/tools/obs_check.mjs        # 观测管线逐像素
node web/tools/nn_check.mjs         # 前向逐位
node web/tools/parity_check.mjs     # 物理与管道生成
node web/tools/worker_check.mjs     # worker 回放 vs 直接推理
node web/tools/gap_check.mjs        # demo 缝隙下界 + 三处 gapRange 一致
node web/tools/stall_check.mjs      # 玩家死后 AI 不许卡住
node web/tools/viewport_check.mjs   # 7 个视口布局 + 操作提示
node web/tools/e2e_check.mjs        # 误操作与生命周期
```

全部退 0 = 过，退 1 = 有失败并打印现场，退 2 = 环境问题（浏览器/服务没起来）。

`web/tools/*.py` 需要 torch + pygame，**只在本机跑**；参考文件
（`nn_ref.json` / `obs_ref.json` / `trace_ai.json`）已经对着当前权重生成并提交，
换模型才需要重跑，命令见 `dump_nn_ref.py` 的头注释（认 `FLAPPY_WEB_CKPT` 环境变量）。

---

## 2. 自动化覆盖了什么

### 2.1 移植正确性（对着 PyTorch 逐位比）

| 检查 | 覆盖 | 实测 |
|---|---|---|
| `obs_check` | 1200 帧观测 | 逐像素一致 |
| `nn_check` | 300 次决策 | 帧栈逐位一致、动作 300/300 一致，Q 最大偏差 6.13e-3（fp16 权重 + 累加顺序） |
| `parity_check` | 1200 帧物理 + 管道生成 | 逐位一致，随机数消费次数也一致 |
| `worker_check` | 3 个 seed | worker 预跑回放 = 主线程直接推理，逐位一致 |

### 2.2 行为与布局

| 检查 | 判据 | 负向对照 |
|---|---|---|
| `gap_check` | 三处 `gapRange` 一致；实测 gap 下界 ≥ 100 | 两份坏页面（aiGame 忘传 / start 忘带）各自退 1 |
| `stall_check` | 玩家死后 AI 继续飞到自己撞死 | 有 |
| `viewport_check` | 7 视口：无横向溢出 / 首屏 / 两画布不重叠 / 无 JS 报错 / 提示文案匹配输入方式 | 五条各有对照，实测都能红 |

### 2.3 误操作与生命周期（`e2e_check`）

| 用例 | 判据 | 状态 |
|---|---|---|
| `boot-spam` | 限速 4 Mbps，权重没下完就狂点狂按 → 之后仍能开局 | 过 |
| `refresh` | 游戏中刷新 → 仍能重新开局 | 过 |
| `rapid-restart` | 连点 10 次换管道序列 → `aiFrame` 不得越过 `plan` | 过 |
| `background-tab` | 冻结 3 秒恢复 → 分数跳幅 ≤ 2 根 | 过 |
| `worker-fallback` | `new Worker` 抛异常 → 退化到主线程推理，仍能玩 | 过 |
| `weights-dead` | 权重下不来 → 给出 "could not load the agent"，不是白屏 | 过 |
| `touch-misfire` | `touch-action: manipulation`、`user-select: none`、双击不缩放 | 过 |
| `slow-network` | 慢网下进度条要动 | **未能验证**，见下 |

---

## 3. ⚠️ 没测到的，以及为什么

### 3.1 `slow-network` —— 造不出可用的慢网

计划 §3.2 第 6 条要求"3G 下进度条要动，不能卡在 0%"。**这条现在没有验证**，
默认跳过，加 `--slow` 才会执行。试过三种办法，都不行：

1. **页面会话上 `Network.emulateNetworkConditions`**：没用。权重是在
   **worker 内部** fetch 的，而 worker 是独立的 CDP target，页面这条 Network 域
   管不到它。实测限到 400 kbps 之后页面照样 `ready=true`、`plan=608`。
2. **挂到 worker 会话上限速**（`Target.setAutoAttach` + `waitForDebuggerOnStart`
   + 对 worker sessionId 发限速）：本地 2.5 MB 两百毫秒就下完，等 attach 上去
   早结束了。实测第一次采样就已经 `ready=true`。
3. **自己写一个按字节滴流的静态服务器**：服务器本身是对的（Node 侧
   `fetch` 3 秒收到 50 个分块），但**它会让 Chrome 的 module worker 坏掉** ——
   页面自己的模块图正常（index.html / game.js / render.js / assets 都请求到了），
   `ai-worker.js` 也服出去了，但 **worker 的 import 一个都没发**
   （nn.js / obs.js / model/weights-meta.js 全无请求），worker 既不报错也不 ready。
   把滴流关掉、退化成普通静态服务器，**症状一模一样**，所以不是"喂得慢"的问题。
   换回 `python -m http.server` 立刻正常。原因没查清。

要做完这条，得先解决第 3 点那个不兼容，或者找到能真正限住 worker 请求的办法。

### 3.2 真机：一次都没跑过

`Emulation.setDeviceMetricsOverride` 只是让 Blink 按给定尺寸和 DPR 布局，
**它不会把渲染引擎换成 WebKit**。下面这些在 iOS 上有可能和自动化结果不同，
**必须人工过一遍**：

- **module worker**：iOS Safari 16 之前对 `new Worker(url, {type:'module'})`
  的支持不完整。跑不起来时应该走 `fallbackToMainThread()`，
  但那条路在真 Safari 上没人验证过。
- **`res.body.getReader()` 流式读取**：拿不到流时 `fetchWeights` 会退回
  `arrayBuffer()`（没有进度条但能下完）。同样没在真机上验证。
- **`image-rendering: pixelated`**：Safari 的实现和 Blink 不同，
  像素画面可能被平滑成糊的。
- **`(pointer: coarse)`**：B7 的提示分支全靠它。iPadOS 上接了妙控键盘/触控板
  时它的取值，没有实测。
- **`touch-action: manipulation` 与双击缩放**：iOS 的处理和 Blink 有出入。
- **`visualViewport.scale`**：`touch-misfire` 用它判断有没有被缩放，
  iOS 上的语义未必一致。

### 3.3 线上环境

用户报告"在 Cloudflare 上十分不稳定"。**本地一次都没复现出来。**
自动化跑的全是 `localhost`，没有 CDN、没有 `_headers` 的缓存策略、
没有真实网络抖动。想定位得先拿到线上的复现步骤或错误日志。

### 3.4 `mobile_check.mjs` 里有一条不会红的判据

它断言 `documentElement.scrollWidth <= innerWidth`。Blink 的移动端模拟会
shrink-to-fit：内容撑宽时布局视口跟着涨，这条不等式**恒成立**
（拿 `#app{width:1200px}` 做对照，实测 scrollWidth=1200 innerWidth=1200，还是绿的）。
`viewport_check.mjs` 已经改成拿设给 `setDeviceMetricsOverride` 的宽度当尺子；
`mobile_check.mjs` 的老判据**没动**，别拿它的绿当数。

---

## 4. 人工验收清单（真机）

每条都写了"点什么、看什么"。做完请把结果记到 `web/PROGRESS.md`。

### iPhone（Safari，最好 iOS 16 和 17 各一台）

1. 打开页面，**别动**，看进度条：应该从 0 走到 100%，不是一直空着。
2. 首屏应该看到 **"Tap to fly"**（不是 "Press SPACE"），页脚是
   "tap the left panel to flap"。
3. 点一下开局，故意撞管。**玩家死后 AI 必须继续飞**，右边计数器要继续涨，
   直到它自己撞死才停。**这条是最重要的**：它在 headless 上测不出来
   （CDP 的 `setCPUThrottlingRate` 不节流 worker 线程），是真机才暴露的 bug。
4. 连着玩 10 局以上，注意**有没有管道不显示**。B2 在本地
   8651 帧 / 33810 根次里一次都没复现到，只能靠真机碰。碰到了请截图。
5. 双击画布：页面**不能**被放大。长按：**不能**弹出选择/复制菜单。
6. 玩到一半切到别的 app，等 5 秒切回来：小鸟**不能**瞬移一大段。
7. 横屏转竖屏各来一次，两块画布都要在，不能错位或被裁。

### Android（Chrome 与三星浏览器各一）

同上 1–7。另外：

8. 开省电模式再玩一局 —— worker 被降频时缓冲会变薄，看 AI 会不会卡住。

### 平板（iPad）

9. 竖屏、横屏各看一次布局。
10. 接上妙控键盘/蓝牙键盘，刷新页面：提示应该变成
    **"Press SPACE to fly"**（`pointer: coarse` 变成 false）。拔掉再看一次。

### 弱网

11. 手机开飞行模式再关，或者用系统的网络限速，在 3G 档下打开页面：
    进度条要动，不能长时间停在 0%。
