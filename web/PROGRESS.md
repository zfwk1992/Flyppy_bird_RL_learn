# demo/web 进度

> 这个文件是跨会话的唯一状态来源。定时任务每次启动都是冷启动，
> **先读这里**，做完一小步就更新这里并提交。不要凭记忆假设做过什么。

## 当前阶段

**阶段 1、阶段 2、阶段 3：全部完成。** 左右分屏的对战页面（`web/index.html`）
已经能跑：玩家一边、AI 一边、同一个 seed、AI 全程不停，页面下方有六段英文讲解，
移动端布局已验证。下一步是**阶段 4：发布**（Cloudflare Pages 配置 / OG 卡片图 /
LinkedIn 帖子草稿）。

## 阶段清单

- [x] **阶段 1　游戏本体**
  - [x] `web/game.js`：物理与管道生成，逐条对照 `game/flappy_env.py`
  - [x] 种子化 PRNG（两只鸟必须共用同一组管道）
  - [x] Canvas 渲染：管道 / 小鸟 / 地面，黑底
  - [x] 键盘 + 触摸控制，一键重开（`web/index.html`，秒重开无停顿）
  - [x] **验收**：见 `web/tools/parity_game.md` —— 1200 帧 / 144 次抽样 / 5 回合逐位一致（PARITY OK）
- [x] **阶段 2　AI 上场**
  - [x] 模型导出 ONNX fp16 —— `web/model/flappy_dqn_fp16.onnx`（2.53 MB，本机已导出并提交）
  - [x] 观测管线（`web/obs.js`）：解析式重建 288×512 的 R 通道 → 照抄 OpenCV
        权重表的 INTER_AREA 降到 80×128 → 阈值 1 二值化 → 4 帧栈。
        **没有读画布**，理由见 `web/tools/parity_ai.md`（一句话：阶段 3 要往
        画布上加装饰，读画布会让纯视觉改动污染观测）。
  - [x] 浏览器推理（`web/nn.js` + `web/ai.js`）：手写前向，零依赖，
        权重 `web/model/weights_fp16.bin`（2.52 MB）+ `weights-meta.js`。
        **没用 onnxruntime-web** —— 首屏体积不划算，实测手写前向 28 ms/次、预算 133 ms。
  - [x] **验收**：见 `web/tools/parity_ai.md`。画面与观测 1200/1200 帧逐位一致，
        帧栈 300/300 逐位一致，动作 300/300 与 PyTorch 一致（Q 值最大偏差 4e-3），
        浏览器与 Node 逐局分数 5/5 完全相同。30 局实测平均 86.4 根（标准误 14.8，
        与 Python 侧 100 局的 78.2 在噪声之内）。
- [x] **阶段 3　同台竞技 + 讲解**
  - [x] 左右分屏：玩家一屏 / AI 一屏，各自独立实例与画布，同一个 seed 所以管道相同
  - [x] 计分板常驻：你的本局 / 你的最佳 / 尝试次数 vs AI 本局 / 最佳 / 崩溃次数
  - [x] 英文讲解：RL 基础 / Q-V-A / 网络架构 / 训练结果（`web/index.html` 里
        新增的 `#learn` 区块，footer 下方，六个小节）
  - [x] 移动端布局：320/390px 视口下用 headless Chromium 截图 + 实际点击/触摸
        模拟验证过，两屏保持并排不堆叠，讲解区文字不溢出，控制台无报错
- [ ] **阶段 4　发布**
  - [ ] Cloudflare Pages 配置
  - [ ] OG 卡片图
  - [ ] LinkedIn 帖子草稿

## 阻塞项

（暂无。原来的 torch 阻塞项已解除 —— 模型已在本机导出成 ONNX 并提交，
云端不再需要 torch。）

## 本机（非云端）才能做的事

云端跑不了这些，需要在有 torch + pygame 的机器上做：

- `python web/tools/dump_python_trace.py` —— 重新生成参考轨迹。
  **只有改了 `game/flappy_env.py` 才需要重跑**；`trace.json` 已提交，
  改 `web/game.js` 之后只要跑 `node web/tools/parity_check.mjs` 就行。
- 阶段 2 的 ONNX 导出（若云端装不上 torch）。

## 给云端的提醒

- **改完 `web/game.js` 必须重跑两条 parity 比对**，这是阶段 1 的验收标准，别让它退化：
  ```
  node web/tools/parity_check.mjs trace.json
  node web/tools/parity_check.mjs trace_ai.json
  ```
  两条都必须 `PARITY OK`。不需要 Python —— trace 已提交。
- **碰撞必须传 `hitmasks`**（`web/assets/hitmasks.js`）。退化成包围盒会让
  AI 擦边飞过被误判成撞击 —— 已实测：`--bbox` 跑 AI 轨迹会提前撞死。
- **改完 `web/obs.js` 或 `web/nn.js` 必须重跑阶段 2 的两条比对**：
  ```
  node web/tools/obs_check.mjs      # 画面 + 观测，1200 帧逐位
  node web/tools/nn_check.mjs       # 帧栈 + Q 值 + 动作，300 次决策
  ```
  同样不需要 Python —— `obs_ref.json` / `nn_ref.json` 已提交。
  方法与结论见 `web/tools/parity_ai.md`。
- **AI 不从画布取像素**，`web/obs.js` 是解析式重建画面的。所以可见画布可以随便
  加装饰（计分板、边框、特效），不会影响 AI。**但反过来**：改了
  `web/assets/obs-sprites.js` 或 blit 顺序就会影响 AI，必须重跑 obs_check。
- **页面不加载那个 .onnx**。推理用的是 `web/model/weights_fp16.bin` +
  手写前向（`web/nn.js`）。onnx 留着只是同一份权重的可验证副本，别去接它。
- **界面是左右分屏**：玩家一屏、AI 一屏，同一个 seed 所以管道相同。
  玩家死了立刻用**同一个 seed** 重开，AI 全程不停 —— 差距就靠 AI 那个一直在涨的
  计数体现出来。这是刻意设计，不要改成"两边一起重开"。

## 变更日志

- 2026-09-04：新建 `web/game.js`，ES module，`FlappyGame` 类。逐条对照
  `game/flappy_env.py`（物理常量、`_sample_pipe`/`_plan_next`/`_center_range`/
  `_min_spacing_for`/`_sample_spacing`/`_travel_slack` 的域随机化管道生成、
  显式跨越计分、管道生成/回收、动画索引、地面滚动 `basex`）与
  `game/resources.py`（`checkCrash`/`pixelCollision`，碰撞用可选 hitmask
  参数，未提供时退化为包围盒判定，留给渲染阶段接入像素级 hitmask）。
  默认参数对齐 `flappy/config.py` 的 CONFIG（训练 `final_v1_best.pt` 实际用的
  超参数：`pipe_gap_range=(85,165)`，注意与 `flappy_env.py` 模块级默认值
  `(80,165)` 不同，已在代码注释里说明并采用前者）。随机源通过构造函数
  `rng` 参数注入（默认 `Math.random`），为下一步的种子化 PRNG 预留接口，
  这一步本身还没做种子化。用 Node 跑了简单的 smoke test（xorshift32 伪随机
  + 追踪缝隙中心的启发式策略），确认：域随机化缝隙落在配置的 85-165px
  范围内、管道正常生成/回收、跨越计分正常触发（15 局启发式平均约 10 根，
  最高 23 根，量级合理）、参数校验的异常路径能正确抛错。
  下一步：种子化 PRNG（两只鸟必须共用同一组管道），做完之后才能补
  `web/tools/parity_game.md` 里的 Python/JS 管道序列比对。

- 2026-09-04：加了 `createSeededRng(seed)`（mulberry32，`web/game.js`），
  `FlappyGame` 构造函数新增 `seed` 参数（未显式传 `rng` 时生效，`rng` 优先级
  更高）。两只鸟各自用相同的整数 seed 构造实例（各自独立的生成器对象，
  不是共享同一个生成器）就能拿到逐帧相同的管道序列 —— 因为管道生成只依赖
  帧数推进和内部状态（`_nextGap`/`_lastGapCenter`/`_lastSlack`），完全不依赖
  玩家动作，所以两边只要 `step()` 调用次数同步，序列天然一致，不需要跨实例
  共享随机源。新增 `web/tools/smoke_test_seed.mjs`（`node web/tools/smoke_test_seed.mjs`
  跑），用一个"追缝隙中心"的简易启发式让两只鸟能撑到 176 帧，验证了：同 seed
  两个独立实例、完全不同的动作序列下管道逐帧一致；不同 seed 产生不同管道
  （排除 seed 参数被忽略的退化情况）；同一 seed 三次独立重放结果完全一致；
  `createSeededRng` 本身确定性、值域 [0,1)、1000 采样里 999+ 个不同值，分布
  正常。也验证了不传 seed 时默认 `Math.random` 行为不受影响（向后兼容）。
  注意：这一步只解决"JS 内两只鸟共用同一组管道"，**没有**让 JS 的随机数
  算法和 Python 的 `random` 模块（Mersenne Twister）比特对齐 —— 那是
  `web/tools/parity_game.md`（阶段 1 的验收项）要做的事，目前还没做，
  `web/tools/` 目录也还没有那个文件。
  下一步：Canvas 渲染（管道 / 小鸟 / 地面，黑底）。

- 2026-09-05：新增 `web/render.js`（`loadSprites()` + `Renderer` 类），
  逐条对照 `game/flappy_env.py: _draw()` 的 blit 顺序（背景→管道→地面→小鸟）
  和 `game/flappy_bird_utils.py` 的精灵加载（上管道 = pipe-green.png 旋转
  180°，玩家动画 0/1/2 = upflap/midflap/downflap）。地面按 Python 的做法
  单次 blit、不平铺（`base.png` 336px 比屏幕 288px 宽 48px，`basex` 取值
  范围 `(-48, 0]` 天然铺满，和 Python 一致）。
  精灵没有以 PNG 文件提交：仓库根 `.gitignore` 里 `*.png` 只给
  `images/*.png` 开了例外，改 `.gitignore` 超出了这次改动被允许碰的范围
  （只能改 `web/`、`plan.md`、`web/PROGRESS.md`）。绕开办法：写了
  `web/assets/sprites-data.js`，把 6 张精灵（`background-black.png`、
  `base.png`、`pipe-green.png`、`redbird-{up,mid,down}flap.png`，取自
  仓库根 `assets/sprites/`）转成 base64 data URI 内嵌在 JS 里（原始
  18KB，base64 后约 25KB，可忽略不计），`render.js` 从这个模块加载
  `Image`，不再走独立 PNG 文件请求。
  新增 `web/index.html` 作为渲染层的开发用冒烟测试页（**不是**最终的
  控制/重开体验，那是下一个清单项要做的事）：canvas 288×512，空格/点击
  扇翅，30 步/秒固定时间步物理 + `requestAnimationFrame` 渲染，死亡后
  定格。用 Playwright（Node 版，`/opt/node22/lib/node_modules/playwright`，
  Python 环境没装 playwright 包）起本地 `python3 -m http.server` 实测
  截图验证：黑底背景、管道颜色/朝向正确（上管道确实是倒过来的）、小鸟
  正常飞行/扇翅动画、地面滚动条纹正常、从两根管道之间飞过没有触发误碰撞、
  以及不扇翅时小鸟正确坠地并在死亡帧定格（没有卡死或抛异常）。
  控制台只有一条无害的 `favicon.ico 404`，没有其他错误。
  下一步：键盘 + 触摸控制、一键重开（把 `index.html` 里现在这段临时的
  开发用输入处理换成正式实现），然后补 `web/tools/parity_game.md` 的
  Python/JS 管道序列比对（阶段 1 的验收项，目前还没做）。

（每次运行在这里追加一行：日期 + 做了什么 + 下一步）

- 2026-09-05（本机）：补上阶段 1 的验收项。设计并跑通了 JS/Python 逐帧一致性
  比对，结果 **PARITY OK**：1200 帧、144 次随机抽样、5 个回合，物理
  （`playery`/`playerVelY`/`basex`）与管道生成（`x`/`y`/`gap`）逐位一致，
  随机数消费次数也相等，计分与 done/reset 逐帧一致。
  方法不是"两边同一个 seed"（Python 是 Mersenne Twister，JS 没有等价实现，
  移植 MT19937 验证的是 PRNG 而非游戏逻辑），而是把 Python 消费的每一个
  `random()` 原始抽样记下来让 JS 重放 —— 两边 uniform 公式等价，同一个 r
  进去逐位相同。详见 `web/tools/parity_game.md`。
  顺带确认了 `web/assets/sprites-data.js` 里 6 张内嵌精灵与
  `assets/sprites/*.png` **字节完全一致**（md5 逐个比对），渲染源可信 ——
  这对阶段 2 的观测管线很关键。
  本机装了 Node v24.19.0。
  下一步：阶段 1 只剩「键盘 + 触摸控制、一键重开」。注意 `index.html` 里
  现在那段输入处理是渲染层的临时冒烟测试，要换成正式实现；
  重开要**瞬间**生效（plan.md 已定：不削弱 AI，靠快速重开维持体验）。

- 2026-09-05（本机）：把阶段 2 的前置全部备好，阶段 1 收尾。
  1) **ONNX 导出**：`web/model/flappy_dqn_fp16.onnx`（2.53 MB）。fp32 是 5.05 MB，
     fp16 减半且 300 组随机输入下 **argmax 一次都没变**（Q 值误差 8e-3）。
     导出脚本一并提交（`web/tools/export_onnx.py` / `pack_onnx.py`），
     要 fp32 随时能重新生成。**云端不再需要 torch。**
  2) **像素碰撞掩码**：`web/assets/hitmasks.js`，从 Python 的 `HITMASKS` 直接
     导出（不是 JS 从 PNG 现算 —— Node 里没 canvas，现算就跑不了 parity）。
     顺带修了一个真 bug：`pygame.Rect` 会把浮点坐标**向零截断**（`Math.trunc`
     不是 `Math.floor`，负数上不同），JS 侧原本直接用浮点，接掩码时
     `mask[166.5]` 直接抛 undefined。
  3) **第二条 parity 轨迹**：`trace_ai.json`，由**训练好的 CNN** 驱动，
     1200 帧一次没死（真正的擦边飞行）。这条轨迹有鉴别力：同样的数据用
     `--bbox` 跑会让 JS 提前撞死。原来那条启发式轨迹撞得太干脆，测不出
     包围盒和掩码的差异。
  4) **阶段 1 最后一项**：`web/index.html` 换成正式实现 —— 空格/点击/R 重开，
     死后秒重开无停顿，接了 hitmasks，移动端 viewport + `pointerdown`（避开
     click 的 300ms 延迟），切后台回来限制物理追帧。
  下一步：阶段 2 的「离屏画布 + 覆盖判定降采样到 80×128 + 4 帧栈」。
  降采样**不必**复刻 cv2 的 INTER_AREA，用「格子内有没有非黑像素」的覆盖
  判定即可（已实测模型不敏感，见 plan.md 第三节）。

- 2026-09-05（本机）：**阶段 2 整段做完并通过验收**，顺手把阶段 3 的左右分屏
  和计分板也做了。定时任务 10:05 那次因为共用的 5 小时额度被本机会话用光，
  开机 3 秒就退出（`rate_limit: rejected (five_hour)`），一个文件都没动，
  所以这一段是在本机做的。

  1) **观测管线 `web/obs.js`**。没有走 `getImageData` 读画布，而是按 `_draw()`
     的顺序自己把 288×512 的 **R 通道**拼出来。三个理由写在
     `web/tools/parity_ai.md`，最关键的一条是：阶段 3 要往画布上加计分板和特效，
     只要 AI 从画布取像素，一个纯视觉的改动就能让 AI 变笨且极难定位。
     新增 `web/assets/obs-sprites.js`（五张精灵合成到黑底后的 R 通道，75 KB）。
     导出脚本 `web/tools/export_obs_sprites.py` 里断言了让这件事成立的性质：
     `R>1`、`任意通道>1`、`alpha>0` 三者逐像素相同。
  2) **降采样照抄了 OpenCV 的 INTER_AREA 权重表**，没有用原计划的覆盖判定近似。
     算了一下发现那个近似的偏差是**系统性**的不是随机的：缩放比 288/80=3.6
     非整数，最小非零重叠 0.2，乘最暗像素 R=83 再除格子面积 14.4 得 1.15，
     round 成 1 而阈值要求 >1 —— 管道边缘那一列会稳定差一格。照抄只要三十行。
     附带的坑：OpenCV 收尾是 cvRound（**.5 进偶数**），不是 Math.round。
  3) **推理走手写 JS 前向**（`web/nn.js`），没用 onnxruntime-web。
     这是给 LinkedIn 用的，wasm 运行时那几 MB 首屏体积不划算；而网络只有
     3 卷积 + 3 线性、一次前向 1230 万次乘加、每秒只需 7.5 次。
     实测 **28 ms/次**（预算 133 ms）。新增 `web/model/weights_fp16.bin`（2.52 MB）
     + `weights-meta.js`，导出脚本 `web/tools/export_weights.py`。
  4) **验收**（详见 `web/tools/parity_ai.md`）：画面重建与二值观测在 trace_ai 的
     1200 帧上**逐位一致**；帧栈 300/300 逐位一致；动作 300/300 与 PyTorch 一致，
     Q 值最大偏差 4.02e-3（fp16 权重 + 累加顺序）；`web/tools/selftest.html`
     在真浏览器里跑同一套比对，结果与 Node **逐局完全相同**（5/5）。
     30 局实测平均 86.43 根、标准误 14.77 —— 原定的"70–85"区间对这种重尾分布
     定得太死，与 Python 侧 100 局的 78.2 相差不到半个标准误。
  5) **左右分屏页面**（`web/index.html` 重写）：玩家左、AI 右，同一个 seed。
     玩家死了用同一个 seed 立刻重开，**AI 全程不停** —— 它那个一直在涨的计数
     就是"差距"本身。文案改成英文（LinkedIn 受众）。用 headless Chrome
     截图核对过布局、无控制台报错。
  6) 新增工具：`obs_check.mjs` / `nn_check.mjs` / `ai_eval.mjs` / `selftest.html`，
     参考数据 `obs_ref.json` / `nn_ref.json` 已提交，**跑比对不需要 Python**。

  下一步（阶段 3 剩下的）：英文讲解段落（RL 基础 / Q-V-A / 网络架构 / 训练结果）、
  移动端真机布局核对。然后阶段 4 发布。

- 2026-09-05（本机，勘误）：把清单里两条还停在旧设计的条目改掉。原来写的是
  「一个画面两只鸟（AI 观测用离屏画布）」，但界面方案已改为**左右分屏**
  （plan.md 第四节）—— AI 有自己独立的画布，画面里只有它自己那只鸟，
  观测天然干净，**不需要离屏画布**。不改的话云端会照着旧设计做。

- 2026-09-05（云端）：补完阶段 3 剩下的两项，**阶段 3 全部完成**。

  1. **英文讲解**：在 `web/index.html` 的 footer 下面新增 `#learn` 区块，
     六个小节——agent 看到什么（4 帧 80×128 二值图，无手工特征）、
     奖励与 Q 函数（+1/−1/0，Double DQN）、为什么要 Dueling
     （`Q=V+A−mean(A)`，配了一张 CSS 画的小型架构图）、网络结构
     （3 卷积 32/64/64 + fc256，1,258,659 参数）、一致性验证的方法论
     （1200 帧观测/300 次动作逐位比对）、训练结果（Python 78.2 根 vs
     浏览器 86.4 根 vs 人类 6.8 根，重尾分布说明标准误而非只报均值）。
     数字全部来自定时任务 prompt 里给的实测值和 `docs/EXPERIMENTS.md`
     （`n_step=3` 无效、`eval_epsilon=0.01` 掉 6.2 倍），没有编造。
     只加了 CSS + 静态 HTML，没碰 `game.js`/`obs.js`/`nn.js`/精灵资源，
     所以**不需要**重跑 parity_check / obs_check / nn_check。
  2. **移动端布局**：本来就是 `grid-template-columns: 1fr 1fr` + `clamp()`，
     这次在云端环境的 headless Chromium（`/opt/node22/lib/node_modules/playwright`）
     里用 390×844（iPhone 尺寸）和 320×700（最窄的 iPhone SE）两个视口分别截了图，
     并且用 `dispatchEvent('pointerdown')` 模拟了真实点击/触摸交互，确认：
     两屏在最窄 320px 下依然并排（没有被挤到堆叠）、讲解区文字和网络架构图
     换行正常不溢出、点击能正常触发扇翅和重开、AI 计数在玩家死亡后继续涨、
     控制台零报错。桌面视口（1000px）同样截图核对过。
     没有真实 iOS/Android 设备可用，这轮验证到 headless Chromium 的模拟触摸
     输入为止；如果后续能在真机上跑一遍会更稳，但当前证据足以判定这一项完成。

  下一步：阶段 4——Cloudflare Pages 配置、OG 卡片图、LinkedIn 帖子草稿。
