# demo/web 进度

> 这个文件是跨会话的唯一状态来源。定时任务每次启动都是冷启动，
> **先读这里**，做完一小步就更新这里并提交。不要凭记忆假设做过什么。

## 当前阶段

**阶段 1：完成。** 下一步进阶段 2（AI 上场）——
模型权重和碰撞掩码都已备好，云端不需要 torch。

## 阶段清单

- [x] **阶段 1　游戏本体**
  - [x] `web/game.js`：物理与管道生成，逐条对照 `game/flappy_env.py`
  - [x] 种子化 PRNG（两只鸟必须共用同一组管道）
  - [x] Canvas 渲染：管道 / 小鸟 / 地面，黑底
  - [x] 键盘 + 触摸控制，一键重开（`web/index.html`，秒重开无停顿）
  - [x] **验收**：见 `web/tools/parity_game.md` —— 1200 帧 / 144 次抽样 / 5 回合逐位一致（PARITY OK）
- [ ] **阶段 2　AI 上场**
  - [x] 模型导出 ONNX fp16 —— `web/model/flappy_dqn_fp16.onnx`（2.53 MB，本机已导出并提交）
  - [ ] 离屏画布 + 覆盖判定降采样到 80×128 + 4 帧栈
  - [ ] 浏览器推理
  - [ ] **验收**：同种子同初始状态下动作序列与 Python 一致；浏览器跑 30 局平均落在 70–85
- [ ] **阶段 3　同台竞技 + 讲解**
  - [ ] 一个画面两只鸟（AI 观测用离屏画布，不能含玩家的鸟）
  - [ ] 计分板：你的本局 / 你的最佳 / AI 本局
  - [ ] 英文讲解：RL 基础 / Q-V-A / 网络架构 / 训练结果
  - [ ] 移动端布局
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
- **模型**：`web/model/flappy_dqn_fp16.onnx`，输入 `obs` float32 `(1,4,80,128)`
  值域 **[0,1]**（不是 0/255），输出 `q` `(1,2)`。已验证 300 组 argmax 与
  PyTorch 完全一致。
- **界面是左右分屏**（plan.md 第四节已更新）：玩家一屏、AI 一屏，同一个 seed
  所以管道相同。AI 有独立画布，观测天然干净，**不需要离屏画布**。


- **ONNX 导出需要 torch**，云端环境未必装得上。如果装不上，
  跳过阶段 2 的导出，先把阶段 1、3 的非 AI 部分做完，并在这里记下来。

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
