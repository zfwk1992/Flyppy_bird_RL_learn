# demo/web 进度

> 这个文件是跨会话的唯一状态来源。定时任务每次启动都是冷启动，
> **先读这里**，做完一小步就更新这里并提交。不要凭记忆假设做过什么。

## 当前阶段

**plan.md 里能由 agent 做的部分全部完成**（阶段 1～4）。剩下的是需要人做的事：
Cloudflare Pages 控制台的实际部署操作（agent 没有账号访问权限，`web/DEPLOY.md`
是给人看的清单）、`web/LINKEDIN_POST.md` 草稿的语气润色、拿到真实部署 URL 后
替换帖子里的占位符、真机走一遍触摸体验。**如果冷启动看到这行，先去确认这几件
"只能人做"的事有没有进展，而不是重新翻 plan.md 找活干——没有代码活了。**

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
  - [x] Cloudflare Pages 配置：`web/_headers`（缓存策略）+ `web/DEPLOY.md`
        （dashboard 里要填的字段，重点是 build output directory 必须设成
        `web`，否则会把 `flappy/`/`models/` 等无关的训练代码也发布出去）
  - [x] OG 卡片图：`web/assets/og-card.jpg`（1200×630，78.5KB），源文件
        `web/tools/og_card_source.html` + 渲染脚本
        `web/tools/render_og_card.mjs`，`index.html` 已加 og:/twitter: 元标签
  - [x] LinkedIn 帖子草稿：`web/LINKEDIN_POST.md`（正文 + 第一条评论 + 话题标签
        + 发布检查表）。**这是草稿**，`plan.md` 第九节仍然标注需要人工润色，
        发布前还要把占位符 URL 换成真实部署地址。

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
- **改完 `web/ai-worker.js`、`web/ai.js` 或动了动作计划的收发逻辑，必须重跑**：
  ```
  node web/tools/worker_check.mjs          # 预跑回放 vs 直接推理，逐帧
  ```
  必须 `WORKER PARITY OK`。这条守的是 worker 方案的**全部合法性前提** ——
  "AI 那一局与玩家操作无关，所以可以提前算"。前提一旦被破坏（比如让 AI 的
  观测里混进玩家那只鸟、或 `AiPlayer` 引入跨局残留状态），表现出来**不是报错，
  而是 AI 悄悄变笨**，正是这个项目历史上最难查的那一类 bug。需要本地起服务：
  `python -m http.server 8123 --directory web`。
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
- **验移动端布局用 `node web/tools/mobile_check.mjs`，不要用 `--window-size` 截图。**
  `--window-size` 设的是窗口尺寸，不是移动端布局视口 —— `width=device-width`
  不会按手机的方式生效，dpr 也还是 1，截出来的图会把一个**完全正常**的响应式
  页面拍成"右边被裁掉"的样子。这个坑已经踩过一次，差点去修一个不存在的 bug。
  `mobile_check.mjs` 走 CDP 的 `Emulation.setDeviceMetricsOverride`（`mobile:true`
  + 真实 dpr），量的是 `scrollWidth` 和每个元素的 right 边界，**给可判定的结论**，
  不用靠人眼看截图猜；顺带收集页面自身抛的 JS 错。零依赖，不需要 Playwright。
- **界面是左右分屏**：同一个 seed、同一帧起跑，所以两边跑的是同一根管道。
  **两边始终同帧推进**（一方的计划没算出来就整体等一拍，绝不让玩家单独往前走）——
  一旦帧数错开就不在同一根管道上了，左右对照立刻失效。
  **但玩家撞了之后 AI 不停**：这一局对你结束、计数照常结算，AI 继续飞，
  你犹豫的那几秒能看着它的计数往上爬。下一局由你点击触发，届时两边**一起**
  用新 seed 重开。这是 `plan.md` 第八节定的"玩家死后 AI 继续飞，让差距可见"；
  中间一版为了对照把它改成了"谁先撞两边同时归零"，代价是 AI 的分数全程等于
  你的、"它比你强"从数字上根本看不出来 —— 2026-09-06 改回来了，对照和差距两头都保住。
  **每局必须换 seed**：AI 是确定性的（eval_epsilon=0、无 Dropout/BatchNorm），
  seed 不变它就会逐帧重放同一条轨迹、死在同一根管道上。第一版整个会话共用一个
  seed，抽到一条早死的序列就会看着它一遍遍死在第 12 根，像是模型坏了。
  两边同时活着的那段，AI 的当前分数必然等于玩家的（同一组管道、鸟的 x 也相同），
  所以差距是在**你掉下去之后**才显现的，另外由 `outlasted you`（你先掉下去的
  局数）和讲解区里的 78.2 / 6.8 承担。

## 变更日志

- 2026-09-05（本机，复核云端阶段 3/4 的产出）：四条 parity + 浏览器自检全部重跑
  通过，没有退化。改了三处**事实错误**的英文文案 —— 都是会被同行一眼看出来的：
  1) "lowering eval epsilon to 0.01 dropped the score 6.2×" **方向反了**。
     实测是 ε=0 得 389.6 根、ε=0.01 只得 63.1 根（`docs/learn/07-exploration.md`），
     所以是**留着** 1% 随机才掉 6.2 倍，不是"降到 0.01"。
  2) "loads these weights (fp16, 2.52 MB) into a hand-written **~2.5 MB JavaScript**
     forward pass" —— 2.52 MB 是权重，不是那段 JS（`web/nn.js` 只有几 KB）。
     改成"没有 ONNX 运行时，所以整个下载量就是权重本身"。
  3) "raw pixels, the same **rendering pipeline** that draws the canvas on the right"
     —— AI 根本不读画布，`web/obs.js` 是解析式重建的，这句和 `parity_ai.md`
     自相矛盾。改成"same sprites and geometry"。
     `LINKEDIN_POST.md` 里同样的 2.5 MB 歧义也一并改了。
  其余数字逐条核对过**都是真的**：1,258,659 参数、1.3 根 / 10 万局、75% 死亡信号
  被丢、目标网络卡在 train 模式、奖励尺度撑爆 Huber、n_step=3 横盘、
  78.2 / 86.4 / 6.8、28 ms vs 133 ms 预算 —— 均有 `docs/` 出处。
  `demo_wechat_30s.mp4` 确实存在（1.15 MB）。
  新增 `web/tools/mobile_check.mjs`：起因是我先用 `chrome --window-size=390,844`
  截图，看到"内容被右边裁掉"，差点判定移动端布局是坏的。**是截图方法错了** ——
  `--window-size` 不构成移动端布局视口。换成 CDP 的 `setDeviceMetricsOverride`
  （`mobile:true` + 真实 dpr）之后，320 / 390 / 414 三档 `scrollWidth` 全部
  等于 `innerWidth`、零元素越界、零 JS 报错，云端那句"移动端已验证"是对的。
  这个工具给出可判定结论而不是让人看截图猜，省得下次再来一遍。

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

- 2026-09-05（云端）：阶段 4 做了两项——Cloudflare Pages 配置和 OG 卡片图。

  1. **Cloudflare Pages 配置**：新增 `web/_headers`（Pages 的响应头约定文件，
     文件名不能改）——`/model/*` 和 `/assets/*` 给一天的强缓存，`/index.html`
     和 `/` 不缓存以便随时改文案，外加两条基本安全头。另外写了
     `web/DEPLOY.md`，记录 dashboard 里要手填的字段。**这里没有"配置文件
     一键部署"这回事**——Cloudflare Pages 连 GitHub 仓库走的是网页向导，
     没有等价于 `vercel.json` 的、提交进仓库就能生效的项目配置文件，所以
     `DEPLOY.md` 是给操作的人看的清单，不是可执行配置。**最关键的一条**：
     build output directory 必须设成 `web`，因为仓库根目录还有
     `flappy/`/`models/`/`train.py` 这些和站点无关的训练代码，不隔离会被
     一起发布出去。
  2. **OG 卡片图**：`web/tools/og_card_source.html`（1200×630 的独立静态页，
     不被 `index.html` 引用，只用来截图）+ `web/tools/render_og_card.mjs`
     （headless Chromium 截图脚本，云端用 `/opt/node22/lib/node_modules/playwright`
     的绝对路径导入，因为 ESM 的 bare specifier 解析不认 `NODE_PATH`，
     云端也没有本地 npm 依赖）。产出 `web/assets/og-card.jpg`（78.5KB），
     配色和站点一致（深色背景 + 淡化的管道剪影装饰），核心信息是
     "78.2 vs 6.8" 的数字对比。`index.html` 的 `<head>` 加了 og:/twitter:
     系列元标签，`og:image` 用相对路径——域名还没定（Cloudflare Pages 建好
     项目才会分配 `*.pages.dev`），`DEPLOY.md` 里记了"部署后要用 LinkedIn
     Post Inspector 验一遍相对路径能不能被正确解析，不行就换绝对 URL"。

  下一步：阶段 4 最后一项——LinkedIn 帖子草稿。做完这个 plan.md 就全部完成了。

- 2026-09-05（云端）：阶段 4 最后一项——**LinkedIn 帖子草稿**，`plan.md`
  里 agent 能做的部分至此全部完成。

  新增 `web/LINKEDIN_POST.md`：正文（钩子句 + agent 看到什么/优化什么 + 为什么
  用 Dueling 架构 + 78.2/86.4/6.8 三个结果数字 + "曾经卡在 1.3 根、问题出在
  数据管线不在算法"的工程故事 + CTA）、第一条评论（demo 链接占位符 + GitHub
  仓库链接，故意不放进正文——`plan.md` 第一节定的策略是 LinkedIn 算法会压低
  带外链帖子的曝光）、话题标签、发布前检查表。数字和技术细节全部对得上
  `web/index.html` 的 `#learn` 区块和这次任务 prompt 里给的实测数据，没有编造。
  同时在 `plan.md` 第九节记了两件还需要**人**做的事：帖子文案润色、
  Cloudflare Pages 的实际部署操作（agent 没有控制台访问权限）。

  **`plan.md` 里所有能由 agent 完成的条目到此为止全部做完。** 剩下全是需要
  人工操作的部分（部署到 Cloudflare、买不买域名、文案润色、真机测试、
  实际发帖）——这些不在 agent 的能力范围内，冷启动后不用再重新扫一遍
  plan.md 找代码活了；如果这几件人工事项之后又衍生出新的代码需求
  （比如部署后发现某个真机 bug），到时候再处理。

- 2026-09-06（本机）：**把推理搬进 Web Worker，修掉手机上的卡顿**；顺带按
  `plan.md` 第八节把"玩家死后 AI 继续飞"改了回来。核心模块（`game.js` /
  `obs.js` / `nn.js` / `ai.js` / 精灵 / 权重）**一行没动**，改的只有
  `index.html`，新增 `web/ai-worker.js` 和 `web/tools/worker_check.mjs`。

  **病因**：推理原来同步跑在主线程的 rAF 回调里。桌面 45 ms / 预算 133 ms
  很宽裕，所以一直没暴露；但用 CDP 的 CPU 节流量了一遍真机档位，中端手机
  178 ms、低端 276 ms —— **超预算**，整个页面被堵住。实测渲染帧率：

  | 档位 | 改之前 | 改之后 |
  |---|---|---|
  | 桌面（无节流） | 49.8 fps | 60 fps |
  | 中端手机（4x CPU） | **9.5 fps** | **60 fps** |
  | 低端手机（6x CPU） | **6.3 fps** | **51.4 fps** |

  对一个要发 LinkedIn（流量以手机为主）的 demo，这是发布前必须修的。
  注意这跟托管、CDN、域名都无关，纯粹是 JS 前向在手机 CPU 上就是这么慢。

  **做法**：`web/ai-worker.js` 按 seed 预跑出**逐帧动作**，主线程只回放。
  能这么做是因为 **AI 那一局与玩家操作完全无关** —— `aiGame` 是独立实例，
  管道只由 seed 决定。同一个 seed + 同一串动作喂进同一份 `game.js`，轨迹
  **逐位一致**，所以**没有削弱 AI**（`plan.md` 第八节的硬约束），只是换了线程。
  这一点由新增的 `worker_check.mjs` 钉死：3 个 seed、约 600 帧，
  worker 回放与主线程直接推理**逐帧全等**，末状态与分数也相等。

  两个踩过的坑，都记下来免得重来：
  1. **每局开头会卡一下**。第一版每开一局 worker 都从零算，计划是空的，
     主线程只能干等。加了**预取**：玩家一撞就让 worker 去算下一局的开头，
     你看结算那一两秒正好抢出缓冲。加之前测出 73～83 次停顿，加之后
     **三档全是 0 次**。
  2. **CDP 的 `setCPUThrottlingRate` 不节流 worker 线程。** 上表里"改之后"
     的 worker 推理耗时三档都是 38～45 ms，就是这个原因。所以真机上
     低端设备的**计划生产速度**仍可能跟不上 30 fps（估算 6x 档约 58% 速度）——
     主线程卡顿这个大头已经解决（渲染和输入完全跟手），但如果真机上还嫌慢，
     下一步是优化 `nn.js` 的卷积（im2col + 循环重排，或利用二值输入的稀疏性），
     那属于改 `nn.js`，**必须重跑 `nn_check`**。

  **还顺手修的两处**：
  - 首屏不再空白等 2.5 MB。对战区立刻显示，加载进度画在玩家那侧的遮罩里
    （慢 4G 实测要 15 秒，纯空白是真实的跳出损耗）。
  - `pointerdown` 从只绑玩家那半边改成绑整个 `#arena`。手机上单边只有约
    160 px 宽，太容易点空。

  验证：四条原有 parity + `mobile_check` + 新增 `worker_check` + 浏览器
  `selftest.html` **全部通过**，无退化，控制台除 favicon 404 外无报错。

  下一步：仍然是那几件只能人做的事（Cloudflare Pages 实际部署、帖子润色、
  真机走一遍）。真机测完如果低端安卓仍偏慢，再考虑优化 `nn.js`。

- 2026-09-06（本机，真机反馈修 bug）：**玩家死后 AI 飞一两根就卡在半空** ——
  用户在手机上实测发现的，headless 六项检查全绿也没抓到。

  **病因**是我自己前一条改动里的一个错误假设。当时的逻辑是：玩家一撞就让
  worker 掉头去预跑下一局，当前这一局的计划不再往前补，注释里写着"死的时候
  手上还有约 LOOKAHEAD 帧（20 秒）的余量"。**那个假设只在开发机上成立。**
  真机上 worker 一次决策约 150~270 ms，而实时消耗要 133 ms/次，缓冲区常年
  接近空，所以玩家一死 AI 就停在一两根管子之后。

  **为什么六项检查全没测出来**：CDP 的 `setCPUThrottlingRate` **不节流 worker
  线程**——这一条我上一条日志里明明记下来了，却没把它和这个失败模式联系起来。
  测试里 worker 全速跑，缓冲区一直很厚，于是"AI 续飞"每次都通过。
  **教训**：这个工具链能测主线程卡顿，测不了计划生产速度跟不跟得上。凡是
  依赖 worker 产出速率的行为，headless 的结论都不作数。

  **修法**：优先保证当前这局的计划不断流（`stepBoth` 里续要计划的条件去掉
  `youAlive`），预取推迟到 **AI 也撞了之后**再做——那时 worker 才真的空出来。
  代价是玩家在 AI 还活着时点"再来一局"会有一次几百毫秒的启动等待，
  比起 AI 卡死在半空，这个代价可以接受。

  **A/B 实测**（新增 scratchpad 探针，盯的是"玩家死后 plan 还增不增长"）：

  | | 计划帧数 | AI 多飞了 |
  |---|---|---|
  | 修之前（线上那版） | 224 -> **224，冻结** | 6 根后**完全停住** |
  | 修之后 | 288 -> **976，持续增长** | 14 根且仍在飞（30.1 fps，实时） |

  顺带确认了另一件**不是 bug** 的事：用户报告"AI 飞 3 个管子就没了"。
  `ai_eval.mjs` 跑 12 局：平均 103.9 根、中位 88，但**最低 3、最高 290**。
  分布极度重尾，12 局里有 1 局死在第 3 根（约 8%）。换模型没换错，
  这是真实行为。但对 demo 是个体验风险：访客第一局要是抽到这种 seed，
  "AI 很强"的印象就没了。**这一条还没解决，留给后面决定**（可选做法：
  开局第一局用一个预筛过的好 seed，之后再随机——但那要小心别变成作弊）。

  验证：四条 parity + `worker_check` + `mobile_check` 全过。
---

## 批次进度（定时 routine 用，勿手改格式）

`web_plan.md` §7 把加固计划切成 B1–B10。每轮 routine 冷启动后读这张表，
认领第一个状态不是"已完成"的批次，**只做那一块**。

认领方式：把该行状态改成 `进行中 <UTC 时间戳>` 并立刻 commit + push，
做完再改成 `已完成` / `部分完成（卡在…）` / `跳过（原因）`。

| 批次 | 内容 | 状态 | 备注 |
|---|---|---|---|
| B1 | 玩家死后 AI 卡住 | 待办 | 根因已定位到行，见 web_plan.md §1 |
| B2 | 管道不显示：写复现脚本 | 待办 | 只复现，不许改渲染代码 |
| B3 | 管道不显示：修 | 待办 | 依赖 B2；B2 没复现就跳过 |
| B4 | 设备矩阵 7 视口 | 待办 | |
| B5 | 误操作用例 1–4 | 待办 | |
| B6 | 误操作用例 5–7 + TESTING.md | 待办 | |
| B7 | 操作提示按输入方式区分 | 待办 | 依赖 B4 |
| B8 | 训练循环图 | 待办 | |
| B9 | 页面数字核对 | 待办 | 放最后，前面可能改数字 |
| B10 | 全量巡检 + 总结 | 待办 | |
