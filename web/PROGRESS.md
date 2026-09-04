# demo/web 进度

> 这个文件是跨会话的唯一状态来源。定时任务每次启动都是冷启动，
> **先读这里**，做完一小步就更新这里并提交。不要凭记忆假设做过什么。

## 当前阶段

**阶段 1：游戏本体（JS）** — 未开始

## 阶段清单

- [ ] **阶段 1　游戏本体**
  - [x] `web/game.js`：物理与管道生成，逐条对照 `game/flappy_env.py`
  - [ ] 种子化 PRNG（两只鸟必须共用同一组管道）
  - [ ] Canvas 渲染：管道 / 小鸟 / 地面，黑底
  - [ ] 键盘 + 触摸控制，一键重开
  - [ ] **验收**：`web/tools/parity_game.md` 里记录同种子下 JS 与 Python 的管道序列比对结果
- [ ] **阶段 2　AI 上场**
  - [ ] 模型导出 ONNX fp16（需要本机 torch，见下方「阻塞项」）
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

（每次运行在这里追加一行：日期 + 做了什么 + 下一步）
