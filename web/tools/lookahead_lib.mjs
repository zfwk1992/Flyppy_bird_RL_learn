/**
 * 前瞻模拟的共享工具：状态快照/恢复 + 追缝隙启发式。
 *
 * 给 oracle_hazard.mjs（实验 A）和 flip_on_critical.mjs（实验 D）用。
 * **不修改 web/game.js**——`FlappyGame` 的字段虽然带下划线前缀，但都是
 * 普通的实例属性（JS 没有真正的私有字段），从外面读写不算改动它的公开行为。
 *
 * 为什么不能直接用 `seed` 选项然后指望"重建一个同 seed 的新实例"来做克隆：
 * `createSeededRng(seed)` 内部的整数状态 `s` 被闭包捕获，外面读不到，
 * 也就没法在游戏进行到一半时把 RNG 的"当前状态"复制到另一个实例上。
 *
 * 解法：`FlappyGame` 的构造函数本身支持传 `rng`（一个返回 [0,1) 的函数），
 * 优先级高于 `seed`。这里自己造一个状态可读写的 RNG 包装（把内部整数状态
 * 存在一个外部可见的对象里），代数上和 `game.js: createSeededRng` 里的
 * mulberry32 逐行相同（同一个公开算法，见 game.js 里的 github 链接），
 * 只是状态从闭包变量搬到了外部对象的字段——这样克隆时只要复制一个整数。
 *
 * 克隆的正确性不是猜的，跑 lookahead_selfcheck.mjs 验证过：
 * 同一个真实对局，在某一帧原地克隆一份，之后用完全相同的动作序列分别推进
 * 原对象和克隆对象很多帧，逐帧比对所有可观察字段（位置、速度、分数、
 * 管道数组、动画帧、地面滚动、done），必须逐帧相同。
 */
import { FlappyGame } from '../game.js';

// ---------------------------------------------------------------------
// 状态可读写的 mulberry32——和 game.js: createSeededRng 算法逐行相同，
// 唯一区别是内部状态 s 存在外部对象 state.s 上，可以被复制。
// ---------------------------------------------------------------------
export function makeRngState(seed) {
  return { s: seed >>> 0 };
}

export function rngNext(state) {
  return function () {
    state.s = (state.s + 0x6d2b79f5) | 0;
    let t = state.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 用可快照的 RNG 构造一局游戏；调用方后续可以读 `game.__rngState.s`。 */
export function makeSeededGame(seed, ctorOpts = {}) {
  const rngState = makeRngState(seed);
  const game = new FlappyGame({ ...ctorOpts, rng: rngNext(rngState) });
  game.__rngState = rngState;
  return game;
}

/**
 * itertools.cycle(seq) 的等价物（对照 game.js: makeCycle），额外支持
 * "预先快进 k 次调用"——克隆时用来重建小鸟扇翅动画的下一次取值位置。
 */
function fastForwardCycle(seq, k) {
  let i = -1;
  for (let n = 0; n < k; n++) i = (i + 1) % seq.length;
  return function () {
    i = (i + 1) % seq.length;
    return seq[i];
  };
}

/**
 * 深拷贝一局游戏的当前状态到一个全新、独立的 FlappyGame 实例。
 *
 * `ctorOpts` 必须和原实例构造时用的选项一致（randomize/gapRange/...
 * /hitmasks），因为这些字段构造后不再改变，克隆时靠重新构造而不是复制。
 * 构造过程本身会跑一次 reset()（消耗一次性的占位 rng），跑完之后
 * 下面会把所有会变的字段整体覆盖掉，所以占位 rng 传什么都无所谓。
 *
 * 动画帧的重建：`playerIndex` 每 3 帧前进一格，且只由 `frames`（已经是
 * 公开字段）决定，与随机数无关，所以 `Math.floor(frames/3)` 就是目前为止
 * cycle 已经被调用的次数——不需要偷看原实例的闭包状态。
 */
export function cloneGame(game, ctorOpts) {
  if (!game.__rngState) {
    throw new Error('cloneGame 需要用 makeSeededGame() 创建的实例（要能读到 __rngState）');
  }
  const clone = new FlappyGame({ ...ctorOpts, rng: () => 0.5 });

  clone.score = game.score;
  clone.frames = game.frames;
  clone.playerIndex = game.playerIndex;
  clone.loopIter = game.loopIter;
  clone._nextPlayerIndex = fastForwardCycle([0, 1, 2, 1], Math.floor(game.frames / 3));

  clone.playerx = game.playerx;
  clone.playery = game.playery;
  clone.basex = game.basex;
  clone.baseShift = game.baseShift;

  clone._lastGapCenter = game._lastGapCenter;
  clone._lastSlack = game._lastSlack;
  clone._nextGap = game._nextGap;
  clone._nextSpacing = game._nextSpacing;

  clone.upperPipes = game.upperPipes.map((p) => ({ ...p }));
  clone.lowerPipes = game.lowerPipes.map((p) => ({ ...p }));

  clone.pipeVelX = game.pipeVelX;
  clone.playerVelY = game.playerVelY;
  clone.playerMaxVelY = game.playerMaxVelY;
  clone.playerMinVelY = game.playerMinVelY;
  clone.playerAccY = game.playerAccY;
  clone.playerFlapAcc = game.playerFlapAcc;
  clone.playerFlapped = game.playerFlapped;
  clone._done = game._done;

  const rngState = { s: game.__rngState.s };
  clone.__rngState = rngState;
  clone._rng = rngNext(rngState);

  return clone;
}

/**
 * 追缝隙中心的朴素 bang-bang 启发式，逐行对照
 * `flappy/diagnostics.py: _chase_gap` 与 `web/tools/dump_python_trace.py: chase_gap`。
 * 不追求成绩，只求确定性地覆盖真实游戏状态。
 */
export function chaseGap(g, { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT }) {
  let nxt = null;
  for (let i = 0; i < g.upperPipes.length; i++) {
    const u = g.upperPipes[i];
    if (u.x + PIPE_WIDTH > g.playerx) { nxt = u; break; }
  }
  if (nxt === null) return 0;
  const center = nxt.y + PIPE_HEIGHT + nxt.gap / 2.0;
  return (g.playery + PLAYER_HEIGHT / 2.0 > center) ? 1 : 0;
}

/**
 * 前瞻安全检查器（实验 A/D 共用）：对 `firstAction` 做一次
 * "执行该动作 + 之后按启发式 rollout 至多 N-1 步" 的模拟，
 * 返回 N 步内是否活下来（true=安全）。
 *
 * 每次调用都会克隆一份全新状态，不会污染传入的 `game`。
 */
export function survivesLookahead(game, firstAction, N, ctorOpts, dims) {
  const g = cloneGame(game, ctorOpts);
  if (g.step(firstAction).done) return false;
  for (let i = 1; i < N; i++) {
    if (g.step(chaseGap(g, dims)).done) return false;
  }
  return true;
}

/**
 * 前瞻 oracle 的决策函数：两个动作都做安全检查，
 * 都安全就退回启发式的选择；只有一个安全就选那个；
 * 两个都不安全（N 步内必死无法避免）就退回启发式（死马当活马医）。
 */
export function oracleAction(game, N, ctorOpts, dims) {
  const heuristic = chaseGap(game, dims);
  const safe0 = survivesLookahead(game, 0, N, ctorOpts, dims);
  const safe1 = survivesLookahead(game, 1, N, ctorOpts, dims);
  if (safe0 && safe1) return heuristic;
  if (safe0) return 0;
  if (safe1) return 1;
  return heuristic;
}
