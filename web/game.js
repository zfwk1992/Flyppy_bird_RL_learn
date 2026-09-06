/**
 * FlappyGame —— 纯 JS 物理与管道生成，逐条对照 Python 侧的
 * `game/flappy_env.py`（FlappyEnv 类）与 `game/resources.py`（尺寸常量 /
 * checkCrash）。训练时实际生效的超参数来自 `flappy/config.py` 的 CONFIG
 * （randomize_pipes=True 时 flappy_env.py 自身的模块默认值只是从未被用到
 * 的回退值），本文件的默认参数对齐 CONFIG，而不是 flappy_env.py 的模块默认值。
 *
 * 本文件只负责"游戏是什么"（物理 + 管道生成 + 碰撞 + 计分），不负责渲染、
 * 不负责输入。
 * 随机源通过构造函数的 `rng` 注入（默认 Math.random），也可以只传整数
 * `seed`，内部会用下方的 `createSeededRng` 构造确定性随机源 —— 这是
 * "两只鸟共用同一组管道"的机制：给两个实例传相同的 seed 即可，见
 * `createSeededRng` 的注释。
 *
 * 坐标系与 Python 完全一致：screen 288x512，(x,y) 原点在左上角，
 * y 向下为正。BASE_Y（地面线）是浮点数，不取整，逐位对照
 * `game/resources.py: BASEY = SCREENHEIGHT * 0.79`。
 */

// ---- 屏幕与精灵尺寸（对照 game/resources.py） --------------------------
// 数值来自 assets/sprites/ 下实际 PNG 的像素尺寸，而不是猜测值。
export const SCREEN_WIDTH = 288;
export const SCREEN_HEIGHT = 512;
export const BASE_Y = SCREEN_HEIGHT * 0.79; // 404.48，故意不取整，与 Python 一致

export const PLAYER_WIDTH = 34;   // redbird-*.png
export const PLAYER_HEIGHT = 24;
export const PIPE_WIDTH = 52;     // pipe-green.png
export const PIPE_HEIGHT = 320;
export const BACKGROUND_WIDTH = 288; // background-black.png
export const BASE_IMAGE_WIDTH = 336; // base.png，用于地面滚动的 baseShift

// ---- 固定管道分布（randomize=False 时用，对照 flappy_env.py） ----------
export const GAP_Y_CHOICES = [30, 40, 50, 60, 70, 80, 90, 100];

// ---- 域随机化默认范围 ----------------------------------------------------
// 对齐 flappy/config.py 的 CONFIG（训练 models/final_v1_best.pt 时实际使用的
// 超参数），而不是 flappy_env.py 模块级的 DEFAULT_GAP_RANGE=(80,165)。
// 两者的差别只有 gap_range 下界：CONFIG 是 85，flappy_env.py 模块默认是 80。
export const DEFAULT_GAP_RANGE = [85.0, 165.0];
export const DEFAULT_SPACING_RANGE = [115.0, 200.0];
export const DEFAULT_EDGE_MARGIN = 38.0;
export const DEFAULT_MAX_DELTA_FRAC = 0.6;
export const DEFAULT_PIPE_GAP = 100; // CONFIG['pipe_gap']；randomize=True 时只是未使用的回退值

// 小鸟维持"稳住身形"锯齿振荡所需的最小竖直振幅（像素）。
// 见 flappy_env.py 顶部注释：frame_skip=4 时，每 5 次决策扇一次翅
// 净漂移最小，对应振幅 22.5px。
export const HOVER_AMPLITUDE = 22.5;

// 缝隙很窄时，即使几何上过得去，也几乎必死，理论上不该生成。
// flappy_env.py 中定义了但没有被其余逻辑实际引用，这里保留只是为了文档对照。
export const MIN_AIM_MARGIN = 8.0;

// ---- 奖励尺度（对照 flappy_env.py 的 PIPE_REWARD / DEATH_REWARD / ALIVE_REWARD）
export const PIPE_REWARD = 1.0;
export const DEATH_REWARD = -1.0;
export const ALIVE_REWARD = 0.0;

// ---- 观测尺寸（AI 阶段要用；这里只是常量占位，本文件不做降采样） --------
export const OBS_W = 80;
export const OBS_H = 128;

/** Python 的 `%` 对正除数恒返回 [0, b) 内的值；JS 的 `%` 对负数被除数会
 *  返回负值。basex 滚动的公式需要 Python 语义，这里显式修正。 */
function pymod(a, b) {
  return ((a % b) + b) % b;
}

/**
 * 32 位可复现 PRNG（mulberry32），签名与 Math.random 兼容：调用一次前进一步，
 * 返回 [0,1)。同一个 seed 永远产出同一条序列 —— 这是"两只鸟必须共用同一组
 * 管道"的基础：给两个 FlappyGame 各自传 `{ seed: sameValue }`（或各自传
 * `createSeededRng(sameValue)`），两边各自拿到独立但序列相同的生成器。
 * 管道生成完全不依赖玩家动作，只依赖帧数推进和已生成的内部状态
 * （`_nextGap` / `_lastGapCenter` / `_lastSlack`，全部由随机数派生），
 * 所以只要两只鸟的 step() 调用次数同步，管道序列就会逐位相同 ——
 * 不需要两只鸟共享同一个生成器实例，各自独立即可。
 *
 * 算法：https://github.com/bryc/code/blob/master/jshash/PRNGs.md#mulberry32
 * 不追求密码学强度，只追求「确定性 + [0,1) 均匀分布」，对游戏采样够用。
 */
export function createSeededRng(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** itertools.cycle(arr) 的等价物：每次调用推进一格，循环取值。 */
function makeCycle(arr) {
  let i = -1;
  return () => {
    i = (i + 1) % arr.length;
    return arr[i];
  };
}

/**
 * 两个矩形是否真的重叠（不只是包围盒相交），对照
 * `game/resources.py: pixelCollision`。
 *
 * mask1 / mask2 是可选的二维数组 mask[x][y] -> boolean（对应精灵 alpha 通道，
 * 见 `web/assets/hitmasks.js`，那份是从 Python 的 HITMASKS 直接导出的）。
 * 缺省时退化为纯包围盒判定 —— 但小鸟只有 72% 不透明、管道 93%，
 * 退化版会让擦边飞行误判为撞击，**正式使用必须传 hitmasks**。
 */
function pixelCollision(rect1, rect2, mask1, mask2) {
  const x1 = Math.max(rect1.x, rect2.x);
  const y1 = Math.max(rect1.y, rect2.y);
  const x2 = Math.min(rect1.x + rect1.w, rect2.x + rect2.w);
  const y2 = Math.min(rect1.y + rect1.h, rect2.y + rect2.h);
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) return false;

  if (!mask1 || !mask2) return true; // 无 hitmask 时按包围盒相交即碰撞

  const ox1 = x1 - rect1.x;
  const oy1 = y1 - rect1.y;
  const ox2 = x1 - rect2.x;
  const oy2 = y1 - rect2.y;
  for (let dx = 0; dx < w; dx++) {
    for (let dy = 0; dy < h; dy++) {
      if (mask1[ox1 + dx][oy1 + dy] && mask2[ox2 + dx][oy2 + dy]) {
        return true;
      }
    }
  }
  return false;
}

export class FlappyGame {
  /**
   * @param {object} opts
   * @param {boolean} [opts.randomize=true] 是否开启管道域随机化（训练默认开）
   * @param {number} [opts.pipeGap=DEFAULT_PIPE_GAP] randomize=false 时的固定缝隙
   * @param {[number,number]} [opts.gapRange=DEFAULT_GAP_RANGE]
   * @param {[number,number]} [opts.spacingRange=DEFAULT_SPACING_RANGE]
   * @param {number} [opts.edgeMargin=DEFAULT_EDGE_MARGIN]
   * @param {number} [opts.maxDeltaFrac=DEFAULT_MAX_DELTA_FRAC]
   * @param {number} [opts.pipeReward=PIPE_REWARD]
   * @param {number} [opts.deathReward=DEATH_REWARD]
   * @param {number} [opts.aliveReward=ALIVE_REWARD]
   * @param {() => number} [opts.rng=Math.random] 返回 [0,1) 的随机源。
   *   显式传入时优先于 `seed`。
   * @param {number|null} [opts.seed=null] 传入整数时，内部用
   *   `createSeededRng(seed)` 构造确定性随机源（`rng` 未显式给出时才生效）。
   *   两只鸟各自用相同的 seed 构造实例，即可共用同一组管道序列。
   * @param {object|null} [opts.hitmasks=null]
   *   { player: [mask0, mask1, mask2], pipeUpper: mask, pipeLower: mask }
   *   缺省时碰撞退化为包围盒判定（见 pixelCollision）。
   */
  constructor(opts = {}) {
    const {
      randomize = true,
      pipeGap = DEFAULT_PIPE_GAP,
      gapRange = DEFAULT_GAP_RANGE,
      spacingRange = DEFAULT_SPACING_RANGE,
      edgeMargin = DEFAULT_EDGE_MARGIN,
      maxDeltaFrac = DEFAULT_MAX_DELTA_FRAC,
      pipeReward = PIPE_REWARD,
      deathReward = DEATH_REWARD,
      aliveReward = ALIVE_REWARD,
      rng = null,
      seed = null,
      hitmasks = null,
    } = opts;

    this.pipeReward = pipeReward;
    this.deathReward = deathReward;
    this.aliveReward = aliveReward;

    this.pipeGap = pipeGap;
    if (this.pipeGap < PLAYER_HEIGHT + 8) {
      throw new Error(
        `pipeGap=${this.pipeGap} 对高度 ${PLAYER_HEIGHT} 的小鸟来说无法通过（至少要 ${PLAYER_HEIGHT + 8}）`
      );
    }

    this.randomize = randomize;
    this.gapRange = [gapRange[0], gapRange[1]];
    this.spacingRange = [spacingRange[0], spacingRange[1]];
    this.edgeMargin = edgeMargin;
    this.maxDeltaFrac = maxDeltaFrac;
    if (this.randomize) {
      if (this.gapRange[0] < PLAYER_HEIGHT + 8) {
        throw new Error(
          `gapRange 下界 ${this.gapRange[0]} 对高度 ${PLAYER_HEIGHT} 的小鸟无法通过（至少要 ${PLAYER_HEIGHT + 8}）`
        );
      }
      if (this.gapRange[1] + 2 * this.edgeMargin > BASE_Y) {
        throw new Error(
          `gapRange 上界 ${this.gapRange[1]} 加上下各 ${this.edgeMargin} 的余量超过了可用高度 ${BASE_Y}`
        );
      }
    }

    this._rng = rng || (seed !== null ? createSeededRng(seed) : Math.random);
    this.hitmasks = hitmasks;

    this._done = true; // 强制先 reset()
    this.reset();
  }

  // -- 随机源：全部经过这两个helper，方便未来替换成种子化 PRNG -----------
  _uniform(lo, hi) {
    return lo + this._rng() * (hi - lo);
  }

  _choice(arr) {
    return arr[Math.floor(this._rng() * arr.length)];
  }

  // ----------------------------------------------------------------------
  // 生命周期
  // ----------------------------------------------------------------------
  reset() {
    this.score = 0;
    this.frames = 0;
    this.playerIndex = 0;
    this.loopIter = 0;
    this._nextPlayerIndex = makeCycle([0, 1, 2, 1]);

    this.playerx = Math.floor(SCREEN_WIDTH * 0.2);
    this.playery = Math.floor((SCREEN_HEIGHT - PLAYER_HEIGHT) / 2) - 20;
    this.basex = 0;
    this.baseShift = BASE_IMAGE_WIDTH - BACKGROUND_WIDTH;

    // 第一根缝隙要从小鸟出生高度够得着，可达性链条从 playery 起算。
    this._lastGapCenter = this.playery + PLAYER_HEIGHT / 2.0;
    this._lastSlack = 0.0;
    this._planNext();

    const first = this._samplePipe(SCREEN_WIDTH); // 内部会再调用 _planNext()
    const second = this._samplePipe(SCREEN_WIDTH + this._nextSpacing);

    this.upperPipes = [first[0], second[0]];
    this.lowerPipes = [first[1], second[1]];

    // 物理参数：与 Python 逐位一致
    this.pipeVelX = -5;
    this.playerVelY = 0;
    this.playerMaxVelY = 5;
    this.playerMinVelY = -5; // Python 侧同样定义但未被其余逻辑引用，仅保留做对照
    this.playerAccY = 0.5;
    this.playerFlapAcc = -5;
    this.playerFlapped = false;

    this._done = false;
    return this.observeState();
  }

  /**
   * 推进一帧。
   * @param {0|1} action 0 = 不跳, 1 = 扇翅
   * @returns {{reward: number, done: boolean, info: object}}
   */
  step(action) {
    if (this._done) {
      throw new Error('step() called on a finished episode; call reset() first');
    }

    this.frames += 1;
    let reward = this.aliveReward;

    // 1. 先施加动作 —— 必须在物理更新之前
    if (action === 1) {
      if (this.playery > -2 * PLAYER_HEIGHT) {
        this.playerVelY = this.playerFlapAcc;
        this.playerFlapped = true;
      }
    }

    // 2. 物理更新（顺序与 Python 一致）
    if (this.playerVelY < this.playerMaxVelY && !this.playerFlapped) {
      this.playerVelY += this.playerAccY;
    }
    if (this.playerFlapped) {
      this.playerFlapped = false;
    }
    this.playery += Math.min(this.playerVelY, BASE_Y - this.playery - PLAYER_HEIGHT);
    if (this.playery < 0) {
      this.playery = 0;
    }

    // 3. 管道移动 + 得分判定（显式跨越测试，见 flappy_env.py 注释）
    const playerMid = this.playerx + PLAYER_WIDTH / 2.0;
    let scored = 0;
    for (let i = 0; i < this.upperPipes.length; i++) {
      const uPipe = this.upperPipes[i];
      const lPipe = this.lowerPipes[i];
      const prevMid = uPipe.x + PIPE_WIDTH / 2.0;
      uPipe.x += this.pipeVelX;
      lPipe.x += this.pipeVelX;
      const newMid = uPipe.x + PIPE_WIDTH / 2.0;
      if (newMid <= playerMid && playerMid < prevMid) {
        scored += 1;
      }
    }
    if (scored) {
      this.score += scored;
      reward += this.pipeReward * scored;
    }

    // 4. 生成 / 回收管道
    const lastUpper = this.upperPipes[this.upperPipes.length - 1];
    if (lastUpper.x <= SCREEN_WIDTH - this._nextSpacing) {
      const newPipe = this._samplePipe(lastUpper.x + this._nextSpacing);
      this.upperPipes.push(newPipe[0]);
      this.lowerPipes.push(newPipe[1]);
    }
    if (this.upperPipes[0].x < -PIPE_WIDTH) {
      this.upperPipes.shift();
      this.lowerPipes.shift();
    }

    // 5. 动画索引
    if ((this.loopIter + 1) % 3 === 0) {
      this.playerIndex = this._nextPlayerIndex();
    }
    this.loopIter = (this.loopIter + 1) % 30;
    this.basex = -pymod(-this.basex + 100, this.baseShift);

    // 6. 碰撞检测 —— 与 Python 一致：这里不内部 reset()
    if (this._checkCrash()) {
      this._done = true;
      reward += this.deathReward; // += 而非 =，同帧得分不会被吞掉
    }

    const info = {
      score: this.score,
      frames: this.frames,
      scored,
    };
    return { reward, done: this._done, info };
  }

  get done() {
    return this._done;
  }

  // ----------------------------------------------------------------------
  // 碰撞检测，对照 game/resources.py: checkCrash
  // ----------------------------------------------------------------------
  _checkCrash() {
    // 撞地
    if (this.playery + PLAYER_HEIGHT >= BASE_Y - 1) {
      return true;
    }

    // pygame.Rect 会把浮点坐标**向零截断**成整数（Math.trunc，不是 Math.floor
    // —— 负数上两者不同：-34.976 在 pygame 里是 -34）。JS 侧必须照做，否则
    //   1. 掩码索引会变成小数，mask[166.5] === undefined 直接抛错
    //   2. 即使不抛错，判定边界也会和 Python 差半个像素
    const playerRect = {
      x: Math.trunc(this.playerx), y: Math.trunc(this.playery),
      w: PLAYER_WIDTH, h: PLAYER_HEIGHT,
    };
    const pMask = this.hitmasks ? this.hitmasks.player[this.playerIndex] : null;
    const uMask = this.hitmasks ? this.hitmasks.pipeUpper : null;
    const lMask = this.hitmasks ? this.hitmasks.pipeLower : null;

    for (let i = 0; i < this.upperPipes.length; i++) {
      const uPipe = this.upperPipes[i];
      const lPipe = this.lowerPipes[i];
      const uRect = {
        x: Math.trunc(uPipe.x), y: Math.trunc(uPipe.y),
        w: PIPE_WIDTH, h: PIPE_HEIGHT,
      };
      const lRect = {
        x: Math.trunc(lPipe.x), y: Math.trunc(lPipe.y),
        w: PIPE_WIDTH, h: PIPE_HEIGHT,
      };

      if (
        pixelCollision(playerRect, uRect, pMask, uMask) ||
        pixelCollision(playerRect, lRect, pMask, lMask)
      ) {
        return true;
      }
    }
    return false;
  }

  // ----------------------------------------------------------------------
  // 管道生成，逐条对照 flappy_env.py 的 _sample_pipe / _plan_next /
  // _travel_slack / _center_range / _min_spacing_for / _sample_spacing
  // ----------------------------------------------------------------------
  _samplePipe(pipeX) {
    let gap;
    let gapTop;
    if (!this.randomize) {
      gap = this.pipeGap;
      const gapTopChoice = this._choice(GAP_Y_CHOICES) + Math.floor(BASE_Y * 0.2);
      this._lastGapCenter = gapTopChoice + gap / 2.0;
      this._lastSlack = Math.max(gap - PLAYER_HEIGHT, 0.0) / 2.0;
      gapTop = gapTopChoice;
    } else {
      gap = this._nextGap;
      const [lo, hi] = this._centerRange(gap, this._nextSpacing);
      const center = lo < hi ? this._uniform(lo, hi) : (lo + hi) / 2.0;

      this._lastGapCenter = center;
      this._lastSlack = this._travelSlack(gap);
      gapTop = center - gap / 2.0;
    }

    this._planNext();
    return [
      { x: pipeX, y: gapTop - PIPE_HEIGHT, gap }, // 上管道
      { x: pipeX, y: gapTop + gap }, // 下管道
    ];
  }

  _planNext() {
    if (!this.randomize) {
      this._nextGap = this.pipeGap;
      this._nextSpacing = SCREEN_WIDTH / 2.0;
    } else {
      this._nextGap = this._uniform(this.gapRange[0], this.gapRange[1]);
      this._nextSpacing = this._sampleSpacing(this._nextGap);
    }
  }

  _travelSlack(gap) {
    const band = Math.max(gap - PLAYER_HEIGHT, 0.0);
    return Math.max(band - HOVER_AMPLITUDE, 0.0) / 2.0;
  }

  _centerRange(gap, spacing) {
    let lo = this.edgeMargin + gap / 2.0;
    let hi = BASE_Y - this.edgeMargin - gap / 2.0;
    if (lo > hi) {
      // 缝隙比可用高度还大，退化成居中
      return [BASE_Y / 2.0, BASE_Y / 2.0];
    }
    if (this._lastGapCenter === null || this._lastGapCenter === undefined) {
      return [lo, hi];
    }

    let budget = this.maxDeltaFrac * spacing - this._lastSlack + this._travelSlack(gap);
    budget = Math.max(budget, 0.0);

    lo = Math.max(lo, this._lastGapCenter - budget);
    hi = Math.min(hi, this._lastGapCenter + budget);
    if (lo > hi) {
      const c = Math.min(
        Math.max(this._lastGapCenter, this.edgeMargin + gap / 2.0),
        BASE_Y - this.edgeMargin - gap / 2.0
      );
      return [c, c];
    }
    return [lo, hi];
  }

  _minSpacingFor(gap) {
    if (this._lastGapCenter === null || this._lastGapCenter === undefined) {
      return this.spacingRange[0];
    }
    const need = HOVER_AMPLITUDE;
    return (need + this._lastSlack - this._travelSlack(gap)) / this.maxDeltaFrac;
  }

  _sampleSpacing(gap = null) {
    if (!this.randomize) {
      return SCREEN_WIDTH / 2.0;
    }
    let [lo, hi] = this.spacingRange;
    if (gap !== null) {
      lo = Math.min(Math.max(lo, this._minSpacingFor(gap)), hi);
    }
    return this._uniform(lo, hi);
  }

  // ----------------------------------------------------------------------
  // 势能函数（供未来势能塑形 / AI 决策使用），对照 flappy_env.py: current_potential
  // ----------------------------------------------------------------------
  currentPotential() {
    let nxt = null;
    for (let i = 0; i < this.upperPipes.length; i++) {
      const uPipe = this.upperPipes[i];
      if (uPipe.x + PIPE_WIDTH > this.playerx) {
        nxt = uPipe;
        break;
      }
    }
    if (nxt === null) return 0.0;
    const gapCenter = nxt.y + PIPE_HEIGHT + nxt.gap / 2.0;
    const d = Math.abs(this.playery + PLAYER_HEIGHT / 2.0 - gapCenter) / SCREEN_HEIGHT;
    return -d;
  }

  // ----------------------------------------------------------------------
  // 调试 / 测试用的纯状态快照（不含像素观测，降采样是后续阶段的事）
  // ----------------------------------------------------------------------
  observeState() {
    return {
      score: this.score,
      frames: this.frames,
      playerx: this.playerx,
      playery: this.playery,
      playerVelY: this.playerVelY,
      playerIndex: this.playerIndex,
      basex: this.basex,
      done: this._done,
      upperPipes: this.upperPipes.map((p) => ({ x: p.x, y: p.y, gap: p.gap })),
      lowerPipes: this.lowerPipes.map((p) => ({ x: p.x, y: p.y })),
    };
  }
}
