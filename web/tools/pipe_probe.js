/**
 * 管道渲染探针 —— 只给 `web/tools/pipe_render_check.mjs` 用。
 * **页面本身不加载它**（index.html 里没有任何引用），它是被 CDP 在页面上下文里
 * `await import('/tools/pipe_probe.js')` 拉进去的，所以能直接 import 真正在跑的
 * `../game.js` / `../render.js` / `../assets/hitmasks.js`，不是复制一份对照实现。
 *
 * 为什么要有这么一个东西
 * ----------------------
 * 用户报的是"玩了几次之后有一些 pipe 没有正常显示"。这句话有两种完全不同的成因：
 *   (甲) 状态里就少了 —— `game.js` 的生成/回收出问题，画得再对也没有管道；
 *   (乙) 状态是对的、画面上少了 —— `render.js` 或精灵解码出问题。
 * 这个探针的唯一任务是把两者**分开**：它拿同一帧的 `observeState()` 和
 * 同一帧画布的像素做逐根对拍，答案要么是"状态里少"，要么是"画面上少"，
 * 要么是"两边都对、没能复现"。**在分清之前不许改任何渲染代码。**
 *
 * 怎么判"这根管道画出来了"
 * ------------------------
 * 不用"数绿色连通块"这种拍脑袋的判据，理由有二：
 *   1. 管道 x 是**浮点数**（spacing 来自 `_uniform`），`drawImage` 会做双线性
 *      重采样，边缘像素是管道色和背景色的混色，按固定色值精确匹配必然误判；
 *   2. 小鸟画在管道**上面**，它遮住的那几十个像素不是"漏画"。
 * 改成两步：
 *   - **颜色分类器**：把四张精灵（背景/管道/地面/小鸟）各自的调色板抠出来，
 *     量化到 5bit/通道，建一张 32768 项的最近邻查找表，把任意像素判成
 *     背景/管道/地面/小鸟之一。混色像素落在最近的那一类，两类都出现过的
 *     颜色标成"歧义"并且**永不**被当成管道证据。
 *   - **期望掩码**：管道精灵自己的 alpha ∧ 颜色属于管道类，再**腐蚀一圈**，
 *     只保留内部像素。这样重采样带来的边缘混色不参与判定。
 * 判据就是覆盖率 = 命中/期望。画对了 ≈ 1.0，整根没画 = 0。
 *
 * 反向证据（"多画了"）也查：把所有管道的**几何**区域（精灵的 alpha 掩码、膨胀 2px）
 * 加上小鸟框盖成一张掩码，掩码之外还出现管道色像素就是幽灵管道。
 * 注意这里用的是几何掩码而不是上面那张颜色掩码，理由见 pipeMask() 的注释。
 *
 * 自带负向对照
 * ------------
 * `sabotage` 参数能在指定帧故意**漏画**或**挪偏**一根管道（只动传给
 * `Renderer.draw()` 的那份快照，游戏状态不动）。检查脚本会跑这两局，
 * 要求探针当场报红 —— 不然"十局全绿"只能证明脚本没在干活。
 */
import {
  FlappyGame,
  SCREEN_WIDTH as W,
  SCREEN_HEIGHT as H,
  BASE_Y,
  PIPE_WIDTH as PW,
  PIPE_HEIGHT as PH,
  PLAYER_WIDTH,
  PLAYER_HEIGHT,
} from '../game.js';
import { loadSprites, Renderer } from '../render.js';
import { HITMASKS } from '../assets/hitmasks.js';

// 颜色类别。AMBIG = 这个颜色在两张以上精灵里都出现过，不能作为任何一方的证据。
const BG = 0, PIPE = 1, BASE = 2, PLAYER = 3, AMBIG = 255;

// 地面画在管道之上，地面线以下的管道像素本来就看不见 —— 判定一律只看这条线以上。
const Y_LIMIT = Math.floor(BASE_Y) - 2;

// barsOf() 上下两条带子的高度：要比小鸟高（PLAYER_HEIGHT=24），
// 这样小鸟无论停在哪儿都不可能把一整条带子占满。
const BAND = 31;

const q5 = (r, g, b) => (((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3));

let S = null; // install() 之后的共享状态

function scratch(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return [c, c.getContext('2d', { willReadFrequently: true })];
}

/** 把一张精灵按原尺寸画出来，收集所有不透明像素的量化颜色。 */
function paletteOf(img, w, h) {
  const [, g] = scratch(w, h);
  g.clearRect(0, 0, w, h);
  g.drawImage(img, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h).data;
  const set = new Set();
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 200) set.add(q5(d[i], d[i + 1], d[i + 2]));
  return set;
}

/**
 * 管道精灵的两张掩码，**用途不同，不能混用**：
 *   - `alpha`：纯几何，只看 alpha > 200。管道在画布上合法占用的地盘就是它，
 *     幽灵检测（"这里不该有管道色"）必须用这一张。
 *   - `eroded`：alpha ∧ 颜色被判成管道类 ∧ 八邻域也都是（腐蚀一圈）。
 *     覆盖率（"这里应该有管道色"）用它，腐蚀是为了避开重采样的边缘混色。
 *
 * 第一版把 `alpha` 也做成颜色过滤过的，结果几乎每帧都报几十个"幽灵像素"：
 * 管道调色板里有 3 个颜色和地面/小鸟撞了色（判成歧义、被排除），这些像素
 * 在掩码里是 0，可它们画出来之后因为 x/y 是小数、被双线性混成了邻近的
 * **纯管道色**，于是"掩码说这儿没有、画面上是管道色" —— 全是假阳性。
 * 教训：几何和颜色是两件事，别用同一张掩码。
 */
function pipeMask(img, lut) {
  const [, g] = scratch(PW, PH);
  g.clearRect(0, 0, PW, PH);
  g.drawImage(img, 0, 0, PW, PH);
  const d = g.getImageData(0, 0, PW, PH).data;
  const alpha = new Uint8Array(PW * PH);
  const solid = new Uint8Array(PW * PH);
  for (let i = 0, n = 0; n < PW * PH; n++, i += 4) {
    alpha[n] = d[i + 3] > 200 ? 1 : 0;
    solid[n] = (alpha[n] && lut[q5(d[i], d[i + 1], d[i + 2])] === PIPE) ? 1 : 0;
  }
  const er = new Uint8Array(PW * PH);
  for (let y = 1; y < PH - 1; y++) {
    for (let x = 1; x < PW - 1; x++) {
      let all = 1;
      for (let dy = -1; dy <= 1 && all; dy++) {
        for (let dx = -1; dx <= 1; dx++) if (!solid[(y + dy) * PW + (x + dx)]) { all = 0; break; }
      }
      er[y * PW + x] = all;
    }
  }
  return { alpha, eroded: er };
}

/**
 * 装配探针。返回的诊断信息里带几个**必须由调用方检查**的量：
 * 分类器要是把管道色和背景色搅在一起，后面所有"覆盖率"都是废数。
 */
export async function install() {
  if (S) return S.diag;
  const sprites = await loadSprites();

  const pal = {
    [BG]: paletteOf(sprites.background, W, H),
    [PIPE]: paletteOf(sprites.pipeLower, PW, PH),
    [BASE]: paletteOf(sprites.base, sprites.base.width, sprites.base.height),
    [PLAYER]: new Set(),
  };
  for (const p of sprites.player) for (const c of paletteOf(p, PLAYER_WIDTH, PLAYER_HEIGHT)) pal[PLAYER].add(c);

  // 同一个量化颜色被多类占用 -> 歧义，谁都不能拿它当证据
  const owner = new Map();
  for (const cls of [BG, PIPE, BASE, PLAYER]) {
    for (const c of pal[cls]) owner.set(c, owner.has(c) && owner.get(c) !== cls ? AMBIG : cls);
  }
  const keys = Int32Array.from(owner.keys());
  const kcls = Uint8Array.from(keys, (k) => owner.get(k));
  const kr = Int32Array.from(keys, (k) => ((k >> 10) & 31) * 8 + 4);
  const kg = Int32Array.from(keys, (k) => ((k >> 5) & 31) * 8 + 4);
  const kb = Int32Array.from(keys, (k) => (k & 31) * 8 + 4);

  // 32768 项最近邻查表：建一次，之后每像素一次数组索引就分好类了
  const lut = new Uint8Array(32768);
  for (let i = 0; i < 32768; i++) {
    const r = ((i >> 10) & 31) * 8 + 4, g = ((i >> 5) & 31) * 8 + 4, b = (i & 31) * 8 + 4;
    let best = Infinity, bc = AMBIG;
    for (let j = 0; j < keys.length; j++) {
      const dr = kr[j] - r, dg = kg[j] - g, db = kb[j] - b;
      const dd = dr * dr + dg * dg + db * db;
      if (dd < best) { best = dd; bc = kcls[j]; }
    }
    lut[i] = bc;
  }

  const mUpper = pipeMask(sprites.pipeUpper, lut);
  const mLower = pipeMask(sprites.pipeLower, lut);
  const [canvas, ctx] = scratch(W, H);

  S = {
    sprites, lut, mUpper, mLower, canvas, ctx,
    renderer: new Renderer(ctx, sprites),
    cover: new Uint8Array(W * H),
    diag: {
      paletteSizes: { bg: pal[BG].size, pipe: pal[PIPE].size, base: pal[BASE].size, player: pal[PLAYER].size },
      // 管道调色板里有多少被判成歧义 —— 太多就说明分类器不可用
      pipeAmbiguous: [...pal[PIPE]].filter((c) => owner.get(c) === AMBIG).length,
      maskPixels: { upper: mUpper.eroded.reduce((a, b) => a + b, 0), lower: mLower.eroded.reduce((a, b) => a + b, 0) },
      spriteSize: { pipe: [sprites.pipeLower.width, sprites.pipeLower.height], base: [sprites.base.width, sprites.base.height] },
      yLimit: Y_LIMIT,
    },
  };
  return S.diag;
}

/** 简单自动驾驶：只为把局跑长一点、多生成几根管道，与 AI 策略无关。 */
function pilot(g) {
  let nxt = null;
  for (const u of g.upperPipes) if (u.x + PW > g.playerx) { nxt = u; break; }
  if (!nxt) return 0;
  const center = nxt.y + PH + nxt.gap / 2;
  return (g.playery + PLAYER_HEIGHT / 2) > center + 12 ? 1 : 0;
}

/** 一根管道的覆盖率。exp = 期望像素数，hit = 实际是管道色的像素数。 */
function coverOne(p, mask, st, data) {
  const x0 = Math.max(0, Math.ceil(p.x) + 1);
  const x1 = Math.min(W - 1, Math.floor(p.x + PW) - 1);
  const y0 = Math.max(0, Math.ceil(p.y) + 1);
  const y1 = Math.min(Y_LIMIT, Math.floor(p.y + PH) - 1);
  // 小鸟画在管道之上，它盖住的像素不算漏画（±2 是给重采样留的边）
  const bx0 = Math.trunc(st.playerx) - 2, bx1 = Math.trunc(st.playerx) + PLAYER_WIDTH + 2;
  const by0 = Math.trunc(st.playery) - 2, by1 = Math.trunc(st.playery) + PLAYER_HEIGHT + 2;
  let exp = 0, hit = 0;
  for (let Y = y0; Y <= y1; Y++) {
    const dy = Math.round(Y - p.y);
    if (dy < 0 || dy >= PH) continue;
    const inBirdRow = Y >= by0 && Y <= by1;
    for (let X = x0; X <= x1; X++) {
      if (inBirdRow && X >= bx0 && X <= bx1) continue;
      const dx = Math.round(X - p.x);
      if (dx < 0 || dx >= PW) continue;
      if (!mask[dy * PW + dx]) continue;
      exp++;
      const o = (Y * W + X) * 4;
      if (S.lut[q5(data[o], data[o + 1], data[o + 2])] === PIPE) hit++;
    }
  }
  return { exp, hit, cov: exp ? hit / exp : 1 };
}

/** 把一根管道的几何掩码（膨胀 2px）盖进覆盖图，用于幽灵像素检测。 */
function stamp(p, mask, map) {
  const x0 = Math.max(0, Math.floor(p.x) - 2), x1 = Math.min(W - 1, Math.ceil(p.x + PW) + 2);
  const y0 = Math.max(0, Math.floor(p.y) - 2), y1 = Math.min(H - 1, Math.ceil(p.y + PH) + 2);
  for (let Y = y0; Y <= y1; Y++) {
    for (let X = x0; X <= x1; X++) {
      const dx = Math.round(X - p.x), dy = Math.round(Y - p.y);
      const cx = Math.min(PW - 1, Math.max(0, dx)), cy = Math.min(PH - 1, Math.max(0, dy));
      if (dx >= -2 && dx < PW + 2 && dy >= -2 && dy < PH + 2 && mask[cy * PW + cx]) map[Y * W + X] = 1;
    }
  }
}

/** 逐帧对拍：状态里可见的每一根 vs 画布上那一根。 */
function judgeFrame(st, data) {
  const rows = [];
  const map = S.cover;
  map.fill(0);
  for (let i = 0; i < st.upperPipes.length; i++) {
    const u = st.upperPipes[i], l = st.lowerPipes[i];
    if (u.x > -PW && u.x < W) {
      rows.push({ i, kind: 'upper', x: u.x, y: u.y, ...coverOne(u, S.mUpper.eroded, st, data) });
      rows.push({ i, kind: 'lower', x: l.x, y: l.y, ...coverOne(l, S.mLower.eroded, st, data) });
      stamp(u, S.mUpper.alpha, map);
      stamp(l, S.mLower.alpha, map);
    }
  }
  // 小鸟框也盖掉（它是红的，本来就不会被判成管道，但重采样混色可能落到管道类）
  const bx0 = Math.trunc(st.playerx) - 2, bx1 = Math.trunc(st.playerx) + PLAYER_WIDTH + 2;
  const by0 = Math.trunc(st.playery) - 2, by1 = Math.trunc(st.playery) + PLAYER_HEIGHT + 2;
  for (let Y = Math.max(0, by0); Y <= Math.min(H - 1, by1); Y++) {
    for (let X = Math.max(0, bx0); X <= Math.min(W - 1, bx1); X++) map[Y * W + X] = 1;
  }
  let ghost = 0;
  const ghostAt = [];
  for (let Y = 0; Y <= Y_LIMIT; Y++) {
    for (let X = 0; X < W; X++) {
      if (map[Y * W + X]) continue;
      const o = (Y * W + X) * 4;
      if (S.lut[q5(data[o], data[o + 1], data[o + 2])] === PIPE) {
        ghost++;
        if (ghostAt.length < 8) ghostAt.push([X, Y, data[o], data[o + 1], data[o + 2]]);
      }
    }
  }
  return { rows, ghost, ghostAt };
}

/**
 * 跑一局并逐帧对拍。
 * @param {object} o
 * @param {number} o.seed
 * @param {[number,number]} o.gapRange
 * @param {number} o.maxFrames
 * @param {number} [o.minExp=300] 期望像素少于这个数的管道（刚从右边缘冒头 /
 *   只剩一条边）不判覆盖率 —— 判了也只是在测重采样的边缘，不是在测漏画。
 * @param {number} [o.minCov=0.9] 覆盖率低于它就算这根没画出来
 * @param {object} [o.sabotage] { mode: 'drop'|'shift', from, to, index } 负向对照
 */
export function runRound(o) {
  const { seed, gapRange, maxFrames, minExp = 300, minCov = 0.9, sabotage = null } = o;
  const game = new FlappyGame({ seed, hitmasks: HITMASKS, gapRange });
  const ctx = S.ctx;
  const bad = [];
  let frames = 0, judged = 0, skipped = 0, ghostFrames = 0, worst = { cov: 2 };
  let maxPipes = 0, minVisible = 99;

  while (frames < maxFrames && !game.done) {
    const st = game.observeState();

    // 画的时候可以喂一份被做过手脚的快照；**判定永远用真状态 st**
    let drawn = st;
    if (sabotage && frames >= sabotage.from && frames <= sabotage.to) {
      const k = Math.min(sabotage.index, st.upperPipes.length - 1);
      drawn = {
        ...st,
        upperPipes: st.upperPipes.map((p) => ({ ...p })),
        lowerPipes: st.lowerPipes.map((p) => ({ ...p })),
      };
      if (sabotage.mode === 'drop') {
        drawn.upperPipes.splice(k, 1);
        drawn.lowerPipes.splice(k, 1);
      } else {
        drawn.upperPipes[k].x += 30;
        drawn.lowerPipes[k].x += 30;
      }
    }
    S.renderer.draw(drawn);
    const data = ctx.getImageData(0, 0, W, H).data;
    const { rows, ghost } = judgeFrame(st, data);

    const visible = rows.length / 2;
    maxPipes = Math.max(maxPipes, visible);
    minVisible = Math.min(minVisible, visible);
    if (ghost > 40) ghostFrames++;
    const badBefore = bad.length;
    for (const r of rows) {
      if (r.exp < minExp) { skipped++; continue; }
      judged++;
      if (r.cov < worst.cov) worst = { cov: r.cov, frame: frames, ...r };
      if (r.cov < minCov && bad.length < 12) {
        bad.push({ frame: frames, ...r, ghost, score: st.score, visible });
      }
    }
    if (bad.length > badBefore && !S.dump) {
      // 第一次对不上：状态 JSON + 那一帧的 PNG 都留下来，供人工核对
      S.dump = { seed, frame: frames, state: st, rows, ghost, png: S.canvas.toDataURL('image/png') };
    }
    game.step(pilot(game));
    frames++;
  }

  return {
    seed, frames, score: game.score, judged, skipped, ghostFrames,
    maxPipes, minVisible, bad,
    worstCov: worst.cov > 1 ? null : worst,
    ok: bad.length === 0 && ghostFrames === 0,
  };
}

/** 取出第一处对不上的现场（PNG 是 data URI），取完清空。 */
export function takeDump() {
  const d = S.dump || null;
  S.dump = null;
  return d;
}

// ---------------------------------------------------------------------------
// 第二段：对着**真的 index.html** 采样。这里读不到 you / aiGame（模块作用域里
// 的 let），所以判据必须是"不需要状态也成立"的那种 —— 见 pipe_render_check.mjs
// 里的连续性判据：管道每帧固定左移 5px，中途不许凭空消失。
// ---------------------------------------------------------------------------
let live = null;

/**
 * 把一块画布上的管道压成"列区间"：管道是竖条，spacing ≥ 115 > 宽度 52，
 * 列区间永远分得开，所以按列投影就够，不必做连通块。
 *
 * 判"这一列是管道"用的是**上下两条带子都得有管道色**，不是"这一列管道色像素够多"。
 * 前两版都栽在这上面，记在这里免得下一轮再试一遍：
 *   1. `col[X] >= 3`：每帧都多出一根纹丝不动的 12px 竖条 —— 那是**小鸟**，
 *      `playery` 是小数，小鸟边缘和黑背景混出的暗色最近邻落到了管道的暗绿上。
 *   2. `col[X] >= 40`：小鸟不再自成一根了，可它**压在管道上**的时候会把那两三列
 *      的计数从 ~50 打到 40 以下，一根管道被劈成两段，判据照样报假阳性。
 * 换成上下带子就与小鸟彻底无关了：管道的上半截必然一直顶到 y=0
 * （gapTop ≤ 266，上管道顶端 = gapTop-320 < 0），下半截必然一直盖过地面线，
 * 而小鸟只有 24px 高，不可能同时出现在两条 31px 宽的带子里。
 * 顺带也躲开了"管子外侧那几列只有帽檐、计数忽高忽低"造成的边界抖动 ——
 * 这个判据量的是**管身**，宽度稳定，帧与帧之间才好比。
 */
export function barsOf(canvas) {
  const g = canvas.getContext('2d');
  const h = Y_LIMIT + 1;
  const d = g.getImageData(0, 0, W, h).data;
  const top = new Uint16Array(W);      // y ∈ [0, BAND) 里的管道色像素数
  const bot = new Uint16Array(W);      // y ∈ [h-BAND, h) 里的
  for (let Y = 0; Y < h; Y++) {
    const isTop = Y < BAND, isBot = Y >= h - BAND;
    if (!isTop && !isBot) continue;
    for (let X = 0; X < W; X++) {
      const o = (Y * W + X) * 4;
      if (S.lut[q5(d[o], d[o + 1], d[o + 2])] === PIPE) { if (isTop) top[X]++; else bot[X]++; }
    }
  }
  const runs = [];
  let s = -1;
  for (let X = 0; X <= W; X++) {
    const on = X < W && top[X] >= 3 && bot[X] >= 3;
    if (on && s < 0) s = X;
    if (!on && s >= 0) { runs.push([s, X - 1]); s = -1; }
  }
  return runs;
}

/**
 * 只看**玩家那块画布的像素**来决定这一帧扇不扇翅膀。
 *
 * 为什么非得从像素里读：页面把 `you` / `aiGame` 关在模块作用域的 let 里，
 * 外面读不到鸟的高度，也读不到下一根管子的缝在哪。而没有一个会瞄准的玩家，
 * 这一段就测不到玩家那块画布 —— 实测两版都不行：
 *   · 每 160 ms 扇一下 -> 鸟直冲天花板，每局 0 分；
 *   · 每 660 ms 扇一下（净漂移为零的悬停节奏）-> 高度是稳住了，可它不会
 *     躲缝，照样 0~1 分。
 * 所以这里干脆按像素做一个和 pilot() 同款的 bang-bang 控制器。
 *
 * **它和 AI 的策略毫无关系**：只驱动玩家那只鸟，AI 走的还是 worker 里的
 * 那份推理，一个字没动。
 */
function decideFlap(canvas) {
  const g = canvas.getContext('2d');
  const h = Y_LIMIT + 1;
  const d = g.getImageData(0, 0, W, h).data;
  const cls = (X, Y) => { const o = (Y * W + X) * 4; return S.lut[q5(d[o], d[o + 1], d[o + 2])]; };

  // 小鸟：playerx 恒为 floor(288*0.2)=57，一局都不变，所以列范围是写死的
  const BX = 57;
  let sum = 0, n = 0;
  for (let Y = 0; Y < h; Y++) {
    for (let X = BX; X < BX + PLAYER_WIDTH; X++) if (cls(X, Y) === PLAYER) { sum += Y; n++; }
  }
  if (!n) return false;                      // 找不到鸟（还没开局）就别乱按
  const birdY = sum / n;

  // 前方第一根管子：从小鸟右边开始，找第一列"上下两条带子都有管道色"的
  let px = -1;
  for (let X = BX + PLAYER_WIDTH + 2; X < W; X++) {
    let top = 0, bot = 0;
    for (let Y = 0; Y < BAND; Y++) if (cls(X, Y) === PIPE) top++;
    for (let Y = h - BAND; Y < h; Y++) if (cls(X, Y) === PIPE) bot++;
    if (top >= 3 && bot >= 3) { px = X; break; }
  }
  if (px < 0) return birdY > BASE_Y * 0.5;   // 前面没管子，别让它掉下去

  // 往里挪 5 列再读，避开管子前缘的重采样混色。那一列从上往下第一段
  // **足够长**的非管道就是缝。
  //
  // "足够长"这个限制是必须的，不能拿第一段非管道就用：管道调色板里有 3 个
  // 颜色和别的精灵撞了色、被判成歧义（不是 PIPE），它们在管身里连成整行，
  // 于是"第一段非管道"经常是管身中间的一行，算出来的缝心在天花板上，
  // 控制器就一路狂扇顶到顶、每局 0 分。缝至少 100px（gapRange 下界），
  // 这里卡 60 行，既滤掉那种一两行的假缝，又不会把真缝判掉。
  const cx = Math.min(W - 1, px + 5);
  let y0 = -1, y1 = -1, s = -1;
  for (let Y = 0; Y <= h; Y++) {
    const isPipe = Y < h && cls(cx, Y) === PIPE;
    if (!isPipe && s < 0) s = Y;
    if (isPipe && s >= 0) {
      if (Y - s >= 60) { y0 = s; y1 = Y - 1; break; }
      s = -1;
    }
  }
  if (y0 < 0) return birdY > BASE_Y * 0.5;
  const center = (y0 + y1) / 2;
  // 和 pilot() 同一个判据：偏置 12 是留给"扇一下要好几帧才升上去"的提前量
  return birdY > center + 12;
}

/**
 * @param {boolean} [autoFly=false] 是否顺便按像素驾驶玩家那只鸟。
 *   开着的时候只在遮罩收起（youState==='playing'）时打键 —— 遮罩在的时候
 *   一下按键等于 `press()` 里的 beginRound，会把这一局重开、把采样序列搅乱。
 */
export function startLive(autoFly = false) {
  if (live) return;
  live = { samples: [], on: true };
  const you = document.getElementById('youCanvas');
  const ai = document.getElementById('aiCanvas');
  const veil = document.getElementById('youVeil');
  const aiVeil = document.getElementById('aiVeil');
  const step = () => {
    if (!live || !live.on) return;
    const st = (window.__stats && window.__stats()) || null;
    if (autoFly && veil && veil.hidden && decideFlap(you)) {
      dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', bubbles: true }));
    }
    live.samples.push({
      t: Math.round(performance.now()),
      aiFrame: st ? st.aiFrame : -1,
      playing: !!veil && veil.hidden,   // 遮罩收起来 = youState==='playing' 且玩家还活着
      // AI 撞了之后 aiFrame 不再增长，可玩家那一局还在往前走 —— 这之后两块画布
      // 的帧数不再同步，玩家那块的连续性判定必须停掉，否则全是假阳性。
      aiDead: !!aiVeil && !aiVeil.hidden,
      youScore: Number(document.getElementById('youLive').textContent || 0),
      aiScore: Number(document.getElementById('aiLive').textContent || 0),
      you: barsOf(you),
      ai: barsOf(ai),
    });
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function drainLive() {
  if (!live) return [];
  const out = live.samples;
  live.samples = [];
  return out;
}

export function stopLive() { if (live) live.on = false; live = null; }
