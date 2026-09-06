/**
 * 实验 C：死亡解剖 + 死前 Q 轨迹。
 *
 * 在 `death_attribution.mjs`（缝隙宽度/偏离中心/撞地面）的基础上新写一份
 * （没有改原文件，原文件继续可用），多记录三件事：
 *
 *   1. 死前最近 10 次决策的 max-Q 轨迹 和 动作差距 |Q1-Q0|
 *   2. 死亡那根管道的几何：缝隙中心与上一根的落差 Δcenter、间距 spacing、gap
 *   3. 对照组：**成功通过**管道瞬间的偏离分布（同一批局里，不是单独跑）
 *
 * 判据（写进结论，不是这里下）：
 *   - Q 在死前 **不下沉** -> 网络根本没看见危险 -> 矛头指向**感知层**
 *   - Q 提前几步就下沉 -> 看见了但已经进了不可逆状态 -> 指向**控制/价值层**
 *
 * 权重用当前模型 base_s0（不用 nn.js 默认的旧模型 weights-meta.js）。
 *
 * 用法：node web/tools/death_attribution_qtrace.mjs [局数=60]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  FlappyGame, OBS_W, OBS_H, PIPE_WIDTH, PLAYER_HEIGHT, PIPE_HEIGHT,
} from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta-base_s0.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const N = Number(process.argv[2] || 60);
const SEED_BASE = 20260906; // 与 oracle_hazard.mjs / hazard_eval.mjs 同一组种子

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(4);
const FRAME_SKIP = 4;
const TRACE_LEN = 10; // 死前追溯的决策数

function currentPipeIdx(g) {
  for (let i = 0; i < g.upperPipes.length; i++) {
    if (g.upperPipes[i].x + PIPE_WIDTH > g.playerx) return i;
  }
  return g.upperPipes.length - 1;
}
function pipeCenter(p) { return p.y + PIPE_HEIGHT + p.gap / 2.0; }

const deathRows = [];   // 每局一条：死亡几何 + Q 轨迹
const passRows = [];    // 每次成功通过一条：偏离中心

for (let ep = 0; ep < N; ep++) {
  const game = new FlappyGame({ seed: SEED_BASE + ep, hitmasks: HITMASKS });
  game.reset();
  let action = 0;
  // 滚动窗口：最近 TRACE_LEN 次决策的 {maxQ, gap}
  const qTrace = [];
  let lastPipeRef = null; // 按对象引用跟踪，而不是数组下标——下标会因 shift() 回收
                           // 旧管道而对同一根管道发生变化，按下标比较会把"数组重排"
                           // 误判成"换了一根新管道"，把刚算好的 Δcenter/spacing 冲成 0
  let lastCenter = null;
  let curDeltaCenter = null;
  let curSpacing = null;
  let last = null;

  for (let i = 0; i < 20000; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obs);
      const arr = i === 0 ? stack.reset(obs) : stack.push(obs);
      const q = net.forward(arr); // [q0, q1]，net 内部复用的数组，立刻拷贝出来
      const q0 = q[0]; const q1 = q[1];
      action = q1 > q0 ? 1 : 0;
      qTrace.push({ maxQ: Math.max(q0, q1), gapQ: Math.abs(q1 - q0), q0, q1 });
      if (qTrace.length > TRACE_LEN) qTrace.shift();
    }

    // 追踪"当前管道"切换，算 Δcenter / spacing。切换阈值 x<=playerx-52
    // 早于回收阈值 x<-52，中间有约 57px 缓冲，所以真正换管道时上一根通常
    // 还在数组里（idx>0 分支能拿到准确 spacing）。
    const idx = currentPipeIdx(game);
    const curPipe = game.upperPipes[idx];
    if (curPipe !== lastPipeRef) {
      const center = pipeCenter(curPipe);
      if (idx > 0) {
        curDeltaCenter = center - pipeCenter(game.upperPipes[idx - 1]);
        curSpacing = game.upperPipes[idx].x - game.upperPipes[idx - 1].x;
      } else if (lastCenter !== null) {
        // 极少数情况：换管道被发现时上一根已经被回收，没法拿准 spacing
        curDeltaCenter = center - lastCenter;
        curSpacing = null; // 标 null 而不是编造
      }
      lastCenter = center;
      lastPipeRef = curPipe;
    }

    const p = game.upperPipes[idx];
    const off = (game.playery + PLAYER_HEIGHT / 2.0) - pipeCenter(p);
    last = {
      gap: p.gap, off, deltaCenter: curDeltaCenter, spacing: curSpacing,
    };

    const r = game.step(action);
    if (r.info.scored > 0) {
      // 用踏入 step() 之前记录的 off/gap（那是"刚越过"那一刻之前的状态）
      passRows.push({ ep, off: last.off, gap: last.gap });
    }
    if (r.done) break;
  }

  const trace = qTrace.map((t, i) => ({ t: i - qTrace.length + 1, ...t }));
  deathRows.push({
    ep, score: game.score, gap: last.gap, off: last.off,
    deltaCenter: last.deltaCenter, spacing: last.spacing, trace,
  });
  console.log(`第 ${String(ep + 1).padStart(3)} 局: ${String(game.score).padStart(4)} 根`
    + `  死时缝隙=${last.gap.toFixed(0)}px  偏离中心=${last.off > 0 ? '+' : ''}${last.off.toFixed(0)}px`
    + `  Δcenter=${last.deltaCenter === null ? 'n/a' : last.deltaCenter.toFixed(0)}`
    + `  spacing=${last.spacing === null ? 'n/a' : last.spacing.toFixed(0)}`
    + `  死前maxQ(t-1)=${trace.length ? trace[trace.length - 1].maxQ.toFixed(2) : 'n/a'}`);
}

// ---------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------
function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function median(a) {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

console.log('\n=========== 死亡几何汇总 ===========');
console.log(`局数 ${deathRows.length}，平均分 ${mean(deathRows.map((r) => r.score)).toFixed(1)}`);
const withDelta = deathRows.filter((r) => r.deltaCenter !== null);
console.log(`死亡缝隙宽度：中位 ${median(deathRows.map((r) => r.gap)).toFixed(0)}px`
  + `（训练分布 85-165 均匀采样，期望中位 125）`);
console.log(`死亡时偏离中心：中位 ${median(deathRows.map((r) => r.off)).toFixed(0)}px`
  + `，范围 [${Math.min(...deathRows.map((r) => r.off)).toFixed(0)}, `
  + `${Math.max(...deathRows.map((r) => r.off)).toFixed(0)}]`);
if (withDelta.length) {
  console.log(`死亡管道 Δcenter（与上一根落差，n=${withDelta.length}）：`
    + `中位 ${median(withDelta.map((r) => r.deltaCenter)).toFixed(0)}px`
    + `，|Δcenter| 中位 ${median(withDelta.map((r) => Math.abs(r.deltaCenter))).toFixed(0)}px`);
}
const withSpacing = deathRows.filter((r) => r.spacing !== null);
if (withSpacing.length) {
  console.log(`死亡管道 spacing（n=${withSpacing.length}）：中位 `
    + `${median(withSpacing.map((r) => r.spacing)).toFixed(0)}px（训练分布 115-200）`);
}

console.log('\n=========== 成功通过时的偏离分布（对照组） ===========');
console.log(`样本数（成功通过次数）：${passRows.length}`);
if (passRows.length) {
  const offs = passRows.map((r) => r.off);
  console.log(`偏离中心：中位 ${median(offs).toFixed(0)}px，`
    + `范围 [${Math.min(...offs).toFixed(0)}, ${Math.max(...offs).toFixed(0)}]`);
  console.log(`|偏离| 中位：${median(offs.map(Math.abs)).toFixed(0)}px`);
}

console.log('\n=========== 死前 Q 轨迹（按相对位置 t=-10..-1 对齐后平均） ===========');
console.log('t\tn\tmaxQ均值\t|Q1-Q0|均值');
for (let t = -TRACE_LEN; t <= -1; t++) {
  const pts = [];
  for (const r of deathRows) {
    const hit = r.trace.find((x) => x.t === t);
    if (hit) pts.push(hit);
  }
  if (!pts.length) continue;
  console.log(`${t}\t${pts.length}\t${mean(pts.map((p) => p.maxQ)).toFixed(3)}`
    + `\t\t${mean(pts.map((p) => p.gapQ)).toFixed(3)}`);
}

// 简单判据：比较 t=-10..-6（早）与 t=-3..-1（临死前）两段的 maxQ 均值
function segMean(field, tFrom, tTo) {
  const pts = [];
  for (const r of deathRows) {
    for (const x of r.trace) if (x.t >= tFrom && x.t <= tTo) pts.push(x[field]);
  }
  return pts.length ? mean(pts) : NaN;
}
const earlyQ = segMean('maxQ', -10, -6);
const lateQ = segMean('maxQ', -3, -1);
const earlyGap = segMean('gapQ', -10, -6);
const lateGap = segMean('gapQ', -3, -1);
console.log(`\n早段(t=-10..-6) maxQ均值=${earlyQ.toFixed(3)}  临死前(t=-3..-1) maxQ均值=${lateQ.toFixed(3)}`
  + `  差=${(lateQ - earlyQ).toFixed(3)}`);
console.log(`早段 |Q1-Q0|均值=${earlyGap.toFixed(3)}  临死前 |Q1-Q0|均值=${lateGap.toFixed(3)}`
  + `  差=${(lateGap - earlyGap).toFixed(3)}`);
console.log(lateQ < earlyQ - 0.05
  ? '判据：临死前 maxQ 明显下沉 -> 支持"看见了但已进入不可逆状态"（控制/价值层）'
  : 'judgement: 临死前 maxQ 没有明显下沉 -> 支持"网络没看见危险"（感知层）');
