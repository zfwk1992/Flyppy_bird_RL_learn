/**
 * 实验 C 对照组的部署分布版本：死前 Q 轨迹 vs "难通过但成功"轨迹，
 * gapRange=[100,165]。
 *
 * 背景：训练分布上 `death_attribution_qtrace_control.mjs` 已经证实死前
 * maxQ 下沉是死亡特异性信号（难通过组下沉幅度明显小于死亡组）。这份是
 * 部署分布版本，回答同一个问题："死前 maxQ 下沉"这个信号在部署分布上
 * 是不是同样具有死亡特异性，而不是任何变难局面都会触发。
 *
 * 做法和判据完全照抄 `death_attribution_qtrace_control.mjs`，只改
 * gapRange=[100,165] 和帧数上限（对齐 eval.py --max-decisions=2000 默认值）。
 * 权重、种子基准与其余部署分布实验一致。
 *
 * 用法：node web/tools/death_attribution_qtrace_control_deploy.mjs [局数=30] [难通过分位数=0.85]
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
const N = Number(process.argv[2] || 30);
const HARD_QUANTILE = Number(process.argv[3] || 0.85);
const SEED_BASE = 20260906;
const GAP_RANGE = [100, 165];
const MAX_DECISIONS = 2000;
const MAX_FRAMES = MAX_DECISIONS * 4;

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(4);
const FRAME_SKIP = 4;
const TRACE_LEN = 10;

function currentPipeIdx(g) {
  for (let i = 0; i < g.upperPipes.length; i++) {
    if (g.upperPipes[i].x + PIPE_WIDTH > g.playerx) return i;
  }
  return g.upperPipes.length - 1;
}
function pipeCenter(p) { return p.y + PIPE_HEIGHT + p.gap / 2.0; }

const deathRows = [];
const passRows = [];
const allScores = [];
const t0 = Date.now();

for (let ep = 0; ep < N; ep++) {
  const game = new FlappyGame({ seed: SEED_BASE + ep, hitmasks: HITMASKS, gapRange: GAP_RANGE });
  game.reset();
  let action = 0;
  const qTrace = [];
  let lastPipeRef = null;
  let lastCenter = null;
  let curDeltaCenter = null;
  let curSpacing = null;
  let last = null;
  let died = false;

  for (let i = 0; i < MAX_FRAMES; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obs);
      const arr = i === 0 ? stack.reset(obs) : stack.push(obs);
      const q = net.forward(arr);
      const q0 = q[0]; const q1 = q[1];
      action = q1 > q0 ? 1 : 0;
      qTrace.push({ maxQ: Math.max(q0, q1), gapQ: Math.abs(q1 - q0), q0, q1 });
      if (qTrace.length > TRACE_LEN) qTrace.shift();
    }

    const idx = currentPipeIdx(game);
    const curPipe = game.upperPipes[idx];
    if (curPipe !== lastPipeRef) {
      const center = pipeCenter(curPipe);
      if (idx > 0) {
        curDeltaCenter = center - pipeCenter(game.upperPipes[idx - 1]);
        curSpacing = game.upperPipes[idx].x - game.upperPipes[idx - 1].x;
      } else if (lastCenter !== null) {
        curDeltaCenter = center - lastCenter;
        curSpacing = null;
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
      const trace = qTrace.map((t, i2) => ({ t: i2 - qTrace.length + 1, ...t }));
      passRows.push({ ep, off: last.off, gap: last.gap, trace });
    }
    if (r.done) { died = true; break; }
  }

  allScores.push(game.score);
  if (died) {
    const trace = qTrace.map((t, i) => ({ t: i - qTrace.length + 1, ...t }));
    deathRows.push({
      ep, score: game.score, gap: last.gap, off: last.off,
      deltaCenter: last.deltaCenter, spacing: last.spacing, trace,
    });
  }
  const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
  console.log(`第 ${String(ep + 1).padStart(3)} 局: ${String(game.score).padStart(4)} 根`
    + `  ${died ? '死亡' : '截断(未死，撞2000决策上限)'}  [累计${elapsed}s]`);
}

function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function quantile(a, q) {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos); const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

console.log(`\n跑了 ${N} 局（部署分布 gapRange=[100,165]），真正死亡 ${deathRows.length} 局，成功通过 ${passRows.length} 次`);

const absOffs = passRows.map((r) => Math.abs(r.off));
const threshold = quantile(absOffs, HARD_QUANTILE);
const hardPasses = passRows.filter((r) => Math.abs(r.off) >= threshold);
const easyPasses = passRows.filter((r) => Math.abs(r.off) < threshold);

console.log(`\n难度切分：|偏离中心| 第 ${(HARD_QUANTILE * 100).toFixed(0)} 分位数 = ${threshold.toFixed(1)}px`);
console.log(`难通过组 n=${hardPasses.length}（|off|>=阈值，偏离幅度中位 `
  + `${quantile(hardPasses.map((r) => Math.abs(r.off)), 0.5).toFixed(0)}px）`);
console.log(`易通过组 n=${easyPasses.length}（|off|<阈值，偏离幅度中位 `
  + `${quantile(easyPasses.map((r) => Math.abs(r.off)), 0.5).toFixed(0)}px）`);

function segMean(rows, field, tFrom, tTo) {
  const pts = [];
  for (const r of rows) {
    for (const x of r.trace) if (x.t >= tFrom && x.t <= tTo) pts.push(x[field]);
  }
  return pts.length ? mean(pts) : NaN;
}

function report(label, rows) {
  const early = segMean(rows, 'maxQ', -10, -6);
  const late = segMean(rows, 'maxQ', -3, -1);
  const earlyGap = segMean(rows, 'gapQ', -10, -6);
  const lateGap = segMean(rows, 'gapQ', -3, -1);
  console.log(`\n[${label}]  n=${rows.length}`);
  console.log(`  maxQ 早段(t=-10..-6)=${early.toFixed(3)}  临事件前(t=-3..-1)=${late.toFixed(3)}`
    + `  差=${(late - early).toFixed(3)}`);
  console.log(`  |Q1-Q0| 早段=${earlyGap.toFixed(3)}  临事件前=${lateGap.toFixed(3)}`
    + `  差=${(lateGap - earlyGap).toFixed(3)}`);
  return {
    label, n: rows.length, early, late, dip: late - early,
  };
}

console.log('\n=========== maxQ 下沉幅度对比（死亡 vs 难通过 vs 易通过，部署分布） ===========');
const rDeath = report('死亡（真正死亡的局）', deathRows);
const rHard = report(`难通过（成功但 |off| 落在前${((1 - HARD_QUANTILE) * 100).toFixed(0)}%）`, hardPasses);
const rEasy = report('易通过（其余成功通过）', easyPasses);

console.log('\n=========== 判据 ===========');
console.log(`死亡组下沉 ${rDeath.dip.toFixed(3)}，难通过组下沉 ${rHard.dip.toFixed(3)}，`
  + `易通过组下沉 ${rEasy.dip.toFixed(3)}`);
if (Number.isFinite(rHard.dip)) {
  const hardVsDeathRatio = rDeath.dip !== 0 ? rHard.dip / rDeath.dip : NaN;
  console.log(`难通过组下沉 / 死亡组下沉 = ${hardVsDeathRatio.toFixed(2)}`);
  if (Math.abs(rHard.dip) > Math.abs(rDeath.dip) * 0.5) {
    console.log('-> 难通过组的下沉幅度达到死亡组的一半以上：支持"Q 下沉是识别到局面变难的'
      + '通用反应"，不是死亡特异性信号——训练分布上"看见了但已进不可逆状态"这个判据在部署'
      + '分布上要打折扣。');
  } else {
    console.log('-> 难通过组下沉幅度明显小于死亡组：下沉更像是死亡特异性信号，不是任何'
      + '变难局面都会触发——训练分布上"控制/价值层"的判据在部署分布上得到复现。');
  }
} else {
  console.log('难通过组样本数为 0 或轨迹不足，无法比较（局数太少或难度分位数设得太极端）。');
}
console.log(`\n总耗时 ${((Date.now() - t0) / 1000).toFixed(0)}s`);
