/**
 * 实验 C 补完：死前 Q 轨迹的"成功但高难度通过"对照组。
 *
 * 上一轮 `death_attribution_qtrace.mjs` 明确标注了一个没做的局限：看到死亡前
 * maxQ 会下沉（早段 t=-10..-6 均值 11.046，临死前 t=-3..-1 均值 7.603，
 * 差 -3.443），但没有对"遇到窄缝/大偏离但最终成功躲过"的情形采集同样的
 * Q 轨迹作对照。如果连成功案例也有类似幅度的下沉，那么"Q 下沉"更多是
 * "正确识别到当前局面变难"的正常反应，不是"预示不可逆死亡"的特异性信号。
 *
 * 做法：不改原文件（继续可用、数字继续有效），新写一份，在原有逻辑基础上
 * 给每一次**成功通过**也存一份 Q 轨迹快照（和死亡时同样的对齐方式：
 * t=-10..-1，t=-1 是这次通过前最后一次决策）。跑完后按"通过时偏离中心
 * 的绝对值"分位数，把所有成功通过切成"难"（高分位，偏离大，几何上和死亡
 * 情形接近）和"易"（其余），分别算 maxQ 早段 vs 临死前（这里是"临通过前"）
 * 的差值，和死亡组的下沉幅度放在一起比较。
 *
 * 判据：
 *   - 难通过组的下沉幅度和死亡组接近 -> "Q 下沉"是"识别到变难"的通用反应，
 *     不是死亡特异性信号
 *   - 难通过组几乎不下沉、死亡组明显下沉 -> 下沉确实是死亡特异性信号，
 *     不是泛化的"变难就下沉"
 *
 * 权重同样用 base_s0（不是网页 demo 的旧模型）。种子与 A/B/C 同一组。
 *
 * 用法：node web/tools/death_attribution_qtrace_control.mjs [局数=60] [难通过分位数=0.85]
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
const HARD_QUANTILE = Number(process.argv[3] || 0.85);
const SEED_BASE = 20260906; // 与 oracle_hazard.mjs / hazard_eval.mjs / 上一轮 qtrace 同一组种子

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(4);
const FRAME_SKIP = 4;
const TRACE_LEN = 10; // 死前/通过前追溯的决策数，和上一轮一致，便于直接对比

function currentPipeIdx(g) {
  for (let i = 0; i < g.upperPipes.length; i++) {
    if (g.upperPipes[i].x + PIPE_WIDTH > g.playerx) return i;
  }
  return g.upperPipes.length - 1;
}
function pipeCenter(p) { return p.y + PIPE_HEIGHT + p.gap / 2.0; }

const deathRows = [];
const passRows = []; // 现在每条也带 trace，供难/易分组用
const allScores = [];

for (let ep = 0; ep < N; ep++) {
  const game = new FlappyGame({ seed: SEED_BASE + ep, hitmasks: HITMASKS });
  game.reset();
  let action = 0;
  const qTrace = [];
  let lastPipeRef = null;
  let lastCenter = null;
  let curDeltaCenter = null;
  let curSpacing = null;
  let last = null;
  let died = false;

  for (let i = 0; i < 20000; i++) {
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
      passRows.push({
        ep, off: last.off, gap: last.gap, trace,
      });
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
  console.log(`第 ${String(ep + 1).padStart(3)} 局: ${String(game.score).padStart(4)} 根`
    + `  ${died ? '死亡' : '截断(未死，帧数上限)'}`);
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

console.log(`\n跑了 ${N} 局，真正死亡 ${deathRows.length} 局，成功通过 ${passRows.length} 次`);

// ---------------------------------------------------------------------
// 按 |off| 分位数把成功通过切成"难"/"易"
// ---------------------------------------------------------------------
const absOffs = passRows.map((r) => Math.abs(r.off));
const threshold = quantile(absOffs, HARD_QUANTILE);
const hardPasses = passRows.filter((r) => Math.abs(r.off) >= threshold);
const easyPasses = passRows.filter((r) => Math.abs(r.off) < threshold);

console.log(`\n难度切分：|偏离中心| 第 ${(HARD_QUANTILE * 100).toFixed(0)} 分位数 = ${threshold.toFixed(1)}px`);
console.log(`难通过组 n=${hardPasses.length}（|off|>=阈值，偏离幅度中位 `
  + `${quantile(hardPasses.map((r) => Math.abs(r.off)), 0.5).toFixed(0)}px）`);
console.log(`易通过组 n=${easyPasses.length}（|off|<阈值，偏离幅度中位 `
  + `${quantile(easyPasses.map((r) => Math.abs(r.off)), 0.5).toFixed(0)}px）`);
console.log(`（对照：上一轮死亡时偏离中心中位 29px，本轮死亡组见下）`);

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
  return { label, n: rows.length, early, late, dip: late - early };
}

console.log('\n=========== maxQ 下沉幅度对比（死亡 vs 难通过 vs 易通过） ===========');
const rDeath = report('死亡（真正死亡的局）', deathRows);
const rHard = report('难通过（成功但 |off| 落在前' + ((1 - HARD_QUANTILE) * 100).toFixed(0) + '%）', hardPasses);
const rEasy = report('易通过（其余成功通过）', easyPasses);

console.log('\n=========== 判据 ===========');
console.log(`死亡组下沉 ${rDeath.dip.toFixed(3)}，难通过组下沉 ${rHard.dip.toFixed(3)}，`
  + `易通过组下沉 ${rEasy.dip.toFixed(3)}`);
if (Number.isFinite(rHard.dip)) {
  const hardVsDeathRatio = rDeath.dip !== 0 ? rHard.dip / rDeath.dip : NaN;
  console.log(`难通过组下沉 / 死亡组下沉 = ${hardVsDeathRatio.toFixed(2)}`);
  if (Math.abs(rHard.dip) > Math.abs(rDeath.dip) * 0.5) {
    console.log('-> 难通过组的下沉幅度达到死亡组的一半以上：支持"Q 下沉是识别到局面变难的'
      + '通用反应"，不是死亡特异性信号——上一轮"看见了但已进不可逆状态"这个判据要打折扣。');
  } else {
    console.log('-> 难通过组下沉幅度明显小于死亡组：下沉更像是死亡特异性信号，不是任何'
      + '变难局面都会触发——上一轮"控制/价值层"的判据得到加强。');
  }
} else {
  console.log('难通过组样本数为 0 或轨迹不足，无法比较（局数太少或难度分位数设得太极端）。');
}
console.log('\n注意：这里的"难"用 |偏离中心| 的分位数代理，不是死亡当时的确切几何匹配——'
  + '偏离中心只是死亡归因里最相关的一个维度，不代表"同样难"的全部含义，是一个近似对照，'
  + '不是严格配对实验。');
