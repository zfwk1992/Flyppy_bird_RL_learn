/**
 * 实验 C 的部署分布版本：死前 10 次决策的 max-Q 轨迹，gapRange=[100,165]。
 *
 * 背景：`death_attribution_qtrace.mjs`（训练分布 [85,165]）和它的对照组
 * `death_attribution_qtrace_control.mjs` 都已经测过——死前 maxQ 明显下沉
 * （早段 11.046 -> 临死前 7.603，差 -3.443），且被证实是死亡特异性信号
 * （难通过对照组几乎不下沉）。但那两轮都在训练分布上做，`LAYER0-RESULTS.md`
 * "部署分布交叉验证"一节明确留了一句"如果想在部署分布上补 Q 轨迹，预算要
 * 先算好"——这次做这个，不改任何已有文件（只改 gapRange，其余逻辑照抄
 * `death_attribution_qtrace.mjs`，方便直接对比两个分布的数字）。
 *
 * 权重同样用 base_s0（不是网页 demo 的旧模型）。种子与其余部署分布实验
 * （`hazard_eval_deploy.mjs` / `oracle_hazard_deploy.mjs` / `value_resolution_deploy.mjs`）
 * 同一组基准 20260906+i。
 *
 * 部署分布单局明显更长（`hazard_eval_deploy.mjs` 实测 61.6s/局，是训练分布
 * 22.2s/局的约 2.8 倍），默认局数比训练分布那次（60）小，按需要传参调大。
 *
 * 用法：node web/tools/death_attribution_qtrace_deploy.mjs [局数=30]
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
const SEED_BASE = 20260906; // 与其余部署分布实验同一组种子
const GAP_RANGE = [100, 165]; // 部署分布——demo 实际在用的难度
const MAX_DECISIONS = 2000; // 对齐本机 eval.py --max-decisions 默认值
const MAX_FRAMES = MAX_DECISIONS * 4; // frame_skip=4

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(4);
const FRAME_SKIP = 4;
const TRACE_LEN = 10; // 死前追溯的决策数，和训练分布那次一致，便于直接对比

function currentPipeIdx(g) {
  for (let i = 0; i < g.upperPipes.length; i++) {
    if (g.upperPipes[i].x + PIPE_WIDTH > g.playerx) return i;
  }
  return g.upperPipes.length - 1;
}
function pipeCenter(p) { return p.y + PIPE_HEIGHT + p.gap / 2.0; }

const deathRows = [];   // 只含真正死亡的局：死亡几何 + Q 轨迹
const passRows = [];    // 每次成功通过一条：偏离中心
const allScores = [];   // 每局一条，含截断局，只用于报平均分
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
      passRows.push({ ep, off: last.off, gap: last.gap });
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
    + `  ${died ? `死时缝隙=${last.gap.toFixed(0)}px  偏离中心=${last.off > 0 ? '+' : ''}${last.off.toFixed(0)}px`
      + `  死前maxQ(t-1)=${qTrace.length ? qTrace[qTrace.length - 1].maxQ.toFixed(2) : 'n/a'}`
      : '截断(未死，撞2000决策上限)——不计入死亡几何/Q轨迹'}`
    + `  [累计${elapsed}s]`);
}

function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function median(a) {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

console.log('\n=========== 死亡几何汇总（部署分布 gapRange=[100,165]） ===========');
console.log(`跑了 ${N} 局，真正死亡 ${deathRows.length} 局`
  + `（${N - deathRows.length} 局撞上 ${MAX_DECISIONS} 决策上限被截断，未计入下面的死亡几何/Q轨迹）`);
console.log(`平均分（含截断局，未做删失校正） ${mean(allScores).toFixed(1)}`);
if (deathRows.length) {
  console.log(`死亡缝隙宽度：中位 ${median(deathRows.map((r) => r.gap)).toFixed(0)}px`
    + `（部署分布 100-165 均匀采样，期望中位 132.5）`);
  console.log(`死亡时偏离中心：中位 ${median(deathRows.map((r) => r.off)).toFixed(0)}px`
    + `，范围 [${Math.min(...deathRows.map((r) => r.off)).toFixed(0)}, `
    + `${Math.max(...deathRows.map((r) => r.off)).toFixed(0)}]`);
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
  ? '判据：临死前 maxQ 明显下沉 -> 支持"看见了但已进入不可逆状态"（控制/价值层），与训练分布结论一致'
  : '判据：临死前 maxQ 没有明显下沉 -> 支持"网络没看见危险"（感知层）——与训练分布结论不一致，需要说明');
console.log(`\n总耗时 ${((Date.now() - t0) / 1000).toFixed(0)}s`);
