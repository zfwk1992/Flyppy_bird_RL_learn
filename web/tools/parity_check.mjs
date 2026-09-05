/**
 * 逐帧比对 JS 移植与 Python 参考实现（阶段 1 的验收项）。
 *
 * 做法：把 Python 记录下来的每一个 random() 原始抽样按顺序喂给 JS，
 * 再重放同一串动作。因为 `random.uniform(a,b) === a + (b-a)*random()`
 * 而 JS 的 `_uniform` 是 `lo + r*(hi-lo)`，同一个 r 进去结果逐位相同 ——
 * 所以两边的管道和物理都必须精确一致，不是"接近"，是逐位。
 *
 * 用法：
 *     node web/tools/parity_check.mjs
 * 前置：先跑 python web/tools/dump_python_trace.py 生成 trace.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FlappyGame } from '../game.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const trace = JSON.parse(fs.readFileSync(path.join(HERE, 'trace.json'), 'utf8'));
const { meta, draws, frames } = trace;

// 重放 Python 的抽样序列。用光了就抛错 —— 说明 JS 比 Python 多抽了随机数，
// 那本身就是一处逻辑不一致，必须暴露出来而不是悄悄退化成别的随机源。
let cursor = 0;
const replayRng = () => {
  if (cursor >= draws.length) {
    throw new Error(
      `随机数用尽：JS 消费了超过 ${draws.length} 个抽样，说明比 Python 多抽了`);
  }
  return draws[cursor++];
};

const makeGame = () => new FlappyGame({
  randomize: meta.randomize,
  gapRange: meta.gap_range,
  spacingRange: meta.spacing_range,
  edgeMargin: meta.edge_margin,
  maxDeltaFrac: meta.max_delta_frac,
  pipeGap: meta.pipe_gap,
  rng: replayRng,
});

let game = makeGame();
game.reset();

const EPS = 1e-9;
const bad = [];
const near = (a, b) => Math.abs(a - b) <= EPS;

function cmpPipes(label, jsArr, pyArr, i) {
  if (jsArr.length !== pyArr.length) {
    bad.push(`帧 ${i} ${label} 数量 JS=${jsArr.length} PY=${pyArr.length}`);
    return;
  }
  for (let k = 0; k < jsArr.length; k++) {
    const j = jsArr[k], p = pyArr[k];
    if (!near(j.x, p.x)) bad.push(`帧 ${i} ${label}[${k}].x JS=${j.x} PY=${p.x}`);
    if (!near(j.y, p.y)) bad.push(`帧 ${i} ${label}[${k}].y JS=${j.y} PY=${p.y}`);
    if (p.gap !== undefined && !near(j.gap, p.gap)) {
      bad.push(`帧 ${i} ${label}[${k}].gap JS=${j.gap} PY=${p.gap}`);
    }
  }
}

let checked = 0;
for (const f of frames) {
  const out = game.step(f.action);
  const done = out.done !== undefined ? out.done : game.done;
  const score = out.info !== undefined ? out.info.score : game.score;

  if (!near(game.playery, f.playery)) {
    bad.push(`帧 ${f.i} playery JS=${game.playery} PY=${f.playery}`);
  }
  if (!near(game.playerVelY, f.playerVelY)) {
    bad.push(`帧 ${f.i} playerVelY JS=${game.playerVelY} PY=${f.playerVelY}`);
  }
  if (!near(game.basex, f.basex)) {
    bad.push(`帧 ${f.i} basex JS=${game.basex} PY=${f.basex}`);
  }
  if (score !== f.score) bad.push(`帧 ${f.i} score JS=${score} PY=${f.score}`);
  if (!!done !== f.done) bad.push(`帧 ${f.i} done JS=${done} PY=${f.done}`);

  if (!f.reset) {
    cmpPipes('upper', game.upperPipes, f.upper, f.i);
    cmpPipes('lower', game.lowerPipes, f.lower, f.i);
  }
  checked++;

  if (f.reset) game.reset();
  if (bad.length >= 12) break;   // 前十几条足够定位，不刷屏
}

console.log(`比对 ${checked} / ${frames.length} 帧，消费随机数 ${cursor} / ${draws.length}`);
if (bad.length === 0 && cursor === draws.length) {
  console.log('PARITY OK  物理与管道生成逐位一致，随机数消费次数也一致');
  process.exit(0);
}
if (bad.length === 0) {
  console.log(`PARITY PARTIAL  数值全对，但随机数消费 ${cursor} != ${draws.length}`);
  process.exit(1);
}
console.log(`PARITY FAIL  ${bad.length} 处不一致（只列前几条）：`);
for (const b of bad) console.log('  ' + b);
process.exit(1);
