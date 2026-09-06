/**
 * 浏览器推理的正确性比对（阶段 2 验收项之二）。
 *
 * 在 trace_ai 的同一条轨迹上，把 JS 手写前向的 Q 值和 PyTorch 的 Q 值逐次决策
 * 对比。比的是 **Q 值本身**而不只是 argmax —— 两个动作的 argmax 碰巧一致的
 * 概率太高，测不出展平顺序、V/A 合并这类错误。
 *
 * 顺带比 **帧栈的 md5**：栈对上而 Q 不对，问题在 nn.js；栈就不对，
 * 问题在 obs.js 或决策时机（frame_skip 窗口）。
 *
 * 用法：
 *     node web/tools/nn_check.mjs
 * 前置：python web/tools/dump_nn_ref.py 生成 nn_ref.json（已提交）。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { FlappyGame, OBS_W, OBS_H } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const trace = JSON.parse(fs.readFileSync(path.join(HERE, 'trace_ai.json'), 'utf8'));
const ref = JSON.parse(fs.readFileSync(path.join(HERE, 'nn_ref.json'), 'utf8'));
const { meta, draws, frames } = trace;
const { frame_skip: SKIP, frame_stack: NSTACK } = ref.meta;

const binPath = path.join(HERE, '..', 'model', WEIGHTS_META.file);
const raw = fs.readFileSync(binPath);
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

let cursor = 0;
const replayRng = () => {
  if (cursor >= draws.length) throw new Error('随机数用尽：JS 比 Python 多抽了');
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
  hitmasks: HITMASKS,
});

const md5 = (b) => crypto.createHash('md5').update(b).digest('hex');

let game = makeGame();
game.reset();

const stack = new FrameStack(NSTACK);
const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stackBytes = new Uint8Array(NSTACK * OBS_W * OBS_H);

let justReset = true;
let curAction = 0;
let d = 0;
let stackOk = 0;
let actionOk = 0;
let maxQErr = 0;
let sumMs = 0;
const bad = [];

for (let i = 0; i < frames.length; i++) {
  if (i % SKIP === 0) {
    renderRed(game, red);
    downsample(red, obs);
    const arr = justReset ? stack.reset(obs) : stack.push(obs);
    justReset = false;
    for (let k = 0; k < arr.length; k++) stackBytes[k] = arr[k] ? 255 : 0;

    const expect = ref.decisions[d];
    if (expect.i !== i) throw new Error(`决策帧对不上：JS ${i} vs PY ${expect.i}`);
    const sh = md5(stackBytes);
    const okStack = sh === expect.stack_md5;
    if (okStack) stackOk++;

    const t0 = process.hrtime.bigint();
    const q = net.forward(arr);
    sumMs += Number(process.hrtime.bigint() - t0) / 1e6;

    const err = Math.max(Math.abs(q[0] - expect.q[0]), Math.abs(q[1] - expect.q[1]));
    if (err > maxQErr) maxQErr = err;
    curAction = q[1] > q[0] ? 1 : 0;
    const okAct = curAction === expect.action;
    if (okAct) actionOk++;
    if ((!okStack || !okAct) && bad.length < 10) {
      bad.push(`决策帧 ${i}: stack=${okStack ? 'ok' : `${sh.slice(0, 8)}≠${expect.stack_md5.slice(0, 8)}`}`
        + ` action=${curAction}/${expect.action}`
        + ` q=[${q[0].toFixed(4)},${q[1].toFixed(4)}] vs [${expect.q[0].toFixed(4)},${expect.q[1].toFixed(4)}]`);
    }
    d++;
  }
  const { done } = game.step(curAction);
  if (done !== frames[i].done) { bad.push(`帧 ${i}: done 不一致`); break; }
  if (done) { game = makeGame(); game.reset(); justReset = true; }
}

const n = ref.decisions.length;
console.log(`决策次数 ${d}/${n}，随机数消耗 ${cursor}/${draws.length}`);
console.log(`帧栈逐位一致   : ${stackOk}/${n}`);
console.log(`动作一致       : ${actionOk}/${n}`);
console.log(`Q 值最大偏差   : ${maxQErr.toExponential(2)}  （fp16 权重 + 不同累加顺序）`);
console.log(`单次前向平均   : ${(sumMs / n).toFixed(1)} ms  （预算 133 ms/决策）`);
if (bad.length) {
  console.log('\n前若干处不一致:');
  for (const b of bad) console.log('  ' + b);
  console.log('\nNN PARITY FAILED');
  process.exit(1);
}
console.log('\nNN PARITY OK');
