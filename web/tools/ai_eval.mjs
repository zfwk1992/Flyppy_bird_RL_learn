/**
 * 阶段 2 的验收：用**浏览器要跑的那份代码**在 Node 里评测 AI。
 *
 * game.js + obs.js + nn.js 三个模块都是浏览器直接 import 的同一份文件，
 * 这里只是换了个宿主。所以这条结果就是网页里的 AI 水平，不是近似。
 * （Python 侧 100 局的基线是 78.2 根，见 docs/EXPERIMENTS.md。）
 *
 * 决策时机与 flappy/rollout.py: skip_step 一致：每 frame_skip 帧决策一次，
 * 窗口内重复同一个动作；新回合用首帧填满帧栈。**不加任何反应延迟**，
 * 也不做 epsilon 探索（eval_epsilon=0 —— 加 1% 随机曾让分数掉 6.2 倍）。
 *
 * 用法：
 *     node web/tools/ai_eval.mjs [局数=30] [每局帧数上限=20000]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FlappyGame, OBS_W, OBS_H } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const N_EPISODES = Number(argv[0] || 30);
const MAX_FRAMES = Number(argv[1] || 20000);
// 默认 seed 是 1000, 1001, ... 这种小连号，方便复现。
// `--browser-seeds` 换成页面实际用的那种 32 位大随机数
// （index.html: `(Date.now() ^ Math.random()*0x7fffffff) >>> 0`）——
// mulberry32 对大 seed 的表现理论上没有区别，但"理论上"不算数，要测。
const BROWSER_SEEDS = argv.includes('--browser-seeds');
const seedFor = (ep) => (BROWSER_SEEDS
  ? ((Date.now() + ep * 7919) ^ (Math.random() * 0x7fffffff)) >>> 0
  : 1000 + ep);
const FRAME_SKIP = 4;
const FRAME_STACK = 4;

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(FRAME_STACK);

const scores = [];
let totalDecisions = 0;
const t0 = performance.now();

for (let ep = 0; ep < N_EPISODES; ep++) {
  const seed = seedFor(ep);
  const game = new FlappyGame({ seed, hitmasks: HITMASKS });
  game.reset();
  let action = 0;
  let i = 0;
  for (; i < MAX_FRAMES; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obs);
      const arr = i === 0 ? stack.reset(obs) : stack.push(obs);
      action = net.act(arr);
      totalDecisions++;
    }
    if (game.step(action).done) break;
  }
  scores.push(game.score);
  process.stdout.write(`  第 ${String(ep + 1).padStart(2)} 局 seed=${String(seed).padStart(10)}: ${String(game.score).padStart(4)} 根`
    + `  (${i} 帧${i >= MAX_FRAMES - 1 ? '，触顶截断' : ''})\n`);
}

const sorted = [...scores].sort((a, b) => a - b);
const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (scores.length - 1));
const median = sorted.length % 2
  ? sorted[(sorted.length - 1) / 2]
  : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
const secs = (performance.now() - t0) / 1000;

console.log('\n---------------------------------------------');
console.log(`局数        ${scores.length}`);
console.log(`平均        ${mean.toFixed(2)} 根   (标准误 ${(sd / Math.sqrt(scores.length)).toFixed(2)})`);
console.log(`中位        ${median}`);
console.log(`最低 / 最高 ${sorted[0]} / ${sorted[sorted.length - 1]}`);
console.log(`决策 ${totalDecisions} 次，耗时 ${secs.toFixed(1)}s，`
  + `单次前向 ${(secs * 1000 / totalDecisions).toFixed(1)} ms`);
console.log('---------------------------------------------');
