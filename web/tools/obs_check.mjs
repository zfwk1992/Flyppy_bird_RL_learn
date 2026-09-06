/**
 * 观测管线的逐帧一致性比对（阶段 2 验收项之一）。
 *
 * 重放 `trace_ai.json` 的抽样与动作，让 JS 走一遍和 Python 完全相同的轨迹，
 * 每一帧比两个 md5：
 *
 *   red  —— 重建出来的 288x512 R 通道画面，验的是 blit 位置/顺序/透明处理
 *   obs  —— 80x128 二值观测，验的是 INTER_AREA 权重和阈值
 *
 * 分成两个是为了定位：只有 obs 挂说明画面是对的、缩放实现有偏差。
 *
 * 用法：
 *     node web/tools/obs_check.mjs
 * 前置：python web/tools/dump_obs_ref.py 生成 obs_ref.json（已提交，
 *       只有改了 game/flappy_env.py 或精灵才需要重跑）。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { FlappyGame, OBS_W, OBS_H } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample } from '../obs.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const trace = JSON.parse(fs.readFileSync(path.join(HERE, 'trace_ai.json'), 'utf8'));
const ref = JSON.parse(fs.readFileSync(path.join(HERE, 'obs_ref.json'), 'utf8'));
const { meta, draws, frames } = trace;

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

const md5 = (buf) => crypto.createHash('md5').update(buf).digest('hex');

let game = makeGame();
game.reset();

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const obsBytes = new Uint8Array(OBS_W * OBS_H);   // {0,255}，与 Python 的 uint8 对齐

const bad = [];
let redOk = 0;
let obsOk = 0;

for (let i = 0; i < frames.length; i++) {
  const f = frames[i];
  const { done } = game.step(f.action);

  renderRed(game, red);
  downsample(red, obs);
  for (let k = 0; k < obs.length; k++) obsBytes[k] = obs[k] ? 255 : 0;

  const rh = md5(red);
  const oh = md5(obsBytes);
  const okRed = rh === ref.red_md5[i];
  const okObs = oh === ref.obs_md5[i];
  if (okRed) redOk++;
  if (okObs) obsOk++;
  if (game.playerIndex !== ref.player_index[i] || !okRed || !okObs) {
    if (bad.length < 12) {
      bad.push(`帧 ${i}: red=${okRed ? 'ok' : `${rh.slice(0, 8)}≠${ref.red_md5[i].slice(0, 8)}`}`
        + ` obs=${okObs ? 'ok' : `${oh.slice(0, 8)}≠${ref.obs_md5[i].slice(0, 8)}`}`
        + ` playerIndex=${game.playerIndex}/${ref.player_index[i]}`);
    }
  }
  if (done !== f.done) { bad.push(`帧 ${i}: done 不一致 ${done}/${f.done}`); break; }
  if (done) { game = makeGame(); game.reset(); }
}

// 抽样帧再做一次位级比对，失败时能看出是差一片还是差几个孤立格子
let sampleNote = '';
for (const [k, b64] of Object.entries(ref.obs_sample_packed)) {
  const bytes = Buffer.from(b64, 'base64');
  let bits = 0;
  for (const b of bytes) bits += (b & 1) + ((b >> 1) & 1) + ((b >> 2) & 1) + ((b >> 3) & 1)
    + ((b >> 4) & 1) + ((b >> 5) & 1) + ((b >> 6) & 1) + ((b >> 7) & 1);
  sampleNote += `  帧 ${k}: Python 点亮 ${bits} / ${OBS_W * OBS_H} 格\n`;
}

const n = frames.length;
console.log(`帧数 ${n}，随机数消耗 ${cursor}/${draws.length}`);
console.log(`R 通道画面逐位一致 : ${redOk}/${n}`);
console.log(`二值观测逐位一致   : ${obsOk}/${n}`);
console.log('抽样帧（Python 侧点亮格数，供人工核对量级）:');
process.stdout.write(sampleNote);
if (bad.length) {
  console.log('\n前若干处不一致:');
  for (const b of bad) console.log('  ' + b);
  console.log('\nOBS PARITY FAILED');
  process.exit(1);
}
console.log('\nOBS PARITY OK');
