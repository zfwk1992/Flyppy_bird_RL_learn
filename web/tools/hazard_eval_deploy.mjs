/**
 * 独立交叉验证：本机在**部署分布**（gapRange=[100,165]，demo 实际在用的难度，
 * 见 commit da00fe8）上测到的 hazard=0.464%±0.028%（`docs/research/
 * LOCAL-RESULTS-2026-09-06.md` "2026-09-07 追加"一节）。
 *
 * 背景：本文件之前的 `hazard_eval.mjs` 一直用 `game.js` 的
 * `DEFAULT_GAP_RANGE=[85,165]`（训练分布，parity 基准要求，见 da00fe8 提交
 * 说明），云端此前所有 oracle/value-resolution 实验也全部在这个分布上做。
 * 但 demo 从 2026-09-07 起实际部署在更窄的 [100,165] 上，本机在这个分布上
 * 重测发现 hazard 直接减半到 0.464%——已经越过"训练分布 0.939% 减半=0.47%"
 * 这条此前被认为摸不到的目标线。这条云端 routine 之前反复扫的"能不能达到
 * 0.47%"问题，答案在部署分布上其实已经是"是"，只是此前没人在这个分布上
 * 独立验证过。这里用**独立的 JS 实现**复测，和实验 B（训练分布上 1.001% vs
 * 本机 0.939%）同一个交叉验证逻辑。
 *
 * 只改了两处：gapRange 传 [100,165]（其余和 hazard_eval.mjs 完全一致，包括
 * 权重、种子、frame_skip）；决策上限对齐本机默认口径 --max-decisions=2000
 * （`eval.py` 默认值），即 maxFrames = 2000*4 = 8000，这样和本机产出 0.464%
 * 用的口径完全一致，不是另起一个更宽松的上限"作弊"。
 *
 * 用法：node web/tools/hazard_eval_deploy.mjs [局数=150] [每局帧数上限=8000]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FlappyGame, OBS_W, OBS_H } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta-base_s0.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const N_EPISODES = Number(argv[0] || 150);
const MAX_FRAMES = Number(argv[1] || 8000); // = 2000 决策 * frame_skip 4，对齐 eval.py 默认上限
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base，和其余诊断脚本用同一组种子
const FRAME_SKIP = 4;
const FRAME_STACK = 4;
const GAP_RANGE = [100, 165]; // 部署分布（da00fe8），不是 game.js 的默认训练分布 [85,165]

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(FRAME_STACK);

const scores = [];
let deaths = 0;
let totalPipes = 0;
let totalDecisions = 0;
let hitCap = 0;
const t0 = performance.now();

for (let ep = 0; ep < N_EPISODES; ep++) {
  const seed = SEED_BASE + ep;
  const game = new FlappyGame({ seed, hitmasks: HITMASKS, gapRange: GAP_RANGE });
  game.reset();
  let action = 0;
  let i = 0;
  let died = false;
  for (; i < MAX_FRAMES; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obs);
      const arr = i === 0 ? stack.reset(obs) : stack.push(obs);
      action = net.act(arr);
      totalDecisions++;
    }
    if (game.step(action).done) { died = true; break; }
  }
  scores.push(game.score);
  totalPipes += game.score;
  if (died) { deaths += 1; } else { hitCap += 1; }
  const elapsed = (performance.now() - t0) / 1000;
  process.stdout.write(`  第 ${String(ep + 1).padStart(3)} 局 seed=${String(seed).padStart(10)}: `
    + `${String(game.score).padStart(4)} 根  ${died ? '死亡' : '截断(未死)'}`
    + `  累计: 死亡=${deaths} 管道=${totalPipes}  用时=${elapsed.toFixed(0)}s\n`);
}

const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (scores.length - 1));
const hazard = totalPipes > 0 ? deaths / totalPipes : NaN;
const relSE = deaths > 0 ? 1 / Math.sqrt(deaths) : NaN;
const secs = (performance.now() - t0) / 1000;

console.log('\n=========== hazard_eval_deploy 结果（神经网络，部署分布 gapRange=[100,165]） ===========');
console.log(`局数=${N_EPISODES}  死亡=${deaths}  截断(撞${MAX_FRAMES}帧上限,未死)=${hitCap}  截断比例=${(hitCap / N_EPISODES * 100).toFixed(1)}%`);
console.log(`总通过管道数=${totalPipes}  平均分(未做删失校正,仅供参考)=${mean.toFixed(1)}  标准差=${sd.toFixed(1)}`);
console.log(`hazard = ${deaths}/${totalPipes} = ${(hazard * 100).toFixed(3)}%/根`
  + `  相对标准误 ≈ 1/sqrt(${deaths}) = ${(relSE * 100).toFixed(1)}%`);
console.log('对照：本机同分布(gapRange=[100,165], --max-decisions=2000 默认)、400 局实测 0.464% ± 0.028%'
  + '（docs/research/LOCAL-RESULTS-2026-09-06.md "2026-09-07 追加"一节）');
console.log(`本次决策 ${totalDecisions} 次，耗时 ${secs.toFixed(1)}s（${(secs / N_EPISODES).toFixed(1)}s/局）`);
if (hitCap > 0) {
  console.log(`\n注意：${hitCap} 局撞上帧数上限(${MAX_FRAMES}帧=${MAX_FRAMES / FRAME_SKIP}决策)被截断，`
    + `只计入分母暴露量、不计入分子死亡数（右删失标准处理）。截断比例较高是部署分布更简单、`
    + `局更长的直接后果，不是 bug——本机在同一口径下也观察到 400 局里 120 局(30%)撞顶。`);
}
console.log('===================================================================');
