/**
 * 实验 B：用**神经网络**（当前模型 runs/base_s0/final.pt）跑 N 关，
 * 按 hazard 口径（死亡数/总通过管道数）报告，并与 oracle_hazard.mjs
 * 的非学习前瞻 oracle 做对比。
 *
 * 动机：现有评测口径是"40 关均值"，相对标准误约 20%（分数近似几何分布，
 * 标准差≈均值，见 docs/research/WHERE_IS_THE_PROBLEM.md）。换成 hazard 后
 * 相对 SE = 1/sqrt(死亡数)，400 关约 5%。而且带网络前向一局约需要跑一整局
 * （帧数上限 20000），比看均分更贵一点点但信息量大得多——不该只跑 40 关。
 *
 * **权重**：显式导入 `weights-meta-base_s0.js`（对应 runs/base_s0/final.pt），
 * 不用 nn.js 默认的 weights-meta.js（那份是网页 demo 在用的旧模型）。
 * 按任务要求：不改 web/nn.js 的默认导入，这里手动传参传进去。
 *
 * 用法：node web/tools/hazard_eval.mjs [局数=150] [每局帧数上限=20000]
 *
 * 时间预算：带网络前向一局约 20 秒（parity_ai.md 实测），400 局约 2.2 小时，
 * 超出单次云端运行预算——这里默认先跑 100-150 局把工具和口径做对，
 * 400 局留给本机（GPU 更快）。
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
const MAX_FRAMES = Number(argv[1] || 20000);
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base，和 oracle_hazard.mjs 用同一组种子
const FRAME_SKIP = 4;
const FRAME_STACK = 4;

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
const t0 = performance.now();

for (let ep = 0; ep < N_EPISODES; ep++) {
  const seed = SEED_BASE + ep;
  const game = new FlappyGame({ seed, hitmasks: HITMASKS });
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
  if (died) deaths += 1;
  const elapsed = (performance.now() - t0) / 1000;
  process.stdout.write(`  第 ${String(ep + 1).padStart(3)} 局 seed=${String(seed).padStart(10)}: `
    + `${String(game.score).padStart(4)} 根  ${died ? '死亡' : '截断(未死)'}`
    + `  累计: 死亡=${deaths} 管道=${totalPipes}  用时=${elapsed.toFixed(0)}s\n`);
}

const sorted = [...scores].sort((a, b) => a - b);
const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (scores.length - 1));
const censored = N_EPISODES - deaths;
const hazard = totalPipes > 0 ? deaths / totalPipes : NaN;
const relSE = deaths > 0 ? 1 / Math.sqrt(deaths) : NaN;
const secs = (performance.now() - t0) / 1000;

console.log('\n=========== hazard_eval 结果（神经网络，base_s0/final.pt） ===========');
console.log(`局数=${N_EPISODES}  死亡=${deaths}  截断(未死)=${censored}`);
console.log(`总通过管道数=${totalPipes}  平均分(未做删失校正,仅供参考)=${mean.toFixed(1)}  标准差=${sd.toFixed(1)}`);
console.log(`hazard = ${deaths}/${totalPipes} = ${(hazard * 100).toFixed(3)}%/根`
  + `  相对标准误 ≈ 1/sqrt(${deaths}) = ${(relSE * 100).toFixed(1)}%`);
console.log(`旧口径参照：docs/research/WHERE_IS_THE_PROBLEM.md 实测 1.457%/根（73 局未截断，均值 68.6 根）`);
console.log(`本次决策 ${totalDecisions} 次，耗时 ${secs.toFixed(1)}s（${(secs / N_EPISODES).toFixed(1)}s/局）`);
if (censored > 0) {
  console.log(`\n注意：${censored} 局撞上帧数上限(${MAX_FRAMES}帧)被截断，只计入分母暴露量、`
    + `不计入分子死亡数（右删失标准处理）。若忽略这一点直接用"总局数"当分母，`
    + `hazard 会被低估——截断局通常是表现最好的局，混进未截断的平均会拉高均值、压低算出的 hazard。`);
}
console.log('=========================================================');
