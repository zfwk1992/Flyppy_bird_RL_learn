/**
 * 把实验 A/B 的 oracle（chaseGap + 前瞻安全检查）搬到**部署分布**
 * （gapRange=[100,165]，demo 实际在用的难度，见 commit da00fe8）上测一遍，
 * 和 `hazard_eval_deploy.mjs`（同一分布上的神经网络交叉验证）配成一对。
 *
 * 背景：`oracle_hazard.mjs` / `lookahead_saturation.mjs` 之前测的 oracle
 * hazard（N=120 v1: 0.706%，v2: 0.559%）全部在 `game.js` 默认的**训练分布**
 * [85,165] 上，那时候的目标线是"训练分布 0.939% 减半 = 0.47%"，oracle 没
 * 摸到。但本机后来发现部署分布 [100,165] 上网络自己就已经到 0.464%，比训练
 * 分布上最好的 oracle（v2 0.559%）还低——这不是矛盾，是换了一把更松的尺子。
 * 这里直接在部署分布上重跑 oracle，看看这把更松的尺子下，非学习 oracle 现在
 * 什么水平，网络与它的相对位置有没有变化。
 *
 * 复用 lookahead_saturation.mjs 的 v1/v2 决策逻辑（只是 CTOR_OPTS 换成
 * gapRange=[100,165]），不改任何已有文件。前瞻本身是纯物理 step，不调用
 * 网络，CPU 计算量比 hazard_eval_deploy.mjs 小得多。
 *
 * 用法：node web/tools/oracle_hazard_deploy.mjs [局数=200] [N=120] [每局帧数上限=30000]
 */
import { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import {
  makeSeededGame, chaseGap, survivesLookahead, survivalSteps,
} from './lookahead_lib.mjs';

const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const CTOR_OPTS = { hitmasks: HITMASKS, gapRange: [100, 165] }; // 部署分布，不是 game.js 默认训练分布
const SEED_BASE = 20260906; // 对齐其余诊断脚本用的同一组种子

const argv = process.argv.slice(2);
const N_EPISODES = Number(argv[0] || 200);
const N = Number(argv[1] || 120);
const MAX_FRAMES = Number(argv[2] || 30000);

function oracleStep(game, version, stats) {
  const heuristic = chaseGap(game, DIMS);
  const safe0 = survivesLookahead(game, 0, N, CTOR_OPTS, DIMS);
  const safe1 = survivesLookahead(game, 1, N, CTOR_OPTS, DIMS);
  stats.total += 1;
  if (safe0 && safe1) return heuristic;
  if (safe0) return 0;
  if (safe1) return 1;
  stats.bothUnsafe += 1;
  if (version === 'v1') return heuristic;
  const s0 = survivalSteps(game, 0, N, CTOR_OPTS, DIMS);
  const s1 = survivalSteps(game, 1, N, CTOR_OPTS, DIMS);
  if (s0 === s1) return heuristic;
  return s0 > s1 ? 0 : 1;
}

function runEpisode(seed, version, stats) {
  const game = makeSeededGame(seed, CTOR_OPTS);
  for (let i = 0; i < MAX_FRAMES; i++) {
    const action = oracleStep(game, version, stats);
    if (game.step(action).done) return { pipes: game.score, died: true };
  }
  return { pipes: game.score, died: false };
}

function runBatch(version, label) {
  const stats = { total: 0, bothUnsafe: 0 };
  let deaths = 0; let totalPipes = 0; let censored = 0;
  const t0 = performance.now();
  for (let ep = 0; ep < N_EPISODES; ep++) {
    const r = runEpisode(SEED_BASE + ep, version, stats);
    totalPipes += r.pipes;
    if (r.died) { deaths += 1; } else { censored += 1; }
    if ((ep + 1) % 20 === 0 || ep === N_EPISODES - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      process.stdout.write(`  [${label}] ${ep + 1}/${N_EPISODES} 局，累计死亡 ${deaths}，`
        + `累计管道 ${totalPipes}，耗时 ${elapsed.toFixed(1)}s\n`);
    }
  }
  const hazard = totalPipes > 0 ? deaths / totalPipes : NaN;
  const relSE = deaths > 0 ? 1 / Math.sqrt(deaths) : NaN;
  const bothUnsafeFrac = stats.total > 0 ? stats.bothUnsafe / stats.total : NaN;
  return {
    label, version, deaths, censored, totalPipes, hazard, relSE, bothUnsafeFrac,
    mean: totalPipes / N_EPISODES,
  };
}

console.log(`oracle hazard（部署分布 gapRange=[100,165]）：局数=${N_EPISODES}，N=${N}\n`);
const r1 = runBatch('v1', 'v1-退回启发式');
const r2 = runBatch('v2', 'v2-选存活更久分支');

console.log('\n=========== 结果（部署分布） ===========');
for (const r of [r1, r2]) {
  console.log(`[${r.label}] 死亡=${r.deaths}  截断=${r.censored}  总管道=${r.totalPipes}  平均分=${r.mean.toFixed(1)}`);
  console.log(`  hazard = ${(r.hazard * 100).toFixed(3)}%/根  相对标准误=${(r.relSE * 100).toFixed(1)}%`
    + `  两分支都不安全比例=${(r.bothUnsafeFrac * 100).toFixed(2)}%`);
}
const seDiff = Math.sqrt((r1.relSE * r1.hazard) ** 2 + (r2.relSE * r2.hazard) ** 2);
const diff = r1.hazard - r2.hazard;
console.log(`\nv1-v2 差 = ${(diff * 100).toFixed(3)} 个百分点，合并标准误 ≈ ${(seDiff * 100).toFixed(3)} `
  + `个百分点（${Math.abs(diff / seDiff).toFixed(1)} 个标准误）`);
console.log('对照：本机同分布网络基线 0.464% ± 0.028%（400 局，--max-decisions 默认 2000）');
console.log('=========================================');
