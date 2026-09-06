/**
 * 实验 A（最高优先级）：oracle hazard —— 环境的物理下限。
 *
 * 背景：当前神经网络的 hazard（死亡数/总通过管道数）实测 1.457%/根
 * （`docs/research/WHERE_IS_THE_PROBLEM.md`）。要把均分从 95 提到 200，
 * 需要把它减半到约 0.5-0.7%。这条实验要回答：一个不学习、只靠前瞻的
 * "最强非学习控制器"，hazard 能压到多低？如果连它都压不到目标区间，
 * 说明"减半"不是网络的锅，是环境参数（缝隙/间距/落差上限）画的硬上限，
 * 后续所有算法层面的优化目标都要重定。
 *
 * 控制器 = chaseGap 启发式 + 前瞻安全检查器：
 *   对两个动作各自克隆当前状态、执行该动作、之后用 chaseGap 继续
 *   rollout 到 N 步决策，检查这 N 步内会不会死。两个动作都安全就用
 *   chaseGap 本来的选择；只有一个安全就选那个；两个都不安全（N 步内
 *   必死无法避免）就退回 chaseGap 的选择（死马当活马医，这时死亡记为
 *   "环境真正不可避免"，不是控制器的锅）。
 *
 * 口径：hazard = 死亡局数 / 总通过管道数（右删失处理——撞上决策上限的
 * 局只计入分母的暴露量，不计入分子）。样本量用 seed = 20260906+i，
 * 和 Python 侧固定评测集起点 `eval_seed_base` 对齐。
 *
 * 用法：
 *   node web/tools/oracle_hazard.mjs [局数=400] [前瞻深度N=15] [每局帧数上限=20000]
 *   node web/tools/oracle_hazard.mjs --sweep [局数=100]   // N=5/10/15/25 对比收敛性
 */
import { FlappyGame, PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { makeSeededGame, chaseGap, oracleAction } from './lookahead_lib.mjs';

const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const CTOR_OPTS = { hitmasks: HITMASKS };
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base

const argv = process.argv.slice(2);
const SWEEP = argv.includes('--sweep');
const rest = argv.filter((a) => a !== '--sweep');

/** 跑一局，返回 {pipes, died, frames}。died=false 表示撞上帧数上限被截断。 */
function runEpisode(seed, N, maxFrames) {
  const game = makeSeededGame(seed, CTOR_OPTS);
  let i = 0;
  for (; i < maxFrames; i++) {
    const action = N > 0 ? oracleAction(game, N, CTOR_OPTS, DIMS) : chaseGap(game, DIMS);
    if (game.step(action).done) return { pipes: game.score, died: true, frames: i + 1 };
  }
  return { pipes: game.score, died: false, frames: maxFrames };
}

function runBatch(nEpisodes, N, maxFrames, label) {
  let deaths = 0;
  let totalPipes = 0;
  const scores = [];
  const t0 = performance.now();
  for (let ep = 0; ep < nEpisodes; ep++) {
    const seed = SEED_BASE + ep;
    const r = runEpisode(seed, N, maxFrames);
    totalPipes += r.pipes;
    if (r.died) deaths += 1;
    scores.push(r.pipes);
    if ((ep + 1) % 20 === 0 || ep === nEpisodes - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      process.stdout.write(`  [${label}] ${ep + 1}/${nEpisodes} 局，累计死亡 ${deaths}，`
        + `累计管道 ${totalPipes}，耗时 ${elapsed.toFixed(1)}s\n`);
    }
  }
  const hazard = totalPipes > 0 ? deaths / totalPipes : NaN;
  const relSE = deaths > 0 ? 1 / Math.sqrt(deaths) : NaN;
  const censored = nEpisodes - deaths;
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  return {
    label, N, nEpisodes, deaths, censored, totalPipes, hazard, relSE, mean, scores,
  };
}

function printResult(r) {
  console.log('---------------------------------------------');
  console.log(`[${r.label}] N=${r.N}  局数=${r.nEpisodes}  死亡=${r.deaths}  截断(未死)=${r.censored}`);
  console.log(`总通过管道数=${r.totalPipes}  平均分=${r.mean.toFixed(1)}`);
  console.log(`hazard = ${r.deaths}/${r.totalPipes} = ${(r.hazard * 100).toFixed(3)}%/根`
    + `  相对标准误 ≈ 1/sqrt(${r.deaths}) = ${(r.relSE * 100).toFixed(1)}%`);
  console.log('---------------------------------------------');
}

if (SWEEP) {
  const nEpisodes = Number(rest[0] || 100);
  const depths = [5, 10, 15, 25];
  console.log(`前瞻深度收敛性检查：N = ${depths.join('/')}，每组 ${nEpisodes} 局（同一组种子）\n`);
  const results = [];
  for (const N of depths) {
    const r = runBatch(nEpisodes, N, 20000, `N=${N}`);
    printResult(r);
    results.push(r);
  }
  console.log('\n=========== 收敛性汇总 ===========');
  console.log('N\t死亡\t管道\thazard%\t相对SE%');
  for (const r of results) {
    console.log(`${r.N}\t${r.deaths}\t${r.totalPipes}\t${(r.hazard * 100).toFixed(3)}\t${(r.relSE * 100).toFixed(1)}`);
  }
  const last = results[results.length - 1];
  const prev = results[results.length - 2];
  const stillDropping = last.hazard < prev.hazard - 2 * Math.max(last.relSE, prev.relSE) * last.hazard;
  console.log(stillDropping
    ? '\n警告：hazard 在 N 增大时仍在单调下降且超出噪声范围——这组测的可能只是上界，不是下界，前瞻深度需要继续加大。'
    : '\n hazard 随 N 增大的变化落在噪声范围内（或已不再下降）——大致收敛。');
} else {
  const nEpisodes = Number(rest[0] || 400);
  const N = Number(rest[1] || 15);
  const maxFrames = Number(rest[2] || 20000);
  console.log(`oracle hazard：${nEpisodes} 局，前瞻深度 N=${N}，帧数上限=${maxFrames}\n`);
  const r = runBatch(nEpisodes, N, maxFrames, `oracle-N${N}`);
  printResult(r);

  // 顺带跑一个纯 chaseGap（N=0，即不加前瞻）做对照，量化前瞻带来的提升
  console.log('\n对照：不加前瞻的纯 chaseGap 启发式（同一批种子）\n');
  const r0 = runBatch(nEpisodes, 0, maxFrames, 'chaseGap-only');
  printResult(r0);
}
