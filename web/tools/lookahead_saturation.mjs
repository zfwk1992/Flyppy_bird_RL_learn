/**
 * 实验 B【次优先】把 oracle 从"存在性证明"往"更接近下界"推一步。
 *
 * 上一轮（`docs/research/LAYER0-RESULTS.md` 实验 A）发现 oracle hazard 在
 * N=120 见底（0.706%），N 再增大反而变差（先降后升，非单调），并给出一个
 * **没有专门验证**的推测：前瞻续跑策略用的是 chaseGap 自己，chaseGap 单独
 * 跑的平均寿命只有约 9.6 根管道；当前瞻窗口 N 逼近/超过这个寿命尺度后，
 * "继续跑 N 步会不会死"这个检查对当前动作选择变得不敏感——不管现在选哪个
 * 动作，chaseGap 长期跑下去本来就有相当概率在窗口期内死于其他原因，两个
 * 分支都容易被判"不安全"，检查器的判别力被稀释。
 *
 * 这里直接测这个推测，做两件事：
 *
 *   (1) --sweep：**"两个分支都判不安全"的决策比例**随 N 的变化。
 *       用 N 前瞻本身驱动的轨迹（自洽——用这个 N 选出的轨迹上测这个 N 的
 *       退化率，而不是用另一个策略生成轨迹再拿这个 N 去评估，因为退化率
 *       本来就该是"用这套前瞻实际跑起来会遇到多少次判不出"）。如果这个
 *       比例在 N>120 后持续上升，上一轮的推测就坐实。
 *
 *   (2) --fix：一个修正尝试。两个分支都不安全时，**不退回 chaseGap 启发式
 *       （v1，上一轮的做法），改成选"用 survivalSteps 打分、撑得更久"的
 *       那个分支（v2）**。在 N=120（上一轮测到的最好点）上对比 v1/v2 的
 *       hazard，看修正能不能把 0.706% 压得更低。
 *
 * 前瞻本身（chaseGap 续跑）不调用网络、不渲染像素，只是纯物理 step，
 * 复用 lookahead_lib.mjs 里上一轮已经自测过克隆保真度的工具
 * （lookahead_selfcheck.mjs，5 个 seed 逐帧比对，全部 PASS），这里不重复
 * 验证克隆本身，只新增了 survivalSteps（和 survivesLookahead 同一套物理
 * 前瞻，只是返回值从布尔换成"实际撑了几步"，逻辑上是同一件事的两种读法）。
 *
 * 用法：
 *   node web/tools/lookahead_saturation.mjs --sweep [局数=100]
 *   node web/tools/lookahead_saturation.mjs --fix [局数=400] [N=120]
 */
import { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import {
  makeSeededGame, chaseGap, survivesLookahead, survivalSteps,
} from './lookahead_lib.mjs';

const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const CTOR_OPTS = { hitmasks: HITMASKS };
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base，与实验 A/B 同一组种子

const argv = process.argv.slice(2);
const MODE = argv.includes('--fix') ? 'fix' : 'sweep';
const rest = argv.filter((a) => a !== '--sweep' && a !== '--fix');

/**
 * oracle 决策 + 统计"两个分支都不安全"的次数。
 * version: 'v1' = 都不安全时退回 chaseGap 启发式（上一轮做法）
 *          'v2' = 都不安全时选 survivalSteps 更大的那个分支
 */
function oracleStep(game, N, version, stats) {
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

function runEpisode(seed, N, version, maxFrames, stats) {
  const game = makeSeededGame(seed, CTOR_OPTS);
  for (let i = 0; i < maxFrames; i++) {
    const action = oracleStep(game, N, version, stats);
    if (game.step(action).done) return { pipes: game.score, died: true };
  }
  return { pipes: game.score, died: false };
}

function runBatch(nEpisodes, N, version, maxFrames, label) {
  const stats = { total: 0, bothUnsafe: 0 };
  let deaths = 0; let totalPipes = 0;
  const t0 = performance.now();
  for (let ep = 0; ep < nEpisodes; ep++) {
    const r = runEpisode(SEED_BASE + ep, N, version, maxFrames, stats);
    totalPipes += r.pipes;
    if (r.died) deaths += 1;
    if ((ep + 1) % 20 === 0 || ep === nEpisodes - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      process.stdout.write(`  [${label}] ${ep + 1}/${nEpisodes} 局，累计死亡 ${deaths}，`
        + `累计管道 ${totalPipes}，耗时 ${elapsed.toFixed(1)}s\n`);
    }
  }
  const hazard = totalPipes > 0 ? deaths / totalPipes : NaN;
  const relSE = deaths > 0 ? 1 / Math.sqrt(deaths) : NaN;
  const bothUnsafeFrac = stats.total > 0 ? stats.bothUnsafe / stats.total : NaN;
  return {
    label, N, version, nEpisodes, deaths, totalPipes, hazard, relSE, bothUnsafeFrac, decisions: stats.total,
  };
}

if (MODE === 'sweep') {
  const nEpisodes = Number(rest[0] || 100);
  const depths = [60, 80, 120, 160, 200, 240, 300];
  console.log(`"两个分支都不安全"比例 vs N：局数=${nEpisodes}，N=${depths.join('/')}\n`);
  const results = [];
  for (const N of depths) {
    const r = runBatch(nEpisodes, N, 'v1', 30000, `N=${N}`);
    console.log(`  >> [N=${N}] 决策数=${r.decisions}  两分支都不安全比例=`
      + `${(r.bothUnsafeFrac * 100).toFixed(2)}%  hazard=${(r.hazard * 100).toFixed(3)}%`
      + `  平均分=${(r.totalPipes / r.nEpisodes).toFixed(1)}`);
    results.push(r);
  }
  console.log('\n=========== 汇总 ===========');
  console.log('N\t决策数\t两分支都不安全%\thazard%\t平均分');
  for (const r of results) {
    console.log(`${r.N}\t${r.decisions}\t${(r.bothUnsafeFrac * 100).toFixed(2)}\t\t`
      + `${(r.hazard * 100).toFixed(3)}\t${(r.totalPipes / r.nEpisodes).toFixed(1)}`);
  }
  const at120 = results.find((r) => r.N === 120);
  const above160 = results.filter((r) => r.N >= 160);
  const rising = at120 && above160.length > 0 && above160.every((r) => r.bothUnsafeFrac > at120.bothUnsafeFrac);
  console.log(rising
    ? '\n"两个分支都不安全"的比例在 N>120 后持续上升，与上一轮的推测方向一致：'
      + 'chaseGap 续跑策略自身寿命尺度在稀释前瞻判别力。'
    : '\n"两个分支都不安全"的比例在 N>120 后没有持续上升，上一轮的推测没有得到这批数据支持。');
} else {
  const nEpisodes = Number(rest[0] || 400);
  const N = Number(rest[1] || 120);
  console.log(`v1(退回启发式) vs v2(选存活更久分支) 对比：局数=${nEpisodes}，N=${N}\n`);
  const r1 = runBatch(nEpisodes, N, 'v1', 30000, 'v1');
  const r2 = runBatch(nEpisodes, N, 'v2', 30000, 'v2');
  console.log('\n=========== 对比 ===========');
  console.log(`v1  hazard=${(r1.hazard * 100).toFixed(3)}%  相对SE=${(r1.relSE * 100).toFixed(1)}%`
    + `  两分支都不安全=${(r1.bothUnsafeFrac * 100).toFixed(2)}%`
    + `  平均分=${(r1.totalPipes / r1.nEpisodes).toFixed(1)}`);
  console.log(`v2  hazard=${(r2.hazard * 100).toFixed(3)}%  相对SE=${(r2.relSE * 100).toFixed(1)}%`
    + `  两分支都不安全=${(r2.bothUnsafeFrac * 100).toFixed(2)}%`
    + `  平均分=${(r2.totalPipes / r2.nEpisodes).toFixed(1)}`);
  const seDiff = Math.sqrt((r1.relSE * r1.hazard) ** 2 + (r2.relSE * r2.hazard) ** 2);
  const diff = r1.hazard - r2.hazard;
  console.log(`\nv1-v2 差 = ${(diff * 100).toFixed(3)} 个百分点，合并标准误 ≈ `
    + `${(seDiff * 100).toFixed(3)} 个百分点（${Math.abs(diff / seDiff).toFixed(1)} 个标准误）`);
  console.log(Math.abs(diff) > 2 * seDiff
    ? (diff > 0 ? '-> v2 显著更好（hazard 更低），修正有效。' : '-> v2 显著更差，修正没有帮助甚至有害。')
    : '-> 差距在 2 个标准误以内，没有区别。');
}
