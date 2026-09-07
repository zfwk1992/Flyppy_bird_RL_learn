/**
 * 价值分辨率假说的"方向"检验——补上上一轮 value_resolution.mjs /
 * value_resolution_deploy.mjs 明确标注过的局限。
 *
 * 上一轮测的是"网络在关键状态上 |Q1-Q0| 是不是明显更大"，两次（训练分布/
 * 部署分布）都测到关键状态比无差别状态大 140-160 倍——但这只回答"网络觉得
 * 这个决策要紧"，不回答"网络选对了没有"。`LAYER0-RESULTS.md` 任务 A 的
 * 局限段原话："价值分辨率假说更精确的版本可能是'关键状态里，网络给出大
 * 差距，但差距的方向经常选错'，这条数据没有直接检验方向对不对"。
 * 部署分布那次 session 的结尾也把这个具体缺口列成"留给下一轮"。这个脚本
 * 就是直接测这一条。
 *
 * 定义：
 *   - 关键状态：K 步前瞻下恰好一个动作安全（另一个必死）——lookahead_lib.mjs
 *     里同一套 chaseGap 续跑 + survivesLookahead，与 value_resolution.mjs
 *     完全一致的分类标准，不是另起一套口径。
 *   - "选错方向"：网络的贪婪动作（argmax Q）不等于前瞻判定的那个安全动作。
 *
 * 判据：
 *   - 选错方向比例接近 0 -> 网络在关键状态上"知道要紧"且"选对了"，
 *     价值分辨率假说的方向版本被弱化
 *   - 选错方向比例明显偏高（两位数百分比量级） -> 支持价值分辨率假说的
 *     方向版本：网络能感知重要性，但经常选错边
 *
 * 额外做一件 value_resolution.mjs 没做的事：把"选错方向"和**真实轨迹**里
 * 该局最终是否死亡、还剩几次决策才死联系起来（不是前瞻的反事实 rollout，
 * 是网络自己实际驱动出来的真实结局）。这不是严格的因果证明——网络选错
 * 之后仍然按自己的策略继续走，不是按 chase_gap 继续走，"选错"和"是不是
 * chase_gap 视角里的必死"本来就是两套动力学——但至少能看一眼"前瞻判定的
 * 错误选择"是否真的对应"这局很快就死了"这个更朴素的相关性，如实标注这是
 * 相关性观察，不是反事实验证。
 *
 * 用法：node web/tools/value_resolution_direction.mjs [关键状态探针数] [K] [gap下界] [gap上界]
 * 例：
 *   node web/tools/value_resolution_direction.mjs 1500 120            # 训练分布 [85,165]（默认，不传后两个参数）
 *   node web/tools/value_resolution_direction.mjs 1500 120 100 165    # 部署分布
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  OBS_W, OBS_H, PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT,
} from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta-base_s0.js';
import { makeSeededGame, cloneGame, survivesLookahead } from './lookahead_lib.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base，和上一轮 value_resolution.mjs 同一组种子

const N_PROBE = Number(process.argv[2] || 1500); // 这是"关键状态"探针数目标，不是总探针数——
// 关键状态只占全部决策的约 20%（上一轮 value_resolution.mjs 测到 21.1%/20.0%），
// 且本脚本要求每局跑到真正结束才能知道死亡结局，不能像 value_resolution.mjs
// 那样凑够数就在局中截断，所以目标数刻意定得比那边的 4000 小，控制预算。
const K = Number(process.argv[3] || 120);
const GAP_LO = process.argv[4] !== undefined ? Number(process.argv[4]) : null;
const GAP_HI = process.argv[5] !== undefined ? Number(process.argv[5]) : null;
const GAP_RANGE = GAP_LO !== null && GAP_HI !== null ? [GAP_LO, GAP_HI] : null; // null = game.js 默认训练分布 [85,165]
const CTOR_OPTS = { hitmasks: HITMASKS, ...(GAP_RANGE ? { gapRange: GAP_RANGE } : {}) };
const DIST_LABEL = GAP_RANGE ? `部署分布 gapRange=${JSON.stringify(GAP_RANGE)}` : '训练分布 gapRange=[85,165]（game.js 默认）';

const FRAME_SKIP = 4;
const STRIDE = 3; // 与 value_resolution.mjs 相同：每隔几个决策采一次样
const PER_EP_CAP = Math.max(1, Math.floor(N_PROBE / 8));
const DEATH_WINDOW = 10; // "很快就死"的窗口：接下来几次决策内死亡算"很快"

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obsBuf = new Float32Array(OBS_W * OBS_H);

console.log(`[${DIST_LABEL}] 探针数目标=${N_PROBE}  前瞻深度K=${K}  单局采样上限=${PER_EP_CAP}`);
console.log('探针来源：网络自己驱动游戏的 on-policy 轨迹（与 value_resolution.mjs 同一套方法）\n');

// 只收集"关键状态"（恰好一个动作安全）的探针，附带该局的死亡信息（跑完该局才知道）。
// 注意：每一局都跑到真正结束（done）或撞上帧数上限，不会为了凑够探针数在
// 局中途截断——因为要读到"这局最终是否死亡、还剩几次决策"，中途截断会让
// 这个信息变成未知数，宁可多跑几局，也不在局内部半途而废。
const criticalRows = [];
let ep = 0;
const t0 = performance.now();
while (criticalRows.length < N_PROBE && ep < 1_000_000) {
  const game = makeSeededGame(SEED_BASE + ep, CTOR_OPTS);
  const stack = new FrameStack(4);
  let action = 0;
  let decisionIdx = 0;
  let takenThisEp = 0;
  const epCriticalBuf = []; // {decisionIdx, matched, gapQ}
  let died = false;
  let finalDecisionIdx = -1;
  for (let i = 0; i < 20000; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obsBuf);
      const arr = i === 0 ? stack.reset(obsBuf) : stack.push(obsBuf);
      const q = net.forward(arr);
      const q0 = q[0]; const q1 = q[1];
      action = q1 > q0 ? 1 : 0;

      if (decisionIdx % STRIDE === 0 && takenThisEp < PER_EP_CAP) {
        const clone = cloneGame(game, CTOR_OPTS);
        const safe0 = survivesLookahead(clone, 0, K, CTOR_OPTS, DIMS);
        const safe1 = survivesLookahead(clone, 1, K, CTOR_OPTS, DIMS);
        if (safe0 !== safe1) { // 恰好一个安全 = 关键状态
          const safeAction = safe0 ? 0 : 1;
          epCriticalBuf.push({
            decisionIdx, matched: action === safeAction, gapQ: Math.abs(q1 - q0),
          });
          takenThisEp += 1;
        }
      }
      decisionIdx += 1;
    }
    if (game.step(action).done) { died = true; finalDecisionIdx = decisionIdx - 1; break; }
  }
  if (!died) finalDecisionIdx = decisionIdx - 1; // 撞帧数上限截断，用最后一次决策下标近似
  // 回填死亡信息：只有真正死亡的局才能算"还剩几次决策会死"，截断局不算。
  for (const row of epCriticalBuf) {
    criticalRows.push({
      ...row,
      diedEventually: died,
      decisionsToDeath: died ? (finalDecisionIdx - row.decisionIdx) : null,
    });
  }
  ep += 1;
  if (ep % 5 === 0 || criticalRows.length >= N_PROBE) {
    const elapsed = (performance.now() - t0) / 1000;
    process.stdout.write(`  已跑 ${ep} 局，采到关键状态探针 ${criticalRows.length}/${N_PROBE} 个，`
      + `耗时 ${elapsed.toFixed(1)}s\n`);
  }
}

// ---------------------------------------------------------------------
// 汇总统计
// ---------------------------------------------------------------------
function quantile(a, q) {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos); const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
function median(a) { return quantile(a, 0.5); }
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }

console.log(`\n采到 ${criticalRows.length} 个关键状态探针，跑了 ${ep} 局，`
  + `耗时 ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

const wrong = criticalRows.filter((r) => !r.matched);
const right = criticalRows.filter((r) => r.matched);

console.log('=========== 关键判据：关键状态里网络选对了没有 ===========');
console.log(`关键状态总数 n=${criticalRows.length}`);
console.log(`选错方向（网络贪婪动作 != 前瞻判定的安全动作）：${wrong.length} 个 `
  + `(${(wrong.length / criticalRows.length * 100).toFixed(1)}%)`);
console.log(`选对方向：${right.length} 个 (${(right.length / criticalRows.length * 100).toFixed(1)}%)`);

console.log('\n=========== |Q1-Q0| 分布：选对 vs 选错（关键状态内部） ===========');
for (const [label, arr] of [['选对', right], ['选错', wrong]]) {
  const gs = arr.map((r) => r.gapQ);
  if (!gs.length) { console.log(`${label}\tn=0`); continue; }
  console.log(`${label}\tn=${gs.length}\tQ25=${quantile(gs, 0.25).toFixed(3)}\t`
    + `中位=${median(gs).toFixed(3)}\tQ75=${quantile(gs, 0.75).toFixed(3)}\t均值=${mean(gs).toFixed(3)}`);
}

console.log('\n=========== 相关性观察（不是反事实验证）：选错方向后这局是不是很快就死了 ===========');
console.log(`（"很快"定义为接下来 ${DEATH_WINDOW} 次决策内；只统计真正死亡的局，截断局不算）`);
for (const [label, arr] of [['选对方向', right], ['选错方向', wrong]]) {
  const withDeath = arr.filter((r) => r.diedEventually && r.decisionsToDeath !== null);
  const diedTotal = arr.filter((r) => r.diedEventually).length;
  const soon = withDeath.filter((r) => r.decisionsToDeath <= DEATH_WINDOW).length;
  console.log(`${label}: n=${arr.length}，所在局最终死亡=${diedTotal} `
    + `(${(diedTotal / arr.length * 100).toFixed(1)}%)，其中 ${DEATH_WINDOW} 次决策内死亡=${soon} `
    + `(${arr.length ? (soon / arr.length * 100).toFixed(1) : '0.0'}%)`);
}

console.log('\n=========== 结论 ===========');
const wrongFrac = wrong.length / criticalRows.length;
if (wrongFrac < 0.05) {
  console.log(`选错方向比例 ${(wrongFrac * 100).toFixed(1)}% 很低 -> 网络在关键状态上不仅"觉得要紧"，`
    + '而且基本选对了。价值分辨率假说的"方向经常选错"这个更精确版本被这条数据弱化。');
} else if (wrongFrac < 0.2) {
  console.log(`选错方向比例 ${(wrongFrac * 100).toFixed(1)}% 不算低但也不算高 -> 网络大多数时候选对，`
    + '但确实存在一部分"觉得要紧但选错"的关键决策，价值分辨率假说的方向版本部分成立，不能完全排除。');
} else {
  console.log(`选错方向比例 ${(wrongFrac * 100).toFixed(1)}% 偏高 -> 支持价值分辨率假说的方向版本：`
    + '网络能感知到决策重要，但经常选错边，这与 hazard 仍未追平最强 oracle 的量级可能有直接关系。');
}
console.log('\n局限：这里的"选错"用 chase_gap 续跑视角的前瞻判定，网络实际选错之后走的是自己的策略，'
  + '不是 chase_gap 的续跑轨迹，两套动力学不同——"选错后这局是否很快死亡"是真实轨迹上的相关性观察，'
  + '不是"如果选对了会不会活下来"的反事实证明，不能倒推因果。');
