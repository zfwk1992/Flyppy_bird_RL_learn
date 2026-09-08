/**
 * 价值分辨率假说的"方向"检验遗留的最后一个开放尾巴：真正的反事实。
 *
 * `value_resolution_direction.mjs` 发现约 22% 的关键状态网络"选错方向"
 * （贪婪动作 != 前瞻判定的安全动作），但那次测到一个反直觉的相关性——
 * "选错方向"之后这局并没有比"选对方向"更快死。该脚本当时明确标注了原因：
 * 比的是"前瞻判定的安全动作"（chase_gap 续跑）vs"网络实际后续怎么走"
 * （网络自己的策略续跑），两套动力学不同，那只是相关性观察，不是反事实证明——
 * "网络自己后续的决策部分弥补了这次选错造成的风险"这个猜测，**没有被直接
 * 检验过**，原文写"这次没有做，标注为局限"。
 *
 * 这个脚本直接做那个反事实：在每一个关键状态上，把游戏状态和帧栈**分叉**
 * 成两个独立世界——第一步分别强制走 0 / 1，之后两个分支都用**网络自己的
 * 贪婪策略**（不是 chase_gap）续跑 H 个决策，比较两个分支在 H 步内的死亡率。
 * 这是同一个状态的两条"如果当时选了另一个动作，网络自己接下来会怎么走"的
 * 平行世界，是真正的反事实（不是"前瞻 vs 真实轨迹"的相关性）。
 *
 * 关键对比在"选错方向"这个子集里：如果强制走"安全"分支的死亡率明显低于
 * 强制走"不安全"分支（网络实际会选的那个），说明纠正这次选择确实能救命——
 * 也就是"选错但没有更快死"是被网络自己后续的决策部分弥补了；如果两个分支
 * 死亡率接近，说明这一次关键决策本身对最终生死影响不大，真正决定生死的是
 * 更长的轨迹，价值分辨率问题即使修好也不足以大幅降低 hazard。
 *
 * "选对方向"的状态同时作为一个方法学健全性检查：这里前瞻判定的"安全"动作
 * 正好是网络本来就要选的动作，如果连这个子集里"不安全分支死亡率更高"这个
 * 最基本的方向都测不出来，说明 H/K 的选择或分叉机制本身有问题，不该直接
 * 相信"选错方向"子集的结论。
 *
 * 用法：node web/tools/wrong_direction_counterfactual.mjs [关键状态目标数=300] [K=120] [H=20] [gap下界] [gap上界]
 * 例：
 *   node web/tools/wrong_direction_counterfactual.mjs 300 120 20            # 训练分布 [85,165]
 *   node web/tools/wrong_direction_counterfactual.mjs 300 120 20 100 165    # 部署分布
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
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base，和此前所有价值分辨率实验同一组种子

const N_CRITICAL = Number(process.argv[2] || 300); // 关键状态探针数目标
const K = Number(process.argv[3] || 120); // 分类用的前瞻深度，与此前所有价值分辨率实验一致
const H = Number(process.argv[4] || 20); // 分叉之后每个分支续跑几个决策（含被强制的第一步）
const GAP_LO = process.argv[5] !== undefined ? Number(process.argv[5]) : null;
const GAP_HI = process.argv[6] !== undefined ? Number(process.argv[6]) : null;
const GAP_RANGE = GAP_LO !== null && GAP_HI !== null ? [GAP_LO, GAP_HI] : null;
const CTOR_OPTS = { hitmasks: HITMASKS, ...(GAP_RANGE ? { gapRange: GAP_RANGE } : {}) };
const DIST_LABEL = GAP_RANGE ? `部署分布 gapRange=${JSON.stringify(GAP_RANGE)}` : '训练分布 gapRange=[85,165]（game.js 默认）';

const FRAME_SKIP = 4;
const STRIDE = 3; // 与 value_resolution.mjs / value_resolution_direction.mjs 相同
const PER_EP_CAP = Math.max(1, Math.floor(N_CRITICAL / 6));

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

// 分叉分支用独立的渲染缓冲区，避免和外层主轨迹的采样循环互相踩踏
// （单线程同步执行，其实复用一份也不会有竞态，这里分开只是让调用关系更清楚）。
const redMain = new Uint8Array(288 * 512);
const obsMain = new Float32Array(OBS_W * OBS_H);
const redBranch = new Uint8Array(288 * 512);
const obsBranch = new Float32Array(OBS_W * OBS_H);

/**
 * 把一个分支从 (baseGame, baseStack) 这一帧分叉出去：第一个决策强制走
 * forcedFirstAction，之后 H-1 个决策全部用网络自己的贪婪策略（net.act）。
 * 返回 H 步内是否死亡。每次调用都克隆一份全新状态，不污染传入对象。
 */
function runBranch(baseGame, baseStack, forcedFirstAction) {
  const g = cloneGame(baseGame, CTOR_OPTS);
  const st = new FrameStack(4);
  st.array.set(baseStack.array);
  let action = forcedFirstAction;
  for (let d = 0; d < H; d++) {
    for (let f = 0; f < FRAME_SKIP; f++) {
      if (g.step(action).done) return true; // 死亡
    }
    if (d === H - 1) break; // 已经活过 H 个决策，不需要再算下一步动作
    renderRed(g, redBranch);
    downsample(redBranch, obsBranch);
    const arr = st.push(obsBranch);
    action = net.act(arr);
  }
  return false; // H 步内没有死亡
}

console.log(`[${DIST_LABEL}] 目标关键状态数=${N_CRITICAL}  分类前瞻K=${K}  分叉后续跑H=${H} 决策（网络自己的策略）`);
console.log('这是真正的反事实：同一状态分叉两个世界，第一步强制走 0/1，之后都用网络续跑，不是前瞻 vs 真实轨迹的相关性\n');

const rows = []; // { matched, safeAction, diedSafeBranch, diedUnsafeBranch }
let ep = 0;
const t0 = performance.now();
while (rows.length < N_CRITICAL && ep < 1_000_000) {
  const game = makeSeededGame(SEED_BASE + ep, CTOR_OPTS);
  const stack = new FrameStack(4);
  let action = 0;
  let decisionIdx = 0;
  let takenThisEp = 0;
  for (let i = 0; i < 20000; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, redMain);
      downsample(redMain, obsMain);
      const arr = i === 0 ? stack.reset(obsMain) : stack.push(obsMain);
      action = net.act(arr);

      if (decisionIdx % STRIDE === 0 && takenThisEp < PER_EP_CAP) {
        const probe = cloneGame(game, CTOR_OPTS);
        const safe0 = survivesLookahead(probe, 0, K, CTOR_OPTS, DIMS);
        const safe1 = survivesLookahead(probe, 1, K, CTOR_OPTS, DIMS);
        if (safe0 !== safe1) { // 恰好一个安全 = 关键状态
          const safeAction = safe0 ? 0 : 1;
          const unsafeAction = 1 - safeAction;
          const matched = action === safeAction;
          const diedSafeBranch = runBranch(game, stack, safeAction);
          const diedUnsafeBranch = runBranch(game, stack, unsafeAction);
          rows.push({
            matched, diedSafeBranch, diedUnsafeBranch,
          });
          takenThisEp += 1;
          // 和 value_resolution.mjs 一致：本局配额用完就提前结束这一局，
          // 不必再把整局跑到自然死亡——分叉测试本身已经比只分类贵得多，
          // 不能再让"跑满整局"的浪费叠加上去。
          if (rows.length >= N_CRITICAL || takenThisEp >= PER_EP_CAP) break;
        }
      }
      decisionIdx += 1;
    }
    if (game.step(action).done) break;
  }
  ep += 1;
  if (ep % 3 === 0 || rows.length >= N_CRITICAL) {
    const elapsed = (performance.now() - t0) / 1000;
    process.stdout.write(`  已跑 ${ep} 局，采到关键状态 ${rows.length}/${N_CRITICAL} 个，耗时 ${elapsed.toFixed(1)}s\n`);
  }
}

console.log(`\n采到 ${rows.length} 个关键状态，跑了 ${ep} 局，耗时 ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

function binomSE(k, n) { return n ? Math.sqrt(((k / n) * (1 - k / n)) / n) : NaN; }

function report(label, arr) {
  const n = arr.length;
  if (!n) { console.log(`${label}: n=0（跳过）`); return; }
  let safeDied = 0; let unsafeDied = 0;
  let bothDied = 0; let neitherDied = 0; let safeOnlyDied = 0; let unsafeOnlyDied = 0;
  for (const r of arr) {
    if (r.diedSafeBranch) safeDied += 1;
    if (r.diedUnsafeBranch) unsafeDied += 1;
    if (r.diedSafeBranch && r.diedUnsafeBranch) bothDied += 1;
    else if (!r.diedSafeBranch && !r.diedUnsafeBranch) neitherDied += 1;
    else if (r.diedSafeBranch) safeOnlyDied += 1;
    else unsafeOnlyDied += 1;
  }
  console.log(`${label}  n=${n}`);
  console.log(`  强制走"安全"分支(前瞻判定) H步内死亡率  = ${safeDied}/${n} = ${(safeDied / n * 100).toFixed(1)}%`
    + `  (二项SE=${(binomSE(safeDied, n) * 100).toFixed(1)}pp)`);
  console.log(`  强制走"不安全"分支(前瞻判定) H步内死亡率 = ${unsafeDied}/${n} = ${(unsafeDied / n * 100).toFixed(1)}%`
    + `  (二项SE=${(binomSE(unsafeDied, n) * 100).toFixed(1)}pp)`);
  console.log(`  配对结果（同一状态分叉的两个分支）：都死=${bothDied}  都不死=${neitherDied}`
    + `  只安全分支死=${safeOnlyDied}  只不安全分支死=${unsafeOnlyDied}`);
  const disc = safeOnlyDied + unsafeOnlyDied;
  if (disc > 0) {
    // McNemar 检验的正态近似：只用两个分支结果不一致的那些配对（disc 个），
    // 在"强制安全/不安全无差别"的原假设下，unsafeOnlyDied 应该接近 disc/2。
    const z = (unsafeOnlyDied - safeOnlyDied) / Math.sqrt(disc);
    console.log(`  McNemar 近似 z = ${z.toFixed(2)}（不一致对 n=${disc}；`
      + '|z|>2 视为显著；z>0 = 不安全分支更容易死，支持"纠正选择能救命"；'
      + 'z<0 = 方向相反）');
  } else {
    console.log('  两分支结果完全一致的配对数=0，无法做 McNemar 检验（样本太少或该子集内没有区分度）');
  }
}

console.log('=========== 整体（所有关键状态，不分选对/选错） ===========');
report('全部', rows);

console.log('\n=========== 按网络实际选择拆分（核心对比） ===========');
report('选对方向子集（网络本来就要走安全分支——健全性检查）', rows.filter((r) => r.matched));
report('选错方向子集（网络本来要走不安全分支——这是真正要回答的问题）', rows.filter((r) => !r.matched));

const wrongRows = rows.filter((r) => !r.matched);
const matchedRows = rows.filter((r) => r.matched);
console.log(`\n（选错方向占比 ${rows.length ? (wrongRows.length / rows.length * 100).toFixed(1) : 'NaN'}%，`
  + '与 value_resolution_direction.mjs 此前测到的 22.3%/22.8% 做交叉核对，不是本脚本的新发现）');

console.log('\n=========== 怎么读这份结果 ===========');
if (matchedRows.length) {
  const md = matchedRows.filter((r) => r.diedUnsafeBranch && !r.diedSafeBranch).length
    - matchedRows.filter((r) => r.diedSafeBranch && !r.diedUnsafeBranch).length;
  console.log(md > 0
    ? '健全性检查通过：在"选对方向"子集里，强制走不安全分支确实比强制走安全分支更容易死，'
      + '说明 H/K 的选择和分叉机制本身能测出预期方向，"选错方向"子集的结论可以相信。'
    : '健全性检查没有通过预期方向——"选对方向"子集里强制不安全分支并不明显比安全分支更容易死，'
      + '说明当前 H 可能太短（网络还没来得及体现"选错"的后果）或分类噪声较大，'
      + '下面"选错方向"子集的结论需要谨慎对待，不能直接采信。');
}
console.log('\n真正的问题：在"选错方向"子集里，如果"强制走安全分支"的死亡率明显低于'
  + '"强制走不安全分支"（网络实际会选的那个），说明这次纠正确实能救命——'
  + '"选错但没有更快死"是被网络自己后续的决策部分弥补的，价值分辨率问题是真实存在且有代价的。'
  + '如果两者死亡率接近，说明这一次关键决策本身对最终生死影响有限，价值分辨率不是主要瓶颈。');
console.log(`\n局限：H=${H} 是一个固定、有限的观察窗口，"H 步内没死"不代表"永远安全"——`
  + '如果两分支的差异要到 H 步之后才会显现，这个窗口会低估真实差异，结论方向仍然可信，'
  + '但幅度可能被低估。');
