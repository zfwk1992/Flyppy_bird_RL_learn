/**
 * 实验 A【最高优先级，本轮】把"价值分辨率"假说变成可测量。
 *
 * 假说（Fable session 的推导，本机已核对算术）：多冒 1% 的死亡风险，在 Q 上
 * 只值 0.01 × Q ≈ 0.12，而训练后期 td_abs_mean ≈ 0.09——信噪比只有约 1.3。
 * 如果这个假说成立，网络在"两个动作真实存活率差很多"的状态上，给出的
 * |Q1-Q0| 应该和"两个动作都安全、怎么选都无所谓"的状态上**分不清**——
 * 因为噪声已经把这点差异淹没了。
 *
 * 判据（任务原文）：
 *   - 网络的动作差距在"关键状态"上明显大于"无差别状态" -> 能区分，
 *     价值分辨率假说弱化
 *   - 两类差距分布重叠（没有明显区分） -> 网络分不清哪里重要，
 *     这是支持价值分辨率假说的强证据
 *
 * 状态分类用实验 A（上一轮）已经建好的前瞻器（lookahead_lib.mjs，
 * chaseGap 续跑 + N 步安全检查），N=120（上一轮测到 oracle hazard 见底的
 * 深度，不是任务原来建议的 N≈15——任务本身的教训是浅前瞻会把结论看反）：
 *   - 两个动作都安全 -> "无差别"
 *   - 只有一个安全 -> "关键"
 *   - 两个都不安全（N 步内必死无法避免） -> "必死"
 *
 * "很多真实状态"取的是**当前网络自己驱动游戏时真正会遇到的状态**
 * （不是像上一轮 flip_on_critical.mjs 那样用与网络无关的 chase_gap 驱动
 * 生成探针）——因为这次要测的是"网络在自己会遇到的局面里，价值分辨率
 * 够不够"，探针分布本身就该是网络的on-policy 状态分布，换成 chase_gap
 * 驱动会引入分布不匹配。跨多个 seed、每隔 STRIDE 个决策采一个，
 * 且单局设采样上限，避免个别长局把统计量拉偏。
 *
 * 前瞻本身（chaseGap 续跑）不调用网络、不渲染像素，只是纯物理 step，
 * 所以 K=120 的双分支前瞻对每个探针只增加轻量开销；真正的耗时大头是
 * 网络前向 + 渲染/降采样（和 ai_eval.mjs 的开销量级一致）。
 *
 * 用法：node web/tools/value_resolution.mjs [探针数=3000] [前瞻深度K=120]
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
const CTOR_OPTS = { hitmasks: HITMASKS };
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base

const N_PROBE = Number(process.argv[2] || 3000);
const K = Number(process.argv[3] || 120);
const FRAME_SKIP = 4;
const STRIDE = 3; // 每隔几个决策采一次样，避免相邻帧高度相关
const PER_EP_CAP = Math.max(1, Math.floor(N_PROBE / 8)); // 单局采样上限，保证跨 seed 多样性

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obsBuf = new Float32Array(OBS_W * OBS_H);

console.log(`探针数目标=${N_PROBE}  前瞻深度K=${K}  单局采样上限=${PER_EP_CAP}`);
console.log('探针来源：网络自己驱动游戏的 on-policy 轨迹（不是 chase_gap 驱动）\n');

const rows = [];
let ep = 0;
const t0 = performance.now();
while (rows.length < N_PROBE && ep < 1_000_000) {
  const game = makeSeededGame(SEED_BASE + ep, CTOR_OPTS);
  const stack = new FrameStack(4);
  let action = 0;
  let decisionIdx = 0;
  let takenThisEp = 0;
  for (let i = 0; i < 20000; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obsBuf);
      const arr = i === 0 ? stack.reset(obsBuf) : stack.push(obsBuf);
      const q = net.forward(arr); // [q0, q1]，net 内部复用的数组
      const q0 = q[0]; const q1 = q[1];
      action = q1 > q0 ? 1 : 0;

      if (decisionIdx % STRIDE === 0) {
        const clone = cloneGame(game, CTOR_OPTS);
        const safe0 = survivesLookahead(clone, 0, K, CTOR_OPTS, DIMS);
        const safe1 = survivesLookahead(clone, 1, K, CTOR_OPTS, DIMS);
        let cls;
        if (safe0 && safe1) cls = 'indifferent';
        else if (safe0 || safe1) cls = 'critical';
        else cls = 'certain_death';
        rows.push({ gapQ: Math.abs(q1 - q0), cls });
        takenThisEp += 1;
        if (rows.length >= N_PROBE || takenThisEp >= PER_EP_CAP) break;
      }
      decisionIdx += 1;
    }
    if (game.step(action).done) break;
  }
  ep += 1;
  if (ep % 20 === 0 || rows.length >= N_PROBE) {
    const elapsed = (performance.now() - t0) / 1000;
    process.stdout.write(`  已跑 ${ep} 局，采到 ${rows.length}/${N_PROBE} 个探针，`
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

const byClass = {
  critical: rows.filter((r) => r.cls === 'critical'),
  indifferent: rows.filter((r) => r.cls === 'indifferent'),
  certain_death: rows.filter((r) => r.cls === 'certain_death'),
};

console.log(`\n采到 ${rows.length} 个探针，跑了 ${ep} 局，`
  + `耗时 ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

console.log('=========== 状态分类占比（K=' + K + ' 步前瞻） ===========');
for (const [name, cn] of [['关键 critical', 'critical'], ['无差别 indifferent', 'indifferent'], ['必死 certain_death', 'certain_death']]) {
  const n = byClass[cn].length;
  console.log(`${name.padEnd(22)} n=${n}  (${(n / rows.length * 100).toFixed(1)}%)`);
}

console.log('\n=========== 网络 |Q1-Q0| 分布（按分类） ===========');
console.log('分类\t\tn\tQ25\t中位\tQ75\t均值');
for (const [label, cn] of [['critical', 'critical'], ['indifferent', 'indifferent'], ['certain_death', 'certain_death']]) {
  const gs = byClass[cn].map((r) => r.gapQ);
  if (!gs.length) { console.log(`${label}\t\t0\t-\t-\t-\t-`); continue; }
  console.log(`${label}\t${gs.length}\t${quantile(gs, 0.25).toFixed(3)}\t`
    + `${median(gs).toFixed(3)}\t${quantile(gs, 0.75).toFixed(3)}\t${mean(gs).toFixed(3)}`);
}

console.log('\n=========== 关键判据 ===========');
const critGaps = byClass.critical.map((r) => r.gapQ);
const indiffGaps = byClass.indifferent.map((r) => r.gapQ);
if (critGaps.length && indiffGaps.length) {
  const mCrit = median(critGaps); const mIndiff = median(indiffGaps);
  const q25Crit = quantile(critGaps, 0.25); const q75Crit = quantile(critGaps, 0.75);
  const q25Indiff = quantile(indiffGaps, 0.25); const q75Indiff = quantile(indiffGaps, 0.75);
  console.log(`关键状态 |Q1-Q0| 中位=${mCrit.toFixed(3)}  IQR=[${q25Crit.toFixed(3)}, ${q75Crit.toFixed(3)}]`);
  console.log(`无差别状态 |Q1-Q0| 中位=${mIndiff.toFixed(3)}  IQR=[${q25Indiff.toFixed(3)}, ${q75Indiff.toFixed(3)}]`);
  const iqrOverlap = !(q75Indiff < q25Crit || q75Crit < q25Indiff);
  console.log(`两类 IQR ${iqrOverlap ? '有重叠' : '不重叠'}`
    + `，中位数比值（关键/无差别）= ${(mCrit / (mIndiff || 1e-9)).toFixed(2)}`);
  if (!iqrOverlap && mCrit > mIndiff) {
    console.log('-> 关键状态的动作差距明显更大且 IQR 不重叠：网络能区分关键与否，'
      + '价值分辨率假说被这条数据弱化。');
  } else if (iqrOverlap) {
    console.log('-> 两类 |Q1-Q0| 的 IQR 有重叠：不能确认网络能清楚分辨关键状态，'
      + '方向上支持价值分辨率假说（但"有重叠"不等于"完全分不清"，需要结合中位数比值判断）。');
  } else {
    console.log('-> 关键状态动作差距反而不比无差别状态大：不支持"网络能区分关键状态"，'
      + '支持价值分辨率假说。');
  }
} else {
  console.log('关键状态或无差别状态样本数为 0，无法比较（多半是 K=120 太深，'
    + '几乎所有探针都被判成"必死"——这本身也是有意义的信息，见下）。');
}

console.log('\n=========== 死亡样本稀缺性 ===========');
const critFrac = byClass.critical.length / rows.length;
console.log(`关键状态占全部探针的比例：${(critFrac * 100).toFixed(1)}%`);
console.log(critFrac < 0.1
  ? '-> 关键状态很稀罕：均匀采样的经验回放池里，"选错会死"的转移天然被淹没在'
    + '大量无差别转移里，直接支持"死亡样本过采样/PER 应该被列为候选"。'
  : '-> 关键状态不算稀罕，"经验池里关键转移被淹没"这条论证的前提没有得到'
    + '这批数据的支持（不代表 PER 没用，只是这条特定论证的证据不强）。');

console.log('\n注意：K=120 是上一轮实验 A 测到 oracle hazard 见底的深度，但上一轮'
  + '同时发现"两个分支都判不安全"的比例可能随 N 增大而上升（本轮任务 B 单独测'
  + '这个比例）。如果这里"必死"类占比很高，说明 K=120 下前瞻本身的判别力可能'
  + '已经被稀释，解释"关键 vs 无差别"对比时要连着任务 B 的结果一起看。');
