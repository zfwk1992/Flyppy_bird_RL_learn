/**
 * value_resolution.mjs 的部署分布版本（gapRange=[100,165]）。
 *
 * 背景：`LAYER0-RESULTS.md` 2026-09-07 那次云端 session 发现，部署分布
 * [100,165] 上网络 hazard=0.452~0.464%，同分布 oracle N=120/v2 只要
 * 0.219%——网络离 oracle 还差约 2.1 倍，**比训练分布上的差距（1.68 倍）
 * 更大**。但此前所有"价值分辨率"测量（`value_resolution.mjs`）都在训练
 * 分布 [85,165] 上做，没人在部署分布上测过网络是否分得清"关键"和
 * "无差别"状态。这个文件补这个空白。
 *
 * 和 `value_resolution.mjs` 唯一的区别：CTOR_OPTS 加 `gapRange:[100,165]`，
 * 其余分类逻辑、统计口径完全一致（权重、frame_skip、stride 不变）。
 *
 * ⚠️ 一个没有解决的口径问题，如实写在这里而不是藏起来：K=120 这个前瞻深度
 * 是在**训练分布**上测到"两个分支都不安全"比例的拐点（`lookahead_saturation.mjs
 * --sweep`），从未在部署分布上重新扫过。部署分布更宽松，chase_gap 续跑策略
 * 的平均寿命大概率比训练分布长（更容易存活），所以 K=120 在这里未必是最优
 * 前瞻深度——比训练分布上的效应更保守还是更激进，不知道。`oracle_hazard_deploy.mjs`
 * 已经先例性地直接复用了训练分布测到的 N=120（`LAYER0-RESULTS.md` 里明确写
 * "此前训练分布上测到的最好前瞻深度"），这里跟随同一个近似，不重新扫，
 * 但明确标注这是一个未验证的假设，不是坐实的最优值。
 *
 * 用法：node web/tools/value_resolution_deploy.mjs [探针数=3000] [前瞻深度K=120]
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
const GAP_RANGE = [100, 165]; // 部署分布（da00fe8），不是训练分布 [85,165]
const CTOR_OPTS = { hitmasks: HITMASKS, gapRange: GAP_RANGE };
const SEED_BASE = 20260906; // 对齐 flappy/config.py: eval_seed_base

const N_PROBE = Number(process.argv[2] || 3000);
const K = Number(process.argv[3] || 120);
const FRAME_SKIP = 4;
const STRIDE = 3;
const PER_EP_CAP = Math.max(1, Math.floor(N_PROBE / 8));

const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obsBuf = new Float32Array(OBS_W * OBS_H);

console.log(`[部署分布 gapRange=${JSON.stringify(GAP_RANGE)}] 探针数目标=${N_PROBE}  前瞻深度K=${K}  单局采样上限=${PER_EP_CAP}`);
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
  for (let i = 0; i < 30000; i++) {
    if (i % FRAME_SKIP === 0) {
      renderRed(game, red);
      downsample(red, obsBuf);
      const arr = i === 0 ? stack.reset(obsBuf) : stack.push(obsBuf);
      const q = net.forward(arr);
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
  if (ep % 10 === 0 || rows.length >= N_PROBE) {
    const elapsed = (performance.now() - t0) / 1000;
    process.stdout.write(`  已跑 ${ep} 局，采到 ${rows.length}/${N_PROBE} 个探针，`
      + `耗时 ${elapsed.toFixed(1)}s\n`);
  }
}

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

console.log('=========== 状态分类占比（部署分布，K=' + K + ' 步前瞻） ===========');
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
  console.log('关键状态或无差别状态样本数为 0，无法比较。');
}

console.log('\n=========== 死亡样本稀缺性 ===========');
const critFrac = byClass.critical.length / rows.length;
console.log(`关键状态占全部探针的比例：${(critFrac * 100).toFixed(1)}%`);
console.log(critFrac < 0.1
  ? '-> 关键状态很稀罕：支持"死亡样本过采样/PER 应该被列为候选"。'
  : '-> 关键状态不算稀罕，"经验池里关键转移被淹没"这条论证的前提没有得到'
    + '这批数据的支持。');
