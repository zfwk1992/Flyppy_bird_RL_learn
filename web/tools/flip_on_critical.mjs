/**
 * 实验 D（可选）：策略翻转落在哪些状态上。
 *
 * 用实验 A 写好的前瞻器（lookahead_lib.mjs）把一批探针状态分成两类：
 *   - **关键状态**：某个动作在 K 步内必死（不是"两个动作都安全"）
 *   - **无差别状态**：两个动作在 K 步内都安全
 * 判据：如果翻转/churn 集中在无差别状态 -> 良性抖动，"压制策略抖动"那类
 * 改动（CHAIN 风格正则化等）可以直接从计划里划掉。
 *
 * !!! 重要限定（如实写在这里，不要在结果里含糊） !!!
 * 任务原本想测的"翻转率"是 `flappy/diagnostics.py: argmax_flip`
 * 那种口径——**同一次训练内**相邻两次诊断快照（间隔约 100 个梯度步）之间
 * 贪婪动作变了的比例。云端只有 `runs/base_s0/final.pt` 这一个模型导出成
 * 的 JS 权重，**没有同一次训练里另一个相邻 checkpoint 的 JS 导出**（导出
 * 需要 torch，云端没有）。所以这里做不了真正的 churn 测量，只能做两件
 * 退而求其次、但仍然真实可信的事：
 *
 *   (a) **Q-gap 代理指标**：用当前模型，比较"关键状态"和"无差别状态"上
 *       |Q1-Q0| 的分布。动作差距越小，越容易被一点点参数扰动翻转——这是
 *       "翻转"的**前提条件**，不是翻转本身，但方向上应该相关。
 *   (b) **跨模型翻转率**（仅供参考，不是 churn）：拿网页 demo 在用的旧模型
 *       （`weights_fp16.bin`，另一次训练，架构相同但权重完全不同）和当前
 *       模型比较贪婪动作，按关键/无差别分组看翻转率。这两个模型之间的差异
 *       远大于"训练晚期 100 个梯度步"，**不能**当成 churn 的度量，只能看
 *       "跨模型的分歧是否也集中在关键状态"这一个弱得多的信号。
 *
 * 真正的 argmax_flip@关键状态 测量需要本机（有 torch）导出同一次训练里
 * 两个相邻 checkpoint 的 JS 权重——已经写进 docs/research/CHECKLIST.md。
 *
 * 用法：node web/tools/flip_on_critical.mjs [探针数=400] [前瞻深度K=15]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT, OBS_W, OBS_H } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META as META_NEW } from '../model/weights-meta-base_s0.js';
import { WEIGHTS_META as META_OLD } from '../model/weights-meta.js';
import {
  makeSeededGame, cloneGame, chaseGap, survivesLookahead,
} from './lookahead_lib.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const CTOR_OPTS = { hitmasks: HITMASKS };
const SEED_BASE = 20260906 + 900_000; // 与训练侧探针集的错峰习惯一致（diagnostics.py 也 +900000）

const N_PROBE = Number(process.argv[2] || 400);
const K = Number(process.argv[3] || 15);
const STRIDE = 3;
const FRAME_SKIP = 4;
const PER_EP_CAP = Math.max(1, Math.floor(N_PROBE / 8));

function loadNet(meta) {
  const raw = fs.readFileSync(path.join(HERE, '..', 'model', meta.file));
  return new DuelingDQN(parseWeights(
    raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));
}
const netNew = loadNet(META_NEW);
const netOld = loadNet(META_OLD);

// ---------------------------------------------------------------------
// 1. 生成探针集：chase_gap 驱动（与被测网络无关），每个探针存
//    {obsStack 的拷贝, 对应时刻的游戏克隆（供前瞻分类用）}
// ---------------------------------------------------------------------
const probes = [];
{
  const red = new Uint8Array(288 * 512);
  const obs = new Float32Array(OBS_W * OBS_H);
  let ep = 0;
  while (probes.length < N_PROBE && ep < 200) {
    const game = makeSeededGame(SEED_BASE + ep, CTOR_OPTS);
    const stack = new FrameStack(4);
    let takenThisEp = 0;
    let decisionIdx = 0;
    let action = 0;
    for (let i = 0; i < 20000; i++) {
      if (i % FRAME_SKIP === 0) {
        renderRed(game, red);
        downsample(red, obs);
        const arr = i === 0 ? stack.reset(obs) : stack.push(obs);
        if (decisionIdx % STRIDE === 0) {
          probes.push({ obs: arr.slice(), gameClone: cloneGame(game, CTOR_OPTS) });
          takenThisEp++;
          if (probes.length >= N_PROBE || takenThisEp >= PER_EP_CAP) break;
        }
        action = chaseGap(game, DIMS);
        decisionIdx++;
      }
      if (game.step(action).done) break;
    }
    ep++;
  }
}
console.log(`探针集：${probes.length} 个状态（跨 seed，chase_gap 驱动，与被测网络无关）`);

// ---------------------------------------------------------------------
// 2. 用 A 的前瞻器分类：关键 vs 无差别
// ---------------------------------------------------------------------
let nCritical = 0;
let nIndifferent = 0;
const rows = [];
for (const p of probes) {
  const safe0 = survivesLookahead(p.gameClone, 0, K, CTOR_OPTS, DIMS);
  const safe1 = survivesLookahead(p.gameClone, 1, K, CTOR_OPTS, DIMS);
  const critical = !(safe0 && safe1);
  if (critical) nCritical++; else nIndifferent++;

  const qNew = netNew.forward(p.obs);
  const gapNew = Math.abs(qNew[1] - qNew[0]);
  const argmaxNew = qNew[1] > qNew[0] ? 1 : 0;
  const qOld = netOld.forward(p.obs);
  const gapOld = Math.abs(qOld[1] - qOld[0]);
  const argmaxOld = qOld[1] > qOld[0] ? 1 : 0;

  rows.push({
    critical, gapNew, gapOld, flip: argmaxNew !== argmaxOld,
  });
}

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length;
  if (!m) return NaN;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}
function mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN; }

console.log(`\n关键状态 ${nCritical} 个 (${(nCritical / rows.length * 100).toFixed(1)}%)`
  + `，无差别状态 ${nIndifferent} 个 (${(nIndifferent / rows.length * 100).toFixed(1)}%)`
  + `  （K=${K} 步前瞻）`);

const crit = rows.filter((r) => r.critical);
const indiff = rows.filter((r) => !r.critical);

console.log('\n=========== (a) Q-gap 代理指标（当前模型 base_s0） ===========');
console.log(`关键状态   |Q1-Q0| 中位 ${median(crit.map((r) => r.gapNew)).toFixed(3)}`
  + `  均值 ${mean(crit.map((r) => r.gapNew)).toFixed(3)}  (n=${crit.length})`);
console.log(`无差别状态 |Q1-Q0| 中位 ${median(indiff.map((r) => r.gapNew)).toFixed(3)}`
  + `  均值 ${mean(indiff.map((r) => r.gapNew)).toFixed(3)}  (n=${indiff.length})`);
console.log(median(crit.map((r) => r.gapNew)) < median(indiff.map((r) => r.gapNew))
  ? '-> 关键状态的动作差距更小，方向上支持"关键状态更容易被扰动翻转"。'
  : '-> 关键状态的动作差距并不更小，没有支持这个方向（也可能是代理指标本身不敏感）。');

console.log('\n=========== (b) 跨模型翻转率（仅供参考，不是同次训练内的 churn） ===========');
const flipCrit = crit.filter((r) => r.flip).length / (crit.length || 1);
const flipIndiff = indiff.filter((r) => r.flip).length / (indiff.length || 1);
console.log(`关键状态翻转率   ${(flipCrit * 100).toFixed(1)}%  (n=${crit.length})`);
console.log(`无差别状态翻转率 ${(flipIndiff * 100).toFixed(1)}%  (n=${indiff.length})`);
console.log('注意：这两个模型来自不同的训练（架构相同，权重完全独立训练），'
  + '差异量级远超"同次训练内 100 个梯度步"的 churn，这一段数字不能当成 churn 的度量，'
  + '只能看"跨模型分歧是否也集中在关键状态"这一个弱信号，供交叉印证用。');
