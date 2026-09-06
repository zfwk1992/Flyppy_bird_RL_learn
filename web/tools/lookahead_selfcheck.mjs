/**
 * 自测 lookahead_lib.mjs 的克隆是否忠实——这不是 parity_check.mjs 那种
 * Python vs JS 的比对（game.js 完全没改），而是"我们自己写的克隆工具
 * 有没有在骗自己"：如果克隆之后的状态和真实继续推进的状态对不上，
 * oracle_hazard.mjs 测出来的 hazard 就是在测一个不存在的平行世界。
 *
 * 做法：真开一局游戏，用 chaseGap 随便跑若干帧，在随机时刻原地克隆一份，
 * 然后用**同一段后续动作序列**分别推进"原对象"和"克隆对象"很多帧，
 * 逐帧比对所有可观察字段。必须逐帧相同，一步都不能差。
 *
 * 用法：node web/tools/lookahead_selfcheck.mjs
 */
import { FlappyGame, PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import {
  makeSeededGame, cloneGame, chaseGap, rngNext,
} from './lookahead_lib.mjs';

const DIMS = { PIPE_WIDTH, PIPE_HEIGHT, PLAYER_HEIGHT };
const CTOR_OPTS = { hitmasks: HITMASKS };

function snap(g) {
  return JSON.stringify({
    score: g.score, frames: g.frames, playerIndex: g.playerIndex, loopIter: g.loopIter,
    playerx: g.playerx, playery: g.playery, playerVelY: g.playerVelY,
    playerFlapped: g.playerFlapped, basex: g.basex, done: g._done,
    upper: g.upperPipes, lower: g.lowerPipes,
    nextGap: g._nextGap, nextSpacing: g._nextSpacing,
    lastGapCenter: g._lastGapCenter, lastSlack: g._lastSlack,
  });
}

let allOk = true;

// ---- 测试 1：自定义 rng 包装（makeSeededGame）与内置 seed 选项逐位相同 ----
{
  const seed = 424242;
  const gA = new FlappyGame({ seed, hitmasks: HITMASKS }); // 内置 seed 路径
  const gB = makeSeededGame(seed, CTOR_OPTS);              // 自定义 rng 包装路径
  let ok = true;
  for (let i = 0; i < 3000; i++) {
    const a = chaseGap(gA, DIMS);
    const b = chaseGap(gB, DIMS);
    if (a !== b) { ok = false; console.log(`  动作分歧于第 ${i} 帧`); break; }
    const rA = gA.step(a);
    const rB = gB.step(b);
    if (snap(gA) !== snap(gB)) { ok = false; console.log(`  状态分歧于第 ${i} 帧`); break; }
    if (rA.done !== rB.done) { ok = false; console.log(`  done 分歧于第 ${i} 帧`); break; }
    if (rA.done) { gA.reset(); gB.reset(); }
  }
  console.log(`测试 1（自定义 rng 包装 vs 内置 seed，3000 帧 chaseGap）: ${ok ? '一致' : '不一致'}`);
  allOk = allOk && ok;
}

// ---- 测试 2：克隆保真——原地克隆后用同一动作序列推进，逐帧必须相同 ----
for (const seed of [1, 2, 3, 100, 999]) {
  const real = makeSeededGame(seed, CTOR_OPTS);
  // 先跑一段随机长度，制造一个"游戏进行到一半"的状态
  const splitAt = 200 + (seed * 37) % 400;
  for (let i = 0; i < splitAt; i++) {
    if (real.step(chaseGap(real, DIMS)).done) real.reset();
  }
  const clone = cloneGame(real, CTOR_OPTS);
  if (snap(real) !== snap(clone)) {
    console.log(`测试 2 (seed=${seed}): 克隆瞬间状态就不一致！`);
    allOk = false;
    continue;
  }
  let ok = true;
  let steps = 0;
  for (let i = 0; i < 1500; i++) {
    if (real._done || clone._done) break; // 两边应该同时死，下面单独判
    const a = chaseGap(real, DIMS); // 用 real 算出的动作，喂给两边（同一策略、同一状态应给出同一动作）
    const aClone = chaseGap(clone, DIMS);
    if (a !== aClone) { ok = false; console.log(`  seed=${seed}: 动作分歧于克隆后第 ${i} 帧`); break; }
    const rReal = real.step(a);
    const rClone = clone.step(a);
    steps++;
    if (snap(real) !== snap(clone)) {
      ok = false;
      console.log(`  seed=${seed}: 状态分歧于克隆后第 ${i} 帧`);
      break;
    }
    if (rReal.done !== rClone.done) {
      ok = false;
      console.log(`  seed=${seed}: done 分歧于克隆后第 ${i} 帧`);
      break;
    }
    if (rReal.done) break;
  }
  console.log(`测试 2 (seed=${seed}, split=${splitAt}): 克隆后推进 ${steps} 帧，${ok ? '逐帧一致' : '不一致'}`);
  allOk = allOk && ok;
}

// ---- 测试 3：rngNext 状态复制的正确性（不经过 cloneGame，直接测 RNG 本身） ----
{
  const s0 = 777;
  const stateA = { s: s0 };
  const nextA = rngNext(stateA);
  for (let i = 0; i < 137; i++) nextA();
  const stateB = { s: stateA.s }; // 复制状态
  const nextB = rngNext(stateB);
  let ok = true;
  for (let i = 0; i < 500; i++) {
    const a = nextA();
    const b = nextB();
    if (a !== b) { ok = false; console.log(`  rng 分歧于第 ${i} 次调用`); break; }
  }
  console.log(`测试 3（rng 状态复制后续调用逐位相同）: ${ok ? '一致' : '不一致'}`);
  allOk = allOk && ok;
}

console.log('\n' + (allOk ? 'SELFCHECK OK：克隆工具可信，可以用于 oracle_hazard.mjs'
  : 'SELFCHECK FAILED：克隆工具有问题，不要用它的结果'));
process.exit(allOk ? 0 : 1);
