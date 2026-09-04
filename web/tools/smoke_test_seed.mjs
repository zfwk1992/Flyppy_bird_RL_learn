/**
 * 种子化 PRNG 的 smoke test：验证「两只鸟共用同一组管道」这个前提成立。
 * 跑法：node web/tools/smoke_test_seed.mjs
 *
 * 不是单元测试框架接入，只是一个能重复跑的确定性检查脚本，供下一步
 * 接入 Canvas 渲染 / 双鸟对战前先确认地基没问题。
 */
import assert from 'node:assert/strict';
import { FlappyGame, createSeededRng } from '../game.js';

function pipesSnapshot(game) {
  const s = game.observeState();
  return JSON.stringify({ upperPipes: s.upperPipes, lowerPipes: s.lowerPipes });
}

/**
 * 极简"追踪缝隙中心"启发式：够用来让鸟撑过几百帧而不是随机策略的十几帧，
 * 不追求真的会玩。`bias` 只是让两只鸟的动作序列彼此不同，用来证明管道
 * 生成确实与玩家动作无关。
 */
function heuristicAction(game, bias = 0) {
  const s = game.observeState();
  const next = s.upperPipes.find((p) => p.x + 52 > s.playerx) || s.upperPipes[0];
  const gapCenter = next.y + 320 + next.gap / 2.0;
  const playerCenter = s.playery + 12;
  return playerCenter + bias > gapCenter ? 1 : 0;
}

// ---- 1. 同种子、两个独立实例，各自用不同的（甚至完全相反的）动作序列，
//         管道序列必须逐帧完全一致 —— 管道生成不依赖玩家动作。
{
  const SEED = 42;
  const gameA = new FlappyGame({ seed: SEED });
  const gameB = new FlappyGame({ seed: SEED });

  assert.equal(pipesSnapshot(gameA), pipesSnapshot(gameB), 'reset() 后初始管道就不一致');

  let steps = 0;
  while (!gameA.done && !gameB.done && steps < 3000) {
    // 两只鸟走完全不同的动作（不同的 bias），验证管道生成确实与玩家动作无关
    gameA.step(heuristicAction(gameA, 0));
    gameB.step(heuristicAction(gameB, -15));
    steps += 1;
    assert.equal(
      pipesSnapshot(gameA),
      pipesSnapshot(gameB),
      `第 ${steps} 帧管道序列分叉了（同 seed=${SEED}，理论上不应该发生）`
    );
  }
  assert.ok(steps > 50, `两只鸟都死得太快了（${steps} 帧），没测出什么东西`);
  console.log(`[PASS] 同 seed 两实例，${steps} 帧管道逐帧一致（动作序列完全不同）`);
}

// ---- 2. 不同种子应该（几乎必然）产生不同的管道序列，排除"seed 参数被忽略、
//         实际上一直在用 Math.random 或某个固定分支"这种退化情况。
{
  const gameA = new FlappyGame({ seed: 1 });
  const gameB = new FlappyGame({ seed: 2 });
  assert.notEqual(
    pipesSnapshot(gameA),
    pipesSnapshot(gameB),
    'seed=1 和 seed=2 生成了相同的初始管道，seed 参数可能没生效'
  );
  console.log('[PASS] 不同 seed 产生不同的初始管道');
}

// ---- 3. 同一个 seed 反复构造，序列必须可重放（确定性），不能受到调用
//         次数、GC、时间之类的副作用影响。
{
  const runs = [1, 2, 3].map(() => {
    const g = new FlappyGame({ seed: 777 });
    const snaps = [pipesSnapshot(g)];
    for (let i = 0; i < 500 && !g.done; i++) {
      g.step(heuristicAction(g)); // 固定的确定性策略
      snaps.push(pipesSnapshot(g));
    }
    return snaps.join('|');
  });
  assert.equal(runs[0], runs[1]);
  assert.equal(runs[1], runs[2]);
  console.log('[PASS] 同一个 seed 三次独立重放，结果逐帧一致');
}

// ---- 4. createSeededRng 本身：同 seed 同序列，值落在 [0,1)，且不是常数。
{
  const gen1 = createSeededRng(2026);
  const gen2 = createSeededRng(2026);
  const seq1 = Array.from({ length: 1000 }, () => gen1());
  const seq2 = Array.from({ length: 1000 }, () => gen2());
  assert.deepEqual(seq1, seq2, 'createSeededRng(同seed) 两次调用产生的序列不一致');
  for (const v of seq1) {
    assert.ok(v >= 0 && v < 1, `随机值 ${v} 超出 [0,1) 范围`);
  }
  const distinct = new Set(seq1);
  assert.ok(distinct.size > 990, `1000 次采样只有 ${distinct.size} 个不同值，分布可疑`);
  console.log('[PASS] createSeededRng 自身：确定性 + 值域正确 + 无明显重复');
}

console.log('\n全部通过。');
