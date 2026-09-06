/**
 * 死亡归因：AI 每一局死在什么样的管道上？
 *
 * 回答一个决定后续该修什么的问题 —— 分数方差大（5 到 321 根）到底是
 *   (a) **任务本身**的方差：域随机化偶尔生成又窄又密的序列，谁来都难；还是
 *   (b) **策略脆弱**：在明明宽松的缝隙上也会掉下去。
 * 两者的修法完全相反：(a) 要改环境/课程或者干脆接受方差，(b) 才该动网络。
 *
 * 跑的是浏览器那份 JS 实现（与 PyTorch 逐位一致，见 parity_ai.md），
 * 所以不需要 torch，也比 Python 快。
 *
 * 用法：node web/tools/death_attribution.mjs [局数=25]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FlappyGame, OBS_W, OBS_H, PIPE_WIDTH, PLAYER_HEIGHT, PIPE_HEIGHT } from '../game.js';
import { HITMASKS } from '../assets/hitmasks.js';
import { renderRed, downsample, FrameStack } from '../obs.js';
import { DuelingDQN, parseWeights } from '../nn.js';
import { WEIGHTS_META } from '../model/weights-meta.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const N = Number(process.argv[2] || 25);
const raw = fs.readFileSync(path.join(HERE, '..', 'model', WEIGHTS_META.file));
const net = new DuelingDQN(parseWeights(
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)));

const red = new Uint8Array(288 * 512);
const obs = new Float32Array(OBS_W * OBS_H);
const stack = new FrameStack(4);

/** 小鸟正在穿越（或即将穿越）的那根管道。 */
function currentPipe(g) {
  for (let i = 0; i < g.upperPipes.length; i++) {
    if (g.upperPipes[i].x + PIPE_WIDTH > g.playerx) return g.upperPipes[i];
  }
  return g.upperPipes[g.upperPipes.length - 1];
}

const rows = [];
for (let ep = 0; ep < N; ep++) {
  const game = new FlappyGame({ seed: 5000 + ep, hitmasks: HITMASKS });
  game.reset();
  let action = 0;
  let last = null;
  for (let i = 0; i < 20000; i++) {
    if (i % 4 === 0) {
      renderRed(game, red);
      downsample(red, obs);
      action = net.act(i === 0 ? stack.reset(obs) : stack.push(obs));
    }
    // 撞之前记下当时那根管道的形状和鸟的位置
    const p = currentPipe(game);
    last = {
      gap: p.gap,
      dx: p.x - game.playerx,
      // 鸟中心相对缝隙中心的偏移：正数=偏低
      off: (game.playery + PLAYER_HEIGHT / 2) - (p.y + PIPE_HEIGHT + p.gap / 2),
      vel: game.playerVelY,
      y: game.playery,
    };
    if (game.step(action).done) break;
  }
  // 撞地面 vs 撞管道：地面在 y=404.48，鸟高 24
  const hitGround = last.y + PLAYER_HEIGHT >= 404.48 - 0.5;
  rows.push({ ep, score: game.score, ...last, hitGround });
  console.log(`第 ${String(ep + 1).padStart(2)} 局: ${String(game.score).padStart(4)} 根`
    + `  死时缝隙=${last.gap.toFixed(0)}px  距管道=${last.dx.toFixed(0)}px`
    + `  偏离中心=${last.off > 0 ? '+' : ''}${last.off.toFixed(0)}px`
    + `  ${hitGround ? '撞地面' : '撞管道'}`);
}

const gaps = rows.map((r) => r.gap).sort((a, b) => a - b);
const q = (p) => gaps[Math.floor(p * (gaps.length - 1))];
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
console.log('\n---------------------------------------------');
console.log(`局数 ${rows.length}，平均 ${mean(rows.map((r) => r.score)).toFixed(1)} 根`);
console.log(`死亡时的缝隙宽度: 最小 ${q(0).toFixed(0)}  中位 ${q(0.5).toFixed(0)}  最大 ${q(1).toFixed(0)}  (训练分布是 85-165，均匀采样，期望中位 125)`);
console.log(`撞地面 ${rows.filter((r) => r.hitGround).length} / ${rows.length}，撞管道 ${rows.filter((r) => !r.hitGround).length} / ${rows.length}`);
const narrow = rows.filter((r) => r.gap < 105);
console.log(`死在窄缝(<105px)的局数: ${narrow.length} / ${rows.length}`
  + `  —— 窄缝只占管道总数的约 25%，这个比例若明显高于 25% 就说明死亡集中在窄缝`);
const short = rows.filter((r) => r.score < 30);
console.log(`早死(<30 根)的 ${short.length} 局，其死亡缝隙中位 `
  + `${short.length ? mean(short.map((r) => r.gap)).toFixed(0) : 'n/a'}px`);
console.log('---------------------------------------------');
