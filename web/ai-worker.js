/**
 * AI 预跑 worker：把整条推理链路搬离主线程。
 *
 * 为什么需要这个文件
 * ------------------
 * 手写前向单次约 12.3M 次乘加。桌面上 45 ms、预算 133 ms，很宽裕；
 * 但在中端手机上实测 178 ms、低端 276 ms —— **超预算**。推理原来是同步跑在
 * 主线程的 rAF 回调里，超预算的直接后果不是"AI 变笨"，而是整个页面被堵住：
 * 实测 4x CPU 节流下渲染掉到 9.5 fps、6x 下 6.3 fps，玩家自己那只鸟也跟着卡。
 * 对一个要发到 LinkedIn（流量以手机为主）的 demo，这是致命的。
 *
 * 关键观察：**AI 那一局和玩家的操作完全无关。**`aiGame` 是独立实例，
 * 管道序列只由 seed 决定，AI 的动作只由它自己看到的画面决定。所以整条
 * 轨迹在玩家按下第一次之前就可以算出来 —— 这里算，主线程只负责回放。
 *
 * 回放是**逐位一致**的，不是近似：同一个 seed + 同一串动作喂进同一份
 * `game.js`，状态必然相同。所以这个改动**没有削弱 AI**，只是换了执行时机
 * （plan.md 第八节：不加反应延迟、不降难度、不改推理逻辑）。
 *
 * 产出的是**逐帧**动作（不是逐决策）：frame_skip 的重复由 `AiPlayer.decide`
 * 内部处理，一帧一字节，主线程直接 `plan[frame]` 索引，不用再算决策下标。
 *
 * 产多少：不是一口气把一局算完 —— AI 能飞 322 根管道（约 1 万帧），
 * 那是几十秒的满线程计算，玩家一重开就全扔了，手机上纯属烧电。改成
 * 主线程说"我要到第 N 帧"，这边生产到 N 就停（`want`），边玩边补。
 */
import { FlappyGame } from './game.js';
import { AiPlayer } from './ai.js';
import { parseWeights } from './nn.js';
import { HITMASKS } from './assets/hitmasks.js';
import { WEIGHTS_META } from './model/weights-meta.js';

// 一局最多算这么多帧。AI 最好成绩 322 根管道约 1 万帧，2 万帧是安全上限；
// 真跑到这儿说明这一局它基本不会死了，停下来不影响观感。
const MAX_FRAMES = 20000;
// 每批算这么多帧就让出一次事件循环 —— 不让出的话，玩家点"下一局"发来的
// start 消息要等整批算完才能处理，表现为换局迟钝。16 帧 = 4 次决策。
const BATCH_FRAMES = 16;

let player = null;              // AiPlayer，权重加载完才有
let job = null;                 // 当前这一局：{ gen, seed, game, actions, want, ended }
let gen = 0;                    // 每次 start 递增，用来作废上一局还在跑的生产循环
let pendingStart = null;        // 权重还没加载完就收到的 start，等 ready 之后补跑

const yieldToLoop = () => new Promise((r) => setTimeout(r, 0));

/**
 * 流式下载权重，边下边报进度。
 *
 * 没有直接用 `nn.js: loadWeights()` —— 那个用的是 `res.arrayBuffer()`，
 * 拿不到中间进度。2.5 MB 在慢 4G 上要 15 秒，没有进度条的话首屏就是一段
 * 无反馈的空白，这是真实的跳出率损耗。这里自己读 ReadableStream，
 * 解析仍然交给 `nn.js: parseWeights()`，**权重解析只有一份实现**。
 */
async function fetchWeights() {
  const res = await fetch('./model/' + WEIGHTS_META.file);
  if (!res.ok) throw new Error(`权重加载失败：${res.status}`);
  const total = WEIGHTS_META.bytes;
  const buf = new Uint8Array(total);
  let at = 0;

  // 个别环境（老 Safari、某些代理）拿不到 body 流，退回一次性读取：
  // 没有进度总比下不下来强。
  if (!res.body || !res.body.getReader) {
    const all = new Uint8Array(await res.arrayBuffer());
    postMessage({ type: 'progress', loaded: all.byteLength, total });
    return all.buffer;
  }

  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (at + value.byteLength > total) throw new Error('权重文件比 meta 声明的长');
    buf.set(value, at);
    at += value.byteLength;
    postMessage({ type: 'progress', loaded: at, total });
  }
  if (at !== total) throw new Error(`权重长度不对：${at} != ${total}`);
  return buf.buffer;
}

/**
 * 生产循环：把 job.actions 补到 job.want（或这一局结束 / 到上限）。
 *
 * `myGen !== gen` 就立刻退出 —— 玩家开了新的一局，这一局的结果没人要了。
 */
async function produce(myGen) {
  while (job && job.gen === myGen && !job.ended) {
    if (job.actions.length >= job.want || job.actions.length >= MAX_FRAMES) {
      await yieldToLoop();               // 已经够前了，等主线程再要
      continue;
    }
    const chunkFrom = job.actions.length;
    const chunk = [];
    let msAcc = 0, msN = 0;
    for (let i = 0; i < BATCH_FRAMES && job.actions.length + chunk.length < MAX_FRAMES; i++) {
      // decide() 必须在 step() 之前 —— 观测的是当前这一帧的画面，与训练时序一致
      const a = player.decide(job.game);
      // lastMs 只在真正跑了网络的那一帧非零（frame_skip 内的其余三帧是复用）
      if (player.lastMs) { msAcc += player.lastMs; msN++; player.lastMs = 0; }
      chunk.push(a);
      if (job.game.step(a).done) { job.ended = true; break; }
    }
    if (job.gen !== gen) return;         // 算这一批的时候已经换局了，扔掉
    for (const a of chunk) job.actions.push(a);

    postMessage({
      type: 'plan', seed: job.seed, from: chunkFrom,
      actions: Int8Array.from(chunk),
      ended: job.ended, score: job.game.score,
      ms: msN ? msAcc / msN : 0, msN,
    });
    await yieldToLoop();
  }
}

function startJob(msg) {
  gen++;
  const myGen = gen;
  const game = new FlappyGame({ seed: msg.seed, hitmasks: HITMASKS });
  game.reset();
  player.reset();
  job = { gen: myGen, seed: msg.seed, game, actions: [], want: msg.want, ended: false };
  produce(myGen);
}

onmessage = (e) => {
  const msg = e.data;

  if (msg.type === 'start') {
    // 权重还在下载 —— 记下来，ready 之后立刻补跑，不用主线程重发
    if (!player) { pendingStart = msg; return; }
    startJob(msg);
    return;
  }

  if (msg.type === 'want') {
    if (pendingStart && pendingStart.seed === msg.seed && msg.want > pendingStart.want) {
      pendingStart.want = msg.want;
      return;
    }
    // 只认当前这一局的请求，晚到的旧消息不能把 want 拉回去
    if (job && job.seed === msg.seed && msg.want > job.want) job.want = msg.want;
    return;
  }
};

// 权重加载与消息处理并行：主线程可以先把 start 发过来排队。
try {
  player = new AiPlayer(parseWeights(await fetchWeights()));
  postMessage({ type: 'ready' });
  if (pendingStart) { const m = pendingStart; pendingStart = null; startJob(m); }
} catch (err) {
  postMessage({ type: 'error', message: String(err && err.message || err) });
}
