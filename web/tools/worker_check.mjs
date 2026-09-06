/**
 * worker 预跑一致性核查：**回放出来的轨迹必须和主线程直接推理逐位相同。**
 *
 * 为什么需要这条检查
 * ------------------
 * `web/ai-worker.js` 把推理搬进了 worker：给定 seed，worker 预先算出逐帧动作，
 * 主线程只回放。这么做的**全部合法性**建立在一个前提上 ——
 * AI 那一局与玩家操作完全无关，所以同一个 seed + 同一串动作喂进同一份
 * `game.js`，轨迹必然逐位一致。
 *
 * 这个前提要是哪天被破坏（比如有人让 AI 的观测里混进玩家那只鸟、或者
 * `AiPlayer` 里引入了跨局残留的状态），表现出来**不是报错，而是 AI 悄悄变笨** ——
 * 正是这个项目历史上最难查的那一类 bug。所以这里把它钉死成一条可执行的检查。
 *
 * 比对的是两条独立算出来的动作序列：
 *   A. worker 按 seed 预跑（`ai-worker.js` 走的路径）
 *   B. 主线程直接 `new AiPlayer(...).decide(game)` 逐帧算（老路径，nn_check 验过的那条）
 * 逐帧全等才算过；顺带比对两条轨迹的最终分数和存活帧数。
 *
 * 用法（需要先起本地服务）：
 *     python -m http.server 8123 --directory web
 *     node web/tools/worker_check.mjs [url]
 *
 * 退出码 0 = 逐位一致。零依赖，走 CDP 直连，不需要 Playwright。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const url = process.argv.find((a, i) => i >= 2 && !a.startsWith('--'))
  || 'http://localhost:8123/index.html';

const CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/opt/node22/lib/node_modules/playwright/.local-browsers/chromium/chrome-linux/chrome',
].filter(Boolean);
const chrome = CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch { return false; } });
if (!chrome) { console.error('没找到 Chrome/Chromium，设 CHROME_PATH 指过去'); process.exit(2); }

const PORT = 9300 + Math.floor(Math.random() * 300);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`,
  '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-worker-check'),
  'about:blank',
], { stdio: 'ignore' });

let target = null;
for (let i = 0; i < 80 && !target; i++) {
  await sleep(250);
  try {
    target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json())
      .find((t) => t.type === 'page');
  } catch { /* 还没起来 */ }
}
if (!target) { proc.kill(); console.error('Chromium 没能起 CDP'); process.exit(2); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
const pending = new Map();
let seq = 0;
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
await new Promise((r) => { ws.onopen = r; });
const send = (method, params = {}) => new Promise((res) => {
  const id = ++seq;
  pending.set(id, (m) => res(m.result ?? m.error));
  ws.send(JSON.stringify({ id, method, params }));
});
await send('Page.enable');
await send('Runtime.enable');

// 三个 seed，每个比对到 FRAMES 帧（或 AI 先撞死为止）。
// 600 帧 = 20 秒飞行、150 次决策，足以覆盖多根管道和若干次擦边。
const SEEDS = [12345, 987654321, 42];
const FRAMES = 600;

const EXPR = `(async () => {
  const seeds = ${JSON.stringify(SEEDS)}, FRAMES = ${FRAMES};
  const [{ FlappyGame }, { AiPlayer }, { parseWeights }, { HITMASKS }, { WEIGHTS_META }] =
    await Promise.all([import('./game.js'), import('./ai.js'), import('./nn.js'),
                       import('./assets/hitmasks.js'), import('./model/weights-meta.js')]);

  // B 路：主线程直接推理（老路径）
  const res = await fetch('./model/' + WEIGHTS_META.file);
  const local = new AiPlayer(parseWeights(await res.arrayBuffer()));

  // A 路：worker 预跑
  const w = new Worker('./ai-worker.js', { type: 'module' });
  const planFor = (seed) => new Promise((resolve, reject) => {
    const acc = [];
    let ended = false, score = 0;
    w.onerror = (e) => reject(new Error('worker error: ' + e.message));
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'error') return reject(new Error(m.message));
      if (m.type === 'plan') {
        if (m.seed !== seed) return;
        for (let i = 0; i < m.actions.length; i++) acc[m.from + i] = m.actions[i];
        if (m.ended) { ended = true; score = m.score; }
        if (acc.length >= FRAMES || ended) resolve({ actions: acc, ended, score });
        else w.postMessage({ type: 'want', seed, want: FRAMES });
      }
    };
    w.postMessage({ type: 'start', seed, want: FRAMES });
  });

  const out = [];
  for (const seed of seeds) {
    const got = await planFor(seed);

    // B 路重算一遍同一个 seed
    const g = new FlappyGame({ seed, hitmasks: HITMASKS });
    g.reset(); local.reset();
    const ref = [];
    for (let f = 0; f < Math.min(FRAMES, got.actions.length); f++) {
      const a = local.decide(g);
      ref.push(a);
      if (g.step(a).done) break;
    }

    // 逐帧比对；同时用 A 路的动作重放一遍游戏，确认状态也一致
    const n = Math.min(ref.length, got.actions.length);
    let firstDiff = -1;
    for (let f = 0; f < n; f++) if (ref[f] !== got.actions[f]) { firstDiff = f; break; }

    const g2 = new FlappyGame({ seed, hitmasks: HITMASKS });
    g2.reset();
    let f2 = 0;
    for (; f2 < n; f2++) if (g2.step(got.actions[f2]).done) { f2++; break; }
    const sameState = (g2.score === g.score)
      && Math.abs(g2.observeState().playery - g.observeState().playery) < 1e-9;

    out.push({ seed, frames: n, firstDiff, refScore: g.score, replayScore: g2.score, sameState });
  }
  w.terminate();
  return JSON.stringify(out);
})()`;

await send('Page.navigate', { url });
await sleep(1500);
const r = await send('Runtime.evaluate',
  { expression: EXPR, awaitPromise: true, returnByValue: true, timeout: 300000 });
ws.close();
proc.kill();

if (!r || !r.result || r.result.value === undefined) {
  console.error('页面内比对没跑起来：', JSON.stringify(r).slice(0, 500));
  process.exit(2);
}
const rows = JSON.parse(r.result.value);
let bad = false;
for (const x of rows) {
  const ok = x.firstDiff === -1 && x.sameState && x.refScore === x.replayScore;
  if (!ok) bad = true;
  console.log(`${ok ? ' ok ' : 'FAIL'}  seed=${String(x.seed).padStart(10)}`
    + `  比对 ${String(x.frames).padStart(3)} 帧`
    + `  分数 worker回放=${x.replayScore} 直接推理=${x.refScore}`
    + (x.firstDiff === -1 ? '' : `  首个不同的帧: ${x.firstDiff}`)
    + (x.sameState ? '' : '  末状态不一致'));
}
console.log(bad ? '\nWORKER PARITY FAILED' : '\nWORKER PARITY OK  预跑回放与直接推理逐位一致');
process.exit(bad ? 1 : 0);
