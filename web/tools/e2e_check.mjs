/**
 * 误操作与生命周期的端到端检查（web_plan.md §3.2）。
 *
 * 为什么单独一个脚本，而不是并进 viewport_check
 * ---------------------------------------------
 * viewport_check 问的是"**静止**的页面在这个尺寸下长得对不对"，一次加载量一遍。
 * 这里问的是"**被乱操作**之后还活不活得下来"，每个用例都要一次干净的加载
 * （上一个用例把页面搞成什么样，不能影响下一个）。两件事混在一个循环里
 * 会互相污染，出了问题也分不清是谁的锅。
 *
 * 判据的写法
 * ----------
 * 每个用例都必须**有可能变红**。只断言"没有 JS 报错"是不够的 —— 这个页面
 * 大部分故障（卡住、白屏、两局叠加）都不抛异常。所以每条都额外断言一个
 * **正向的功能事实**：还能开局、分数还在涨、锁步没断。
 *
 * 每条判据下面都写了它对应的负向对照怎么造（`--only <名字>` 单独跑一条，
 * 配合 `--injectjs` 注入破坏）。没有负向对照的判据不算判据。
 *
 * 用法：
 *     python3 -m http.server 8123 --directory web &
 *     node web/tools/e2e_check.mjs                 # 全部用例
 *     node web/tools/e2e_check.mjs --only refresh  # 只跑一条
 *     node web/tools/e2e_check.mjs --list
 * 退出码：0 = 全过；1 = 有用例失败；2 = 环境问题（起不来浏览器 / 服务没开）。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : dflt;
};
const url = argv.find((a, i) => !a.startsWith('--')
  && (i === 0 || !argv[i - 1].startsWith('--'))) || 'http://localhost:8123/index.html';
const only = opt('only', null);
const injectJs = opt('injectjs', null);
const shotDir = opt('shots', null);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 浏览器查找：和 viewport_check / stall_check 用同一套（云端把 Chromium 放在
// /opt/pw-browsers/chromium-<版本>/ 下，目录名会变，写死路径等于找不到）。
function findChrome() {
  const cands = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const p of cands) { try { if (fs.existsSync(p)) return p; } catch { /* 忽略 */ } }
  try {
    const root = '/opt/pw-browsers';
    for (const d of fs.readdirSync(root).filter((x) => x.startsWith('chromium')).sort().reverse()) {
      const p = path.join(root, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  } catch { /* 不是这个环境 */ }
  return null;
}

const chrome = findChrome();
if (!chrome) { console.error('没找到 Chrome/Chromium，设 CHROME_PATH 指过去'); process.exit(2); }

try {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  // **必须把 body 读掉**：只看 r.ok 而不消费响应体的话，undici 会让这条 socket
  // 停在 paused 状态；python http.server 是 HTTP/1.0、响应完就关连接，于是命中
  // Node 内部的 assert(!this.paused)，整个进程崩在一段看不懂的 undici 栈里。
  await r.arrayBuffer();
} catch (e) {
  console.error(`打不开 ${url}（${e.message}）。先起服务：`);
  console.error('    python3 -m http.server 8123 --directory web &');
  process.exit(2);
}

const PORT = 9700 + Math.floor(Math.random() * 200);
// profile 目录带 pid：固定名字的话，上一次没退干净的 Chromium 占着目录，
// 新实例起不来 CDP，表现成"超时"，很难查。
const profile = path.join(os.tmpdir(), 'flappy-e2e-' + process.pid);
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`, '--user-data-dir=' + profile,
  'about:blank',
], { stdio: 'ignore' });

let target = null;
for (let i = 0; i < 60 && !target; i++) {
  await sleep(250);
  try {
    const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    target = (await res.json()).find((t) => t.type === 'page');
  } catch { /* 浏览器还没起来 */ }
}
if (!target) { proc.kill(); console.error(`Chromium 没能在 ${PORT} 上起 CDP`); process.exit(2); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
const pending = new Map();
let seq = 0;
// worker 是**独立的 CDP target**。页面这条会话上的 Network 域管不到它 ——
// 权重就是在 worker 里下的，所以限速/拦截想作用到权重，必须挂到 worker 的会话上。
// 这一点踩过两次坑（weights-dead 和 slow-network 第一版都白测了），
// 所以这里留了 sessionId 通道。
const workerSessions = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Target.attachedToTarget'
      && /worker/i.test(m.params.targetInfo.type)) {
    workerSessions.push(m.params.sessionId);
  }
};
await new Promise((r) => { ws.onopen = r; });
const send = (method, params = {}, sessionId) => new Promise((res) => {
  const id = ++seq;
  pending.set(id, (m) => res(m.result ?? m.error));
  ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
});

await send('Page.enable');
await send('Runtime.enable');
await send('Network.enable');
// flatten 模式下 worker 起来就自动附着，用来拿它的 sessionId。
// **waitForDebuggerOnStart 必须是 false。** 开成 true 的话每个 worker 一创建
// 就挂在等调试器上、权重永远下不完，后面每条用例都报"没 ready" ——
// 单跑每条都过、一起跑挂四条，就是这么来的。只有 slow-network 用例
// 在真要限速时才会临时打开它。
await send('Target.setAutoAttach',
  { autoAttach: true, waitForDebuggerOnStart: false, flatten: true });

const evalIn = async (expr) => {
  const r = await send('Runtime.evaluate',
    { expression: expr, returnByValue: true, awaitPromise: true });
  return r && r.result ? r.result.value : undefined;
};

// 页面里读得到的一切都从这里出。刻意只用 DOM + window.__stats，
// 不往 index.html 里加测试专用的钩子 —— 页面不该为了被测而长出旁路。
const SNAP = `(() => {
  const s = (window.__stats && window.__stats()) || {};
  const txt = (sel) => { const el = document.querySelector(sel); return el ? el.textContent : null; };
  const veil = document.querySelector('#youVeil');
  return JSON.stringify({
    ready: !!s.ready, worker: !!s.worker, plan: s.plan | 0, aiFrame: s.aiFrame | 0,
    stalls: s.stalls | 0, prefetch: s.prefetch | 0,
    youScore: Number(txt('#youLive') || -1), aiScore: Number(txt('#aiLive') || -1),
    rounds: Number(txt('#youRounds') || -1), best: Number(txt('#youBest') || -1),
    playing: !!(veil && veil.hidden),
    errors: window.__pageErrors || [],
  });
})()`;

const snap = async () => JSON.parse(await evalIn(SNAP));

// 每个用例都从一次干净的加载开始
// 上一次 load 注入的那段脚本的 id。**必须删掉再加新的** ——
// addScriptToEvaluateOnNewDocument 是累加的，不删的话
// worker-fallback 里那句 `window.Worker=throw` 会一直留到后面每个用例，
// 于是后面的用例测的根本不是它以为的那个页面。
let priorScript = null;

async function load(pre = '') {
  // 每个用例开头都把生命周期状态掰回 active。background-tab 会把页面 frozen 掉，
  // 恢复之后 requestAnimationFrame 未必真的重新跑起来 —— 下一个需要画面推进的
  // 用例就会看到 aiFrame 一直是 0，症状像"页面坏了"，其实是上一条留下的。
  // （实测：worker-fallback 单跑必过，跟在 background-tab 后面必挂。）
  await send('Page.setWebLifecycleState', { state: 'active' });
  if (priorScript) {
    await send('Page.removeScriptToEvaluateOnNewDocument', { identifier: priorScript });
    priorScript = null;
  }
  // pre 在页面脚本之前执行，用来造"这个环境没有 X"这类前置条件。
  const added = await send('Page.addScriptToEvaluateOnNewDocument', {
    source: 'window.__pageErrors=[];'
      + 'addEventListener("error",e=>__pageErrors.push(String(e.message)));'
      + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));'
      + pre
      + (injectJs ? injectJs : ''),
  });
  priorScript = added && added.identifier;
  await send('Page.navigate', { url });
  await sleep(400);
}

async function waitReady(maxMs = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    const s = await snap();
    if (s.ready) return s;
    await sleep(200);
  }
  return null;
}

async function tap() {
  // 页面监听的是 #arena 上的 pointerdown。鼠标事件在 Chrome 里会产出 pointerdown，
  // 所以这里用 mousePressed/Released，不需要专门造 PointerEvent。
  const box = JSON.parse(await evalIn(`(() => { const r =
    document.querySelector('#arena').getBoundingClientRect();
    return JSON.stringify({ x: r.left + r.width * 0.25, y: r.top + r.height * 0.5 }); })()`));
  for (const type of ['mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent',
      { type, x: box.x, y: box.y, button: 'left', clickCount: 1 });
  }
}

async function key(code = 'Space') {
  for (const type of ['keyDown', 'keyUp']) {
    await send('Input.dispatchKeyEvent',
      { type, code, key: code === 'Space' ? ' ' : code, windowsVirtualKeyCode: code === 'Space' ? 32 : 82 });
  }
}

/**
 * 开一局，等到分数涨起来为止。用来证明"页面还活着"。
 *
 * **只点一次**。第一版写成"只要 veil 可见就再点一下"，结果是：脚本不会扇翅膀，
 * 玩家一两秒就撞了、veil 重新出现、脚本又点一下开新局，把 **AI 的进度一起清零** ——
 * 于是分数永远回不到 1，用例报"开不了局"，而页面其实好好的。
 * 玩家死后 AI 会继续飞（这正是 B1 修好的行为），所以点一次之后安静等着看 aiScore 就行。
 */
async function playUntilScore(minScore = 1, maxMs = 25000) {
  const t0 = Date.now();
  let s = await snap();
  if (!s.playing) await tap();
  while (Date.now() - t0 < maxMs) {
    await sleep(300);
    s = await snap();
    if (s.youScore >= minScore || s.aiScore >= minScore) return s;
  }
  return s;
}


/** 限速。用例 1 需要"未 ready"这个窗口真的存在，用例 6 直接测慢网。 */
async function throttle(kbps) {
  await send('Network.emulateNetworkConditions', {
    offline: false, latency: kbps ? 150 : 0,
    downloadThroughput: kbps ? (kbps * 1024) / 8 : -1,
    uploadThroughput: kbps ? (kbps * 1024) / 8 : -1,
  });
}


/**
 * 自带一个"按字节滴流"的静态服务器，专门给 slow-network 用例。
 *
 * 为什么不用 CDP 的 Network.emulateNetworkConditions：**权重是在 worker 里下的**，
 * worker 是独立 target，页面会话上的限速管不到它。挂到 worker 会话上也不行 ——
 * 本地 2.5 MB 两百毫秒就下完，等我们 attach 上去再限速，早下完了
 * （实测第一次采样就已经 ready=true）。
 * 从服务端把字节喂慢，本来是唯一不依赖 CDP 时序的做法。
 *
 * **但这个服务器本身把 module worker 弄坏了，所以这条用例目前是废的。**
 * 实测：页面自己的模块图正常（index.html / game.js / render.js / assets 都请求到了），
 * `ai-worker.js` 也服出去了，但**worker 的 import 一个都没发**（nn.js / obs.js /
 * model/weights-meta.js 全都没有请求），worker 既不报错也不 ready，就那么挂着。
 * 把滴流整个关掉、变成一个普通的静态服务器，症状**一模一样** —— 所以不是"喂得慢"
 * 的问题，是这个手写服务器和 Chrome 的 module worker 之间有别的不兼容。
 * 换回 python -m http.server 立刻正常。
 *
 * 也试过 CDP，两条都不行：
 *   - 页面会话上 Network.emulateNetworkConditions：**权重是在 worker 里下的**，
 *     worker 是独立 target，页面这条域管不到（weights-dead 也栽在这个上）。
 *   - 挂到 worker 会话上限速：本地 2.5 MB 两百毫秒下完，等 attach 上去早结束了；
 *     加了 Target.setAutoAttach + waitForDebuggerOnStart 也没能真的把 worker 拦住，
 *     实测第一次采样就已经 ready=true。
 *
 * 所以这条现在的状态是**未能验证**，不是"通过"。要做完得先解决那个不兼容
 * （或者换成真的能限速 worker 的办法）。
 */
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
  '.mjs': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml' };

function startDripServer(rootDir, bytesPerSec) {
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    if (process.env.DRIP_LOG) console.log('  [drip] ' + rel);
    const file = path.join(rootDir, rel);
    if (!file.startsWith(rootDir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('nope'); return;
    }
    const buf = fs.readFileSync(file);
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
      'Content-Length': String(buf.length),
    });
    if (!/_fp16\.bin$/.test(file)) { res.end(buf); return; }
    // 只有权重走滴流。Content-Length 照常给，页面的进度条才有分母。
    const chunk = Math.max(1, Math.floor(bytesPerSec / 20));   // 每 50 ms 一小口
    let at = 0;
    const timer = setInterval(() => {
      if (at >= buf.length) { clearInterval(timer); res.end(); return; }
      res.write(buf.subarray(at, at + chunk));
      at += chunk;
    }, 50);
    res.on('close', () => clearInterval(timer));
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

const cases = [];
const def = (name, desc, fn) => cases.push({ name, desc, fn });

// ---------------------------------------------------------------- 用例 1
def('boot-spam', '权重还没下完就狂点狂按空格', async () => {
  const notes = [];
  // **必须限速**：本地 2.5 MB 的权重几百毫秒就下完了，不限速的话
  // "未 ready 时点击"这个窗口根本不存在，用例会在什么都没测到的情况下变绿。
  // 4 Mbps 下大约 5 秒，足够把十几次乱点全打在加载期里。
  await throttle(4000);
  await load();
  for (let i = 0; i < 12; i++) { await tap(); await key('Space'); }
  const before = await snap();
  if (before.ready) {
    notes.push('限速了还是在点击前就 ready —— 这一轮没测到未 ready 时的点击，判为失败');
  }
  const r = await waitReady();
  await throttle(0);
  if (!r) { notes.push('权重加载完不成，ready 一直是 false'); return notes; }
  // 关键判据：乱点过之后**还能正常开局**。加载期的点击不能把状态机顶进
  // 一个"看起来 ready 但按了没反应"的死角。
  const s = await playUntilScore(1);
  if (s.youScore < 1 && s.aiScore < 1) {
    notes.push(`乱点之后开不了局：playing=${s.playing} you=${s.youScore} ai=${s.aiScore}`);
  }
  for (const e of s.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// ---------------------------------------------------------------- 用例 2
def('refresh', '游戏进行中刷新，必须能重新开局', async () => {
  const notes = [];
  await load();
  if (!await waitReady()) { notes.push('首次加载就没 ready'); return notes; }
  await playUntilScore(1);
  await send('Page.reload', { ignoreCache: false });
  await sleep(600);
  const r = await waitReady();
  if (!r) { notes.push('刷新之后 ready 一直是 false'); return notes; }
  const s = await playUntilScore(1);
  if (s.youScore < 1 && s.aiScore < 1) {
    notes.push(`刷新之后开不了局：playing=${s.playing} you=${s.youScore} ai=${s.aiScore}`);
  }
  if (s.rounds < 0) notes.push('刷新后计数器读不到');
  for (const e of s.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// ---------------------------------------------------------------- 用例 3
def('rapid-restart', '连点"下一局"/换管道序列，不能出现两局叠加', async () => {
  const notes = [];
  await load();
  if (!await waitReady()) { notes.push('没 ready'); return notes; }
  await playUntilScore(1);
  // 连点换管道序列按钮：每次都会 gen++ 作废上一局，最容易暴露竞态。
  for (let i = 0; i < 10; i++) {
    await evalIn(`document.querySelector('#reseed').click()`);
    await sleep(60);
  }
  await sleep(1200);
  const s1 = await snap();
  // 判据 a：aiFrame 不能跑到 plan 前面 —— 那意味着在回放一段还没算出来的计划，
  // 也就是两局的数据串了。
  if (s1.aiFrame > s1.plan) {
    notes.push(`回放越过了计划：aiFrame=${s1.aiFrame} > plan=${s1.plan}（两局叠加）`);
  }
  // 判据 b：连点之后还能正常玩
  const s2 = await playUntilScore(1);
  if (s2.youScore < 1 && s2.aiScore < 1) {
    notes.push(`连点之后开不了局：playing=${s2.playing} you=${s2.youScore} ai=${s2.aiScore}`);
  }
  if (s2.aiFrame > s2.plan) {
    notes.push(`回放越过了计划（第二次量）：aiFrame=${s2.aiFrame} > plan=${s2.plan}`);
  }
  for (const e of s2.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// ---------------------------------------------------------------- 用例 5
def('worker-fallback', 'worker 起不来时退化到主线程推理，而不是白屏', async () => {
  const notes = [];
  // 造法是让 `new Worker` 直接抛，正好命中 index.html 里那句
  //     try { worker = new Worker(...) } catch { worker = null; }
  // 这就是"老浏览器 / CSP 禁掉 worker"在页面里的真实表现。
  //
  // 试过 Network.setBlockedURLs('*ai-worker.js')，**没用**：模块 worker 的脚本
  // 请求不走页面那条网络域，拦不到；而且 `new Worker` 本身是同步返回的，
  // 脚本 404 要等 onerror 才知道，快照里 worker 仍然是"在的"。
  //
  // 也没有按计划原文去拦 ./model/*.bin —— 那样主线程的 createAi() 同样下不到
  // 权重，两条路一起断，测的就不是退化路径了。那个场景单独放在 weights-dead。
  await load('window.Worker=function(){throw new Error("worker blocked (test)")};');
  const r = await waitReady(40000);
  if (!r) { notes.push('worker 拦掉之后 ready 一直是 false —— 没退化成功'); return notes; }
  if (r.worker) notes.push('worker 说自己还在，拦截没生效，这一轮没测到退化路径');
  // 判据是"**帧在推进**"，不是"过了几根管子"。退化路径的推理是同步跑在主线程上的
  // （约 30 ms 一次决策），机器一忙就可能几十秒都过不了一根 —— 拿分数当判据
  // 会变成一条看机器心情的用例（实测：单跑两次都过，整套一起跑就挂）。
  // "没白屏"的真正含义是画面在动，aiFrame 在涨就够了。
  const a = await snap();
  if (!a.playing) await tap();
  // 最多等 20 秒。固定睡 4 秒是不够的：整套用例连着跑的时候机器已经很忙，
  // 实测出现过"单跑两次都过、整套一起跑 aiFrame 0->0"。
  // 判据没有放水 —— 帧终究必须动起来，只是给够时间。
  let b = a;
  const t0 = Date.now();
  while (Date.now() - t0 < 20000) {
    await sleep(500);
    b = await snap();
    if (b.aiFrame > a.aiFrame) break;
    if (!b.playing) await tap();
  }
  if (b.aiFrame <= a.aiFrame) {
    notes.push(`退化路径下画面不动：aiFrame ${a.aiFrame} -> ${b.aiFrame}（20 秒内没推进）`);
  }
  for (const e of b.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// ---------------------------------------------------------------- 用例 5b
def('weights-dead', '权重下不来时要说人话，不能是一片空白', async () => {
  const notes = [];
  // 这里**必须同时**禁掉 worker。
  // 只发 Network.setBlockedURLs 是没用的：权重是在 **worker 内部** fetch 的，
  // 而 worker 是独立的 CDP target，页面这条 Network 域管不到它 ——
  // 第一版就是这样，实测 ready=true / plan=608，页面根本没受影响，
  // 用例却在报"没给出失败提示"，是测试自己错了。
  // 禁掉 worker 之后走主线程的 createAi()，那条 fetch 归页面管，拦得住，
  // 命中的正是 fallbackToMainThread() 里的 catch。
  await send('Network.setBlockedURLs', { urls: ['*_fp16.bin'] });
  await load('window.Worker=function(){throw new Error("worker blocked (test)")};');
  await sleep(8000);
  await send('Network.setBlockedURLs', { urls: [] });
  const st = await snap();
  if (st.ready) {
    notes.push(`权重被拦了 ready 却是 true（plan=${st.plan}）—— 拦截没生效，这一轮没测到失败路径`);
  }
  const veil = await evalIn(
    `(document.querySelector('#youVeil') || {}).textContent || ''`);
  const hidden = await evalIn(
    `!!(document.querySelector('#youVeil') || {}).hidden`);
  // 判据：遮罩要在、并且要有字。白屏或者一个空遮罩都是不合格的失败方式。
  if (hidden) notes.push('权重下不来，遮罩却被藏起来了 —— 玩家看到的是一片空白');
  if (!/could not load|reload/i.test(veil)) {
    notes.push(`没有给出可读的失败提示，遮罩文字是："${veil}"`);
  }
  return notes;
});

// ---------------------------------------------------------------- 用例 6
// **默认不跑**，要加 --slow 才会执行。原因见下面的长注释：这条目前**测不出来**，
// 我不想让一条其实什么都没验证的用例挂在全绿的列表里冒充覆盖率。
def('slow-network', '慢网下进度条要动，不能卡在 0%（--slow 才跑，见注释）', async () => {
  if (!argv.includes('--slow')) {
    return ['（提示）跳过：这条目前没有能用的造慢网手段，见脚本里的注释'];
  }
  const notes = [];
  const webRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
  const { server, port } = await startDripServer(webRoot, 400 * 1024 / 8);   // 约 400 kbps
  try {
    await load();
    await send('Page.navigate', { url: `http://127.0.0.1:${port}/index.html` });
    const seen = new Set();
    let sawNonZero = false;
    for (let i = 0; i < 60; i++) {
      await sleep(400);
      const st = await evalIn(`(() => {
        const f = document.querySelector('#bootFill');
        const p = document.querySelector('#bootPct');
        return JSON.stringify({ w: f ? f.style.width : null, t: p ? p.textContent : null });
      })()`);
      const o = JSON.parse(st || '{}');
      if (o.w) { seen.add(o.w); if (parseFloat(o.w) > 0) sawNonZero = true; }
      if (o.t) seen.add('t:' + o.t);
      if (seen.size >= 5 && sawNonZero) break;
    }
    if (!sawNonZero) notes.push('进度条宽度全程是 0 —— 慢网下首屏没有任何反馈');
    if (seen.size < 3) {
      notes.push(`进度只出现了 ${seen.size} 个不同的值，看不出在动：${[...seen].join(' | ')}`);
    }
  } finally {
    server.close();
  }
  return notes;
});

// ---------------------------------------------------------------- 用例 7
def('touch-misfire', '移动端误触不能缩放页面或选中文字', async () => {
  const notes = [];
  await load();
  if (!await waitReady()) { notes.push('没 ready'); return notes; }
  // 造一个真的触摸环境：mobile:true 不会让 (pointer: coarse) 成立，
  // 也不会给出 maxTouchPoints，必须单独发这条（B4 那轮实测过）。
  await send('Emulation.setDeviceMetricsOverride',
    { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
  await send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
  await sleep(400);

  const css = JSON.parse(await evalIn(`(() => {
    const b = getComputedStyle(document.body);
    const a = getComputedStyle(document.querySelector('#arena'));
    return JSON.stringify({
      touchAction: b.touchAction,
      userSelect: b.userSelect || b.webkitUserSelect,
      arenaSelect: a.userSelect || a.webkitUserSelect,
    });
  })()`));
  // touch-action: manipulation 就是"认单击和滑动，但不认双击缩放"。
  if (!/manipulation|none/.test(css.touchAction)) {
    notes.push(`body 的 touch-action 是 "${css.touchAction}"，双击会缩放页面`);
  }
  if (css.userSelect !== 'none') {
    notes.push(`body 的 user-select 是 "${css.userSelect}"，长按会选中文字`);
  }

  // 双击：两次快速点击，之后视口缩放必须还是 1
  for (let i = 0; i < 2; i++) { await tap(); await sleep(60); }
  await sleep(600);
  const scale = await evalIn(
    `(window.visualViewport && visualViewport.scale) || 1`);
  if (Math.abs(scale - 1) > 0.01) notes.push(`双击之后页面被缩放到 ${scale}`);

  await send('Emulation.clearDeviceMetricsOverride');
  await send('Emulation.setTouchEmulationEnabled', { enabled: false });
  const s = await snap();
  for (const e of s.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// ---------------------------------------------------------------- 用例 4（放最后）
// **这条必须排在最后。** 它把页面 frozen 掉，恢复之后 requestAnimationFrame
// 不一定真的重新跑起来（headless 里页面还处在 hidden），于是**后面**任何需要
// 画面推进的用例都会看到 aiFrame 恒为 0，症状像"页面坏了"，其实是这条留下的。
// 实测：worker-fallback 单跑必过、跟在这条后面必挂；在 load() 里补一句
// Page.setWebLifecycleState('active') 也救不回来。排到最后是最省事且可靠的做法。
def('background-tab', '切后台再切回来，不能一次性补几百帧', async () => {
  const notes = [];
  await load();
  if (!await waitReady()) { notes.push('没 ready'); return notes; }
  const before = await playUntilScore(1);
  // 冻结页面：rAF 停摆，回来时 dt 是好几秒。index.html 里
  // `acc = Math.min(acc + dt, STEP_MS * 4)` 就是为了钳住这一下。
  await send('Page.setWebLifecycleState', { state: 'frozen' });
  await sleep(3000);
  await send('Page.setWebLifecycleState', { state: 'active' });
  await sleep(120);          // 只给一帧多一点的时间，量的是"回来那一下"
  const after = await snap();
  const jump = Math.max(after.youScore - before.youScore, after.aiScore - before.aiScore);
  // 3 秒冻结 = 约 180 帧。没有钳制的话回来那一下会一口气推完，分数跳好几根；
  // 钳到 4 步（STEP_MS*4）意味着最多推进不到一根管子。给 2 根的余量。
  if (jump > 2) {
    notes.push(`回来那一下跳了 ${jump} 根（you ${before.youScore}->${after.youScore}, `
      + `ai ${before.aiScore}->${after.aiScore}）—— 时间累积量没被钳住`);
  }
  for (const e of after.errors) notes.push('JS 报错: ' + e);
  return notes;
});

// 收尾。**不能**直接 ws.close(); proc.kill(); process.exit() 一把梭 ——
// Windows 上 libuv 会在句柄还没关完的时候撞上
// `Assertion failed: !(handle->flags & UV_HANDLE_CLOSING)`，
// 进程带着一个和测试内容毫无关系的 crash 退出。让出一拍再退。
function finish(code) {
  try { ws.close(); } catch { /* 已经关了 */ }
  try { proc.kill(); } catch { /* 已经没了 */ }
  setTimeout(() => process.exit(code), 120);
}

if (argv.includes('--list')) {
  for (const c of cases) console.log(`${c.name.padEnd(18)} ${c.desc}`);
  finish(0);
}

const run = only ? cases.filter((c) => c.name === only) : cases;
if (!run.length) { console.error(`没有叫 ${only} 的用例，--list 看全部`); finish(2); }

console.log(`目标 ${url}\n浏览器 ${chrome}\n`);
let failed = 0;
for (const c of run) {
  let notes;
  try {
    notes = await c.fn();
  } catch (e) {
    notes = ['用例自己抛了：' + (e && e.message)];
  }
  const hard = notes.filter((n) => !n.startsWith('（提示）'));
  const bad = hard.length > 0;
  if (bad) failed++;
  console.log(`${bad ? 'FAIL' : ' ok '}  ${c.name.padEnd(18)} ${c.desc}`);
  for (const n of notes) console.log('        ' + n);
  if (shotDir) {
    fs.mkdirSync(shotDir, { recursive: true });
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    if (shot && shot.data) {
      fs.writeFileSync(path.join(shotDir, `e2e-${c.name}.png`), Buffer.from(shot.data, 'base64'));
    }
  }
}

console.log(`\n${run.length} 个用例，${run.length - failed} 过 ${failed} 挂`);
console.log('注意：这是 Blink 的无头浏览器，不是 iOS 的 WebKit —— 真机结论见 web/TESTING.md');
console.log(failed ? 'E2E CHECK FAILED' : 'E2E CHECK OK');
finish(failed ? 1 : 0);
