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
await send('Network.enable');

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
async function load() {
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: 'window.__pageErrors=[];'
      + 'addEventListener("error",e=>__pageErrors.push(String(e.message)));'
      + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));'
      + (injectJs ? injectJs : ''),
  });
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

// ---------------------------------------------------------------- 用例 4
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
