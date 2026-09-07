/**
 * 断流回归：**玩家撞死之后，AI 的动作计划必须继续生产。**
 *
 * 这条检查钉的是哪个 bug
 * ----------------------
 * 推理搬进 worker 之后有过一版：玩家一撞，`endRound()` 立刻为**下一局**发
 * `start`，而 `ai-worker.js: startJob()` 第一件事是 `gen++`，当前这局的
 * `produce()` 循环条件 `job.gen === myGen` 当场不成立、生产线程退出；
 * 同时 `stepBoth()` 里续要计划的条件带着 `youAlive`，玩家死了就再也不发 `want`。
 * 两处叠加的后果是：玩家一死，AI 只能回放缓冲区里剩下的存货，放完就永远卡在
 * 半空 —— 计数器不动，`The AI crashed` 永远不出现。
 *
 * 这个 bug **headless 六项检查全绿也抓不到**，是真机用户报的。原因写在
 * PROGRESS.md 里：CDP 的 `setCPUThrottlingRate` 不节流 worker 线程，测试里
 * worker 全速跑、缓冲区一直很厚，"AI 续飞"每次都过。所以这个脚本**不看
 * worker 快不快**，只看一件与机器速度无关的事：
 *
 *     玩家死亡那一刻的 plan.length = P0；
 *     之后 plan.length 必须继续涨，aiFrame 必须越过 P0。
 *
 * 断流版本里这两个量在玩家死后立刻冻住（PROGRESS 记的实测：224 -> 224），
 * 修好的版本持续增长（288 -> 976）。这个判据不依赖 worker 的速度，
 * 快机器慢机器都成立 —— 快机器只是让 P0 更大、跑得更久才越得过去。
 *
 * 为什么判据不是"玩家在第 5 根管子撞死"
 * -----------------------------------
 * 计划里写的是让脚本驱动玩家飞到第 5 根再主动撞。但这里没有可靠的办法让
 * 外部脚本把那只鸟飞过 5 根管子：页面只暴露 `window.__stats()`，读不到鸟的
 * 高度，而 30 fps 下靠 CDP 往回打按键有几十毫秒的抖动，飞多远纯看运气。
 * 更要紧的是**玩家死在第几根与这个 bug 无关** —— 断流是由"玩家死"这个事件
 * 触发的，不是由分数触发的。所以脚本让玩家扇几下翅膀然后松手摔死，
 * 玩家分数只打印、不作判据。别为了凑计划里的措辞把判据写成一个随机量。
 *
 * 判据（全部满足才退出 0）
 * ------------------------
 *   1. 玩家死后 `plan.length` 至少涨过 3 次，且末值 > P0
 *   2. `aiFrame` 越过 P0 至少 60 帧（= AI 吃完了死亡时的存货还在飞）
 *   3. AI 分数达到 --target（默认 35，断流版本的天花板约 18~20 根）
 *      —— 或者 AI 自己撞死且页面确实显示了 `The AI crashed`
 *   4. 页面没有 JS 报错
 *
 * AI 的分数是重尾的（12 局实测：中位 88，但最低 3、最高 290），
 * 抽到早死的 seed 会让第 3 条无从判断，所以这种局判为"无效"并自动换一局重来，
 * 最多 --rounds 次。**这不是重试掩盖失败** —— 只有"AI 没到 target 就自己死了"
 * 才换局，卡住（1、2 条不成立）当场判失败，不换局。
 *
 * 用法（需要先起本地服务）：
 *     python3 -m http.server 8123 --directory web &
 *     node web/tools/stall_check.mjs [url] [--target 35] [--rounds 3] [--budget 180]
 *
 * 负向对照（确认这条检查真的抓得住 bug）：把 index.html 改回断流版本
 * （`stepBoth` 的续命条件加回 `youAlive`、`endRound()` 里加回预取），
 * 另存一份再 `node web/tools/stall_check.mjs http://localhost:8123/那份.html`，
 * 必须退出 1。做法与实测结果记在 PROGRESS.md 的 B1 那一条里。
 *
 * 零依赖：Node 内置 fetch + WebSocket 直连 CDP，不需要 Playwright。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? Number(argv[i + 1]) : dflt;
};
const url = argv.find((a) => !a.startsWith('--') && !/^\d+$/.test(a))
  || 'http://localhost:8123/index.html';
const TARGET = flag('target', 35);      // AI 要飞到几根才算证明"没卡住"
const MAX_ROUNDS = flag('rounds', 3);   // AI 早死属于无效局，最多换几次
const BUDGET_S = flag('budget', 180);   // 单局观察上限（秒）
const WARMUP_MS = flag('warmup', 3000); // 玩家扇翅多久之后松手摔死

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 常见的 Chromium 位置。云端的 routine 容器把浏览器放在 /opt/pw-browsers 下，
// 版本号会变，所以那一条按目录名扫，不要写死版本 —— 写死的话下次镜像一升级
// 这个脚本就静默地找不到浏览器。
function findChrome() {
  const fixed = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const p of fixed) { try { if (fs.existsSync(p)) return p; } catch { /* 忽略 */ } }
  const root = '/opt/pw-browsers';
  try {
    for (const d of fs.readdirSync(root).filter((d) => d.startsWith('chromium')).sort().reverse()) {
      const p = path.join(root, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  } catch { /* 不是这个环境 */ }
  return null;
}
const chrome = findChrome();
if (!chrome) { console.error('没找到 Chrome/Chromium，设 CHROME_PATH 指过去'); process.exit(2); }

try {
  const r = await fetch(url, { method: 'GET' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
} catch (e) {
  console.error(`打不开 ${url}（${e.message}）。先起服务：`);
  console.error('    python3 -m http.server 8123 --directory web &');
  process.exit(2);
}

const PORT = 9600 + Math.floor(Math.random() * 300);
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`,
  '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-stall-check'),
  'about:blank',
], { stdio: 'ignore' });

let target = null;
for (let i = 0; i < 60 && !target; i++) {
  await sleep(250);
  try {
    target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json())
      .find((t) => t.type === 'page');
  } catch { /* 浏览器还没起来 */ }
}
if (!target) { proc.kill(); throw new Error(`Chromium 没能在 ${PORT} 上起 CDP：${chrome}`); }

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
const evaluate = async (expr) => (await send('Runtime.evaluate',
  { expression: expr, returnByValue: true, awaitPromise: true }))?.result?.value;

await send('Page.enable');
await send('Runtime.enable');
await send('Page.addScriptToEvaluateOnNewDocument', {
  source: 'window.__pageErrors=[];addEventListener("error",e=>__pageErrors.push(String(e.message)));'
    + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));',
});
await send('Page.navigate', { url });

/* 探针只读页面**已有**的东西：`window.__stats()` 加三个 DOM 节点。
   不往 index.html 里加测试专用的钩子 —— 玩家死没死、AI 死没死都从遮罩上读，
   读的就是访客真正看到的那一层，钩子再准也不能证明画面对。 */
const PROBE = `(() => {
  const veil = document.getElementById('youVeil');
  const aiVeil = document.getElementById('aiVeil');
  const s = (window.__stats && window.__stats()) || null;
  return JSON.stringify({
    ok: !!s, stats: s,
    youLive: Number(document.getElementById('youLive')?.textContent || 0),
    aiLive: Number(document.getElementById('aiLive')?.textContent || 0),
    youCrashed: !!veil && !veil.hidden && /You crashed/.test(veil.innerHTML),
    aiCrashed: !!aiVeil && !aiVeil.hidden,
    aiVeilText: aiVeil && !aiVeil.hidden ? aiVeil.textContent : '',
    errors: window.__pageErrors || [],
  });
})()`;
const probe = async () => JSON.parse(await evaluate(PROBE));

// 按键走真的 keydown：press() 挂在 window 上，dispatchEvent 就能触发。
// 不用 Input.dispatchKeyEvent 是因为它要 rawKeyDown/char 两条才稳，没必要。
const tap = () => evaluate(
  `dispatchEvent(new KeyboardEvent('keydown',{code:'Space',bubbles:true}))`);

const fail = (msg) => { console.log('FAIL  ' + msg); failed = true; };
let failed = false;

// 等权重（2.5 MB）加载完
let boot = null;
for (let i = 0; i < 120; i++) {
  await sleep(500);
  boot = await probe().catch(() => null);
  if (boot?.ok && boot.stats.ready) break;
}
if (!boot?.ok || !boot.stats.ready) {
  console.log('FAIL  60 秒内页面没就绪（__stats().ready 一直是 false）');
  ws.close(); proc.kill(); process.exit(1);
}
if (!boot.stats.worker) {
  // 退化路径是主线程同步推理，根本没有"计划"这回事，这条检查无从谈起。
  // 静默跳过等于以后 worker 挂了也没人知道，所以判失败。
  console.log('FAIL  worker 没起来，页面走的是主线程退化路径，这条检查测不了');
  ws.close(); proc.kill(); process.exit(1);
}
console.log(`就绪：worker=on  单次推理 ${boot.stats.lastMs?.toFixed?.(0) ?? '?'} ms`);

let verdict = null;   // { round, ... } 一局有效局的结果
for (let round = 1; round <= MAX_ROUNDS && !verdict && !failed; round++) {
  // ---- 开局：扇一会儿翅膀，然后松手，让玩家自己摔死 ----
  await tap();
  const warmEnd = Date.now() + WARMUP_MS;
  while (Date.now() < warmEnd) { await tap(); await sleep(320); }

  let p = await probe();
  for (let i = 0; i < 60 && !p.youCrashed; i++) { await sleep(250); p = await probe(); }
  if (!p.youCrashed) { fail('玩家 15 秒内没死成，脚本驱动有问题'); break; }

  const P0 = p.stats.plan, F0 = p.stats.aiFrame, S0 = p.aiLive, ST0 = p.stats.stalls;
  console.log(`\n第 ${round} 局  玩家撞死在第 ${p.youLive} 根`
    + `  |  此刻 plan=${P0} aiFrame=${F0} aiScore=${S0} stalls=${ST0}`);

  // ---- 玩家死后：盯着 plan 还长不长 ----
  let grows = 0, lastPlan = P0, last = p;
  const deadline = Date.now() + BUDGET_S * 1000;
  while (Date.now() < deadline) {
    await sleep(500);
    last = await probe();
    if (last.stats.plan > lastPlan) { grows++; lastPlan = last.stats.plan; }
    if (last.aiCrashed || last.aiLive >= TARGET) break;
  }

  const dPlan = last.stats.plan - P0;
  const dFrame = last.stats.aiFrame - P0;   // 注意是越过 P0，不是越过 F0
  const dStall = last.stats.stalls - ST0;
  console.log(`  观察结束  plan=${last.stats.plan}(+${dPlan}, 涨了 ${grows} 次)`
    + `  aiFrame=${last.stats.aiFrame}(越过 P0 ${dFrame} 帧)`
    + `  aiScore=${last.aiLive}  stalls +${dStall}`);
  if (last.aiCrashed) console.log(`  AI 自己撞了：${JSON.stringify(last.aiVeilText)}`);

  // 判据 1、2：与机器速度无关，任何时候不成立都是真失败，不换局。
  if (grows < 3 || dPlan <= 0) {
    fail(`玩家死后 plan 不再增长（P0=${P0} -> ${last.stats.plan}，涨了 ${grows} 次）`
      + ' —— 当前局的生产被掐断了');
  }
  if (dFrame < 60) {
    fail(`aiFrame 没能越过玩家死亡时的存货（P0=${P0}，只到 ${last.stats.aiFrame}）`
      + ' —— AI 卡在缓冲区末尾');
  }
  if (failed) break;

  // 判据 3：AI 没到 target 就自己死了 = 抽到早死的 seed，这局无效，换一局。
  if (last.aiLive >= TARGET) {
    verdict = { round, P0, plan: last.stats.plan, aiFrame: last.stats.aiFrame,
      aiScore: last.aiLive, youScore: p.youLive, crashed: last.aiCrashed };
  } else if (last.aiCrashed) {
    console.log(`  这一局 AI 只飞了 ${last.aiLive} 根（< target ${TARGET}）——`
      + '重尾分布里的早死局，判为无效，换一局重来');
    await tap();          // youState === 'dead'，这一下点击就是开新的一局
    await sleep(1500);
  } else {
    fail(`${BUDGET_S} 秒内 AI 既没飞到 ${TARGET} 根也没撞死（当前 ${last.aiLive} 根）`
      + ' —— 页面可能整体停住了');
  }
}

const errs = (await probe().catch(() => ({ errors: [] }))).errors || [];
for (const e of errs) { console.log('  JS 报错: ' + e); }
if (errs.length) failed = true;
if (!verdict && !failed) fail(`${MAX_ROUNDS} 局都抽到早死的 seed，没能验到 ${TARGET} 根`);

ws.close();
proc.kill();
if (verdict) {
  console.log(`\n有效局：第 ${verdict.round} 局，玩家死在第 ${verdict.youScore} 根，`
    + `AI 继续飞到第 ${verdict.aiScore} 根`
    + `（plan ${verdict.P0} -> ${verdict.plan}，aiFrame ${verdict.aiFrame}）`);
}
console.log(failed ? '\nSTALL CHECK FAILED' : '\nSTALL CHECK OK  玩家死后计划继续生产，AI 没有卡住');
process.exit(failed ? 1 : 0);
