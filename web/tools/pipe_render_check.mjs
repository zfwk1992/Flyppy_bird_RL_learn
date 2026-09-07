/**
 * 管道渲染复现脚本（web_plan.md §2 / 批次 B2）。
 *
 * 用户报的现象："玩了几次后有一些 pipe 没有正常显示。"
 * 这句话有两种完全不同的成因，**修之前必须先分开**：
 *   (甲) `game.js` 的状态里就少了一根 —— 生成/回收的逻辑问题；
 *   (乙) 状态里有、画面上没有 —— `render.js` 或精灵解码的问题。
 * 这个脚本的任务只有一个：给出"是甲、是乙、还是复现不出来"。
 * **它不改任何渲染代码，也不该被改成去掩盖问题。**
 *
 * 两段检查
 * --------
 * 【A 段·确定性对拍】在页面上下文里 import 真正在跑的 `game.js` / `render.js`，
 *   用离屏画布把每一帧画出来，然后**逐根管道**拿 `observeState()` 和画布像素
 *   对拍（判据见 `pipe_probe.js` 的头注释）。固定 seed、固定动作，
 *   同一条命令重跑结果一样；不依赖机器快慢，也不依赖 worker。
 *   每局跑到小鸟撞死为止，**每一帧都判**，不是抽帧。
 *
 * 【B 段·真页面连续性】A 段跑的是离屏画布，证明不了 index.html 那两块画布上
 *   发生的事。但页面没有暴露 `you` / `aiGame`（模块作用域里的 let），
 *   外部读不到状态。所以 B 段换一个**不需要状态**的判据：
 *
 *       管道每帧恒定左移 5px（pipeVelX = -5）。所以任何一根还在屏幕中间的
 *       竖条，下一次采样时必须仍然在，且位置正好挪了 5 x Δ帧。
 *       "有些管道没显示"这件事一旦发生，必然表现为某根竖条在半路凭空消失。
 *
 *   AI 那块画布的 Δ帧能从 `window.__stats().aiFrame` 精确读出来；玩家那块
 *   在遮罩收起（youState==='playing'）期间和 AI 逐帧同步（两者都在 `stepBoth()`
 *   里走），所以用同一个 Δ。
 *
 * 负向对照（这是这个脚本可信的前提）
 * ----------------------------------
 * A 段最后会额外跑两局**故意画错**的：一局漏画中间那根管道、一局把它挪偏 30px
 * （只动传给 `Renderer.draw()` 的快照，游戏状态不动）。这两局必须被判成红，
 * 否则"十局全绿"只说明脚本没在干活，直接退 1。
 *
 * 用法（需要先起本地服务）：
 *     python3 -m http.server 8123 --directory web &
 *     node web/tools/pipe_render_check.mjs [url] [--rounds 10] [--frames 3000]
 *                                          [--seed 20260907] [--live 3] [--dump 目录]
 *     --live N   B 段观察 N 局真页面（默认 2；给 0 就只跑 A 段）
 *     --dump 目录 把第一处对不上的现场（状态 JSON + PNG）写出来
 *     环境变量 PIPE_DEBUG=1 会把 B 段每一处可疑点的前后帧现场也打出来
 *     （前后两帧各自的竖条区间、aiFrame、两边分数），查假阳性时很有用。
 *
 * 退出码：0 = 全过（含负向对照确实报红）；1 = 发现问题或负向对照没报红；
 *         2 = 环境起不来（浏览器 / 本地服务）。
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
const str = (name, dflt) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const url = argv.find((a) => !a.startsWith('--') && !/^\d+$/.test(a))
  || 'http://localhost:8123/index.html';
const ROUNDS = flag('rounds', 10);
const FRAMES = flag('frames', 3000);
const SEED0 = flag('seed', 20260907);
const LIVE_ROUNDS = flag('live', 2);
const DUMP_DIR = str('dump', null);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 云端 routine 的容器把 Chromium 放在 /opt/pw-browsers 下，版本号会变，
// 所以那一条按目录名扫，不要写死版本（写死的话镜像一升级就静默找不到浏览器）。
function findChrome() {
  const fixed = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const p of fixed) { try { if (fs.existsSync(p)) return p; } catch { /* 忽略 */ } }
  try {
    for (const d of fs.readdirSync('/opt/pw-browsers').filter((x) => x.startsWith('chromium')).sort().reverse()) {
      const p = path.join('/opt/pw-browsers', d, 'chrome-linux', 'chrome');
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
  // **必须把 body 读掉**：只看 r.ok 而不消费响应体的话，undici 会让这条
  // socket 停在 paused 状态；python http.server 用 HTTP/1.0、响应完就关连接，
  // 于是命中 Node 内部的 assert(!this.paused)，整个进程带着一段看不懂的
  // undici 栈崩掉 —— 表现为"检查挂了"，但和被测页面毫无关系。
  await r.arrayBuffer();
} catch (e) {
  console.error(`打不开 ${url}（${e.message}）。先起服务：`);
  console.error('    python3 -m http.server 8123 --directory web &');
  process.exit(2);
}

const PORT = 9900 + Math.floor(Math.random() * 300);
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`,
  // 每次跑用独立的 profile 目录：同一个目录被上一次没退干净的实例占着的话，
  // Chromium 会静默地不起 CDP，表现成"浏览器起不来"，很难查。
  '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-pipe-check-' + process.pid),
  'about:blank',
], { stdio: 'ignore' });

let target = null;
for (let i = 0; i < 60 && !target; i++) {
  await sleep(250);
  try {
    target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === 'page');
  } catch { /* 浏览器还没起来 */ }
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

const bail = async (code, msg) => {
  if (msg) console.error(msg);
  try { ws.close(); } catch { /* 已经关了 */ }
  proc.kill();
  process.exit(code);
};

/** 在页面里跑一段表达式并把返回值取回来；页面抛异常就原样报出来，不要吞掉。 */
async function evaluate(expression) {
  const r = await send('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true, timeout: 600000,
  });
  if (r?.exceptionDetails) {
    throw new Error('页面里抛异常: ' + (r.exceptionDetails.exception?.description
      || r.exceptionDetails.text));
  }
  return r?.result?.value;
}

await send('Page.enable');
await send('Runtime.enable');
await send('Page.addScriptToEvaluateOnNewDocument', {
  source: 'window.__pageErrors=[];addEventListener("error",e=>__pageErrors.push(String(e.message)));'
    + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));',
});
await send('Page.navigate', { url });
await sleep(2500);

let failed = false;
const fail = (m) => { console.log('FAIL  ' + m); failed = true; };

// 探针模块挂到 window 上，后面每次调用都走它（import 一次，状态复用）
const diag = await evaluate(`(async () => {
  const P = await import('/tools/pipe_probe.js');
  window.__pipeProbe = P;
  return await P.install();
})()`).catch((e) => { console.error(String(e.message)); return null; });
if (!diag) await bail(2, '探针装不上，A 段无从谈起');

console.log('探针就绪：');
console.log(`  精灵尺寸  管道 ${diag.spriteSize.pipe.join('x')}  地面 ${diag.spriteSize.base.join('x')}`);
console.log(`  调色板    背景 ${diag.paletteSizes.bg} 色 / 管道 ${diag.paletteSizes.pipe} 色 / `
  + `地面 ${diag.paletteSizes.base} 色 / 小鸟 ${diag.paletteSizes.player} 色`
  + `（管道里与别类重名的 ${diag.pipeAmbiguous} 色，已排除）`);
console.log(`  期望掩码  上管道 ${diag.maskUpperPx ?? diag.maskPixels.upper} px / `
  + `下管道 ${diag.maskPixels.lower} px（腐蚀一圈后）；地面线取 y<${diag.yLimit}`);

// 分类器要是把管道色和别的搅在一起，后面所有覆盖率都是废数，直接判环境失败
if (diag.paletteSizes.pipe - diag.pipeAmbiguous < 4) {
  await bail(2, '管道调色板几乎全被判成歧义色，颜色分类器不可用');
}
if (diag.maskPixels.upper < 5000 || diag.maskPixels.lower < 5000) {
  await bail(2, '管道期望掩码太小，说明精灵没解码出来或颜色分类错了');
}

// ---------------------------------------------------------------------------
// A 段：确定性逐帧对拍
// ---------------------------------------------------------------------------
console.log(`\n=== A 段：${ROUNDS} 局确定性对拍（seed ${SEED0}..${SEED0 + ROUNDS - 1}，每局最多 ${FRAMES} 帧，逐帧判） ===`);
let totFrames = 0, totJudged = 0, totSkipped = 0, worstAll = null;
const seeds = [];
for (let i = 0; i < ROUNDS; i++) {
  const seed = SEED0 + i;
  const r = await evaluate(`(() => JSON.stringify(window.__pipeProbe.runRound({
    seed: ${seed}, gapRange: [100.0, 165.0], maxFrames: ${FRAMES} })))()`).then(JSON.parse);
  seeds.push(r);
  totFrames += r.frames; totJudged += r.judged; totSkipped += r.skipped;
  if (!worstAll || (r.worstCov && r.worstCov.cov < worstAll.cov)) {
    if (r.worstCov) worstAll = { ...r.worstCov, seed };
  }
  const w = r.worstCov ? r.worstCov.cov.toFixed(4) : 'n/a';
  console.log(`${r.ok ? ' ok ' : 'FAIL'}  seed=${seed}  ${String(r.frames).padStart(4)} 帧`
    + `  分数 ${String(r.score).padStart(3)}  判了 ${String(r.judged).padStart(5)} 根次`
    + `（跳过 ${r.skipped} 根次：刚冒头/只剩一条边）  最低覆盖率 ${w}`
    + `  幽灵帧 ${r.ghostFrames}`);
  for (const b of r.bad.slice(0, 4)) {
    console.log(`        第 ${b.frame} 帧 第 ${b.i} 根 ${b.kind} x=${b.x.toFixed(1)} `
      + `覆盖率 ${b.cov.toFixed(3)}（期望 ${b.exp} px，命中 ${b.hit} px）`);
  }
  if (!r.ok) failed = true;
}
console.log(`A 段合计：${totFrames} 帧，判了 ${totJudged} 根次（另有 ${totSkipped} 根次太小没判）`);
if (worstAll) {
  console.log(`全局最低覆盖率 ${worstAll.cov.toFixed(4)}`
    + `（seed=${worstAll.seed} 第 ${worstAll.frame} 帧 第 ${worstAll.i} 根 ${worstAll.kind}）`);
}
if (totJudged < 1000) fail(`只判了 ${totJudged} 根次，样本太少，这一轮的结论不作数`);

if (DUMP_DIR) {
  const d = await evaluate('(() => JSON.stringify(window.__pipeProbe.takeDump()))()').then(JSON.parse);
  if (d) {
    fs.mkdirSync(DUMP_DIR, { recursive: true });
    fs.writeFileSync(path.join(DUMP_DIR, `mismatch-${d.seed}-${d.frame}.json`),
      JSON.stringify({ seed: d.seed, frame: d.frame, state: d.state, rows: d.rows, ghost: d.ghost }, null, 2));
    fs.writeFileSync(path.join(DUMP_DIR, `mismatch-${d.seed}-${d.frame}.png`),
      Buffer.from(d.png.split(',')[1], 'base64'));
    console.log(`现场已写到 ${DUMP_DIR}/mismatch-${d.seed}-${d.frame}.{json,png}`);
  }
}

// ---- 负向对照：故意画错，脚本必须当场报红 ----
console.log('\n--- 负向对照（故意画错，必须报红；否则上面的"全过"没有意义）---');
for (const sab of [
  { mode: 'drop', label: '漏画中间那根管道' },
  { mode: 'shift', label: '把中间那根管道挪偏 30px' },
]) {
  const r = await evaluate(`(() => JSON.stringify(window.__pipeProbe.runRound({
    seed: ${SEED0}, gapRange: [100.0, 165.0], maxFrames: 400,
    sabotage: { mode: '${sab.mode}', from: 120, to: 400, index: 0 } })))()`).then(JSON.parse);
  const caught = !r.ok;
  console.log(`${caught ? ' ok ' : 'FAIL'}  ${sab.label}：`
    + `${caught ? '报红了' : '**没报红**'}（对不上 ${r.bad.length} 处，幽灵帧 ${r.ghostFrames}`
    + `，最低覆盖率 ${r.worstCov ? r.worstCov.cov.toFixed(3) : 'n/a'}）`);
  if (!caught) fail(`负向对照"${sab.label}"没被抓到 —— A 段的绿是假绿`);
}
await evaluate('(() => { window.__pipeProbe.takeDump(); return 1; })()');

// ---------------------------------------------------------------------------
// B 段：真页面上的连续性
// ---------------------------------------------------------------------------
const liveIssues = [];
let liveStats = null;
if (LIVE_ROUNDS > 0) {
  console.log(`\n=== B 段：真页面 ${LIVE_ROUNDS} 局，看两块画布上的管道会不会半路消失 ===`);
  const probeState = `(() => { const s = window.__stats && window.__stats();
    const v = document.getElementById('youVeil'); const av = document.getElementById('aiVeil');
    return JSON.stringify({ ready: !!s && s.ready, worker: !!s && s.worker,
      aiFrame: s ? s.aiFrame : -1, playing: !!v && v.hidden,
      aiCrashed: !!av && !av.hidden,
      youScore: Number(document.getElementById('youLive').textContent || 0),
      aiScore: Number(document.getElementById('aiLive').textContent || 0),
      errors: window.__pageErrors || [] }); })()`;
  const st = async () => JSON.parse(await evaluate(probeState));
  const tap = () => evaluate(`dispatchEvent(new KeyboardEvent('keydown',{code:'Space',bubbles:true}))`);

  let boot = null;
  for (let i = 0; i < 120; i++) { await sleep(500); boot = await st(); if (boot.ready) break; }
  if (!boot?.ready) {
    fail('60 秒内页面没就绪，B 段没跑成');
  } else {
    console.log(`页面就绪：worker=${boot.worker ? 'on' : 'off(退化路径)'}`);
    // startLive(true) 会顺带按像素驾驶玩家那只鸟（见 pipe_probe.js 的
    // decideFlap）—— 不这么做玩家每局 0~1 分就摔了，玩家那块画布等于没测到。
    await evaluate('(() => { window.__pipeProbe.startLive(true); return 1; })()');
    const samples = [];
    for (let round = 1; round <= LIVE_ROUNDS; round++) {
      await tap();
      const deadline = Date.now() + 70000;
      let cur = await st();
      let nextDrain = 0;
      while (Date.now() < deadline) {
        cur = await st();
        if (Date.now() >= nextDrain) {
          samples.push(...JSON.parse(await evaluate('(() => JSON.stringify(window.__pipeProbe.drainLive()))()')));
          nextDrain = Date.now() + 1500;
        }
        if (cur.aiCrashed) break;
        await sleep(300);
      }
      samples.push(...JSON.parse(await evaluate('(() => JSON.stringify(window.__pipeProbe.drainLive()))()')));
      console.log(`  第 ${round} 局：玩家 ${cur.youScore} 根 / AI ${cur.aiScore} 根`
        + `${cur.aiCrashed ? '（AI 自己撞了）' : '（到时了）'}  累计采样 ${samples.length} 帧`);
      if (round < LIVE_ROUNDS) { await tap(); await sleep(1200); }
    }
    await evaluate('(() => { window.__pipeProbe.stopLive(); return 1; })()');

    // --- 连续性判据 ---
    // 管道每帧固定左移 5px。所以上一次采样里右边缘还在屏幕里的竖条，
    // 这一次必须还能在 x - 5*Δ 附近找到；找不到 = 它在半路凭空消失了。
    const TOL = 8;             // ±1 帧（5px）+ 边缘检测抖动
    let checked = 0;
    const analyse = (key, prev, cur, dFrames) => {
      const shift = 5 * dFrames;
      for (const [x0, x1] of prev[key]) {
        const px1 = x1 - shift;
        if (px1 < 10) continue;                 // 本来就该滑出左边了，不算
        if (x1 >= 287) continue;                // 右边缘还在进场，形状会变
        checked++;
        const hit = cur[key].some(([, b1]) => Math.abs(b1 - px1) <= TOL);
        if (!hit && liveIssues.length < 20) {
          liveIssues.push({ key, t: cur.t, dFrames, was: [x0, x1],
            expect: px1, got: cur[key].map((r) => r[1]),
            prevBars: prev[key], curBars: cur[key],
            aiFrame: [prev.aiFrame, cur.aiFrame], playing: [prev.playing, cur.playing],
            scores: [prev.youScore, prev.aiScore, cur.youScore, cur.aiScore] });
        }
      }
    };
    for (let i = 1; i < samples.length; i++) {
      const a = samples[i - 1], b = samples[i];
      const d = b.aiFrame - a.aiFrame;
      if (d < 0 || d > 12) continue;            // 跨局重开 / 长时间停顿，不判
      if (b.aiScore < a.aiScore) continue;      // 分数回退 = 换局了
      analyse('ai', a, b, d);
      // 玩家那块只在"玩家在玩 && AI 还活着"时判 —— 只有这时两块画布才逐帧同步
      // （AI 一死 aiFrame 就不动了，而 you 还在走，拿 d 去推它的位置必然算错）
      if (a.playing && b.playing && !a.aiDead && !b.aiDead) analyse('you', a, b, d);
    }
    liveStats = { samples: samples.length, checked, issues: liveIssues.length };
    console.log(`  连续性判定：采样 ${samples.length} 帧，判了 ${checked} 根次，`
      + `半路消失 ${liveIssues.length} 次`);
    for (const q of liveIssues.slice(0, 6)) {
      console.log(`        ${q.key} 画布：竖条右边缘 ${q.was[1]} 过了 ${q.dFrames} 帧后`
        + `应在 ${q.expect}，实际只有 [${q.got.join(', ')}]`);
      if (process.env.PIPE_DEBUG) {
        console.log(`            aiFrame ${q.aiFrame.join('->')}  playing ${q.playing.join('->')}`
          + `  分数 you/ai ${q.scores[0]}/${q.scores[1]} -> ${q.scores[2]}/${q.scores[3]}`
          + `  竖条 ${JSON.stringify(q.prevBars)} -> ${JSON.stringify(q.curBars)}`);
      }
    }
    if (checked < 200) fail(`B 段只判了 ${checked} 根次，样本太少，这一段的结论不作数`);
    if (liveIssues.length) fail(`真页面上有 ${liveIssues.length} 次管道半路消失 —— 复现到了`);
  }
}

const errs = (await evaluate('JSON.stringify(window.__pageErrors||[])').then(JSON.parse).catch(() => []));
for (const e of errs) console.log('  页面 JS 报错: ' + e);
if (errs.length) fail('页面有 JS 报错');

// ---------------------------------------------------------------------------
console.log('\n=== 结论 ===');
if (failed) {
  console.log('复现到了问题（或负向对照失效）。按上面的现场分两种情况处理：');
  console.log('  · A 段覆盖率低 -> 状态里有、画面上没有 -> 渲染侧（render.js / 精灵解码）');
  console.log('  · A 段全绿但 B 段有竖条消失 -> 问题不在 game.js+render.js 这一对，');
  console.log('    而在 index.html 的调度（换局、帧推进）或浏览器合成这一层');
} else {
  console.log(`未能复现：A 段 ${totFrames} 帧 / ${totJudged} 根次逐帧对拍，`
    + `状态与画布完全一致（最低覆盖率 ${worstAll ? worstAll.cov.toFixed(4) : 'n/a'}）；`);
  if (liveStats) {
    console.log(`B 段真页面 ${liveStats.samples} 帧采样、${liveStats.checked} 根次连续性判定，`
      + '没有竖条半路消失。');
  }
  console.log('负向对照两种画错法都被当场抓住，所以这个"没复现"不是脚本没干活。');
}

ws.close();
proc.kill();
console.log(failed ? '\nPIPE RENDER CHECK FAILED' : '\nPIPE RENDER CHECK OK');
process.exit(failed ? 1 : 0);
