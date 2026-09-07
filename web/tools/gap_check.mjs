/**
 * demo 难度一致性核查：**you / aiGame / worker 三处的 gapRange 必须完全相同。**
 *
 * 这条检查钉的是哪个 bug
 * ----------------------
 * demo 把缝隙下界从 85 抬到了 100（web_plan.md §8），做法是在 index.html 里
 * 定义 `DEMO_GAP_RANGE` 并传给三处：玩家那一局 `you`、主线程显示用的 `aiGame`、
 * 以及 worker 里真正跑推理的那一局（通过 `start` 消息传，不能在 worker 里写死）。
 *
 * 人机对战的**全部前提**是"同一个 seed 生成同一串管道"。三处里只要有一处
 * gapRange 不同，同一个 seed 就长出不同的管道序列，左右两块画面不再是同一关。
 * 而这种 bug 在画面上极其隐蔽 —— **两边都能正常玩、都不报错**，只是管道
 * 长得不一样，整个对照悄悄失去意义。这正是这个项目里最难查的那一类 bug，
 * 所以钉成一条可执行的检查。
 *
 * 五条判据（全过才退出 0）
 * ------------------------
 *   1. `you.gapRange`、`aiGame.gapRange`、worker 收到的 `start.gapRange`、
 *      以及常量本身，四者**逐值相等**且等于 [100, 165]。
 *      读的是实例上的 `this.gapRange`（`window.__gapInfo()`），不是常量 ——
 *      "常量改对了但某一处构造忘了传"也要能抓到。
 *   2. **源码层**：index.html 里每一处发 `start` 的 `postMessage` 都带 gapRange。
 *      为什么还要看源码：`endRound()` 里那处预取的 `start` 只有 AI 撞死才发，
 *      跑一趟 headless 未必碰得到，运行时探针会漏掉它。
 *   3. 同一 seed 下，用三处**各自实测到的** gapRange 各构造一局，各自用真的
 *      `AiPlayer` 驱动，逐帧比对 `observeState()` 里的管道 x/y/gap，
 *      600 帧全等（或三局在同一帧同时结束）。
 *      注意"各自"两个字：三局都喂同一份配置的话这条就退化成恒真，什么也测不到。
 *   4. 采样 ≥ 200 根管道：gap 最小值 ≥ 100，最大值 ≤ 165。
 *   5. **负向对照**：把 gapRange 换成 [85, 165] 重跑 3 和 4，
 *      管道序列必须**不同**、gap 最小值必须 < 100。
 *      没有这一条，前面几条可能只是"比了个寂寞"。
 *
 * 四项 parity 不受影响：它们用 `game.js` 的 `DEFAULT_GAP_RANGE = [85,165]`
 * 构造，而覆盖只发生在 demo 页面这一侧。**不要去改那个默认值** ——
 * 重新生成参考文件需要 torch + pygame，云端 routine 没有，会把自己卡死。
 *
 * 用法（需要先起本地服务）：
 *     python3 -m http.server 8123 --directory web &
 *     node web/tools/gap_check.mjs [url] [--frames 600] [--pipes 200]
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
const FRAMES = flag('frames', 600);   // 判据 3 逐帧比对多少帧
const MIN_PIPES = flag('pipes', 200); // 判据 4 至少采样多少根管道

// demo 应当生效的范围。改这里之前先改 index.html 的 DEMO_GAP_RANGE。
const EXPECT = [100, 165];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 常见的 Chromium 位置。云端 routine 容器把浏览器放在 /opt/pw-browsers 下，
// 版本号会变，所以那一条按目录名扫，不要写死版本。
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

let html = null;
try {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  html = await r.text();
} catch (e) {
  console.error(`打不开 ${url}（${e.message}）。先起服务：`);
  console.error('    python3 -m http.server 8123 --directory web &');
  process.exit(2);
}

let failed = false;
const fail = (msg) => { console.log('FAIL  ' + msg); failed = true; };
const ok = (msg) => console.log(' ok   ' + msg);

/* ---- 判据 2：源码层，每一处 start 都带 gapRange ------------------------
   运行时探针只看得到实际发出去的消息。`endRound()` 里那处预取的 start
   要 AI 撞死才发，headless 跑一趟未必碰得到 —— 漏掉它，锁步照样会坏。
   所以这里直接在源码里把所有 postMessage 调用抠出来数一遍。 */
{
  const calls = html.match(/postMessage\(\{[^}]*\}\)/g) || [];
  const starts = calls.filter((c) => /'start'/.test(c));
  if (!starts.length) {
    fail('index.html 里一处发 start 的 postMessage 都没找到 —— 正则该更新了');
  } else {
    const bad = starts.filter((c) => !/gapRange/.test(c));
    if (bad.length) {
      fail(`index.html 有 ${bad.length}/${starts.length} 处 start 没带 gapRange：\n      `
        + bad.join('\n      '));
    } else {
      ok(`源码：${starts.length} 处发 start 的 postMessage 都带了 gapRange`);
    }
  }
}

const PORT = 9900 + Math.floor(Math.random() * 300);
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`,
  '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-gap-check'),
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
const evaluate = async (expr, timeout = 300000) => (await send('Runtime.evaluate',
  { expression: expr, returnByValue: true, awaitPromise: true, timeout }))?.result?.value;

await send('Page.enable');
await send('Runtime.enable');

/* 探针：在页面脚本跑起来之前劫持 Worker.prototype.postMessage，
   把每一条 start 消息里的 gapRange 记下来。这是唯一能看到"worker 那一侧
   到底收到了什么"的办法 —— worker 内部的 game 实例不对外暴露。 */
await send('Page.addScriptToEvaluateOnNewDocument', {
  source: `
    window.__pageErrors = [];
    addEventListener('error', (e) => __pageErrors.push(String(e.message)));
    addEventListener('unhandledrejection', (e) => __pageErrors.push('unhandled: ' + e.reason));
    window.__workerStarts = [];
    const __origPost = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (msg, ...rest) {
      try {
        if (msg && msg.type === 'start') {
          __workerStarts.push({ seed: msg.seed, gapRange: msg.gapRange ? Array.from(msg.gapRange) : null });
        }
      } catch { /* 记录失败不能影响页面 */ }
      return __origPost.call(this, msg, ...rest);
    };
  `,
});
await send('Page.navigate', { url });

// 等权重（2.5 MB）加载完
let boot = null;
for (let i = 0; i < 120; i++) {
  await sleep(500);
  boot = await evaluate('JSON.stringify({ s: (window.__stats && __stats()) || null,'
    + ' g: (window.__gapInfo && __gapInfo()) || null, w: window.__workerStarts || [],'
    + ' e: window.__pageErrors || [] })').then((v) => (v ? JSON.parse(v) : null)).catch(() => null);
  if (boot?.s?.ready) break;
}
if (!boot?.s?.ready) {
  console.log('FAIL  60 秒内页面没就绪（__stats().ready 一直是 false）');
  ws.close(); proc.kill(); process.exit(1);
}
if (!boot.s.worker) {
  // 退化路径根本不发 start，第 1 条判据无从谈起。静默跳过等于以后没人知道。
  console.log('FAIL  worker 没起来，页面走的是主线程退化路径，这条检查测不了');
  ws.close(); proc.kill(); process.exit(1);
}
if (!boot.g) {
  console.log('FAIL  页面没有 window.__gapInfo()，index.html 里的调试钩子被删了？');
  ws.close(); proc.kill(); process.exit(1);
}

/* ---- 判据 1：四者逐值相等 ---------------------------------------------- */
const same = (a, b) => Array.isArray(a) && Array.isArray(b)
  && a.length === b.length && a.every((v, i) => v === b[i]);
const fmt = (a) => (Array.isArray(a) ? `[${a.join(', ')}]` : String(a));

const workerStarts = boot.w;
if (!workerStarts.length) {
  fail('没截到任何发给 worker 的 start 消息 —— 探针失效了，后面的结论都不可信');
}
const sources = [
  ['常量 DEMO_GAP_RANGE', boot.g.constant],
  ['you.gapRange', boot.g.you],
  ['aiGame.gapRange', boot.g.ai],
  ...workerStarts.map((w, i) => [`worker start#${i + 1}(seed=${w.seed})`, w.gapRange]),
];
for (const [name, val] of sources) {
  if (!same(val, EXPECT)) fail(`${name} = ${fmt(val)}，期望 ${fmt(EXPECT)}`);
}
if (!failed) ok(`四方一致：${sources.map((s) => s[0].split('(')[0]).join(' = ')} = ${fmt(EXPECT)}`);

// 判据 3/4 喂进去的是**三个来源各自实测到的**那一份，而不是 EXPECT。
// 这一点很要紧：如果三处都填 you 那一份，判据 3 就退化成"同一份配置跑三遍"
// —— 恒真，抓不到任何东西。分别喂各自的值，锁步比对才真的在比三处是否一致。
// worker 那份没截到（探针失效）时用 null 占位，判据 1 已经先报过失败了。
const gYou = boot.g.you || EXPECT;
const gAi = boot.g.ai || EXPECT;
const gWorker = (workerStarts[0] && workerStarts[0].gapRange) || null;

/* ---- 判据 3/4/5：全在页面里跑，因为要用页面已经缓存好的权重 ------------ */
const EXPR = `(async () => {
  const FRAMES = ${FRAMES}, MIN_PIPES = ${MIN_PIPES};
  // 三处各自实测到的 gapRange，外加负向对照用的默认下界。
  // G_WORKER 为 null 表示 start 消息里压根没带 —— worker 会静默落回
  // game.js 的 DEFAULT_GAP_RANGE，所以这里也用同一份默认值来复现它的行为。
  const G_YOU = ${JSON.stringify(gYou)}, G_AI = ${JSON.stringify(gAi)};
  const G_WORKER = ${JSON.stringify(gWorker)}, CTRL = [85.0, 165.0];
  const [{ FlappyGame }, { AiPlayer }, { parseWeights }, { HITMASKS }, { WEIGHTS_META }] =
    await Promise.all([import('./game.js'), import('./ai.js'), import('./nn.js'),
                       import('./assets/hitmasks.js'), import('./model/weights-meta.js')]);
  const buf = await (await fetch('./model/' + WEIGHTS_META.file)).arrayBuffer();
  const SEED = 20260907;

  // 一局：用真的 AiPlayer 驱动，逐帧记下管道的 x/y/gap（就是 stateForRender
  // 画出来的那份）。三处 gapRange 相同的话，这串东西必须逐位相同。
  const rollout = (gapRange) => {
    const g = new FlappyGame({ seed: SEED, gapRange, hitmasks: HITMASKS });
    g.reset();
    const p = new AiPlayer(parseWeights(buf.slice(0)));
    p.reset();
    const frames = [];
    for (let f = 0; f < FRAMES; f++) {
      const st = g.observeState();
      frames.push(st.upperPipes.map((u, i) =>
        [u.x, u.y, u.gap, st.lowerPipes[i].y].join(',')).join('|'));
      if (g.step(p.decide(g)).done) break;   // 撞死就停，帧数本身也是对照量
    }
    return { frames, score: g.score };
  };

  const A = rollout(G_YOU);                       // 玩家那一局
  const B = rollout(G_AI);                        // 主线程显示用的 aiGame
  const C = rollout(G_WORKER || CTRL);            // worker 里真正跑推理的那一局
  const N = rollout(CTRL);                        // 负向对照：默认的 85 下界

  const diffAt = (x, y) => {
    if (x.frames.length !== y.frames.length) return -2;   // 帧数就不同
    for (let i = 0; i < x.frames.length; i++) if (x.frames[i] !== y.frames[i]) return i;
    return -1;
  };

  // 采样管道分布。_samplePipe 是 step() 内部生成管道用的同一个方法，
  // 直接调它才能在不依赖小鸟活多久的前提下拿到足够多的样本。
  const sampleGaps = (gapRange) => {
    const gaps = [];
    for (let s = 0; gaps.length < MIN_PIPES; s++) {
      const g = new FlappyGame({ seed: 4000 + s, gapRange });
      g.reset();
      gaps.push(g.upperPipes[0].gap, g.upperPipes[1].gap);
      for (let i = 0; i < 24; i++) gaps.push(g._samplePipe(0)[0].gap);
    }
    return gaps;
  };
  const stat = (a) => ({ n: a.length, min: Math.min(...a), max: Math.max(...a),
                         mean: a.reduce((s, v) => s + v, 0) / a.length });

  return JSON.stringify({
    lockstep: { frames: A.frames.length, score: A.score,
                ab: diffAt(A, B), ac: diffAt(A, C),
                ctrlFrames: N.frames.length, ctrlScore: N.score, an: diffAt(A, N) },
    gaps: stat(sampleGaps(G_YOU)),
    ctrlGaps: stat(sampleGaps(CTRL)),
  });
})()`;

const raw = await evaluate(EXPR);
ws.close();
proc.kill();

if (raw === undefined || raw === null) {
  console.log('FAIL  页面内的比对没跑起来');
  process.exit(1);
}
const r = JSON.parse(raw);

/* ---- 判据 3 ---- */
{
  const { frames, score, ab, ac } = r.lockstep;
  if (ab !== -1 || ac !== -1) {
    // -1 = 这一对相同，别把它也印成"第 -1 帧不同"；只报真的对不上的那一对
    const why = (d) => (d === -2 ? '两局帧数就不同' : `第 ${d} 帧起管道不同`);
    const bad = [['you-vs-aiGame', ab], ['you-vs-worker', ac]]
      .filter(([, d]) => d !== -1).map(([n, d]) => `${n} ${why(d)}`);
    fail(`同一 seed 下三局的管道序列不一致：${bad.join('；')}`);
  } else if (frames < 60) {
    // 60 帧连一根管子都过不完，比了也说明不了什么
    fail(`锁步比对只跑了 ${frames} 帧就结束了，样本太短，结论不成立`);
  } else {
    ok(`锁步：三局逐帧管道全等，比对 ${frames} 帧（AI 分数 ${score}）`);
  }
}

/* ---- 判据 5（对照 A）：换成 85 下界必须看得出不同 ---- */
{
  const { an, ctrlFrames, ctrlScore } = r.lockstep;
  if (an === -1) {
    fail('负向对照没红：gapRange 换成 [85,165] 之后管道序列竟然完全一样 —— '
      + '说明这条比对根本没在比 gapRange，判据 3 不算数');
  } else {
    const why = an === -2 ? `帧数不同（${r.lockstep.frames} vs ${ctrlFrames}）` : `第 ${an} 帧起管道不同`;
    ok(`负向对照：[85,165] 与实测配置${why}（对照局 AI 分数 ${ctrlScore}）—— 比对确实在比 gapRange`);
  }
}

/* ---- 判据 4 ---- */
{
  const g = r.gaps;
  if (g.n < MIN_PIPES) fail(`只采到 ${g.n} 根管道，不足 ${MIN_PIPES}`);
  else if (g.min < EXPECT[0] - 1e-9) fail(`采样 ${g.n} 根，gap 最小值 ${g.min.toFixed(2)} < ${EXPECT[0]}`);
  else if (g.max > EXPECT[1] + 1e-9) fail(`采样 ${g.n} 根，gap 最大值 ${g.max.toFixed(2)} > ${EXPECT[1]}`);
  else ok(`分布：${g.n} 根管道，gap ∈ [${g.min.toFixed(2)}, ${g.max.toFixed(2)}]，均值 ${g.mean.toFixed(2)}`);
}

/* ---- 判据 5（对照 B）：同一个采样器在 85 下界上必须给出 < 100 的缝 ---- */
{
  const c = r.ctrlGaps;
  if (c.min >= EXPECT[0]) {
    fail(`负向对照没红：[85,165] 采样 ${c.n} 根，最小值 ${c.min.toFixed(2)} 也 ≥ ${EXPECT[0]} —— `
      + '采样器没在读 gapRange，判据 4 不算数');
  } else {
    ok(`负向对照：[85,165] 采样 ${c.n} 根，gap ∈ [${c.min.toFixed(2)}, ${c.max.toFixed(2)}]，`
      + `确实有 < ${EXPECT[0]} 的窄缝 —— 采样器有效`);
  }
}

const errs = (boot.e || []).filter((m) => m && !/favicon/i.test(m));
if (errs.length) fail('页面有 JS 报错：' + errs.slice(0, 3).join(' | '));

console.log('');
console.log(failed ? 'GAP CHECK FAILED' : 'GAP CHECK OK  三处 gapRange 一致，demo 缝隙下界已抬到 ' + EXPECT[0]);
process.exit(failed ? 1 : 0);
