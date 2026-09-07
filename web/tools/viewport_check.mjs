/**
 * 设备矩阵：7 个视口逐个跑，量布局，给退出码。（web_plan.md §3.1 / 批次 B4）
 *
 * 为什么不是 `chrome --headless --window-size=390,844 --screenshot`
 * ---------------------------------------------------------------
 * **`--window-size` 不是移动端视口。** 它设的是窗口尺寸：页面拿到的布局视口
 * 未必是 390，`<meta name=viewport content="width=device-width">` 不按手机的
 * 方式生效，`devicePixelRatio` 还是 1。这个坑 `mobile_check.mjs` 顶部记过一次
 * —— 用它截出来的图会把一个**完全正常**的响应式页面拍成"右边被裁掉"的样子。
 * 正确做法是 CDP 的 `Emulation.setDeviceMetricsOverride`，**必须**同时带
 * `mobile: true` 和真实的 `deviceScaleFactor`。
 *
 * 和 `mobile_check.mjs` 的关系
 * ---------------------------
 * 那个是本工具的前身，只有 3 个手机视口、只判横向溢出。这里按 §3.1 的表
 * 补齐 7 个视口（含平板竖/横屏和桌面），并把判据从 1 条加到 4 条。
 * 没有删掉 mobile_check —— 它更小、跑得更快，改 CSS 时手边先跑它更顺手。
 *
 * ⚠️ 横向溢出**不能**拿 `innerWidth` 当尺子（这一版踩到的坑）
 * ----------------------------------------------------------
 * §3.1 的原话是断言 `documentElement.scrollWidth <= innerWidth`。照着写出来的
 * 第一版**永远是绿的**：负向对照注入 `#app{width:1200px}` 之后实测
 * `scrollWidth=1200, innerWidth=1200` —— 因为 Blink 的移动端模拟会
 * **shrink-to-fit**，内容撑宽时它把页面整体缩小、布局视口跟着涨到 1200，
 * `innerWidth` 追着 `scrollWidth` 走，这条不等式于是恒成立。
 * 一条永远不会红的检查等于没有检查。
 *
 * 所以尺子换成**我们自己设给 `setDeviceMetricsOverride` 的那个宽度**
 * （320 / 390 / ...）：那才是真机的 CSS 像素宽度，不会被 shrink-to-fit 改。
 * 顺带把 shrink-to-fit 本身也当失败报出来 —— 真机上它的表现就是
 * "一进页面字全是小的、要双指放大"，那不是可接受的移动端布局。
 * （`mobile_check.mjs` 里那条老判据有同样的毛病，见 PROGRESS.md 的 B4。）
 *
 * 每个视口的判据（任一不成立 => 退出 1）
 * -------------------------------------
 *   1. **无横向溢出**：`documentElement.scrollWidth <= 设备宽度`，没有元素的
 *      right 越过设备宽度，且没有发生 shrink-to-fit（`innerWidth` 必须等于
 *      设备宽度、`visualViewport.scale` 必须是 1）。
 *   2. **首屏可见**：两块 canvas 的 `getBoundingClientRect().bottom <= 设备高度`；
 *      退一步，至少玩家那块 canvas + 操作提示可见（§3.1 原话是"或者至少
 *      '操作提示'可见"）。**两块都进不去、玩家那块也进不去，就是失败** ——
 *      一个不滚动就看不到自己那只鸟的首屏，等于没有首屏。
 *   3. **两块画布不重叠**：横屏 1024x768 那一行专门盯这个（§3.1 备注：
 *      "横屏时两个画布 + 讲解区会不会挤爆"）。栅格塌了会表现为两块叠在一起，
 *      而这**不会**让 scrollWidth 超标，判据 1 抓不到。
 *   4. **`window.__pageErrors` 为空**：布局对但脚本挂了，页面照样是坏的。
 *
 * 判据 2 为什么用"底边"而不是"元素可见"
 * ------------------------------------
 * `bottom <= innerHeight` 是"整块画布都在首屏里"。用 IntersectionObserver 之类的
 * "可见即可"会把只露出 3 个像素的画布判成过，那不是这条判据想保的东西。
 *
 * 关于真机：**这里跑的是 Blink 的设备模拟，不是 iOS 的 WebKit。**
 * `Emulation.setDeviceMetricsOverride` 只是让 Blink 按手机的规则布局，
 * 它证明不了 Safari 上的行为。真机结论只能进 `web/TESTING.md` 的人工清单，
 * 不许拿这个脚本的绿灯去写"iPhone 上测过了"。
 *
 * 用法：
 *     python3 -m http.server 8123 --directory web &
 *     node web/tools/viewport_check.mjs [url] [--shots <目录>] [--wait 20]
 *
 * 负向对照：`--inject <css>` / `--injectjs <js>` 在页面加载时注入一段
 * CSS/JS，**只为了把这四条判据分别打红**，用完即弃，不改 index.html。
 * 四条判据各有一条对照，全都必须退出 1：
 *     1 溢出   --inject '#app{max-width:none;width:1200px}'
 *     2 首屏   --inject 'canvas{height:900px!important}'
 *     3 重叠   --inject '#arena{display:block;position:relative;height:0}
 *                        .side{position:absolute;top:0;left:0;width:48%}'
 *     4 报错   --injectjs 'setTimeout(()=>{throw new Error("negative control")},300)'
 * 实测结果记在 PROGRESS.md 的 B4 那一条里。
 *
 * 注意有一类"看着像 bug"的改动**不该**红：把 `#arena` 改成单栏
 * （`grid-template-columns:1fr`）之后 AI 那块画布确实掉出首屏，但玩家那块和
 * 操作提示还在 —— §3.1 的判据 2 明确留了这条退路，所以它照样是绿的。
 * 试过，记在这里，免得下一轮再拿它当对照然后困惑半天。
 *
 * 零依赖：Node 内置 fetch + WebSocket 直连 CDP，不需要 Playwright。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const url = argv.find((a, i) => !a.startsWith('--') && !argv[i - 1]?.startsWith('--'))
  || 'http://localhost:8123/index.html';
const shotDir = opt('shots', null);
const WAIT_S = Number(opt('wait', 20));      // 等 __stats().ready 的上限（秒）
const inject = opt('inject', null);          // 负向对照用：注入一段 CSS
const injectJs = opt('injectjs', null);      // 负向对照用：注入一段 JS

// §3.1 的表，一行不多一行不少。dpr 按真机写（iPhone 13 是 3，SE 是 2）——
// dpr 不只影响截图清晰度，`image-rendering: pixelated` 在非整数缩放下的表现
// 也跟它有关，写 1 就等于没测到手机。
// `touch` 那一列不是摆设：**`mobile: true` 并不会让 `(pointer: coarse)` 成立。**
// 本轮实测，只设 mobile:true 时 coarse=false、fine 也=false（headless 里压根没有
// 指针）、maxTouchPoints=0；要再发一条 `Emulation.setTouchEmulationEnabled`
// 才变成 coarse=true / maxTouchPoints=5。B7 要按 `pointer: coarse` 分操作提示，
// 少了这条的话七个视口全走桌面分支，断言等于没写。
const VIEWPORTS = [
  { w: 320,  h: 568,  dpr: 2, touch: true,  name: '小屏手机 iPhone SE 一代' },
  { w: 390,  h: 844,  dpr: 3, touch: true,  name: '主流 iPhone' },
  { w: 414,  h: 896,  dpr: 2, touch: true,  name: '大屏 iPhone' },
  { w: 360,  h: 800,  dpr: 3, touch: true,  name: '主流 Android' },
  { w: 768,  h: 1024, dpr: 2, touch: true,  name: '平板竖屏' },
  { w: 1024, h: 768,  dpr: 2, touch: true,  name: '平板横屏（两画布 + 讲解区会不会挤爆）' },
  { w: 1440, h: 900,  dpr: 1, touch: false, name: '桌面' },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 云端 routine 容器把浏览器放在 /opt/pw-browsers 下，目录名带版本号且会变，
// 所以那一条按目录名扫，不要写死 —— 写死的话镜像一升级脚本就静默找不到浏览器。
function findChrome() {
  const fixed = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const p of fixed) { try { if (fs.existsSync(p)) return p; } catch { /* 忽略 */ } }
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
} catch (e) {
  console.error(`打不开 ${url}（${e.message}）。先起服务：`);
  console.error('    python3 -m http.server 8123 --directory web &');
  process.exit(2);
}

// user-data-dir 带 pid：固定名字的话，上一次没退干净的 Chromium 占着目录会让
// 新实例**静默不起 CDP**，表现成"浏览器起不来"。这个坑 B2 那轮踩过。
const PORT = 9300 + Math.floor(Math.random() * 300);
const profile = path.join(os.tmpdir(), `flappy-viewport-check-${process.pid}`);
const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`, '--user-data-dir=' + profile,
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
// 报错钩子要在**文档创建前**装，装晚了加载期抛的错就漏了。
await send('Page.addScriptToEvaluateOnNewDocument', {
  source: 'window.__pageErrors=[];addEventListener("error",e=>__pageErrors.push(String(e.message)));'
    + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));'
    + (inject ? `document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');`
        + `s.textContent=${JSON.stringify(inject)};document.head.appendChild(s);});` : '')
    + (injectJs ? `document.addEventListener('DOMContentLoaded',()=>{${injectJs}\n});` : ''),
});

/* 探针全部读页面**已有**的东西（DOM + `window.__stats()`），不往 index.html
   里加测试专用钩子 —— 量的就是访客真正看到的那一层。 */
const PROBE = (deviceW) => `(() => {
  const DW = ${deviceW};   // 设给 setDeviceMetricsOverride 的 CSS 宽度，见顶部注释
  const R = (sel) => { const el = document.querySelector(sel); if (!el) return null;
    const r = el.getBoundingClientRect();
    return { top: r.top, bottom: r.bottom, left: r.left, right: r.right,
             w: r.width, h: r.height, hidden: !!el.hidden }; };
  // 越界元素：列出来，报错才能直接指向要改的那条 CSS。用 right > DW + 0.5
  // 是因为亚像素布局下 right 常常是 390.0000001 这种，0.5 的余量避免假阳性。
  const over = [...document.querySelectorAll('body *')]
    .filter((el) => el.getBoundingClientRect().right > DW + 0.5)
    .slice(0, 8)
    .map((el) => {
      const r = el.getBoundingClientRect();
      const id = el.id ? '#' + el.id
        : el.tagName.toLowerCase() + (typeof el.className === 'string' && el.className.trim()
            ? '.' + el.className.trim().split(/\\s+/)[0] : '');
      return id + ' right=' + r.right.toFixed(0) + ' w=' + r.width.toFixed(0);
    });
  const s = (window.__stats && window.__stats()) || null;
  return JSON.stringify({
    innerWidth, innerHeight, dpr: devicePixelRatio,
    scale: (window.visualViewport && visualViewport.scale) || 1,
    scrollWidth: document.documentElement.scrollWidth,
    overflowing: over,
    you: R('#youCanvas'), ai: R('#aiCanvas'),
    veil: R('#youVeil'), hint: R('footer > span'),
    coarse: matchMedia('(pointer: coarse)').matches,
    touchPoints: navigator.maxTouchPoints,
    ready: !!(s && s.ready), worker: !!(s && s.worker),
    errors: window.__pageErrors || [],
  });
})()`;

const F = (x) => (x == null ? '?' : x.toFixed(0));
let failed = false;
const rows = [];

for (const vp of VIEWPORTS) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: vp.w, height: vp.h, deviceScaleFactor: vp.dpr, mobile: true,
  });
  await send('Emulation.setTouchEmulationEnabled',
    { enabled: !!vp.touch, maxTouchPoints: vp.touch ? 5 : 1 });
  // 每个视口都重新导航：布局是在加载时算的，而且换一次视口就换一次干净的
  // __pageErrors —— 否则上一个视口的报错会挂在后面每一行上，看不出是谁的。
  await send('Page.navigate', { url });

  // 等页面就绪。**不是**在等布局（布局早就摆好了），是在等 2.5 MB 权重下完：
  // 加载期间玩家那侧的遮罩显示的是进度条，不是操作提示，判据 2 的退路会误判。
  let r = null;
  for (let i = 0; i < WAIT_S * 4; i++) {
    await sleep(250);
    try { r = JSON.parse(await evaluate(PROBE(vp.w))); } catch { continue; }
    if (r && r.ready) break;
  }
  // ready 之后再等一拍重新量一次。两个原因：布局在字体/图片落地后还会微调；
  // 更要紧的是 `__pageErrors` 是**快照**，就绪那一刻之后抛的错读不到 ——
  // 负向对照 `--injectjs 'setTimeout(...,300)'` 第一版就是这么漏过去的。
  // 这一拍只是把窗口拉宽一点，**不等于**覆盖了整个生命周期的报错：
  // 开局之后才发生的错（比如 AI 撞死时的分支）归 stall_check 那类脚本管。
  if (r) { await sleep(800); try { r = JSON.parse(await evaluate(PROBE(vp.w))); } catch { /* 用上一份 */ } }
  if (!r) { console.log(`FAIL  ${vp.w}x${vp.h}  探针取不到数`); failed = true; continue; }

  const notes = [];
  // 判据 1：横向溢出。尺子是 vp.w，不是 innerWidth —— 理由见顶部注释
  // （shrink-to-fit 会让 innerWidth 追着 scrollWidth 涨，那条不等式恒成立）。
  if (r.scrollWidth > vp.w + 0.5) {
    notes.push(`横向溢出 scrollWidth=${r.scrollWidth} > 设备宽度 ${vp.w}`);
  }
  if (r.innerWidth !== vp.w || r.scale < 0.999) {
    notes.push(`发生了 shrink-to-fit：innerWidth=${r.innerWidth}（应为 ${vp.w}）`
      + ` scale=${r.scale.toFixed(3)} —— 真机上表现为"进页面就得双指放大"`);
  }
  for (const o of r.overflowing) notes.push('元素越界: ' + o);

  // 判据 2：首屏可见。同样用 vp.h 而不是 innerHeight。
  const fold = vp.h + 0.5;
  const bothAbove = r.you && r.ai && r.you.bottom <= fold && r.ai.bottom <= fold;
  // 退路里的"操作提示"：加载完之后玩家画布上的遮罩就是提示（IDLE_VEIL），
  // 页脚那条 `space flap / R restart` 是桌面端的那份。任一可见都算。
  const hintVisible = (r.veil && !r.veil.hidden && r.veil.bottom <= fold)
    || (r.hint && r.hint.bottom <= fold);
  const youAbove = r.you && r.you.bottom <= fold;
  if (!bothAbove && !(youAbove && hintVisible)) {
    notes.push(`首屏放不下：you.bottom=${F(r.you?.bottom)} ai.bottom=${F(r.ai?.bottom)}`
      + ` 设备高度=${vp.h}，且操作提示也不在首屏`);
  }

  // 判据 3：两块画布不重叠。竖直方向重叠 && 水平方向重叠 才算真的叠在一起；
  // 单看水平会把"上下堆叠"的单栏布局误判成重叠。
  if (r.you && r.ai) {
    const ox = Math.min(r.you.right, r.ai.right) - Math.max(r.you.left, r.ai.left);
    const oy = Math.min(r.you.bottom, r.ai.bottom) - Math.max(r.you.top, r.ai.top);
    if (ox > 0.5 && oy > 0.5) notes.push(`两块画布重叠 ${F(ox)}x${F(oy)} px`);
    if (r.you.w < 40 || r.ai.w < 40) {
      notes.push(`画布被压得太窄（you=${F(r.you.w)} ai=${F(r.ai.w)}），288 宽的像素画面基本没法看`);
    }
  } else {
    notes.push('找不到 #youCanvas 或 #aiCanvas');
  }

  // 判据 4：页面自身的 JS 报错
  for (const e of r.errors) notes.push('JS 报错: ' + e);

  // 自检：模拟到底生没生效。这一条盯的不是页面而是**这个脚本自己** ——
  // 模拟悄悄失效的话，七行照样全绿，但量的是桌面 Chrome，全表作废。
  if (r.coarse !== !!vp.touch) {
    notes.push(`触摸模拟没生效：(pointer: coarse)=${r.coarse}，这一行应为 ${!!vp.touch}`);
  }

  const bad = notes.length > 0;
  if (bad) failed = true;
  rows.push({ vp, r, notes });
  console.log(`${bad ? 'FAIL' : ' ok '}  ${String(vp.w).padStart(4)}x${String(vp.h).padEnd(4)}`
    + ` dpr=${vp.dpr}  ${vp.name}`);
  console.log(`        scrollWidth=${r.scrollWidth}/${vp.w} scale=${r.scale.toFixed(2)}`
    + `  canvas ${F(r.you?.w)}x${F(r.you?.h)}`
    + `  底边 you=${F(r.you?.bottom)} ai=${F(r.ai?.bottom)} / ${vp.h}`
    + `  首屏=${bothAbove ? '两块' : (youAbove && hintVisible ? '玩家+提示' : '放不下')}`
    + `  coarse=${r.coarse}(${r.touchPoints})  worker=${r.worker ? 'on' : 'off'}`);
  for (const n of notes) console.log('        ' + n);

  if (shotDir) {
    fs.mkdirSync(shotDir, { recursive: true });
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    const dst = path.join(shotDir, `vp-${vp.w}x${vp.h}.png`);
    fs.writeFileSync(dst, Buffer.from(shot.data, 'base64'));
    console.log(`        截图（仅首屏）: ${dst}`);
  }
}

ws.close();
proc.kill();
try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* 无所谓 */ }

const nBad = rows.filter((x) => x.notes.length).length;
console.log(`\n${VIEWPORTS.length} 个视口，${VIEWPORTS.length - nBad} 过 ${nBad} 挂`);
console.log('注意：这是 Blink 的设备模拟，不是 iOS 的 WebKit —— 真机结论见 web/TESTING.md');
console.log(failed ? 'VIEWPORT CHECK FAILED' : 'VIEWPORT CHECK OK');
process.exit(failed ? 1 : 0);
