/**
 * 移动端布局核查：真·设备模拟，量出窄视口下有没有横向溢出。
 *
 * 为什么需要这个工具，而不是直接 `chrome --headless --window-size=390,844 --screenshot`
 * ------------------------------------------------------------------------------
 * **`--window-size` 不等于移动端视口。** 它设的是窗口尺寸，页面拿到的布局视口
 * 未必是 390，`<meta name=viewport content="width=device-width">` 也不会按手机
 * 的方式生效，devicePixelRatio 还是 1。用它截出来的图会把一个**完全正常**的
 * 响应式页面拍成"内容被右边裁掉"的样子 —— 这个坑真的踩过一次，差点去"修"一个
 * 根本不存在的 bug。
 *
 * 正确做法是走 CDP 的 `Emulation.setDeviceMetricsOverride`，把 `mobile: true`
 * 和 deviceScaleFactor 一起设上。这个脚本就干这件事，并且不只截图 ——
 * 它直接量 `documentElement.scrollWidth` 和每个元素的 right 边界，
 * 给出**可判定**的结论，不用靠人眼看截图猜。
 *
 * 零依赖：用 Node 内置的 fetch + WebSocket 直连 CDP，不需要 Playwright。
 *
 * 用法：
 *     python -m http.server 8123 --directory web &
 *     node web/tools/mobile_check.mjs [url] [--shots <目录>]
 *
 * 退出码 0 = 所有视口都没有横向溢出；1 = 有溢出，并列出是哪些元素撑宽的。
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith('--')) || 'http://localhost:8123/index.html';
const shotDir = args.includes('--shots') ? args[args.indexOf('--shots') + 1] : null;

// 常见的 Chromium 位置。云端一般是 chromium/google-chrome 在 PATH 里；
// Windows 上给出两条常规安装路径。找不到就报错退出，不要静默跳过检查。
const CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/opt/node22/lib/node_modules/playwright/.local-browsers/chromium/chrome-linux/chrome',
].filter(Boolean);
const chrome = CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch { return false; } })
  || 'google-chrome';

// 要测的视口。320 是还在用的最窄的主流手机（iPhone SE 一代），
// 390 是 iPhone 12/13/14 那一档，414 是 Plus/Max 一档。
const VIEWPORTS = [
  { w: 320, h: 568, dpr: 2, name: 'iPhone SE' },
  { w: 390, h: 844, dpr: 3, name: 'iPhone 13' },
  { w: 414, h: 896, dpr: 2, name: 'iPhone 11 Pro Max' },
];

const PORT = 9222 + Math.floor(Math.random() * 300);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
  `--remote-debugging-port=${PORT}`,
  '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-mobile-check'),
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

await send('Page.enable');
await send('Runtime.enable');

// 在页面里量：scrollWidth 超过 innerWidth 就是有横向溢出；
// 再把越界的元素列出来，这样报错能直接指向要改的那条 CSS。
const PROBE = `(() => {
  const over = [...document.querySelectorAll('body *')]
    .filter((el) => el.getBoundingClientRect().right > innerWidth + 0.5)
    .slice(0, 10)
    .map((el) => {
      const r = el.getBoundingClientRect();
      const id = el.id ? '#' + el.id
        : el.tagName.toLowerCase() + (el.className && typeof el.className === 'string'
            ? '.' + el.className.trim().split(/\\s+/)[0] : '');
      return id + ' right=' + r.right.toFixed(0) + ' width=' + r.width.toFixed(0);
    });
  return JSON.stringify({
    innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    overflowing: over,
    errors: window.__pageErrors || [],
  });
})()`;

let failed = false;
for (const vp of VIEWPORTS) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: vp.w, height: vp.h, deviceScaleFactor: vp.dpr, mobile: true,
  });
  // 记录页面自身抛的错，光看布局不够 —— 脚本挂了页面也可能"看起来正常"
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: 'window.__pageErrors=[];addEventListener("error",e=>__pageErrors.push(String(e.message)));'
      + 'addEventListener("unhandledrejection",e=>__pageErrors.push("unhandled: "+e.reason));',
  });
  await send('Page.navigate', { url });
  await sleep(4000);        // 等权重加载（2.5 MB）+ 首屏渲染

  const raw = (await send('Runtime.evaluate',
    { expression: PROBE, returnByValue: true, awaitPromise: true }))?.result?.value;
  const r = JSON.parse(raw);
  const bad = r.scrollWidth > r.innerWidth || r.overflowing.length || r.errors.length;
  if (bad) failed = true;
  console.log(`${bad ? 'FAIL' : ' ok '}  ${String(vp.w).padStart(3)}x${vp.h} dpr=${vp.dpr}`
    + `  (${vp.name})  scrollWidth=${r.scrollWidth} innerWidth=${r.innerWidth}`);
  for (const o of r.overflowing) console.log(`        溢出: ${o}`);
  for (const e of r.errors) console.log(`        JS 报错: ${e}`);

  if (shotDir) {
    fs.mkdirSync(shotDir, { recursive: true });
    const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    const dst = path.join(shotDir, `mobile-${vp.w}.png`);
    fs.writeFileSync(dst, Buffer.from(shot.data, 'base64'));
    console.log(`        截图: ${dst}`);
  }
}

ws.close();
proc.kill();
console.log(failed ? '\nMOBILE LAYOUT FAILED' : '\nMOBILE LAYOUT OK');
process.exit(failed ? 1 : 0);
