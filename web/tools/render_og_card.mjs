// 把 og_card_source.html 截图成 web/assets/og-card.jpg（1200x630，LinkedIn/OG 标准尺寸）。
// 用法：node web/tools/render_og_card.mjs
//
// 两条路径，先试 Playwright，失败就退到裸 CDP：
//   - 云端预装了 Node 版 Playwright，但 ESM 的 bare specifier 解析不认 NODE_PATH，
//     所以只能用 file:// 绝对路径去 import 它；
//   - **本机（Windows）没有 Playwright**，那条 import 必然失败。以前这里没有退路，
//     结果就是改了卡片文案却没法在本机重新生成图片。现在退到 mobile_check.mjs
//     用的那套零依赖 CDP：Node 内置 fetch + WebSocket 直连 headless Chrome。
// 两条路径产出的都是同一个 #card 元素的截图，尺寸由 CSS 钉死，所以结果一致。
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcPath = path.join(__dirname, 'og_card_source.html');
// jpeg 不是画质取舍——仓库根 .gitignore 里 `*.png` 只给 `images/*.png` 开了例外
// （阶段 2 的 obs 精灵也踩过这个坑，见 PROGRESS.md 变更日志）。这张图全是纯色块
// 和文字，jpeg q92 肉眼看不出压缩痕迹，体积还更小。
const outPath = path.join(__dirname, '..', 'assets', 'og-card.jpg');
const fileUrl = 'file://' + srcPath.replace(/\\/g, '/');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function viaPlaywright() {
  const { chromium } = await import('/opt/node22/lib/node_modules/playwright/index.mjs');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
  await page.goto(fileUrl);
  await page.locator('#card').screenshot({ path: outPath, type: 'jpeg', quality: 92 });
  await browser.close();
  return 'playwright';
}

async function viaCDP() {
  const CANDIDATES = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  const chrome = CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch { return false; } });
  if (!chrome) throw new Error('既没有 Playwright，也找不到 Chrome/Edge，无法生成卡片');

  const port = 9222 + Math.floor(Math.random() * 300);
  const proc = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--mute-audio',
    `--remote-debugging-port=${port}`,
    '--user-data-dir=' + path.join(os.tmpdir(), 'flappy-og-card'),
    'about:blank',
  ], { stdio: 'ignore' });

  try {
    let target = null;
    for (let i = 0; i < 60 && !target; i++) {
      await sleep(250);
      try {
        target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json())
          .find((t) => t.type === 'page');
      } catch { /* 浏览器还没起来 */ }
    }
    if (!target) throw new Error(`Chromium 没能在 ${port} 上起 CDP：${chrome}`);

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
    await send('Emulation.setDeviceMetricsOverride',
      { width: 1200, height: 630, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: fileUrl });
    await sleep(2500);      // 纯静态页，等字体和布局稳定即可

    // 按 #card 的实际盒子裁，而不是按视口裁 —— 卡片尺寸是 CSS 定的，
    // 万一以后改了尺寸，这里不用跟着改。
    const box = JSON.parse((await send('Runtime.evaluate', {
      expression: `JSON.stringify((()=>{const r=document.querySelector('#card').getBoundingClientRect();
        return {x:r.x,y:r.y,width:r.width,height:r.height};})())`,
      returnByValue: true,
    }))?.result?.value);
    if (!box || !box.width) throw new Error('页面里找不到 #card');

    const shot = await send('Page.captureScreenshot', {
      format: 'jpeg', quality: 92,
      clip: { ...box, scale: 1 },
      captureBeyondViewport: true,
    });
    if (!shot?.data) throw new Error('captureScreenshot 没有返回数据');
    fs.writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
    ws.close();
  } finally {
    proc.kill();
  }
  return 'cdp';
}

let how;
try {
  how = await viaPlaywright();
} catch (e) {
  console.log('Playwright 不可用（' + e.message.split('\n')[0] + '），改走 CDP');
  how = await viaCDP();
}
const { size } = fs.statSync(outPath);
console.log(`wrote ${outPath}  (${(size / 1024).toFixed(0)} KB, via ${how})`);
