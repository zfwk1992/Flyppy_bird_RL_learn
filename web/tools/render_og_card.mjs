// 把 og_card_source.html 截图成 web/assets/og-card.jpg（1200x630，LinkedIn/OG 标准尺寸）。
// 用法：node web/tools/render_og_card.mjs
// ESM 的 bare specifier 解析不认 NODE_PATH，云端没有本地 npm 依赖，所以用 file:// 相对
// 定位云端预装的 Node 版 Playwright（/opt/node22/lib/node_modules/playwright）；
// 本机如果全局/本地装了 playwright 包，把下面这行换回 `import { chromium } from 'playwright'` 也行。
import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcPath = path.join(__dirname, 'og_card_source.html');
// jpeg 不是画质取舍——仓库根 .gitignore 里 `*.png` 只给 `images/*.png` 开了例外
// （阶段 2 的 obs 精灵也踩过这个坑，见 PROGRESS.md 变更日志），改 .gitignore
// 超出了这次改动被允许碰的范围（只能改 web/、plan.md）。这张图全是纯色块和文字，
// jpeg q92 肉眼看不出压缩痕迹，体积还更小。
const outPath = path.join(__dirname, '..', 'assets', 'og-card.jpg');

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.goto('file://' + srcPath);
await page.locator('#card').screenshot({ path: outPath, type: 'jpeg', quality: 92 });
await browser.close();
console.log('wrote', outPath);
