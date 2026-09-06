# 部署到 Cloudflare

纯静态站点，零构建步骤，零后端。整个"服务端"只做一件事：把 `web/` 里的
文件递出去。神经网络跑在**访问者自己的浏览器**里（`web/ai-worker.js`），
所以没有任何需要服务器算的东西，也就没有运行成本。

## 走的是 Workers Static Assets，不是 Pages

配置就是仓库根目录的 [`wrangler.jsonc`](../wrangler.jsonc)，**提交进 git 就生效**。

早先这份文档写的是 Cloudflare Pages。改掉的原因有两条：

1. **Pages 的配置只存在于网页控制台里**，没有可提交的等价文件。这份文档
   当时只能写成"人在 dashboard 里该填什么"的清单，换个人、换台机器就得
   照着重点一遍，还容易点错（下面那条就是真点错过的）。
2. Cloudflare 现在把静态站往 Workers Static Assets 上迁，新建 Pages 项目
   是在走下坡路。

顺带一个实际好处：纯资源投递（`wrangler.jsonc` 里**故意不写 `main` 字段**，
即没有 Worker 脚本）不消耗 Workers 的请求额度，也没有冷启动。

## 一次性设置（Cloudflare dashboard）

**Workers & Pages → Create application → 连接 GitHub 仓库
`zfwk1992/Flyppy_bird_RL_learn`。**

建好之后进 **Settings → Build**，确认这四项：

| 字段 | 值 |
|---|---|
| Branch（部署分支） | **`demo/web`** |
| Build command | 留空 / `None` |
| Deploy command | `npx wrangler deploy` |
| Root directory | `/` |

环境变量一个都不需要——这个站点没有 API key，没有后端调用。

### ⚠️ 两个真踩过的坑

1. **Branch 必须改成 `demo/web`。** 默认是 `main`，而 **`main` 分支上根本
   没有 `web/` 目录**——网站代码只在 `demo/web` 上。不改的话构建会拉到一个
   没有站点文件的分支，`wrangler deploy` 直接失败。
2. **别走 Pages 那个标签页。** 两条路的配置字段完全不一样：Pages 要在控制台
   里填 "Build output directory"，Workers 则读 `wrangler.jsonc`。混着来会得到
   一堆对不上的报错。

### `assets.directory` 是全部要害

`wrangler.jsonc` 里 `"assets": { "directory": "./web" }` —— **只有 `web/`
会被发布出去。** 仓库根目录还有 `flappy/`、`game/`、`models/*.pt`、`train.py`
这些训练代码和权重，跟站点毫无关系且体积不小。不限定范围的话，要么把训练
代码一起公开发布出去，要么 Cloudflare 把仓库根的 `README.md` 当首页、
找不到 `web/index.html`。

## `web/_headers` 是什么

已经提交在仓库里，放在资源目录里会被自动读取，不需要在 dashboard 里额外配置：

- `/model/*`、`/assets/*`：中等时长的强缓存（模型权重 2.5MB + 精灵数据，
  改动频率低，但没有做文件名指纹，所以不用 `immutable`，留了 1 天余地）。
- `/index.html`、`/`：不缓存，改文案/修 bug 后用户能立刻看到最新版本。
- 基本的 `X-Content-Type-Options` / `Referrer-Policy`。静态站点没有表单，
  给个保守默认值即可，不需要更复杂的 CSP（也没这个必要——零第三方脚本）。

部署后值得确认一次缓存头确实生效了（`curl -I` 看 `cache-control`）。

## 部署后要手动确认的三件事

1. **首页是不是游戏。** 如果看到的是仓库的 `README.md`，说明发布范围没限定
   在 `web/`，回去检查 `wrangler.jsonc` 的 `assets.directory`。
2. **OG 卡片图。** `index.html` 里 `og:image` 用的是相对路径
   `assets/og-card.jpg`。拿到域名后丢进
   [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/) 跑一遍，
   确认抓取器能正确解析出图片——不同平台对相对路径 og:image 的支持程度不一样，
   这一步不能跳。解析不出来就换成绝对 URL 再测一次，否则帖子的预览卡片是空的。
3. **真机走一遍。** 这是目前唯一还没有证据的一项。手机上重点看两处：
   开局有没有顿挫、以及**你掉下去之后 AI 的计数是不是继续往上爬**。
   原因见 `web/PROGRESS.md` 2026-09-06 那条：CDP 的 `setCPUThrottlingRate`
   **不节流 worker 线程**，所以低端安卓上"计划生产速度跟不跟得上 30 fps"
   在 headless 里测不出来，只有真机能回答。真嫌慢的话下一步是优化
   `web/nn.js` 的卷积，那要重跑 `nn_check`。

## 自定义域名（可选，plan.md 里定的是"先不买"）

不买域名也能用——Cloudflare 会给一个免费的 `*.workers.dev` 子域名，自带 HTTPS，
功能上和自购域名完全一样。想要 `yourname.dev` 这类好看的名字（约 $10/年）
再去 **Domains → Registrations** 买，然后在项目的 **Domains** 里加一条，
走 Cloudflare 自己的 DNS 几分钟内自动签发证书，不需要改这个仓库里的任何文件。
