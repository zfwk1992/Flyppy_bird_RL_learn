# 部署到 Cloudflare Pages

纯静态站点，零构建步骤，零后端。这份文档记录 dashboard 里要填的字段——
Cloudflare Pages 连 GitHub 仓库的向导本身不能用 git 提交的文件配置，
所以这里只能写"人在网页控制台里该填什么"，不是一个可执行的配置文件。

## 一次性设置（Cloudflare dashboard → Workers & Pages → 创建）

1. **Connect to Git** → 选这个仓库 `zfwk1992/Flyppy_bird_RL_learn`。
2. **Production branch**：先填 `demo/web`（这个分支还没合并到主线；等
   demo 稳定、决定合并回主分支之后再改成那时的主分支名）。
3. **Framework preset**：`None`。
4. **Build command**：留空——没有构建步骤，`web/` 目录本身就是产物。
5. **Build output directory / Root directory**：`web`。
   **这一步是全部配置里唯一容易出错、也最关键的一步**——这个仓库根目录下
   还有 `flappy/`、`game/`、`train.py`、`models/*.pt` 这些训练用的代码和权重
   文件，体积不小，且和这个静态站点毫无关系。必须把 Pages 的服务根目录
   限定在 `web/`，否则要么把无关文件也发布出去（暴露仓库其他部分、拖慢
   部署），要么 Pages 直接把仓库根的 `README.md` 当首页、找不到
   `web/index.html`。
6. **环境变量**：不需要任何一个。这个站点没有 API key、没有后端调用。
7. 保存后 Cloudflare 会给一个 `*.pages.dev` 的预览域名，每次 push 到
   `demo/web` 都会自动重新部署。

## `web/_headers` 是什么

已经提交在仓库里（`web/_headers`），Cloudflare Pages 部署时会自动读取——
不需要在 dashboard 里额外配置。内容是：
- `/model/*`、`/assets/*`：中等时长的强缓存（模型权重 2.5MB+精灵数据，
  改动频率低，但没有做文件名指纹，所以不用 `immutable`，留了 1 天余地）。
- `/index.html`、`/`：不缓存，改文案/修 bug 后用户能立刻看到最新版本。
- 基本的 `X-Content-Type-Options` / `Referrer-Policy`，静态站点没有表单，
  给个保守默认值即可，不需要更复杂的 CSP（也没有这个必要——零第三方脚本）。

## 部署后要手动确认的两件事

1. **OG 卡片图**：`index.html` 里 `og:image` 用的是相对路径
   `assets/og-card.jpg`。拿到 `*.pages.dev` 域名（或最终的自定义域名）后，
   丢进 [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)
   跑一遍，确认抓取器能正确解析出图片——不同平台对相对路径 og:image 的
   支持程度不一样，这一步不能跳过。如果 LinkedIn 解析不出来，把
   `assets/og-card.jpg` 换成绝对 URL（`https://<你的域名>/assets/og-card.jpg`）
   再测一次。
2. **真实设备走一遍**：桌面 Chrome/Safari 各开一次，手机上至少点一次
   "tap to fly"，确认触摸事件、音效（如果有）、canvas 尺寸在真实设备
   DPI 下不糊。云端这边只验证到 headless Chromium 模拟触摸的程度
   （见 `web/PROGRESS.md` 变更日志），没有真机可用。

## 自定义域名（可选，plan.md 里定的是"先不买"）

如果之后买了域名，在 Pages 项目的 **Custom domains** 里加一条，
走 Cloudflare 自己的 DNS 的话几分钟内自动签发 HTTPS 证书，不需要
额外配置这个仓库里的任何文件。
