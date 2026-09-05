# LinkedIn 帖子草稿

英文，受众是 LinkedIn 上的招聘方 / RL 同行（`plan.md` 第七节的既定受众）。
按 `plan.md` 第一节定的现实路径写：帖子正文不放外链（LinkedIn 算法会压低带
外链帖子的曝光），demo 链接放在**第一条评论**里；原生视频直接传到帖子里
（信息流自动播放），不是链接。

---

## 视频

`demo_wechat_30s.mp4`（1.2 MB，H.264，576×1024 竖屏，那局 100 根管道的结尾）——
这份素材已经存在，不在这个仓库里，发帖时手动上传为原生视频附件，不要放链接。
建议开头 3 秒就能看到 AI 连续过管道的画面，LinkedIn 信息流里没有声音、默认
静音自动播放，前 3 秒决定有没有人停下来看完。

## 正文（复制这段进帖子编辑框，不要带链接）

```
I built an AI that beats humans at Flappy Bird by roughly 11x — and it never
sees a single number. Only pixels.

What it gets: four stacked 80x128 black-and-white frames. No bird height, no
pipe distance, no velocity handed to it as features — it infers all of that
from how four frames differ from each other.

What it optimizes: +1 for clearing a pipe, -1 for dying, 0 otherwise. That's
the entire reward function. It learns a Q-function on top of that and acts
greedily every 4 frames.

The architecture detail I like most: a Dueling Double-DQN, which splits
"how good is this situation" from "does my next move actually matter." In
Flappy Bird, most frames the flap decision is close to irrelevant — the bird
is centered, the next pipe is far off. Only in a narrow band near a gap edge
does the choice carry real weight. Splitting the network into those two
heads means the value estimate learns from every single frame, not just the
one where an action got played.

Results: 78.2 pipes cleared on average over 100 episodes in the Python
training environment. I then ported the exact same weights to a
hand-written, dependency-free JavaScript forward pass running in a browser —
2.5 MB, 28ms per decision — and verified it bit-for-bit against the PyTorch
model: 1,200 reconstructed frames pixel-identical, 300 sampled decisions
action-for-action identical. In the browser it clears 86.4 pipes on average
over 30 runs. Humans, in my own small sample: about 7.

The part of this project I'm actually proudest of isn't the score. It's that
this exact architecture once plateaued at 1.3 pipes after 100,000 episodes
on an earlier version of the pipeline — and the fix wasn't a smarter
algorithm. It was finding and closing about a dozen structural bugs: a
terminal-state flag that silently got dropped on 75% of deaths, a target
network left in training mode with BatchNorm quietly invalidating the
Bellman equation's fixed-point guarantee, a reward scale that saturated the
loss function so "cleared one pipe" and "cleared five" produced the same
gradient. None of that shows up in a loss curve. All of it shows up in the
score.

You can play against it yourself, right in your browser, no install — link
in the first comment. Same seed, same physics — you and the agent see the
exact same pipes. It doesn't get a difficulty setting turned down for you.
```

（正文字数约在 LinkedIn 桌面端"see more"折叠点之上没关系——这是故意的，
钩子句已经在前两行，折叠不影响转化；但如果贴的时候平台限制变了，
可以从"Results:"那段开始砍，保留开头钩子 + 最后一段的工程故事 + CTA。）

## 第一条评论（发完正文立刻跟一条，带链接）

```
Play it here: <替换成 Cloudflare Pages 部署好之后的正式 URL>

Full code, training logs, and a 13-part written breakdown of the RL
fundamentals (MDPs, Q-learning, Double DQN, Dueling architecture, why the
first version didn't converge) — all in the repo:
https://github.com/zfwk1992/Flyppy_bird_RL_learn
```

**发布前必须把第一个 URL 换掉** —— `web/DEPLOY.md` 走完 Cloudflare Pages
的设置之后才会有真实域名，草稿阶段这里先留占位符。

## 话题标签（放正文最后或第一条评论都行，3-6 个足够，别刷屏）

```
#ReinforcementLearning #DeepLearning #MachineLearning #PyTorch #GameAI
```

## 发布检查表

- [ ] Cloudflare Pages 已经部署好，拿到真实 URL（`web/DEPLOY.md`）
- [ ] 用 LinkedIn Post Inspector 测过 OG 卡片图能正常抓取（`web/DEPLOY.md` 有链接）
- [ ] 正文里的 URL 占位符已替换成真实链接
- [ ] 视频已经上传为**原生视频**，不是外链
- [ ] 正文本身**不含任何链接**（链接只放第一条评论）
- [ ] 发布后立刻跟一条第一条评论（带链接），不要等
