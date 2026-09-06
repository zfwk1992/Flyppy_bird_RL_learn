# 本机待办清单（2026-09-06 第三版重写）

> **这一版和前两版的关系**：前两版的 #1（画训练曲线/D2）、#2（D0 固定评测集）
> 已经在本机做完并合入主分支（提交 `51151da`、`b139f60`、`d577edc`）。
> 云端这次能直接从 `docs/IMPROVEMENT_PLAN.md`、`docs/EXPERIMENTS.md`、
> `docs/research/UPGRADE_ANALYSIS.md` 里读到证据，**不需要再靠"仓库里有没有
> 出现某个文件"去猜有没有做完**——而且本机这期间做的事情比前两版清单预想的
> 多得多：不仅完成了 D0/D2，还做完了 D3（配对回测三个存档）和网络内部诊断
> （`experiments/netdiag_dormant.py`），**查到了根因是可塑性丢失**（休眠单元/
> 有效秩塌缩/动作差距塌缩），写了完整分析 `docs/research/UPGRADE_ANALYSIS.md`，
> 并且已经把诊断量接进了训练循环（`flappy/diagnostics.py`，提交 `1dd848d`，
> 产出 `runs/<run>/diag.csv`）。
>
> 前两版清单里的 #1-#5 **不再重复列出**——它们描述的工作全部已完成或已被
> 更精确的后续工作取代。这一版的清单完全对应**下一步**：diag.csv 刚接好，
> 还没有一次真实的长训练跑过它（只有 `--smoke` 到 ep2000 的自检），
> 所以当前最靠前的开放问题是**验证 UPGRADE_ANALYSIS.md 因果链的最后一环**
> （"动作差距塌缩 → argmax 翻转 → 分数震荡"）以及**运行间方差**（D4，
> 至今没测过——D3 量的是同一次训练内部的峰谷差异，不是独立重跑的方差）。
>
> 排序即优先级，**最多 4 条 + 1 条不紧急的设计提醒**。#1 是硬前置——
> 没有真实的 diag.csv，#2/#3 都无法进行。#3 依赖 #1/#2 的结果支持因果链
> 才值得投入（如果 #2 显示相关性方向不对，先回来重新想，不要盲目上 LayerNorm）。

---

## #1【现成脚本，但耗时很长——这是这轮最重要的一条】跑 2 个新种子的基线训练，
## 同时拿到 D4（运行间方差）和第一份真实 diag.csv

**背景**：`docs/research/UPGRADE_ANALYSIS.md` 第五节自己写的执行顺序第一步就是
"先加诊断日志（不改算法），跑 2 个 seed 的基线——这同时解决了 D4"。诊断日志
已经加完了（`1dd848d`），**这一步现在可以直接做，不需要再写代码**。

D3 的配对比较（`best.pt` vs `resume_1.pt` vs `resume_0.pt`）量化的是**同一次
训练内部**峰谷之间的方差（3.7 倍），这**不是** D4 要问的"独立重跑多个 torch
seed 之间的方差"——两者是不同的量，不要混着当同一件事已经做完。
`UPGRADE_ANALYSIS.md` 第四节明确把这个缺口列为"可能落空的地方"：如果两次
同配置训练本来就差 2 倍，那么后面任何"LayerNorm/ReDo 有没有用"的结论都读不出来。

**跑什么**（和 `EXPERIMENTS.md` 里 final_v1 的训练命令完全一样，只换 seed 和
run-dir；`--seed` 会同时播种 `random`/`numpy`/`torch`，和评测用的固定
`eval_seed_base` 互不干扰）：

```bash
python train.py --buffer 60000 --max-hours 6 --seed 1 --run-dir runs/diag_seed1
python train.py --buffer 60000 --max-hours 6 --seed 2 --run-dir runs/diag_seed2
```

**大概要跑多久**：参照 final_v1 的实测速率（4.5 小时 / 19850 局 / 233 万决策步，
约 51.8 万决策步/小时），`--max-hours 6` 封顶。两次可以顺序跑（总共最多约
12 小时）；如果显卡显存够、想并行跑两个进程也可以，两边 `--run-dir` 不同不会
冲突。**这条没有办法压缩时间**——UPGRADE_ANALYSIS 里观察到的震荡发生在
ep15000-19500 这个晚期区间，跑得太短根本进不到这个阶段，diag.csv 也就看不到
真正的震荡对不对得上诊断量。如果时间实在紧张，**先跑一个种子也比不跑强**，
第二个种子可以留到下一轮。

**看什么算成功**：

1. **D4（运行间方差）**：两次跑（可以的话，连同 final_v1 原来的 seed=0 一起）
   在训练末期的固定集配对评测（跑完后用 `eval.py --fixed-set --episodes 100`
   评每次跑的 `final.pt`/`best.pt`）比较中位数和 IQR。如果三次跑的"最终表现"
   区间本身就跟 D3 观测到的峰谷差异（3.7 倍）同量级，说明后面任何算法改动的
   "提升"都可能只是抽到了好种子，必须先把这个不确定性钉死再谈优化。
2. **diag.csv 的时间序列**：训练进入利用期后，`dorm_fc`、`eff95_fc`、
   `q_gap_median`、`q_ceil_ratio`、`argmax_flip` 随 episode 的变化曲线，
   和同一份 `eval.csv` 的 `eval_pipes_mean` 曲线放在一起看——分数掉的那几个
   点，这几个诊断量是不是同步变差。**不需要在这一步就下结论**，只需要把数据
   跑出来，下一条（#2）会自动做这个比较。

**结果回填到**：`docs/IMPROVEMENT_PLAN.md` D4 那一行（运行间方差的具体数字）
以及第 5 节新增"5.4 运行间方差（D4）"小节；两份 `diag.csv`/`eval.csv` 的路径
记一笔，供 #2 使用。

---

## #2【代码已经写好、已自测——本机只需要跑】用新脚本验证"塌缩 → 翻转 → 震荡"
## 这条因果链最后一环

**背景**：`UPGRADE_ANALYSIS.md` 自己说得很清楚——"动作差距塌缩 → argmax 翻转"
这一步"还没有直接验证"，目前的证据只是"休眠/秩塌缩"和"分数震荡"**同时发生**，
不构成因果证明。这一条云端已经把分析代码写好并自测过了：

新增 `experiments/analyze_diag_vs_score.py`（文件头第一行已注明"未在本机验证"，
只依赖 numpy，不 import torch/pygame）。**云端已经跑过它的自测**（构造合成数据，
不依赖真实训练数据），输出如下：

```
=== 自测 1/3：spearman 实现在已知数据上的表现 ===
强正相关样本 -> r=0.998（预期接近 +1）
强负相关样本 -> r=-0.998（预期接近 -1）
无关样本     -> r=0.050（预期接近 0）
=== 自测 2/3：并列值（ties）处理 ===
完全相同（含并列）的两列 -> r=1.000（预期接近 +1）
=== 自测 3/3：csv 读取 + episode 对齐 + 端到端相关性方向 ===
/tmp/xxx: 8 个配对点（按 episode 对齐，eval.csv 共 8 行，diag.csv 共 8 行）
指标                 spearman r        符合预期
dorm_fc                -1.000           是
eff95_fc                1.000           是
q_gap_median            0.000           否
q_ceil_ratio            0.000           否
argmax_flip             0.000           否
全部自测通过。
```
（后三项判定"否"是因为自测数据里这三列全是常数、没有构造相关性，脚本正确地
没有瞎报相关——这正是自测要验证的行为：不构造相关性就不该看到相关性。）

**跑什么**（依赖 #1 产出的 `diag.csv`/`eval.csv`）：

```bash
python3 experiments/analyze_diag_vs_score.py runs/diag_seed1
python3 experiments/analyze_diag_vs_score.py runs/diag_seed2
```

**大概要跑多久**：几秒钟，纯 CPU 读 csv 算相关系数。

**看什么算成功**：`dorm_fc`/`eff95_fc`/`q_gap_median`/`q_ceil_ratio`/
`argmax_flip` 和 `eval_pipes_mean` 的 spearman 相关系数符号是否和
`UPGRADE_ANALYSIS.md` 假设的方向一致（休眠越多分数越低、有效秩越高分数越高、
动作差距越大分数越高、越贴近"永不死"上限分数越低、翻转越频繁分数越低）。
**配对点数大概率只有 10-40 个**（`eval_every_episodes=500`，一次训练几万局），
spearman 在这个样本量下噪声很大——**只看符号方向，不要报具体数值当结论**，
符号一致算"支持因果链"，符号不一致或接近零算"这条链在这个尺度上站不住"，
如实记录，不要为了让故事完整而选择性解读。

**结果回填到**：`docs/research/UPGRADE_ANALYSIS.md` 第一节新增"1.4 因果链的
直接验证"小节，把两个种子的相关系数表格贴进去，并且明确写这一步是**支持**了
还是**没有支持**因果链——如果没支持，`docs/IMPROVEMENT_PLAN.md` T1.4 里
"这条尚未验证"那句话要改成"验证结果是……"，不能放着不管。

---

## #3【依赖 #1/#2 结果，需要写代码】如果因果链得到支持，实现 LayerNorm 并配对比较

**只有 #2 显示相关性方向基本符合预期时才做这一步**——如果 #2 的结果说因果链
站不住，先停下来重新想，不要跳过诊断直接上改动（这正是 IMPROVEMENT_PLAN.md
第 1 节反复强调的"没有这一步，后面任何 A/B 对比都读不出信号"）。

**要写的代码**（`UPGRADE_ANALYSIS.md` 第三节①已经给出设计，且**已经本机验证过**
批次无关性等三项性质，只是原型代码没有提交，需要重新落地进 `flappy/model.py`
之类的文件）：

1. 卷积层后插 `nn.GroupNorm(1, C)`（等价于在 C,H,W 上做 LayerNorm，不依赖
   batch，满足 Bellman 算子需要的批次无关性），fc 层后插 `nn.LayerNorm`。
2. 跑 `test/test_env_and_buffer.py` 里的 `assert_deterministic`，确认新架构
   仍然通过——这是仓库硬性要求（CLAUDE.md 第 2 条的字面意思是禁止
   BatchNorm/Dropout，`UPGRADE_ANALYSIS.md` 已经论证 LayerNorm 不违反这条
   背后的原因，但**验证步骤不能省**）。
3. **顺手加一个几乎零成本的对照组**（本轮检索新增的弱证据候选，见
   `docs/IMPROVEMENT_PLAN.md` T2.4）：给 `torch.optim.Adam` 加一个可选的
   `betas` 覆盖（比如 CLI 加 `--adam-betas 0.9 0.9`），在跑 LayerNorm 对照的
   同时也跑一组 `betas=(0.9, 0.9)` 的对照，成本只是多一次训练，不需要专门
   为这条弱证据单独设计实验。**这条证据强度弱（两篇来源都只读到摘要转述，
   互相还有矛盾），不要因为加了这个对照就推迟 LayerNorm 本身的验证。**

**跑什么**（配对用同一个 seed，和 #1 的某一次基线做直接对比）：

```bash
python train.py --buffer 60000 --max-hours 6 --seed 1 --run-dir runs/layernorm_seed1
python eval.py runs/layernorm_seed1/final.pt --fixed-set --episodes 100 --q-stats
python eval.py runs/diag_seed1/final.pt --fixed-set --episodes 100 --q-stats   # baseline 对照
```

**大概要跑多久**：训练本身和 #1 同量级（几小时），评测各几分钟。

**看什么算成功**：

- 训练末期 `dorm_fc`/`never_fc` 显著低于 baseline（#1 的对应种子）——
  这是"LayerNorm 有没有真的复活休眠单元"的直接判据。
- 固定集 100 局的中位数/25 分位数**不低于** baseline（主要图稳，不图涨，
  `UPGRADE_ANALYSIS.md` 已经说清楚"峰值不一定提高，要买的是峰值之后不塌"）。
- `argmax_flip` 曲线是否比 baseline 更平。
- 差距小于 2 个标准误一律记作"没有区别"，失败也要写进 `EXPERIMENTS.md`。

**结果回填到**：`docs/EXPERIMENTS.md` 新增一节（格式参考 `n-step` 那条失败
记录的写法），以及 `docs/research/UPGRADE_ANALYSIS.md` 第三节①标注验证结果。

---

## #4【不紧急，设计/资源规划提醒】接下来几轮都需要多次几小时训练，提前规划

这不是一条要跑的命令，是提醒：#1、#3，以及 `UPGRADE_ANALYSIS.md` 后续还要做的
ReDo + EMA（第三节②③），每一条要出干净的配对结果都得跑至少一次几小时的完整
训练。累积起来是好几个"过夜跑"量级的工作，不是一次坐下来就能做完的。建议：

- 如果硬件允许（多 GPU 或显存够跑两个进程），#1 的两个种子可以并行跑，
  节省一半时间。
- 每次开新训练前确认磁盘够放（`runs/` 不进 git，但每次跑的 `episodes.csv`/
  `train.csv`/`eval.csv`/`diag.csv` 加上 `resume_0.pt`/`resume_1.pt`/
  `best.pt`/`final.pt` 存档，几个跑下来也有几十 GB，`--buffer 60000`
  本身占约 3.1GB 显存，和磁盘占用是两回事）。
- 云端完全无法核实 `docs/IMPROVEMENT_PLAN.md` T2.4（Adam β 匹配 / weight
  clipping）那两篇来源的原始论文——`openreview.net`、`arxiv.org`、
  `rlj.cs.umass.edu` 在云端环境全部被出口代理拦截，只读到 WebSearch 摘要
  转述和一份 GitHub README。如果本机能正常访问这些网站，**建议人工核对一遍
  原文的消融数字**，因为目前这条候选的证据强度全靠转述，云端没有能力再深挖。

---

## 已完成，不再重复发（供核对，不是待办）

- ✅ D0：固定评测 seed 集（`flappy/rollout.py: evaluate(seed_base=...)`，
  提交 `51151da`）
- ✅ D2：训练曲线复盘，推翻"单向崩塌"判断，改判"3-4 倍震荡"
  （`docs/IMPROVEMENT_PLAN.md` 第 0 节 A、第 5.2 节）
- ✅ D3：三个存档在固定集上的配对回测（`best.pt`/`resume_1.pt`/`resume_0.pt`，
  `docs/IMPROVEMENT_PLAN.md` 第 5.3 节）
- ✅ 网络内部诊断：`experiments/netdiag_dormant.py`，查到根因是可塑性丢失
  （休眠单元 47%→73%、有效秩 44→22、动作差距中位 1.10→0.32），完整分析见
  `docs/research/UPGRADE_ANALYSIS.md`
- ✅ 诊断日志接入训练循环：`flappy/diagnostics.py` + `runs/<run>/diag.csv`
  （提交 `1dd848d`），`--smoke` 到 ep2000 验证过管线通（`dorm_fc=0.4%
  eff95_fc=91 gap_med=1.160 q/ceil=0.15`），**但还没有一次真实长训练跑过它**
  ——这正是这版清单 #1 要做的事
- ✅ 死亡归因（B：评测期方差）：两批独立 25-30 局都指向策略脆弱而非任务太难
  （`docs/IMPROVEMENT_PLAN.md` 第 5.1 节），**这个方向目前证据已经足够，
  不需要更多样本量，不再列入清单**
