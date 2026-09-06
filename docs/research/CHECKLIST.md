# 本机待办清单（2026-09-06 第四版重写）

## 【2026-09-06 晚间新增，置顶】云端跑完 Layer-0 四个诊断实验后，留给本机的两条

> 背景：这一轮云端 session 按 `PLAN-2026-09-06-next.md` §1 的 0.1/0.2/0.4 三条
> （物理下限 oracle / 死亡解剖+Q轨迹 / 翻转落在哪些状态）在 Node 里全部真跑了，
> 数字都在 `docs/research/LAYER0-RESULTS.md`。下面两条是云端**做不到**、必须
> 本机（有 torch）补的部分。这两条和上面"第四版"的 #1/#2/#3（两个种子的
> 诊断训练）是**两条独立的线**，不冲突，但如果时间只够做一条，建议先做
> #1（诊断训练），因为下面这两条都依赖它产出的 checkpoint。

### 新-1【需要 torch，本质是一行导出命令】补一次真正的 churn / argmax_flip 测量

**背景**：云端写了 `web/tools/flip_on_critical.mjs`，能把探针状态分成"关键
状态"（前瞻 K 步内某个动作必死）和"无差别状态"（两个动作都安全），然后比较
两个模型在这两类状态上的贪愿动作翻转率。但云端只有 `runs/base_s0/final.pt`
一个模型导出的 JS 权重，没有**同一次训练**里相邻的第二个 checkpoint，
只能拿网页 demo 的旧模型（完全独立的另一次训练）凑数——这个"跨模型翻转率"
在 `LAYER0-RESULTS.md` 实验 D 里已经标注**不能当 churn 用**，只是弱佐证。

**跑什么**：

```bash
# 把同一次训练里相隔约 100 个梯度步的两个 checkpoint 都导出成 JS 权重
# （若 runs/base_s0 下没有存到这么细粒度的 checkpoint，退而求其次用
#  resume_0.pt / resume_1.pt 这种间隔更大的，也比跨训练强得多）
python web/tools/export_weights.py --ckpt runs/base_s0/<早一点的ckpt>.pt --tag base_s0_a
python web/tools/export_weights.py --ckpt runs/base_s0/<晚一点的ckpt>.pt --tag base_s0_b
```

导出后会各自生成 `web/model/weights_base_s0_a_fp16.bin` + `weights-meta-base_s0_a.js`
（`_b` 同理，具体命名看 `export_weights.py --tag` 的实现）。然后把
`web/tools/flip_on_critical.mjs` 里两行 `import { WEIGHTS_META as META_OLD }
from '../model/weights-meta.js'`（以及 `META_NEW`）换成这两个新 tag 对应的
meta 文件——**这是本机唯一需要写的代码改动**，其余分类/统计逻辑都已经现成。

**大概要跑多久**：导出几秒钟；`flip_on_critical.mjs 800 15` 本身云端实测
47.8 秒（Node，无需 GPU，本机会更快）。

**看什么算成功**：真正的 `argmax_flip@关键状态` vs `argmax_flip@无差别状态`
哪个更高。如果无差别状态显著更高（云端的跨模型代理指标弱倾向支持这个方向，
且和 `timescale_probe.csv` 的"churn 大概率良性"结论一致），"压制策略抖动"
那条改动方向（CHAIN 风格正则化）可以从计划里划掉；如果关键状态更高，
说明抖动确实可能伤到关键决策，这条不能划掉。

**结果回填到**：`docs/research/LAYER0-RESULTS.md` 实验 D 新增一段，明确标注
"这次是真正的同次训练 churn 测量"，替换掉云端那段"跨模型代理，仅供参考"
的限定语。

### ✅ 新-2 已完成（云端第二轮，2026-09-06 晚）：oracle hazard 非单调曲线的根因

云端第二轮直接测了这个（`web/tools/lookahead_saturation.mjs --sweep 400`）：
"两个分支都判不安全"的比例从 N=60 的 4.81% 单调升到 N=300 的 37.33%，
和 hazard 的"先降后升"完全对应——**推测被坐实**，不需要本机再做。
顺带试了个修正（都不安全时选 `survivalSteps` 更大的分支，而不是退回
启发式）：N=120 上 hazard 从 0.706% 压到 0.559%（3.3 个标准误，显著），
但仍未摸到 0.47% 的目标线。完整数字见 `docs/research/LAYER0-RESULTS.md`
"本轮新增"一节的任务 B。

---


> **和第三版的关系**：`git log` 最新提交仍是第三版写的 `49a05dd`——本机在
> 这一轮检索期间**没有新的提交**，所以第三版清单的 #1（两个种子的诊断
> 训练）按规矩视为**仍未完成**，原样保留置顶，不重复发一份新的。
> 这一版只做了两件事：①云端做了第四轮检索，找到一些新证据（见下方
> "本轮新增背景"）；②据此给 #1 之后的步骤补充了几条可以直接抄的具体
> 参数建议，减少本机做 #3 时还要自己再查一轮的成本。**任务本身没有变**：
> 硬前置还是 #1，没有真实的 diag.csv，#2/#3 都无法进行。

## 本轮新增背景（不影响优先级排序，只是给后面步骤补细节）

云端第四轮检索找到两篇之前三轮没搜到的论文，和本项目"动作差距塌缩 → argmax
翻转"这条待验证的因果链直接相关：

1. **CHAIN**（NeurIPS 2024）把这个现象叫"greedy action deviation"，
   在 **MinAtar + DoubleDQN**（本项目尺度上最接近的一篇 peer-review 工作）
   上直接测过，但云端没拿到具体数字（`arxiv.org` 被环境拦截）。
2. **C-CHAIN**（ICML 2025）给出"秩塌缩通过 NTK 驱动 churn 恶化"的机制
   论证——**部分**填上了本项目因果链缺失的一环，但它的实验场景是
   continual RL + policy-gradient，不是本项目这种单任务 DQN。

**结论：两篇都不能替代本机用 diag.csv 做的直接验证**，因为没有一篇论文在
"单任务/DQN/小尺度/稀疏奖励"这个和本项目完全一致的组合上验证过。它们的
价值是：①确认 `argmax_flip` 这个诊断量本身是有 peer-review 工作背书的
标准做法，不是本项目自己发明的；②给 #3 的 EMA 实现提供了一个具体的 τ
起点（见下）；③多了一个新候选 T2.5（CHAIN 风格 churn 正则化），但优先级
排在 LayerNorm/ReDo/EMA 之后，不影响下面 #1-#3 的顺序。完整细节见
`docs/research/NOTES-2026-09-06.md` §1.15/1.16 和
`docs/research/UPGRADE_ANALYSIS.md` 第 1.4 节、第三节④。

---

## #1【现成脚本，但耗时很长——这是这轮最重要的一条，硬前置】跑 2 个新种子的
## 基线训练，同时拿到 D4（运行间方差）和第一份真实 diag.csv

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

**多一个解读维度（本轮新增，不改变判据本身）**：如果符号方向和 CHAIN/C-CHAIN
（见上方"本轮新增背景"）的机制方向一致，这个结论不只是本项目内部自洽，也
和外部文献吻合，可信度更高，回填时可以顺带提一句；如果不一致，同样值得记录——
说明本项目的具体情况和那两篇论文假设的机制不完全一样，本身就是一个信息量。

**结果回填到**：`docs/research/UPGRADE_ANALYSIS.md` 第一节新增"1.5 因果链的
直接验证"小节（1.4 这个编号本轮已经被云端用于外部文献综述，本机写实测结果时
用 1.5 或更靠后的编号，避免覆盖云端刚写的内容），把两个种子的相关系数表格
贴进去，并且明确写这一步是**支持**了还是**没有支持**因果链——如果没支持，
`docs/IMPROVEMENT_PLAN.md` T1.4 里"这条尚未验证"那句话要改成"验证结果
是……"，不能放着不管。

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

**如果 #3 完成后还想接着做 EMA 目标网络（`UPGRADE_ANALYSIS.md` 第三节③）**：
τ 直接从 **5e-3** 起步（第四轮检索找到的参考值，来自一篇大规模 Atari 计算
扩展律论文的 τ 敏感度扫描，方向性结论是"τ 在合理范围内选择不太关键，偏离
一个数量级只降约 19% 数据效率"），不需要专门为本项目尺度重新扫一遍——
如果时间紧张，这一条可以先用 5e-3 跑一次，观察 diag.csv 里 `argmax_flip`
曲线是否比硬同步更平，而不是自己再花一轮训练去调 τ。

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
ReDo + EMA（第三节②③）、以及本轮新增的 T2.5（CHAIN 风格 churn 正则化，
`UPGRADE_ANALYSIS.md` 第三节④），每一条要出干净的配对结果都得跑至少一次几小时
的完整训练。累积起来是好几个"过夜跑"量级的工作，不是一次坐下来就能做完的。建议：

- 如果硬件允许（多 GPU 或显存够跑两个进程），#1 的两个种子可以并行跑，
  节省一半时间。
- 每次开新训练前确认磁盘够放（`runs/` 不进 git，但每次跑的 `episodes.csv`/
  `train.csv`/`eval.csv`/`diag.csv` 加上 `resume_0.pt`/`resume_1.pt`/
  `best.pt`/`final.pt` 存档，几个跑下来也有几十 GB，`--buffer 60000`
  本身占约 3.1GB 显存，和磁盘占用是两回事）。
- 云端完全无法核实 `docs/IMPROVEMENT_PLAN.md` T2.4（Adam β 匹配 / weight
  clipping）、T1.4/T2.5 引用的 CHAIN/C-CHAIN/τ 扫描三篇论文的原始数字——
  `openreview.net`、`arxiv.org`、`rlj.cs.umass.edu`、`bluecontra.github.io`、
  `value-scaling.github.io`、`alphaxiv.org` 在云端环境全部被出口代理拦截，
  只读到 WebSearch 摘要转述和一份第三方 GitHub 论文笔记。**如果本机能正常
  访问这些网站，建议人工核对一遍原文的消融数字**，尤其是 CHAIN 论文里
  MinAtar+DoubleDQN 那组"greedy action deviation"的具体图表——那是目前
  找到的和本项目 argmax_flip 诊断最直接对应的一手数据，云端始终没能读到
  具体数字，只确认了"这个实验存在"。

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
- ✅（云端第四轮检索）为"塌缩→翻转"因果链找到部分外部文献支持
  （CHAIN/C-CHAIN）、补上 EMA τ 的具体起点建议（5e-3）、新增候选 T2.5——
  这些都已写进 `UPGRADE_ANALYSIS.md`/`IMPROVEMENT_PLAN.md`，**不需要本机
  额外做什么**，只是给 #1-#3 的执行提供了更具体的参数参考
