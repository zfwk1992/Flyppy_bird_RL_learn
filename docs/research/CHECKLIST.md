# 本机待办清单（2026-09-06 生成）

> 这是这个仓库第一次跑这个定时任务，**没有上一份清单**，所以这里全部是新的。
> 下一轮如果这份清单里有条目还没做，那一轮会保留并置顶，不会因为想到新点子就冲掉。
>
> 排序即优先级。**#1 和 #2 互不依赖，可以并行/任选先后；#3、#4 依赖 #2 先完成。**
> #5 是一条设计建议，不紧急，供你判断要不要现在做。

---

## #1【只需要跑现成脚本，零前置，强烈建议第一个跑】画训练曲线，看崩塌是渐变还是突变

**背景**：IMPROVEMENT_PLAN.md 的 D2 要回答"崩塌是渐变还是突变？max-Q 是否在
崩塌前先发散？"——这直接决定后面该冲 buffer 策略（T1.3）还是目标网络方差
（T1.4/T2.1）。这份数据**可能已经躺在你本机的 `runs/final_v1/` 目录里**，
不需要任何新代码、任何新训练：训练时每 500 局会写一行贪婪评测（`eval.csv`），
每 100 个梯度步会写一行 loss/Q 值（`train.csv`），`plot.py` 已经能直接画。

**先检查文件还在不在**（`.gitignore` 挡住了 `runs/`，本地清理过的话可能已经没了）：

```bash
ls runs/final_v1/*.csv
```

- 如果三个 csv（`episodes.csv` / `train.csv` / `eval.csv`）都在，跑：

  ```bash
  python plot.py runs/final_v1 --out docs/research/final_v1_diagnostics.png
  ```

  大概几秒钟。**看什么算成功**：
  1. `eval.csv` 里 `eval_pipes_mean` 这条曲线在 ep15493 附近（当时的 `best.pt`）
     到 ep19831 附近（`resume_0`）之间是**平滑下降**还是**某一小段突然掉下去**？
  2. 同一段时间窗口里 `train.csv` 的 `q_max` / `q_mean` 有没有先于分数下降
     开始发散或者跳变？如果 Q 值先发散，指向价值高估/崩塌；如果 Q 值稳定
     但分数掉了，更像是策略层面的问题（比如 policy churn，见 IMPROVEMENT_PLAN
     T1.4 里新加的那条候选解释）。
  3. 顺手看一眼 `td_abs_mean` 和 `loss` 在这段窗口有没有异常尖峰。

- 如果 csv 已经不在了：这条数据永久丢失了，不用补救，直接跳到 #5，
  并在下次真正训练时把这份诊断纳入起跑前检查清单。

**结果回填到**：`docs/IMPROVEMENT_PLAN.md` 第 1 节 D2 那一行的"产出"列，
贴曲线截图或者关键数字（哪个 episode 附近开始掉、Q 值曲线什么形状），
以及 `docs/research/NOTES-2026-09-06.md` 末尾补一段。

---

## #2【前置，必须先做，需要写代码】D0：修好评测的"固定 seed 集"

**背景**：读代码（不是猜测）确认了 `flappy/rollout.py: evaluate()` 现在
每局只调用一次 `env.reset()`，**不会逐局重新播种**，管道生成用的是全局
`random` 模块。哪怕命令行传了固定 `--seed`，只要两个模型的存活局长不同
（几乎总是不同），从第 2 局起消耗的随机数数量就不同，管道序列会**错位**——
固定 seed 现在只保证第 1 局可比。**这是 IMPROVEMENT_PLAN.md 全部工作里
优先级最高的一条，不修它后面所有 A/B 对比都不可信。**

`web/tools/death_attribution.mjs` / `ai_eval.mjs` 已经用对的方式做了这件事：
每一局单独 `new FlappyGame({ seed: BASE + ep })`。Python 侧要照抄同一个模式。

**要写的代码**（在本机、不在这次云端会话里做，因为 `flappy/` 和 `eval.py`
是仓库锁定不让云端改的文件）：

1. 新建 `flappy/evalset.py`：定义一个固定的 seed 列表（建议 200 个，
   比如 `list(range(5000, 5200))`——刻意和这次云端跑的
   `death_attribution.mjs`/`ai_eval.mjs` 用的基数 5000 保持一致，方便以后
   直接和 JS 侧的结果对表）。
2. 改 `flappy/rollout.py: evaluate()`（或者新写一个不改现有函数、只加一个
   新入口的版本，看你觉得哪个更不容易破坏现有调用方）：加一个可选的
   `episode_seeds` 参数，若提供，则每局开始前 `random.seed(s); np.random.seed(s)`
   再 `env.reset()`，s 取 `episode_seeds[i]`。
3. 改 `eval.py` 加 `--fixed-set` 开关，打开时从 `flappy/evalset.py` 读 seed
   列表传给 `evaluate()`，替代当前的单次全局 `--seed`。
4. 跑 `python test/test_env_and_buffer.py` 确认 11 个单测还过（改了
   `rollout.py` 必须过这一步，这是仓库硬性要求）。

**大概要跑多久**：写代码 + 调试，看你熟练程度，几十分钟到一两小时；
不是一个"跑脚本等结果"的任务。

**看什么算成功**：用 `--fixed-set` 对同一个 `models/final_v1_best.pt`
跑两次 `--episodes 100`，**两次结果必须逐位相同**（不止均值一样，
每一局的比分都要一样）——这是"真固定"和"看起来固定但没固定"的区别。

**结果回填到**：`flappy/evalset.py` 本身就是产出；跑通后把验证结果
（两次 100 局逐位相同的证明，或者截图/校验和）记一笔到
`docs/EXPERIMENTS.md`（新起一节，格式参考现有条目）。

---

## #3【依赖 #2】用固定 seed 集重新评一遍现存的存档，把 D3 的"崩塌点"钉死

**背景**：IMPROVEMENT_PLAN.md 原来设想的 D3 是"把 `runs/final_v1/` 里所有
周期存档都评一遍"，但读 `train.py` 发现**这不现实**——本项目只有三类存档：
`best.pt`（门控触发时才存，本次训练只有 ep15493 那一次触发）、
`resume_0.pt`/`resume_1.pt`（每 30 分钟双槽轮转，只保留最新两份，
更早的会被覆盖冲掉）、`final.pt`（收尾时存一次）。所以能拿到的最多就是
**3-4 个时间点**，不是一条密集曲线。**这是对原计划 D3 描述的一处修正**：
数量有限，但仍然值得跑——目的从"画完整曲线"降级为"用干净的配对比较
确认崩塌确实发生、且发生在什么范围"。

**先看 `runs/final_v1/` 里实际还剩什么**：

```bash
ls -la runs/final_v1/*.pt
```

**跑什么**（`--fixed-set` 是 #2 做完之后才有的开关）：

```bash
python eval.py runs/final_v1/best.pt --fixed-set --episodes 100 --q-stats
python eval.py runs/final_v1/resume_1.pt --fixed-set --episodes 100 --q-stats   # 如果还在
python eval.py runs/final_v1/resume_0.pt --fixed-set --episodes 100 --q-stats
python eval.py runs/final_v1/final.pt --fixed-set --episodes 100 --q-stats      # 如果还在
```

**大概要跑多久**：每条 100 局贪婪评测，参考 `death_attribution.mjs` 的速度
（约 14 秒/局），Python+torch 大概率更快，估计每条几分钟，四条合计
15-30 分钟。

**看什么算成功 / 怎么解读**：

- 拿到的每个存档的 `pipes_mean / pipes_median / pipes_std`，**用固定集
  的结果去对比 EXPERIMENTS.md 里已经记过的旧数字**（best.pt 78.2 / resume_0
  24.2，那两个数字不是固定 seed 集测的）——如果新旧数字差很多，
  说明旧结论本身就有一部分是测量噪声，不是真的崩塌那么剧烈；
  如果新旧数字量级一致，说明崩塌是真实存在、不是噪声，前一份结论站得住。
- 三四个点连起来，看崩塌是"单调掉"还是"先掉后有反弹又掉"。

**结果回填到**：`docs/IMPROVEMENT_PLAN.md` 第 5 节新起一小节
"5.2 固定 seed 集下的多存档回测"，以及更新 `docs/EXPERIMENTS.md`
final_v1 那一节（标注"以下用固定 seed 集重测"，不要覆盖旧数字，
两份并排放，方便看差异）。

---

## #4【依赖 #2，可以和 #3 一起跑】检验"policy churn"是不是崩塌的另一个原因

**背景**：这次检索找到一个 IMPROVEMENT_PLAN.md 之前没有的候选解释——
policy churn（Schaul et al. 2022）：DQN 类算法的贪婪策略会因为"动作差距小"
而在几次梯度更新内大范围抖动，这是一个**不依赖 buffer 淘汰**的现象 A
备选机制。已经写进 `docs/IMPROVEMENT_PLAN.md` T1.4 那一条。

`eval.py --q-stats` 已经会打印 `mean |Q1-Q0|`（动作差距）——如果 #3 已经跑了
`--q-stats`，这条不需要重新跑，直接对比 #3 四个存档输出里的
`mean |Q1-Q0|` 那一行即可。

**看什么算成功**：

- 如果动作差距从 best.pt 到 resume_0 **明显缩小**（比如掉了一半以上），
  支持 policy churn 是崩塌的重要成分，T1.4/T2.1（降方差的目标网络手段）
  优先级应该继续维持在前面。
- 如果动作差距基本不变，崩塌更可能纯粹是 buffer 淘汰死亡样本导致的
  Q 值失去下界约束，T1.3（保留死亡样本专池）的针对性更强。
- 两者都变化不大 / 都变化很大，如实记录"没有区分开"，不要强行下结论。

**结果回填到**：`docs/IMPROVEMENT_PLAN.md` T1.4 那条新加的"诊断方法"段落，
把结论写在后面。

---

## #5【设计建议，不紧急】下次正式训练前，给周期性存档留一份不会被冲掉的历史

这不是本轮要跑的命令，是提醒：`resume_0.pt`/`resume_1.pt` 的双槽轮转设计
是为了"断点续训"服务的，天然不适合拿来做"崩塌发生在哪一步"的事后取证——
更早的中间状态会被覆盖冲掉。如果 #1 发现 `runs/final_v1/` 的 csv 已经丢了，
或者 #3 发现能用的存档只剩 1-2 个，那么这条判断（"密集回测崩塌曲线"）
在这次训练上已经无法达成，**只能等下一次正式训练跑起来的时候**，
提前想好要不要额外存一份不覆盖的周期快照（比如每 2000 局存一个
`ckpt_ep{N}.pt`，成本是多占硬盘，换来的是这次拿不到的"分数-步数"曲线）。
这个决定留给你，不在这轮清单里定死——如果不想改 `train.py`，
`eval.csv` 每 500 局一条本身已经是一种轻量级的替代方案，只是这次
不确定文件还在不在（见 #1）。
