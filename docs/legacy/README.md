# 旧管线文档存档

> ⚠️ **这里的内容全部已经过时，不要照着它们改代码。**
> 当前管线的说明在仓库根目录的 [README.md](../../README.md)。

这些文档描述的是重写之前的那套实现：

- `deep_Q_oneStep.py`（FixedOptimizedDQN，单步预测）
- `deep_Q_dueling_DQN.py` / `deep_Q_dueling_fine_training.py`
- `continue_training.py`（预训练模型继续训练）

那套管线跑了 4 轮、最长 10 万局 / 13 小时，近百局均分始终停在约 1.3 根管道。
诊断确认了 12 个互相掩盖的缺陷，其中 5 个是结构性的、无法靠调参绕开
（帧跳过错位、目标网络几乎不同步、BatchNorm/Dropout 破坏贪婪与不动点、
跨局帧栈拼接、PER 的 α 被施加两次）。详见根 README 的"关键设计"一节 ——
每一条都对应 `test/test_env_and_buffer.py` 里的一个单测。

其中已废弃的机制包括：

- **智能目标网络选择**（按历史最佳表现挑目标网络）—— 现在按固定的
  1000 个**梯度步**硬同步。原机制建立在"最佳网络"这个概念上，
  而当时的 best 判定本身是坏的（阈值每局被重置，每个正分局都是"新纪录"）。
- **预训练模型继续训练 + 观察期 50/50 策略** —— 现在的 warmup 是纯随机
  （p_flap=0.2），不需要预训练模型来"提高经验质量"。
- **BATCH=512 / 大批次优化** —— 实测小网络下 GPU 被 kernel 启动延迟主导，
  batch 128 与 32 的每步耗时几乎一样，512 没有额外收益。

保留这批文档是因为其中的推导过程和调试记录仍有参考价值
（尤其是 `memory_error_analysis.md` 和 `instruction/` 下的原理说明）。
旧代码本身已删除，可在 git 历史中查阅：

```bash
git log --diff-filter=D --name-only    # 找到删除它们的提交
git show <commit>^:deep_Q_oneStep.py   # 取回某个旧文件
```

## 目录

| 文件 | 内容 |
|------|------|
| `CLAUDE.md` | 旧管线的项目指导（已被根目录新版取代） |
| `Dueling.md` | Dueling 架构说明 |
| `dueling_dqn_documentation.md` | 旧 dueling 实现的文档 |
| `dueling_dqn_reward_q_calculation.md` | 奖励与 Q 值计算 |
| `dueling_dqn_vs_original_dqn_comparison.md` | Dueling vs 原始 DQN |
| `Optimized_Dueling_DQN_Analysis.md` | 旧优化分析 |
| `Enhanced_Training_System_Summary.md` | 旧训练系统总结 |
| `memory_error_analysis.md` | 显存/内存问题的排查记录 |
| `Adversarial_DQN_Beginner_Guide.md` | 对抗式 DQN 入门（未落地的方向） |
| `Adversarial_Learning_Theory_Analysis.md` | 对抗学习理论分析（同上） |
| `learning.md` | 学习笔记 |
| `instruction/` | 原理与优化说明（9 篇） |

> `Docker_Tutorial.md` 和 `init_env_error.md` 已一并删除 —— 项目不再提供
> Docker 环境（那份 Dockerfile 从未被实际运行过），留着教程只会误导。
> 安装用根目录的 `setup.ps1`。
