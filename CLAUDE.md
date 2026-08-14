# CLAUDE.md

此文件为 Claude Code 在此代码库中工作时提供指导。
用法见 [README.md](README.md)，架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，
算法原理见 [docs/learn/](docs/learn/README.md)。

**这个仓库的主要目的是教学**（学 RL 和神经网络）。所以改代码时，
解释"为什么"和代码本身同等重要 —— 现有代码里的注释密度是刻意的，请保持。

## 这是什么

Flappy Bird 的 Dueling Double-DQN，纯像素输入（4×80×80 二值帧），两个动作。

**当前管线**（唯一在用的）：

- 入口：`train.py` / `eval.py` / `play.py` / `plot.py` / `monitor.py`
- 算法：`flappy/`（config / model / replay / rollout / agent / csvlog / checkpoint）
- 环境：`game/flappy_env.py`（规则）+ `game/resources.py`（pygame 底座）
- 测试：`test/test_env_and_buffer.py`

## 硬性约定

1. **超参数只在 `flappy/config.py` 里改。** 不要在入口脚本里写死数值。
   计数量的单位必须写进名字（`*_decisions` / `*_grad_steps` / `*_episodes`）——
   旧管线最难查的一类 bug 就是帧/决策步/局数三个计数器被混用。

2. **网络里不许加 BatchNorm 或 Dropout。** 目标网络处于 train 模式时
   BatchNorm 会让 Bellman 算子随批次变化，没有不动点，训练不可能收敛。
   `flappy/model.py: assert_deterministic` 会在三条路径启动时拦截这类改动。

3. **动作选择只走 `flappy/rollout.py` 和 `Agent.act*`。** 不要在评测或
   可视化脚本里复制一份 —— 旧管线正是这样让所有评测都变成了随机策略。

4. **不要在动作选择里加"按训练进度短路"的分支。** warmup 期间的全随机
   靠 `epsilon_at()` 返回 1.0 实现，这样 epsilon 曲线永远等于实际随机比例。

5. **`env.step()` 绝不内部 reset。** 回合结束必须由调用方显式 `reset()`
   并重置帧栈（`FrameStack.reset`），否则会产生跨局拼接的假样本。

6. **奖励一律用 `+=` 累加，不要用赋值。** 同一帧可能既过管道又撞死。

7. **热循环里不要 print，尤其不要 emoji。** Windows cp1252 的 stdout
   会抛 UnicodeEncodeError 直接终止训练。用 `RunLogger.say()`（纯 ASCII）。

8. **分析脚本（`plot.py` / `monitor.py`）只读 CSV，不 import 项目代码。**
   它们要能对着拷贝过来的 `runs/` 目录跑，也不该触发 pygame 初始化。

9. **不要用 PowerShell 读-改-写仓库里的中文文本文件。**
   PS 5.1 的 `Get-Content -Raw` 不指定编码时按系统 ANSI 码页读，
   UTF-8 中文读进来就已经错了，再 `Set-Content` 写回去就固化成乱码。
   改文件用编辑工具。同理 `setup.ps1` **必须存成 UTF-8 with BOM**，
   否则 PS 5.1 解析不了里面的中文。

## 环境

Windows 用 `.\setup.ps1`（国内加 `-Mirror`）。**本项目不提供 Docker** ——
之前那份 Dockerfile 从未被实际运行过，已删除。

## 改动后必须验证

```bash
python test/test_env_and_buffer.py     # 11 个单测，约 30 秒
python train.py --smoke --allow-cpu    # 管线自检
```

每个单测都对应一个真实存在过的 bug。测试挂了先看是不是把旧缺陷改回来了，
不要直接改断言。

## docs/legacy/

旧管线（`deep_Q_oneStep.py` / `deep_Q_dueling_DQN.py` / `continue_training.py`）
的文档存档，**内容已过时**，其中的"智能目标网络选择""预训练模型继续训练"
等机制均已废弃。不要照着它们改代码。旧代码已删除，见 git 历史。
