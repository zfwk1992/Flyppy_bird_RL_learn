# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个使用 PyTorch 深度 Q 网络 (DQN) 的 Flappy Bird 强化学习项目。该项目实现了一个通过深度强化学习来学习玩 Flappy Bird 的 AI 智能体，具有优化的神经网络和多种训练变体。

## 核心架构

### 主要组件

1. **DQN 网络实现**:
   - `deep_q_network_pytorch_optimized.py`: 复杂版本，包含多层网络架构
   - `deep_Q_oneStep.py`: **推荐版本**，使用 `FixedOptimizedDQN` 单步预测网络
     * 2个卷积层 + 批归一化 + 自适应池化
     * 2个全连接层 (1024→512→2)
     * AdamW 优化器 + 梯度裁剪
     * Double DQN + Huber 损失

2. **游戏环境** (`game/`):
   - `wrapped_flappy_bird_fast.py`: 500 FPS 的优化游戏包装器
   - `flappy_bird_utils.py`: 游戏工具和资源加载
   - `wrapped_flappy_bird.py`: 标准游戏包装器

3. **程序入口**:
   - `deep_Q_oneStep.py`: **主要推荐训练脚本** (单步预测，稳定可靠)
   - `deep_q_network_pytorch_optimized.py`: 复杂版本训练
   - `final_flappy_bird.py`: 手动游戏，支持 R 键重启

### 关键特性

- **多帧输入**: 使用 4 个连续帧 (4x80x80) 作为网络输入
- **经验回放**: 20,000 内存缓冲区，批次大小 64
- **优化游戏参数**: 更大的管道间隙 (150px) 和调整的小鸟物理
- **CUDA 支持**: 自动 GPU 检测和内存管理
- **全面日志**: 训练日志保存到 `logs/` 目录

## 开发命令

### 环境设置

```bash
# CPU 版本 (推荐开发使用)
pip install -r requirements_cpu.txt

# GPU 版本 (需要 CUDA 11.8+)  
pip install -r requirements_cuda.txt

# 基础版本
pip install -r requirements.txt
```

### 训练命令

```bash
# 推荐训练脚本 (单步预测，稳定)
python deep_Q_oneStep.py

# 复杂版本训练 (多层网络)
python deep_q_network_pytorch_optimized.py

# 手动游戏测试
python final_flappy_bird.py
```

### 开发工作流

1. **GPU 检查**: 代码自动检测 CUDA 并设置内存占用为 0.8
2. **日志记录**: 所有训练运行在 `logs/` 目录创建时间戳日志
3. **模型保存**: 网络保存到 `saved_networks/` 为 `.pth` 文件
4. **备份脚本**: `DL_train_Backup/` 中的替代实现

## 关键配置

### DQN 超参数 (deep_Q_oneStep.py 推荐配置)
- `OBSERVE = 3000`: 训练前的观察步数 (减少等待时间)
- `EXPLORE = 20000`: ε衰减的探索步数 (优化收敛速度)
- `REPLAY_MEMORY = 20000`: 经验回放缓冲区大小
- `BATCH = 64`: 训练批次大小
- `GAMMA = 0.99`: 折扣因子
- `FRAME_PER_ACTION = 2`: 每2帧做一次决策 (降低计算量)

### FixedOptimizedDQN 网络配置
- 卷积层1: Conv2d(4→32, 8×8, stride=4, padding=2) + BatchNorm
- 卷积层2: Conv2d(32→64, 4×4, stride=2, padding=1) + BatchNorm  
- 自适应池化: AdaptiveAvgPool2d(4×4) 确保固定输出尺寸
- 全连接层: Linear(1024→512) + Linear(512→2)
- 优化器: AdamW(lr=1e-4, weight_decay=1e-5) + 梯度裁剪

### 游戏参数
- `FPS = 500`: 快速训练模式
- `PIPEGAPSIZE = 150`: 更大间隙便于学习
- 输入: 4帧 80x80 灰度图像
- 动作: 2个 (跳跃/不跳跃)

## 模型选择建议

### 推荐使用 deep_Q_oneStep.py
- **优点**: 单步预测，避免时序复杂性，训练稳定
- **适用**: 新手学习、快速验证、生产环境
- **特性**: FixedOptimizedDQN 网络，自适应池化，梯度裁剪

### 备选 deep_q_network_pytorch_optimized.py  
- **优点**: 复杂网络架构，更多优化技术
- **适用**: 深度研究、性能调优、实验对比
- **特性**: 多层网络，复杂损失函数，高级优化策略

## 训练监控

### 关键指标
- **训练步数**: 观察期(3K) → 探索期(20K) → 稳定期
- **ε值衰减**: 1.0 → 0.001 线性衰减
- **平均得分**: 目标达到50+分稳定性能
- **损失值**: Huber损失应逐渐收敛

### 日志分析
```bash
# 查看实时训练日志
tail -f logs/training_oneStep_*.log

# 监控模型保存
ls -la saved_networks/bird-dqn-oneStep-*.pth
```

## 文档资源

### 代码文档
- `instruction/`: 详细的技术说明文档
- `FixedOptimizedDQN_architecture.html`: 网络架构可视化
- `FlappyBird_DQN_Complete_Guide.html`: 完整项目指南
- `WSL_Installation_Guide.md`: WSL环境安装指南

### 项目文件
- `assets/`: 游戏精灵图和音频文件
- `images/`: 文档图片和演示GIF
- `logs/`: 训练日志 (按时间戳命名)
- `saved_networks/`: 模型检查点 (.pth文件)
- `DL_train_Backup/`: 实验版本和备份实现

## 测试和验证

### 训练验证
1. **实时监控**: 通过日志观察损失收敛和得分提升
2. **模型保存**: 每100轮自动保存，便于回滚和对比
3. **手动测试**: 使用 `final_flappy_bird.py` 验证智能体表现

### 性能基准
- **收敛时间**: 通常2-4小时 (取决于硬件)
- **目标性能**: 平均得分50+，最高分100+
- **稳定性**: 训练后期ε=0.001，主要利用学习策略