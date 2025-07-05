# Flappy Bird DQN 项目安装指南

## 📋 系统要求

### 操作系统
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04+ / CentOS 7+

### Python版本
- Python 3.8 - 3.11 (推荐 3.9+)

### 硬件要求
- **CPU版本**: 任意现代CPU
- **GPU版本**: NVIDIA GPU (推荐RTX 2060或更高)
  - CUDA 11.8+ (GPU版本)
  - 至少4GB显存

## 🚀 快速安装

### 方法1: 使用pip安装 (推荐)

#### 1. 克隆项目
```bash
git clone https://github.com/yourusername/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
```

#### 2. 创建虚拟环境
```bash
# 使用conda (推荐)
conda create -n flappy-bird-dqn python=3.9
conda activate flappy-bird-dqn

# 或使用venv
python -m venv flappy-bird-dqn
# Windows
flappy-bird-dqn\Scripts\activate
# Linux/macOS
source flappy-bird-dqn/bin/activate
```

#### 3. 安装依赖

**CPU版本** (推荐新手):
```bash
pip install -r requirements_cpu.txt
```

**GPU版本** (需要NVIDIA GPU):
```bash
pip install -r requirements_cuda.txt
```

**通用版本**:
```bash
pip install -r requirements.txt
```

#### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 🔧 详细安装步骤

### Windows安装

#### 1. 安装Python
1. 访问 [python.org](https://www.python.org/downloads/)
2. 下载Python 3.9+ (勾选"Add to PATH")
3. 验证安装: `python --version`

#### 2. 安装CUDA (GPU版本)
1. 访问 [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
2. 下载CUDA 11.8+
3. 安装CUDA Toolkit
4. 验证安装: `nvcc --version`

#### 3. 安装依赖
```bash
# 创建虚拟环境
python -m venv flappy-bird-dqn
flappy-bird-dqn\Scripts\activate

# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements_cuda.txt  # GPU版本
# 或
pip install -r requirements_cpu.txt   # CPU版本
```

### Linux安装

#### Ubuntu/Debian
```bash
# 安装系统依赖
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# 安装CUDA (GPU版本)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# 安装项目依赖
git clone https://github.com/yourusername/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python3 -m venv flappy-bird-dqn
source flappy-bird-dqn/bin/activate
pip install -r requirements_cuda.txt
```

#### CentOS/RHEL
```bash
# 安装系统依赖
sudo yum install python3 python3-pip git

# 安装CUDA (GPU版本)
sudo yum install cuda

# 安装项目依赖
git clone https://github.com/yourusername/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python3 -m venv flappy-bird-dqn
source flappy-bird-dqn/bin/activate
pip install -r requirements_cuda.txt
```
## 🚀 快速开始

安装完成后，可以立即开始使用：

```bash
# 开始训练
python deep_q_network_pytorch_optimized.py

# 或使用命令行工具
flappy-bird-train

# 查看游戏演示
flappy-bird-play
```

## 📞 获取帮助

如果遇到安装问题：

1. 查看 [常见问题](#故障排除)
2. 检查 [系统要求](#系统要求)
3. 提交 [Issue](https://github.com/yourusername/DeepLearningFlappyBird/issues)
4. 查看 [文档](https://github.com/yourusername/DeepLearningFlappyBird/blob/main/README.md)

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。 