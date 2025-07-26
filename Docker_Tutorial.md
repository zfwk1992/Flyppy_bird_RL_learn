# Flappy Bird DQN Docker 构建教程

## 当前镜像状态分析

根据 `docker images` 输出，当前系统中有以下镜像：

```bash
REPOSITORY                             TAG                                 IMAGE ID       CREATED             SIZE
flyppy_bird_rl_learn_flappy-bird-dqn   latest                              bda9822d52ef   11 minutes ago      10.3GB
<none>                                 <none>                              46e2870b54ab   58 minutes ago      10.3GB
<none>                                 <none>                              38be7e6e8f91   About an hour ago   4.47GB
<none>                                 <none>                              f5335dff9582   About an hour ago   5.18GB
<none>                                 <none>                              e5e16ffe5e08   2 hours ago         4.66GB
nvidia/cuda                            11.8.0-cudnn8-runtime-ubuntu22.04   2d49de6afba5   20 months ago       3.74GB
nvidia/cuda                            11.8.0-base-ubuntu22.04             1e75b7decac0   20 months ago       239MB
```

### 镜像分析
- **主镜像**: `flyppy_bird_rl_learn_flappy-bird-dqn:latest` (10.3GB) - 最新构建成功的版本
- **悬挂镜像**: 4个 `<none>` 镜像 - 构建过程中产生的中间镜像
- **基础镜像**: 
  - `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (3.74GB) - CUDA 运行时环境
  - `nvidia/cuda:11.8.0-base-ubuntu22.04` (239MB) - CUDA 基础环境

## 快速开始指南

### 立即开始训练（推荐）
你的环境已准备就绪，可以直接开始训练：

```bash
# 使用现有镜像直接开始训练
docker-compose up flappy-bird-dqn

# 或者进入容器进行调试
docker-compose run --rm flappy-bird-dqn bash
```

### 清理空间（建议）
当前有多个悬挂镜像占用约 20GB 空间：
```bash
# 清理悬挂镜像，释放空间
docker image prune -f

# 查看清理效果
docker images
```

## Docker 构建 vs 运行 vs 进入容器

### 1. 构建镜像 (Build Image)
```bash
# 构建新镜像
docker-compose build flappy-bird-dqn
# 或
docker build -t flappy-bird-dqn .
```
**作用**: 
- 创建一个新的 Docker 镜像
- 执行 Dockerfile 中的所有指令
- 安装依赖、复制文件、设置环境
- **结果**: 生成一个可复用的镜像模板

### 2. 运行容器 (Run Container)
```bash
# 运行容器（一次性）
docker-compose up flappy-bird-dqn

sudo service docker start
# 或
docker run --gpus all flappy-bird-dqn
```
**作用**:
- 基于镜像创建并启动一个新容器
- 执行 CMD 指令（默认运行 `deep_Q_oneStep.py`）
- 自动开始训练
- **结果**: 容器运行训练程序，训练完成后容器停止

### 3. 进入容器 (Enter Container)
```bash
# 交互式进入容器
docker-compose run --rm flappy-bird-dqn bash
# 或进入运行中的容器
docker exec -it flappy_bird_rl_gpu bash
```
**作用**:
- 进入容器的命令行环境
- 可以手动执行命令、调试代码
- 不会自动运行训练程序
- **结果**: 获得容器内的 Shell 访问权限

## Docker 构建过程详解

### 1. 基础镜像选择
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```
**作用**: 
- 基于 Ubuntu 22.04 系统
- 预装 CUDA 11.8.0 运行时环境
- 预装 cuDNN 8 深度学习加速库
- 提供 GPU 计算支持

### 2. 环境变量设置
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
```
**作用**:
- `DEBIAN_FRONTEND=noninteractive`: 禁用交互式安装提示
- `PYTHONUNBUFFERED=1`: Python 输出不缓冲，实时显示日志

### 3. 系统依赖安装

#### Python 环境
```dockerfile
python3                # Python 解释器
python3-pip           # Python 包管理器
python3-dev           # Python 开发头文件
```

#### 系统工具
```dockerfile
git                   # 版本控制工具
```

#### 图像处理库
```dockerfile
libglib2.0-0          # GLib 核心库
libsm6                # X11 会话管理库
libxext6              # X11 扩展库
libxrender-dev        # X11 渲染库
libgomp1              # OpenMP 并行计算库
libgtk-3-0            # GTK 图形界面库
```

#### 游戏开发库 (Pygame 支持)
```dockerfile
libsdl2-dev           # SDL2 核心开发库
libsdl2-image-dev     # SDL2 图像处理
libsdl2-mixer-dev     # SDL2 音频混合
libsdl2-ttf-dev       # SDL2 字体渲染
libportmidi-dev       # MIDI 音频支持
```

#### 视频处理库 (OpenCV 支持)
```dockerfile
libswscale-dev        # FFmpeg 图像缩放
libavformat-dev       # FFmpeg 格式处理
libavcodec-dev        # FFmpeg 编解码器
libv4l-dev            # Video4Linux 摄像头支持
```

#### 图像格式支持
```dockerfile
zlib1g-dev            # 压缩库
libjpeg-dev           # JPEG 图像支持
libpng-dev            # PNG 图像支持
libtiff-dev           # TIFF 图像支持
```

### 4. Python 依赖安装

#### 深度学习框架
```python
torch                 # PyTorch 深度学习框架 (CUDA 版本)
torchvision          # PyTorch 计算机视觉工具包
```

#### 图像处理
```python
opencv-python        # OpenCV 计算机视觉库
Pillow              # Python 图像处理库
```

#### 数值计算
```python
numpy               # 数值计算核心库
scipy               # 科学计算库
```

#### 机器学习
```python
scikit-learn        # 机器学习工具包
```

#### 可视化和工具
```python
tqdm                # 进度条显示
matplotlib          # 绘图库
seaborn             # 统计可视化
```

#### 游戏环境
```python
pygame              # 游戏开发库
```

#### 开发工具
```python
jupyter             # Jupyter Notebook
ipython             # 增强的 Python 交互环境
```

### 5. 容器配置

#### 工作目录
```dockerfile
WORKDIR /app
```
设置容器内工作目录为 `/app`

#### 数据卷挂载
```dockerfile
# 在 docker-compose.yml 中配置
volumes:
  - ./logs:/app/logs                    # 训练日志
  - ./saved_networks:/app/saved_networks # 模型保存
  - ./assets:/app/assets                # 游戏资源
```

#### 环境变量
```dockerfile
ENV CUDA_VISIBLE_DEVICES=0    # 指定使用 GPU 0
ENV PYTHONPATH=/app           # Python 模块搜索路径
```

#### 端口暴露
```dockerfile
EXPOSE 8888                   # Jupyter Notebook 端口
```

## Docker 构建命令详解

### 1. 构建镜像
```bash
docker build -t flappy-bird-dqn .
```
- `-t flappy-bird-dqn`: 为镜像设置标签名
- `.`: 使用当前目录作为构建上下文

### 2. 运行容器
```bash
# 使用 docker-compose (推荐)
docker-compose up flappy-bird-dqn

# 手动运行
docker run --gpus all -v $(pwd)/logs:/app/logs flappy-bird-dqn
```
- `--gpus all`: 启用所有 GPU
- `-v`: 挂载数据卷

### 3. 进入容器
```bash
# 交互模式
docker-compose run --rm flappy-bird-dqn bash

# 查看运行中的容器
docker exec -it flappy_bird_rl_gpu bash
```

## 构建优化

### 1. 层缓存
Docker 按层构建，每个 `RUN` 命令创建一个层：
- 系统依赖安装合并到一个 `RUN` 命令
- 清理包管理器缓存 `rm -rf /var/lib/apt/lists/*`

### 2. 构建顺序
- 先复制 `requirements_cuda.txt`
- 后复制项目文件
- 利用缓存机制，依赖不变时重用层

### 3. .dockerignore
排除不必要的文件，减少构建上下文：
```
logs/
saved_networks/
.git/
__pycache__/
```

## GPU 支持配置

### 1. NVIDIA Docker 要求
```bash
# 安装 nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### 2. 测试 GPU 支持
```bash
# 检查 GPU 可用性
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

### 3. Docker Compose GPU 配置
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 镜像管理和维护

### 1. 清理悬挂镜像
当前系统中有 4 个悬挂镜像（`<none>`），占用大量空间：

```bash
# 查看所有悬挂镜像
docker images -f "dangling=true"

# 清理悬挂镜像（释放约 20GB 空间）
docker image prune -f

# 查看空间使用情况
docker system df
```

### 2. 重新构建策略
```bash
# 情况1: 代码修改（推荐）
docker-compose build flappy-bird-dqn  # 利用缓存，快速构建

# 情况2: 依赖修改或环境问题
docker-compose build --no-cache flappy-bird-dqn  # 完全重新构建

# 情况3: 仅运行已构建镜像
docker-compose up flappy-bird-dqn  # 直接使用现有镜像
```

### 3. 镜像大小优化
当前镜像 10.3GB 较大，可以通过以下方式优化：

```dockerfile
# 在 Dockerfile 中添加清理步骤
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 使用多阶段构建（高级）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder
# ... 安装构建依赖
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as runtime
# ... 只复制运行时需要的文件
```

## 常见使用场景

### 场景1: 快速训练（推荐）
```bash
# 直接运行训练，使用现有镜像
docker-compose up flappy-bird-dqn

# 查看实时日志
docker-compose logs -f flappy-bird-dqn
```

### 场景2: 调试和开发
```bash
# 进入容器进行调试
docker-compose run --rm flappy-bird-dqn bash

# 在容器内手动运行
cd /app
python3 deep_Q_oneStep.py
python3 final_flappy_bird.py  # 手动游戏测试
```

### 场景3: 代码修改后重新训练
```bash
# 1. 修改代码
vim deep_Q_oneStep.py

# 2. 重新构建（利用缓存）
docker-compose build flappy-bird-dqn

# 3. 运行新版本
docker-compose up flappy-bird-dqn
```

## 故障排除

### 1. 权限问题
```bash
# WSL 中需要 sudo
sudo docker-compose up flappy-bird-dqn
```

### 2. GPU 不可用
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

### 3. 音频问题（已修复）
项目已修复 pygame 音频初始化问题：
```python
# game/flappy_bird_utils.py 中使用 DummySound 类
# game/wrapped_flappy_bird_fast.py 中设置 SDL_VIDEODRIVER=dummy
```

### 4. 内存不足
```bash
# 清理 Docker 系统
docker system prune -a

# 停止所有容器
docker stop $(docker ps -aq)

# 删除悬挂镜像
docker image prune -f
```

### 5. 镜像拉取失败
```bash
# 手动拉取基础镜像
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 检查网络连接
ping registry-1.docker.io
```

## 使用流程

1. **克隆项目**
   ```bash
   git clone <repository>
   cd Flyppy_bird_RL_learn
   ```

2. **构建并运行**
   ```bash
   sudo docker-compose up flappy-bird-dqn
   ```

3. **查看日志**
   ```bash
   # 实时查看训练日志
   tail -f logs/training_oneStep_*.log
   ```

4. **保存模型**
   ```bash
   # 模型自动保存到 saved_networks/
   ls -la saved_networks/
   ```

通过 Docker 容器化，用户无需安装复杂的 CUDA、PyTorch 环境，一键即可开始强化学习训练。

---

# Docker 原理深入解析

## Docker 镜像分层原理

### 1. Union File System (联合文件系统)
Docker 使用分层存储，每个指令创建一个只读层：

```
最终镜像结构:
┌─────────────────────────────────────┐
│ 应用层: COPY . . (可写)               │  ← 容器运行时层
├─────────────────────────────────────┤
│ 项目文件层: COPY . .                  │  ← 50MB
├─────────────────────────────────────┤
│ Python 包层: pip install            │  ← 800MB
├─────────────────────────────────────┤
│ 系统包层: apt-get install           │  ← 200MB
├─────────────────────────────────────┤
│ 工作目录层: WORKDIR /app             │  ← 几KB
├─────────────────────────────────────┤
│ 环境变量层: ENV DEBIAN_FRONTEND...   │  ← 几KB
└─────────────────────────────────────┘
│ CUDA 基础层: Ubuntu + CUDA + cuDNN   │  ← 1.5GB (共享)
└─────────────────────────────────────┘
```

### 2. 层共享机制
- **同一主机**: 相同的层在不同镜像间共享存储
- **缓存复用**: 未更改的层直接复用，大幅减少构建时间
- **增量更新**: 只传输和构建变更的层

### 3. Copy-on-Write (写时复制)
- **只读层**: 所有镜像层都是只读的
- **容器层**: 运行时在顶部添加可写层
- **文件修改**: 修改文件时从只读层复制到可写层

## Docker 镜像管理完全指南

### 1. 镜像查看和检查

#### 基本查看命令
```bash
# 查看所有本地镜像
docker images
docker image ls

# 查看特定镜像
docker images flappy-bird-dqn

# 查看镜像详细信息
docker inspect flappy-bird-dqn

# 查看镜像构建历史
docker history flappy-bird-dqn

# 查看镜像层大小分布
docker system df -v
```

#### 镜像信息解读
```bash
REPOSITORY        TAG       IMAGE ID       CREATED        SIZE
flappy-bird-dqn  latest    abc123def456   2 hours ago    2.5GB
nvidia/cuda      11.8.0... fed789abc123   3 weeks ago    1.5GB
```
- **REPOSITORY**: 镜像名称
- **TAG**: 版本标签 (默认 latest)
- **IMAGE ID**: 唯一标识符
- **CREATED**: 创建时间
- **SIZE**: 镜像大小

### 2. 构建策略和缓存优化

#### 🔄 **构建时机判断**
```bash
# 检查是否需要重新构建
docker-compose config  # 验证配置文件

# 情况1: 首次构建 (必须)
docker-compose build flappy-bird-dqn

# 情况2: 代码变更 (需要)
# 修改了 .py 文件
docker-compose build flappy-bird-dqn

# 情况3: 依赖变更 (需要，清除缓存)
# 修改了 requirements_cuda.txt
docker-compose build --no-cache flappy-bird-dqn

# 情况4: 仅运行 (无需构建)
docker-compose up flappy-bird-dqn  # 直接使用已存在镜像
```

#### 🚀 **构建优化技巧**
```dockerfile
# 优化1: 分离变化频率不同的层
COPY requirements_cuda.txt ./      # 依赖文件先复制
RUN pip3 install -r requirements_cuda.txt  # 依赖安装 (缓存)
COPY . .                          # 项目文件后复制 (经常变化)

# 优化2: 合并 RUN 命令减少层数
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 优化3: 使用 .dockerignore 减少构建上下文
# .dockerignore 内容:
logs/
saved_networks/
__pycache__/
.git/
```

### 3. 镜像生命周期管理

#### 开发阶段工作流
```bash
# 1. 修改代码
vim deep_Q_oneStep.py

# 2. 重新构建 (只重建变更层)
docker-compose build flappy-bird-dqn

# 3. 测试运行
docker-compose up flappy-bird-dqn

# 4. 查看日志
docker-compose logs -f flappy-bird-dqn
```

#### 生产阶段工作流
```bash
# 1. 构建发布版本
docker-compose build flappy-bird-dqn

# 2. 标记版本
docker tag flappy-bird-dqn flappy-bird-dqn:v1.0

# 3. 多次运行 (复用镜像)
docker-compose up flappy-bird-dqn  # 秒启动
```

### 4. 镜像清理和维护

#### 存储空间管理
```bash
# 查看 Docker 空间使用情况
docker system df

# 清理未使用的镜像
docker image prune

# 清理悬挂镜像 (dangling)
docker image prune -f

# 清理所有未使用资源 (镜像、容器、网络、卷)
docker system prune -a

# 强制清理所有内容
docker system prune -a --volumes
```

#### 镜像版本管理
```bash
# 创建版本标签
docker tag flappy-bird-dqn flappy-bird-dqn:v1.0
docker tag flappy-bird-dqn flappy-bird-dqn:stable

# 删除特定标签
docker rmi flappy-bird-dqn:v1.0

# 删除所有相关镜像
docker rmi $(docker images flappy-bird-dqn -q)
```

## Docker 镜像分享和分发

### 1. 镜像导出/导入 (离线分享)

#### 导出镜像为文件
```bash
# 导出单个镜像
docker save -o flappy-bird-dqn.tar flappy-bird-dqn

# 导出多个镜像
docker save -o flappy-bird-complete.tar flappy-bird-dqn nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 压缩导出 (节省空间)
docker save flappy-bird-dqn | gzip > flappy-bird-dqn.tar.gz
```

#### 导入镜像文件
```bash
# 导入镜像
docker load -i flappy-bird-dqn.tar

# 导入压缩镜像
gunzip -c flappy-bird-dqn.tar.gz | docker load

# 验证导入
docker images flappy-bird-dqn
```

### 2. Docker Hub 分享 (在线分享)

#### 准备发布
```bash
# 1. 注册 Docker Hub 账号 (https://hub.docker.com)

# 2. 登录
docker login

# 3. 重新标记镜像 (添加用户名前缀)
docker tag flappy-bird-dqn yourusername/flappy-bird-dqn:latest
docker tag flappy-bird-dqn yourusername/flappy-bird-dqn:v1.0
```

#### 发布镜像
```bash
# 推送镜像到 Docker Hub
docker push yourusername/flappy-bird-dqn:latest
docker push yourusername/flappy-bird-dqn:v1.0

# 其他人使用
docker pull yourusername/flappy-bird-dqn:latest
docker run --gpus all yourusername/flappy-bird-dqn:latest
```

### 3. 私有镜像仓库

#### 本地私有仓库
```bash
# 启动本地 Registry
docker run -d -p 5000:5000 --name registry registry:2

# 标记镜像
docker tag flappy-bird-dqn localhost:5000/flappy-bird-dqn

# 推送到本地仓库
docker push localhost:5000/flappy-bird-dqn

# 从本地仓库拉取
docker pull localhost:5000/flappy-bird-dqn
```

### 4. 企业级分发策略

#### 多平台构建 (支持不同架构)
```bash
# 创建多平台构建器
docker buildx create --name multiplatform --use

# 构建多架构镜像
docker buildx build --platform linux/amd64,linux/arm64 -t flappy-bird-dqn:multi .

# 推送多架构镜像
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/flappy-bird-dqn:multi --push .
```

#### 镜像安全扫描
```bash
# 使用 Docker Scout 扫描漏洞
docker scout quickview flappy-bird-dqn

# 详细漏洞报告
docker scout cves flappy-bird-dqn
```

## 最佳实践和建议

### 1. 镜像命名规范
```bash
# 推荐命名格式
[registry]/[namespace]/[repository]:[tag]

# 示例
docker.io/mycompany/flappy-bird-dqn:v1.0
registry.mycompany.com/ai/flappy-bird-dqn:stable
localhost:5000/dev/flappy-bird-dqn:latest
```

### 2. 版本管理策略
```bash
# 语义化版本
flappy-bird-dqn:1.0.0      # 主版本
flappy-bird-dqn:1.0.1      # 补丁版本
flappy-bird-dqn:latest     # 最新版本
flappy-bird-dqn:stable     # 稳定版本
flappy-bird-dqn:dev        # 开发版本
```

### 3. 自动化工作流
```yaml
# .github/workflows/docker.yml (GitHub Actions)
name: Build and Push Docker Image
on:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and push
        run: |
          docker build -t ${{ github.repository }}:${{ github.ref_name }} .
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ${{ github.repository }}:${{ github.ref_name }}
```

通过这套完整的 Docker 管理体系，你可以高效地开发、部署和分享 Flappy Bird DQN 项目，无论是个人学习还是团队协作都能游刃有余。