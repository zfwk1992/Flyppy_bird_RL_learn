# Flappy Bird DQN Docker Environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装核心系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements_cuda.txt ./

# 安装 Python 依赖（GPU 版本）
RUN pip3 install --no-cache-dir -r requirements_cuda.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs saved_networks

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

# 暴露端口（如果需要 Jupyter）
EXPOSE 8888

# 默认命令
CMD ["python3", "deep_Q_oneStep.py"]