#!/bin/bash

# Flappy Bird DQN 训练启动脚本
# 解决 docker-compose ContainerConfig 问题

echo "🐦 Flappy Bird DQN 训练启动脚本"
echo "================================"

# 构建镜像
echo "📦 构建 Flappy Bird DQN 镜像..."
docker build -t flappy-bird-dqn .

if [ $? -ne 0 ]; then
    echo "❌ 镜像构建失败！"
    exit 1
fi

# 清理旧容器
echo "🧹 清理旧容器..."
docker stop flappy_bird_dqn 2>/dev/null || true
docker rm flappy_bird_dqn 2>/dev/null || true

# 确保目录存在
echo "📁 确保输出目录存在..."
mkdir -p logs saved_networks

# 显示配置信息
echo "⚙️  训练配置:"
echo "   - GPU: CUDA_VISIBLE_DEVICES=0"
echo "   - 日志目录: ./logs"
echo "   - 模型保存: ./saved_networks"
echo "   - 游戏资源: ./assets"
echo ""

# 运行训练
echo "🚀 启动训练..."
docker run --gpus all \
    --rm \
    --name flappy_bird_dqn \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/saved_networks:/app/saved_networks \
    -v $(pwd)/assets:/app/assets \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    flappy-bird-dqn

echo ""
echo "✅ 训练完成！"
echo "📊 查看日志: ls -la logs/"
echo "💾 查看模型: ls -la saved_networks/"