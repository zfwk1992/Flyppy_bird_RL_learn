#!/bin/bash

echo "🐋 从Docker容器复制模型脚本"
echo "=========================="

# 查找正在运行的容器
echo "🔍 查找正在运行的Docker容器..."
docker ps

# 获取第一个运行中的容器ID
CONTAINER_ID=$(docker ps -q | head -1)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ 没有找到正在运行的容器"
    exit 1
fi

echo ""
echo "🎯 使用容器ID: $CONTAINER_ID"

# 检查容器中的模型文件
echo ""
echo "🔍 检查容器中的模型文件..."
docker exec $CONTAINER_ID ls -la /app/saved_networks/

# 复制模型文件
echo ""
echo "🚀 复制模型文件到宿主机..."
docker cp $CONTAINER_ID:/app/saved_networks/. ./saved_networks/

# 验证复制结果
echo ""
echo "✅ 复制完成！宿主机模型文件:"
ls -la ./saved_networks/

echo ""
echo "🎉 现在可以运行测试: python run_test.py"