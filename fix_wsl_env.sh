#!/bin/bash

echo "🔧 修复WSL环境下的pygame问题"

# 设置显示和音频环境变量
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy  
export XDG_RUNTIME_DIR=/tmp/runtime-dir
export DISPLAY=:0

# 创建运行时目录
mkdir -p /tmp/runtime-dir

# 启动虚拟环境并运行训练
source flappy-bird-env/bin/activate

echo "✅ 环境变量已设置:"
echo "SDL_VIDEODRIVER=$SDL_VIDEODRIVER"
echo "SDL_AUDIODRIVER=$SDL_AUDIODRIVER"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo ""

echo "🚀 开始训练..."
python deep_Q_oneStep.py