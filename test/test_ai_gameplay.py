#!/usr/bin/env python3
"""
Flappy Bird Dueling DQN 推理测试脚本
只用于加载并评估训练好的Dueling DQN模型（RobustDuelingDQN/StableDQNAgent）
"""
import sys
import os
import time
import argparse
import numpy as np
import torch
from datetime import datetime

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
os.chdir(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "game"))

# 导入游戏和Agent
import game.wrapped_flappy_bird as game
from deep_Q_dueling_DQN import StableDQNAgent

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 命令行参数
parser = argparse.ArgumentParser(description="Flappy Bird Dueling DQN 推理测试")
parser.add_argument('--model', type=str, required=True, help='模型文件路径（.pth）')
parser.add_argument('--episodes', type=int, default=5, help='测试局数')
parser.add_argument('--fps', type=int, default=60, help='游戏帧率')
parser.add_argument('--delay', type=float, default=0.01, help='每帧延迟（秒）')
parser.add_argument('--no-q-values', action='store_true', help='不显示Q值')
args = parser.parse_args()

# Agent初始化
agent = StableDQNAgent(actions=2)
if not agent.load_model(args.model):
    print(f"❌ 模型加载失败: {args.model}")
    sys.exit(1)

# 游戏初始化
game_state = game.GameState()
ACTIONS = 2
FRAME_PER_ACTION = 4  # 与训练保持一致

def preprocess_state(state):
    import cv2
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return state

print(f"\n🎮 Flappy Bird Dueling DQN 推理测试 | 模型: {args.model}")
print(f"测试局数: {args.episodes} | 帧率: {args.fps} | 显示Q值: {'否' if args.no_q_values else '是'}")

for episode in range(args.episodes):
    print(f"\n🚀 第 {episode+1} 局开始")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    step = 0
    episode_score = 0
    action_history = []
    start_time = time.time()
    while not terminal:
        if step % FRAME_PER_ACTION == 0:
            # [H,W,C] -> [1,C,H,W]
            state_tensor = torch.FloatTensor(s_t).unsqueeze(0).to(DEVICE)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            action = agent.select_action(s_t)
            # Q值显示（可选）
            if not args.no_q_values and step % (FRAME_PER_ACTION*5) == 0:
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)
                    q_vals = q_values.squeeze().cpu().numpy()
                    print(f"步{step:3d} | Q值: [{q_vals[0]:6.2f}, {q_vals[1]:6.2f}] | 动作: {'跳跃' if action==1 else '不跳'}")
        a_t = np.zeros([ACTIONS])
        a_t[action] = 1
        x_t1_colored, reward, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        s_t = s_t1
        episode_score += reward
        action_history.append(action)
        step += 1
        time.sleep(args.delay)
    duration = time.time() - start_time
    print(f"🏁 第{episode+1}局结束 | 得分: {episode_score:.2f} | 步数: {step} | 时长: {duration:.1f}s")
    jump_ratio = sum(action_history)/len(action_history) if action_history else 0
    print(f"   跳跃比例: {jump_ratio*100:.1f}% | 不跳: {(1-jump_ratio)*100:.1f}%")
    if episode < args.episodes-1:
        print("⏱️  3秒后开始下一局...")
        time.sleep(3)
print("\n✅ 测试完成！")