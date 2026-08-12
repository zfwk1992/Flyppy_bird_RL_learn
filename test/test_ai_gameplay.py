#!/usr/bin/env python3
"""
Flappy Bird Dueling DQN 推理测试脚本
用于加载并评估训练好的改进版Dueling DQN模型
完全匹配训练脚本的逻辑
"""
import sys
import os
import time
import argparse
import numpy as np
import torch
import cv2
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) #if os.path.basename(current_dir) != project_root else current_dir
os.chdir(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "game"))

# 导入游戏
import game.wrapped_flappy_bird as game
#import game.wrapped_flappy_bird_fast as game

# 导入改进版的模型类
try:
    from improved_dueling_dqn import StableDQNAgent, RobustDuelingDQN
except ImportError:
    # 如果找不到改进版，尝试导入原始版本
    from deep_Q_dueling_DQN import StableDQNAgent, RobustDuelingDQN

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 命令行参数
parser = argparse.ArgumentParser(description="Flappy Bird Dueling DQN 推理测试")
parser.add_argument('--model', type=str, required=True, help='模型文件路径（.pth）')
parser.add_argument('--episodes', type=int, default=5, help='测试局数')
parser.add_argument('--fps', type=int, default=500, help='游戏帧率')
parser.add_argument('--delay', type=float, default=0.01, help='每帧延迟（秒）')
parser.add_argument('--no-q-values', action='store_true', help='不显示Q值')
parser.add_argument('--epsilon', type=float, default=0.0, help='测试时的epsilon（默认0为纯利用）')
parser.add_argument('--verbose', action='store_true', help='显示详细信息')
args = parser.parse_args()

# Agent初始化
print(f"🔧 初始化Agent...")
agent = StableDQNAgent(actions=2)

# 设置测试模式的epsilon
agent.epsilon = args.epsilon
agent.epsilon_min = args.epsilon  # 防止epsilon衰减

# 加载模型
if not agent.load_model(args.model):
    print(f"❌ 模型加载失败: {args.model}")
    sys.exit(1)

# 将网络设置为评估模式
agent.q_network.eval()
agent.target_network.eval()

# 游戏初始化
game_state = game.GameState()
ACTIONS = 2
TRAINING_FREQ = agent.training_freq  # 使用agent的训练频率

print(f"\n🎮 Flappy Bird Dueling DQN 推理测试")
print(f"模型: {args.model}")
print(f"测试局数: {args.episodes} | 帧率: {args.fps} | 显示Q值: {'否' if args.no_q_values else '是'}")
print(f"决策频率: 每{TRAINING_FREQ}帧决策一次 | 非决策帧: 强制不跳跃")
print(f"设备: {DEVICE} | Epsilon: {agent.epsilon}")
print("-" * 60)

# 开始测试循环
total_scores = []
total_decision_count = 0
total_steps = 0
all_q_values = []

for episode in range(args.episodes):
    print(f"\n🚀 第 {episode+1}/{args.episodes} 局开始")
    
    # 初始化游戏状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    
    # 使用agent的预处理方法
    x_t = agent.preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
    # 局内统计变量
    step = 0
    decision_step = 0
    episode_score = 0
    action_history = []
    decision_history = []
    q_value_history = []
    start_time = time.time()
    current_action = 0  # 当前执行的动作
    
    while not terminal:
        # 🎯 核心逻辑：完全匹配训练脚本的决策模式
        #print(f"决策_step{decision_step:3d} | step{step:3d}")
        if step % TRAINING_FREQ == 0:
            # 决策帧：使用模型选择动作
            with torch.no_grad():
                current_action = agent.select_action(s_t)
           # print(f"进行决策_step{decision_step:3d} | step{step:3d}| 选择动作: {current_action}")
            decision_step += 1
            decision_history.append(current_action)
            
            # 获取并记录Q值
            if not args.no_q_values or args.verbose:
                try:
                    state_tensor = agent.get_state_tensor(s_t)
                    with torch.no_grad():
                        q_values = agent.q_network(state_tensor)
                        q_vals = q_values.squeeze().cpu().numpy()
                        q_value_history.append(q_vals)
                        all_q_values.append(q_vals)
                        
                        # 显示Q值（根据参数）
                        if not args.no_q_values and (decision_step % 5 == 0 or args.verbose):
                            action_str = '跳跃' if current_action == 1 else '不跳'
                            q_diff = q_vals[1] - q_vals[0]
                           # print(f"决策{decision_step:3d} | Q值: [不跳:{q_vals[0]:6.2f}, 跳跃:{q_vals[1]:6.2f}] | 差值:{q_diff:6.2f} | 选择: {action_str}")
                except Exception as e:
                    if args.verbose:
                        print(f"Q值计算错误: {e}")
            
            # 执行模型选择的动作
            a_t = np.zeros([ACTIONS])
            a_t[current_action] = 1
        else:
            # 🔥 非决策帧：强制不跳跃（完全匹配训练脚本）
            a_t = np.array([1, 0])  # 强制不跳跃
            current_action = 0  # 记录为不跳跃
        
        # 执行动作并获取下一个状态
        print(f"a_t不跳{a_t[0]}| 跳 {a_t[1]} | 当前动作: {current_action}")
        x_t1_colored, reward, terminal = game_state.frame_step(a_t)
        x_t1 = agent.preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.concatenate([x_t1, s_t[:, :, :3]], axis=2)
        
        # 更新状态
        s_t = s_t1
        episode_score += reward
        action_history.append(current_action)
        step += 1
        
        # 控制游戏速度
        time.sleep(args.delay)
    
    # 统计这一局的结果
    duration = time.time() - start_time
    total_scores.append(episode_score)
    total_decision_count += decision_step
    total_steps += step
    
    # 计算动作统计
    jump_ratio = sum(action_history) / len(action_history) if action_history else 0
    decision_jump_ratio = sum(decision_history) / len(decision_history) if decision_history else 0
    
    # 计算Q值统计
    if q_value_history:
        q_array = np.array(q_value_history)
        mean_q = np.mean(q_array, axis=0)
        std_q = np.std(q_array, axis=0)
        mean_q_diff = np.mean(q_array[:, 1] - q_array[:, 0])
    
    print(f"\n🏁 第{episode+1}局结束")
    print(f"   得分: {episode_score:.1f} | 总步数: {step} | 决策数: {decision_step}")
    print(f"   时长: {duration:.1f}秒 | 平均FPS: {step/duration:.1f}")
    print(f"   跳跃统计: 所有帧{jump_ratio*100:.1f}% | 决策帧{decision_jump_ratio*100:.1f}%")
    print(f"   决策频率: 每{step/decision_step:.1f}帧一次 (目标: {TRAINING_FREQ}帧)")
    
    if q_value_history and args.verbose:
        print(f"   Q值统计:")
        print(f"     - 平均Q(不跳): {mean_q[0]:.2f} ± {std_q[0]:.2f}")
        print(f"     - 平均Q(跳跃): {mean_q[1]:.2f} ± {std_q[1]:.2f}")
        print(f"     - 平均Q差值: {mean_q_diff:.2f}")
    
    if episode < args.episodes - 1:
        print("\n⏱️  3秒后开始下一局...")
        time.sleep(3)

# 整体测试结果
print(f"\n{'='*60}")
print(f"📊 测试完成！总共{args.episodes}局")
print(f"{'='*60}")

# 分数统计
print(f"\n📈 分数统计:")
print(f"   平均得分: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")
print(f"   最高得分: {max(total_scores):.1f}")
print(f"   最低得分: {min(total_scores):.1f}")
print(f"   中位数: {np.median(total_scores):.1f}")

# 动作统计
print(f"\n🎮 动作统计:")
print(f"   总决策数: {total_decision_count}")
print(f"   总步数: {total_steps}")
print(f"   平均决策频率: 每{total_steps/total_decision_count:.2f}帧 (目标: {TRAINING_FREQ}帧)")

# Q值总体统计
if all_q_values:
    all_q_array = np.array(all_q_values)
    overall_mean_q = np.mean(all_q_array, axis=0)
    overall_std_q = np.std(all_q_array, axis=0)
    overall_mean_diff = np.mean(all_q_array[:, 1] - all_q_array[:, 0])
    
    print(f"\n🧠 Q值总体统计:")
    print(f"   平均Q(不跳): {overall_mean_q[0]:.2f} ± {overall_std_q[0]:.2f}")
    print(f"   平均Q(跳跃): {overall_mean_q[1]:.2f} ± {overall_std_q[1]:.2f}")
    print(f"   平均Q差值: {overall_mean_diff:.2f}")
    print(f"   Q值稳定性: {1/(overall_std_q.mean()+1e-6):.2f}")

# 模型信息
print(f"\n🤖 模型信息:")
print(f"   模型路径: {args.model}")
print(f"   设备: {DEVICE}")
print(f"   Epsilon: {agent.epsilon}")
print(f"   批次大小: {agent.batch_size}")

# 保存测试结果（可选）
if args.verbose:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"test_results_{timestamp}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"测试局数: {args.episodes}\n")
        f.write(f"平均得分: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}\n")
        f.write(f"最高得分: {max(total_scores):.1f}\n")
        f.write(f"各局得分: {total_scores}\n")
    print(f"\n💾 测试结果已保存到: {result_file}")

print(f"\n✅ 测试完成！")