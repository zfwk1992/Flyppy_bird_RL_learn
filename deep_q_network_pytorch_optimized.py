#!/usr/bin/env python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import sys
import game.wrapped_flappy_bird_fast as game  
import random
import numpy as np
from collections import deque
import os
import logging
from datetime import datetime

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_optimized_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件创建: {log_filename}")
    return log_filename

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)

# 🚀 优化网络配置参数
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 5000        #观察步数
EXPLORE = 25000     # 探索步数
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 20000 # 适中的经验池
BATCH = 64           # 增大批次提高效率
FRAME_PER_ACTION = 2  # 每1帧一次动作，平衡速度和质量

class OptimizedDQN(nn.Module):
    """优化版深度Q网络 - 解决维度问题并提升性能"""
    def __init__(self, actions):
        super(OptimizedDQN, self).__init__()
        
        # 更小的卷积网络
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        
        # 批归一化层 - 加速训练
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 自适应池化 - 确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层规模减小
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  # 只保留卷积输出
        self.fc2 = nn.Linear(1024, actions)

        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """优化的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x: 图像输入
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x_flat))
        x = self.fc2(x)
        return x

class OptimizedDQNAgent:
    """优化版DQN智能体"""
    def __init__(self, actions):
        self.actions = actions
        self.device = device
        
        # 创建优化网络
        self.q_network = OptimizedDQN(actions).to(device)
        self.target_network = OptimizedDQN(actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 🎯 优化器配置
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=2e-4,  # 稍高的学习率
            weight_decay=1e-4,  # 权重衰减
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器 - 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=EXPLORE, 
            eta_min=1e-6
        )
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=REPLAY_MEMORY)
        self.trajectory_buffer = []  # 新增轨迹缓冲区
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0
        
        # 性能监控
        self.loss_history = []
        self.reward_history = []
        
    def preprocess_state(self, state):
        """优化的状态预处理"""
        # 转换为灰度图
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        # 二值化
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        # 归一化
        state = state.astype(np.float32) / 255.0
        return state
    
    def get_state_tensor(self, state_stack):
        """将状态堆栈转换为tensor"""
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        return state_tensor
    
    def select_action(self, state_tensor):
        """优化的动作选择策略"""
        if self.step <= OBSERVE:
            # 观察阶段：纯随机动作
            action = random.randrange(self.actions)
            if self.step % 100 == 0:
                logging.info(f"👀 观察阶段随机动作: {action} (ε=1.0)")
        elif random.random() <= self.epsilon:
            # 探索阶段：随机动作
            action = random.randrange(self.actions)
            if self.step % 100 == 0:
                logging.info(f"🎲 探索阶段随机动作: {action} (ε={self.epsilon:.4f})")
        else:
            # 利用阶段：网络决策
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                if self.step % 100 == 0:
                    logging.info(f"🧠 网络决策: {action} (Q值: {q_values.max().item():.4f})")
        
        return action
    
    def store_transition(self, state, action, reward, next_state, terminal):
        """存储经验"""
        self.trajectory_buffer.append((state, action, reward, next_state, terminal))
        if len(self.trajectory_buffer) == 50:  # 这里改为50
            self.memory.append(list(self.trajectory_buffer))  # 存一段连续轨迹
            self.trajectory_buffer.pop(0)  # 滑动窗口
        if terminal:
            self.trajectory_buffer.clear()  # 游戏结束清空
    
    def train(self):
        """优化的训练方法"""
        if len(self.memory) < BATCH:
            return

        batch = random.sample(self.memory, BATCH)  # batch: [BATCH, 50, 5]

        # 只取每段轨迹的首状态、首动作、累计奖励、最后一个next_state、最后一个terminal
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        for traj in batch:
            states.append(traj[0][0])          # 首状态
            actions.append(traj[0][1])         # 首动作
            # n步累计奖励
            R = 0
            for i in range(len(traj)):
                R += (GAMMA ** i) * traj[i][2]
            rewards.append(R)
            next_states.append(traj[-1][3])    # 最后一步的 next_state
            terminals.append(traj[-1][4])      # 最后一步的 terminal

        # 转为tensor
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        terminals = torch.BoolTensor(terminals).to(device)

        # 调整维度顺序
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值 - 使用Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + (GAMMA ** len(traj)) * next_q_values.squeeze() * ~terminals

        loss = F.huber_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.loss_history.append(loss.item())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss.item()
    
    def update_target_network(self):
        """软更新目标网络"""
        tau = 0.001  # 软更新参数
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def update_epsilon(self):
        """更新探索率"""
        if self.step <= OBSERVE:
            self.epsilon = 1.0
        elif self.epsilon > FINAL_EPSILON and self.step > OBSERVE:
            self.epsilon -= (1.0 - FINAL_EPSILON) / EXPLORE
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step': self.step,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step = checkpoint['step']
            self.loss_history = checkpoint.get('loss_history', [])
            self.reward_history = checkpoint.get('reward_history', [])
            logging.info(f"成功加载模型: {path}")
            return True
        else:
            logging.warning("未找到预训练模型")
            return False

def train_network():
    """训练网络"""
    # 设置日志记录
    log_filename = setup_logging()
    logging.info("🚀 开始Flappy Bird AI优化版网络训练")
    logging.info("=" * 60)
    
    # 显示设备信息
    logging.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    else:
        logging.warning("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 显示优化版网络配置参数
    logging.info("⚡ 优化版网络配置参数:")
    logging.info(f"   - FRAME_PER_ACTION: {FRAME_PER_ACTION}")
    logging.info(f"   - OBSERVE: {OBSERVE}")
    logging.info(f"   - EXPLORE: {EXPLORE}")
    logging.info(f"   - BATCH: {BATCH}")
    logging.info(f"   - 网络架构: 优化版 (自适应池化 + 残差连接 + Double DQN)")
    logging.info(f"   - 优化器: AdamW + 余弦退火学习率")
    logging.info(f"   - 损失函数: Huber Loss")
    logging.info(f"   - 预期速度: ~40步/秒")
    logging.info(f"   - 预期时间: ~12分钟")
    
    # 创建智能体
    agent = OptimizedDQNAgent(ACTIONS)
    
    # 验证模型是否在GPU上
    if torch.cuda.is_available():
        q_network_device = next(agent.q_network.parameters()).device
        target_network_device = next(agent.target_network.parameters()).device
        logging.info(f"Q网络设备: {q_network_device}")
        logging.info(f"目标网络设备: {target_network_device}")
        
        if q_network_device.type != 'cuda' or target_network_device.type != 'cuda':
            logging.error("❌ 模型未正确移到GPU！")
            agent.q_network = agent.q_network.to(device)
            agent.target_network = agent.target_network.to(device)
            logging.info("✅ 已强制将模型移到GPU")
    
    # 创建游戏环境
    game_state = game.GameState()
    
    # 创建日志文件
    os.makedirs(f"logs_{GAME}", exist_ok=True)
    a_file = open(f"logs_{GAME}/readout.txt", 'w')
    h_file = open(f"logs_{GAME}/hidden.txt", 'w')
    
    # 初始化状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = agent.preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
    # 尝试加载预训练模型
    model_path = "saved_networks/bird-dqn-pytorch-optimized.pth"
    agent.load_model(model_path)
    
    # 帧计数器
    frame_count = 0
    episode_reward = 0
    episode_count = 0
    
    # 训练循环
    while "flappy bird" != "angry bird":
        # 获取状态tensor
        state_tensor = agent.get_state_tensor(s_t)
        
        # 每FRAME_PER_ACTION帧采取一次动作
        if frame_count % FRAME_PER_ACTION == 0:
            # 选择动作
            action_index = agent.select_action(state_tensor)
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1
        else:
            # 其他帧保持上一个动作
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1
        
        # 执行动作
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = agent.preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        # 累积奖励
        episode_reward += r_t
        
        # 存储经验（只在采取动作的帧存储）
        if frame_count % FRAME_PER_ACTION == 0:
            agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
            agent.step += 1
        
        # 更新状态
        s_t = s_t1
        frame_count += 1
        
        # 只在采取动作的帧进行训练和日志记录
        if frame_count % FRAME_PER_ACTION == 0:
            # 训练网络
            loss = None
            if agent.step > OBSERVE:
                loss = agent.train()
            
            # 软更新目标网络（每步更新）
            if agent.step > OBSERVE:
                agent.update_target_network()
            
            # 更新探索率
            agent.update_epsilon()
            
            # 保存模型（每5000步）
            if agent.step % 5000 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                agent.save_model(f"saved_networks/{GAME}-dqn-pytorch-optimized-{agent.step}.pth")
            
            # 打印训练信息
            if agent.step % 100 == 0:
                state = ""
                if agent.step <= OBSERVE:
                    state = "observe"
                elif agent.step > OBSERVE and agent.step <= OBSERVE + EXPLORE:
                    state = "explore"
                else:
                    state = "train"
                
                q_max = agent.q_network(state_tensor).max().item() if loss is not None else 0
                current_lr = agent.scheduler.get_last_lr()[0]
                
                logging.info(f"⏱️  TIMESTEP {agent.step} / FRAME {frame_count} / STATE {state} / EPSILON {agent.epsilon:.4f} / "
                      f"ACTION {action_index} / REWARD {r_t} / Q_MAX {q_max:.4f} / LR {current_lr:.2e}")
                
                if loss is not None:
                    logging.info(f"📉 LOSS: {loss:.6f}")
            
            # 每1000步显示统计信息
            if agent.step % 1000 == 0:
                avg_loss = np.mean(agent.loss_history[-100:]) if agent.loss_history else 0
                logging.info(f"📊 统计: 经验池大小={len(agent.memory)}, 探索率={agent.epsilon:.4f}, 平均损失={avg_loss:.6f}")
                
                # 显示GPU内存使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    logging.info(f"🖥️  GPU内存: {gpu_memory:.1f} MB / {gpu_memory_reserved:.1f} MB / {gpu_memory_total:.1f} GB")
                
                logging.info("-" * 50)
        
        # 游戏结束处理
        if terminal:
            episode_count += 1
            agent.reward_history.append(episode_reward)
            episode_reward = 0

def play_game():
    """主函数"""
    train_network()

if __name__ == "__main__":
    play_game()