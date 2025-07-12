#!/usr/bin/env python
# 修复版本：单步预测DQN训练脚本

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import sys
import random
import numpy as np
from collections import deque
import os
import logging
from datetime import datetime

# 添加游戏路径
sys.path.append("game/")
import game.wrapped_flappy_bird_fast as game

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)

# 超参数配置
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 3000
EXPLORE = 20000
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 20000
BATCH = 64
FRAME_PER_ACTION = 2

# 设置日志
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_oneStep_{timestamp}.log"
    
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
# 该文件使用FixedOptimizedDQN网络进行单步预测训练
# 避免了复杂的多步预测时序问题，采用经典DQN架构


class FixedOptimizedDQN(nn.Module):
    """修复的网络结构：输出单步Q值而非多步"""
    def __init__(self, actions):
        super(FixedOptimizedDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 修改：只输出单步Q值
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, actions)  # 简化输出
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # [batch, actions]


class DQNAgent:
    """DQN智能体"""
    def __init__(self, actions):
        self.actions = actions
        self.device = device
        
        # 创建网络
        self.q_network = FixedOptimizedDQN(actions).to(device)
        self.target_network = FixedOptimizedDQN(actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=REPLAY_MEMORY)
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0
        self.reward_history = []
        
    def preprocess_state(self, state):
        """预处理状态"""
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return state
    
    def get_state_tensor(self, state_stack):
        """将状态堆栈转换为tensor"""
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
        return state_tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
    
    def select_action(self, state_tensor):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练网络"""
        if len(self.memory) < BATCH:
            return None
        
        # 随机采样
        batch = random.sample(self.memory, BATCH)
        states = torch.FloatTensor([e[0] for e in batch]).to(device).permute(0, 3, 1, 2)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device).permute(0, 3, 1, 2)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (GAMMA * next_q * ~dones.unsqueeze(1))
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """软更新目标网络"""
        if self.step % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """更新探索率"""
        if self.step < OBSERVE:
            self.epsilon = 1.0
        elif self.step < OBSERVE + EXPLORE:
            self.epsilon = 1.0 - (self.step - OBSERVE) / EXPLORE * (1.0 - FINAL_EPSILON)
        else:
            self.epsilon = FINAL_EPSILON


def main():
    """主训练函数"""
    # 设置日志
    log_file = setup_logging()
    
    # 显示设备信息
    logging.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA版本: {torch.version.cuda}")
    
    # 初始化游戏环境
    game_state = game.GameState()
    
    # 初始化智能体
    agent = DQNAgent(ACTIONS)
    
    # 获取初始状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = agent.preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
    # 训练统计
    episode_count = 0
    episode_reward = 0
    max_score = 0
    action_index = 0
    
    logging.info("开始训练...")
    logging.info(f"观察步数: {OBSERVE}, 探索步数: {EXPLORE}, 批次大小: {BATCH}")
    
    while True:
        # 每FRAME_PER_ACTION帧做一次决策
        if agent.step % FRAME_PER_ACTION == 0:
            state_tensor = agent.get_state_tensor(s_t)
            action_index = agent.select_action(state_tensor)
        
        # 构建动作向量
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1
        
        # 执行动作
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = agent.preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        episode_reward += r_t
        
        # 只在决策帧存储经验
        if agent.step % FRAME_PER_ACTION == 0:
            agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
            agent.step += 1
            
            # 训练网络
            if agent.step > OBSERVE:
                loss = agent.train()
                agent.update_target_network()
                
                # 记录训练信息
                if agent.step % 1000 == 0:
                    avg_reward = np.mean(agent.reward_history[-100:]) if agent.reward_history else 0
                    logging.info(f"步数: {agent.step}, ε: {agent.epsilon:.4f}, "
                               f"损失: {loss:.4f}, 平均奖励: {avg_reward:.2f}, "
                               f"最高分数: {max_score}")
            
            # 更新探索率
            agent.update_epsilon()
        
        # 更新状态
        s_t = s_t1
        
        # 游戏结束处理
        if terminal:
            episode_count += 1
            agent.reward_history.append(episode_reward)
            
            if episode_reward > max_score:
                max_score = episode_reward
                
            logging.info(f"游戏 {episode_count} 结束, 得分: {episode_reward}, "
                        f"最高分: {max_score}, 当前ε: {agent.epsilon:.4f}")
            
            episode_reward = 0
            
            # 保存模型
            if episode_count % 100 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          f"saved_networks/bird-dqn-oneStep-{episode_count}.pth")
                logging.info(f"模型已保存: bird-dqn-oneStep-{episode_count}.pth")
        
        # 阶段提示
        if agent.step == OBSERVE:
            logging.info("观察阶段结束，开始探索和训练...")
        elif agent.step == OBSERVE + EXPLORE:
            logging.info("探索阶段结束，进入稳定训练...")


if __name__ == "__main__":
    main()