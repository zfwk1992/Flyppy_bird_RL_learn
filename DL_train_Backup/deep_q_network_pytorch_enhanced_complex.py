#!/usr/bin/env python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import sys
sys.path.append("game/")
import game.wrapped_flappy_bird_fast as game  # 使用优化游戏版本
import random
import numpy as np
from collections import deque
import os
import logging
from datetime import datetime

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_enhanced_complex_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"日志文件创建: {log_filename}")
    return log_filename

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 显示GPU详细信息（在日志配置后显示）
if torch.cuda.is_available():
    # 设置GPU内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存

# 🚀 增强复杂度网络配置参数 (通过游戏优化提升速度)
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 3000        # 适度观察步数
EXPLORE = 30000       # 适度探索步数
FINAL_EPSILON = 0.001  # 最终探索率
REPLAY_MEMORY = 25000  # 较大经验池
BATCH = 32           # 较大批次大小
FRAME_PER_ACTION = 2  # 每2帧一次动作 (平衡速度和质量)

class EnhancedDQN(nn.Module):
    """增强复杂度深度Q网络"""
    def __init__(self, actions):
        super(EnhancedDQN, self).__init__()
        
        # 增强卷积层 - 增加通道数和层数
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=4)   # 增加通道数
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # 增加通道数
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1) # 新增卷积层
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 增强全连接层 - 增加神经元数和层数
        self.fc1 = nn.Linear(64, 1024)  # 增加神经元数
        self.fc2 = nn.Linear(1024, 512) # 新增隐藏层
        self.fc3 = nn.Linear(512, 256)  # 新增隐藏层
        self.fc4 = nn.Linear(256, actions)
        
        # 批归一化层
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)
        
        # Dropout层
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 增强卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.bn4(self.conv4(x)))  # 新增卷积层
        
        # 展平
        x = x.view(x.size(0), -1)  # 展平为64维
        
        # 增强全连接层
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

class DQNAgent:
    """DQN智能体"""
    def __init__(self, actions):
        self.actions = actions
        self.device = device
        
        # 创建增强网络
        self.q_network = EnhancedDQN(actions).to(device)
        self.target_network = EnhancedDQN(actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-6, weight_decay=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=REPLAY_MEMORY)
        
        # 训练参数
        self.epsilon = 1.0  # 初始为纯随机
        self.step = 0
        
    def preprocess_state(self, state):
        """预处理状态"""
        # 转换为灰度图
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        # 二值化
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return state
    
    def get_state_tensor(self, state_stack):
        """将状态堆栈转换为tensor"""
        # 转换为tensor并添加batch维度
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
        # 调整维度顺序 (batch, height, width, channels) -> (batch, channels, height, width)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        return state_tensor
    
    def select_action(self, state_tensor):
        """选择动作（ε-贪婪策略）"""
        if self.step <= OBSERVE:
            # 观察阶段：纯随机动作
            action = random.randrange(self.actions)
            if self.step % 100 == 0:  # 保持日志频率
                logging.info(f"👀 观察阶段随机动作: {action} (ε=1.0)")
        elif random.random() <= self.epsilon:
            # 探索阶段：随机动作
            action = random.randrange(self.actions)
            if self.step % 100 == 0:  # 保持日志频率
                logging.info(f"🎲 探索阶段随机动作: {action} (ε={self.epsilon:.4f})")
        else:
            # 利用阶段：网络决策
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                if self.step % 100 == 0:  # 保持日志频率
                    logging.info(f"🧠 网络决策: {action} (Q值: {q_values.max().item():.4f})")
        
        return action
    
    def store_transition(self, state, action, reward, next_state, terminal):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, terminal))
    
    def train(self):
        """训练网络"""
        if len(self.memory) < BATCH:
            return
        
        # 随机采样批次
        batch = random.sample(self.memory, BATCH)
        
        # 分离批次数据并移到GPU
        states = torch.FloatTensor([d[0] for d in batch]).to(device, non_blocking=True)
        actions = torch.LongTensor([d[1] for d in batch]).to(device, non_blocking=True)
        rewards = torch.FloatTensor([d[2] for d in batch]).to(device, non_blocking=True)
        next_states = torch.FloatTensor([d[3] for d in batch]).to(device, non_blocking=True)
        terminals = torch.BoolTensor([d[4] for d in batch]).to(device, non_blocking=True)
        
        # 调整维度顺序
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~terminals)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 清理GPU缓存（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """更新探索率"""
        if self.step <= OBSERVE:
            # 观察阶段：保持纯随机
            self.epsilon = 1.0
        elif self.epsilon > FINAL_EPSILON and self.step > OBSERVE:
            # 探索阶段：从1.0逐渐降低到FINAL_EPSILON
            self.epsilon -= (1.0 - FINAL_EPSILON) / EXPLORE
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step': self.step
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
            logging.info(f"成功加载模型: {path}")
            return True
        else:
            logging.warning("未找到预训练模型")
            return False

def train_network():
    """训练网络"""
    # 设置日志记录
    log_filename = setup_logging()
    logging.info("🚀 开始Flappy Bird AI增强复杂度网络训练")
    logging.info("=" * 60)
    
    # 显示设备信息
    logging.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logging.info(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    else:
        logging.warning("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 显示增强复杂度网络配置参数
    logging.info("⚡ 增强复杂度网络配置参数:")
    logging.info(f"   - FRAME_PER_ACTION: {FRAME_PER_ACTION}")
    logging.info(f"   - OBSERVE: {OBSERVE}")
    logging.info(f"   - EXPLORE: {EXPLORE}")
    logging.info(f"   - BATCH: {BATCH}")
    logging.info(f"   - 网络复杂度: 增强版 (更多层、更多神经元、批归一化、Dropout)")
    logging.info(f"   - 预期速度: ~30步/秒 (通过游戏优化)")
    logging.info(f"   - 预期时间: ~18分钟")
    logging.info(f"   - 使用优化游戏版本")
    
    # 创建智能体
    agent = DQNAgent(ACTIONS)
    
    # 验证模型是否在GPU上
    if torch.cuda.is_available():
        q_network_device = next(agent.q_network.parameters()).device
        target_network_device = next(agent.target_network.parameters()).device
        logging.info(f"Q网络设备: {q_network_device}")
        logging.info(f"目标网络设备: {target_network_device}")
        
        if q_network_device.type != 'cuda' or target_network_device.type != 'cuda':
            logging.error("❌ 模型未正确移到GPU！")
            # 强制移到GPU
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
    model_path = "saved_networks/bird-dqn-pytorch-enhanced-complex.pth"
    agent.load_model(model_path)
    
    # 帧计数器
    frame_count = 0
    
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
            
            # 更新目标网络（每5000步）
            if agent.step % 5000 == 0:
                agent.update_target_network()
            
            # 更新探索率
            agent.update_epsilon()
            
            # 保存模型（每5000步）
            if agent.step % 5000 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                agent.save_model(f"saved_networks/{GAME}-dqn-pytorch-enhanced-complex-{agent.step}.pth")
            
            # 打印训练信息（保持日志频率）
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
                logging.info(f"📊 统计: 经验池大小={len(agent.memory)}, 探索率={agent.epsilon:.4f}")
                
                # 显示GPU内存使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    logging.info(f"🖥️  GPU内存: {gpu_memory:.1f} MB / {gpu_memory_reserved:.1f} MB / {gpu_memory_total:.1f} GB")
                    
                    # 检查模型参数是否在GPU上
                    q_network_device = next(agent.q_network.parameters()).device
                    logging.info(f"🔍 Q网络设备: {q_network_device}")
                
                logging.info("-" * 50)

def play_game():
    """主函数"""
    train_network()

if __name__ == "__main__":
    play_game() 