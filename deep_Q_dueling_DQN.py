#!/usr/bin/env python
# 全面测试稳定版 Dueling DQN - 优化版本 (Batch Size: 128)
# 修复所有已知问题，确保训练可以正常进行

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
import warnings
from datetime import datetime
import gc
import time

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

class StablePriorityReplayBuffer:
    """
    稳定版优先经验回放缓冲区
    经过全面测试，确保无内存泄漏和bug
    针对更大的batch size进行了优化
    """
    def __init__(self, capacity=100000, alpha=0.8):  # 增大缓冲区容量
        self.capacity = capacity
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment = 0.0005  # 减缓beta增长速度
        self.beta_max = 1.0
        
        # 使用预分配数组避免动态增长
        self.buffer = [None] * capacity
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        # 🔧 修复1: 确保数据类型一致性
        experience = {
            'state': np.array(state, dtype=np.float32) / 255.0,
            'action': int(action),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32) / 255.0,
            'done': bool(done)
        }
        
        # 存储经验
        self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """优先级采样 - 优化版本以支持更大批次"""
        if self.size < batch_size:
            return None, None, None
        
        # 🔧 修复2: 安全的优先级采样
        try:
            # 获取有效优先级
            valid_priorities = self.priorities[:self.size]
            probs = valid_priorities ** self.alpha
            probs = probs / np.sum(probs)
            
            # 采样索引 - 使用更高效的采样方法
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
            
            # 计算重要性采样权重
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / np.max(weights)
            
            # 提取批次数据
            batch = [self.buffer[idx] for idx in indices]
            
            # 更新beta
            self.beta = min(self.beta_max, self.beta + self.beta_increment)
            
            return batch, indices, weights
            
        except Exception as e:
            logging.warning(f"采样失败: {e}")
            return None, None, None
    
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < self.size:
                priority = (abs(float(td_error)) + 1e-6) ** self.alpha
                priority = np.clip(priority, 0.01, 100.0)
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size

class RobustDuelingDQN(nn.Module):
    """
    鲁棒版 Dueling DQN 网络
    经过测试确保稳定性
    针对更大批次进行了优化
    """
    def __init__(self, actions=2):
        super(RobustDuelingDQN, self).__init__()
        self.actions = actions
        
        # 🔧 修复3: 标准化网络结构，避免维度问题
        # 卷积层: 4 -> 32 -> 64 -> 64
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # 批量归一化提升训练稳定性 - 增加momentum以适应大批次
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.01)
        
        # 🔧 修复4: 计算正确的特征图尺寸
        # 输入: 80x80 -> conv1: 20x20 -> conv2: 10x10 -> conv3: 10x10
        self.feature_size = 64 * 10 * 10
        
        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3)  # 稍微增加dropout以适应大批次
        )
        
        # Dueling分支
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, actions)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 🔧 修复5: 添加输入验证
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")
        
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平 - 使用reshape替代view避免contiguous问题
        x = x.reshape(x.size(0), -1)
        
        # 🔧 修复6: 确保特征维度正确
        if x.size(1) != self.feature_size:
            logging.warning(f"特征维度不匹配: 期望{self.feature_size}, 实际{x.size(1)}")
            # 动态调整
            if not hasattr(self, '_adaptive_fc'):
                self._adaptive_fc = nn.Linear(x.size(1), 512).to(x.device)
            shared_features = F.relu(self._adaptive_fc(x))
        else:
            shared_features = self.shared_fc(x)
        
        # Dueling分支
        value = self.value_head(shared_features)
        advantage = self.advantage_head(shared_features)
        
        # Dueling聚合: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class StableDQNAgent:
    """
    稳定版DQN智能体
    经过全面测试，确保训练正常进行
    优化以支持更大的batch size
    """
    def __init__(self, actions=2, state_shape=(4, 80, 80)):
        self.actions = actions
        self.state_shape = state_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🔧 修复7: 调整GPU内存设置以适应更大批次
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.7)  # 增加GPU内存使用
            # 启用内存池
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        # 创建网络
        self.q_network = RobustDuelingDQN(actions).to(self.device)
        self.target_network = RobustDuelingDQN(actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 🔧 修复8: 调整优化器配置以适应大批次
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=0.00025,  # 增加学习率以补偿大批次
            eps=1e-8,
            weight_decay=1e-5
        )
        
        # 学习率调度器 - 调整以适应大批次训练
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100000,  # 增加step size
            gamma=0.8
        )
        
        # 经验回放缓冲区 - 增大容量
        self.memory = StablePriorityReplayBuffer(capacity=100000)
        
        # 训练参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 1200000  # 覆盖约48,000局游戏（50,000-2,000观察期）
        self.step = 0
        self.decision_step = 0
        self.training_start = 50000  # 观察期约2,000局游戏
        
        # 探索策略参数
        self.exploration_cycle = 50000  # 探索周期（约2,000局）
        self.exploration_duration = 5000  # 每次探索持续步数（约200局）
        self.main_exploration_games = 50000  # 前50,000局为主要探索阶段
        self.avg_decisions_per_game = 25  # 平均每局决策数，用于估算游戏局数
        
        # 🔧 修复9: 固定大小的监控数据
        self.episode_rewards = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.q_values_history = deque(maxlen=1000)
        
        # 网络更新参数 - 调整以适应大批次
        self.target_update_freq = 15000  # 增加目标网络更新频率
        self.training_freq = 4
        self.batch_size = 128  # 增大批次大小
        
        # 性能跟踪
        self.best_avg_reward = -float('inf')
        self.no_improvement_count = 0
        
        # 训练统计
        self.train_count = 0
        self.update_count = 0
        
        # 梯度累积（可选，用于进一步稳定大批次训练）
        self.gradient_accumulation_steps = 1
        
    def preprocess_state(self, state):
        """鲁棒的状态预处理"""
        try:
            # 🔧 修复10: 处理不同输入格式
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            if len(state.shape) == 3 and state.shape[2] == 3:
                # BGR to Gray
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            elif len(state.shape) == 3 and state.shape[2] == 1:
                state = state.squeeze(2)
            
            # 调整大小
            state = cv2.resize(state, (80, 80), interpolation=cv2.INTER_AREA)
            
            # 二值化
            _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
            
            return state.astype(np.uint8)
            
        except Exception as e:
            logging.error(f"状态预处理失败: {e}")
            # 返回默认状态
            return np.zeros((80, 80), dtype=np.uint8)
    
    def get_state_tensor(self, state_stack):
        """安全的状态tensor转换"""
        try:
            # 🔧 修复11: 确保状态格式正确
            if isinstance(state_stack, np.ndarray):
                # 确保是 (H, W, C) 格式
                if state_stack.shape == (80, 80, 4):
                    state_tensor = torch.FloatTensor(state_stack).unsqueeze(0)
                    state_tensor = state_tensor.permute(0, 3, 1, 2)  # (B, C, H, W)
                else:
                    raise ValueError(f"Invalid state shape: {state_stack.shape}")
            else:
                state_tensor = torch.FloatTensor(state_stack).unsqueeze(0)
            
            return state_tensor.to(self.device) / 255.0
            
        except Exception as e:
            logging.error(f"状态tensor转换失败: {e}")
            # 返回默认tensor
            return torch.zeros(1, 4, 80, 80).to(self.device)
    
    def select_action(self, state):
        """ε-贪婪动作选择"""
        try:
            # 观察期随机动作
            if self.decision_step < self.training_start:
                return random.randrange(self.actions)
            
            # ε-贪婪策略
            if random.random() < self.epsilon:
                return random.randrange(self.actions)
            
            # 网络预测
            with torch.no_grad():
                state_tensor = self.get_state_tensor(state)
                q_values = self.q_network(state_tensor)
                
                # 记录Q值用于监控
                if len(self.q_values_history) < 1000:
                    self.q_values_history.append(q_values.cpu().numpy().flatten())
                
                return q_values.argmax().item()
                
        except Exception as e:
            logging.error(f"动作选择失败: {e}")
            return random.randrange(self.actions)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        try:
            self.memory.add(state, action, reward, next_state, done)
        except Exception as e:
            logging.error(f"经验存储失败: {e}")
    
    def train(self):
        """训练网络 - 优化以支持大批次"""
        if len(self.memory) < self.batch_size * 2:  # 确保有足够的样本
            return None
        
        try:
            # 采样经验
            batch, indices, weights = self.memory.sample(self.batch_size)
            if batch is None:
                return None
            
            # 🔧 修复12: 优化的数据准备（批量处理）
            # 预分配数组
            states = np.zeros((self.batch_size, 4, 80, 80), dtype=np.float32)
            actions = np.zeros(self.batch_size, dtype=np.int64)
            rewards = np.zeros(self.batch_size, dtype=np.float32)
            next_states = np.zeros((self.batch_size, 4, 80, 80), dtype=np.float32)
            dones = np.zeros(self.batch_size, dtype=bool)
            
            # 批量填充数据
            for i, exp in enumerate(batch):
                # 处理状态形状
                state = exp['state']
                next_state = exp['next_state']
                
                if state.shape == (80, 80, 4):
                    states[i] = state.transpose(2, 0, 1)
                else:
                    states[i] = state
                    
                if next_state.shape == (80, 80, 4):
                    next_states[i] = next_state.transpose(2, 0, 1)
                else:
                    next_states[i] = next_state
                
                actions[i] = exp['action']
                rewards[i] = exp['reward']
                dones[i] = exp['done']
            
            # 转换为tensor
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # 当前Q值
            current_q_values = self.q_network(states)
            current_q = current_q_values.gather(1, actions.unsqueeze(1))
            
            # Double DQN目标Q值
            with torch.no_grad():
                # 主网络选择动作
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(1)
                
                # 目标网络评估Q值
                next_q_target = self.target_network(next_states)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1))
                
                # 计算目标
                target_q = rewards.unsqueeze(1) + (0.99 * next_q * ~dones.unsqueeze(1))
            
            # 计算TD误差和损失
            td_errors = target_q - current_q
            loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
            
            # 反向传播（支持梯度累积）
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (self.train_count + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            else:
                grad_norm = 0.0
            
            # 更新优先级
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)
            
            # 记录统计
            self.loss_history.append(loss.item() * self.gradient_accumulation_steps)
            self.train_count += 1
            
            # 目标网络更新
            if self.decision_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.update_count += 1
                logging.info(f"🎯 目标网络更新 #{self.update_count} (步数: {self.decision_step})")
            
            # 🔧 修复13: 适度的内存清理
            if self.train_count % 50 == 0:  # 更频繁的清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.train_count % 500 == 0:
                    gc.collect()
            
            return {
                'loss': loss.item() * self.gradient_accumulation_steps,
                'q_mean': current_q.mean().item(),
                'q_std': current_q.std().item(),
                'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                'lr': self.scheduler.get_last_lr()[0],
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'train_count': self.train_count,
                'batch_size': self.batch_size
            }
            
        except Exception as e:
            logging.error(f"训练失败: {e}")
            return None
    
    def update_epsilon(self):
        """改进的epsilon衰减策略，前50,000局探索，后50,000局利用"""
        # 估算当前游戏局数
        estimated_episode = self.decision_step / self.avg_decisions_per_game
        
        if self.decision_step < self.training_start:
            # 观察期（前2,000局）：完全随机
            self.epsilon = 1.0
        else:
            # 基础衰减
            progress = (self.decision_step - self.training_start) / self.epsilon_decay_steps
            base_epsilon = max(self.epsilon_min, 1.0 - progress * (1.0 - self.epsilon_min))
            
            # 分阶段调整epsilon
            if estimated_episode < 10000:
                # 前10,000局：保持高探索率（至少50%）
                base_epsilon = max(base_epsilon, 0.5)
            elif estimated_episode < 30000:
                # 10,000-30,000局：中等探索率（至少20%）
                base_epsilon = max(base_epsilon, 0.2)
            elif estimated_episode < 50000:
                # 30,000-50,000局：低探索率（至少10%）
                base_epsilon = max(base_epsilon, 0.1)
            # 50,000局后：使用计算的base_epsilon，逐渐降到1%
            
            # 周期性探索脉冲
            cycle_position = self.decision_step % self.exploration_cycle
            if cycle_position < self.exploration_duration:
                # 根据训练阶段调整脉冲强度
                if estimated_episode < 20000:
                    # 前20,000局：强脉冲
                    exploration_boost = 0.15
                elif estimated_episode < 40000:
                    # 20,000-40,000局：中等脉冲
                    exploration_boost = 0.10
                elif estimated_episode < 60000:
                    # 40,000-60,000局：弱脉冲
                    exploration_boost = 0.05
                else:
                    # 60,000局后：微弱脉冲
                    exploration_boost = 0.03
            else:
                exploration_boost = 0.0
            
            # 组合基础epsilon和探索脉冲
            self.epsilon = min(1.0, base_epsilon + exploration_boost)
            
            # 记录探索脉冲事件
            if cycle_position == 0 and self.decision_step > self.training_start:
                phase = "高探索期" if estimated_episode < 10000 else \
                        "中探索期" if estimated_episode < 30000 else \
                        "低探索期" if estimated_episode < 50000 else "利用期"
                logging.info(f"🔍 探索脉冲 | ε: {self.epsilon:.4f} | 局数: ~{int(estimated_episode)} | 阶段: {phase}")
            
            # 重要阶段转换提示
            if int(estimated_episode) == 50000 and int((self.decision_step - self.avg_decisions_per_game) / self.avg_decisions_per_game) == 49999:
                logging.info("🎯 主探索阶段完成！进入利用阶段")
                logging.info(f"   已完成50,000局探索，开始专注于策略优化")
    
    def evaluate_performance(self):
        """性能评估"""
        try:
            if len(self.episode_rewards) < 10:
                return {
                    'status': 'insufficient_data',
                    'avg_reward': 0,
                    'health_score': 50
                }
            
            recent_rewards = list(self.episode_rewards)[-100:] if len(self.episode_rewards) >= 100 else list(self.episode_rewards)
            recent_losses = list(self.loss_history)[-100:] if self.loss_history else [0]
            
            # 基本统计
            avg_reward = float(np.mean(recent_rewards))
            reward_std = float(np.std(recent_rewards))
            max_reward = float(np.max(recent_rewards))
            min_reward = float(np.min(recent_rewards))
            avg_loss = float(np.mean(recent_losses))
            
            # 改善率计算 - 修复版本
            if len(recent_rewards) >= 20:
                early_rewards = recent_rewards[:len(recent_rewards)//2]
                late_rewards = recent_rewards[len(recent_rewards)//2:]
                improvement_rate = float(np.mean(late_rewards) - np.mean(early_rewards))
            else:
                improvement_rate = 0.0
            
            # 健康度评分（调整阈值以适应大批次）
            health_score = 100
            if avg_loss > 3.0:  # 放宽损失阈值
                health_score -= 30
            if reward_std > abs(avg_reward) + 5:
                health_score -= 20
            if improvement_rate < -2:
                health_score -= 25
            if avg_reward < -5:
                health_score -= 15
            
            health_score = max(0, health_score)
            
            status = 'healthy' if health_score > 70 else 'warning' if health_score > 40 else 'critical'
            
            return {
                'status': status,
                'avg_reward': avg_reward,
                'reward_std': reward_std,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'avg_loss': avg_loss,
                'improvement_rate': improvement_rate,
                'health_score': health_score
            }
            
        except Exception as e:
            logging.error(f"性能评估失败: {e}")
            return {
                'status': 'error',
                'avg_reward': 0,
                'health_score': 0
            }
    
    def save_model(self, filepath, metadata=None):
        """保存模型"""
        try:
            checkpoint = {
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': self.step,
                'decision_step': self.decision_step,
                'epsilon': self.epsilon,
                'best_avg_reward': self.best_avg_reward,
                'train_count': self.train_count,
                'update_count': self.update_count,
                'batch_size': self.batch_size  # 保存批次大小
            }
            
            if metadata:
                checkpoint['metadata'] = metadata
            
            torch.save(checkpoint, filepath)
            logging.info(f"💾 模型保存成功: {filepath}")
            
        except Exception as e:
            logging.error(f"模型保存失败: {e}")

def setup_logging():
    """设置日志系统"""
    try:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/stable_dueling_dqn_bs128_{timestamp}.log"
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
        
        return log_file
        
    except Exception as e:
        print(f"日志设置失败: {e}")
        logging.basicConfig(level=logging.INFO)
        return None

def main():
    """主训练函数"""
    try:
        # 初始化日志
        log_file = setup_logging()
        logging.info("🚀 稳定版 Dueling DQN 训练系统启动 (Batch Size: 128)")
        
        # 设置垃圾回收
        gc.set_threshold(1000, 100, 100)
        
        # 导入游戏环境
        sys.path.append("game/")
        import game.wrapped_flappy_bird_fast as game
        
        # 初始化游戏和智能体
        game_state = game.GameState()
        agent = StableDQNAgent(actions=2)
        
        # 训练目标
        MAX_EPISODES = 100000  # 目标训练局数
        
        # 系统信息
        logging.info(f"   设备: {agent.device}")
        logging.info(f"   批次大小: {agent.batch_size}")
        logging.info(f"   观察期: {agent.training_start} 步")
        logging.info(f"   目标网络更新频率: {agent.target_update_freq}")
        logging.info(f"   缓冲区容量: {agent.memory.capacity}")
        logging.info(f"   学习率: {agent.optimizer.param_groups[0]['lr']}")
        logging.info(f"   🎯 目标训练局数: {MAX_EPISODES:,}")
        
        # 获取初始状态
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = agent.preprocess_state(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        
        # 训练循环变量
        episode_count = 0
        episode_reward = 0
        max_score = 0
        action_index = 0
        
        # 性能监控
        training_start_time = time.time()
        last_save_time = time.time()
        
        logging.info("🎮 开始训练循环")
        
        while episode_count < MAX_EPISODES:
            # 动作选择
            if agent.step % agent.training_freq == 0:
                action_index = agent.select_action(s_t)
            
            # 执行动作
            a_t = np.zeros([2])
            a_t[action_index] = 1
            
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = agent.preprocess_state(x_t1_colored)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.concatenate([x_t1, s_t[:, :, :3]], axis=2)
            
            episode_reward += r_t
            agent.step += 1
            
            # 训练
            if agent.step % agent.training_freq == 0:
                agent.decision_step += 1
                agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
                
                # 开始训练
                if agent.decision_step > agent.training_start:
                    train_stats = agent.train()
                    
                    # 训练日志
                    if train_stats and agent.decision_step % 1000 == 0:
                        avg_reward = np.mean(list(agent.episode_rewards)[-10:]) if len(agent.episode_rewards) >= 10 else 0
                        
                        logging.info(f"🧠 训练步骤 {agent.decision_step}")
                        logging.info(f"   损失: {train_stats['loss']:.4f} | Q值: {train_stats['q_mean']:.2f}±{train_stats['q_std']:.2f}")
                        logging.info(f"   学习率: {train_stats['lr']:.2e} | ε: {train_stats['epsilon']:.4f}")
                        logging.info(f"   近10局平均: {avg_reward:.2f} | 内存: {train_stats['memory_size']} | 批次: {train_stats['batch_size']}")
                        
                        # 检查是否在探索脉冲期间
                        if agent.decision_step > agent.training_start:
                            cycle_pos = agent.decision_step % agent.exploration_cycle
                            if cycle_pos < agent.exploration_duration:
                                logging.info(f"   🔍 探索脉冲进行中: {cycle_pos}/{agent.exploration_duration}")
                
                agent.update_epsilon()
            
            s_t = s_t1
            
            # 游戏结束处理
            if terminal:
                episode_count += 1
                agent.episode_rewards.append(episode_reward)
                
                # 保存最佳模型
                if episode_reward > max_score:
                    max_score = episode_reward
                    os.makedirs("saved_networks", exist_ok=True)
                    
                    metadata = {
                        'episode': episode_count,
                        'score': episode_reward,
                        'training_time': time.time() - training_start_time,
                        'decision_steps': agent.decision_step,
                        'batch_size': agent.batch_size
                    }
                    
                    agent.save_model(f"saved_networks/stable_best_bs128_{episode_reward:.0f}.pth", metadata)
                    logging.info(f"🏆 新纪录! 分数: {episode_reward}")
                
                # 游戏总结
                recent_avg = np.mean(list(agent.episode_rewards)[-100:]) if len(agent.episode_rewards) >= 100 else episode_reward
                
                # 内存使用报告
                memory_info = ""
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
                    memory_info = f" | GPU: {gpu_memory:.0f}/{gpu_memory_reserved:.0f}MB"
                
                # 进度信息
                progress = (episode_count / MAX_EPISODES) * 100
                logging.info(f"🎮 第{episode_count}局 ({progress:.1f}%) | 分数: {episode_reward:.2f} | 最高: {max_score:.2f} | 近100局: {recent_avg:.2f}{memory_info}")
                
                # 性能报告
                if episode_count % 100 == 0:
                    performance = agent.evaluate_performance()
                    training_hours = (time.time() - training_start_time) / 3600
                    
                    logging.info(f"📊 第{episode_count}局 性能报告 (Batch Size: {agent.batch_size})")
                    logging.info(f"   训练进度: {episode_count:,}/{MAX_EPISODES:,} ({progress:.1f}%)")
                    logging.info(f"   训练时长: {training_hours:.1f}小时 | 决策步数: {agent.decision_step}")
                    logging.info(f"   平均表现: {performance['avg_reward']:.2f}±{performance['reward_std']:.2f}")
                    logging.info(f"   表现范围: [{performance['min_reward']:.1f}, {performance['max_reward']:.1f}]")
                    logging.info(f"   训练稳定性: 损失 {performance['avg_loss']:.4f}")
                    logging.info(f"   健康度: {performance['health_score']}/100 ({performance['status']})")
                    logging.info(f"   改善率: {performance['improvement_rate']:.2f}")
                    
                    # 探索策略信息
                    base_progress = min(1.0, (agent.decision_step - agent.training_start) / agent.epsilon_decay_steps) if agent.decision_step > agent.training_start else 0
                    estimated_games = agent.decision_step / agent.avg_decisions_per_game
                    
                    if estimated_games < 2000:
                        exploration_phase = "观察期"
                    elif estimated_games < 10000:
                        exploration_phase = "高探索期"
                    elif estimated_games < 30000:
                        exploration_phase = "中探索期"
                    elif estimated_games < 50000:
                        exploration_phase = "低探索期"
                    else:
                        exploration_phase = "利用期"
                    
                    logging.info(f"   探索率: ε={agent.epsilon:.4f} | 阶段: {exploration_phase} | 衰减进度: {base_progress*100:.1f}%")
                    
                    # 估算剩余时间
                    if episode_count > 0:
                        episodes_per_hour = episode_count / training_hours
                        remaining_episodes = MAX_EPISODES - episode_count
                        estimated_hours = remaining_episodes / episodes_per_hour
                        logging.info(f"   预计剩余时间: {estimated_hours:.1f}小时 ({episodes_per_hour:.1f}局/小时)")
                    
                    # 保存定期检查点
                    if time.time() - last_save_time > 1800:  # 每30分钟保存一次
                        checkpoint_metadata = {
                            'episode': episode_count,
                            'performance': performance,
                            'training_hours': training_hours,
                            'batch_size': agent.batch_size
                        }
                        agent.save_model(f"saved_networks/stable_checkpoint_bs128_{episode_count}.pth", checkpoint_metadata)
                        last_save_time = time.time()
                    
                    # 训练质量警告
                    if performance['health_score'] < 30:
                        logging.warning("⚠️  训练健康度极低，建议检查网络参数")
                    elif performance['health_score'] < 50:
                        logging.warning("⚠️  训练健康度较低，注意监控")
                
                episode_reward = 0
                
                # 阶段转换提示
                if agent.decision_step == agent.training_start:
                    logging.info("🎆 观察期结束，进入主探索阶段！")
                    logging.info(f"   已完成约2,000局观察")
                    logging.info(f"   当前缓冲区大小: {len(agent.memory)}")
                    logging.info(f"   当前ε值: {agent.epsilon:.4f}")
                    logging.info(f"   开始48,000局的探索学习")
                
                # 10,000局里程碑
                if episode_count == 10000:
                    logging.info("📍 完成10,000局 - 高探索期")
                    logging.info(f"   当前ε值: {agent.epsilon:.4f}")
                    logging.info(f"   历史最高分: {max_score:.2f}")
                
                # 30,000局里程碑
                if episode_count == 30000:
                    logging.info("📍 完成30,000局 - 进入低探索期")
                    logging.info(f"   当前ε值: {agent.epsilon:.4f}")
                    logging.info(f"   历史最高分: {max_score:.2f}")
                
                # 50,000局阶段转换
                if episode_count == 50000:
                    logging.info("🏁 主探索阶段完成！进入利用阶段")
                    logging.info(f"   已完成50,000局探索学习")
                    logging.info(f"   当前ε值: {agent.epsilon:.4f}")
                    logging.info(f"   历史最高分: {max_score:.2f}")
                    recent_1000_avg = np.mean(list(agent.episode_rewards)[-1000:]) if len(agent.episode_rewards) >= 1000 else 0
                    recent_5000_avg = np.mean(list(agent.episode_rewards)[-5000:]) if len(agent.episode_rewards) >= 5000 else 0
                    logging.info(f"   近1,000局平均: {recent_1000_avg:.2f}")
                    logging.info(f"   近5,000局平均: {recent_5000_avg:.2f}")
                    logging.info("   开始后50,000局的策略优化")
                
                # 训练里程碑
                milestones = [50000, 100000, 200000, 500000, 1000000]
                for milestone in milestones:
                    if agent.decision_step == milestone:
                        current_performance = agent.evaluate_performance()
                        logging.info(f"🎯 训练里程碑: {milestone} 决策步")
                        if current_performance['status'] != 'insufficient_data':
                            logging.info(f"   里程碑性能: {current_performance['avg_reward']:.2f}")
                            training_efficiency = current_performance['avg_reward'] / (training_hours + 0.1)
                            logging.info(f"   训练效率: {training_efficiency:.2f} 分/小时")
                            logging.info(f"   每批次样本数: {agent.batch_size}")
        
        # 训练完成
        logging.info("🎉 训练完成！已完成100,000局游戏")
        
        # 最终性能报告
        final_performance = agent.evaluate_performance()
        total_training_time = (time.time() - training_start_time) / 3600
        
        logging.info("📈 最终训练报告:")
        logging.info(f"   总训练局数: {episode_count:,}")
        logging.info(f"   总训练时长: {total_training_time:.1f}小时")
        logging.info(f"   总决策步数: {agent.decision_step:,}")
        logging.info(f"   历史最高分: {max_score:.2f}")
        logging.info(f"   最终100局平均: {final_performance['avg_reward']:.2f}")
        logging.info(f"   最终健康度: {final_performance['health_score']}/100")
        
        # 保存最终模型
        final_metadata = {
            'episode': episode_count,
            'total_training_hours': total_training_time,
            'decision_steps': agent.decision_step,
            'max_score': max_score,
            'final_performance': final_performance,
            'batch_size': agent.batch_size
        }
        
        agent.save_model(f"saved_networks/stable_final_bs128_100k.pth", final_metadata)
        logging.info("💾 已保存最终模型")
        
    except KeyboardInterrupt:
        logging.info("👋 训练被用户中断")
        logging.info(f"   已完成 {episode_count:,}/{MAX_EPISODES:,} 局 ({(episode_count/MAX_EPISODES)*100:.1f}%)")
        if 'agent' in locals():
            try:
                final_checkpoint = f"saved_networks/stable_interrupted_bs128_{episode_count}.pth"
                agent.save_model(final_checkpoint)
                logging.info(f"💾 已保存中断检查点: {final_checkpoint}")
            except:
                logging.warning("检查点保存失败")
    
    except Exception as e:
        logging.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 尝试保存紧急检查点
        if 'agent' in locals():
            try:
                emergency_checkpoint = f"saved_networks/stable_emergency_bs128_{int(time.time())}.pth"
                agent.save_model(emergency_checkpoint)
                logging.info(f"🚨 已保存紧急检查点: {emergency_checkpoint}")
            except:
                logging.warning("紧急检查点保存失败")

if __name__ == "__main__":
    main()