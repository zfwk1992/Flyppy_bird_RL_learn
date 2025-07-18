#!/usr/bin/env python
# Dueling DQN 训练脚本：基于deep_Q_oneStep.py，使用Dueling DQN架构
# 核心创新：使用Value stream和Advantage stream分别估算状态价值和动作优势

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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加游戏路径
sys.path.append("game/")
import game.wrapped_flappy_bird_fast as game

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)

# 优化的超参数配置
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.995
OBSERVE = 5000  # 减少观察期，更快开始训练
EXPLORE = 25000
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 20000
BATCH = 256  # 减少批次大小，更频繁更新
FRAME_PER_ACTION = 4

# 设置日志
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_dueling_DQN_{timestamp}.log"
    
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


def plot_training_progress(agent, episode_count, timestamp):
    """
    绘制训练进度图表
    
    Args:
        agent: DQN智能体
        episode_count: 当前局数
        timestamp: 时间戳
    """
    try:
        # 创建保存目录
        os.makedirs("logs/plots", exist_ok=True)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'增强版 Dueling DQN 训练进度 - 局数: {episode_count}', fontsize=16)
        
        # 图1: 奖励历史
        if agent.episode_rewards:
            episodes = list(range(1, len(agent.episode_rewards) + 1))
            ax1.plot(episodes, agent.episode_rewards, 'b-', alpha=0.7, linewidth=1)
            
            # 添加移动平均线
            if len(agent.episode_rewards) >= 10:
                window_size = min(50, len(agent.episode_rewards) // 2)
                moving_avg = []
                for i in range(window_size-1, len(agent.episode_rewards)):
                    avg = np.mean(agent.episode_rewards[i-window_size+1:i+1])
                    moving_avg.append(avg)
                
                avg_episodes = list(range(window_size, len(agent.episode_rewards) + 1))
                ax1.plot(avg_episodes, moving_avg, 'r-', linewidth=2, 
                        label=f'移动平均({window_size}局)')
                ax1.legend()
            
            ax1.set_title('每局奖励历史')
            ax1.set_xlabel('局数')
            ax1.set_ylabel('奖励')
            ax1.grid(True, alpha=0.3)
            
            # 显示统计信息
            max_reward = max(agent.episode_rewards)
            avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            ax1.text(0.02, 0.98, f'最高: {max_reward:.2f}\n近100局平均: {avg_reward:.2f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 图2: 损失历史
        if agent.loss_history and agent.training_steps:
            ax2.plot(agent.training_steps, agent.loss_history, 'g-', alpha=0.7, linewidth=1)
            
            # 添加移动平均线
            if len(agent.loss_history) >= 10:
                window_size = min(100, len(agent.loss_history) // 2)
                moving_avg_loss = []
                for i in range(window_size-1, len(agent.loss_history)):
                    avg = np.mean(agent.loss_history[i-window_size+1:i+1])
                    moving_avg_loss.append(avg)
                
                avg_steps = agent.training_steps[window_size-1:]
                ax2.plot(avg_steps, moving_avg_loss, 'orange', linewidth=2, 
                        label=f'移动平均({window_size}步)')
                ax2.legend()
            
            ax2.set_title('训练损失历史')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('损失')
            ax2.grid(True, alpha=0.3)
            
            # 显示统计信息
            if agent.loss_history:
                recent_loss = np.mean(agent.loss_history[-100:]) if len(agent.loss_history) >= 100 else np.mean(agent.loss_history)
                ax2.text(0.02, 0.98, f'近100步平均损失: {recent_loss:.4f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 图3: 探索率变化
        decision_steps = list(range(0, agent.decision_step + 1, 100))
        epsilons = []
        for step in decision_steps:
            if step < OBSERVE:
                epsilon = 1.0
            elif step < OBSERVE + EXPLORE:
                epsilon = 1.0 - (step - OBSERVE) / EXPLORE * (1.0 - FINAL_EPSILON)
            else:
                epsilon = FINAL_EPSILON
            epsilons.append(epsilon)
        
        ax3.plot(decision_steps, epsilons, 'm-', linewidth=2)
        ax3.axvline(x=OBSERVE, color='r', linestyle='--', alpha=0.7, label='观察期结束')
        ax3.axvline(x=OBSERVE + EXPLORE, color='b', linestyle='--', alpha=0.7, label='探索期结束')
        ax3.axvline(x=agent.decision_step, color='g', linestyle='-', alpha=0.7, label='当前位置')
        ax3.set_title('探索率 (ε) 变化')
        ax3.set_xlabel('决策步数')
        ax3.set_ylabel('探索率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 显示当前探索率
        current_epsilon = agent.epsilon
        ax3.text(0.02, 0.98, f'当前ε: {current_epsilon:.4f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 图4: 训练阶段显示
        ax4.axis('off')
        
        # 计算当前阶段
        if agent.decision_step < OBSERVE:
            current_phase = "观察期"
            phase_progress = agent.decision_step / OBSERVE
            phase_color = 'orange'
        elif agent.decision_step < OBSERVE + EXPLORE:
            current_phase = "探索期"
            phase_progress = (agent.decision_step - OBSERVE) / EXPLORE
            phase_color = 'blue'
        else:
            current_phase = "利用期"
            phase_progress = 1.0
            phase_color = 'green'
        
        # 显示训练信息
        info_text = f"""
增强版 Dueling DQN 训练状态

当前阶段: {current_phase}
进度: {phase_progress:.1%}

决策步数: {agent.decision_step:,}
总局数: {episode_count}

网络架构:
- 卷积层: 2层 + BatchNorm
- 共享层: 2层 (1024→ 1024)
- Value 分支: 3层
- Advantage 分支: 3层
- Dropout: 0.3

超参数:
- 批次大小: {BATCH}
- 学习率: 5e-4
- 折扣因子: {GAMMA}
        """
        
        ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=phase_color, alpha=0.2))
        
        # 保存图表
        plt.tight_layout()
        plot_filename = f"logs/plots/dueling_dqn_progress_{timestamp}_{episode_count:04d}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📈 训练进度图已保存: {plot_filename}")
        
    except Exception as e:
        logging.warning(f"⚠️ 绘图失败: {e}")


class EnhancedDuelingDQN(nn.Module):
    """
    增强版 Dueling DQN 网络架构
    
    优化特性：
    - 更深的共享特征提取层
    - 增强的Value和Advantage分支
    - 改进的权重初始化
    - 更大的网络容量
    """
    def __init__(self, actions):
        super(EnhancedDuelingDQN, self).__init__()
        self.actions = actions
        
        # 共享卷积层（特征提取）
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 增强的共享全连接层
        self.shared_fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.shared_fc2 = nn.Linear(1024, 1024)  # 新增共享层
        self.dropout = nn.Dropout(0.3)  # 防止过拟合
        
        # 增强的Value stream (状态价值分支)
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 1024),  # 增加容量
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # 输出标量：V(s)
        )
        
        # 增强的Advantage stream (动作优势分支)
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 1024),  # 增加容量
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, actions)  # 输出向量：A(s,a)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        增强版前向传播
        
        Args:
            x: 输入状态 [batch_size, 4, 80, 80]
        
        Returns:
            Q值 [batch_size, actions]
        """
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        
        # 增强的共享全连接层
        x = F.relu(self.shared_fc1(x))
        x = self.dropout(x)
        shared_features = F.relu(self.shared_fc2(x))
        
        # Value stream: 估算状态价值
        value = self.value_stream(shared_features)  # [batch_size, 1]
        
        # Advantage stream: 估算动作优势
        advantage = self.advantage_stream(shared_features)  # [batch_size, actions]
        
        # Dueling DQN 合并公式
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # 减去均值确保网络能正确学习V(s)和A(s,a)的分离
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value_and_advantage(self, x):
        """
        获取Value和Advantage的分离值（用于分析）
        
        Returns:
            tuple: (value, advantage)
        """
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        
        # 增强的共享全连接层
        x = F.relu(self.shared_fc1(x))
        x = self.dropout(x)
        shared_features = F.relu(self.shared_fc2(x))
        
        # 分别计算Value和Advantage
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        return value, advantage


class EnhancedDuelingDQNAgent:
    """增强版 Dueling DQN 智能体"""
    def __init__(self, actions):
        self.actions = actions
        self.device = device
        
        # 创建增强网络
        self.q_network = EnhancedDuelingDQN(actions).to(device)
        self.target_network = EnhancedDuelingDQN(actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 智能目标网络选择
        self.best_reward_network = EnhancedDuelingDQN(actions).to(device)
        self.best_reward_network.load_state_dict(self.q_network.state_dict())
        self.best_reward = -float('inf')
        self.best_reward_step = 0
        self.last_target_update_step = 0
        
        # 优化器 - 降低学习率
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=5e-4, weight_decay=1e-5)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=REPLAY_MEMORY)
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0  # 总帧数计数器
        self.decision_step = 0  # 决策步数计数器
        self.reward_history = []
        self.loss_history = []  # 记录损失历史
        self.episode_rewards = []  # 记录每局奖励
        self.training_steps = []  # 记录训练步数
        
    def preprocess_state(self, state):
        """预处理状态"""
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return state
    
    def get_state_tensor(self, state_stack):
        """将状态堆栈转换为tensor"""
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device, non_blocking=True)
        return state_tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
    
    def select_action(self, state_tensor):
        """选择动作（标准ε-贪婪策略）"""
        if self.decision_step < OBSERVE:
            # 观察期：纯随机探索
            return random.randrange(self.actions)
        else:
            # 训练期：ε-贪婪策略
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
        states = torch.FloatTensor([e[0] for e in batch]).to(device, non_blocking=True).permute(0, 3, 1, 2)
        actions = torch.LongTensor([e[1] for e in batch]).to(device, non_blocking=True)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device, non_blocking=True)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device, non_blocking=True).permute(0, 3, 1, 2)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device, non_blocking=True)
        
        # 当前Q值
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # 目标Q值 (Double DQN with智能目标网络)
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.max(1)[1]
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (GAMMA * next_q * ~dones.unsqueeze(1))
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 返回训练统计信息（包括Dueling DQN特有的分析）
        train_stats = {
            'loss': loss.item(),
            'current_q_mean': current_q.mean().item(),
            'current_q_max': current_q.max().item(),
            'current_q_min': current_q.min().item(),
            'target_q_mean': target_q.mean().item(),
            'reward_mean': rewards.mean().item(),
            'q_values_action0': current_q_values[:, 0].mean().item(),
            'q_values_action1': current_q_values[:, 1].mean().item()
        }
        
        # 添加Dueling DQN特有的分析
        with torch.no_grad():
            value, advantage = self.q_network.get_value_and_advantage(states)
            train_stats['value_mean'] = value.mean().item()
            train_stats['value_std'] = value.std().item()
            train_stats['advantage_mean'] = advantage.mean().item()
            train_stats['advantage_std'] = advantage.std().item()
            train_stats['advantage_action0'] = advantage[:, 0].mean().item()
            train_stats['advantage_action1'] = advantage[:, 1].mean().item()
        
        # 添加GPU内存使用监控
        if torch.cuda.is_available():
            train_stats['gpu_memory_used'] = torch.cuda.memory_allocated(device) / 1024**2
            train_stats['gpu_memory_cached'] = torch.cuda.memory_reserved(device) / 1024**2
        
        # 记录损失用于绘图
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss.item())
            self.training_steps.append(self.decision_step)
        
        return train_stats
    
    def update_target_network(self):
        """阶段性智能目标网络更新策略（使用决策步数）"""
        if self.decision_step % 350 == 0:  # 350个决策更新周期
            
            # 阶段判断（使用决策步数）
            if self.decision_step < OBSERVE:
                strategy = "observe_best_only"
            elif self.decision_step < OBSERVE + 500:
                strategy = "explore_early_best_only"
            else:
                strategy = "intelligent_selection"
            
            # 执行对应策略
            if strategy == "observe_best_only":
                if self.best_reward_step > 0:
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    logging.info(f"")
                    logging.info(f"🔬📚 DUELING DQN OBSERVE PHASE -> BEST NETWORK ONLY 📚🔬")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 阶段策略: 观察期专注积累经验，仅使用最佳网络")
                    logging.info(f"   └─ 决策步数: {self.decision_step}/{OBSERVE}")
                    logging.info(f"")
                else:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    logging.info(f"")
                    logging.info(f"🔬⚡ DUELING DQN OBSERVE PHASE -> WAITING FOR BEST NETWORK ⚡🔬")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 阶段策略: 观察期暂用当前网络，等待首个最佳网络出现")
                    logging.info(f"   └─ 决策步数: {self.decision_step}/{OBSERVE}")
                    logging.info(f"")
                    
            elif strategy == "explore_early_best_only":
                if self.best_reward_step > 0:
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    remaining_steps = OBSERVE + 500 - self.decision_step
                    logging.info(f"")
                    logging.info(f"🚀💎 DUELING DQN EXPLORE EARLY -> FORCE BEST NETWORK 💎🚀")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 阶段策略: 探索前500步强制使用最佳网络稳定学习")
                    logging.info(f"   └─ 剩余强制步数: {remaining_steps}步")
                    logging.info(f"")
                else:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    remaining_steps = OBSERVE + 500 - self.decision_step
                    logging.info(f"")
                    logging.info(f"🚀⚡ DUELING DQN EXPLORE EARLY -> REGULAR UPDATE ⚡🚀")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 阶段策略: 探索早期，尚无最佳网络可用")
                    logging.info(f"   └─ 剩余早期步数: {remaining_steps}步")
                    logging.info(f"")
                    
            else:  # intelligent_selection
                has_recent_best = (self.best_reward_step > self.last_target_update_step and 
                                 self.step - self.best_reward_step < 1000)
                
                if has_recent_best:
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    network_age = self.decision_step - self.best_reward_step
                    phase = "探索后期" if self.decision_step < OBSERVE + EXPLORE else "利用期"
                    logging.info(f"")
                    logging.info(f"🎯🔥 DUELING DQN {phase.upper()} -> INTELLIGENT BEST NETWORK 🔥🎯")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 网络年龄: {network_age}步 (< 1000步有效期)")
                    logging.info(f"   └─ 切换原因: 智能选择最佳历史表现网络")
                    logging.info(f"")
                else:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    self.last_target_update_step = self.decision_step
                    phase = "探索后期" if self.decision_step < OBSERVE + EXPLORE else "利用期"
                    logging.info(f"")
                    logging.info(f"🔄⏰ DUELING DQN {phase.upper()} -> INTELLIGENT REGULAR UPDATE ⏰🔄")
                    logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
                    logging.info(f"   ├─ 当前决策步数: {self.decision_step}")
                    if self.best_reward_step > 0:
                        network_age = self.decision_step - self.best_reward_step
                        logging.info(f"   ├─ 最佳网络年龄: {network_age}步 (> 1000步过期)")
                        logging.info(f"   └─ 切换原因: 最佳网络过期，智能选择定时更新")
                    else:
                        logging.info(f"   └─ 切换原因: 智能选择定时更新 (尚无最佳网络)")
                    logging.info(f"")
    
    def update_best_reward_network(self, episode_reward):
        """更新最佳奖励网络"""
        should_update = False
        if self.decision_step < OBSERVE:
            should_update = (episode_reward > self.best_reward and episode_reward > 0)
        else:
            should_update = (episode_reward > self.best_reward)
            
        if should_update:
            old_best = self.best_reward
            improvement = episode_reward - old_best if old_best > -float('inf') else episode_reward
            self.best_reward = episode_reward
            self.best_reward_step = self.decision_step
            self.best_reward_network.load_state_dict(self.q_network.state_dict())
            
            phase = "观察期" if self.decision_step < OBSERVE else "训练期"
            logging.info(f"")
            logging.info(f"🏆⭐ NEW BEST DUELING DQN NETWORK SAVED! ({phase}) ⭐🏆")
            logging.info(f"   ├─ 新纪录: {episode_reward:.3f}")
            logging.info(f"   ├─ 网络架构: Value + Advantage 分支")
            if old_best > -float('inf'):
                logging.info(f"   ├─ 提升幅度: +{improvement:.3f} (从 {old_best:.3f})")
            logging.info(f"   ├─ 决策步数: {self.decision_step}")
            logging.info(f"   └─ Dueling DQN优势: 分离状态价值和动作优势学习")
            logging.info(f"")
    
    def update_epsilon(self):
        """更新探索率（使用决策步数）"""
        if self.decision_step < OBSERVE:
            self.epsilon = 1.0  # 观察期全随机
        elif self.decision_step < OBSERVE + EXPLORE:
            self.epsilon = 1.0 - (self.decision_step - OBSERVE) / EXPLORE * (1.0 - FINAL_EPSILON)
        else:
            self.epsilon = FINAL_EPSILON


def main():
    """主训练函数"""
    # 设置日志
    log_file = setup_logging()
    
    # 显示设备信息
    logging.info(f"使用设备: {device}")
    logging.info(f"🎯 网络架构: Dueling DQN (Value + Advantage 分支)")
    logging.info(f"🔥 核心优势: 分离状态价值和动作优势学习，提升训练效率")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA版本: {torch.version.cuda}")
        
        # 估算内存需求
        batch_memory_mb = (BATCH * 4 * 80 * 80 * 4) / (1024 * 1024)
        logging.info(f"预计批次内存需求: {batch_memory_mb:.1f}MB (BATCH={BATCH})")
        logging.info(f"GPU显存: 4096MB, 预计利用率: {batch_memory_mb/4096*100:.1f}%")
    
    # 初始化游戏环境
    game_state = game.GameState()
    
    # 初始化增强版智能体
    agent = EnhancedDuelingDQNAgent(ACTIONS)
    
    logging.info(f"🚀 增强版 Dueling DQN 智能体初始化完成")
    logging.info(f"📊 网络架构: 增强版 Dueling DQN (更深的Value + Advantage 分支)")
    logging.info(f"🎯 权重初始化: Kaiming 初始化 (更好的收敛性)")
    logging.info(f"📉 观察期策略: 100%随机探索 (从头开始学习)")
    logging.info(f"⚡ 优化超参数: 减少观察期 + 降低学习率 + 更频繁更新")
    
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
    
    logging.info("🎮 开始增强版 Dueling DQN 训练...")
    logging.info(f"观察步数: {OBSERVE}, 探索步数: {EXPLORE}, 批次大小: {BATCH}")
    logging.info(f"决策间隔: {FRAME_PER_ACTION}帧/动作, 游戏速度: 500FPS")
    logging.info(f"🎯 增强版特色: 更深的Value/Advantage分支 + Dropout正则化")
    logging.info(f"💡 理论优势: 更大网络容量 + 更好梯度流 + 防过拟合")
    logging.info("🎯 智能目标网络: 自动选择最佳奖励网络或定时网络，提升训练稳定性")
    
    # 估算计算负载
    decisions_per_sec = 500 // FRAME_PER_ACTION
    logging.info(f"预计决策频率: {decisions_per_sec}次/秒 (vs 原250次/秒，减少{(1-decisions_per_sec/250)*100:.0f}%)")
    
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
        agent.step += 1
        
        # 只在决策帧存储经验和训练
        if agent.step % FRAME_PER_ACTION == 0:
            agent.decision_step += 1
            agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
            
            # 训练网络（仅训练期）
            if agent.decision_step > OBSERVE:
                train_stats = agent.train()
            
            # 更新目标网络
            agent.update_target_network()
            
            # 记录训练信息并绘图
            if agent.decision_step > OBSERVE and agent.decision_step % 1000 == 0 and train_stats is not None:
                # 绘制训练进度图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_training_progress(agent, episode_count, timestamp)
                avg_reward = np.mean(agent.reward_history[-100:]) if agent.reward_history else 0
                if agent.decision_step < OBSERVE:
                    status = f"观察期 {agent.decision_step}/{OBSERVE}"
                    strategy_info = "50%随机+50%预训练"
                elif agent.decision_step < OBSERVE + EXPLORE:
                    status = f"探索期 {agent.decision_step}/{OBSERVE + EXPLORE}"
                    strategy_info = f"ε={agent.epsilon:.4f}"
                else:
                    status = "利用期"
                    strategy_info = f"ε={agent.epsilon:.4f}"
                
                logging.info(f"[{status}] 决策步数: {agent.decision_step} | 策略: {strategy_info}")
                logging.info(f"  损失: {train_stats['loss']:.4f} | 平均奖励: {avg_reward:.3f} | 最高分: {max_score:.3f}")
                logging.info(f"  Q值 - 平均: {train_stats['current_q_mean']:.3f} | 最大: {train_stats['current_q_max']:.3f} | 最小: {train_stats['current_q_min']:.3f}")
                logging.info(f"  动作Q值 - 不跳: {train_stats['q_values_action0']:.3f} | 跳跃: {train_stats['q_values_action1']:.3f}")
                
                # Dueling DQN特有的分析
                logging.info(f"  🎯 Value分析 - 平均: {train_stats['value_mean']:.3f} | 标准差: {train_stats['value_std']:.3f}")
                logging.info(f"  🎯 Advantage分析 - 平均: {train_stats['advantage_mean']:.3f} | 标准差: {train_stats['advantage_std']:.3f}")
                logging.info(f"  🎯 动作优势 - 不跳: {train_stats['advantage_action0']:.3f} | 跳跃: {train_stats['advantage_action1']:.3f}")
                
                # GPU使用情况
                if torch.cuda.is_available() and 'gpu_memory_used' in train_stats:
                    logging.info(f"  GPU内存 - 已用: {train_stats['gpu_memory_used']:.1f}MB | 缓存: {train_stats['gpu_memory_cached']:.1f}MB")
                    
                    # 智能目标网络状态
                    if agent.best_reward_step > 0:
                        target_network_age = agent.decision_step - agent.best_reward_step
                        
                        if agent.decision_step < OBSERVE:
                            network_type = "🎯最佳网络"
                            status_icon = "📚观察期"
                        elif agent.decision_step < OBSERVE + 500:
                            network_type = "🎯最佳网络"
                            status_icon = "🚀早期强制"
                        else:
                            is_using_best = (agent.best_reward_step > agent.last_target_update_step and target_network_age < 1000)
                            network_type = "🎯最佳网络" if is_using_best else "🔄定时网络"
                            status_icon = "✅有效" if target_network_age < 1000 else "⏰过期"
                            
                        logging.info(f"  🎯 Dueling目标网络 - 当前使用: {network_type} | 最佳奖励: {agent.best_reward:.3f} | 年龄: {target_network_age}步 ({status_icon})")
                    else:
                        logging.info(f"  🎯 Dueling目标网络 - 当前使用: 🔄定时网络 | 状态: 尚未发现最佳网络")
            
            # 更新探索率
            agent.update_epsilon()
        
        # 更新状态
        s_t = s_t1
        
        # 游戏结束处理
        if terminal:
            episode_count += 1
            agent.reward_history.append(episode_reward)
            agent.episode_rewards.append(episode_reward)
            
            # 保存最佳模型
            if episode_reward > max_score:
                max_score = episode_reward
                os.makedirs("saved_networks", exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          f"saved_networks/bird-dueling-dqn-best-{episode_reward:.3f}.pth")
                logging.info(f"🏆 新的最佳增强版 Dueling DQN 模型已保存: bird-dueling-dqn-best-{episode_reward:.3f}.pth")
            
            # 更新最佳奖励网络
            agent.update_best_reward_network(episode_reward)
            
            # 计算训练状态
            if agent.decision_step < OBSERVE:
                status = f"观察期 ({agent.decision_step}/{OBSERVE})"
                strategy_display = "100%随机探索"
            elif agent.decision_step < OBSERVE + EXPLORE:
                status = f"探索期 ({agent.decision_step}/{OBSERVE + EXPLORE})"
                strategy_display = f"ε: {agent.epsilon:.4f}"
            else:
                status = "利用期"
                strategy_display = f"ε: {agent.epsilon:.4f}"
            
            # 当前目标网络状态
            if agent.best_reward_step > 0:
                if agent.decision_step < OBSERVE:
                    network_type = "🎯最佳"
                elif agent.decision_step < OBSERVE + 500:
                    network_type = "🎯最佳"
                else:
                    target_network_age = agent.decision_step - agent.best_reward_step
                    is_using_best = (agent.best_reward_step > agent.last_target_update_step and target_network_age < 1000)
                    network_type = "🎯最佳" if is_using_best else "🔄定时"
            else:
                network_type = "🔄定时"
            
            logging.info(f"🎮 增强版 Dueling DQN 游戏 {episode_count} 结束 | 决策步数: {agent.decision_step} | {status} | "
                        f"得分: {episode_reward:.3f} | 最高分: {max_score:.3f} | 策略: {strategy_display} | 目标网络: {network_type}")
            
            episode_reward = 0
            
            # 定期保存模型
            if episode_count % 100 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          f"saved_networks/bird-dueling-dqn-{episode_count}.pth")
                logging.info(f"定期增强版 Dueling DQN 模型已保存: bird-dueling-dqn-{episode_count}.pth")
        
        # 阶段提示
        if agent.decision_step == OBSERVE:
            logging.info("🎯 重要节点：观察期结束，开始增强版 Dueling DQN 训练！预期Value和Advantage分支开始分化...")
        elif agent.decision_step == OBSERVE + EXPLORE:
            logging.info("🎯 重要节点：探索期结束，进入利用期！增强版 Dueling DQN 已充分学习状态价值和动作优势...")
        
        # 定期提示进度
        if agent.decision_step % 5000 == 0 and agent.decision_step > 0:
            if agent.decision_step < OBSERVE:
                remaining = OBSERVE - agent.decision_step
                logging.info(f"📊 增强版 Dueling DQN 观察期进度：还需 {remaining} 步开始训练")
            elif agent.decision_step < OBSERVE + EXPLORE:
                remaining = OBSERVE + EXPLORE - agent.decision_step
                logging.info(f"📊 增强版 Dueling DQN 探索期进度：还需 {remaining} 步进入利用期")


if __name__ == "__main__":
    main()