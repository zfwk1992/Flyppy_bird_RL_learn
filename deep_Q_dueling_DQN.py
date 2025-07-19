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
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 抑制PNG sRGB警告
warnings.filterwarnings("ignore", message=".*iCCP.*")
os.environ['PYTHONWARNINGS'] = 'ignore'


class SumTree:
    """
    SumTree数据结构用于优先经验回放
    支持O(log n)时间复杂度的采样和更新操作
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """根据累积和检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]
    
    def add(self, priority, data):
        """添加新的经验和优先级"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """更新指定索引的优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """根据累积值获取数据"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    基于TD-error的优先级采样
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样参数
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def add(self, experience):
        """添加经验,使用最大优先级"""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """优先级采样"""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        # 计算重要性采样权重
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        return batch, idxs, is_weight
    
    def update_priorities(self, idxs, errors):
        """根据TD-error更新优先级"""
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries

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
- 共享层: 2层 (1024->1024)
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
        
        # logging.info(f"📈 训练进度图已保存: {plot_filename}")  # 减少冗余日志
        
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
        self.ln1 = nn.LayerNorm([32, 20, 20])  # LayerNorm替代BatchNorm
        self.ln2 = nn.LayerNorm([64, 10, 10])  # LayerNorm替代BatchNorm
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
            elif isinstance(module, nn.LayerNorm):
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
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
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
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
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
        
        # 软更新参数
        self.tau = 5e-3  # 软更新系数 (从1e-3优化为5e-3，提升目标网络响应性)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)
        
        # 优先经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(REPLAY_MEMORY, alpha=0.6, beta=0.4, beta_increment=0.01)
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0  # 总帧数计数器
        self.decision_step = 0  # 决策步数计数器
        self.reward_history = []
        self.loss_history = []  # 记录损失历史
        self.episode_rewards = []  # 记录每局奖励
        self.training_steps = []  # 记录训练步数
        
    def load_checkpoint(self, checkpoint_path):
        """加载完整的训练检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载网络状态
            self.q_network.load_state_dict(checkpoint['main_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # 加载训练状态
            self.decision_step = checkpoint.get('decision_step', 0)
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.reward_history = checkpoint.get('reward_history', [])
            self.loss_history = checkpoint.get('loss_history', [])
            
            # 恢复目标网络更新计数
            if 'target_update_count' in checkpoint:
                self.target_update_count = checkpoint['target_update_count']
            
            print(f"✅ 检查点加载成功: {checkpoint_path}")
            print(f"   📊 恢复状态: 决策步{self.decision_step} | ε:{self.epsilon:.4f}")
            print(f"   🎯 目标网络更新次数: {getattr(self, 'target_update_count', 0)}")
            
            return checkpoint.get('episode_count', 0), checkpoint.get('max_score', 0)
        else:
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return 0, 0
        
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
        """存储经验到优先缓冲区"""
        experience = (state, action, reward, next_state, done)
        self.memory.add(experience)
    
    def train(self):
        """使用优先经验回放的训练网络"""
        if len(self.memory) < BATCH:
            return None
        
        # 优先级采样
        batch, idxs, is_weights = self.memory.sample(BATCH)
        states = torch.FloatTensor([e[0] for e in batch]).to(device, non_blocking=True).permute(0, 3, 1, 2)
        actions = torch.LongTensor([e[1] for e in batch]).to(device, non_blocking=True)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device, non_blocking=True)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device, non_blocking=True).permute(0, 3, 1, 2)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device, non_blocking=True)
        is_weights = torch.FloatTensor(is_weights).to(device, non_blocking=True)
        
        # 当前Q值
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.max(1)[1]
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (GAMMA * next_q * ~dones.unsqueeze(1))
        
        # 计算TD错误
        td_errors = target_q - current_q
        
        # 使用重要性采样权重的损失
        loss = (is_weights.unsqueeze(1) * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
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
            'current_q_std': current_q.std().item(),
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
        
        # 更新优先级
        td_errors_cpu = td_errors.detach().cpu().numpy().flatten()
        self.memory.update_priorities(idxs, td_errors_cpu)
        
        # 记录损失用于绘图
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss.item())
            self.training_steps.append(self.decision_step)
        
        # 软更新目标网络
        self.soft_update_target_network()
        
        # 更新学习率 (每10000个决策步调用一次，而不是每次训练都调用)
        if self.decision_step % 10000 == 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"🔧 学习率更新: {current_lr:.2e} (步数:{self.decision_step})")
            if current_lr < 5e-5:
                print(f"⚠️  学习率过低警告! 当前:{current_lr:.2e}, 可能影响学习效果")
        
        # 添加PER相关统计信息
        train_stats['td_error_mean'] = np.abs(td_errors_cpu).mean()
        train_stats['td_error_max'] = np.abs(td_errors_cpu).max()
        train_stats['is_weight_mean'] = is_weights.mean().item()
        train_stats['per_beta'] = self.memory.beta
        
        # 训练健康监控指标
        train_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        train_stats['epsilon'] = self.epsilon
        train_stats['advantage_range'] = (advantage.max() - advantage.min()).item()
        train_stats['q_value_range'] = (current_q_values.max() - current_q_values.min()).item()
        
        # 目标网络与主网络差异监控
        with torch.no_grad():
            target_q_values = self.target_network(states)
            target_q_selected = target_q_values.gather(1, actions.unsqueeze(1))
            
            # 网络差异指标
            train_stats['target_main_q_diff'] = torch.abs(target_q_selected - current_q).mean().item()
            train_stats['target_main_q_ratio'] = (target_q_selected.mean() / (current_q.mean() + 1e-8)).item()
            
            # 训练稳定性指标
            train_stats['gradient_norm'] = sum(p.grad.norm().item() for p in self.q_network.parameters() if p.grad is not None)
            train_stats['target_update_count'] = getattr(self, 'target_update_count', 0)
            
            # 网络收敛监控
            train_stats['q_value_stability'] = current_q.std().item() / (current_q.mean().abs().item() + 1e-8)
            train_stats['action_preference'] = torch.abs(current_q_values[:, 0].mean() - current_q_values[:, 1].mean()).item()
        
        # 检查训练停滞预警
        if train_stats['learning_rate'] < 1e-6:
            print("⚠️  警告: 学习率过低 ({:.2e})，可能导致训练停滞！".format(train_stats['learning_rate']))
        
        if train_stats['advantage_std'] < 0.01:
            print("⚠️  警告: Advantage标准差过小 ({:.4f})，动作区分度不足！".format(train_stats['advantage_std']))
        
        return train_stats
    
    def soft_update_target_network(self):
        """使用Polyak平均法软更新目标网络"""
        # 记录更新前的参数范数用于监控
        if not hasattr(self, 'target_update_count'):
            self.target_update_count = 0
        
        # 计算更新前后的参数差异
        param_changes = []
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            old_target = target_param.data.clone()
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
            # 记录参数变化幅度
            if self.target_update_count % 1000 == 0:  # 每1000次更新记录一次
                change = torch.norm(target_param.data - old_target).item()
                param_changes.append(change)
        
        self.target_update_count += 1
        
        # 每1000次更新输出监控信息
        if self.target_update_count % 1000 == 0:
            avg_change = np.mean(param_changes) if param_changes else 0
            print(f"🎯 目标网络更新 #{self.target_update_count} | 平均参数变化: {avg_change:.6f} | τ={self.tau}")
    
    # 传统固定间隔更新和最优网络策略已完全移除
    # 当前仅使用软更新策略
    
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
    
    # 系统初始化信息
    device_info = str(device).upper()
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
    logging.info(f"🚀 Dueling DQN 初始化 | 设备:{device_info} | 批次:{BATCH}")
    
    # 初始化游戏环境
    game_state = game.GameState()
    
    # 初始化增强版智能体
    agent = EnhancedDuelingDQNAgent(ACTIONS)
    
    logging.info(f"⚙️  核心优化: LayerNorm+软更新(τ={agent.tau:.3f})+PER(α={agent.memory.alpha},β={agent.memory.beta:.2f})")
    
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
    
    logging.info(f"🎮 开始训练 | 观察:{OBSERVE} 探索:{EXPLORE} | {500//FRAME_PER_ACTION}决策/秒")
    
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
            
            # 软更新目标网络已在train()中完成
            
            # 训练监控 - 每1000步详细日志
            if agent.decision_step > OBSERVE and agent.decision_step % 1000 == 0 and train_stats is not None:
                plot_training_progress(agent, episode_count, datetime.now().strftime("%Y%m%d_%H%M%S"))
                avg_reward = np.mean(agent.reward_history[-100:]) if agent.reward_history else 0
                
                # 训练阶段状态
                if agent.decision_step < OBSERVE + EXPLORE:
                    phase = f"探索期 {agent.decision_step}/{OBSERVE + EXPLORE}"
                else:
                    phase = "利用期"
                
                # 核心训练指标
                logging.info(f"📊 [{phase}] 步数:{agent.decision_step} | 损失:{train_stats['loss']:.4f} | 均分:{avg_reward:.2f} | 最高:{max_score:.2f}")
                logging.info(f"   🎯 Q值: {train_stats['current_q_mean']:.2f}±{train_stats['current_q_std']:.2f} | TD误差:{train_stats['td_error_mean']:.3f}")
                logging.info(f"   🧠 V:{train_stats['value_mean']:.2f}±{train_stats['value_std']:.2f} | A-Range:{train_stats['advantage_range']:.2f}")
                logging.info(f"   ⚙️  LR:{train_stats['learning_rate']:.2e} | ε:{train_stats['epsilon']:.4f} | β:{train_stats['per_beta']:.3f}")
                
                # 目标网络监控
                logging.info(f"   🎯 目标网络: Q差异:{train_stats['target_main_q_diff']:.3f} | 比率:{train_stats['target_main_q_ratio']:.3f} | 更新次数:{train_stats['target_update_count']}")
                
                # 训练稳定性监控  
                logging.info(f"   📈 训练稳定性: 梯度范数:{train_stats['gradient_norm']:.3f} | Q稳定性:{train_stats['q_value_stability']:.3f} | 动作偏好:{train_stats['action_preference']:.3f}")
                
                # GPU和内存监控
                if torch.cuda.is_available():
                    logging.info(f"   💾 GPU:{train_stats['gpu_memory_used']:.0f}MB | 缓存:{train_stats['gpu_memory_cached']:.0f}MB")
                
                # 训练健康度评估
                health_score = 100
                warnings = []
                
                if train_stats['learning_rate'] < 1e-6:
                    health_score -= 30
                    warnings.append("学习率过低")
                    
                if train_stats['advantage_std'] < 0.01:
                    health_score -= 20
                    warnings.append("动作区分度不足")
                    
                if train_stats['gradient_norm'] < 0.001:
                    health_score -= 15
                    warnings.append("梯度过小")
                    
                if train_stats['target_main_q_diff'] > 10:
                    health_score -= 25
                    warnings.append("网络差异过大")
                
                health_status = "🟢 健康" if health_score >= 80 else "🟡 注意" if health_score >= 60 else "🔴 警告"
                warning_text = f" ({', '.join(warnings)})" if warnings else ""
                logging.info(f"   🏥 训练健康度: {health_status} {health_score}/100{warning_text}")
            
            # 更新探索率
            agent.update_epsilon()
        
        # 更新状态
        s_t = s_t1
        
        # 游戏结束处理
        if terminal:
            episode_count += 1
            agent.reward_history.append(episode_reward)
            agent.episode_rewards.append(episode_reward)
            
            # 保存最佳模型 (主网络 + 目标网络)
            if episode_reward > max_score:
                max_score = episode_reward
                os.makedirs("saved_networks", exist_ok=True)
                
                # 保存主网络
                main_net_path = f"saved_networks/bird-dueling-dqn-best-{episode_reward:.3f}.pth"
                torch.save(agent.q_network.state_dict(), main_net_path)
                
                # 保存目标网络
                target_net_path = f"saved_networks/bird-dueling-dqn-target-best-{episode_reward:.3f}.pth"
                torch.save(agent.target_network.state_dict(), target_net_path)
                
                # 保存完整训练状态
                checkpoint_path = f"saved_networks/bird-dueling-dqn-checkpoint-best-{episode_reward:.3f}.pth"
                checkpoint = {
                    'main_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'episode_count': episode_count,
                    'decision_step': agent.decision_step,
                    'max_score': max_score,
                    'epsilon': agent.epsilon,
                    'target_update_count': getattr(agent, 'target_update_count', 0)
                }
                torch.save(checkpoint, checkpoint_path)
                
                logging.info(f"🏆 新纪录! 分数:{episode_reward:.2f}")
                logging.info(f"   💾 主网络: {main_net_path}")
                logging.info(f"   🎯 目标网络: {target_net_path}")
                logging.info(f"   📋 完整检查点: {checkpoint_path}")
            
            # 训练阶段显示
            if agent.decision_step < OBSERVE:
                phase = f"观察期({agent.decision_step}/{OBSERVE})"
            elif agent.decision_step < OBSERVE + EXPLORE:
                phase = f"探索期({agent.decision_step}/{OBSERVE + EXPLORE})"
            else:
                phase = "利用期"
            
            # 游戏结束日志 - 简洁格式  
            logging.info(f"🎮 游戏{episode_count} | {phase} | 分数:{episode_reward:.2f} | 最高:{max_score:.2f} | ε:{agent.epsilon:.4f}")
            
            # 每10局检查训练健康状况
            if episode_count % 10 == 0 and episode_count > 0:
                recent_scores = agent.reward_history[-10:] if len(agent.reward_history) >= 10 else agent.reward_history
                avg_recent = np.mean(recent_scores) if recent_scores else 0
                
                if agent.decision_step > OBSERVE:
                    # 检查性能停滞
                    if len(agent.reward_history) >= 50:
                        last_50 = np.mean(agent.reward_history[-50:])
                        prev_50 = np.mean(agent.reward_history[-100:-50]) if len(agent.reward_history) >= 100 else last_50
                        improvement = last_50 - prev_50
                        
                        if improvement < 0.1 and agent.decision_step > OBSERVE + 5000:
                            logging.info(f"⚠️  性能停滞警告! 近50局平均分:{last_50:.2f}, 改善:{improvement:.2f}")
                    
                    logging.info(f"📊 第{episode_count}局 | 近10局平均:{avg_recent:.2f} | 决策步:{agent.decision_step}")
            
            episode_reward = 0
            
            # 定期保存模型 (每100局)
            if episode_count % 100 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                
                # 保存主网络
                main_net_path = f"saved_networks/bird-dueling-dqn-{episode_count}.pth"
                torch.save(agent.q_network.state_dict(), main_net_path)
                
                # 保存目标网络  
                target_net_path = f"saved_networks/bird-dueling-dqn-target-{episode_count}.pth"
                torch.save(agent.target_network.state_dict(), target_net_path)
                
                # 保存完整检查点
                checkpoint_path = f"saved_networks/bird-dueling-dqn-checkpoint-{episode_count}.pth"
                checkpoint = {
                    'main_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'episode_count': episode_count,
                    'decision_step': agent.decision_step,
                    'max_score': max_score,
                    'epsilon': agent.epsilon,
                    'target_update_count': getattr(agent, 'target_update_count', 0),
                    'reward_history': agent.reward_history[-100:],  # 保存最近100局的奖励
                    'loss_history': agent.loss_history[-1000:] if hasattr(agent, 'loss_history') else []
                }
                torch.save(checkpoint, checkpoint_path)
                
                # 详细的保存日志
                current_avg = np.mean(agent.reward_history[-10:]) if len(agent.reward_history) >= 10 else 0
                logging.info(f"💾 定期保存 (第{episode_count}局)")
                logging.info(f"   📊 当前状态: 决策步{agent.decision_step} | 近10局均分:{current_avg:.2f} | ε:{agent.epsilon:.4f}")
                logging.info(f"   💾 主网络: {main_net_path}")
                logging.info(f"   🎯 目标网络: {target_net_path}")
                logging.info(f"   📋 完整检查点: {checkpoint_path}")
        
            # 阶段转换提示 (只在决策帧检查)
            if agent.decision_step == OBSERVE:
                logging.info(f"🎆 观察期结束! 开始 Dueling DQN 训练...")
            elif agent.decision_step == OBSERVE + EXPLORE:
                logging.info(f"🏆 进入利用期! 主要使用已学策略...")
            
            # 进度提示
            if agent.decision_step % 5000 == 0 and agent.decision_step > 0:
                if agent.decision_step < OBSERVE:
                    remaining = OBSERVE - agent.decision_step
                    logging.info(f"🔍 观察期: 还需{remaining}步开始训练")
                elif agent.decision_step < OBSERVE + EXPLORE:
                    remaining = OBSERVE + EXPLORE - agent.decision_step
                    logging.info(f"🔍 探索期: 还需{remaining}步进入利用期")


if __name__ == "__main__":
    main()