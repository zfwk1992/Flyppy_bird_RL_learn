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
# 移除matplotlib依赖，使用GPU监控代替可视化

# 抑制PNG sRGB警告
warnings.filterwarnings("ignore", message=".*iCCP.*")
os.environ['PYTHONWARNINGS'] = 'ignore'


class SumTree:
    """
    SumTree数据结构用于优先经验回放
    支持O(log n)时间复杂度的采样和更新操作
    🚨 内存优化：使用预分配固定数组避免对象引用泄漏
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # 🚨 修复内存泄漏：使用None初始化的list代替object数组
        # 避免numpy object数组持有强引用导致的内存泄漏
        self.data = [None] * capacity  # 改为普通list，GC可正常回收
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
        
        # 🚨 内存优化：显式删除旧数据，帮助GC回收
        if self.data[self.write] is not None:
            del self.data[self.write]  # 显式删除旧经验
        
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
    torch.cuda.set_per_process_memory_fraction(0.7)  # 💪 合理使用GPU内存，平衡性能与稳定性

# 优化的超参数配置
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.995
OBSERVE = 5000  # 减少观察期，更快开始训练
EXPLORE = 25000
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 20000  # 💪 恢复合理的内存缓冲区大小，利用28GB内存优势
BATCH = 256            # 💪 恢复大批次训练，提升GPU利用率和训练稳定性
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


def monitor_gpu_status():
    """
    监控GPU状态和系统资源
    
    Returns:
        dict: GPU状态信息
    """
    gpu_info = {}
    
    if torch.cuda.is_available():
        try:
            # GPU基本信息
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name()
            
            # 内存使用情况
            gpu_info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            gpu_info['max_memory_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
            gpu_info['max_memory_reserved'] = torch.cuda.max_memory_reserved() / 1024**2  # MB
            
            # 内存使用率
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_info['total_memory'] = total_memory
            gpu_info['memory_usage_percent'] = (gpu_info['memory_allocated'] / total_memory) * 100
            
            # GPU利用率检查
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info['gpu_utilization'] = float(result.stdout.strip())
                else:
                    gpu_info['gpu_utilization'] = None
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                gpu_info['gpu_utilization'] = None
                
            # 温度监控
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info['temperature'] = float(result.stdout.strip())
                else:
                    gpu_info['temperature'] = None
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                gpu_info['temperature'] = None
                
        except Exception as e:
            logging.warning(f"GPU状态监控失败: {e}")
            gpu_info['error'] = str(e)
    else:
        gpu_info['available'] = False
        gpu_info['message'] = "CUDA不可用，使用CPU训练"
    
    return gpu_info


def log_training_statistics(agent, episode_count, train_stats=None):
    """
    记录详细的训练统计信息
    
    Args:
        agent: DQN智能体
        episode_count: 当前局数
        train_stats: 训练统计数据
    """
    try:
        # GPU状态监控
        gpu_info = monitor_gpu_status()
        
        # 训练进度统计
        if agent.episode_rewards:
            recent_rewards = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
            reward_stats = {
                'count': len(agent.episode_rewards),
                'mean': np.mean(recent_rewards),
                'max': max(agent.episode_rewards),
                'min': min(recent_rewards),
                'std': np.std(recent_rewards)
            }
        else:
            reward_stats = {'count': 0, 'mean': 0, 'max': 0, 'min': 0, 'std': 0}
        
        # 损失统计
        if hasattr(agent, 'loss_history') and agent.loss_history:
            recent_losses = agent.loss_history[-50:] if len(agent.loss_history) >= 50 else agent.loss_history
            loss_stats = {
                'count': len(agent.loss_history),
                'mean': np.mean(recent_losses),
                'min': min(recent_losses),
                'max': max(recent_losses),
                'std': np.std(recent_losses)
            }
        else:
            loss_stats = {'count': 0, 'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        # 训练阶段
        if agent.decision_step < OBSERVE:
            phase = "观察期"
            progress = agent.decision_step / OBSERVE
        elif agent.decision_step < OBSERVE + EXPLORE:
            phase = "探索期"
            progress = (agent.decision_step - OBSERVE) / EXPLORE
        else:
            phase = "利用期"
            progress = 1.0
        
        # 获取内存信息
        memory_summary = ""
        try:
            import psutil
            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_summary += f"CPU:{cpu_memory_mb:.0f}MB"
            
            if gpu_info.get('available', False):
                gpu_memory_used = gpu_info.get('memory_allocated', 0)
                gpu_memory_total = gpu_info.get('total_memory', 0)
                memory_summary += f" | GPU:{gpu_memory_used:.0f}MB/{gpu_memory_total:.0f}MB"
        except ImportError:
            memory_summary = "N/A"
        
        # 详细日志记录
        logging.info(f"📊 训练统计报告 - 第{episode_count}局 | 内存: {memory_summary}")
        logging.info(f"   🎯 阶段: {phase} ({progress:.1%}) | 决策步: {agent.decision_step}")
        logging.info(f"   🏆 奖励: 平均{reward_stats['mean']:.2f} | 最高{reward_stats['max']:.2f} | 标准差{reward_stats['std']:.2f}")
        
        if train_stats:
            logging.info(f"   🧠 网络: Q值{train_stats['current_q_mean']:.2f}±{train_stats['current_q_std']:.2f} | 损失{loss_stats['mean']:.4f}")
            logging.info(f"   ⚙️  训练: LR{train_stats['learning_rate']:.2e} | ε{train_stats['epsilon']:.4f} | 梯度{train_stats['gradient_norm']:.3f}")
        
        # GPU状态报告
        if gpu_info.get('available', True):
            logging.info(f"   🖥️  GPU: {gpu_info.get('device_name', 'Unknown')}")
            logging.info(f"   💾 内存: {gpu_info.get('memory_allocated', 0):.0f}MB/{gpu_info.get('total_memory', 0):.0f}MB ({gpu_info.get('memory_usage_percent', 0):.1f}%)")
            
            if gpu_info.get('gpu_utilization') is not None:
                logging.info(f"   📈 利用率: {gpu_info['gpu_utilization']:.1f}%")
            if gpu_info.get('temperature') is not None:
                logging.info(f"   🌡️  温度: {gpu_info['temperature']:.0f}°C")
        else:
            logging.info(f"   ⚠️  {gpu_info.get('message', 'GPU监控失败')}")
        
    except Exception as e:
        logging.warning(f"统计记录失败: {e}")


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
        self.tau = 1e-3  # 软更新系数 (恢复标准值，确保训练稳定性)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)
        
        # 优先经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(REPLAY_MEMORY, alpha=0.6, beta=0.4, beta_increment=0.01)
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0  # 总帧数计数器
        self.decision_step = 0  # 决策步数计数器
        
        # 🚨 修复内存泄漏：严格限制历史记录长度，防止无限增长
        self.reward_history = []  # 只保留最近50条
        self.loss_history = []    # 只保留最近100条
        self.episode_rewards = [] # 只保留最近30条
        self.training_steps = []  # 只保留最近100条
        
        # 🚨 内存优化：严格限制历史记录长度，防止内存泄漏
        self.max_reward_history = 30    # 进一步减少到30，够用即可
        self.max_loss_history = 50      # 进一步减少到50，保持训练监控
        self.max_episode_rewards = 20   # 减少到20，足够统计
        self.max_training_steps = 50    # 大幅减少到50，减少内存占用
        
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
            
            logging.info(f"✅ 检查点加载成功: {checkpoint_path}")
            logging.info(f"   📊 恢复状态: 决策步{self.decision_step} | ε:{self.epsilon:.4f}")
            logging.info(f"   🎯 目标网络更新次数: {getattr(self, 'target_update_count', 0)}")
            
            return checkpoint.get('episode_count', 0), checkpoint.get('max_score', 0)
        else:
            logging.warning(f"❌ 检查点文件不存在: {checkpoint_path}")
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
        
        # 🚨 修复内存泄漏：优化tensor创建，减少CPU端内存堆积
        # 使用torch.stack避免list comprehension创建中间对象
        # 移除non_blocking=True防止CPU端缓存堆积
        states = torch.stack([torch.from_numpy(e[0]).float() for e in batch]).to(device).permute(0, 3, 1, 2)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float, device=device)
        next_states = torch.stack([torch.from_numpy(e[3]).float() for e in batch]).to(device).permute(0, 3, 1, 2)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=device)
        is_weights = torch.tensor(is_weights, dtype=torch.float, device=device)
        
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
        
        # 反向传播和GPU内存监控
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 🚨 更频繁的GPU缓存清理，防止OOM
        if hasattr(self, '_train_count'):
            self._train_count += 1
        else:
            self._train_count = 1
            
        # 保存统计信息（在清理前）
        target_q_mean = target_q.mean().item()
        current_q_mean = current_q.mean().item()
        current_q_max = current_q.max().item()
        current_q_min = current_q.min().item()
        current_q_std = current_q.std().item()
        reward_mean = rewards.mean().item()
        q_values_action0 = current_q_values[:, 0].mean().item()
        q_values_action1 = current_q_values[:, 1].mean().item()
        
        # 保存Dueling DQN特有的分析（在清理前）
        with torch.no_grad():
            value, advantage = self.q_network.get_value_and_advantage(states)
            value_mean = value.mean().item()
            value_std = value.std().item()
            advantage_mean = advantage.mean().item()
            advantage_std = advantage.std().item()
            advantage_action0 = advantage[:, 0].mean().item()
            advantage_action1 = advantage[:, 1].mean().item()
            
        # 保存td_errors用于优先级更新（在清理前）
        td_errors_cpu = td_errors.detach().cpu().numpy().flatten()
        
        # 保存其他统计信息（在清理前）
        is_weight_mean = is_weights.mean().item()
        advantage_range = (advantage.max() - advantage.min()).item()
        q_value_range = (current_q_values.max() - current_q_values.min()).item()
        
        # 保存目标网络统计（在清理前）
        with torch.no_grad():
            target_q_values = self.target_network(states)
            target_q_selected = target_q_values.gather(1, actions.unsqueeze(1))
            network_diff_mean = (current_q - target_q_selected).abs().mean().item()
            network_diff_max = (current_q - target_q_selected).abs().max().item()
            target_q_values_mean = target_q_values.mean().item()
            target_q_values_std = target_q_values.std().item()
        
        # 🚨 内存优化：更频繁的清理，防止内存堆积
        if self._train_count % 50 == 0:  # 从100改为50，更频繁清理
            # GPU内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated(device) / 1024**2
                reserved_memory = torch.cuda.memory_reserved(device) / 1024**2
                if current_memory > 150:  # 降低阈值，更早清理
                    logging.warning(f"🚨 GPU内存较高: {current_memory:.0f}MB(已用)/{reserved_memory:.0f}MB(预留)")
                    torch.cuda.empty_cache()
            
            # 🚨 强制CPU内存垃圾回收，清理循环引用
            import gc
            gc.collect()  # 先回收一般垃圾
            gc.collect()  # 再次回收循环引用
            
            # 🚨 清理大的tensor变量引用
            try:
                del states, next_states, current_q_values, target_q, td_errors
            except NameError:
                pass
        
        # 返回训练统计信息（包括Dueling DQN特有的分析）
        train_stats = {
            'loss': loss.item(),
            'current_q_mean': current_q_mean,
            'current_q_max': current_q_max,
            'current_q_min': current_q_min,
            'current_q_std': current_q_std,
            'target_q_mean': target_q_mean,
            'reward_mean': reward_mean,
            'q_values_action0': q_values_action0,
            'q_values_action1': q_values_action1
        }
        
        # 添加Dueling DQN特有的分析（使用预先保存的值）
        train_stats['value_mean'] = value_mean
        train_stats['value_std'] = value_std
        train_stats['advantage_mean'] = advantage_mean
        train_stats['advantage_std'] = advantage_std
        train_stats['advantage_action0'] = advantage_action0
        train_stats['advantage_action1'] = advantage_action1
        
        # 添加详细GPU监控
        if torch.cuda.is_available():
            train_stats['gpu_memory_used'] = torch.cuda.memory_allocated(device) / 1024**2
            train_stats['gpu_memory_cached'] = torch.cuda.memory_reserved(device) / 1024**2
            train_stats['gpu_memory_max'] = torch.cuda.max_memory_allocated(device) / 1024**2
            
            # GPU利用率监控
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    train_stats['gpu_utilization'] = float(result.stdout.strip())
            except:
                train_stats['gpu_utilization'] = None
        
        # 更新优先级（使用预先保存的值）
        self.memory.update_priorities(idxs, td_errors_cpu)
        
        # 🚨 修复内存泄漏：严格控制历史记录，立即清理
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss.item())
            # 🚨 内存优化：立即清理而非等到超限，保持固定大小
            if len(self.loss_history) >= self.max_loss_history:
                # 保留最新的一半，立即释放旧数据
                old_history = self.loss_history
                self.loss_history = self.loss_history[-self.max_loss_history//2:]
                del old_history  # 显式删除旧list
                
            self.training_steps.append(self.decision_step)
            if len(self.training_steps) >= self.max_training_steps:
                old_steps = self.training_steps
                self.training_steps = self.training_steps[-self.max_training_steps//2:]
                del old_steps  # 显式删除旧list
        
        # 软更新目标网络
        self.soft_update_target_network()
        
        # 更新学习率 (每10000个决策步调用一次，而不是每次训练都调用)
        if self.decision_step % 10000 == 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"🔧 学习率更新: {current_lr:.2e} (步数:{self.decision_step})")
            if current_lr < 5e-5:
                logging.warning(f"⚠️  学习率过低警告! 当前:{current_lr:.2e}, 可能影响学习效果")
        
        # 添加PER相关统计信息
        train_stats['td_error_mean'] = np.abs(td_errors_cpu).mean()
        train_stats['td_error_max'] = np.abs(td_errors_cpu).max()
        train_stats['is_weight_mean'] = is_weight_mean
        train_stats['per_beta'] = self.memory.beta
        
        # 训练健康监控指标（使用预先保存的值）
        train_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        train_stats['epsilon'] = self.epsilon
        train_stats['advantage_range'] = advantage_range
        train_stats['q_value_range'] = q_value_range
        
        # 目标网络与主网络差异监控（使用预先保存的值）
        train_stats['network_diff_mean'] = network_diff_mean
        train_stats['network_diff_max'] = network_diff_max
        train_stats['target_q_values_mean'] = target_q_values_mean
        train_stats['target_q_values_std'] = target_q_values_std
        
        # 额外的网络差异指标（使用预先保存的值）
        train_stats['target_main_q_diff'] = network_diff_mean
        train_stats['target_main_q_ratio'] = (target_q_values_mean / (current_q_mean + 1e-8))
        
        # 训练稳定性指标
        train_stats['gradient_norm'] = sum(p.grad.norm().item() for p in self.q_network.parameters() if p.grad is not None)
        train_stats['target_update_count'] = getattr(self, 'target_update_count', 0)
        
        # 网络收敛监控（使用预先保存的值）
        train_stats['q_value_stability'] = current_q_std / (abs(current_q_mean) + 1e-8)
        train_stats['action_preference'] = abs(q_values_action0 - q_values_action1)
        
        # 检查训练停滞预警和GPU状态
        if train_stats['learning_rate'] < 1e-6:
            logging.warning("⚠️  警告: 学习率过低 ({:.2e})，可能导致训练停滞！".format(train_stats['learning_rate']))
        
        if train_stats['advantage_std'] < 0.01:
            logging.warning("⚠️  警告: Advantage标准差过小 ({:.4f})，动作区分度不足！".format(train_stats['advantage_std']))
            
        # GPU状态检查 - 修复错误的内存监控逻辑
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(device) / 1024**2
            reserved_memory = torch.cuda.memory_reserved(device) / 1024**2
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
            
            # 使用预留内存与总内存的比例进行检查
            if reserved_memory > total_memory * 0.9:
                logging.warning(f"⚠️  GPU内存预留接近极限: {reserved_memory:.0f}MB/{total_memory:.0f}MB ({reserved_memory/total_memory*100:.1f}%)")
                torch.cuda.empty_cache()
        
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
            logging.info(f"🎯 目标网络更新 #{self.target_update_count} | 平均参数变化: {avg_change:.6f} | τ={self.tau}")
    
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
    # 获取初始化时的内存状态
    try:
        import psutil
        process = psutil.Process()
        init_cpu_mem = process.memory_info().rss / 1024 / 1024
        init_mem_info = f"CPU:{init_cpu_mem:.0f}MB"
        if torch.cuda.is_available():
            init_gpu_mem = torch.cuda.memory_allocated(device) / 1024**2
            init_mem_info += f" | GPU:{init_gpu_mem:.0f}MB"
    except ImportError:
        init_mem_info = "N/A"
    
    logging.info(f"🚀 Dueling DQN 初始化 | 设备:{device_info} | 批次:{BATCH} | 内存:{init_mem_info}")
    
    # 初始化GPU监控
    if torch.cuda.is_available():
        gpu_info = monitor_gpu_status()
        logging.info(f"🖥️  GPU初始化: {gpu_info.get('device_name', 'Unknown')}")
        logging.info(f"   总内存: {gpu_info.get('total_memory', 0):.0f}MB | 当前使用: {gpu_info.get('memory_allocated', 0):.1f}MB")
        if gpu_info.get('temperature'):
            logging.info(f"   温度: {gpu_info['temperature']:.0f}°C")
        
        # 设置GPU内存监控
        torch.cuda.empty_cache()  # 清理未使用的缓存
        logging.info(f"   ✨ GPU缓存已清理，开始训练监控")
    
    # 初始化游戏环境
    game_state = game.GameState()
    
    # 初始化增强版智能体
    agent = EnhancedDuelingDQNAgent(ACTIONS)
    
    logging.info(f"⚙️  核心优化: LayerNorm+软更新(τ={agent.tau:.3f})+PER(α={agent.memory.alpha},β={agent.memory.beta:.2f})")
    logging.info(f"🎯 科学奖励机制: 管道+20分 | 衰减生存1e-4*(0.999^t) | 死亡-2分 | 潜能函数塑形")
    logging.info(f"🚨 内存优化配置: 缓冲区{REPLAY_MEMORY} | 批次{BATCH} | GPU内存40% | 50步清理")
    
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
    
    # 🚨 最小化奖励统计变量
    pipe_rewards_count = 0        # 管道奖励次数  
    survival_frames = 0           # 当前局存活帧数
    
    # 获取初始内存状态
    try:
        import psutil
        process = psutil.Process()
        initial_cpu_mem = process.memory_info().rss / 1024 / 1024
        mem_info = f"CPU:{initial_cpu_mem:.0f}MB"
        if torch.cuda.is_available():
            initial_gpu_mem = torch.cuda.memory_allocated(device) / 1024**2
            mem_info += f" | GPU:{initial_gpu_mem:.0f}MB"
    except ImportError:
        mem_info = "N/A"
    
    logging.info(f"🎮 开始训练 | 观察:{OBSERVE} 探索:{EXPLORE} | {500//FRAME_PER_ACTION}决策/秒 | 初始内存:{mem_info}")
    
    # 训练开始时的系统检查
    if torch.cuda.is_available():
        logging.info(f"🖥️  GPU训练准备就绪，开启实时监控...")
    
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
        # 🚨 优化内存：使用concatenate而不是append，避免额外内存分配
        s_t1 = np.concatenate([x_t1, s_t[:, :, :3]], axis=2)
        
        episode_reward += r_t
        agent.step += 1
        
        # 🚨 极简化奖励统计，最小化内存使用
        survival_frames += 1
        
        # 🚨 只在关键时刻记录，减少内存和计算
        if r_t > 15:  # 管道奖励 (20分)
            pipe_rewards_count += 1
            # 大幅减少日志输出，每10个管道报告一次
            if pipe_rewards_count % 10 == 1:
                logging.info(f"🏆 管道里程碑! {pipe_rewards_count}个管道 | 总分: {episode_reward:.1f}")
        # 🚨 移除死亡惩罚的详细日志，减少输出
        # 🚨 移除复杂的奖励累加，只保留必要统计
        
        # 只在决策帧存储经验和训练
        if agent.step % FRAME_PER_ACTION == 0:
            agent.decision_step += 1
            agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
            
            # 训练网络（仅训练期）
            if agent.decision_step > OBSERVE:
                train_stats = agent.train()
                
                # 每100步简要训练日志 + 内存监控（修复BUG：确保在训练期）
                if agent.decision_step % 100 == 0 and train_stats is not None:
                    avg_reward = np.mean(agent.reward_history[-10:]) if len(agent.reward_history) >= 10 else 0
                    
                    # 添加内存使用监控
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        memory_info = f" | 内存:{memory_mb:.0f}MB"
                    except ImportError:
                        memory_info = ""
                    
                    logging.info(f"🧠 训练步{agent.decision_step} | 损失:{train_stats['loss']:.4f} | 近10局均分:{avg_reward:.2f} | Q值:{train_stats['current_q_mean']:.2f}±{train_stats['current_q_std']:.2f}{memory_info}")
            
            # 软更新目标网络已在train()中完成
            
            # 训练监控 - 每250步详细日志（更频繁的监控）
            if agent.decision_step > OBSERVE and agent.decision_step % 250 == 0 and train_stats is not None:
                avg_reward = np.mean(agent.reward_history[-100:]) if agent.reward_history else 0
                
                # 🎯 新奖励机制下的表现分析 + 内存监控
                estimated_pipes_per_episode = avg_reward / 20 if avg_reward > 0 else 0
                logging.info(f"🎯 奖励分析: 平均{avg_reward:.2f}分/局 ≈ {estimated_pipes_per_episode:.1f}个管道/局")
                
                # 🚨 内存使用监控
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logging.info(f"🧠 系统内存: {memory_mb:.0f}MB | 奖励历史:{len(agent.reward_history)} | 损失历史:{len(agent.loss_history)}")
                    if memory_mb > 8000:  # 超过8GB警告
                        logging.warning(f"⚠️ 内存使用过高: {memory_mb:.0f}MB，接近容器限制")
                except ImportError:
                    pass  # psutil不可用时忽略
                
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
                    gpu_util_text = f" | 利用率:{train_stats['gpu_utilization']:.1f}%" if train_stats.get('gpu_utilization') else ""
                    logging.info(f"   💾 GPU:{train_stats['gpu_memory_used']:.0f}MB | 缓存:{train_stats['gpu_memory_cached']:.0f}MB{gpu_util_text}")
                    
                    # GPU健康检查 - 使用预留内存作为参考
                    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
                    if train_stats['gpu_memory_cached'] > total_memory * 0.8:
                        logging.warning(f"   ⚠️  GPU内存预留率较高: {train_stats['gpu_memory_cached']:.0f}MB/{total_memory:.0f}MB ({train_stats['gpu_memory_cached']/total_memory*100:.1f}%)")
                    
                    if train_stats.get('gpu_utilization') and train_stats['gpu_utilization'] < 50:
                        logging.warning(f"   ⚠️  GPU利用率较低: {train_stats['gpu_utilization']:.1f}%，可能存在性能瓶颈")
                
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
            
            # 🚨 内存优化：严格控制历史记录增长，立即清理防止泄漏
            agent.reward_history.append(episode_reward)
            # 立即清理而非等到超限，保持固定大小
            if len(agent.reward_history) >= agent.max_reward_history:
                old_reward_history = agent.reward_history
                agent.reward_history = agent.reward_history[-agent.max_reward_history//2:]
                del old_reward_history  # 显式删除旧数据
                
            agent.episode_rewards.append(episode_reward)
            if len(agent.episode_rewards) >= agent.max_episode_rewards:
                old_episode_rewards = agent.episode_rewards
                agent.episode_rewards = agent.episode_rewards[-agent.max_episode_rewards//2:]
                del old_episode_rewards  # 显式删除旧数据
            
            # 🚨 更频繁的垃圾回收和内存清理，防止内存泄漏
            if episode_count % 10 == 0:
                import gc
                gc.collect()
                # 强制清理PyTorch缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 🚨 最小化奖励统计报告
            if episode_count % 50 == 0 and pipe_rewards_count > 0:  # 每50局报告一次
                logging.info(f"🎯 进度检查点 [第{episode_count}局]: {pipe_rewards_count}个管道 | 分数: {episode_reward:.1f}")
            
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
            
            # 游戏结束日志 - 简洁格式 + 完整内存监控
            memory_info = ""
            try:
                import psutil
                process = psutil.Process()
                cpu_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_info += f" | CPU内存:{cpu_memory_mb:.0f}MB"
                
                # GPU内存监控
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**2
                    gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**2
                    memory_info += f" | GPU:{gpu_memory_used:.0f}MB/{gpu_memory_cached:.0f}MB"
                    
            except ImportError:
                memory_info = " | 内存:N/A"
            
            logging.info(f"🎮 游戏{episode_count} | {phase} | 分数:{episode_reward:.2f} | 最高:{max_score:.2f} | ε:{agent.epsilon:.4f}{memory_info}")
            
            # 每10局检查训练健康状况
            if episode_count % 10 == 0 and episode_count > 0:
                recent_scores = agent.reward_history[-10:] if len(agent.reward_history) >= 10 else agent.reward_history
                avg_recent = np.mean(recent_scores) if recent_scores else 0
                
                # GPU健康检查
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
                    reserved_gpu_memory = torch.cuda.memory_reserved(device) / 1024**2
                    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
                    
                    # 使用预留内存作为参考，如果预留内存超过总内存的80%则警告
                    if reserved_gpu_memory > total_gpu_memory * 0.8:
                        logging.warning(f"⚠️  GPU内存预留量较高: {reserved_gpu_memory:.0f}MB/{total_gpu_memory:.0f}MB ({reserved_gpu_memory/total_gpu_memory*100:.1f}%)")
                        torch.cuda.empty_cache()  # 清理缓存
                        logging.info(f"   ✨ 已清理GPU缓存")
                
                if agent.decision_step > OBSERVE:
                    # 检查性能停滞
                    if len(agent.reward_history) >= 25:
                        last_50 = np.mean(agent.reward_history[-50:])
                        prev_25 = np.mean(agent.reward_history[-50:-25]) if len(agent.reward_history) >= 50 else last_50
                        improvement = last_50 - prev_25
                        
                        # 🎯 更新性能停滞检查阈值（新奖励机制下）
                        # 考虑新奖励机制：管道奖励20分，期望改善阈值提高
                        improvement_threshold = 1.0  # 从0.1提升到1.0，匹配新奖励规模
                        if improvement < improvement_threshold and agent.decision_step > OBSERVE + 5000:
                            logging.warning(f"⚠️  性能停滞警告! 近50局平均分:{last_50:.2f}, 改善:{improvement:.2f} (阈值:{improvement_threshold})")
                            
                            # 🎯 新奖励机制下的性能解释
                            estimated_pipes = int(last_50 / 20)  # 估算通过的管道数
                            logging.info(f"   📊 性能解读: 约通过{estimated_pipes}个管道/局 | 管道奖励占比≈{(estimated_pipes*20/last_50*100) if last_50 > 0 else 0:.1f}%")
                    
                    # 获取当前内存状态
                    try:
                        import psutil
                        process = psutil.Process()
                        cpu_mem = process.memory_info().rss / 1024 / 1024
                        mem_info = f"CPU:{cpu_mem:.0f}MB"
                        if torch.cuda.is_available():
                            gpu_mem = torch.cuda.memory_allocated(device) / 1024**2
                            mem_info += f" | GPU:{gpu_mem:.0f}MB"
                    except ImportError:
                        mem_info = "N/A"
                    
                    logging.info(f"📊 第{episode_count}局 | 近10局平均:{avg_recent:.2f} | 决策步:{agent.decision_step} | 内存:{mem_info}")
            
            episode_reward = 0
            
            # 🚨 重置奖励统计（最小化）
            pipe_rewards_count = 0
            survival_frames = 0
            
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
                    'reward_history': agent.reward_history[-50:],   # 保存最近50局的奖励
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
                
                # 保存时的GPU状态
                if torch.cuda.is_available():
                    gpu_info = monitor_gpu_status()
                    logging.info(f"   🖥️  GPU: {gpu_info.get('memory_allocated', 0):.0f}MB已用 | {gpu_info.get('memory_usage_percent', 0):.1f}%利用率")
        
            # 阶段转换提示 (只在决策帧检查)
            if agent.decision_step == OBSERVE:
                logging.info(f"🎆 观察期结束! 开始 Dueling DQN 训练...")
                # 训练开始时的GPU检查
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_info = monitor_gpu_status()
                    logging.info(f"   🖥️  训练初始GPU状态: {gpu_info.get('memory_allocated', 0):.0f}MB已用")
            elif agent.decision_step == OBSERVE + EXPLORE:
                logging.info(f"🏆 进入利用期! 主要使用已学策略...")
                # 进入利用期时的GPU检查
                if torch.cuda.is_available():
                    gpu_info = monitor_gpu_status()
                    logging.info(f"   🖥️  利用期GPU状态: {gpu_info.get('memory_allocated', 0):.0f}MB已用 | {gpu_info.get('memory_usage_percent', 0):.1f}%利用率")
            
            # 进度提示和GPU监控
            if agent.decision_step % 5000 == 0 and agent.decision_step > 0:
                # 进度信息
                if agent.decision_step < OBSERVE:
                    remaining = OBSERVE - agent.decision_step
                    logging.info(f"🔍 观察期: 还需{remaining}步开始训练")
                elif agent.decision_step < OBSERVE + EXPLORE:
                    remaining = OBSERVE + EXPLORE - agent.decision_step
                    logging.info(f"🔍 探索期: 还需{remaining}步进入利用期")
                
                # 定期GPU状态检查
                if torch.cuda.is_available():
                    gpu_info = monitor_gpu_status()
                    logging.info(f"🖥️  GPU状态: 使用{gpu_info.get('memory_usage_percent', 0):.1f}% | 温度{gpu_info.get('temperature', 'N/A')}°C")
                    
                    # GPU健康检查
                    if gpu_info.get('memory_usage_percent', 0) > 85:
                        logging.warning(f"⚠️  GPU内存使用率过高: {gpu_info['memory_usage_percent']:.1f}%")
                    if gpu_info.get('temperature') and gpu_info['temperature'] > 80:
                        logging.warning(f"⚠️  GPU温度过高: {gpu_info['temperature']}°C")


if __name__ == "__main__":
    main()