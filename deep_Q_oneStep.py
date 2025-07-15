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
OBSERVE = 10000
EXPLORE = 30000
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 20000
BATCH = 512  # 进一步增大批次：4帧决策释放GPU资源，支持更大批次训练
FRAME_PER_ACTION = 4  # 优化：4帧一个动作，匹配状态帧数，提高信息效率

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
# 🎯 智能目标网络选择：如果最佳奖励网络比定时网络更新，优先使用最佳奖励网络


class FixedOptimizedDQN(nn.Module):
    """修复的网络结构：输出单步Q值而非多步"""
    def __init__(self, actions):
        super(FixedOptimizedDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 修改：增大网络规模提高GPU利用率
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  # 1024输入 → 1024输出
        self.fc2 = nn.Linear(1024, 512)         # 1024输入 → 512输出  
        self.fc3 = nn.Linear(512, actions)      # 512输入 → 2输出
        
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
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # 新增中间层
        x = self.fc3(x)
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
        
        # 智能目标网络选择
        self.best_reward_network = FixedOptimizedDQN(actions).to(device)  # 最高奖励网络
        self.best_reward_network.load_state_dict(self.q_network.state_dict())
        self.best_reward = -float('inf')  # 最佳奖励记录
        self.best_reward_step = 0  # 最佳奖励对应的步数
        self.last_target_update_step = 0  # 上次定时更新步数
        
        # 优化器
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=REPLAY_MEMORY)
        
        # 训练参数
        self.epsilon = 1.0
        self.step = 0  # 总帧数计数器
        self.decision_step = 0  # 决策步数计数器
        self.reward_history = []
        
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
        """选择动作（专门的观察期预训练模型策略）"""
        if self.decision_step < OBSERVE:
            # 观察期专用策略：50%随机探索 + 50%预训练模型
            if random.random() < 0.5:
                return random.randrange(self.actions)  # 50%随机探索
            else:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    return q_values.max(1)[1].item()  # 50%预训练模型
        else:
            # 探索期+利用期：正常ε-贪婪策略
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
        
        # 返回训练统计信息（包括GPU使用情况）
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
        
        # 添加GPU内存使用监控
        if torch.cuda.is_available():
            train_stats['gpu_memory_used'] = torch.cuda.memory_allocated(device) / 1024**2  # MB
            train_stats['gpu_memory_cached'] = torch.cuda.memory_reserved(device) / 1024**2  # MB
        
        return train_stats
    
    def update_target_network(self):
        """阶段性智能目标网络更新策略（使用决策步数）"""
        if self.decision_step % 350 == 0:  # 350个决策更新周期：平衡稳定性和响应性
            
            # 阶段判断（使用决策步数）
            if self.decision_step < OBSERVE:
                # 观察阶段：只使用最佳网络
                strategy = "observe_best_only"
            elif self.decision_step < OBSERVE + 500:
                # 探索早期：强制使用最佳网络
                strategy = "explore_early_best_only"
            else:
                # 探索后期+利用期：智能选择
                strategy = "intelligent_selection"
            
            # 执行对应策略
            if strategy == "observe_best_only":
                if self.best_reward_step > 0:
                    # 有最佳网络，使用最佳网络
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    logging.info(f"")
                    logging.info(f"🔬📚 OBSERVE PHASE -> BEST NETWORK ONLY 📚🔬")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 阶段策略: 观察期专注积累经验，仅使用最佳网络")
                    logging.info(f"   └─ 决策步数: {self.decision_step}/{OBSERVE}")
                    logging.info(f"")
                else:
                    # 观察期无最佳网络，先使用当前网络并提示
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    logging.info(f"")
                    logging.info(f"🔬⚡ OBSERVE PHASE -> WAITING FOR BEST NETWORK ⚡🔬")
                    logging.info(f"   ├─ 阶段策略: 观察期暂用当前网络，等待首个最佳网络出现")
                    logging.info(f"   ├─ 当前表现: 随机策略，寻找突破性表现")
                    logging.info(f"   └─ 决策步数: {self.decision_step}/{OBSERVE}")
                    logging.info(f"")
                    
            elif strategy == "explore_early_best_only":
                if self.best_reward_step > 0:
                    # 探索早期：强制使用最佳网络
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    remaining_steps = OBSERVE + 500 - self.decision_step
                    logging.info(f"")
                    logging.info(f"🚀💎 EXPLORE EARLY -> FORCE BEST NETWORK 💎🚀")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 阶段策略: 探索前500步强制使用最佳网络稳定学习")
                    logging.info(f"   └─ 剩余强制步数: {remaining_steps}步")
                    logging.info(f"")
                else:
                    # 无最佳网络，使用当前网络
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    remaining_steps = OBSERVE + 500 - self.decision_step
                    logging.info(f"")
                    logging.info(f"🚀⚡ EXPLORE EARLY -> REGULAR UPDATE ⚡🚀")
                    logging.info(f"   ├─ 阶段策略: 探索早期，尚无最佳网络可用")
                    logging.info(f"   └─ 剩余早期步数: {remaining_steps}步")
                    logging.info(f"")
                    
            else:  # intelligent_selection
                # 探索后期+利用期：智能选择机制
                has_recent_best = (self.best_reward_step > self.last_target_update_step and 
                                 self.step - self.best_reward_step < 1000)  # 1000步内的最佳网络
                
                if has_recent_best:
                    # 使用最佳奖励网络
                    self.target_network.load_state_dict(self.best_reward_network.state_dict())
                    network_age = self.decision_step - self.best_reward_step
                    phase = "探索后期" if self.decision_step < OBSERVE + EXPLORE else "利用期"
                    logging.info(f"")
                    logging.info(f"🎯🔥 {phase.upper()} -> INTELLIGENT BEST NETWORK 🔥🎯")
                    logging.info(f"   ├─ 最佳奖励: {self.best_reward:.3f}")
                    logging.info(f"   ├─ 网络年龄: {network_age}步 (< 1000步有效期)")
                    logging.info(f"   └─ 切换原因: 智能选择最佳历史表现网络")
                    logging.info(f"")
                else:
                    # 使用定时网络（默认策略）
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    self.last_target_update_step = self.decision_step
                    phase = "探索后期" if self.decision_step < OBSERVE + EXPLORE else "利用期"
                    logging.info(f"")
                    logging.info(f"🔄⏰ {phase.upper()} -> INTELLIGENT REGULAR UPDATE ⏰🔄")
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
        # 观察期降低门槛，任何正奖励都保存
        should_update = False
        if self.decision_step < OBSERVE:
            # 观察期：任何正向改善都保存
            should_update = (episode_reward > self.best_reward and episode_reward > 0)
        else:
            # 训练期：严格按最佳奖励更新
            should_update = (episode_reward > self.best_reward)
            
        if should_update:
            old_best = self.best_reward
            improvement = episode_reward - old_best if old_best > -float('inf') else episode_reward
            self.best_reward = episode_reward
            self.best_reward_step = self.decision_step
            self.best_reward_network.load_state_dict(self.q_network.state_dict())
            
            phase = "观察期" if self.decision_step < OBSERVE else "训练期"
            logging.info(f"")
            logging.info(f"🏆⭐ NEW BEST REWARD NETWORK SAVED! ({phase}) ⭐🏆")
            logging.info(f"   ├─ 新纪录: {episode_reward:.3f}")
            if old_best > -float('inf'):
                logging.info(f"   ├─ 提升幅度: +{improvement:.3f} (从 {old_best:.3f})")
            logging.info(f"   ├─ 决策步数: {self.decision_step}")
            if self.step < OBSERVE:
                logging.info(f"   └─ 观察期发现正向表现，保存为参考网络")
            else:
                logging.info(f"   └─ 此网络将用于未来1000步的目标网络更新")
            logging.info(f"")
    
    def update_epsilon(self):
        """更新探索率（使用决策步数）"""
        if self.decision_step < OBSERVE:
            # 观察期：使用专门的混合策略，不依赖epsilon
            self.epsilon = 0.5  # 仅作为显示用，实际不使用
        elif self.decision_step < OBSERVE + EXPLORE:
            # 探索期：从1.0线性衰减到0.001
            self.epsilon = 1.0 - (self.decision_step - OBSERVE) / EXPLORE * (1.0 - FINAL_EPSILON)
        else:
            # 利用期：固定为0.001
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
        
        # 估算内存需求
        batch_memory_mb = (BATCH * 4 * 80 * 80 * 4) / (1024 * 1024)  # 4字节float32
        logging.info(f"预计批次内存需求: {batch_memory_mb:.1f}MB (BATCH={BATCH})")
        logging.info(f"GPU显存: 4096MB, 预计利用率: {batch_memory_mb/4096*100:.1f}%")
    
    # 初始化游戏环境
    game_state = game.GameState()
    
    # 初始化智能体
    agent = DQNAgent(ACTIONS)
    
    # 加载预训练模型作为初始智能体
    pretrained_model = "saved_networks/bird-dqn-oneStep-1300.pth"
    if os.path.exists(pretrained_model):
        logging.info(f"🔄 加载预训练模型作为初始智能体: {pretrained_model}")
        agent.q_network.load_state_dict(torch.load(pretrained_model, map_location=device))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        agent.best_reward_network.load_state_dict(agent.q_network.state_dict())
        
        # 保持正常训练流程：观察期->探索期->利用期
        agent.step = 0  # 从观察期开始（帧数）
        agent.decision_step = 0  # 从观察期开始（决策步数）
        agent.epsilon = 1.0  # 从观察期的ε值开始
        
        # 设置预训练模型的基础表现作为最佳奖励参考
        agent.best_reward = 30.0  # 假设1300轮模型有较好表现
        agent.best_reward_step = 0
        
        logging.info(f"✅ 预训练模型加载成功！")
        logging.info(f"📊 训练状态: 从观察期开始，使用成熟智能体玩游戏")
        logging.info(f"🎯 观察期专用策略: 50%随机探索 + 50%预训练模型")
        logging.info(f"🚀 预训练智能体将在观察期展示技能，探索期作为目标网络基础")
        logging.info(f"🎮 预期观察期表现: 高分游戏，快速积累高质量经验")
        logging.info(f"📈 优势: 预训练模型即使在观察期也能提供高质量决策")
    else:
        logging.info(f"⚠️ 预训练模型不存在，从头开始训练")
        logging.info(f"📉 观察期使用标准策略: 100%随机探索")
    
    # 获取初始状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = agent.preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
    # 训练统计
    episode_count = 0  # 重新开始计数，但使用预训练智能体
    episode_reward = 0
    max_score = 0
    action_index = 0
    
    logging.info("开始训练...")
    logging.info(f"观察步数: {OBSERVE}, 探索步数: {EXPLORE}, 批次大小: {BATCH}")
    logging.info(f"决策间隔: {FRAME_PER_ACTION}帧/动作, 游戏速度: 500FPS")
    logging.info(f"优化配置: 4帧状态匹配4帧决策间隔, 大批次训练({BATCH})")
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
        
        # 每帧都增加帧数
        agent.step += 1
        
        # 只在决策帧存储经验和训练
        if agent.step % FRAME_PER_ACTION == 0:
            agent.decision_step += 1  # 增加决策步数
            agent.store_transition(s_t, action_index, r_t, s_t1, terminal)
            
            # 训练网络（仅训练期）
            if agent.decision_step > OBSERVE:
                train_stats = agent.train()
            
            # 更新目标网络（所有阶段都需要，按350个决策周期）
            agent.update_target_network()
            
            # 记录训练信息 (降低频率以便看清目标网络切换信息)
            if agent.decision_step > OBSERVE and agent.decision_step % 1000 == 0 and train_stats is not None:
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
                logging.info(f"  目标Q值: {train_stats['target_q_mean']:.3f} | 批次奖励: {train_stats['reward_mean']:.3f}")
                
                # 添加GPU使用情况和优化效果日志
                if torch.cuda.is_available() and 'gpu_memory_used' in train_stats:
                    logging.info(f"  GPU内存 - 已用: {train_stats['gpu_memory_used']:.1f}MB | 缓存: {train_stats['gpu_memory_cached']:.1f}MB")
                    
                    # 计算当前配置的理论改善
                    original_decisions_per_sec = 250  # 原2帧决策
                    current_decisions_per_sec = 500 // FRAME_PER_ACTION
                    decision_efficiency = (original_decisions_per_sec - current_decisions_per_sec) / original_decisions_per_sec * 100
                    
                    logging.info(f"  优化效果 - 决策效率提升: {decision_efficiency:.0f}% | 批次规模: {BATCH} (vs原64)")
                    logging.info(f"  信息效率 - 状态重叠度: 0% (vs原75%) | 每步信息增益: 4x")
                    
                    # 智能目标网络状态（与实际更新逻辑保持一致）
                    if agent.best_reward_step > 0:
                        target_network_age = agent.decision_step - agent.best_reward_step
                        
                        if agent.decision_step < OBSERVE:
                            # 观察期：有最佳网络就使用
                            network_type = "🎯最佳网络"
                            status_icon = "📚观察期"
                        elif agent.decision_step < OBSERVE + 500:
                            # 探索早期：有最佳网络就使用
                            network_type = "🎯最佳网络"
                            status_icon = "🚀早期强制"
                        else:
                            # 探索后期+利用期：智能选择逻辑
                            is_using_best = (agent.best_reward_step > agent.last_target_update_step and target_network_age < 1000)
                            network_type = "🎯最佳网络" if is_using_best else "🔄定时网络"
                            status_icon = "✅有效" if target_network_age < 1000 else "⏰过期"
                            
                        logging.info(f"  智能目标网络 - 当前使用: {network_type} | 最佳奖励: {agent.best_reward:.3f} | 年龄: {target_network_age}步 ({status_icon})")
                    else:
                        logging.info(f"  智能目标网络 - 当前使用: 🔄定时网络 | 状态: 尚未发现最佳网络")
            
            # 更新探索率
            agent.update_epsilon()
        
        # 更新状态
        s_t = s_t1
        
        # 游戏结束处理
        if terminal:
            episode_count += 1
            agent.reward_history.append(episode_reward)
            
            # 保存最佳模型（先保存文件，再更新内存网络）
            if episode_reward > max_score:
                max_score = episode_reward
                os.makedirs("saved_networks", exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          f"saved_networks/bird-dqn-oneStep-best-{episode_reward:.3f}.pth")
                logging.info(f"🏆 新的最佳模型已保存: bird-dqn-oneStep-best-{episode_reward:.3f}.pth (得分: {episode_reward:.3f})")
            
            # 更新最佳奖励网络（在内存中）
            agent.update_best_reward_network(episode_reward)
                
            # 计算训练状态
            if agent.decision_step < OBSERVE:
                status = f"观察期 ({agent.decision_step}/{OBSERVE})"
                strategy_display = "50%随机+50%预训练"
            elif agent.decision_step < OBSERVE + EXPLORE:
                status = f"探索期 ({agent.decision_step}/{OBSERVE + EXPLORE})"
                strategy_display = f"ε: {agent.epsilon:.4f}"
            else:
                status = "利用期"
                strategy_display = f"ε: {agent.epsilon:.4f}"
            
            # 当前目标网络状态（与实际更新逻辑保持一致）
            if agent.best_reward_step > 0:
                if agent.decision_step < OBSERVE:
                    # 观察期：有最佳网络就使用
                    network_type = "🎯最佳"
                elif agent.decision_step < OBSERVE + 500:
                    # 探索早期：有最佳网络就使用
                    network_type = "🎯最佳"
                else:
                    # 探索后期+利用期：智能选择逻辑
                    target_network_age = agent.decision_step - agent.best_reward_step
                    is_using_best = (agent.best_reward_step > agent.last_target_update_step and target_network_age < 1000)
                    network_type = "🎯最佳" if is_using_best else "🔄定时"
            else:
                network_type = "🔄定时"
            
            logging.info(f"游戏 {episode_count} 结束 | 决策步数: {agent.decision_step} | {status} | "
                        f"得分: {episode_reward:.3f} | 最高分: {max_score:.3f} | 策略: {strategy_display} | 目标网络: {network_type}")
            
            episode_reward = 0
            
            # 定期保存模型
            if episode_count % 100 == 0:
                os.makedirs("saved_networks", exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          f"saved_networks/bird-dqn-oneStep-{episode_count}.pth")
                logging.info(f"定期模型已保存: bird-dqn-oneStep-{episode_count}.pth")
        
        # 阶段提示
        if agent.decision_step == OBSERVE:
            logging.info("🎯 重要节点：观察期结束，开始DQN训练！预期得分开始提升...")
        elif agent.decision_step == OBSERVE + EXPLORE:
            logging.info("🎯 重要节点：探索期结束，进入利用期！网络已充分学习...")
        
        # 定期提示进度
        if agent.decision_step % 5000 == 0 and agent.decision_step > 0:
            if agent.decision_step < OBSERVE:
                remaining = OBSERVE - agent.decision_step
                logging.info(f"📊 观察期进度：还需 {remaining} 步开始训练")
            elif agent.decision_step < OBSERVE + EXPLORE:
                remaining = OBSERVE + EXPLORE - agent.decision_step
                logging.info(f"📊 探索期进度：还需 {remaining} 步进入利用期")


if __name__ == "__main__":
    main()