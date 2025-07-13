#!/usr/bin/env python3
"""
AI游戏测试脚本 - 基于deep_Q_oneStep.py
加载最新保存的模型，观看AI如何游玩Flappy Bird
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import os
import glob
import numpy as np
import time
import logging
import argparse
from datetime import datetime

# 添加项目路径 - 修复Windows路径问题
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 获取项目根目录

# 🎮 启用游戏显示 - 覆盖训练时的无头模式设置
os.environ.pop('SDL_VIDEODRIVER', None)  # 移除无头模式
os.environ['SDL_VIDEODRIVER'] = ''  # 确保使用默认显示驱动

# 确保工作目录在项目根目录
os.chdir(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "game"))

try:
    # 🎮 使用可视化版本的游戏模块 (而不是快速训练版本)
    import game.wrapped_flappy_bird as game
except ImportError as e:
    print(f"❌ 无法导入游戏模块: {e}")
    print(f"📂 当前工作目录: {os.getcwd()}")
    print(f"📂 项目根目录: {project_root}")
    print("💡 请确保从项目根目录运行: python run_test.py")
    sys.exit(1)

# 设置设备 - 与deep_Q_oneStep.py保持一致  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)

# 超参数配置 - 与deep_Q_oneStep.py完全一致
ACTIONS = 2
FRAME_PER_ACTION = 4  # 与训练时保持一致

# 网络架构定义 - 直接从deep_Q_oneStep.py复制
class FixedOptimizedDQN(nn.Module):
    """修复的网络结构：输出单步Q值而非多步 - 与deep_Q_oneStep.py完全一致"""
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

class AIGameTester:
    """AI游戏测试器"""
    
    def __init__(self, target_fps=500):
        self.device = device
        self.actions = 2  # 跳跃/不跳跃
        self.network = None
        self.game_state = None
        self.target_fps = target_fps  # 目标FPS
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"test/ai_gameplay_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"🎮 AI游戏测试开始 | 日志文件: {log_filename}")
        logging.info(f"🖥️  使用设备: {self.device}")
        logging.info(f"🎯 目标FPS: {self.target_fps} (训练时为500 FPS)")
        logging.info(f"👁️  显示模式: 已启用游戏窗口显示 (覆盖训练时的无头模式)")
        
    def find_latest_model(self):
        """查找最新保存的模型 - 优先查找deep_Q_oneStep.py生成的模型"""
        model_patterns = [
            "saved_networks/bird-dqn-oneStep-*.pth",      # deep_Q_oneStep.py 主脚本
            "saved_networks/bird-dqn-pytorch-optimized-*.pth",  # 旧优化版本
            "saved_networks/*.pth"  # 其他所有模型
        ]
        
        latest_model = None
        latest_time = 0
        model_source = ""
        
        for i, pattern in enumerate(model_patterns):
            models = glob.glob(pattern)
            for model_path in models:
                model_time = os.path.getmtime(model_path)
                if model_time > latest_time:
                    latest_time = model_time
                    latest_model = model_path
                    if i == 0:
                        model_source = "deep_Q_oneStep.py"
                    elif i == 1:
                        model_source = "优化版本"
                    else:
                        model_source = "其他脚本"
        
        if latest_model:
            model_time_str = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"🔍 找到最新模型: {latest_model}")
            logging.info(f"📅 模型时间: {model_time_str}")
            logging.info(f"🏷️  模型来源: {model_source}")
        else:
            logging.error("❌ 未找到任何保存的模型文件！")
            logging.info("💡 请先运行 'python deep_Q_oneStep.py' 训练模型")
            
        return latest_model
    
    def load_specific_model(self, model_path):
        """加载指定的模型文件"""
        if not os.path.exists(model_path):
            logging.error(f"❌ 指定的模型文件不存在: {model_path}")
            return None
        
        model_time = os.path.getmtime(model_path)
        model_time_str = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"🎯 加载指定模型: {model_path}")
        logging.info(f"📅 模型时间: {model_time_str}")
        
        return model_path
    
    def load_model(self, model_path):
        """加载模型 - 基于deep_Q_oneStep.py的FixedOptimizedDQN架构"""
        try:
            # 创建网络 - 使用与deep_Q_oneStep.py完全相同的架构
            self.network = FixedOptimizedDQN(self.actions).to(self.device)
            
            # 加载检查点
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查保存格式并加载相应的权重
            if isinstance(checkpoint, dict):
                # 检查是否包含完整训练状态
                if 'q_network_state_dict' in checkpoint:
                    state_dict = checkpoint['q_network_state_dict']
                    logging.info("🔄 检测到完整训练状态格式")
                    logging.info(f"   训练步数: {checkpoint.get('step', 'Unknown')}")
                    logging.info(f"   Epsilon: {checkpoint.get('epsilon', 'Unknown')}")
                    if 'reward_history' in checkpoint:
                        history = checkpoint['reward_history']
                        if history:
                            avg_reward = sum(history[-10:]) / min(len(history), 10)
                            logging.info(f"   近期平均得分: {avg_reward:.3f}")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logging.info("🔄 检测到标准模型格式")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logging.info("🔄 检测到state_dict格式")
                else:
                    # 检查是否为deep_Q_oneStep.py的简单state_dict格式
                    if 'conv1.weight' in checkpoint and 'fc3.weight' in checkpoint:
                        state_dict = checkpoint
                        logging.info("🔄 检测到deep_Q_oneStep.py的state_dict格式")
                        logging.info("   这是来自deep_Q_oneStep.py的模型文件")
                    else:
                        # 假设整个字典就是state_dict
                        state_dict = checkpoint
                        logging.info("🔄 检测到直接字典格式")
            else:
                # 直接的state_dict格式（通常不会是这种情况）
                state_dict = checkpoint
                logging.info("🔄 检测到直接state_dict格式")
            
            # 加载网络权重
            self.network.load_state_dict(state_dict)
            self.network.eval()  # 设置为评估模式
            
            logging.info(f"✅ 模型加载成功: {model_path}")
            
            # 显示网络信息
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            
            logging.info(f"🧠 网络参数: 总数 {total_params:,} | 可训练 {trainable_params:,}")
            
            # 测试网络是否正常工作
            test_input = torch.randn(1, 4, 80, 80).to(self.device)
            with torch.no_grad():
                test_output = self.network(test_input)
                logging.info(f"🧪 网络测试: 输入 {list(test_input.shape)} -> 输出 {list(test_output.shape)}")
                logging.info(f"   Q值范围: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            logging.error("💡 解决方案:")
            logging.error("   1. 确保模型来自 deep_Q_oneStep.py 训练")
            logging.error("   2. 检查模型文件是否完整")
            logging.error("   3. 尝试重新训练模型")
            logging.error(f"   4. 运行: python deep_Q_oneStep.py")
            return False
    
    def preprocess_state(self, state):
        """预处理游戏状态（与训练时保持一致）"""
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return state
    
    def get_action(self, state_tensor):
        """获取AI的动作决策"""
        with torch.no_grad():
            q_values = self.network(state_tensor)
            action = q_values.max(1)[1].item()
            
            # 返回动作和Q值（用于显示）
            q_vals = q_values.squeeze().cpu().numpy()
            return action, q_vals
    
    def run_game(self, max_episodes=5, show_q_values=True, delay=0.01):
        """运行AI游戏测试"""
        if self.network is None:
            logging.error("❌ 请先加载模型！")
            return
            
        # 初始化游戏
        self.game_state = game.GameState()
        
        logging.info(f"🎮 开始AI游戏测试 | 最大局数: {max_episodes}")
        logging.info(f"⚙️  显示Q值: {show_q_values} | 延迟: {delay}秒")
        logging.info("📝 控制说明:")
        logging.info("   - 'q': 退出测试")
        logging.info("   - 'p': 暂停/继续")
        logging.info("   - 'r': 重启当前游戏")
        logging.info("   - 's': 截图保存")
        
        episode = 0
        paused = False
        
        try:
            while episode < max_episodes:
                logging.info(f"\n🚀 第 {episode + 1} 局游戏开始")
                
                # 初始化游戏状态
                do_nothing = np.zeros(self.actions)
                do_nothing[0] = 1
                x_t, r_0, terminal = self.game_state.frame_step(do_nothing)
                x_t = self.preprocess_state(x_t)
                s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
                
                step = 0
                episode_score = 0
                start_time = time.time()
                action_history = []  # 记录动作历史
                
                while not terminal:
                    if not paused:
                        # 与deep_Q_oneStep.py保持一致：每FRAME_PER_ACTION帧做一次决策
                        if step % FRAME_PER_ACTION == 0:
                            # 准备网络输入
                            state_tensor = torch.FloatTensor(s_t).unsqueeze(0).to(self.device)
                            state_tensor = state_tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
                            
                            # 获取AI决策
                            action_index, q_values = self.get_action(state_tensor)
                            action_history.append(action_index)
                            
                            # 显示决策信息
                            if show_q_values and len(action_history) % 5 == 0:  # 每5次决策显示一次
                                action_name = "跳跃" if action_index == 1 else "不跳"
                                decision_count = len(action_history)
                                logging.info(f"决策 {decision_count:3d} (步数{step:3d}) | 动作: {action_name} | Q值: [{q_values[0]:6.2f}, {q_values[1]:6.2f}] | 得分: {episode_score:.3f}")
                        
                        # 构建动作向量
                        action = np.zeros([self.actions])
                        action[action_index] = 1
                        
                        # 执行动作
                        x_t1_colored, reward, terminal = self.game_state.frame_step(action)
                        x_t1 = self.preprocess_state(x_t1_colored)
                        x_t1 = np.reshape(x_t1, (80, 80, 1))
                        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
                        
                        # 更新状态
                        s_t = s_t1
                        episode_score += reward
                        step += 1
                    
                    # 控制游戏速度
                    time.sleep(delay)
                
                # 游戏结束统计
                end_time = time.time()
                duration = end_time - start_time
                
                # 决策统计（与训练保持一致）
                decision_count = len(action_history)
                jump_count = sum(action_history)
                stay_count = decision_count - jump_count
                jump_ratio = jump_count / decision_count if decision_count else 0
                
                # 计算实际决策频率
                decisions_per_sec = decision_count / duration if duration > 0 else 0
                expected_decisions = self.target_fps // FRAME_PER_ACTION  # 基于目标FPS的期望决策频率
                
                logging.info(f"🏁 第 {episode + 1} 局结束")
                logging.info(f"📊 游戏统计:")
                logging.info(f"   总步数: {step}")
                logging.info(f"   决策次数: {decision_count} (每{FRAME_PER_ACTION}帧一次)")
                logging.info(f"   最终得分: {episode_score:.3f}")
                logging.info(f"   游戏时长: {duration:.2f}秒")
                logging.info(f"   游戏FPS: {step/duration:.1f} | 决策频率: {decisions_per_sec:.1f}/秒 (期望{expected_decisions})")
                logging.info(f"   动作分布: 跳跃 {jump_count}次 ({jump_ratio*100:.1f}%) | 不跳 {stay_count}次 ({(1-jump_ratio)*100:.1f}%)")
                
                # 基于得分和行为的性能评估
                if episode_score > 50:
                    logging.info("🏆 专家级表现！AI已完全掌握游戏策略")
                elif episode_score > 20:
                    logging.info("🎉 优秀表现！AI展现了良好的游戏技巧")
                elif episode_score > 10:
                    logging.info("👍 不错的表现！AI基本掌握了游戏")
                elif episode_score > 5:
                    logging.info("📈 有进步空间，AI还在学习中")
                elif episode_score > 1:
                    logging.info("🤔 基础水平，需要更多训练")
                else:
                    logging.info("⚠️  表现较差，可能需要检查模型或重新训练")
                
                # 动作模式分析
                if jump_ratio > 0.6:
                    logging.info("📝 行为分析: AI过于激进，跳跃频率偏高")
                elif jump_ratio < 0.1:
                    logging.info("📝 行为分析: AI过于保守，跳跃频率偏低")
                else:
                    logging.info("📝 行为分析: AI动作分布合理，策略平衡")
                
                episode += 1
                
                if episode < max_episodes:
                    logging.info(f"⏱️  准备下一局... (3秒后开始)")
                    time.sleep(3)
        
        except KeyboardInterrupt:
            logging.info("\n⏹️  用户中断测试")
        except Exception as e:
            logging.error(f"❌ 测试过程中出错: {e}")
        
        logging.info(f"\n🎯 AI游戏测试完成！共完成 {episode} 局游戏")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Flappy Bird AI 游戏测试器")
    parser.add_argument('--model', type=str, help='指定要加载的模型文件路径')
    parser.add_argument('--fps', type=int, default=500, help='游戏目标FPS (默认500)')
    parser.add_argument('--episodes', type=int, default=5, help='测试局数 (默认5)')
    parser.add_argument('--delay', type=float, default=0.01, help='帧间延迟秒数 (默认0.01)')
    parser.add_argument('--no-q-values', action='store_true', help='不显示Q值信息')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("🎮 Flappy Bird AI 游戏测试器")
    print("=" * 50)
    
    # 创建测试器
    tester = AIGameTester(target_fps=args.fps)
    
    # 确定要加载的模型
    if args.model:
        # 使用指定的模型
        model_path = tester.load_specific_model(args.model)
        if not model_path:
            print("❌ 指定的模型文件无效")
            return
    else:
        # 查找最新模型
        model_path = tester.find_latest_model()
        if not model_path:
            print("❌ 未找到模型文件，请先运行训练脚本")
            return
    
    # 加载模型
    if not tester.load_model(model_path):
        print("❌ 模型加载失败")
        return
    
    # 显示测试配置
    print("\n⚙️  测试配置:")
    print(f"   模型文件: {model_path}")
    print(f"   最大局数: {args.episodes}")
    print(f"   目标FPS: {args.fps} {'(与训练一致)' if args.fps == 500 else '(自定义)'}")
    print(f"   显示Q值: {'否' if args.no_q_values else '是'}")
    print(f"   帧间延迟: {args.delay}秒")
    
    # 用户确认
    input("\n按 Enter 开始测试...")
    
    # 运行测试
    tester.run_game(
        max_episodes=args.episodes,
        show_q_values=not args.no_q_values,
        delay=args.delay
    )
    
    print("\n✅ 测试完成！查看日志文件获取详细信息")

if __name__ == "__main__":
    main()