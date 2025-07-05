#!/usr/bin/env python
"""
距离奖励系统实现示例
演示如何在Flappy Bird中实现基于距离的奖励
"""

import numpy as np
import math

class DistanceRewardSystem:
    """距离奖励系统"""
    
    def __init__(self, max_reward=0.5, sigma=50, distance_threshold=100):
        """
        初始化距离奖励系统
        
        Args:
            max_reward: 最大距离奖励 (0.3-0.5)
            sigma: 高斯函数标准差 (30-70)
            distance_threshold: 距离阈值 (80-120)
        """
        self.max_reward = max_reward
        self.sigma = sigma
        self.distance_threshold = distance_threshold
        
    def calculate_distance_reward(self, bird_x, bird_y, pipe_x, pipe_y, pipe_gap_center_y):
        """
        计算基于距离的奖励
        
        Args:
            bird_x: 小鸟x坐标
            bird_y: 小鸟y坐标
            pipe_x: 管道x坐标
            pipe_y: 管道y坐标
            pipe_gap_center_y: 管道空隙中心y坐标
            
        Returns:
            float: 距离奖励值
        """
        # 计算水平距离
        horizontal_distance = abs(bird_x - pipe_x)
        
        # 计算垂直距离（小鸟中心到空隙中心）
        vertical_distance = abs(bird_y - pipe_gap_center_y)
        
        # 只在接近管道时给予距离奖励
        if horizontal_distance < self.distance_threshold:
            # 计算综合距离
            distance = math.sqrt(horizontal_distance**2 + vertical_distance**2)
            
            # 使用高斯函数计算奖励
            distance_reward = self.max_reward * np.exp(-distance**2 / (2 * self.sigma**2))
            
            return distance_reward
        
        return 0.0
    
    def calculate_enhanced_reward(self, bird_x, bird_y, pipe_x, pipe_y, pipe_gap_center_y, 
                                pipe_gap_size, base_reward=0.1):
        """
        计算增强奖励（包含距离奖励）
        
        Args:
            bird_x: 小鸟x坐标
            bird_y: 小鸟y坐标
            pipe_x: 管道x坐标
            pipe_y: 管道y坐标
            pipe_gap_center_y: 管道空隙中心y坐标
            pipe_gap_size: 管道空隙大小
            base_reward: 基础存活奖励
            
        Returns:
            dict: 包含各种奖励的字典
        """
        # 基础存活奖励
        total_reward = base_reward
        
        # 距离奖励
        distance_reward = self.calculate_distance_reward(
            bird_x, bird_y, pipe_x, pipe_y, pipe_gap_center_y
        )
        total_reward += distance_reward
        
        # 空隙中心奖励（额外奖励）
        gap_center_reward = 0.0
        if abs(bird_y - pipe_gap_center_y) < pipe_gap_size * 0.3:  # 在空隙中心30%范围内
            gap_center_reward = 0.2
            total_reward += gap_center_reward
        
        # 接近奖励（额外奖励）
        approach_reward = 0.0
        if abs(bird_x - pipe_x) < 50:  # 距离管道50像素内
            approach_reward = 0.1
            total_reward += approach_reward
        
        return {
            'total_reward': total_reward,
            'base_reward': base_reward,
            'distance_reward': distance_reward,
            'gap_center_reward': gap_center_reward,
            'approach_reward': approach_reward
        }

def demonstrate_distance_reward():
    """演示距离奖励计算"""
    print("🎯 距离奖励系统演示")
    print("=" * 60)
    
    # 创建奖励系统
    reward_system = DistanceRewardSystem(
        max_reward=0.4,      # 最大距离奖励
        sigma=40,            # 高斯函数标准差
        distance_threshold=90 # 距离阈值
    )
    
    # 游戏参数
    pipe_gap_size = 100      # 管道空隙大小
    pipe_gap_center_y = 300  # 管道空隙中心y坐标
    
    # 测试不同位置
    test_positions = [
        # (bird_x, bird_y, pipe_x, description)
        (200, 300, 250, "小鸟在空隙中心"),
        (200, 350, 250, "小鸟在空隙下方"),
        (200, 250, 250, "小鸟在空隙上方"),
        (150, 300, 250, "小鸟在空隙中心但距离较远"),
        (300, 300, 250, "小鸟在空隙中心且非常接近"),
        (200, 400, 250, "小鸟远离空隙"),
    ]
    
    print("📊 不同位置的奖励计算:")
    print()
    
    for bird_x, bird_y, pipe_x, description in test_positions:
        # 计算奖励
        rewards = reward_system.calculate_enhanced_reward(
            bird_x, bird_y, pipe_x, 0, pipe_gap_center_y, pipe_gap_size
        )
        
        print(f"📍 {description}:")
        print(f"   位置: 小鸟({bird_x}, {bird_y}), 管道x={pipe_x}")
        print(f"   总奖励: {rewards['total_reward']:.3f}")
        print(f"   - 基础奖励: {rewards['base_reward']:.3f}")
        print(f"   - 距离奖励: {rewards['distance_reward']:.3f}")
        print(f"   - 空隙中心奖励: {rewards['gap_center_reward']:.3f}")
        print(f"   - 接近奖励: {rewards['approach_reward']:.3f}")
        print()

def show_reward_visualization():
    """显示奖励可视化"""
    print("📈 距离奖励可视化")
    print("=" * 60)
    
    reward_system = DistanceRewardSystem(max_reward=0.4, sigma=40, distance_threshold=90)
    
    # 创建网格
    bird_x = 200
    pipe_x = 250
    pipe_gap_center_y = 300
    
    print("垂直距离 vs 奖励值 (水平距离=50像素):")
    print("距离(像素) | 奖励值")
    print("-" * 20)
    
    for vertical_distance in range(0, 200, 20):
        reward = reward_system.calculate_distance_reward(
            bird_x, pipe_gap_center_y + vertical_distance, 
            pipe_x, 0, pipe_gap_center_y
        )
        print(f"{vertical_distance:>8} | {reward:.4f}")
    
    print()
    print("水平距离 vs 奖励值 (垂直距离=0像素):")
    print("距离(像素) | 奖励值")
    print("-" * 20)
    
    for horizontal_distance in range(0, 150, 15):
        reward = reward_system.calculate_distance_reward(
            bird_x + horizontal_distance, pipe_gap_center_y, 
            pipe_x, 0, pipe_gap_center_y
        )
        print(f"{horizontal_distance:>8} | {reward:.4f}")

def compare_reward_systems():
    """比较不同奖励系统"""
    print("🔄 奖励系统对比")
    print("=" * 60)
    
    # 原始奖励系统
    def original_reward(bird_x, bird_y, pipe_x, pipe_gap_center_y, passed_pipe=False, crashed=False):
        if crashed:
            return -1.0
        elif passed_pipe:
            return 1.0
        else:
            return 0.1
    
    # 距离奖励系统
    distance_system = DistanceRewardSystem(max_reward=0.4, sigma=40, distance_threshold=90)
    
    # 测试场景
    scenarios = [
        ("小鸟在空隙中心", 200, 300, 250, False, False),
        ("小鸟接近空隙", 200, 320, 250, False, False),
        ("小鸟远离空隙", 200, 400, 250, False, False),
        ("通过管道", 200, 300, 250, True, False),
        ("碰撞死亡", 200, 300, 250, False, True),
    ]
    
    print("场景 | 原始奖励 | 距离奖励 | 改进效果")
    print("-" * 50)
    
    for scenario, bird_x, bird_y, pipe_x, passed_pipe, crashed in scenarios:
        # 原始奖励
        orig_reward = original_reward(bird_x, bird_y, pipe_x, 300, passed_pipe, crashed)
        
        # 距离奖励
        if not passed_pipe and not crashed:
            dist_rewards = distance_system.calculate_enhanced_reward(
                bird_x, bird_y, pipe_x, 0, 300, 100
            )
            dist_reward = dist_rewards['total_reward']
        else:
            dist_reward = orig_reward
        
        # 改进效果
        improvement = dist_reward - orig_reward
        
        print(f"{scenario} | {orig_reward:>8.3f} | {dist_reward:>8.3f} | {improvement:>+8.3f}")

def provide_implementation_guide():
    """提供实现指南"""
    print("🛠️ 实现指南")
    print("=" * 60)
    
    print("📝 在游戏中的实现步骤:")
    print()
    
    print("1. 修改 game/wrapped_flappy_bird.py:")
    print("   - 添加 numpy 导入: import numpy as np")
    print("   - 添加 DistanceRewardSystem 类")
    print("   - 在 GameState 类中初始化奖励系统")
    print()
    
    print("2. 在 frame_step 函数中添加距离奖励:")
    print("   ```python")
    print("   # 计算距离奖励")
    print("   if self.upperPipes:")
    print("       nearest_pipe = self.upperPipes[0]")
    print("       pipe_x = nearest_pipe['x'] + PIPE_WIDTH / 2")
    print("       gap_center_y = nearest_pipe['y'] + PIPE_HEIGHT + PIPEGAPSIZE / 2")
    print("       ")
    print("       bird_x = self.playerx + PLAYER_WIDTH / 2")
    print("       bird_y = self.playery + PLAYER_HEIGHT / 2")
    print("       ")
    print("       distance_reward = self.reward_system.calculate_distance_reward(")
    print("           bird_x, bird_y, pipe_x, nearest_pipe['y'], gap_center_y")
    print("       )")
    print("       reward += distance_reward")
    print("   ```")
    print()
    
    print("3. 参数调优建议:")
    print("   - 保守设置: max_reward=0.3, sigma=50, threshold=100")
    print("   - 平衡设置: max_reward=0.4, sigma=40, threshold=90")
    print("   - 激进设置: max_reward=0.5, sigma=30, threshold=80")
    print()
    
    print("4. 监控和调试:")
    print("   - 添加奖励日志记录")
    print("   - 观察小鸟行为变化")
    print("   - 调整参数以获得最佳效果")

if __name__ == "__main__":
    demonstrate_distance_reward()
    print("\n" + "="*60 + "\n")
    show_reward_visualization()
    print("\n" + "="*60 + "\n")
    compare_reward_systems()
    print("\n" + "="*60 + "\n")
    provide_implementation_guide() 