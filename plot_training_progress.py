#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 黑体、微软雅黑、DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def parse_log_file(log_file_path):
    """解析训练日志文件，提取分数数据"""
    episodes = []
    scores = []
    max_scores = []
    avg_scores = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配游戏局数日志格式：🎮 第X局 (X.X%) | 分数: XX.XX | 最高: XX.XX | 近100局: XX.XX
            match = re.search(r'🎮 第(\d+)局.*分数: ([-\d.]+).*最高: ([-\d.]+).*近100局: ([-\d.]+)', line)
            if match:
                episode = int(match.group(1))
                score = float(match.group(2))
                max_score = float(match.group(3))
                avg_score = float(match.group(4))
                
                episodes.append(episode)
                scores.append(score)
                max_scores.append(max_score)
                avg_scores.append(avg_score)
    
    return episodes, scores, max_scores, avg_scores

def plot_training_progress(log_file_path):
    """绘制训练进度图像"""
    episodes, scores, max_scores, avg_scores = parse_log_file(log_file_path)
    
    if not episodes:
        print("❌ 未找到有效的训练数据")
        return
    
    print(f"📊 解析到 {len(episodes)} 局游戏数据")
    print(f"📈 分数范围: {min(scores):.2f} ~ {max(scores):.2f}")
    print(f"🎯 最高分: {max(max_scores):.2f}")
    
    # 创建图像
    plt.figure(figsize=(15, 10))
    
    # 子图1: 每局分数
    plt.subplot(2, 2, 1)
    plt.plot(episodes, scores, 'lightblue', alpha=0.6, linewidth=0.5, label='Episode Score')
    # 添加移动平均线
    if len(scores) > 50:
        window_size = min(50, len(scores)//10)
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        plt.plot(moving_episodes, moving_avg, 'red', linewidth=2, label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.title('Score per Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2: 最高分记录
    plt.subplot(2, 2, 2)
    plt.plot(episodes, max_scores, 'green', linewidth=2, label='Best Score')
    plt.xlabel('Episode Number')
    plt.ylabel('Best Score')
    plt.title('Best Score Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图3: 近100局平均分
    plt.subplot(2, 2, 3)
    plt.plot(episodes, avg_scores, 'orange', linewidth=2, label='Recent 100 Avg')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Score')
    plt.title('Recent 100 Episodes Average')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图4: 分数分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_file = 'training_progress_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 图像已保存到: {output_file}")
    
    # 显示统计信息
    print(f"\n📊 训练统计信息:")
    print(f"   总局数: {len(episodes)}")
    print(f"   平均分: {np.mean(scores):.2f}")
    print(f"   标准差: {np.std(scores):.2f}")
    print(f"   最高分: {max(max_scores):.2f}")
    print(f"   最低分: {min(scores):.2f}")
    print(f"   最终100局均分: {avg_scores[-1]:.2f}")
    
    # 分析训练进展
    if len(scores) > 100:
        first_100_avg = np.mean(scores[:100])
        last_100_avg = np.mean(scores[-100:])
        improvement = last_100_avg - first_100_avg
        print(f"   前100局均分: {first_100_avg:.2f}")
        print(f"   后100局均分: {last_100_avg:.2f}")
        print(f"   改善程度: {improvement:+.2f} ({'提升' if improvement > 0 else '下降'})")
    
    plt.show()

if __name__ == "__main__":
    log_file = "logs/improved_dueling_dqn_no_obs_20250802_104047.log"
    plot_training_progress(log_file)