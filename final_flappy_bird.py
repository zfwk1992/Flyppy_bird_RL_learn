#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import pygame
import numpy as np
import time

def final_flappy_bird():
    """最终版Flappy Bird游戏，确保R键重新开始功能正常"""
    
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((288, 512))
    pygame.display.set_caption('Flappy Bird - Final Version')
    
    print("=== Flappy Bird 游戏 ===")
    print("控制说明：")
    print("- 按 SPACE 键或 鼠标左键 让小鸟跳跃")
    print("- 按 R 键重新开始游戏")
    print("- 按 ESC 键退出游戏")
    print("=" * 30)
    
    def start_new_game():
        """开始新游戏"""
        return game.GameState(), 0, 0
    
    # 初始化游戏
    game_state, score, frame_count = start_new_game()
    running = True
    
    try:
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_r:
                        # 重新开始游戏
                        game_state, score, frame_count = start_new_game()
                        print("🎮 游戏重新开始！")
                        continue
            
            # 确定动作
            keys = pygame.key.get_pressed()
            mouse_clicked = pygame.mouse.get_pressed()[0]
            
            if keys[pygame.K_SPACE] or mouse_clicked:
                action = [0, 1]  # 跳跃
            else:
                action = [1, 0]  # 不动作
            
            # 执行游戏步骤
            image_data, reward, terminal = game_state.frame_step(action)
            frame_count += 1
            
            # 更新分数
            if reward == 1:
                score += 1
                print(f"🎉 得分！当前分数: {score}")
            
            # 游戏结束处理
            if terminal:
                print(f"💥 游戏结束！最终分数: {score}")
                print("按 R 键重新开始，或按 ESC 键退出")
                
                # 等待用户选择 - 使用更可靠的方法
                waiting = True
                while waiting and running:
                    # 处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                waiting = False
                                break
                            elif event.key == pygame.K_r:
                                game_state, score, frame_count = start_new_game()
                                print("🎮 游戏重新开始！")
                                waiting = False
                                break
                    
                    # 短暂延迟，避免CPU占用过高
                    time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n游戏被中断。")
    finally:
        print(f"\n游戏结束！")
        print(f"最终分数: {score}")
        print(f"总共运行了 {frame_count} 帧")
        pygame.quit()

if __name__ == "__main__":
    final_flappy_bird() 