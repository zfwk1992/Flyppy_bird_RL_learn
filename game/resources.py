"""pygame 底座：无头初始化、精灵资源、像素级碰撞检测。

只放"游戏规则之外"的东西。物理、奖励、得分判定、观测全部在 flappy_env.py 里 ——
本文件此前还带着一个 GameState 类（旧管线的环境），它与 FlappyEnv 是两份
互相偏离的物理实现，且带着四个已确诊的缺陷（崩溃时先 reset 再绘制、奖励用
赋值覆盖、势能差分算的是上一步、4 像素窗口判定得分）。那份实现已删除，
只留下这里的共享底座。
"""

import os

import pygame

from . import flappy_bird_utils

# 必须在 pygame.init() 之前设置：无头模式下 SDL 不需要真实显示设备。
# 注意这是**导入时**的无条件副作用，外部无法覆盖 —— play.py 用 cv2 显示
# 而不是 pygame 显示，根源就在这里。
os.environ['SDL_VIDEODRIVER'] = 'dummy'

SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """撞地或撞管道则返回 True。

    走 hitmask + 坐标，不依赖渲染结果 —— 所以帧跳过窗口内可以安全地
    render=False 只跑物理。
    """
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # 撞地
    if player['y'] + player['h'] >= BASEY - 1:
        return True

    playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

        pHitMask = HITMASKS['player'][pi]
        uHitmask = HITMASKS['pipe'][0]
        lHitmask = HITMASKS['pipe'][1]

        if (pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                or pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)):
            return True

    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """两个对象是否真的重叠（而不只是包围盒相交）。"""
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
