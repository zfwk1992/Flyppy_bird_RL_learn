"""Flappy Bird 游戏环境。

``resources`` 在导入时就会初始化 pygame 并设置 SDL_VIDEODRIVER=dummy，
所以这里不做任何 eager 导入 —— 只 import flappy_env 的调用方不该被迫
承担这个副作用之外的东西。
"""
