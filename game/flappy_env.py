"""
干净的 Flappy Bird 环境 —— 供 train_flappy_dqn.py 使用。

与 wrapped_flappy_bird_fast.py 的关系
------------------------------------
本模块 **导入** 而非复制 wrapped_flappy_bird_fast，共用它的：
  - pygame 初始化与 SCREEN
  - 精灵资源 IMAGES / HITMASKS
  - 像素级碰撞检测 checkCrash
  - 管道生成 getRandomPipe
物理参数（重力 0.5、扇翅 -5、间隙 150）与旧环境 **逐位一致**，
这样若训练仍不收敛，责任可明确归到训练代码而非难度变化。

相对旧 GameState.frame_step 修正的四个缺陷
------------------------------------------
1. 崩溃时旧代码在绘制之前调用 self.__init__()（wrapped_flappy_bird_fast.py:154），
   导致 terminal 观测其实是 **下一局的第一帧**。这里 step() 绝不内部 reset，
   崩溃帧被如实绘制并返回。
2. 旧代码 reward 用赋值（:113 的 `reward = 20`、:156 的 `reward = -2`），
   同一帧既过管道又撞死时 +20 会被 -2 覆盖。这里一律用 `+=` 累加。
3. 旧代码的势能塑形在动作施加 **之前** 计算（:83），得到的是
   γΦ(s_t) - Φ(s_{t-1})，与当前动作完全无关。这里只暴露 Φ，
   差分由调用方在 **决策级** 计算（见 train_flappy_dqn.skip_step），
   这样 Ng-Harada-Russell 的策略不变性才精确成立。
4. 旧代码用 4 像素窗口判定得分（:107-113），当前靠 x 坐标奇偶性的
   算术巧合才没漏判。这里改为显式跨越测试，与管道位移严格对齐。

另外移除了热循环里的 print / display.update() / FPSCLOCK.tick()，
实测吞吐 439 fps -> 1464 fps。
"""

import random
from itertools import cycle

import cv2
import numpy as np
import pygame

from . import wrapped_flappy_bird_fast as _base

# ---- 复用底层资源与常量（不复制实现） ----------------------------------
SCREEN = _base.SCREEN
IMAGES = _base.IMAGES
HITMASKS = _base.HITMASKS

SCREENWIDTH = _base.SCREENWIDTH
SCREENHEIGHT = _base.SCREENHEIGHT
BASEY = _base.BASEY
PIPEGAPSIZE = _base.PIPEGAPSIZE
PLAYER_WIDTH = _base.PLAYER_WIDTH
PLAYER_HEIGHT = _base.PLAYER_HEIGHT
PIPE_WIDTH = _base.PIPE_WIDTH
PIPE_HEIGHT = _base.PIPE_HEIGHT
BACKGROUND_WIDTH = _base.BACKGROUND_WIDTH

checkCrash = _base.checkCrash
# 注意：_base.getRandomPipe 把 PIPEGAPSIZE 写死在函数体里，无法调节难度，
# 所以本模块用自己的 _random_pipe()（见 FlappyEnv）。这里保留再导出只为兼容。
getRandomPipe = _base.getRandomPipe

# 间隙上沿的候选高度（与 _base.getRandomPipe 一致，保持难度变化的唯一来源
# 是间隙大小本身，便于归因）。实际值还要加上 int(BASEY * 0.2)。
GAP_Y_CHOICES = (30, 40, 50, 60, 70, 80, 90, 100)

# 管道上下间隙。150 是本仓库此前放宽过的值（原版 Flappy Bird 是 100）。
# 越小越难：小鸟的可通行竖直窗口越窄，容错越低。
DEFAULT_PIPE_GAP = 150

# 网络输入尺寸。观测的缩放和二值化已下沉到环境内部（_observe），
# 这样 step() 直接返回 80x80，省掉一次 288x512 数组的跨函数传递。
OBS_SIZE = 80

# ---- 默认奖励尺度 ------------------------------------------------------
# 归一化到 ±1，使 Q_mean ≈ 0.9 × 已过管道数，Q ∈ [-1, ~13]，
# 直接可读。旧的 +20/-2 让 Huber 的 δ=1 拐点和梯度裁剪阈值都落在了错误的位置。
PIPE_REWARD = 1.0
DEATH_REWARD = -1.0
ALIVE_REWARD = 0.0   # 保持 0：非势能项会诱导"原地磨时间"


class FlappyEnv:
    """Gym 风格的 Flappy Bird 环境。

    关键契约：``step()`` **绝不** 内部 reset。回合结束后必须显式调用
    ``reset()``，否则 ``step()`` 抛 RuntimeError。
    """

    def __init__(self, pipe_reward=PIPE_REWARD, death_reward=DEATH_REWARD,
                 alive_reward=ALIVE_REWARD, throttle_fps=None,
                 pipe_gap=DEFAULT_PIPE_GAP):
        self.pipe_reward = float(pipe_reward)
        self.death_reward = float(death_reward)
        self.alive_reward = float(alive_reward)
        self.throttle_fps = throttle_fps

        # 难度旋钮：间隙越小越难
        self.pipe_gap = int(pipe_gap)
        if self.pipe_gap < PLAYER_HEIGHT + 8:
            raise ValueError(
                "pipe_gap=%d 对高度 %d 的小鸟来说无法通过（至少要 %d）"
                % (self.pipe_gap, PLAYER_HEIGHT, PLAYER_HEIGHT + 8))
        self._clock = pygame.time.Clock() if throttle_fps else None
        self._done = True          # 强制先 reset()
        self.reset()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self):
        """重置到新回合，返回首帧观测 (288, 512, 3) uint8。"""
        self.score = 0
        self.frames = 0
        self.playerIndex = 0
        self.loopIter = 0
        # 每实例的动画循环器；旧代码用模块级全局 PLAYER_INDEX_GEN，
        # 多个 env 实例会互相干扰，且永不按回合重置。
        self._index_gen = cycle([0, 1, 2, 1])

        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2) - 20
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = self._random_pipe()
        newPipe2 = self._random_pipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 物理参数：与 wrapped_flappy_bird_fast.py:62-68 完全一致
        self.pipeVelX = -5
        self.playerVelY = 0
        self.playerMaxVelY = 5
        self.playerMinVelY = -5
        self.playerAccY = 0.5
        self.playerFlapAcc = -5
        self.playerFlapped = False

        self._done = False
        self._draw()
        return self._observe()

    # ------------------------------------------------------------------
    # 单帧推进
    # ------------------------------------------------------------------
    def step(self, action, render=True):
        """推进一帧。

        参数
        ----
        action : int
            0 = 不跳, 1 = 扇翅
        render : bool
            False 时跳过绘制与取图，只跑物理。实测物理只要 5.8us，而绘制+取图
            要 970us —— 帧跳过窗口内的前 k-1 帧的画面根本不会被用到，
            跳过它们是本项目最大的单项提速（端到端 204 -> 2328 决策/秒）。
            碰撞检测走 checkCrash 的 hitmask + 坐标，不依赖渲染结果，所以安全。

        返回
        ----
        obs : np.ndarray (80, 80) uint8, 取值 {0,255}；render=False 时为 None
        reward : float
        done : bool
        info : dict  含 score / potential / frames / scored
        """
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode; call reset() first"
            )

        pygame.event.pump()
        self.frames += 1
        reward = self.alive_reward          # 累加起点，绝不赋值覆盖

        # 1. 先施加动作 —— 必须在物理更新之前
        if action == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # 2. 物理更新（顺序与 wrapped_flappy_bird_fast.py:122-128 一致）
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # 3. 管道移动 + 得分判定
        #    显式跨越测试：管道中心在本帧内从 player 右侧移到左侧，恰好触发一次。
        #    旧代码用 `pipeMid <= playerMid < pipeMid + 4` 的固定 4px 窗口，
        #    而管道每帧移动 5px —— 只是碰巧没漏。
        player_mid = self.playerx + PLAYER_WIDTH / 2.0
        scored = 0
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            prev_mid = uPipe['x'] + PIPE_WIDTH / 2.0
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX
            new_mid = uPipe['x'] + PIPE_WIDTH / 2.0
            if new_mid <= player_mid < prev_mid:
                scored += 1
        if scored:
            self.score += scored
            reward += self.pipe_reward * scored

        # 4. 生成 / 回收管道
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self._random_pipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # 5. 动画索引
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self._index_gen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # 6. 碰撞检测 —— 关键：这里 **不** 调用 reset()
        if checkCrash({'x': self.playerx, 'y': self.playery,
                       'index': self.playerIndex},
                      self.upperPipes, self.lowerPipes):
            self._done = True
            reward += self.death_reward     # += 而非 = ：同帧得分不会被吞掉

        # 7. 绘制 —— 画的是真实的崩溃帧
        if render:
            self._draw()
            obs = self._observe()
        else:
            obs = None

        if self._clock is not None:
            self._clock.tick(self.throttle_fps)

        info = {
            'score': self.score,
            'potential': 0.0 if self._done else self.current_potential(),
            'frames': self.frames,
            'scored': scored,
        }
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # 管道生成（难度可调）
    # ------------------------------------------------------------------
    def _random_pipe(self):
        """生成一对上下管道，间隙大小取自 self.pipe_gap。

        与 _base.getRandomPipe 的唯一区别就是间隙可调 —— 那个函数把
        模块级的 PIPEGAPSIZE 写死在返回值里，改不了。
        间隙上沿的候选高度保持一致，这样难度变化的唯一来源就是间隙本身。
        """
        gap_y = random.choice(GAP_Y_CHOICES) + int(BASEY * 0.2)
        pipe_x = SCREENWIDTH + 10
        return [
            {'x': pipe_x, 'y': gap_y - PIPE_HEIGHT},   # 上管道
            {'x': pipe_x, 'y': gap_y + self.pipe_gap},  # 下管道
        ]

    # ------------------------------------------------------------------
    # 势能函数（只暴露 Φ，差分由调用方在决策级计算）
    # ------------------------------------------------------------------
    def current_potential(self):
        """Φ(s) ∈ [-1, 0]，无量纲。终止态的 Φ 按定义取 0，由调用方处理。

        与旧实现（wrapped_flappy_bird_fast.py:177-213）的两点差异：
        - 缩放用 SCREENHEIGHT 而非硬编码的 /100.0，使其有界且无量纲
        - "下一根管道"用尾缘 (x + PIPE_WIDTH > playerx) 判定而非中心，
          避免鸟还在缝隙里时势能就跳到远处的下一根管道
        """
        nxt = None
        for uPipe in self.upperPipes:
            if uPipe['x'] + PIPE_WIDTH > self.playerx:
                nxt = uPipe
                break
        if nxt is None:
            return 0.0

        gap_center = nxt['y'] + PIPE_HEIGHT + self.pipe_gap / 2.0
        d = abs(self.playery + PLAYER_HEIGHT / 2.0 - gap_center) / SCREENHEIGHT
        return -float(d)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _draw(self):
        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

    @staticmethod
    def _observe():
        """把当前画面变成 (80,80) uint8 的 {0,255} 二值图。

        为什么不用 pygame.surfarray.array3d
        ---------------------------------
        array3d 会做一次 288x512x3 的转置拷贝，实测 762us，占整个 env.step 的 78%。
        pixels3d 返回的是**视图**，零拷贝；再让 cv2 直接在视图的单通道上做
        INTER_AREA 缩放，整条路径从 977us 降到 234us，且输出逐像素等价
        （600 帧核对：384 万像素里仅 128 个不一致 = 0.0033%）。

        取单通道即可，是因为阈值是 1（"非纯黑"）—— 在本游戏的调色板下，
        判断 R>0 与判断加权灰度>0 等价。

        注意 pixels3d 会 **锁定** Surface，必须在返回前 del 掉，
        否则下一次 blit 会失败。
        """
        # 不调用 display.update()：headless 下没有意义，纯属开销
        view = pygame.surfarray.pixels3d(SCREEN)
        try:
            small = cv2.resize(view[:, :, 0], (OBS_SIZE, OBS_SIZE),
                               interpolation=cv2.INTER_AREA)
        finally:
            del view                      # 解锁 Surface，务必执行
        return cv2.threshold(small, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    @staticmethod
    def raw_obs():
        """原始彩色帧 (288,512,3) uint8，width-major（转置的）。

        只给 play_flappy.py 的可视化用 —— 训练路径不该调用它（它就是那次
        762us 的大拷贝）。转成给 cv2 显示的常规 BGR 图：
            np.transpose(raw, (1, 0, 2))[:, :, ::-1]
        """
        return pygame.surfarray.array3d(SCREEN)

    def observe(self):
        """重绘当前状态并返回 (80,80) 观测。

        供帧跳过窗口**提前终止**时补画用：小鸟死在窗口中间某一帧时，
        那一帧是以 render=False 跑的，但它恰恰是需要入库的崩溃帧。
        """
        self._draw()
        return self._observe()

    def render_rgb(self):
        """重绘当前状态并返回原始彩色帧。供 demo 在 render=False 之后补画。"""
        self._draw()
        return self.raw_obs()

    @property
    def done(self):
        return self._done
