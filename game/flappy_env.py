"""Flappy Bird 环境 —— Gym 风格，供 train.py / eval.py / play.py 共用。

与 game/resources.py 的分工
---------------------------
本模块 **导入** 底座而非复制它，共用：
  - pygame 无头初始化与 SCREEN
  - 精灵资源 IMAGES / HITMASKS
  - 像素级碰撞检测 checkCrash
游戏规则（物理、奖励、得分、观测）全部在这里，只此一份。

相对旧 GameState.frame_step 修正的四个缺陷
------------------------------------------
（旧实现已删除，可在 git 历史中查阅 game/wrapped_flappy_bird_fast.py）

1. 崩溃时旧代码在绘制之前调用 self.__init__()，导致 terminal 观测其实是
   **下一局的第一帧**。这里 step() 绝不内部 reset，崩溃帧被如实绘制并返回。
2. 旧代码 reward 用赋值（`reward = 20`、`reward = -2`），同一帧既过管道
   又撞死时 +20 会被 -2 覆盖。这里一律用 `+=` 累加。
3. 旧代码的势能塑形在动作施加 **之前** 计算，得到的是 γΦ(s_t) - Φ(s_{t-1})，
   与当前动作完全无关。这里只暴露 Φ，差分由调用方在 **决策级** 计算
   （见 flappy.rollout.skip_step），这样 Ng-Harada-Russell 的策略不变性
   才精确成立。
4. 旧代码用 4 像素窗口判定得分，当前靠 x 坐标奇偶性的算术巧合才没漏判。
   这里改为显式跨越测试，与管道位移严格对齐。

另外移除了热循环里的 print / display.update() / FPSCLOCK.tick()，
实测吞吐 439 fps -> 1464 fps。
"""

import random
from itertools import cycle

import cv2
import numpy as np
import pygame

from . import resources as _base

# ---- 复用底层资源与常量（不复制实现） ----------------------------------
SCREEN = _base.SCREEN
IMAGES = _base.IMAGES
HITMASKS = _base.HITMASKS

SCREENWIDTH = _base.SCREENWIDTH
SCREENHEIGHT = _base.SCREENHEIGHT
BASEY = _base.BASEY
PLAYER_WIDTH = _base.PLAYER_WIDTH
PLAYER_HEIGHT = _base.PLAYER_HEIGHT
PIPE_WIDTH = _base.PIPE_WIDTH
PIPE_HEIGHT = _base.PIPE_HEIGHT
BACKGROUND_WIDTH = _base.BACKGROUND_WIDTH

checkCrash = _base.checkCrash

# 间隙上沿的候选高度（randomize=False 时用）。实际值还要加上 int(BASEY * 0.2)。
#
# 实测这套固定值有多窄：缝隙中心只有 8 个离散取值、跨度 70px，
# 而留 38px 余量后理论可用范围是 224px —— 只覆盖了 31%。
# 缝隙大小完全固定，水平间距也几乎固定（144/145/150/151）。
# 于是网络只需要应付一个很窄的分布，学到的是一套窄策略而非"会玩这个游戏"。
# randomize=True 打开域随机化来解决这件事，见 _sample_pipe()。
GAP_Y_CHOICES = (30, 40, 50, 60, 70, 80, 90, 100)

# ---- 域随机化的默认范围 ------------------------------------------------
# 缝隙大小：下界要能过（小鸟高 24px），上界给到相当宽松
DEFAULT_GAP_RANGE = (80.0, 165.0)
# 相邻管道的水平间距。原来恒为 144。
DEFAULT_SPACING_RANGE = (115.0, 200.0)
# 缝隙上下沿距天花板 / 地面的最小余量：太贴边等于必死
DEFAULT_EDGE_MARGIN = 38.0
# 相邻缝隙中心的最大落差 / 水平间距。见 _sample_pipe() 里的可达性推导。
DEFAULT_MAX_DELTA_FRAC = 0.6

# 小鸟能维持的最小竖直振荡幅度（像素）。
#
# 动作以 frame_skip 为粒度，一次扇翅把 velY 瞬间设成 -5，之后重力 +0.5/帧。
# 于是"悬停"必然是锯齿形的，振幅有硬下限。实测（scratchpad/hover.py）：
#   每 4 决策扇一次 -> 振幅 22.5px，但净漂移 -20px/周期（在往上飘）
#   每 5 决策扇一次 -> 振幅 22.5px，净漂移  -5px/周期  <- 唯一接近悬停的节奏
#   每 6 决策扇一次 -> 振幅 42.5px，净漂移 +15px/周期（在往下掉）
# 所以 22.5px 是小鸟"稳住身形"所需的最小竖直空间，和缝隙宽度无关。
HOVER_AMPLITUDE = 22.5

# 缝隙很窄时，除了悬停振荡还得留一点纯余量给瞄准误差。
# 低于这个值的缝隙即使几何上过得去，实际也几乎必死，不该生成。
MIN_AIM_MARGIN = 8.0

# 管道上下间隙。150 是本仓库此前放宽过的值（原版 Flappy Bird 是 100）。
# 越小越难：小鸟的可通行竖直窗口越窄，容错越低。
DEFAULT_PIPE_GAP = 150

# 网络输入尺寸。观测的缩放和二值化已下沉到环境内部（_observe），
# 这样 step() 直接返回观测，省掉一次 288x512 数组的跨函数传递。
#
# **各向异性**：屏幕是 288x512，任务的精度需求几乎全在竖直方向
# （要对准缝隙），所以竖直方向给更高的分辨率。
#   OBS_W = 80  <- 屏幕宽 288，压缩 3.6 倍
#   OBS_H = 128 <- 屏幕高 512，压缩 4.0 倍（原来压 6.4 倍）
# 实测：80x80 时 60px 的缝隙只剩 3 个像素行，容错约 2 像素，
# 比 conv1 的 stride(4 像素) 还小；加到 128 后翻倍。
#
# **注意数组方向**：pygame.surfarray 是 width-major，所以观测数组的
#   axis 0 = 屏幕宽方向 (OBS_W)
#   axis 1 = 屏幕高方向 (OBS_H)
# 也就是说画面在数组里是转置的。这对卷积网络没有影响（它不在乎哪边是"上"），
# 但决定了 cv2.resize 的 dsize 必须写成 (OBS_H, OBS_W)。
OBS_W = 80
OBS_H = 128

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
                 pipe_gap=DEFAULT_PIPE_GAP,
                 randomize=False,
                 gap_range=DEFAULT_GAP_RANGE,
                 spacing_range=DEFAULT_SPACING_RANGE,
                 edge_margin=DEFAULT_EDGE_MARGIN,
                 max_delta_frac=DEFAULT_MAX_DELTA_FRAC,
                 obs_w=OBS_W, obs_h=OBS_H):
        # 观测尺寸是实例属性而不是模块常量 —— 否则改配置不会改环境实际输出，
        # 会在第一次前向时以形状不匹配的方式炸出来
        self.obs_w = int(obs_w)
        self.obs_h = int(obs_h)
        self.pipe_reward = float(pipe_reward)
        self.death_reward = float(death_reward)
        self.alive_reward = float(alive_reward)
        self.throttle_fps = throttle_fps

        # 难度旋钮：间隙越小越难。randomize=True 时它只作为回退值，
        # 实际缝隙大小逐根从 gap_range 采样。
        self.pipe_gap = int(pipe_gap)
        if self.pipe_gap < PLAYER_HEIGHT + 8:
            raise ValueError(
                "pipe_gap=%d 对高度 %d 的小鸟来说无法通过（至少要 %d）"
                % (self.pipe_gap, PLAYER_HEIGHT, PLAYER_HEIGHT + 8))

        # ---- 域随机化 ----
        self.randomize = bool(randomize)
        self.gap_range = (float(gap_range[0]), float(gap_range[1]))
        self.spacing_range = (float(spacing_range[0]), float(spacing_range[1]))
        self.edge_margin = float(edge_margin)
        self.max_delta_frac = float(max_delta_frac)
        if self.randomize:
            if self.gap_range[0] < PLAYER_HEIGHT + 8:
                raise ValueError(
                    "gap_range 下界 %.0f 对高度 %d 的小鸟无法通过（至少要 %d）"
                    % (self.gap_range[0], PLAYER_HEIGHT, PLAYER_HEIGHT + 8))
            # 最大的缝隙也必须塞得进上下余量之间，否则采样区间是空的
            if self.gap_range[1] + 2 * self.edge_margin > BASEY:
                raise ValueError(
                    "gap_range 上界 %.0f 加上下各 %.0f 的余量超过了可用高度 %.0f"
                    % (self.gap_range[1], self.edge_margin, BASEY))

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

        # 第一根缝隙要从小鸟的出生高度够得着，所以可达性链条从 playery 起算。
        # 出生时小鸟是静止的、不受缝隙约束，所以偏移余量记为 0。
        self._last_gap_center = float(self.playery + PLAYER_HEIGHT / 2.0)
        self._last_slack = 0.0
        self._plan_next()

        first = self._sample_pipe(SCREENWIDTH)          # 内部会 _plan_next()
        second = self._sample_pipe(SCREENWIDTH + self._next_spacing)

        self.upperPipes = [first[0], second[0]]
        self.lowerPipes = [first[1], second[1]]

        # 物理参数：与旧环境逐位一致（重力 0.5、扇翅 -5、最大下落 5），
        # 这样若训练不收敛，责任可明确归到训练代码而非难度变化
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

        # 2. 物理更新（顺序与旧环境一致）
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
        #    触发条件按"最后一根管道距屏幕右缘已经够 _next_spacing"来判断，
        #    而不是原来的"第一根管道快出左边界了"—— 后者把间距锁死在
        #    屏幕宽度的一半上，间距根本没法随机。
        if self.upperPipes[-1]['x'] <= SCREENWIDTH - self._next_spacing:
            newPipe = self._sample_pipe(self.upperPipes[-1]['x'] + self._next_spacing)
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
    def _sample_pipe(self, pipe_x):
        """采样一对上下管道，返回 (upper, lower)。

        上管道的 dict 里额外存 ``gap`` —— 缝隙大小逐根变化之后，
        势能函数不能再读 self.pipe_gap，必须读这根管道自己的值。

        randomize=False
        ---------------
        走原来的固定分布（8 个离散高度 + 固定缝隙），保持与既有存档可比。

        randomize=True —— 域随机化
        --------------------------
        缝隙大小、竖直位置、水平间距三者逐根独立采样，所以**同一局之内**
        地图就在不断变化，而不只是局与局之间不同。

        竖直位置有一个**可达性约束**，这是这段代码唯一有技术含量的地方：
        管道以 5px/帧左移，相邻管道间距 S px，于是小鸟有 S/5 帧的时间来
        完成竖直转移。而小鸟的极限竖直速度两个方向都是 5px/帧
        （持续扇翅时 velY 恒为 -5；自由落体的终端速度 playerMaxVelY = +5），
        所以这段时间内最多移动 ±S px。

        若不加约束地独立采样，相邻缝隙中心可能相差 224px，而间距 115px 时
        物理上根本到不了 —— 那不是"难"，是**不可学**：网络会收到一批
        无论如何都会死的样本，白白污染价值估计。所以把落差限制在
        ``max_delta_frac × S``（默认 0.6，留出余量给减速和缝隙本身的宽度）。
        """
        if not self.randomize:
            gap = float(self.pipe_gap)
            gap_top = random.choice(GAP_Y_CHOICES) + int(BASEY * 0.2)
            self._last_gap_center = gap_top + gap / 2.0
            self._last_slack = max(gap - PLAYER_HEIGHT, 0.0) / 2.0
        else:
            # 缝隙大小和水平间距由 _plan_next() 一起定好了 —— 必须一起，
            # 因为窄缝要求更大的间距，而间距在管道生成之前就要知道
            gap = self._next_gap
            lo, hi = self._center_range(gap, self._next_spacing)
            center = random.uniform(lo, hi) if lo < hi else (lo + hi) / 2.0

            self._last_gap_center = center
            self._last_slack = self._travel_slack(gap)
            gap_top = center - gap / 2.0

        self._plan_next()
        return (
            {'x': pipe_x, 'y': gap_top - PIPE_HEIGHT, 'gap': gap},  # 上管道
            {'x': pipe_x, 'y': gap_top + gap},                      # 下管道
        )

    def _plan_next(self):
        """预先决定下一根管道的缝隙大小和水平间距。

        两者必须一起定：间距的下界依赖缝隙大小（越窄要求越大的间距，
        好给小鸟更多时间对准），而间距又要在管道真正生成之前就用来
        判断"该不该生成了"。
        """
        if not self.randomize:
            self._next_gap = float(self.pipe_gap)
            self._next_spacing = SCREENWIDTH / 2.0
        else:
            self._next_gap = random.uniform(*self.gap_range)
            self._next_spacing = self._sample_spacing(self._next_gap)

    # ------------------------------------------------------------------
    # 可通过性模型
    # ------------------------------------------------------------------
    @staticmethod
    def _travel_slack(gap):
        """小鸟穿过这个缝隙时，中心位置还能偏离多少（像素）。

        缝隙的净空是 gap - PLAYER_HEIGHT，但其中 HOVER_AMPLITUDE 那部分
        被"稳住身形"的锯齿振荡占掉了，不能拿来当瞄准余量。
        剩下的一半就是中心可以偏移的范围。
        """
        band = max(gap - PLAYER_HEIGHT, 0.0)
        return max(band - HOVER_AMPLITUDE, 0.0) / 2.0

    def _center_range(self, gap, spacing):
        """这根管道的缝隙中心允许落在哪个区间 [lo, hi]。

        三重约束，缺一不可：

        **① 屏幕边界** —— 缝隙上下沿都要留出 edge_margin。

        **② 可达性** —— 管道以 5px/帧左移，间距 S 给小鸟 S/5 帧；
        小鸟两个方向的极限竖直速度都是 5px/帧，所以最多移动 ±S px。
        乘 max_delta_frac 留余量。

        **③ 小鸟并不在上一个缝隙的正中心。** 这一条是最初版本漏掉的，
        也是窄缝配大落差会变得几乎不可能的真正原因：

            小鸟可以合法地贴着上一个缝隙的边缘通过（偏移量 _last_slack），
            而它只需要摸到这个缝隙的边缘就算过（偏移量 slack）。
            所以最坏情况下需要走的距离是

                |Δcenter| + 上一个缝隙的偏移余量 − 这个缝隙的偏移余量

            缝隙越窄，slack 越小（60px 只有 6.75px，165px 有 59.25px），
            于是"从一个宽缝跳到一个窄缝"要求的行程远大于中心间距。

        合起来：

            |Δcenter| ≤ max_delta_frac·S − _last_slack + slack

        效果正是想要的：**上一根缝隙在上方、这一根又窄又在下方**这种组合，
        预算会直接变成负数，于是新缝隙被强制贴近上一根的高度。
        缝隙越窄，它能离上一根越远的余地就越小。
        """
        lo = self.edge_margin + gap / 2.0
        hi = BASEY - self.edge_margin - gap / 2.0
        if lo > hi:                      # 缝隙比可用高度还大，退化成居中
            return (BASEY / 2.0, BASEY / 2.0)
        if self._last_gap_center is None:
            return (lo, hi)

        budget = (self.max_delta_frac * spacing
                  - self._last_slack + self._travel_slack(gap))
        # 预算可能为负（宽缝 -> 窄缝且间距小）。夹到 0，表示"必须原地高度"。
        budget = max(budget, 0.0)

        lo = max(lo, self._last_gap_center - budget)
        hi = min(hi, self._last_gap_center + budget)
        if lo > hi:
            # 边界和可达性打架时，边界优先（不能把管道放到屏幕外），
            # 取屏幕内最接近上一根高度的那个点。
            c = min(max(self._last_gap_center, self.edge_margin + gap / 2.0),
                    BASEY - self.edge_margin - gap / 2.0)
            return (c, c)
        return (lo, hi)

    def _min_spacing_for(self, gap):
        """要让这个缝隙有起码的竖直活动余地，至少需要多大的水平间距。

        把上面的不等式反解：想让 budget 至少有 need 像素，就需要

            S ≥ (need + _last_slack − slack) / max_delta_frac

        窄缝的 slack 很小，所以自动会要求更大的间距 —— 也就是给小鸟
        更多时间去对准。这正是"窄缝要留出飞行空间"的直接实现。
        """
        if self._last_gap_center is None:
            return self.spacing_range[0]
        need = HOVER_AMPLITUDE          # 至少要能在一个悬停振幅内调整
        s = (need + self._last_slack - self._travel_slack(gap)) / self.max_delta_frac
        return float(s)

    def _sample_spacing(self, gap=None):
        """下一根管道与当前最后一根的水平间距。

        ``gap`` 给出时（随机化模式），窄缝会自动获得更大的间距下界。
        """
        if not self.randomize:
            return SCREENWIDTH / 2.0            # 原来的固定值 144
        lo, hi = self.spacing_range
        if gap is not None:
            lo = min(max(lo, self._min_spacing_for(gap)), hi)
        return random.uniform(lo, hi)

    # ------------------------------------------------------------------
    # 真实状态向量（诊断用，训练主管线不走这条路）
    # ------------------------------------------------------------------
    def state_vector(self):
        """把游戏的真实内部状态压成 8 维向量，各分量大致归一到 [-1,1]。

        这**不是**给主管线用的 —— 主管线的全部意义就是从像素学。
        它的用途是做一个"感知上界"对照：如果一个只有几万参数的 MLP
        拿着完美状态也过不了 N 根管道，那说明瓶颈在控制/物理，
        再怎么改网络和分辨率都没用；反之则说明瓶颈在感知。

        含**下两根**管道，因为 CNN 在 80x80 的画面里原则上也能同时看到两根，
        对照才公平。
        """
        nxt = []
        for u in self.upperPipes:
            if u['x'] + PIPE_WIDTH > self.playerx:
                nxt.append(u)
            if len(nxt) == 2:
                break
        while len(nxt) < 2:                      # 屏幕上不足两根时补一个远处的哨兵
            nxt.append({'x': SCREENWIDTH * 2.0,
                        'y': BASEY / 2 - PIPE_HEIGHT,
                        'gap': float(self.pipe_gap)})

        v = [self.playery / BASEY * 2.0 - 1.0,
             self.playerVelY / self.playerMaxVelY]
        for u in nxt:
            gap_center = u['y'] + PIPE_HEIGHT + u['gap'] / 2.0
            v += [(u['x'] - self.playerx) / SCREENWIDTH,
                  gap_center / BASEY * 2.0 - 1.0,
                  u['gap'] / 200.0]
        return np.asarray(v, dtype=np.float32)

    # ------------------------------------------------------------------
    # 势能函数（只暴露 Φ，差分由调用方在决策级计算）
    # ------------------------------------------------------------------
    def current_potential(self):
        """Φ(s) ∈ [-1, 0]，无量纲。终止态的 Φ 按定义取 0，由调用方处理。

        与旧实现的两点差异：
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

        # 缝隙大小逐根变化，所以必须读这根管道自己的 gap，不能读 self.pipe_gap
        gap_center = nxt['y'] + PIPE_HEIGHT + nxt['gap'] / 2.0
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

    def _observe(self):
        """把当前画面变成 (obs_w, obs_h) uint8 的 {0,255} 二值图。

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
            # view[:,:,0] 是 (288, 512) = (屏幕宽, 屏幕高)。
            # cv2 的 dsize 是 (cols, rows)，这里 cols 对应屏幕高、rows 对应屏幕宽，
            # 所以要写成 (obs_h, obs_w)，得到的数组形状是 (obs_w, obs_h)。
            small = cv2.resize(view[:, :, 0], (self.obs_h, self.obs_w),
                               interpolation=cv2.INTER_AREA)
        finally:
            del view                      # 解锁 Surface，务必执行
        return cv2.threshold(small, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    @staticmethod
    def raw_obs():
        """原始彩色帧 (288,512,3) uint8，width-major（转置的）。

        只给 play.py 的可视化用 —— 训练路径不该调用它（它就是那次
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
