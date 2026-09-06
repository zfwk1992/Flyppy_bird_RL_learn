"""导出 Python 侧的参考轨迹，供 JS 逐帧比对（阶段 1 的验收项）。

为什么不能"两边用同一个 seed"
------------------------------
Python 的 `random` 是 Mersenne Twister，JS 里没有等价实现，硬对齐要把
MT19937 移植过去 —— 没必要。真正要验的是**移植的逻辑**，不是两个 PRNG。

所以这里把 Python 消费的每一个 `random()` 原始值记下来，让 JS 按同样顺序
重放。`random.uniform(a,b)` 就是 `a + (b-a)*random()`，JS 的 `_uniform` 是
`lo + r*(hi-lo)` —— 同一个 r 进去，IEEE754 下逐位相同。

于是比对拆成两件互不干扰的事：
  1. **物理**：小鸟 y / 速度只由动作决定，与随机数无关 —— 必须逐帧精确一致
  2. **管道生成**：喂同一串 random()，管道 x / y / gap 必须逐位一致

动作也一并记录下来让 JS **重放**（而不是重算）。这样即使策略依赖状态，
一点点分歧也不会让两条轨迹发散成完全不同的东西，第一处不一致能直接定位。

用法：
    python web/tools/dump_python_trace.py        # 写到 web/tools/trace.json
"""
import json
import os
import random
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# 必须 patch 类方法：random.uniform 绑定在模块级 _inst 上，内部调 self.random()，
# patch 模块级 random.random 抓不到。
_DRAWS = []
_orig_random = random.Random.random


def _recording_random(self):
    v = _orig_random(self)
    _DRAWS.append(v)
    return v


random.Random.random = _recording_random

from flappy.config import resolve_config                                # noqa: E402
from flappy.rollout import make_env                                     # noqa: E402
from game.flappy_env import PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT      # noqa: E402

N_FRAMES = 1200
SEED = 12345

# 第二条轨迹用**训练好的模型**驱动。理由：bang-bang 启发式撞得很干脆，
# 覆盖不到"擦着管道边缘飞过"的情形，于是包围盒和像素掩码的差异测不出来
# —— 那样 parity 就没有鉴别力。真实模型有 4.6% 的通过是擦边的，
# 正好压在两种碰撞判定会分歧的地方。
USE_MODEL = os.environ.get("TRACE_POLICY", "heuristic") == "model"


def chase_gap(e):
    """朴素 bang-bang 控制，只为让轨迹覆盖更多管道，不追求成绩。"""
    nxt = None
    for u in e.upperPipes:
        if u['x'] + PIPE_WIDTH > e.playerx:
            nxt = u
            break
    if nxt is None:
        return 0
    center = nxt['y'] + PIPE_HEIGHT + nxt['gap'] / 2.0
    return 1 if e.playery + PLAYER_HEIGHT / 2.0 > center else 0


def snapshot(e, i, action, score, done, did_reset):
    return {
        "i": i, "action": action, "done": bool(done), "reset": bool(did_reset),
        "playery": e.playery, "playerVelY": e.playerVelY,
        "score": score, "basex": e.basex,
        "upper": [{"x": p["x"], "y": p["y"], "gap": p["gap"]} for p in e.upperPipes],
        "lower": [{"x": p["x"], "y": p["y"]} for p in e.lowerPipes],
    }


cfg = resolve_config()
random.seed(SEED)
env = make_env(cfg)

policy = None
if USE_MODEL:
    import numpy as np
    import torch
    from flappy import checkpoint
    from flappy.rollout import FrameStack
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 权重路径可以用环境变量覆盖。网页换模型时参考轨迹必须跟着重录，
    # 否则 dump_nn_ref.py 里那条“重放出的动作要和 trace 记录的一致”的断言
    # 必然失败 —— 不同权重在同一状态上本来就会选不同动作。
    _ckpt = os.environ.get("FLAPPY_WEB_CKPT", "models/final_v1_best.pt")
    _net, _mcfg, _ = checkpoint.load_for_inference(_ckpt, _dev)
    _stack = FrameStack(cfg["frame_stack"])
    _state = {"stack": None}

    def policy(e, first):
        obs = e.observe()
        _state["stack"] = (_stack.reset(obs) if first
                           else _stack.push(obs))
        with torch.no_grad():
            t = torch.from_numpy(_state["stack"]).unsqueeze(0).to(_dev)
            return int(_net(t).argmax(1).item())

env.reset()

frames = []
episodes = 1
_just_reset = True
for i in range(N_FRAMES):
    if policy is not None:
        # 模型按决策级动作（frame_skip=4）；这里按帧记录，所以每 4 帧问一次，
        # 窗口内重复同一个动作 —— 与 rollout.skip_step 的语义一致
        if i % cfg["frame_skip"] == 0:
            _cur_action = policy(env, _just_reset)
            _just_reset = False
        action = _cur_action
    else:
        action = chase_gap(env)
    # 模型策略要读画面，所以必须渲染
    obs, r, done, info = env.step(action, render=(policy is not None))
    # 快照必须在 reset **之前** 取：done 那一帧要记的是撞死时的状态，
    # 不是重开后的状态。写反了会让比对在每个 done 帧上假报不一致
    # （PY 记成 224/0/0 的重开值，JS 记的是崩溃值）。
    frames.append(snapshot(env, i, action, info["score"], done, done))
    if done:
        # 立刻重开，让轨迹足够长以覆盖更多管道；顺带验证 reset 的一致性。
        # reset 也会消费随机数，JS 端必须在比对完这一帧之后同样 reset，
        # 两边的抽样顺序才对得上。
        env.reset()
        episodes += 1
        _just_reset = True

out = {
    "meta": {
        "seed": SEED, "n_frames": len(frames), "n_draws": len(_DRAWS),
        "episodes": episodes,
        "gap_range": list(cfg["pipe_gap_range"]),
        "spacing_range": list(cfg["pipe_spacing_range"]),
        "edge_margin": cfg["pipe_edge_margin"],
        "max_delta_frac": cfg["pipe_max_delta_frac"],
        "randomize": cfg["randomize_pipes"],
        "pipe_gap": cfg["pipe_gap"],
    },
    "draws": _DRAWS,
    "frames": frames,
}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "trace_ai.json" if USE_MODEL else "trace.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f)
print("wrote %s : %d frames, %d draws, %d episodes"
      % (dst, len(frames), len(_DRAWS), episodes))
