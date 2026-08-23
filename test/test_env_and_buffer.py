"""
game/flappy_env.py 与 flappy/ 包的正确性单测。

每一个测试都对应一个在旧管线里真实存在过的 bug —— 注释里说明了旧实现错在哪，
以及它会在哪一行断言上失败。旧代码已删除，可在 git 历史中查阅。

运行：
    python test/test_env_and_buffer.py        # 独立运行
    pytest test/test_env_and_buffer.py -v     # 或用 pytest
"""

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import flappy_env
from game.flappy_env import (BASEY, PIPE_HEIGHT, PIPE_WIDTH, PLAYER_HEIGHT,
                             PLAYER_WIDTH, FlappyEnv)

GAMMA = 0.99
SHAPING_COEF = 0.05


# ======================================================================
# 1. 奖励累加 + 势能塑形的望远镜求和
# ======================================================================
def test_reward_accumulation_and_telescoping():
    """奖励必须逐项累加，且势能塑形必须精确望远镜求和。

    旧环境对塑形项做了 max(-0.01, min(0.01, ...)) 截断，破坏了
    Ng et al. 1999 的策略不变性，必然无法通过最后一条断言。
    """
    env = FlappyEnv()
    env.reset()

    # 关掉碰撞，让回合足够长，能跨过多根管道
    original_crash = flappy_env.checkCrash
    flappy_env.checkCrash = lambda *a, **kw: False
    try:
        phi_prev = env.current_potential()
        phi_0 = phi_prev

        raw_rewards = []
        discounted_shaping = 0.0
        crossings = 0
        t = 0

        player_mid = env.playerx + PLAYER_WIDTH / 2.0

        for _ in range(400):
            # 独立统计跨越次数：记录移动前后的管道中心
            mids_before = [p['x'] + PIPE_WIDTH / 2.0 for p in env.upperPipes]

            obs, r, done, info = env.step(0)
            raw_rewards.append(r)

            mids_after = [p['x'] + PIPE_WIDTH / 2.0 for p in env.upperPipes]
            for before, after in zip(mids_before, mids_after):
                if after <= player_mid < before:
                    crossings += 1

            phi_next = info['potential']          # done 时环境已返回 0.0
            discounted_shaping += (GAMMA ** t) * (GAMMA * phi_next - phi_prev)
            phi_prev = phi_next
            t += 1

            if done:
                break
    finally:
        flappy_env.checkCrash = original_crash

    # 得分判定与独立统计一致（旧代码的 4px 窗口 vs 5px 位移，只是碰巧没漏）
    assert env.score == crossings, f"score={env.score} vs crossings={crossings}"

    # 原始奖励精确等于 管道数×PIPE + 是否死亡×DEATH
    expected = env.pipe_reward * env.score + (env.death_reward if done else 0.0)
    assert abs(sum(raw_rewards) - expected) < 1e-9, \
        f"sum(rewards)={sum(raw_rewards)} != {expected}"

    # 望远镜求和的一般形式：Σ_{t<T} γ^t (γΦ(s_{t+1}) - Φ(s_t)) == γ^T Φ(s_T) - Φ(s_0)
    # 这里回合被人为拉长且未终止，所以 Φ(s_T) 不为 0，用一般形式。
    expected_tel = (GAMMA ** t) * phi_prev - phi_0
    assert abs(discounted_shaping - expected_tel) < 1e-6, \
        f"telescoping broken: {discounted_shaping} != {expected_tel}"

    print(f"  [1] OK  score={env.score} crossings={crossings} steps={t} "
          f"telescope_err={abs(discounted_shaping - expected_tel):.2e}")


def test_telescoping_on_terminating_episode():
    """自然终止的回合上，Φ(s_T) 按定义为 0，望远镜求和退化为 -Φ(s_0)。

    这是塑形项 **总量有界** 的保证：整局塑形回报的绝对值 ≤ |Φ(s_0)| ≤ 1，
    乘上 SHAPING_COEF=0.05 后比一根管道的 +1 低 20 倍，
    因此不可能盖过真实奖励。旧环境的 ±0.01 截断让这个保证不成立。
    """
    env = FlappyEnv()
    env.reset()

    phi_prev = env.current_potential()
    phi_0 = phi_prev
    discounted_shaping = 0.0
    t = 0

    while True:
        obs, r, done, info = env.step(0)
        phi_next = info['potential']          # done 时环境返回 0.0
        discounted_shaping += (GAMMA ** t) * (GAMMA * phi_next - phi_prev)
        phi_prev = phi_next
        t += 1
        if done:
            break

    assert abs(discounted_shaping - (-phi_0)) < 1e-6, \
        f"telescoping on terminal episode: {discounted_shaping} != {-phi_0}"

    # 塑形总量有界性
    assert abs(SHAPING_COEF * discounted_shaping) < 0.1, \
        f"塑形总量 {SHAPING_COEF * discounted_shaping} 过大"

    print(f"  [1b] OK  steps={t} Phi_0={phi_0:.4f} "
          f"shaping_return={SHAPING_COEF * discounted_shaping:+.5f} (|.|<0.1)")


# ======================================================================
# 2. 同一帧既得分又撞死 —— 两份奖励都要保留
# ======================================================================
def test_score_and_crash_same_frame():
    """旧代码的 `reward = -2` 是赋值，会把前面的 `reward = 20` 整个覆盖掉，
    同一帧既过管道又撞死时只返回 -2。
    """
    # 故意用不对称的奖励值：正确结果 -2.0 与"只拿死亡奖励"的 -3.0
    # 和"只拿管道奖励"的 +1.0 都不同，断言才有判别力
    env = FlappyEnv(pipe_reward=1.0, death_reward=-3.0)
    env.reset()

    # 构造：管道中心本帧从 player_mid 右侧越到左侧
    # player_mid = 57 + 17 = 74.0；需要 prev_mid ∈ (74, 79]，即 x ∈ (48, 53]
    pipe_x = 50
    env.upperPipes = [{'x': pipe_x, 'y': 0 - flappy_env.PIPE_HEIGHT}]
    env.lowerPipes = [{'x': pipe_x, 'y': 0 + env.pipe_gap}]

    # 构造：本帧撞地。checkCrash 判定 playery + PLAYER_HEIGHT >= BASEY - 1
    env.playery = 385
    env.playerVelY = 0.5

    obs, r, done, info = env.step(0)

    assert info['scored'] == 1, f"expected scored=1, got {info['scored']}"
    assert done, "expected crash"
    expected = env.pipe_reward + env.death_reward       # +1.0 + (-3.0) = -2.0
    assert abs(r - expected) < 1e-9, \
        f"reward={r}, expected {expected} (旧环境的赋值语义会返回 {env.death_reward})"
    assert abs(r - env.death_reward) > 1e-9, "管道奖励被死亡奖励覆盖了"
    assert abs(r - env.pipe_reward) > 1e-9, "死亡惩罚被管道奖励覆盖了"

    print(f"  [2] OK  scored=1 done=True reward={r:+.1f} "
          f"(= {env.pipe_reward:+.1f} {env.death_reward:+.1f}; "
          f"旧语义会给 {env.death_reward:+.1f})")


# ======================================================================
# 3. terminal 观测必须是真实的崩溃帧，且 done 后不可再 step
# ======================================================================
def test_terminal_frame_is_crash_frame():
    """旧代码在绘制之前调用 self.__init__()，
    于是 terminal 那一帧返回的其实是下一局的第一帧 —— 旧环境里
    obs_terminal 与 reset() 的观测是 **相等** 的。
    """
    env = FlappyEnv()
    env.reset()

    obs_T = None
    crash_y = None
    for _ in range(2000):
        obs_T, r, done, info = env.step(0)      # 一直不跳，必然坠地
        if done:
            crash_y = env.playery
            break
    assert obs_T is not None and done, "episode did not terminate"

    obs_0 = env.reset()

    assert not np.array_equal(obs_T, obs_0), \
        "terminal 观测与 reset 观测相同 —— 说明崩溃帧被下一局首帧覆盖了"

    # 崩溃时鸟确实在地面附近，证明画的是崩溃姿态而非重置姿态
    assert crash_y + PLAYER_HEIGHT >= BASEY - 1 - 1e-6, \
        f"crash_y={crash_y} 不在地面"

    # done 之后再 step 必须报错，而不是静默重置
    env2 = FlappyEnv()
    env2.reset()
    while True:
        _, _, done2, _ = env2.step(0)
        if done2:
            break
    try:
        env2.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("done 之后 step() 应该抛 RuntimeError")

    print(f"  [3] OK  crash_y={crash_y:.1f} (BASEY-H={BASEY - PLAYER_HEIGHT:.1f}), "
          f"terminal!=reset, post-done step raises")


# ======================================================================
# 4. 经验池往返：帧栈必须严格对齐，且绝不跨回合
# ======================================================================
def test_replay_roundtrip_and_stack_integrity():
    """采样出来的 s 必须是 [i, i-1, i-2, i-3]，s1 必须是 [i+1, i, i-1, i-2]。

    对应旧管线的缺陷：terminal 那一轮仍执行无条件的 s_t = s_t1，
    导致新回合前 3 次决策的帧栈里混着上一局的画面，且以 done=False 存入
    —— 约 23% 的经验是物理不可能的状态。
    """
    from flappy.replay import ReplayBuffer

    STACK = 4
    # 刻意用非正方形，才能抓到把 h/w 写反的错误
    H, W = 80, 128
    buf = ReplayBuffer(capacity=500, stack=STACK, h=H, w=W)

    # 合成帧：第 ep 回合的第 i 帧全部填充值 (ep*50 + i)，可从像素值反推来源
    def make_frame(ep, i):
        return np.full((H, W), (ep * 50 + i) % 256, dtype=np.uint8)

    records = []          # (ep, i) —— 这条经验的 state 最新帧应当是第 i 帧
    for ep in range(3):
        frame = make_frame(ep, 0)
        stack = np.repeat(frame[None], STACK, axis=0)     # 回合开始：重置帧栈
        for i in range(1, 11):
            nf = make_frame(ep, i)
            done = (i == 10)
            buf.add(stack, i % 2, float(i), nf, done)
            records.append((ep, i - 1))                   # state 最新帧是第 i-1 帧
            if done:
                break
            stack = np.concatenate([nf[None], stack[:STACK - 1]], axis=0)

    assert len(buf) == len(records)

    # 逐条核对整个缓冲区
    for k, (ep, newest) in enumerate(records):
        s = buf.states[k]
        nf = buf.next_frames[k, 0]
        s1 = np.concatenate([nf[None], s[:STACK - 1]], axis=0)

        for j in range(STACK):
            want = make_frame(ep, max(newest - j, 0))     # 回合开头是重复的第 0 帧
            assert np.array_equal(s[j], want), \
                f"record {k}: s[{j}] mismatch (ep={ep}, newest={newest})"
        for j in range(STACK):
            want = make_frame(ep, max(newest + 1 - j, 0))
            assert np.array_equal(s1[j], want), \
                f"record {k}: s1[{j}] mismatch (ep={ep}, newest={newest})"

        # 无跨回合：整个帧栈的像素值必须落在本回合的编码区间内
        for j in range(STACK):
            v = int(s[j][0, 0])
            assert ep * 50 <= v <= ep * 50 + 10, \
                f"record {k}: frame from another episode (value={v}, ep={ep})"

    # sample() 的拼接与上面手工拼的一致
    s, a, r, s1, d = buf.sample(64)
    assert s.shape == (64, STACK, H, W) and s1.shape == (64, STACK, H, W)
    assert s.dtype == np.uint8 and s1.dtype == np.uint8
    assert np.array_equal(s1[:, 1:], s[:, :STACK - 1]), \
        "sample() 拼出的 next_state 与 state 不满足滑窗关系"

    print(f"  [4] OK  {len(records)} records, stacks aligned, no cross-episode frames, "
          f"dtype=uint8")


# ======================================================================
# 5. 帧跳过对齐：动作真被重复 k 帧，奖励全额累加
# ======================================================================
def test_frame_skip_alignment():
    """对应旧管线最严重的缺陷：`agent.step += 1` 夹在两个 `% k` 判断之间，
    动作在 step≡0 时选、经验在 step≡3 时存，
    两者永不同时成立 —— 存进去的动作标签是"选择的动作"，而那一帧实际执行的
    是硬编码的不跳 [1,0]，同时 75% 的奖励与 terminal 被丢弃。
    """
    from flappy.config import CONFIG
    from flappy.rollout import skip_step

    cfg = dict(CONFIG)
    K = cfg['frame_skip']

    env = FlappyEnv(pipe_reward=cfg['pipe_reward'],
                    death_reward=cfg['death_reward'],
                    alive_reward=cfg['alive_reward'],
                    pipe_gap=cfg['pipe_gap'])
    env.reset()

    # 记录每一帧实际施加的动作和产生的原始奖励
    applied, frame_rewards = [], []
    raw_step = env.step

    rendered = []

    def spy_step(action, render=True):
        out = raw_step(action, render=render)
        applied.append(action)
        frame_rewards.append(out[1])
        rendered.append(render)
        return out

    env.step = spy_step

    # 关掉碰撞，保证跑满 60 次决策（含跨管道），覆盖面才够
    original_crash = flappy_env.checkCrash
    flappy_env.checkCrash = lambda *a, **kw: False
    try:
        phi = env.current_potential()
        decisions, skip_lens, dec_rewards, boundaries = [], [], [], []
        for n in range(60):
            action = 1 if n % 3 == 0 else 0    # 有跳有不跳，两个分支都覆盖
            start = len(applied)
            nf, r_dec, done, info, phi = skip_step(env, action, phi, cfg)
            decisions.append(action)
            skip_lens.append(len(applied) - start)
            boundaries.append((start, len(applied)))
            dec_rewards.append(r_dec)
            if done:
                break
    finally:
        flappy_env.checkCrash = original_crash

    # 每次决策消耗 K 帧（终止那次可能不足 K）
    for j, n_frames in enumerate(skip_lens[:-1]):
        assert n_frames == K, f"decision {j} consumed {n_frames} frames, expected {K}"
    assert skip_lens[-1] <= K

    # 动作确实在整个窗口里重复
    for j, (lo, hi) in enumerate(boundaries):
        assert applied[lo:hi] == [decisions[j]] * (hi - lo), \
            f"decision {j}: applied {applied[lo:hi]}, expected {[decisions[j]]*(hi-lo)}"

    # 帧数守恒：没有任何一帧落在窗口之外
    assert sum(skip_lens) == len(applied)

    # 决策奖励 = 窗口内所有帧奖励之和 + 塑形项（一分不丢）
    shaping_total = 0.0
    for j, (lo, hi) in enumerate(boundaries):
        raw_sum = sum(frame_rewards[lo:hi])
        # 塑形项绝对值有上界，用它来核对 r_dec 与 raw_sum 的差
        diff = dec_rewards[j] - raw_sum
        shaping_total += diff
        assert abs(diff) < 0.01, \
            f"decision {j}: r_dec-raw_sum={diff} 超出塑形项的量级上界"

    # 帧跳过窗口内只有最后一帧渲染 —— 这是 11 倍提速的来源，顺便断言它确实生效
    for j, (lo, hi) in enumerate(boundaries):
        want = [False] * (hi - lo - 1) + [True]
        assert rendered[lo:hi] == want, \
            f"decision {j}: rendered={rendered[lo:hi]}, expected {want}"
    n_rendered = sum(rendered)
    assert n_rendered == len(decisions), \
        f"rendered {n_rendered} frames for {len(decisions)} decisions"

    print(f"  [5] OK  {len(decisions)} decisions, {len(applied)} frames, "
          f"action repeat exact, reward loss = 0 (shaping total={shaping_total:+.5f}), "
          f"rendered only {n_rendered}/{len(applied)} frames")


# ======================================================================
# 6. 动作选择必须是确定性的
# ======================================================================
def test_act_time_determinism():
    """一行断言，当初就能抓到旧网络里的 Dropout(0.3) ——
    它让"贪婪"动作变成随机，同时让目标网络的 BatchNorm 用 minibatch 统计量，
    使 TD 目标随批次组成而变，Bellman 算子失去不动点。
    """
    import torch

    from flappy.model import DuelingDQN

    net = DuelingDQN()
    x = torch.randint(0, 2, (5, 4, flappy_env.OBS_W, flappy_env.OBS_H), dtype=torch.uint8) * 255

    net.train()                       # 即便处于 train 模式也必须确定
    q1, q2 = net(x), net(x)
    assert torch.allclose(q1, q2), "网络在 act 时是随机的（Dropout/BatchNorm 未关）"

    net.eval()
    q3 = net(x)
    assert torch.allclose(q1, q3), "train/eval 模式下输出不一致（存在 BatchNorm/Dropout）"

    # 确认网络里确实没有这两类层
    bad = [type(m).__name__ for m in net.modules()
           if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                             torch.nn.BatchNorm2d))]
    assert not bad, f"网络里仍存在 {bad}"

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  [6] OK  deterministic in train() and eval(), no Dropout/BatchNorm, "
          f"params={n_params} ({n_params * 4 / 1e6:.2f} MB)")


# ======================================================================
# 7. render=False 绝不改变游戏本身
# ======================================================================
def test_render_flag_does_not_change_physics():
    """跳过渲染是整个项目最大的单项提速（端到端 11.4 倍），前提是它对
    游戏状态完全没有副作用。碰撞检测走 checkCrash 的 hitmask + 坐标，
    不读渲染结果 —— 这个测试把这一点钉死。
    """
    FIELDS = ('playery', 'playerVelY', 'playerFlapped', 'score', 'frames',
              'basex', 'loopIter', 'playerIndex')

    def snapshot(e):
        base = {f: getattr(e, f) for f in FIELDS}
        base['upper'] = [(p['x'], p['y']) for p in e.upperPipes]
        base['lower'] = [(p['x'], p['y']) for p in e.lowerPipes]
        base['done'] = e.done
        return base

    actions = [1 if i % 3 == 0 else 0 for i in range(300)]

    traj = []
    for render in (True, False):
        random.seed(1234)                     # 管道生成用全局 random，必须同种子
        env = FlappyEnv()
        env.reset()
        states, rewards = [], []
        for a in actions:
            if env.done:
                random.seed(9999)             # reset 也会抽管道，保持两次一致
                env.reset()
            _, r, _, info = env.step(a, render=render)
            states.append(snapshot(env))
            rewards.append(r)
        traj.append((states, rewards))

    (s_on, r_on), (s_off, r_off) = traj
    assert len(s_on) == len(s_off)
    for i, (a, b) in enumerate(zip(s_on, s_off)):
        assert a == b, f"step {i}: physics differ between render=True/False\n{a}\n{b}"
    assert r_on == r_off, "rewards differ between render=True/False"

    print(f"  [7] OK  {len(actions)} steps, physics + rewards bit-identical "
          f"with and without rendering")


# ======================================================================
# 8. 新取图路径与旧 array3d 路径逐像素等价
# ======================================================================
def test_observation_matches_legacy_pipeline():
    """新路径用 pixels3d 视图 + 单通道 INTER_AREA（234us），
    旧路径用 array3d 拷贝 + BGR2GRAY + INTER_AREA（977us）。
    速度差 4.2 倍，但网络看到的东西必须一样，否则等于偷偷换了任务。
    """
    env = FlappyEnv()
    env.reset()

    def legacy():
        raw = env.raw_obs()                                   # (288,512,3)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # dsize 是 (cols, rows)：cols 对应屏幕高、rows 对应屏幕宽，
        # 所以是 (OBS_H, OBS_W)，与 flappy_env._observe 一致
        small = cv2.resize(gray, (flappy_env.OBS_H, flappy_env.OBS_W),
                           interpolation=cv2.INTER_AREA)
        return cv2.threshold(small, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    random.seed(7)
    mismatched = total = 0
    expected = (flappy_env.OBS_W, flappy_env.OBS_H)
    for i in range(200):
        if env.done:
            env.reset()
        new = env.step(1 if random.random() < 0.2 else 0, render=True)[0]
        old = legacy()                       # raw_obs 读的是同一块已绘制的 SCREEN
        assert new.shape == expected and new.dtype == np.uint8, \
            f"观测形状 {new.shape} != {expected}"
        assert set(np.unique(new)).issubset({0, 255}), "观测必须是 {0,255} 二值"
        mismatched += int((new != old).sum())
        total += new.size

    rate = mismatched / total
    assert rate < 1e-4, f"新旧取图路径不一致率 {rate:.6%} 超过 0.01%"

    print(f"  [8] OK  {total} pixels compared, {mismatched} differ "
          f"({rate:.6%} < 0.01%)")


# ======================================================================
# 9. 难度旋钮：管道间隙可调且真的生效
# ======================================================================
def test_pipe_gap_is_configurable():
    """间隙是难度的唯一旋钮，必须实测生效。

    旧实现把间隙作为模块级常量 PIPEGAPSIZE 写死在管道生成函数的返回值里，
    根本调不了。所以 FlappyEnv 用自己的 _random_pipe()，间隙走实例属性。
    """
    from game.flappy_env import DEFAULT_PIPE_GAP

    for gap in (DEFAULT_PIPE_GAP, 100, 80):
        env = FlappyEnv(pipe_gap=gap)
        env.reset()
        # 跑一段，让管道自然生成/回收，覆盖 reset 和 step 两条生成路径
        seen = 0
        for i in range(500):
            if env.done:
                env.reset()
            env.step(1 if i % 5 == 0 else 0, render=False)
            for u, l in zip(env.upperPipes, env.lowerPipes):
                measured = l['y'] - (u['y'] + PIPE_HEIGHT)
                assert measured == gap, \
                    f"gap={gap} 但实测间隙是 {measured}"
                seen += 1
        assert seen > 0
    # 太小的间隙必须拒绝，而不是训出一个永远撞死的智能体
    try:
        FlappyEnv(pipe_gap=PLAYER_HEIGHT)
    except ValueError:
        pass
    else:
        raise AssertionError("间隙小于小鸟高度时应当报错")

    print(f"  [9] OK  gap configurable ({DEFAULT_PIPE_GAP}/100/80 all verified), "
          f"impossible gaps rejected")


# ======================================================================
# 10. 域随机化：分布要真的变宽，且不能生成物理上到不了的管道
# ======================================================================
def test_pipe_randomization():
    """随机化必须同时满足三件事，缺一不可。

    (a) 分布真的变宽 —— 否则等于没做。固定模式下缝隙中心只有 8 个离散取值、
        跨度 70px（可用空间的 17%），缝隙大小和水平间距完全固定。

    (b) **最坏情况下**的竖直需求速度不超过物理上限 —— 否则会生成
        无论如何都过不去的管道。那不是"更难"，是不可学：网络收到一批
        必死样本，白白污染价值估计。

        注意这里检查的**不是** |Δcenter| / 间距。那个比值现在可以超过
        max_delta_frac，而且是正确的：从窄缝跳到宽缝时，落点范围更大，
        本来就可以跳得更远。真正该守的不变量是把两端的落点余量都算进去的
        需求行程：

            need = |Δcenter| + slack_prev − slack_next
            need / (S/5) ≤ 5 · max_delta_frac

        其中 slack 是扣掉悬停振荡之后、缝隙中心还能偏离多少
        （见 flappy_env._travel_slack）。

    (c) 窄缝必须拿到更多飞行空间 —— 缝隙越窄，允许的落差越小。
        否则会出现"上一根缝隙在最上面，下一根又窄又在最下面"这种
        几何上够得着、实际却几乎必死的组合。
    """
    import numpy as np

    original_crash = flappy_env.checkCrash
    flappy_env.checkCrash = lambda *a, **kw: False      # 让回合跑得够长
    try:
        FRAC = 0.6
        PIPE_VEL = 5.0            # 管道左移速度 px/帧
        BIRD_VEL = 5.0            # 小鸟极限竖直速度 px/帧（两个方向都是）

        def survey(**kw):
            random.seed(7)
            env = FlappyEnv(**kw)
            centers, gaps = [], []
            pairs = []            # (间距, 上一根gap, 这一根gap, 落差)
            for _ in range(12):
                env.reset()
                for _ in range(600):
                    env.step(0, render=False)
                    ps = sorted(env.upperPipes, key=lambda p: p['x'])
                    for p in ps:
                        centers.append(p['y'] + PIPE_HEIGHT + p['gap'] / 2.0)
                        gaps.append(p['gap'])
                    for a, b in zip(ps, ps[1:]):
                        ca = a['y'] + PIPE_HEIGHT + a['gap'] / 2.0
                        cb = b['y'] + PIPE_HEIGHT + b['gap'] / 2.0
                        pairs.append((b['x'] - a['x'], a['gap'], b['gap'],
                                      abs(cb - ca)))
            return np.array(centers), np.array(gaps), np.array(pairs)

        c0, g0, p0 = survey(pipe_gap=100)
        c1, g1, p1 = survey(pipe_gap=100, randomize=True, max_delta_frac=FRAC)
        s0, s1 = p0[:, 0], p1[:, 0]

        # (a) 每一个维度都必须显著变宽
        assert np.ptp(c1) > 3 * np.ptp(c0), \
            f"缝隙中心跨度 {np.ptp(c1):.0f} 没有明显超过固定模式的 {np.ptp(c0):.0f}"
        assert len(np.unique(np.round(c1, 1))) > 100, \
            f"缝隙中心只有 {len(np.unique(np.round(c1,1)))} 个取值，太离散"
        assert np.ptp(g0) == 0 and np.ptp(g1) > 50, \
            f"缝隙大小跨度：固定 {np.ptp(g0):.0f} -> 随机 {np.ptp(g1):.0f}"
        assert np.ptp(s0) == 0 and np.ptp(s1) > 50, \
            f"水平间距跨度：固定 {np.ptp(s0):.0f} -> 随机 {np.ptp(s1):.0f}"

        # (b) 最坏情况的需求速度不许超过物理上限
        spacing, gap_a, gap_b, delta = p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 3]
        slack_a = np.array([flappy_env.FlappyEnv._travel_slack(g) for g in gap_a])
        slack_b = np.array([flappy_env.FlappyEnv._travel_slack(g) for g in gap_b])
        need = np.maximum(delta + slack_a - slack_b, 0.0)
        speed = need / (spacing / PIPE_VEL)
        limit = BIRD_VEL * FRAC
        assert speed.max() <= limit + 1e-6, (
            f"最坏需求速度 {speed.max():.3f} px/帧 超过了 {limit:.3f} "
            f"—— 会生成实际过不去的管道")

        # (c) 窄缝必须比宽缝拿到更小的落差（= 更多飞行空间）。
        #     默认 gap_range 下界是 80，所以要专门开一组含 60px 窄缝的样本。
        _, _, pn = survey(pipe_gap=100, randomize=True, max_delta_frac=FRAC,
                          gap_range=(60., 165.))
        n_spacing, n_gap_a, n_gap_b, n_delta = pn[:, 0], pn[:, 1], pn[:, 2], pn[:, 3]
        narrow = n_gap_b <= 70
        wide = n_gap_b >= 120
        assert narrow.any() and wide.any(), "样本里没有覆盖到窄缝或宽缝"
        assert n_delta[narrow].mean() < n_delta[wide].mean(), (
            f"窄缝平均落差 {n_delta[narrow].mean():.1f} 没有小于宽缝的 "
            f"{n_delta[wide].mean():.1f} —— 窄缝没有拿到额外的飞行空间")

        # 窄缝那一组同样不许突破速度上限
        n_slack_a = np.array([flappy_env.FlappyEnv._travel_slack(g) for g in n_gap_a])
        n_slack_b = np.array([flappy_env.FlappyEnv._travel_slack(g) for g in n_gap_b])
        n_speed = np.maximum(n_delta + n_slack_a - n_slack_b, 0.0) / (n_spacing / PIPE_VEL)
        assert n_speed.max() <= limit + 1e-6, (
            f"含 60px 窄缝时最坏需求速度 {n_speed.max():.3f} 超过了 {limit:.3f}")

        # 缝隙必须始终留在屏幕内（上下沿都不能越界）
        assert (c1 - g1 / 2.0).min() >= 0, "缝隙上沿越过了天花板"
        assert (c1 + g1 / 2.0).max() <= BASEY, "缝隙下沿越过了地面"

        print(f"  [10] OK  center span {np.ptp(c0):.0f}->{np.ptp(c1):.0f}px "
              f"({len(np.unique(np.round(c1,1)))} values), "
              f"gap {np.ptp(g0):.0f}->{np.ptp(g1):.0f}px, "
              f"spacing {np.ptp(s0):.0f}->{np.ptp(s1):.0f}px")
        print(f"           worst-case speed {speed.max():.2f} (60px set {n_speed.max():.2f}) "
              f"<= {limit:.2f} px/frame; "
              f"delta narrow {n_delta[narrow].mean():.1f} < wide {n_delta[wide].mean():.1f} px")
    finally:
        flappy_env.checkCrash = original_crash


# ======================================================================
# 11. n-step：回报累加正确、s_{t+n} 重建正确、绝不跨回合
# ======================================================================
def test_n_step_returns():
    """n-step 有两个**必须**同时正确、否则会静默学歪的地方。

    (a) **回报**：R_n = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1}，
        回合中途结束时要截断，且剩下的前缀都得吐出来 ——
        否则每个回合末尾的 n-1 条经验会丢失，而那些恰恰包含死亡信息。

    (b) **s_{t+n} 的重建**：1-step 的 next_state = concat(next_frame, s[:3])
        在 n-step 下**不成立**。s_t 与 s_{t+n} 只重叠 stack-n 帧：

            s_t     = [f_t,   f_t-1, f_t-2, f_t-3]
            s_{t+3} = [f_t+3, f_t+2, f_t+1, f_t  ]

        照搬 1-step 的写法会让网络拿 s_{t+1} 当 s_{t+n}，
        而奖励是 n 步累加的 —— 状态和目标错位，loss 照样下降，分数上不去。
    """
    from flappy.replay import NStepAccumulator, ReplayBuffer

    STACK, N, GAMMA = 4, 3, 0.9
    H, W = 8, 12                       # 小尺寸，便于逐帧核对
    acc = NStepAccumulator(N, GAMMA)
    buf = ReplayBuffer(capacity=200, stack=STACK, h=H, w=W, n_step=N)

    def frame(tag):                    # 用填充值编码帧的身份
        return np.full((H, W), tag, dtype=np.uint8)

    def stack_of(t):                   # 时刻 t 的帧栈：新帧在前
        return np.stack([frame((t - i) % 256) for i in range(STACK)])

    # ---- (a) 回报累加 + 截断 ----
    acc.reset()
    rewards = [1.0, 2.0, 4.0, 8.0, 16.0]
    outs = []
    for t, r in enumerate(rewards):
        done = (t == len(rewards) - 1)
        outs += acc.push(stack_of(t + STACK), t % 2, r, frame(t + STACK + 1), done)

    # 5 步、n=3：应当产出 5 条（前 3 条是完整 n-step，末尾 2 条被截断）
    assert len(outs) == 5, f"应产出 5 条经验，实得 {len(outs)}"
    for i, (_, _, R, _, dn) in enumerate(outs):
        span = min(N, len(rewards) - i)
        want = sum((GAMMA ** k) * rewards[i + k] for k in range(span))
        assert abs(R - want) < 1e-9, \
            f"第 {i} 条 n-step 回报 {R} != {want}"
        # 只有窗口覆盖到最后一步的那些才该是 done
        assert dn == (i + span == len(rewards)), \
            f"第 {i} 条 done={dn} 不对"

    # ---- (b) s_{t+n} 的重建 ----
    buf_state = stack_of(10)                      # [10, 9, 8, 7]
    nf = np.stack([frame(11), frame(12), frame(13)])   # 时间正序
    buf.add(buf_state, 1, 0.5, nf, False)
    s, a, r, s1, d = buf.sample(1)

    want_s1 = np.stack([frame(13), frame(12), frame(11), frame(10)])
    assert np.array_equal(s1[0], want_s1), (
        "s_{t+n} 重建错误\n实得首帧值 %s\n期望 %s"
        % (s1[0][:, 0, 0], want_s1[:, 0, 0]))
    # 1-step 的错误写法会得到 [11, 10, 9, 8]，这里必须区分开
    wrong = np.stack([frame(11), frame(10), frame(9), frame(8)])
    assert not np.array_equal(s1[0], wrong), "退化成了 1-step 的重建"

    # ---- 跨回合污染 ----
    acc.reset()
    acc.push(stack_of(100), 0, 1.0, frame(101), False)
    acc.reset()                                    # 新回合
    out2 = acc.push(stack_of(200), 0, 1.0, frame(201), True)
    assert len(out2) == 1 and abs(out2[0][2] - 1.0) < 1e-9, \
        "reset() 之后仍混入了上一回合的奖励"

    # n_step > stack 必须被拒绝（重叠帧不够，压缩存储不成立）
    try:
        ReplayBuffer(capacity=10, stack=4, h=H, w=W, n_step=5)
    except ValueError:
        pass
    else:
        raise AssertionError("n_step > stack 时应当报错")

    print(f"  [11] OK  n={N} returns exact (5 transitions incl. truncated tail), "
          f"s_t+n rebuilt correctly, no cross-episode leak, n>stack rejected")


# ======================================================================
def _run_all():
    tests = [
        test_reward_accumulation_and_telescoping,
        test_telescoping_on_terminating_episode,
        test_score_and_crash_same_frame,
        test_terminal_frame_is_crash_frame,
        test_replay_roundtrip_and_stack_integrity,
        test_frame_skip_alignment,
        test_act_time_determinism,
        test_render_flag_does_not_change_physics,
        test_observation_matches_legacy_pipeline,
        test_pipe_gap_is_configurable,
        test_pipe_randomization,
        test_n_step_returns,
    ]
    print(f"running {len(tests)} env tests...")
    for fn in tests:
        fn()
    print("all env tests passed")


if __name__ == "__main__":
    _run_all()
