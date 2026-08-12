"""
game/flappy_env.py 与 train_flappy_dqn.py 的正确性单测。

每一个测试都对应一个在旧管线里真实存在过的 bug —— 注释里标注了旧代码的位置
以及旧实现会在哪一行断言上失败。

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

    旧环境在 wrapped_flappy_bird_fast.py:213 对塑形项做了
    max(-0.01, min(0.01, ...)) 截断，破坏了 Ng et al. 1999 的策略不变性，
    必然无法通过最后一条断言。
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
    """旧代码 wrapped_flappy_bird_fast.py:156 的 `reward = -2` 是赋值，
    会把 :113 的 `reward = 20` 整个覆盖掉，返回 -2。
    """
    # 故意用不对称的奖励值：正确结果 -2.0 与"只拿死亡奖励"的 -3.0
    # 和"只拿管道奖励"的 +1.0 都不同，断言才有判别力
    env = FlappyEnv(pipe_reward=1.0, death_reward=-3.0)
    env.reset()

    # 构造：管道中心本帧从 player_mid 右侧越到左侧
    # player_mid = 57 + 17 = 74.0；需要 prev_mid ∈ (74, 79]，即 x ∈ (48, 53]
    pipe_x = 50
    env.upperPipes = [{'x': pipe_x, 'y': 0 - flappy_env.PIPE_HEIGHT}]
    env.lowerPipes = [{'x': pipe_x, 'y': 0 + flappy_env.PIPEGAPSIZE}]

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
    """旧代码 wrapped_flappy_bird_fast.py:154 在绘制之前调用 self.__init__()，
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

    对应旧管线的缺陷：terminal 那一轮仍执行无条件的 s_t = s_t1
    (deep_Q_dueling_DQN.py:785)，导致新回合前 3 次决策的帧栈里混着上一局的
    画面，且以 done=False 存入 —— 约 23% 的经验是物理不可能的状态。
    """
    from train_flappy_dqn import ReplayBuffer

    STACK = 4
    buf = ReplayBuffer(capacity=500, stack=STACK, size=80)

    # 合成帧：第 ep 回合的第 i 帧全部填充值 (ep*50 + i)，可从像素值反推来源
    def make_frame(ep, i):
        return np.full((80, 80), (ep * 50 + i) % 256, dtype=np.uint8)

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
    assert s.shape == (64, STACK, 80, 80) and s1.shape == (64, STACK, 80, 80)
    assert s.dtype == np.uint8 and s1.dtype == np.uint8
    assert np.array_equal(s1[:, 1:], s[:, :STACK - 1]), \
        "sample() 拼出的 next_state 与 state 不满足滑窗关系"

    print(f"  [4] OK  {len(records)} records, stacks aligned, no cross-episode frames, "
          f"dtype=uint8")


# ======================================================================
# 5. 帧跳过对齐：动作真被重复 k 帧，奖励全额累加
# ======================================================================
def test_frame_skip_alignment():
    """对应旧管线最严重的缺陷：`agent.step += 1` 夹在两个 `% k` 判断之间
    (continue_training.py:758/773/776)，动作在 step≡0 时选、经验在 step≡3 时存，
    两者永不同时成立 —— 存进去的动作标签是"选择的动作"，而那一帧实际执行的
    是硬编码的不跳 [1,0]，同时 75% 的奖励与 terminal 被丢弃。
    """
    from train_flappy_dqn import CONFIG, skip_step

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
    """一行断言，当初就能抓到 deep_Q_dueling_DQN.py:134 的 Dropout(0.3) ——
    它让"贪婪"动作变成随机，同时让目标网络的 BatchNorm 用 minibatch 统计量，
    使 TD 目标随批次组成而变，Bellman 算子失去不动点。
    """
    import torch

    from train_flappy_dqn import DuelingDQN

    net = DuelingDQN()
    x = torch.randint(0, 2, (5, 4, 80, 80), dtype=torch.uint8) * 255

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
        small = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
        return cv2.threshold(small, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    random.seed(7)
    mismatched = total = 0
    for i in range(200):
        if env.done:
            env.reset()
        new = env.step(1 if random.random() < 0.2 else 0, render=True)[0]
        old = legacy()                       # raw_obs 读的是同一块已绘制的 SCREEN
        assert new.shape == (80, 80) and new.dtype == np.uint8
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

    不能直接改 wrapped_flappy_bird_fast.py 的 PIPEGAPSIZE —— 那个模块被 4 个
    旧脚本共用；而且 _base.getRandomPipe() 把 PIPEGAPSIZE 写死在返回值里，
    根本改不了。所以 FlappyEnv 用自己的 _random_pipe()。
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
    ]
    print(f"running {len(tests)} env tests...")
    for fn in tests:
        fn()
    print("all env tests passed")


if __name__ == "__main__":
    _run_all()
