"""超参数 —— 全项目唯一的数据源。

命名约定：所有计数量的单位都写在名字里（``*_decisions`` / ``*_grad_steps`` /
``*_episodes``）。旧管线最难查的一类 bug 就是"帧 / 决策步 / 局数"三个计数器
被混用 —— 衰减挂在决策步上、下限却挂在局数上，两条曲线永远对不上。
名字里带单位之后，这类错误一眼可见。
"""

CONFIG = dict(
    # ---- 环境 ----
    frame_skip=4,
    frame_stack=4,

    # 观测尺寸。各向异性：竖直方向（屏幕高 512）给更高分辨率，
    # 因为任务的精度需求几乎全在竖直方向。
    # 观测数组的形状是 (obs_w, obs_h) —— pygame surfarray 是 width-major，
    # 详见 game/flappy_env.py 里 OBS_W/OBS_H 的说明。
    obs_w=80,                        # 屏幕宽 288 -> 80，压 3.6 倍
    obs_h=128,                       # 屏幕高 512 -> 128，压 4.0 倍（原为 80，压 6.4 倍）

    # 全连接层宽度。实测训练好的网络里，512 维的 fc 激活有 14-25% 的单元
    # 从不激活，95% 的方差只活在约 100 维子空间里（99% 也只要约 210 维）。
    # 所以 256 是有实测依据的取值，不是拍脑袋。见 docs/learn/12-network-sizing.md
    fc_hidden=256,
    pipe_reward=1.0,
    death_reward=-1.0,
    alive_reward=0.0,
    shaping_coef=0.05,

    # 难度：管道上下间隙（像素）。小鸟高 24px。
    #   150 = 本仓库此前放宽过的值（第一版训练用的）
    #   100 = 原版 Flappy Bird 的难度
    # 间隙越小，可通行的竖直窗口越窄，容错越低。
    # randomize_pipes=True 时这个值只是回退默认，实际缝隙从 pipe_gap_range 采样。
    pipe_gap=100,

    # ---- 域随机化 ----
    # 关掉的话，缝隙中心只有 8 个离散取值、跨度 70px（可用空间的 17%），
    # 缝隙大小固定 100，水平间距固定 144 —— 网络只需应付一个很窄的分布，
    # 学到的是窄策略而不是"会玩这个游戏"。打开后三者逐根独立采样，
    # 同一局之内地图就在不断变化。
    randomize_pipes=True,
    # 下界 85 是实验定的，不是拍的。
    #
    # 小鸟高 24px，而它能维持的最小竖直振荡是 22.5px（由扇翅冲量 -5、
    # 重力 0.5/帧、frame_skip=4 共同决定，见 docs/learn/12）。所以缝隙的
    # "真实容错" = gap - 24 - 22.5：
    #     gap  60px -> 13.5px      gap  85px -> 38.5px
    #     gap  70px -> 23.5px      gap 100px -> 53.5px
    #
    # 死亡归因（200 局实测）：70-80px 只占 10.5% 的管道，却贡献 53.5% 的
    # 死亡（风险是基准的 5.08 倍）；80-95px 占 15.8%、贡献 25.5%。
    # 而 gap 60-80 上四种网络配置全部只有 2.3-3.3 根 —— 那是物理墙，
    # 不是学习问题。
    #
    # 对照实验（其余参数完全相同，按相同决策步对齐的贪婪评测）：
    #     决策步      gap 70-165    gap 85-165
    #      60,000        15.4          26.9
    #     120,000        17.9          58.5
    # 把不可通过区移出训练分布之后，同等训练量下高 1.7-3.3 倍。
    # 这不是算法改进，是承认那一段不该在考纲里。
    pipe_gap_range=(85.0, 165.0),      # 缝隙大小
    pipe_spacing_range=(115.0, 200.0),  # 相邻管道水平间距（原来恒为 144）
    pipe_edge_margin=38.0,              # 缝隙上下沿距天花板/地面的最小余量
    pipe_max_delta_frac=0.6,            # 相邻缝隙中心落差上限 / 水平间距，
                                        # 保证物理可达（见 flappy_env._sample_pipe）

    # ---- 学习 ----
    gamma=0.99,                      # 决策级折扣
    batch_size=128,                  # RTX 3060 实测：batch 128 = 4.67ms/步，
                                     # batch 32 = 5.04ms/步 —— 小网络下 GPU 被
                                     # kernel 启动延迟主导，梯度步速率恒为
                                     # ~200 步/秒与 batch 无关。所以 batch 白拿。
    lr=1.5e-4,                       # batch 32->128 后按 sqrt 缩放
    adam_eps=1.5e-4,                 # Rainbow 的取值；1e-8 会让 Adam 退化成
                                     # lr*sign(g)，分不清"+1 管道"和数值噪声
    grad_clip=10.0,                  # Huber 已限幅；1.0 会让几乎每批都被重缩放

    # ---- 节奏 ----
    warmup_decisions=20_000,
    train_every_decisions=4,         # 配 batch 128 = 每次决策 32 个样本的学习量
                                     # （标准回放比），同时把梯度步需求压到
                                     # GPU 的 200 步/秒上限之内
    target_sync_grad_steps=1_000,

    # n-step 回报。**实测在本任务上 n=3 反而学不起来，所以默认 1。**
    #
    # 对照实验（gap 85-165，其余参数完全相同，按相同决策步对齐）：
    #     决策步     n=1      n=3
    #      60,000    26.9      5.0
    #     120,000    58.5      4.6
    #     200,000    35.3      5.3
    # n=3 全程在 3-6.5 横盘，训练侧 loss 高 10 倍、|TD| 高 2.4 倍、
    # Q 值只有 n=1 的三分之一（目标又吵又悲观，但不发散）。
    #
    # 实现本身是对的 —— 在真实轨迹上端到端验证过：n 步回报、done 标记、
    # 产出条数、s_{t+n} 重建全部零误差（单测 11 + scratchpad 里的暴力对照）。
    # 失败原因没有完全查清，三个候选未区分开：
    #   (a) off-policy 偏差：中间动作来自行为策略，退火期 eps 很高时
    #       随机动作导致的死亡会被摊到前 n 个状态上
    #   (b) gamma^n 让自举项权重下降，本就稀疏的价值传播被进一步削弱
    #   (c) n 步累加放大了奖励方差，而 Huber 拐点是按单步奖励量级定的
    # 代码和单测保留，想复现或深究把这里改回 3 即可。
    n_step=1,

    # ---- 经验回放 ----
    # 观测从 80x80 变成 80x128 之后，每条经验从 32KB 涨到 51.2KB（1.6 倍）。
    # 原来的 400_000 在新尺寸下要 20.5GB —— 那个默认值从来没被实际跑过，
    # 所有训练都是用 --buffer 覆盖的。这里改成一个真能跑起来的值：
    #   150k * 5 * 80 * 128 = 7.7GB
    # 16GB 的机器用 --buffer 60000（3.1GB），train.py 启动时会打印实际占用。
    buffer_capacity=150_000,

    # ---- 探索 ----
    eps_start=1.0,
    eps_mid=0.05,
    eps_final=0.01,
    eps_anneal1_decisions=250_000,
    eps_anneal2_decisions=500_000,
    eps_random_flap_prob=0.2,        # 见 rollout.sample_random_action 的说明

    # 回合长度上限。一旦策略学好，回合可以无限长（实测 checkpoint 能连过 400+
    # 根管道不死），不设上限会让 episode 级的日志/存档/评测全部卡住。
    # 注意：截断 **不是** 终止 —— 截断处 done 保持 False，价值仍然自举，
    # 否则等于告诉网络"飞太久会凭空死掉"。
    max_episode_decisions=4_000,

    # ---- 记录 ----
    max_episodes=200_000,
    seed=0,
    eval_every_episodes=500,         # 长跑需要更密的评测曲线
    eval_episodes=20,                # 回合变长后评测本身也耗时
    eval_epsilon=0.0,                # 必须是 0。实测：同一个模型 100 局，
                                     # eps=0 得 389.6 根，eps=0.01 只得 63.1 根
                                     # —— 6.2 倍差距。学好之后一局有约 2900 次
                                     # 决策，1% 随机率 = 每局约 29 次随机动作，
                                     # 在 100px 的缝隙里一次错误扇翅就是死。
                                     # 策略越强，这个惩罚越重，评测越失真。
                                     # 防"确定性死循环"由 max_episode_decisions
                                     # 的上限负责，不需要靠注入随机动作。
    eval_seed_base=20260906,         # 固定评测集的起点：第 i 局用 seed_base+i
                                     # 播种。**必须逐局播种**，只在评测开始时
                                     # 播一次是不够的 —— 管道生成走的是全局
                                     # random 模块，两个模型只要存活局长不同
                                     # （几乎总是不同），从第 2 局起消耗的随机
                                     # 数数量就不同，管道序列会错位，"固定种子"
                                     # 只保证了第 1 局可比。这个 bug 让此前所有
                                     # 跨存档比较都不可信，见 docs/IMPROVEMENT_PLAN.md D0。
                                     # 代价：绝对分数是这一组固定关卡上的分数，
                                     # 不等于整个分布上的期望；但跨模型比较变成
                                     # 同一批关卡上的**配对比较**，灵敏度高得多。
    resume_every_minutes=30,
    log_train_every_grad_steps=100,
)

# --smoke：几分钟就能跑完的缩水配置，用来验证管线通不通，不用来看成绩
SMOKE_OVERRIDES = dict(
    warmup_decisions=5_000,
    buffer_capacity=50_000,
    target_sync_grad_steps=300,
    eps_anneal1_decisions=40_000,
    eps_anneal2_decisions=40_000,
    max_episodes=20_000,
    eval_every_episodes=2_000,
)


def resolve_config(smoke=False, **overrides):
    """CONFIG 的副本，依次叠加 --smoke 预设和显式覆盖项。

    值为 None 的覆盖项被忽略，这样命令行参数可以直接透传（未给出的
    argparse 参数就是 None），调用方不必再逐个判空。
    """
    cfg = dict(CONFIG)
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    for key, value in overrides.items():
        if value is not None:
            if key not in cfg:
                raise KeyError("unknown config key: %s" % key)
            cfg[key] = value
    return cfg


def config_from_checkpoint(ckpt, **overrides):
    """存档里记录的配置，缺项用当前 CONFIG 补齐。

    存档必须自带配置，否则用今天的默认值去评测一个昨天的模型 —— 尤其是
    pipe_gap 这种难度旋钮 —— 得到的分数没有可比性。
    """
    saved = ckpt.get('config', {})
    cfg = dict(CONFIG)

    # 域随机化是后加的。存档里没有这个键，说明它是在**未随机化**的环境里
    # 训练的；此时若沿用今天的默认值 True，就等于拿一个更难的分布去考它，
    # 分数会莫名其妙地低。缺项一律按"当时的行为"补，不按"今天的默认"补。
    if 'randomize_pipes' not in saved:
        cfg['randomize_pipes'] = False

    # 各向异性观测和可配置的 fc 宽度也是后加的。老存档记的是正方形的
    # obs_size 和写死的 fc=512 —— 用今天的默认值去建网络会直接形状不匹配。
    # 这类兼容处理必须放在这里，否则每个加载点都要重写一遍。
    if 'obs_size' in saved and 'obs_w' not in saved:
        cfg['obs_w'] = cfg['obs_h'] = saved['obs_size']
    if 'fc_hidden' not in saved:
        cfg['fc_hidden'] = 512

    cfg.update(saved)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg
