"""可视化入口：看训练好的 AI 玩，或者自己上手玩。

    python play.py runs/xxx/best.pt                    # 30 FPS 实时观看
    python play.py runs/xxx/best.pt --fps 60 --scale 2 # 加速 / 放大
    python play.py runs/xxx/best.pt --record demo.mp4  # 同时录像
    python play.py runs/xxx/best.pt --record demo.mp4 --no-window  # 只录像
    python play.py --human                             # 自己玩（不需要存档）

窗口内按键：
    q / ESC   退出
    空格      AI 模式下暂停；--human 模式下扇翅
    r         立刻重开一局
    + / -     加速 / 减速

为什么用 cv2 显示而不是 pygame 显示
------------------------------------
这不是偏好，是硬约束：

  1. game/flappy_bird_utils.py 用 .convert() / .convert_alpha() 加载所有精灵，
     这些 Surface 是**绑定当前显示格式**的。
  2. game/resources.py 在 **导入时** 无条件执行
     os.environ['SDL_VIDEODRIVER'] = 'dummy'，外部无法抢先覆盖。
  3. 实测确认：pygame.display.quit() 之后原 SCREEN 直接作废，
     重建显示会让所有已 convert 的精灵失效。

而 cv2 的 GUI 后端直接显示 numpy 帧，完全绕开 SDL，对环境零改动、零风险。
VideoWriter 也复用同一批帧。

推理速度不是问题：GPU 单样本前向 0.621ms = 1611 次/秒，而 30 FPS 实时游玩
只需要 7.5 次/秒 —— 快了 200 倍。所以这里没有任何针对推理的优化，只有节流。
"""

import argparse
import time

import cv2
import numpy as np
import torch

from flappy import checkpoint
from flappy.config import resolve_config
from flappy.rollout import FrameStack, make_env, sample_random_action

WINDOW = "Flappy Bird - DQN agent"

# BGR
C_TEXT = (255, 255, 255)
C_DIM = (170, 170, 170)
C_PICK = (80, 255, 80)
C_BAR_BG = (60, 60, 60)
C_BAR_FG = (90, 200, 255)

KEY_QUIT = (ord('q'), 27)
KEY_FLAP = (ord(' '), 32, 82, 0)     # 空格 / 上方向键（不同后端码值不同）


def to_bgr(raw):
    """环境的原始观测是 (288,512,3) width-major（转置的）RGB。

    转成 cv2 惯用的 (512,288,3) height-major BGR。
    """
    return np.transpose(raw, (1, 0, 2))[:, :, ::-1]


def draw_hud(img, pipes, n_dec, q, action, fps, eps, paused, human):
    """把分数、Q 值、动作画到画面上。

    用 cv2.putText 而不是往 SCREEN 上 blit —— 后者会污染下一帧的观测
    （网络就看见分数数字了）。HUD 只能画在拷贝出来的图上。
    """
    h, w = img.shape[:2]
    pad = 8

    # 顶部半透明条
    bar = img[0:76, :].copy()
    img[0:76, :] = cv2.addWeighted(bar, 0.35, np.zeros_like(bar), 0.65, 0)

    cv2.putText(img, "PIPES %d" % pipes, (pad, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, "step %d" % n_dec, (pad, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "%.0f fps%s" % (fps, "  PAUSED" if paused else ""),
                (pad, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)

    if human:
        # 一次扇翅令 velY=-5，约 19 帧后净位移归零 —— 所以是"点"不是"按住"
        cv2.putText(img, "HUMAN - TAP SPACE", (w - 190, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_PICK, 1, cv2.LINE_AA)
    elif q is not None:
        labels = ("STAY", "FLAP")
        for i in range(2):
            picked = (i == action)
            col = C_PICK if picked else C_DIM
            cv2.putText(img, "%s %+.2f" % (labels[i], q[i]),
                        (w - 132, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 2 if picked else 1, cv2.LINE_AA)
        # Q 差值条：越长表示网络对该动作越有把握
        gap = float(q[1] - q[0])
        cx, cy, half = w - 70, 66, 55
        cv2.rectangle(img, (cx - half, cy - 4), (cx + half, cy + 4), C_BAR_BG, -1)
        end = int(np.clip(gap / 3.0, -1, 1) * half)
        x0, x1 = (cx, cx + end) if end >= 0 else (cx + end, cx)
        cv2.rectangle(img, (x0, cy - 4), (x1, cy + 4), C_BAR_FG, -1)
        cv2.line(img, (cx, cy - 7), (cx, cy + 7), C_TEXT, 1)

    if eps > 0:
        cv2.putText(img, "eps %.3f" % eps, (pad, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_DIM, 1, cv2.LINE_AA)
    return img


def wait_for_start(env, args):
    """冻结画面等玩家按空格。返回 False 表示玩家要退出。

    顺带解决了"窗口没焦点 -> 按键收不到 -> 小鸟自己摔死"的问题：
    在按到键之前物理根本不推进，所以焦点什么时候拿到都不影响。
    """
    blink = 0
    while True:
        img = to_bgr(env.raw_obs()).copy()
        if args.scale != 1:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h // 2 - 46), (w, h // 2 + 34), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        if (blink // 12) % 2 == 0:                  # 闪烁，提示这里在等输入
            cv2.putText(img, "PRESS SPACE", (int(w * 0.14), h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * args.scale / 2.0 + 0.35,
                        C_PICK, 2, cv2.LINE_AA)
        cv2.putText(img, "click the window first if keys do nothing",
                    (int(w * 0.06), h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, C_DIM, 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(20) & 0xFF
        blink += 1
        if key in KEY_QUIT:
            return False
        if key != 255:                              # 任意键开始（空格最自然）
            return True


@torch.no_grad()
def play(args):
    human = args.human
    if human:
        net, device = None, torch.device('cpu')
        cfg = resolve_config(pipe_gap=args.pipe_gap,
                             randomize_pipes=args.randomize)
        print("human mode: hold SPACE to flap, r to restart, q to quit")
    else:
        device = torch.device(args.device)
        net, cfg, ckpt = checkpoint.load_for_inference(
            args.checkpoint, device, pipe_gap=args.pipe_gap,
            randomize_pipes=args.randomize)
        print(checkpoint.describe(ckpt, cfg, args.checkpoint))
        print("  device: %s" % device)

    # 把环境设置打出来 —— 光看画面很难确认随机化到底开没开
    if cfg['randomize_pipes']:
        # 存档里的范围经过 JSON/torch.save 往返会变成 list，先转回 tuple
        print("  environment: RANDOMISED per pipe")
        print("    gap size    %.0f - %.0f px" % tuple(cfg['pipe_gap_range']))
        print("    spacing     %.0f - %.0f px" % tuple(cfg['pipe_spacing_range']))
        print("    gap centre  free to move +-%.0f%% of the spacing between pipes"
              % (100 * cfg['pipe_max_delta_frac']))
    else:
        print("  environment: FIXED (gap %d px, spacing 144 px, 8 heights)"
              % cfg['pipe_gap'])
    print()

    # 人玩时一次决策 = 一帧，否则 4 帧一动的操作手感完全没法玩
    k = 1 if human else cfg['frame_skip']
    env = make_env(cfg)

    writer = None
    if args.record:
        h, w = 512 * args.scale, 288 * args.scale
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (w, h))
        if not writer.isOpened():
            raise SystemExit("could not open %s for writing" % args.record)
        print("recording to %s (%dx%d @ %d fps)" % (args.record, w, h, args.fps))

    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("keys: q/ESC quit | SPACE %s | r restart | +/- speed"
              % ("flap" if human else "pause"))

    stacker = FrameStack(cfg['frame_stack'])
    results = []
    fps_target = float(args.fps)
    paused = False
    quit_all = False
    flap_held = False
    t_fps, n_fps, fps_shown = time.time(), 0, fps_target

    for ep in range(args.episodes):
        if quit_all:
            break
        stack = stacker.reset(env.reset())
        n_dec, q_np, action = 0, None, 0
        info = {'score': 0}                 # 暂停时第一帧就要用到

        # 人玩时先等一次按键再开始物理。这不只是还原原版 Flappy Bird 的手感 ——
        # cv2 的窗口只有拿到系统焦点之后才收得到键盘事件，而窗口刚弹出来时
        # 往往还没被激活。没有这道闸门的话，小鸟会在玩家反应过来之前
        # 自由落体 36 帧摔死，而且连续好几局都这样（实测 20 局里有 18 局如此）。
        if human and show:
            if not wait_for_start(env, args):
                break

        t_next = time.time()

        while True:
            if not paused:
                if human:
                    action = 1 if flap_held else 0
                    flap_held = False
                else:
                    q = net(torch.from_numpy(stack).unsqueeze(0).to(device))[0]
                    q_np = q.cpu().numpy()
                    action = int(q.argmax().item())
                    if args.epsilon > 0 and np.random.random() < args.epsilon:
                        action = sample_random_action(cfg)

                # 训练时窗口内只渲染最后一帧（为了速度）；这里给人看，
                # k 帧全渲染全显示 —— 否则画面是 4 倍快进且发跳。
                # 渲染开销 ~1ms/帧，在 30fps 下完全无所谓。
                # 动作在整个窗口内保持不变，与训练时的语义完全一致。
                done = False
                obs_last = None
                pending = []
                for _ in range(k):
                    obs_last, _, done, info = env.step(action, render=True)
                    pending.append(env.raw_obs())
                    if done:
                        break
                n_dec += 1
                # 只有窗口最后那一帧进入帧栈（与训练一致）
                if not done:
                    stack = stacker.push(obs_last)
            else:
                pending = [env.raw_obs()]
                done = False

            restart = False
            for rgb in pending:
                img = to_bgr(rgb).copy()
                if args.scale != 1:
                    img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                     interpolation=cv2.INTER_NEAREST)
                n_fps += 1
                if time.time() - t_fps >= 0.5:
                    fps_shown = n_fps / (time.time() - t_fps)
                    t_fps, n_fps = time.time(), 0
                img = draw_hud(img, info['score'], n_dec, q_np, action,
                               fps_shown, args.epsilon, paused, human)

                if writer is not None and not paused:
                    writer.write(img)

                if show:
                    cv2.imshow(WINDOW, img)
                    # 节流到目标 FPS：推理只要 0.6ms，不节流会快到看不清
                    t_next += 1.0 / fps_target
                    wait_ms = max(1, int((t_next - time.time()) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in KEY_QUIT:
                        quit_all = True
                        break
                    elif key == ord('r'):
                        restart = True
                        break
                    elif key in (ord('+'), ord('=')):
                        fps_target = min(fps_target * 1.5, 2000)
                    elif key == ord('-'):
                        fps_target = max(fps_target / 1.5, 2)
                    elif key == ord(' '):
                        if human:
                            flap_held = True
                        else:
                            paused = not paused
                            t_next = time.time()
            if quit_all or restart:
                break

            if done or n_dec >= args.max_decisions:
                results.append((info['score'], n_dec, done))
                print("episode %d: %d pipes in %d decisions  (%s)"
                      % (ep + 1, info['score'], n_dec,
                         "died" if done else "hit the %d-decision cap, still alive"
                         % args.max_decisions))
                break

    if writer is not None:
        writer.release()
        print("saved %s" % args.record)
    if show:
        cv2.destroyAllWindows()

    if results:
        pipes = [r[0] for r in results]
        died = sum(1 for r in results if r[2])
        print()
        print("played %d episode(s): pipes mean %.1f, median %.1f, max %d"
              % (len(results), np.mean(pipes), np.median(pipes), max(pipes)))
        print("  died %d / %d  (the rest hit the %d-decision cap while still alive)"
              % (died, len(results), args.max_decisions))


def main():
    p = argparse.ArgumentParser(
        description="Watch a trained DQN agent play Flappy Bird (or play it yourself)")
    p.add_argument('checkpoint', nargs='?', default=None)
    p.add_argument('--human', action='store_true',
                   help='play it yourself instead of loading a checkpoint')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--scale', type=int, default=1, help='integer upscale of the window')
    p.add_argument('--max-decisions', type=int, default=4000)
    p.add_argument('--epsilon', type=float, default=0.0,
                   help='inject random actions (0 = pure greedy)')
    p.add_argument('--record', type=str, default=None, help='write an mp4 here')
    p.add_argument('--no-window', action='store_true', help='record only, no window')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='override difficulty (px); 150 easy, 100 original, lower harder. '
                        'Only used when randomisation is off.')
    p.add_argument('--no-randomize', dest='randomize', action='store_false',
                   default=None,
                   help='use the old fixed layout (gap 100, spacing 144, 8 heights)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if not args.human and args.checkpoint is None:
        p.error("give a checkpoint path, or --human to play it yourself")
    if args.human:
        # 人玩时每帧一次决策，30fps 的默认节流刚好是原版游戏的手感
        args.max_decisions = max(args.max_decisions, 20000)
    play(args)


if __name__ == '__main__':
    main()
