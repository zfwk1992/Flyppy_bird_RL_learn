"""
看训练好的 AI 自己玩 Flappy Bird —— 实时窗口。

用法
----
    python play_flappy.py runs/gpu_final/best.pt              # 30 FPS 实时观看
    python play_flappy.py <ckpt> --fps 60 --scale 2           # 加速 / 放大
    python play_flappy.py <ckpt> --record demo.mp4            # 同时录像
    python play_flappy.py <ckpt> --episodes 3 --max-decisions 4000
    python play_flappy.py <ckpt> --record demo.mp4 --no-window # 只录像不开窗

窗口内按键：
    q / ESC   退出
    空格      暂停 / 继续
    r         立刻重开一局
    + / -     加速 / 减速

为什么用 cv2 显示而不是 pygame 显示
------------------------------------
这不是偏好，是硬约束：

  1. game/flappy_bird_utils.py:28-69 用 .convert() / .convert_alpha() 加载所有
     精灵，这些 Surface 是**绑定当前显示格式**的。
  2. game/wrapped_flappy_bird_fast.py:12 在 **导入时** 无条件执行
     os.environ['SDL_VIDEODRIVER'] = 'dummy'，外部无法抢先覆盖。
  3. 实测确认：pygame.display.quit() 之后原 SCREEN 直接作废，重建显示会让
     所有已 convert 的精灵失效。

而 cv2 的 GUI 后端是 WIN32UI（已验证可用），直接显示 numpy 帧，
完全绕开 SDL，对环境零改动、零风险。VideoWriter 也复用同一批帧。

注意推理速度根本不是问题：GPU 单样本前向 0.621ms = 1611 次/秒，
而 30 FPS 实时游玩只需要 7.5 次/秒 —— 快了 200 倍。所以这里没有任何
针对推理的优化，只有节流（不然画面会快到看不清）。
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from game.flappy_env import FlappyEnv
from train_flappy_dqn import CONFIG, DuelingDQN, sample_random_action

WINDOW = "Flappy Bird - DQN agent"

# BGR
C_TEXT = (255, 255, 255)
C_DIM = (170, 170, 170)
C_PICK = (80, 255, 80)
C_BAR_BG = (60, 60, 60)
C_BAR_FG = (90, 200, 255)


def to_bgr(raw):
    """环境的原始观测是 (288,512,3) width-major（转置的）RGB。

    转成 cv2 惯用的 (512,288,3) height-major BGR。
    """
    return np.transpose(raw, (1, 0, 2))[:, :, ::-1]


def draw_hud(img, pipes, n_dec, q, action, fps, eps, paused):
    """把分数、Q 值、动作画到画面上。

    用 cv2.putText 而不是环境里的 showScore() —— 后者会往 SCREEN 上 blit，
    会污染下一帧的观测（网络就看见分数数字了）。HUD 只能画在拷贝出来的图上。
    """
    h, w = img.shape[:2]
    pad = 8

    # 顶部半透明条
    bar = img[0:76, :].copy()
    img[0:76, :] = cv2.addWeighted(bar, 0.35,
                                   np.zeros_like(bar), 0.65, 0)

    cv2.putText(img, "PIPES %d" % pipes, (pad, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, "step %d" % n_dec, (pad, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "%.0f fps%s" % (fps, "  PAUSED" if paused else ""),
                (pad, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)

    if q is not None:
        labels = ("STAY", "FLAP")
        for i in range(2):
            picked = (i == action)
            col = C_PICK if picked else C_DIM
            y = 26 + i * 22
            cv2.putText(img, "%s %+.2f" % (labels[i], q[i]),
                        (w - 132, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col,
                        2 if picked else 1, cv2.LINE_AA)
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


@torch.no_grad()
def play(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(CONFIG)
    cfg.update(ckpt.get('config', {}))
    if args.pipe_gap is not None:
        cfg['pipe_gap'] = args.pipe_gap
    K = cfg['frame_skip']
    S = cfg['frame_stack']

    net = DuelingDQN().to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    # 确定性自检（旧管线的 Dropout bug 就是栽在这里）
    dummy = torch.zeros(1, S, cfg['obs_size'], cfg['obs_size'],
                        dtype=torch.uint8, device=device)
    assert torch.allclose(net(dummy), net(dummy)), "network is stochastic at act time"

    print("checkpoint : %s" % args.checkpoint)
    print("  trained to episode %s, decision_step %s"
          % (ckpt.get('episode'), ckpt.get('decision_step')))
    if ckpt.get('recent100_pipes') is not None:
        print("  recent100_pipes at save time: %.2f" % ckpt['recent100_pipes'])
    print("  device: %s | frame_skip=%d frame_stack=%d | pipe_gap=%d"
          % (device, K, S, cfg['pipe_gap']))
    print()

    env = FlappyEnv(pipe_reward=cfg['pipe_reward'],
                    death_reward=cfg['death_reward'],
                    alive_reward=cfg['alive_reward'],
                    pipe_gap=cfg['pipe_gap'])

    writer = None
    if args.record:
        h, w = 512 * args.scale, 288 * args.scale
        writer = cv2.VideoWriter(args.record,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (w, h))
        if not writer.isOpened():
            raise SystemExit("could not open %s for writing" % args.record)
        print("recording to %s (%dx%d @ %d fps)" % (args.record, w, h, args.fps))

    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("keys: q/ESC quit | SPACE pause | r restart | +/- speed")

    results = []
    fps_target = float(args.fps)
    paused = False
    quit_all = False
    t_fps, n_fps, fps_shown = time.time(), 0, fps_target

    for ep in range(args.episodes):
        if quit_all:
            break
        frame = env.reset()
        stack = np.repeat(frame[None], S, axis=0)
        n_dec, q_np, action = 0, None, 0
        info = {'score': 0}                 # 暂停时第一帧就要用到
        t_next = time.time()

        while True:
            if not paused:
                t = torch.from_numpy(stack).unsqueeze(0).to(device)
                q = net(t)[0]
                q_np = q.cpu().numpy()
                action = int(q.argmax().item())
                if args.epsilon > 0 and np.random.random() < args.epsilon:
                    action = sample_random_action(cfg)

                # 训练时窗口内只渲染最后一帧（为了速度）；这里给人看，
                # 4 帧全渲染全显示 —— 否则画面是 4 倍快进且发跳。
                # 渲染开销 ~1ms/帧，在 30fps 下完全无所谓。
                # 动作在整个窗口内保持不变，与训练时的语义完全一致。
                done = False
                obs_last = None
                pending = []
                for i in range(K):
                    obs_last, _, done, info = env.step(action, render=True)
                    pending.append(env.raw_obs())
                    if done:
                        break
                n_dec += 1
                # 只有窗口最后那一帧进入帧栈（与训练一致）
                if not done:
                    stack = np.concatenate([obs_last[None], stack[:S - 1]], axis=0)
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
                               fps_shown, args.epsilon, paused)

                if writer is not None and not paused:
                    writer.write(img)

                if show:
                    cv2.imshow(WINDOW, img)
                    # 节流到目标 FPS：推理只要 0.6ms，不节流会快到看不清
                    t_next += 1.0 / fps_target
                    wait_ms = max(1, int((t_next - time.time()) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord('q'), 27):
                        quit_all = True
                        break
                    elif key == ord(' '):
                        paused = not paused
                        t_next = time.time()
                    elif key == ord('r'):
                        restart = True
                        break
                    elif key in (ord('+'), ord('=')):
                        fps_target = min(fps_target * 1.5, 2000)
                    elif key == ord('-'):
                        fps_target = max(fps_target / 1.5, 2)
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
    p = argparse.ArgumentParser(description="Watch a trained DQN agent play Flappy Bird")
    p.add_argument('checkpoint')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--scale', type=int, default=1, help='integer upscale of the window')
    p.add_argument('--max-decisions', type=int, default=4000)
    p.add_argument('--epsilon', type=float, default=0.0,
                   help='inject random actions (0 = pure greedy)')
    p.add_argument('--record', type=str, default=None, help='write an mp4 here')
    p.add_argument('--no-window', action='store_true', help='record only, no window')
    p.add_argument('--pipe-gap', type=int, default=None,
                   help='override difficulty (px); 150 easy, 100 original, lower harder')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit("checkpoint not found: %s" % args.checkpoint)
    play(args)


if __name__ == '__main__':
    main()
