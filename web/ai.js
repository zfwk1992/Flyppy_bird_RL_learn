/**
 * AI 玩家：把观测管线、网络、frame_skip 决策节奏包成一个能驱动 FlappyGame 的对象。
 *
 * 决策节奏与 `flappy/rollout.py: skip_step` 一致 —— 每 frame_skip 帧决策一次，
 * 窗口内重复同一个动作。**不要**在这里加"反应延迟"或降低决策频率来放水：
 * 这个 demo 的卖点就是它比人强得多，削弱它等于把 demo 本身删掉。
 */
import { loadWeights, DuelingDQN } from './nn.js';
import { renderRed, downsample, FrameStack } from './obs.js';
import { OBS_W, OBS_H } from './game.js';

export const FRAME_SKIP = 4;    // flappy/config.py: CONFIG['frame_skip']
export const FRAME_STACK = 4;   // flappy/config.py: CONFIG['frame_stack']

export class AiPlayer {
  constructor(weights) {
    this.net = new DuelingDQN(weights);
    this.stack = new FrameStack(FRAME_STACK);
    this.red = new Uint8Array(288 * 512);
    this.obs = new Float32Array(OBS_W * OBS_H);
    this.lastMs = 0;
    this.reset();
  }

  /** 新回合：帧计数归零，下一次决策会用首帧填满帧栈。 */
  reset() {
    this.frame = 0;
    this.action = 0;
    this.first = true;
  }

  /**
   * 返回这一帧要执行的动作。**必须在 game.step() 之前调用** ——
   * 观测的是"当前这一帧的画面"，和训练时的时序一致。
   */
  decide(game) {
    if (this.frame % FRAME_SKIP === 0) {
      renderRed(game, this.red);
      downsample(this.red, this.obs);
      const arr = this.first ? this.stack.reset(this.obs) : this.stack.push(this.obs);
      this.first = false;
      const t0 = performance.now();
      this.action = this.net.act(arr);
      this.lastMs = performance.now() - t0;
    }
    this.frame++;
    return this.action;
  }

  /** 最近一次决策的 Q 值，用来在界面上显示"它在想什么"。 */
  get q() { return this.net.q; }
}

/** 加载权重并造一个 AiPlayer。`base` 是 model/ 目录的相对 URL。 */
export async function createAi(base = './model/') {
  return new AiPlayer(await loadWeights(base));
}
