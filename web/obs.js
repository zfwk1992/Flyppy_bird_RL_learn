/**
 * 观测管线：游戏状态 → 网络输入。
 *
 * 复刻的是 `game/flappy_env.py` 的两步：
 *   `_draw()`    把背景/管道/地面/小鸟按序 blit 到 288x512 的 Surface
 *   `_observe()` 取 **R 通道**，INTER_AREA 缩到 80x128，按阈值 1 二值化
 *
 * 三个容易踩的点，都在下面的实现里逐条对上了：
 *
 * 1. **只有 R 通道**。`pixels3d(SCREEN)[:, :, 0]`。已验证这五张精灵上
 *    "R>1"、"任意通道>1"、"alpha>0" 三者逐像素相同（见 export_obs_sprites.py
 *    里的断言），所以取 R 不丢信息，但**数值**必须是真的 R 值 ——
 *    INTER_AREA 是加权平均，不是覆盖判定，暗像素在格子边缘会被平均掉。
 *
 * 2. **数组是转置的**。pygame.surfarray 是 width-major：
 *    源数组 (288, 512) = (屏幕宽, 屏幕高)，输出 (80, 128) = (OBS_W, OBS_H)。
 *    所以本文件里一律用 `x * SCREEN_HEIGHT + y` 索引，和 Python 一致。
 *
 * 3. **blit 坐标向零截断**。pygame 用 Rect 承接目标坐标，浮点会被截断
 *    （`Math.trunc`，**不是** `Math.floor` —— 上管道的 y 是负数，两者不同）。
 *
 * 为什么不读画布见 web/tools/export_obs_sprites.py 的开头。
 */
import { OBS_SPRITES } from './assets/obs-sprites.js';
import {
  SCREEN_WIDTH, SCREEN_HEIGHT, BASE_Y, PIPE_WIDTH, PIPE_HEIGHT,
  OBS_W, OBS_H,
} from './game.js';

// ---------------------------------------------------------------------
// 精灵：上管道是 pipe-green 旋转 180°，和 flappy_bird_utils.py 一样只做一次
// ---------------------------------------------------------------------
function rotate180(sp) {
  const { w, h, data } = sp;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = data[w * h - 1 - i];
  return { w, h, data: out };
}

const PIPE_LOWER = OBS_SPRITES.pipe;
const PIPE_UPPER = rotate180(OBS_SPRITES.pipe);
const BIRDS = [OBS_SPRITES.birdUp, OBS_SPRITES.birdMid, OBS_SPRITES.birdDown];
const BASE = OBS_SPRITES.base;
// SCREEN.blit(IMAGES['base'], (self.basex, BASEY))；BASEY=404.48 → 截断成 404
const BASE_Y_INT = Math.trunc(BASE_Y);

// ---------------------------------------------------------------------
// blit：透明像素（R===0）不覆盖已画的内容，与 pygame 的 alpha blit 一致。
// 小鸟只有 72% 不透明，写 0 会把身后的管道擦出一个鸟形的洞。
// ---------------------------------------------------------------------
function blit(dst, sp, dx, dy) {
  const { w, h, data } = sp;
  const x0 = Math.trunc(dx);
  const y0 = Math.trunc(dy);
  const sxLo = Math.max(0, -x0);
  const sxHi = Math.min(w, SCREEN_WIDTH - x0);
  const syLo = Math.max(0, -y0);
  const syHi = Math.min(h, SCREEN_HEIGHT - y0);
  for (let sx = sxLo; sx < sxHi; sx++) {
    const col = (x0 + sx) * SCREEN_HEIGHT + y0;
    for (let sy = syLo; sy < syHi; sy++) {
      const v = data[sy * w + sx];
      if (v !== 0) dst[col + sy] = v;
    }
  }
}

/**
 * 重建当前帧的 R 通道，(288, 512) width-major，等价于
 * `pygame.surfarray.pixels3d(SCREEN)[:, :, 0]`。
 *
 * blit 顺序照抄 `_draw()`：背景 → 管道（上、下成对）→ 地面 → 小鸟。
 * 背景是纯黑（R 恒 0），所以清零即可，不必真的画。
 */
export function renderRed(state, out) {
  const dst = out || new Uint8Array(SCREEN_WIDTH * SCREEN_HEIGHT);
  dst.fill(0);
  const { upperPipes, lowerPipes } = state;
  for (let i = 0; i < upperPipes.length; i++) {
    blit(dst, PIPE_UPPER, upperPipes[i].x, upperPipes[i].y);
    blit(dst, PIPE_LOWER, lowerPipes[i].x, lowerPipes[i].y);
  }
  blit(dst, BASE, state.basex, BASE_Y_INT);
  blit(dst, BIRDS[state.playerIndex], state.playerx, state.playery);
  return dst;
}

// ---------------------------------------------------------------------
// INTER_AREA：逐条对着 OpenCV resize.cpp 的 resizeArea 权重表写的。
// 不用"格子里有没有非黑像素"的覆盖判定近似 —— 缩放比 288/80 = 3.6 不是整数，
// 边缘列的权重最小只有 0.2/14.4 = 0.0139，乘上最暗的 R=83 得 1.15 < 1.5，
// 会被 round 成 1 而低于阈值。覆盖判定在这些列上会多点亮一格。
// ---------------------------------------------------------------------
function areaWeights(srcLen, dstLen) {
  const scale = srcLen / dstLen;
  const table = [];
  for (let d = 0; d < dstLen; d++) {
    const fs1 = d * scale;
    const fs2 = fs1 + scale;
    const cell = Math.min(scale, srcLen - fs1);
    let s1 = Math.ceil(fs1);
    let s2 = Math.floor(fs2);
    s2 = Math.min(s2, srcLen - 1);
    s1 = Math.min(s1, s2);
    const idx = [];
    const w = [];
    if (s1 - fs1 > 1e-3) { idx.push(s1 - 1); w.push((s1 - fs1) / cell); }
    for (let s = s1; s < s2; s++) { idx.push(s); w.push(1.0 / cell); }
    if (fs2 - s2 > 1e-3) { idx.push(s2); w.push((Math.min(fs2, srcLen) - s2) / cell); }
    table.push({ idx, w });
  }
  return table;
}

// 屏幕宽 288 → OBS_W 80（比 3.6，权重不齐整）；屏幕高 512 → OBS_H 128（比 4，齐整）
const WX = areaWeights(SCREEN_WIDTH, OBS_W);
const WY = areaWeights(SCREEN_HEIGHT, OBS_H);

/** cvRound：四舍五入、**.5 进偶数**，与 OpenCV 的 saturate_cast<uchar> 一致。 */
function cvRound(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return (f % 2 === 0) ? f : f + 1;
}

const _rowBuf = new Float64Array(OBS_H);

/**
 * R 通道 (288,512) → 二值观测 (80,128)，值域 {0,1}（不是 {0,255}）。
 *
 * PyTorch 侧 uint8 输入会在 forward 里 `div_(255)`，导出的 ONNX 直接吃 [0,1]，
 * 所以这里一步到位输出 0/1，省掉一次全数组除法。
 */
export function downsample(red, out) {
  const dst = out || new Float32Array(OBS_W * OBS_H);
  for (let ox = 0; ox < OBS_W; ox++) {
    const { idx: xi, w: xw } = WX[ox];
    _rowBuf.fill(0);
    for (let k = 0; k < xi.length; k++) {
      const col = xi[k] * SCREEN_HEIGHT;
      const wx = xw[k];
      for (let oy = 0; oy < OBS_H; oy++) {
        const { idx: yi, w: yw } = WY[oy];
        let acc = 0;
        for (let j = 0; j < yi.length; j++) acc += red[col + yi[j]] * yw[j];
        _rowBuf[oy] += acc * wx;
      }
    }
    const base = ox * OBS_H;
    // cv2.threshold(small, 1, 255, THRESH_BINARY)：> 1 才点亮
    for (let oy = 0; oy < OBS_H; oy++) dst[base + oy] = cvRound(_rowBuf[oy]) > 1 ? 1 : 0;
  }
  return dst;
}

/** 一步到位：游戏状态 → 单帧观测 (80,128) 的 {0,1} Float32Array。 */
export function observe(state, scratch) {
  return downsample(renderRed(state, scratch && scratch.red), scratch && scratch.obs);
}

// ---------------------------------------------------------------------
// 帧栈：新帧在前，丢最旧的一帧。与 flappy/rollout.py: FrameStack 逐位一致 ——
// 顺序写反的话网络看到的时间箭头是反的，训练时学到的速度信息全部失效。
// ---------------------------------------------------------------------
export class FrameStack {
  constructor(n = 4) {
    this.n = n;
    this.frame = OBS_W * OBS_H;
    this.array = new Float32Array(n * this.frame);
  }

  /** 新回合：用首帧填满整个栈（对照 FrameStack.reset）。 */
  reset(frame) {
    for (let i = 0; i < this.n; i++) this.array.set(frame, i * this.frame);
    return this.array;
  }

  /** 新帧插到最前，整体后移一帧（对照 FrameStack.push）。 */
  push(frame) {
    this.array.copyWithin(this.frame, 0, (this.n - 1) * this.frame);
    this.array.set(frame, 0);
    return this.array;
  }
}
