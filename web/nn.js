/**
 * Dueling Double-DQN 的前向推理，手写、零依赖。
 *
 * 为什么不挂 onnxruntime-web：见 web/tools/export_weights.py 的开头。
 * 一句话 —— 这网络一次前向约 1230 万次乘加、每秒只需要 7.5 次
 * （frame_skip=4、30fps），为它拖进来几 MB 的 wasm 运行时不划算。
 *
 * 结构逐行对照 `flappy/model.py: DuelingDQN`：
 *
 *   conv1  4->32  k8 s4   (4,80,128) -> (32,19,31)   relu
 *   conv2 32->64  k4 s2               -> (64, 8, 14)  relu
 *   conv3 64->64  k3 s1               -> (64, 6, 12)  relu
 *   flatten 4608 -> fc 256            relu
 *   value 256->1     advantage 256->2
 *   Q = V + A - mean(A)
 *
 * 注意两处**不是**随便写的细节：
 *
 * 1. **输入的两个空间轴是转置的**。观测数组 axis0 = 屏幕宽 (80)、
 *    axis1 = 屏幕高 (128)，因为 pygame.surfarray 是 width-major。
 *    卷积不在乎哪边是"上"，但形状必须和训练时一致。
 * 2. **A 要减自己的均值**。少了这一步 V 和 A 就不可辨识 ——
 *    给 V 加常数、给 A 减同一个常数，Q 不变，训练会在这条零方向上漂移。
 */
import { WEIGHTS_META } from './model/weights-meta.js';

// IEEE-754 半精度解码。不用 DataView.getFloat16：那是很新的 API，
// 而这个 demo 要在别人随便什么浏览器里打开。
function f16to32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h & 0x7C00) >> 10;
  const frac = h & 0x03FF;
  if (exp === 0) return sign * 6.103515625e-5 * (frac / 1024);   // 非规格化
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/** 把 weights_fp16.bin 的 ArrayBuffer 解成 {名字: Float32Array}。 */
export function parseWeights(buf) {
  if (buf.byteLength !== WEIGHTS_META.bytes) {
    throw new Error(`权重文件长度不对：${buf.byteLength} != ${WEIGHTS_META.bytes}`);
  }
  const u16 = new Uint16Array(buf);
  const out = {};
  for (const [key, t] of Object.entries(WEIGHTS_META.tensors)) {
    const start = t.offset / 2;
    const a = new Float32Array(t.count);
    for (let i = 0; i < t.count; i++) a[i] = f16to32(u16[start + i]);
    out[key] = a;
  }
  return out;
}

/** 浏览器里的加载入口。`base` 是 weights-meta.js 所在目录的 URL。 */
export async function loadWeights(base = './model/') {
  const res = await fetch(base + WEIGHTS_META.file);
  if (!res.ok) throw new Error(`权重加载失败：${res.status} ${base}${WEIGHTS_META.file}`);
  return parseWeights(await res.arrayBuffer());
}

// ---------------------------------------------------------------------
// 算子
// ---------------------------------------------------------------------

/**
 * 直接卷积，无 padding。
 *
 * 循环顺序是 oc → ic → (kh,kw) → 输出空间：把权重提成内层循环外的标量，
 * 内层对输出行是连续写、对输入行是等步长读。im2col 更快但要一块
 * 4608xN 的临时内存，而这里的瓶颈根本不在这一层（见 nn_check.mjs 的计时）。
 */
function conv2d(inp, inC, inH, inW, w, b, outC, k, stride, out, outH, outW) {
  const plane = outH * outW;
  for (let oc = 0; oc < outC; oc++) {
    const ob = oc * plane;
    out.fill(b[oc], ob, ob + plane);
    for (let ic = 0; ic < inC; ic++) {
      const ib = ic * inH * inW;
      const wb = (oc * inC + ic) * k * k;
      for (let kh = 0; kh < k; kh++) {
        for (let kw = 0; kw < k; kw++) {
          const wv = w[wb + kh * k + kw];
          if (wv === 0) continue;
          for (let oh = 0; oh < outH; oh++) {
            const irow = ib + (oh * stride + kh) * inW + kw;
            const orow = ob + oh * outW;
            for (let ow = 0; ow < outW; ow++) {
              out[orow + ow] += wv * inp[irow + ow * stride];
            }
          }
        }
      }
    }
  }
}

function relu(a) {
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
}

/** y = W x + b，W 是 (out, in) 行优先，和 PyTorch 的 Linear.weight 一致。 */
function linear(x, w, b, outN, inN, out) {
  for (let o = 0; o < outN; o++) {
    const wb = o * inN;
    let acc = b[o];
    for (let i = 0; i < inN; i++) acc += w[wb + i] * x[i];
    out[o] = acc;
  }
}

// ---------------------------------------------------------------------
// 网络
// ---------------------------------------------------------------------
const outSize = (n, k, s) => Math.floor((n - k) / s) + 1;

export class DuelingDQN {
  constructor(weights) {
    this.w = weights;
    const { channels, h, w } = WEIGHTS_META.input;
    this.inC = channels; this.inH = h; this.inW = w;

    this.h1 = outSize(h, 8, 4); this.w1 = outSize(w, 8, 4);
    this.h2 = outSize(this.h1, 4, 2); this.w2 = outSize(this.w1, 4, 2);
    this.h3 = outSize(this.h2, 3, 1); this.w3 = outSize(this.w2, 3, 1);
    this.flat = 64 * this.h3 * this.w3;

    const expect = WEIGHTS_META.tensors.fc_weight.shape[1];
    if (this.flat !== expect) {
      throw new Error(`展平维度对不上：算出 ${this.flat}，权重要 ${expect}`);
    }

    // 中间张量只分配一次 —— 每秒 7.5 次前向，各分配 500KB 会让 GC 频繁醒来
    this.a1 = new Float32Array(32 * this.h1 * this.w1);
    this.a2 = new Float32Array(64 * this.h2 * this.w2);
    this.a3 = new Float32Array(this.flat);
    this.a4 = new Float32Array(WEIGHTS_META.fcHidden);
    this.v = new Float32Array(1);
    this.adv = new Float32Array(WEIGHTS_META.tensors.advantage_bias.count);
    this.q = new Float32Array(this.adv.length);
  }

  /** obs: Float32Array(4*80*128)，值域 {0,1}。返回内部复用的 Q 数组。 */
  forward(obs) {
    const w = this.w;
    conv2d(obs, this.inC, this.inH, this.inW, w.conv1_weight, w.conv1_bias,
      32, 8, 4, this.a1, this.h1, this.w1);
    relu(this.a1);
    conv2d(this.a1, 32, this.h1, this.w1, w.conv2_weight, w.conv2_bias,
      64, 4, 2, this.a2, this.h2, this.w2);
    relu(this.a2);
    conv2d(this.a2, 64, this.h2, this.w2, w.conv3_weight, w.conv3_bias,
      64, 3, 1, this.a3, this.h3, this.w3);
    relu(this.a3);
    linear(this.a3, w.fc_weight, w.fc_bias, WEIGHTS_META.fcHidden, this.flat, this.a4);
    relu(this.a4);
    linear(this.a4, w.value_weight, w.value_bias, 1, WEIGHTS_META.fcHidden, this.v);
    linear(this.a4, w.advantage_weight, w.advantage_bias,
      this.adv.length, WEIGHTS_META.fcHidden, this.adv);

    let mean = 0;
    for (let i = 0; i < this.adv.length; i++) mean += this.adv[i];
    mean /= this.adv.length;
    for (let i = 0; i < this.q.length; i++) this.q[i] = this.v[0] + this.adv[i] - mean;
    return this.q;
  }

  /** 贪婪动作。eval_epsilon=0 —— 评测里加 1% 随机曾让分数掉 6.2 倍。 */
  act(obs) {
    const q = this.forward(obs);
    let best = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
    return best;
  }
}
