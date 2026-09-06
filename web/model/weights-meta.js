/**
 * 权重元数据（自动生成，勿手改）—— 由 web/tools/export_weights.py 导出。
 *
 * weights_fp16.bin 是这些张量按 `order` 首尾相接的 fp16 裸数据，
 * `offset` 是字节偏移，`shape` 与 PyTorch 的 state_dict 完全一致：
 *   卷积权重 (out, in, kh, kw)，线性权重 (out, in)。
 */
export const WEIGHTS_META = {
  file: 'weights_base_s0_fp16.bin',
  bytes: 2517318,
  input: { channels: 4, h: 80, w: 128 },
  fcHidden: 256,
  tensors: {
    conv1_weight: { shape: [32, 4, 8, 8], offset: 0, count: 8192 },
    conv1_bias: { shape: [32], offset: 16384, count: 32 },
    conv2_weight: { shape: [64, 32, 4, 4], offset: 16448, count: 32768 },
    conv2_bias: { shape: [64], offset: 81984, count: 64 },
    conv3_weight: { shape: [64, 64, 3, 3], offset: 82112, count: 36864 },
    conv3_bias: { shape: [64], offset: 155840, count: 64 },
    fc_weight: { shape: [256, 4608], offset: 155968, count: 1179648 },
    fc_bias: { shape: [256], offset: 2515264, count: 256 },
    value_weight: { shape: [1, 256], offset: 2515776, count: 256 },
    value_bias: { shape: [1], offset: 2516288, count: 1 },
    advantage_weight: { shape: [2, 256], offset: 2516290, count: 512 },
    advantage_bias: { shape: [2], offset: 2517314, count: 2 },
  },
};
