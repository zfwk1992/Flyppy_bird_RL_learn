"""把训练好的网络导成浏览器能直接读的裸权重（fp16）+ 一份形状元数据。

为什么不用 onnxruntime-web
--------------------------
这个 demo 是发到 LinkedIn 上的，首屏体积就是转化率。onnxruntime-web 的
wasm 运行时压缩后仍有好几 MB，而这个网络只有 3 个卷积 + 3 个线性层、
既没有 BatchNorm 也没有 Dropout，单次前向约 1230 万次乘加、每秒只要 7.5 次
（frame_skip=4，30fps）。手写前向完全跑得动，还省掉整个运行时。
`web/model/flappy_dqn_fp16.onnx` 留着不删 —— 它是这套权重的第三方可验证副本，
也是万一要换回 onnxruntime 的退路。

为什么是 fp16
-------------
体积减半（5.05 MB → 2.53 MB），而 300 组随机输入下 argmax 一次都没变过
（Q 值误差 8e-3，见 export_onnx.py）。浏览器端解回 fp32 再算，
精度只会比 fp16 推理更高。

用法：
    python web/tools/export_weights.py
        -> web/model/weights_fp16.bin      裸权重，按下面的顺序首尾相接
        -> web/model/weights-meta.js       每个张量的形状与字节偏移
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from flappy import checkpoint                                   # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default=os.path.join(ROOT, 'models', 'final_v1_best.pt'))
_ap.add_argument('--tag', default='', help='非空时输出 weights_<tag>_fp16.bin / weights-meta-<tag>.js')
_a = _ap.parse_args()
CKPT = _a.ckpt
BIN = os.path.join(ROOT, 'web', 'model', 'weights%s_fp16.bin' % (('_'+_a.tag) if _a.tag else ''))
META = os.path.join(ROOT, 'web', 'model', 'weights-meta%s.js' % (('-'+_a.tag) if _a.tag else ''))

# 顺序 = 前向顺序。改了这里必须同步改 web/nn.js 的解析顺序 ——
# 所以偏移量写进元数据由 JS 按名字取，而不是两边各数一遍。
ORDER = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
         'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias',
         'value.weight', 'value.bias', 'advantage.weight', 'advantage.bias']

net, cfg, ckpt = checkpoint.load_for_inference(CKPT, torch.device('cpu'))
sd = net.state_dict()
assert set(sd.keys()) == set(ORDER), sorted(sd.keys())

blobs, entries, offset = [], [], 0
for name in ORDER:
    t = sd[name].detach().cpu().numpy().astype(np.float16)
    blobs.append(t.tobytes())
    entries.append((name, list(t.shape), offset, t.size))
    offset += t.nbytes

with open(BIN, 'wb') as f:
    for b in blobs:
        f.write(b)

lines = [
    '/**',
    ' * 权重元数据（自动生成，勿手改）—— 由 web/tools/export_weights.py 导出。',
    ' *',
    ' * weights_fp16.bin 是这些张量按 `order` 首尾相接的 fp16 裸数据，',
    ' * `offset` 是字节偏移，`shape` 与 PyTorch 的 state_dict 完全一致：',
    ' *   卷积权重 (out, in, kh, kw)，线性权重 (out, in)。',
    ' */',
    'export const WEIGHTS_META = {',
    f"  file: '%s'," % os.path.basename(BIN),
    f'  bytes: {offset},',
    f"  input: {{ channels: {cfg['frame_stack']}, h: {cfg['obs_w']}, w: {cfg['obs_h']} }},",
    f"  fcHidden: {cfg['fc_hidden']},",
    '  tensors: {',
]
for name, shape, off, n in entries:
    key = name.replace('.', '_')
    lines.append(f"    {key}: {{ shape: {shape}, offset: {off}, count: {n} }},")
lines += ['  },', '};', '']

with open(META, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines))

total = sum(n for _, _, _, n in entries)
print('checkpoint : %s (recent100_pipes=%.2f)'
      % (CKPT, ckpt.get('recent100_pipes', float('nan'))))
print('input      : (1, %d, %d, %d)' % (cfg['frame_stack'], cfg['obs_w'], cfg['obs_h']))
print('params     : %d  ->  %s  %.2f MB' % (total, BIN, offset / 1e6))
for name, shape, off, n in entries:
    print('  %-18s %-22s @%d' % (name, shape, off))
print('wrote', META)
