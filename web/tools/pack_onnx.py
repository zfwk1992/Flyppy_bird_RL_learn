"""把外部权重合并进单文件，并生成 fp16 版本，两者都做数值校验。

需要本机的 torch + pygame 环境，在**仓库根目录**运行：
    python web/tools/pack_onnx.py

import os, sys
import numpy as np, onnx, torch
sys.path.insert(0, os.getcwd())
from flappy import checkpoint

SRC = "web/model/flappy_dqn.onnx"
F32 = "web/model/flappy_dqn.onnx"          # 原地覆盖成单文件
F16 = "web/model/flappy_dqn_fp16.onnx"

m = onnx.load(SRC)                          # 带外部权重一起读进来
onnx.save(m, F32, save_as_external_data=False)
print("fp32 单文件 %.2f MB" % (os.path.getsize(F32) / 1e6))
for f in os.listdir("web/model"):
    if f.endswith(".data"):
        os.remove(os.path.join("web/model", f))
        print("删掉外部权重 %s" % f)

try:
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(F32), keep_io_types=True)
    onnx.save(m16, F16, save_as_external_data=False)
    print("fp16 单文件 %.2f MB" % (os.path.getsize(F16) / 1e6))
    have16 = True
except Exception as e:
    print("fp16 转换失败(%s)，只用 fp32" % type(e).__name__)
    have16 = False

# ---- 校验：两个都要和 PyTorch 对得上 ----
import onnxruntime as ort
dev = torch.device('cpu')
net, cfg, _ = checkpoint.load_for_inference("models/final_v1_best.pt", dev)
net.eval()
rng = np.random.default_rng(0)
xs = [(rng.random((1, 4, 80, 128)) < 0.25).astype(np.float32) for _ in range(300)]
refs = []
with torch.no_grad():
    for x in xs:
        refs.append(net(torch.from_numpy(x)).numpy())

for name, path in [("fp32", F32)] + ([("fp16", F16)] if have16 else []):
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    worst = 0.0; bad = 0
    for x, ref in zip(xs, refs):
        got = s.run(None, {'obs': x})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
        bad += int(ref.argmax(1)[0] != got.argmax(1)[0])
    print("  %-5s Q 最大绝对误差 %.3e   argmax 不一致 %d/300   %s"
          % (name, worst, bad, "OK" if bad == 0 else "**动作会变，不能用**"))
