"""导出 ONNX（fp32，带外部权重）。跑完再跑 pack_onnx.py 合并成单文件。

需要本机的 torch + pygame 环境，在**仓库根目录**运行：
    python web/tools/export_onnx.py
输入约定：float32 (1,4,80,128)，值域 [0,1]。
不导 uint8 输入，因为 model.forward 里那句
    if x.dtype == torch.uint8: x = x.float().div_(255.0)
在 trace 时会被固化成一个分支，反而不透明。JS 端本来就是自己造观测，
直接产出 0.0/1.0 的 float 更干净。
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from flappy import checkpoint

dev = torch.device('cpu')
net, cfg, ck = checkpoint.load_for_inference("models/final_v1_best.pt", dev)
net.eval()
print("加载: ep=%s recent100=%s  obs=%dx%d fc=%d"
      % (ck.get('episode'), ck.get('recent100_pipes'),
         cfg['obs_w'], cfg['obs_h'], cfg['fc_hidden']))

dummy = torch.zeros(1, cfg['frame_stack'], cfg['obs_w'], cfg['obs_h'],
                    dtype=torch.float32)
out = "web/model/flappy_dqn.onnx"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.onnx.export(
    net, dummy, out,
    input_names=['obs'], output_names=['q'],
    dynamic_axes={'obs': {0: 'batch'}, 'q': {0: 'batch'}},
    opset_version=17, do_constant_folding=True,
)
print("导出 %s  (%.2f MB)" % (out, os.path.getsize(out) / 1e6))

# ---- 数值一致性：ONNX 必须和 PyTorch 输出一致，否则等于换了个模型 ----
try:
    import onnxruntime as ort
except ImportError:
    print("!! 没装 onnxruntime，跳过数值校验")
    sys.exit(0)

sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
rng = np.random.default_rng(0)
worst = 0.0
mismatch = 0
for _ in range(200):
    # 用真实分布：二值 {0,1}
    x = (rng.random((1, cfg['frame_stack'], cfg['obs_w'], cfg['obs_h']))
         < 0.25).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    got = sess.run(None, {'obs': x})[0]
    worst = max(worst, float(np.abs(ref - got).max()))
    mismatch += int(ref.argmax(1)[0] != got.argmax(1)[0])
print("数值校验 200 组: Q 最大绝对误差 %.3e, argmax 不一致 %d 组" % (worst, mismatch))
print("结论: %s" % ("一致，可用" if worst < 1e-4 and mismatch == 0 else "**不一致，不能用**"))
