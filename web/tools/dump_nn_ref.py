"""导出 PyTorch 在 trace_ai 轨迹上每次决策的 Q 值，供 web/tools/nn_check.mjs 比对。

阶段 2 的验收要回答两个能互相混淆的问题：观测管线对不对，推理对不对。
obs_check.mjs 已经把前者钉死（1200 帧逐位一致），这里补后者 ——
在**同一条轨迹的同一批帧栈**上比 Q 值本身，而不只是比 argmax。
比 Q 值是因为 argmax 只有 2 个取值，碰巧一致的概率太高；Q 值对上了，
卷积权重、展平顺序、V/A 合并这些才算真的验过。

决策时机严格照抄 `dump_python_trace.py`：每 frame_skip 帧在 **step 之前**
观测一次（此时画面是上一帧 step 之后的状态），窗口内重复同一个动作。

用法：
    python web/tools/dump_nn_ref.py       # 写到 web/tools/nn_ref.json
"""
import hashlib
import json
import os
import random
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

TRACE = os.path.join(HERE, 'trace_ai.json')
DST = os.path.join(HERE, 'nn_ref.json')
CKPT = os.path.join(ROOT, 'models', 'final_v1_best.pt')

with open(TRACE, encoding='utf-8') as f:
    trace = json.load(f)
draws, frames_ref = trace['draws'], trace['frames']

_cursor = {'i': 0}


def _replay(self):
    i = _cursor['i']
    if i >= len(draws):
        raise RuntimeError('随机数用尽：这次重放比录制时多消费了抽样')
    _cursor['i'] = i + 1
    return draws[i]


random.Random.random = _replay

from flappy import checkpoint                        # noqa: E402
from flappy.config import resolve_config             # noqa: E402
from flappy.rollout import make_env, FrameStack      # noqa: E402

cfg = resolve_config()
dev = torch.device('cpu')          # CPU 导出，免得 cuDNN 的算法选择引入抖动
net, mcfg, _ = checkpoint.load_for_inference(CKPT, dev)
env = make_env(cfg)
env.reset()

stack = FrameStack(cfg['frame_stack'])
just_reset = True
decisions = []
cur_action = 0
for i, ref in enumerate(frames_ref):
    if i % cfg['frame_skip'] == 0:
        obs = env.observe()
        arr = stack.reset(obs) if just_reset else stack.push(obs)
        just_reset = False
        with torch.no_grad():
            q = net(torch.from_numpy(arr).unsqueeze(0).to(dev))[0]
        cur_action = int(q.argmax().item())
        assert cur_action == ref['action'], (
            f'帧 {i}: 重放算出的动作 {cur_action} 与 trace 记录的 {ref["action"]} 不一致')
        decisions.append({
            'i': i,
            'q': [float(q[0]), float(q[1])],
            'action': cur_action,
            'stack_md5': hashlib.md5(np.ascontiguousarray(arr)).hexdigest(),
        })
    _, _, done, _ = env.step(cur_action, render=True)
    assert bool(done) == bool(ref['done']), f'帧 {i}: done 与 trace 不一致'
    if done:
        env.reset()
        just_reset = True

flap = sum(d['action'] for d in decisions)
out = {
    'meta': {
        'source': 'trace_ai.json', 'checkpoint': 'models/final_v1_best.pt',
        'frame_skip': cfg['frame_skip'], 'frame_stack': cfg['frame_stack'],
        'n_decisions': len(decisions), 'n_flap': flap,
        'device': 'cpu',
        'note': 'stack_md5 取自 (4,80,128) uint8 {0,255} 的 C 连续数组；'
                'q 是 fp32 PyTorch 的输出',
    },
    'decisions': decisions,
}
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(out, f)
print('wrote %s : %d decisions (%d flap / %.1f%%)'
      % (DST, len(decisions), flap, 100.0 * flap / len(decisions)))
