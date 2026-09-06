"""导出观测管线的参考数据，供 web/tools/obs_check.mjs 逐帧比对（阶段 2 验收）。

做法和阶段 1 的 parity 一样：重放 `trace_ai.json` 里记下的 random() 抽样
和动作，让 Python 走一遍完全相同的轨迹，把每一帧的

  * R 通道原始画面 (288,512) 的 md5
  * 二值观测 (80,128) 的 md5

都记下来。JS 侧解析式重建同一帧，比 md5。这样一次比对同时验两件事：
**画面重建**对不对（第一个 md5），以及 **INTER_AREA + 阈值**对不对（第二个）。
分开记是为了定位 —— 只错第二个就说明重建没问题、缩放实现有偏差。

只存 md5 不存像素：1200 帧的原图是 176 MB，md5 只有 75 KB。
另外抽 5 帧存位压缩的观测，比对失败时能直接打出图看差在哪。

用法：
    python web/tools/dump_obs_ref.py      # 写到 web/tools/obs_ref.json
"""
import base64
import hashlib
import json
import os
import random
import sys

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np                                          # noqa: E402
import pygame                                               # noqa: E402

TRACE = os.path.join(HERE, 'trace_ai.json')
DST = os.path.join(HERE, 'obs_ref.json')
SAMPLE_AT = [0, 300, 600, 900, 1199]

with open(TRACE, encoding='utf-8') as f:
    trace = json.load(f)
draws, frames_ref, meta = trace['draws'], trace['frames'], trace['meta']

# 按顺序重放 Python 当初消费的每一个 random()。用完就报错 —— 多消费本身是 bug。
_cursor = {'i': 0}


def _replay(self):
    i = _cursor['i']
    if i >= len(draws):
        raise RuntimeError('随机数用尽：这次重放比录制时多消费了抽样')
    _cursor['i'] = i + 1
    return draws[i]


random.Random.random = _replay

from flappy.config import resolve_config          # noqa: E402
from flappy.rollout import make_env               # noqa: E402
import game.flappy_env as fenv                    # noqa: E402

cfg = resolve_config()
env = make_env(cfg)
env.reset()

red_md5, obs_md5, player_idx, samples = [], [], [], {}
for i, ref in enumerate(frames_ref):
    obs, _, done, _ = env.step(ref['action'], render=True)
    # step(render=True) 之后 SCREEN 上就是这一帧；array3d 拿的是拷贝，
    # 不会像 pixels3d 那样锁住 Surface
    red = pygame.surfarray.array3d(fenv.SCREEN)[:, :, 0]
    assert red.shape == (288, 512), red.shape
    assert obs.shape == (cfg['obs_w'], cfg['obs_h']), obs.shape
    red_md5.append(hashlib.md5(np.ascontiguousarray(red)).hexdigest())
    obs_md5.append(hashlib.md5(np.ascontiguousarray(obs)).hexdigest())
    player_idx.append(int(env.playerIndex))
    if i in SAMPLE_AT:
        bits = np.packbits((obs > 0).astype(np.uint8).ravel())
        samples[str(i)] = base64.b64encode(bits.tobytes()).decode('ascii')
    assert bool(done) == bool(ref['done']), f'frame {i}: done 与 trace 不一致'
    if done:
        env.reset()

out = {
    'meta': {
        'source': 'trace_ai.json', 'n_frames': len(frames_ref),
        'draws_consumed': _cursor['i'], 'obs_w': cfg['obs_w'], 'obs_h': cfg['obs_h'],
        'screen': [288, 512], 'sample_at': SAMPLE_AT,
        'note': 'md5 取自 C 连续的 uint8 数组：red 是 (288,512) 的 R 通道，'
                'obs 是 (80,128) 的 {0,255} 二值图',
    },
    'red_md5': red_md5, 'obs_md5': obs_md5, 'player_index': player_idx,
    'obs_sample_packed': samples,
}
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(out, f)
print('wrote %s : %d frames, %d draws consumed'
      % (DST, len(red_md5), _cursor['i']))
