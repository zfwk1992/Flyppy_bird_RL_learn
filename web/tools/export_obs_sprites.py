"""导出观测管线要用的**红通道**位图，供 web/obs.js 解析式重建画面。

为什么是红通道
--------------
`game/flappy_env.py: _observe()` 只取 `pygame.surfarray.pixels3d(SCREEN)[:, :, 0]`
—— 也就是 R 通道 —— 再缩放、按阈值 1 二值化。所以浏览器端要复刻的不是
"看起来一样的画面"，而是**逐像素相同的 R 通道**。

为什么不直接 getImageData 读画布
--------------------------------
1. 阶段 3 要往可见画布上加计分板、边框、特效。只要 AI 从画布取像素，
   任何装饰都会污染观测 —— 一个纯视觉的改动就能让 AI 变笨，而且很难发现。
2. 画布不可能在 Node 里跑，解析式重建才能拿 1200 帧对着 Python 做逐位比对
   （见 web/tools/obs_check.mjs）。读画布的版本只能靠肉眼看截图。
3. `ctx.drawImage` 在浮点坐标上会做抗锯齿，pygame 的 blit 是整数截断，
   两者本来就对不齐。

已验证的关键事实（本脚本末尾会重新断言一遍）：
对这五张精灵，**"R>1" 与 "任意通道>1" 与 "alpha>0" 三者逐像素完全相同**。
所以合成到纯黑背景上时，R 通道就是 `sprite_R * alpha/255`，没有半透明的
中间态需要考虑。

用法：
    python web/tools/export_obs_sprites.py     # 写到 web/assets/obs-sprites.js
"""
import base64
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'assets', 'sprites')
DST = os.path.join(ROOT, 'web', 'assets', 'obs-sprites.js')

# 名字对应 game/flappy_bird_utils.py 的加载顺序；背景是纯黑，R 恒为 0，不用导
SPRITES = {
    'base': 'base.png',
    'pipe': 'pipe-green.png',
    'birdUp': 'redbird-upflap.png',
    'birdMid': 'redbird-midflap.png',
    'birdDown': 'redbird-downflap.png',
}


def red_plane(path):
    """合成到纯黑背景后的 R 通道，行优先 (h, w) uint8。"""
    a = np.array(Image.open(path).convert('RGBA')).astype(np.int32)
    rgb, alpha = a[..., :3], a[..., 3]
    comp = rgb * alpha[..., None] // 255          # straight alpha over black
    r = comp[..., 0].astype(np.uint8)
    # 断言那条让整件事成立的性质：R>1 <=> 任意通道>1 <=> alpha>0
    any_lit = comp.sum(axis=-1) > 1
    assert np.array_equal(r > 1, any_lit), path
    assert np.array_equal(r > 1, alpha > 0), path
    return r


out = ['/**',
       ' * 观测管线用的红通道位图（自动生成，勿手改）。',
       ' * 由 web/tools/export_obs_sprites.py 从 assets/sprites/*.png 导出。',
       ' *',
       ' * 每张图是合成到纯黑背景后的 R 通道，行优先（先 y 后 x），base64 编码。',
       ' * 只有 R 通道有意义 —— flappy_env._observe() 就只读这一个通道。',
       ' */',
       'function unpack(b64, w, h) {',
       '  const bin = atob(b64);',
       '  const out = new Uint8Array(w * h);',
       '  for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(i);',
       '  return { w, h, data: out };',
       '}',
       '']
entries = []
for key, fname in SPRITES.items():
    r = red_plane(os.path.join(SRC, fname))
    h, w = r.shape
    b64 = base64.b64encode(r.tobytes()).decode('ascii')
    entries.append(f"  {key}: unpack('{b64}', {w}, {h}),")
    print(f'{key:8s} {w}x{h}  {r.size} bytes -> {len(b64)} base64')

out.append('export const OBS_SPRITES = {')
out += entries
out.append('};')
out.append('')

with open(DST, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(out))
print('wrote', DST)
