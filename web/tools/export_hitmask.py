"""把 Python 的 HITMASKS 导成 web/assets/hitmasks.js。

需要本机的 torch + pygame 环境，在**仓库根目录**运行：
    python web/tools/export_hitmask.py
不在 JS 端从 PNG 的 alpha 现算，原因有二：
  1. Node 里没有 canvas，parity 比对跑不了
  2. 从 Python 直接导出，保证和 checkCrash 用的是**同一份**掩码，
     不会因为解码差异产生偏差
"""
import os, sys, base64
sys.path.insert(0, os.getcwd())
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
from game.resources import HITMASKS

def pack(mask):
    """mask[x][y] -> 按位打包的 bytes，行优先按 x 扫描。"""
    w, h = len(mask), len(mask[0])
    bits = bytearray((w * h + 7) // 8)
    n = 0
    for x in range(w):
        for y in range(h):
            if mask[x][y]:
                bits[n >> 3] |= 1 << (n & 7)
            n += 1
    return w, h, bytes(bits)

entries = []
report = []
for key, masks in (('player', HITMASKS['player']), ('pipe', HITMASKS['pipe'])):
    for i, m in enumerate(masks):
        w, h, b = pack(m)
        dens = sum(bin(x).count('1') for x in b) / (w * h)
        entries.append((key, i, w, h, base64.b64encode(b).decode()))
        report.append((key, i, w, h, dens, len(b)))

lines = [
    '/**',
    ' * 从 Python 的 `game/resources.py: HITMASKS` 直接导出的像素碰撞掩码。',
    ' *',
    ' * 不在 JS 端从 PNG 的 alpha 现算，原因：',
    ' *   1. Node 里没有 canvas，parity 比对就跑不了',
    ' *   2. 直接导出能保证和 Python 的 checkCrash 用的是**同一份**掩码，',
    ' *      不会因为 PNG 解码差异产生偏差',
    ' *',
    ' * 按位打包，行优先按 x 扫描：bit(x*h + y) 对应 mask[x][y]。',
    ' * 由 scratchpad/export_hitmask.py 生成，改了精灵才需要重新生成。',
    ' */',
    '',
    'function unpack(b64, w, h) {',
    '  const bin = atob(b64);',
    '  const mask = [];',
    '  let n = 0;',
    '  for (let x = 0; x < w; x++) {',
    '    const col = new Uint8Array(h);',
    '    for (let y = 0; y < h; y++) {',
    '      col[y] = (bin.charCodeAt(n >> 3) >> (n & 7)) & 1;',
    '      n++;',
    '    }',
    '    mask.push(col);',
    '  }',
    '  return mask;',
    '}',
    '',
    '// Node 没有 atob（18+ 才有全局），补一个',
    "const atob = globalThis.atob || ((s) => Buffer.from(s, 'base64').toString('binary'));",
    '',
]
for key, i, w, h, b64 in entries:
    lines.append('const %s%d = ["%s", %d, %d];' % (key.upper(), i, b64, w, h))
lines += [
    '',
    'export const HITMASKS = {',
    '  player: [%s],' % ', '.join('unpack(...PLAYER%d)' % i for i in range(3)),
    '  pipeUpper: unpack(...PIPE0),',
    '  pipeLower: unpack(...PIPE1),',
    '};',
    '',
]
dst = 'web/assets/hitmasks.js'
with open(dst, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines))

print('wrote %s (%.1f KB)' % (dst, os.path.getsize(dst) / 1024))
for key, i, w, h, dens, nb in report:
    print('  %-7s[%d] %3dx%-3d  不透明像素占比 %5.1f%%  打包 %5d B' % (key, i, w, h, 100 * dens, nb))
