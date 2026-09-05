/**
 * Canvas 渲染 —— 黑底 + 管道 / 地面 / 小鸟。
 *
 * 逐条对照 `game/flappy_env.py: _draw()`（blit 顺序：background → pipes →
 * base → player，管道用 `zip(upperPipes, lowerPipes)` 先上后下）和
 * `game/flappy_bird_utils.py` 的精灵加载（上管道 = pipe-green.png 旋转
 * 180°，玩家三帧顺序 upflap/midflap/downflap 对应 playerIndex 0/1/2）。
 *
 * 只负责画，不负责物理/输入/游戏循环 —— 状态来自
 * `FlappyGame#observeState()`（见 game.js），逐帧只读，不修改。
 */
import {
  SCREEN_HEIGHT,
  BASE_Y,
  PLAYER_WIDTH,
  PLAYER_HEIGHT,
  PIPE_WIDTH,
  PIPE_HEIGHT,
  BACKGROUND_WIDTH,
} from './game.js';
import { SPRITE_DATA_URIS } from './assets/sprites-data.js';

// 精灵内嵌成 base64 data URI（见 assets/sprites-data.js），而不是作为独立
// PNG 文件提交 —— 仓库根 .gitignore 里的 `*.png` 规则只给 `images/*.png`
// 开了口子，改 .gitignore 超出了这次改动被允许触碰的范围
// （只能改 web/、plan.md、web/PROGRESS.md），所以用这种方式绕开，
// 而不是去改仓库级配置。6 张精灵原始体积一共约 18KB，base64 后约 25KB，
// 可忽略不计。
function loadImage(dataUri) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('加载精灵失败（data URI 解码出错）'));
    img.src = dataUri;
  });
}

/**
 * 把一张图旋转 180° 烘焙进一个离屏 canvas，对照 Python 侧
 * `pygame.transform.rotate(pipe_img, 180)` 生成上管道贴图的做法 ——
 * 只在加载时做一次，每帧渲染直接 drawImage，不必每帧都做坐标变换。
 */
function rotate180(img) {
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  const g = c.getContext('2d');
  g.translate(c.width, c.height);
  g.rotate(Math.PI);
  g.drawImage(img, 0, 0);
  return c;
}

/** 预加载全部精灵，返回 Renderer 构造函数要用的 sprites 对象。 */
export async function loadSprites() {
  const [background, base, pipe, upflap, midflap, downflap] = await Promise.all([
    loadImage(SPRITE_DATA_URIS.background),
    loadImage(SPRITE_DATA_URIS.base),
    loadImage(SPRITE_DATA_URIS.pipe),
    loadImage(SPRITE_DATA_URIS.upflap),
    loadImage(SPRITE_DATA_URIS.midflap),
    loadImage(SPRITE_DATA_URIS.downflap),
  ]);
  return {
    background,
    base,
    pipeLower: pipe,
    pipeUpper: rotate180(pipe),
    // 顺序对照 flappy_bird_utils.py 的 PLAYER_PATH：0=upflap 1=midflap 2=downflap
    player: [upflap, midflap, downflap],
  };
}

export class Renderer {
  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {ReturnType<typeof loadSprites> extends Promise<infer T> ? T : never} sprites
   */
  constructor(ctx, sprites) {
    this.ctx = ctx;
    this.sprites = sprites;
  }

  /** @param {ReturnType<import('./game.js').FlappyGame['observeState']>} state */
  draw(state) {
    const { ctx, sprites } = this;

    // 1. 背景：纯黑（对照 IMAGES['background']），先铺满再画其余部分
    ctx.drawImage(sprites.background, 0, 0, BACKGROUND_WIDTH, SCREEN_HEIGHT);

    // 2. 管道：先上后下，对照 _draw() 里 zip(upperPipes, lowerPipes) 的顺序
    for (let i = 0; i < state.upperPipes.length; i++) {
      const u = state.upperPipes[i];
      const l = state.lowerPipes[i];
      ctx.drawImage(sprites.pipeUpper, u.x, u.y, PIPE_WIDTH, PIPE_HEIGHT);
      ctx.drawImage(sprites.pipeLower, l.x, l.y, PIPE_WIDTH, PIPE_HEIGHT);
    }

    // 3. 地面：管道之上、小鸟之下，单次 blit（不平铺）——
    //    对照 _draw() 的 SCREEN.blit(IMAGES['base'], (basex, BASEY))。
    //    base.png (336px) 比屏幕 (288px) 宽 48px，basex 的取值范围
    //    正好是 (-48, 0]，所以单次绘制已经能铺满整个屏幕宽度，
    //    和 Python 侧一样不需要平铺第二张。
    ctx.drawImage(sprites.base, state.basex, BASE_Y);

    // 4. 小鸟
    ctx.drawImage(
      sprites.player[state.playerIndex],
      state.playerx,
      state.playery,
      PLAYER_WIDTH,
      PLAYER_HEIGHT
    );
  }
}
