"""A 组联系表 / 关键帧表：按本组自己的镜头表（units/sec_a_intro_verse1.SHOTS）取帧。
  tools/cpu_slot.sh python3 tools/a_contact.py contact out/round4/A_contact_vN.jpg      每镜 55% 处一帧
  tools/cpu_slot.sh python3 tools/a_contact.py keys out/round4/A_keys_vN.jpg            转场中间帧 + 歌词/冲击关键帧
  python3 tools/a_contact.py times out.jpg 1.0 2.5 ...                                 指定秒"""
import os, sys, time
ROOT = '/workspace/work/mv'; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/units'); sys.path.insert(0, ROOT + '/lib')
os.chdir(ROOT)
from PIL import Image, ImageDraw, ImageFont
import sec_a_intro_verse1 as M

FPS = 30
font = ImageFont.truetype(ROOT + '/fonts/NotoSerifSC-Bold.otf', 22)


def sheet(items, out, cols=4, tw=480, th=270):
    rows = (len(items) + cols - 1) // cols
    sh = Image.new('RGB', (cols * tw, rows * (th + 30)), (12, 12, 12)); d = ImageDraw.Draw(sh)
    cache = {}
    for i, (label, t) in enumerate(items):
        a = time.time()
        f = int(round(t * FPS))
        im = Image.fromarray(M.frame(f, cache)).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * (th + 30)
        sh.paste(im, (x, y + 30)); d.text((x + 6, y + 2), f'{label}  {t:.2f}s', fill=(240, 220, 120), font=font)
        print(label, round(t, 2), round(time.time() - a, 2), 's', flush=True)
    sh.save(out, quality=86); print('saved', out)


if __name__ == '__main__':
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == 'contact':
        items = [(S.__name__, S.t0 + 0.55 * (S.t1 - S.t0)) for S in M.SHOTS]
    elif cmd == 'keys':
        items = []
        for S in M.SHOTS[1:]:
            c = M.CUTS.get(S.__name__)
            if c and c[0] not in ('hard', 'whip'):
                items.append((S.__name__ + ' ' + c[0][:10], S.t0 + (c[2] - c[1]) / 2 * 0.5))
        items += [('title', 8.10), ('credits', 1.6), ('寒光', 21.5), ('罪状', 24.9), ('称量', 28.7), ('杀', 30.3), ('刻刀', 34.2),
                  ('正印', 38.8), ('锋', 43.55), ('exit', 44.0)]
    else:
        items = [('t', float(x)) for x in sys.argv[3:]]
    sheet(items, out)
