"""挑选候选 → assets/gen/<镜号>/<姿势>_raw.png（2560 宽）→ tools/cutout.py → _cut.png
用法：python3 tools/pick_cut.py S035/run=/path/cand.png[:flip] ..."""
import sys, os
from PIL import Image, ImageOps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/tools')
import cutout
for arg in sys.argv[1:]:
    job, src = arg.split('=')
    flip = src.endswith(':flip'); src = src.replace(':flip', '')
    shot, pose = job.split('/')
    od = f'{ROOT}/assets/gen/{shot}'; os.makedirs(od, exist_ok=True)
    im = Image.open(src).convert('RGB')
    im = im.resize((2560, round(2560 * im.size[1] / im.size[0])), Image.LANCZOS)
    if flip: im = ImageOps.mirror(im)
    raw = f'{od}/{pose}_raw.png'; im.save(raw)
    if pose != 'eyes':
        cutout.cutout(raw, f'{od}/{pose}_cut.png', 2560)
    print('done', job, flush=True)
