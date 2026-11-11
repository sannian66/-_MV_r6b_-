"""r6b 单帧：python3 tools/r6b_still.py <帧号|秒s> <out.png> [--sub] [--ref]
  默认渲底层（SUB_OFF=1，与 out/render_r6/base 同口径）；--sub 叠字幕；--ref 另存第六轮同帧（out.png → out_r6.png，取自第六轮无损底层块）。"""
import os, sys, json, subprocess
ROOT = os.environ.get('MV_ROOT', '/workspace/work/mv_r6b')
R6 = '/workspace/work/guitu-s3-mv/out/render_r6/base'
a = sys.argv[1]; outp = sys.argv[2]
f = int(round(float(a[:-1]) * 30)) if a.endswith('s') else int(a)
sub = '--sub' in sys.argv
if not sub: os.environ['SUB_OFF'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.argv = [sys.argv[0]]; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, random
random.seed(f); np.random.seed(f)
import render_sprite_full as R
img = R.frame(f, {})
cv2.imwrite(outp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
if '--ref' in __import__('sys').orig_argv if hasattr(sys, 'orig_argv') else False:
    c = [c for c in json.load(open(ROOT + '/crew/round6/render_r6_chunks.json')) if c['f0'] <= f < c['f1']][0]
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f"{R6}/{c['name']}.mkv", '-vf', f"select=eq(n\\,{f - c['f0']})", '-vframes', '1', outp.replace('.png', '_r6.png')], check=True)
print('ok', f, outp)
