"""副歌一样片渲染：python3 units/render_chorus1.py [contact|seg F0 F1 OUT|all]
contact: 每镜关键帧联系表 out/chorus1_contact.jpg
all: 58.5–89.5s 两进程分段并行 + 混音 → out/sample_chorus1.mp4"""
import sys, os, subprocess, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = '/workspace/work/mv'
sys.path.insert(0, ROOT + '/units'); sys.path.insert(0, ROOT + '/lib'); sys.path.insert(0, ROOT)
import numpy as np, cv2
cv2.setNumThreads(1)
FPS = 30
T0, T1 = 58.5, 89.5
OUT = ROOT + '/out'


def seg(f0, f1, outp):
    import chorus1 as C
    X = C.Chorus1()
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', str(FPS),
                          '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', outp],
                         stdin=subprocess.PIPE)
    a = time.time()
    for f in range(f0, f1):
        t = f / FPS
        fr, _ = X.frame(t, 1 / FPS)
        p.stdin.write(C.to8(fr).tobytes())
        if f % 60 == 0: print(f'{t:.2f}s  {(time.time() - a) / max(1, f - f0 + 1):.2f}s/f', flush=True)
    p.stdin.close(); p.wait()


def contact(times=None, outp=OUT + '/chorus1_contact.jpg', cols=4):
    import chorus1 as C
    from PIL import Image, ImageDraw
    X = C.Chorus1()
    if times is None:
        times = []
        for s in X.shots:
            a, b = max(s.t0, T0), min(s.t1, T1)
            times += [a + (b - a) * 0.35, a + (b - a) * 0.8]
    ims = []
    for t in times:
        a = time.time(); fr, _ = X.frame(t)
        dtm = time.time() - a
        im = Image.fromarray(C.to8(fr)).resize((640, 360))
        d = ImageDraw.Draw(im); d.text((6, 4), f'{X.ids[X.idx(t)]} {t:.2f}s {dtm:.2f}s', fill=(255, 220, 120))
        ims.append(im); print(f'{t:.2f} {X.ids[X.idx(t)]} {dtm:.2f}s', flush=True)
    rows = (len(ims) + cols - 1) // cols
    S = Image.new('RGB', (640 * cols, 360 * rows))
    for k, i in enumerate(ims): S.paste(i, ((k % cols) * 640, (k // cols) * 360))
    S.save(outp, quality=86); print(outp)


def all_():
    f0, f1 = int(round(T0 * FPS)), int(round(T1 * FPS))
    mid = (f0 + f1) // 2
    os.makedirs(OUT + '/c1chunks', exist_ok=True)
    parts = [(f0, mid), (mid, f1)]
    procs = []
    for a, b in parts:
        o = f'{OUT}/c1chunks/p_{a}.mp4'
        procs.append(subprocess.Popen([sys.executable, __file__, 'seg', str(a), str(b), o],
                                      stdout=open(f'{OUT}/c1chunks/log_{a}.txt', 'w'), stderr=subprocess.STDOUT))
    for p in procs: p.wait()
    lst = OUT + '/c1chunks/list.txt'
    open(lst, 'w').write(''.join(f"file 'p_{a}.mp4'\n" for a, b in parts))
    subprocess.check_call(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-ss', '0', '-t', f'{T1 - T0:.3f}',
                           '-i', ROOT + '/song.mp3', '-ss', f'{T0:.3f}', '-t', f'{T1 - T0:.3f}',
                           '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
                           OUT + '/sample_chorus1.mp4'])
    print('done', OUT + '/sample_chorus1.mp4')


CH = 90   # 每块帧数


def chunks(budget=100):
    """分块渲染（每次调用最多跑 budget 秒，2 进程并行；已完成的块跳过），可反复调用直到全部完成。"""
    f0, f1 = int(round(T0 * FPS)), int(round(T1 * FPS))
    os.makedirs(OUT + '/c1chunks', exist_ok=True)
    todo = [(a, min(a + CH, f1)) for a in range(f0, f1, CH)]
    todo = [c for c in todo if not os.path.exists(f'{OUT}/c1chunks/c_{c[0]}.mp4')]
    st = time.time(); run = []
    while (todo or run):
        while todo and len(run) < 2 and time.time() - st < budget:
            a, b = todo.pop(0); o = f'{OUT}/c1chunks/c_{a}.mp4'
            pr = subprocess.Popen([sys.executable, __file__, 'seg', str(a), str(b), o + '.tmp.mp4'],
                                  stdout=open(f'{OUT}/c1chunks/log_{a}.txt', 'w'), stderr=subprocess.STDOUT)
            run.append((pr, o))
        for pr, o in run[:]:
            if pr.poll() is not None:
                run.remove((pr, o))
                if pr.returncode == 0: os.replace(o + '.tmp.mp4', o)
        if not run and (not todo or time.time() - st >= budget): break
        time.sleep(1)
    left = len([c for c in [(a, 0) for a in range(f0, f1, CH)] if not os.path.exists(f'{OUT}/c1chunks/c_{c[0]}.mp4')])
    print('remaining chunks', left)


def mux():
    f0, f1 = int(round(T0 * FPS)), int(round(T1 * FPS))
    lst = OUT + '/c1chunks/list.txt'
    open(lst, 'w').write(''.join(f"file 'c_{a}.mp4'\n" for a in range(f0, f1, CH)))
    subprocess.check_call(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                           '-ss', f'{T0:.3f}', '-t', f'{T1 - T0:.3f}', '-i', ROOT + '/song.mp3',
                           '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
                           '-movflags', '+faststart', OUT + '/sample_chorus1.mp4'])
    print('done', OUT + '/sample_chorus1.mp4')


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'contact'
    if m == 'seg': seg(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif m == 'contact': contact([float(x) for x in sys.argv[2:]] or None)
    elif m == 'all': all_()
    elif m == 'chunks': chunks(float(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif m == 'mux': mux()
