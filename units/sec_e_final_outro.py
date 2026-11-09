"""units/sec_e_final_outro.py — E 组：副歌终「定」+ 尾奏（帧 5797–7241，193.24–241.34s）
设计说明：crew/round4/E_storyboard.md。36 镜：副歌终 26 镜（平均 1.08s）+ 尾奏 10 镜（含片名再现、6 条署名、收刀落印、收墨）。
接口：READY=False；frame(f, cache) -> uint8 (1080,1920,3) 成品帧（含黑边/颗粒/暗角）。
结构仿 units/chorus1.py：每镜一个类（render(t) 场景 + text(fr,t) 屏幕字层），SHOTS 表驱动切点，CUTS 表驱动转场，MOVE 表驱动运镜。"""
import math, os, sys, json
ROOT = '/workspace/work/mv'
for p in (ROOT, ROOT + '/lib', ROOT + '/units'):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, cv2
import c1kit as K
import typo as T
import trans as TR
import fx2
import props as PR
import paper as PA
import rest_sprites as RS
import e_kit as E
from e_kit import (W, H, BAR, GOLD, GOLDW, RED, REDT, WHITE_T, GOLD_T, INK, put, Lyr, row, sstep, c01, eio, eout, ein,
                   G_DAWN, G_HOT, G_SIL)

READY = False
FPS = 30
F0, F1 = 5797, 7241
LINES = T.load_lines()


def CT(li, k):
    """第 li 句第 k 个可见字的视觉帧时刻（字头最近帧再提前 1 帧）。"""
    ts = [x for x in LINES[li]['times'] if x is not None]
    return (round(ts[min(k, len(ts) - 1)] * FPS) - 1) / FPS


def cts(li, ks): return [CT(li, k) for k in ks]


HM = json.load(open(ROOT + '/crew/round4/E_hitmap.json'))
KICKS = [(h['t'], h['strength']) for h in HM['hits'] if 'kick' in h['type'] and h['strength'] > 0.5]


def kick_punch(t):
    p = 0.0
    for th, s in KICKS:
        if th - 1 / 30 <= t < th + 0.45: p = max(p, s * math.exp(-max(0.0, t - th + 1 / 30) / 0.11))
    return p


class Shot:
    def __init__(self): pass
    def u(self, t): return (t - self.t0) / (self.t1 - self.t0)
    def render(self, t): return np.zeros((H, W, 3), np.float32)
    def text(self, fr, t): return fr


def bloom(fr, thr=0.8, amt=0.5, sig=30): return K.bloom(fr, thr, amt, sig)


# =========================================================================================== 副歌终「定」
class E01(Shot):   # S111 ECU 刃口迎日：日光沿刃口扫过，巨字「血」火烧出
    def __init__(self):
        self.sky = E.dawn_sky(111, horizon=0.78, sun=(1560, 600), mid=(0.42, 0.29, 0.16), cloud=0.5)
        self.l1 = Lyr(row('血', [CT(48, 0)], 470, 520, size=440, red='血'), CT(48, 2) + 0.12, seed=1)
        self.l2 = Lyr(row('未冷', cts(48, [1, 2]), 760, 600, size=66, vertical=True, base=GOLD_T, track=0.4), CT(48, 2) + 0.15, seed=2)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (1560, 610), 70, amt=1.1)
        fr = fx2.god_rays(fr, (1560, 610), strength=0.5, length=0.8, t=t)
        # 刀（S118 交叉斩的刀刃特写，旋成斜向横贯画面），轻微滑动
        fr, info = put(fr, t, 'S118/xslash|crop=0.58,0.40,1,0.78', (980 - 120 * u, 610 + 20 * u), w=2500, anc=(0.5, 0.5),
                       rot=24, rim=GOLDW, rim_amt=3.2, rim_dir=(0.6, -0.8), grade=G_HOT, wind=0)
        sx = -200 + 2300 * eio(u * 1.1)                                   # 日光扫刃
        fr = E.flare_streak(fr, (sx, 640 - (sx - 980) * 0.42), amt=1.0, angle=-23)
        fr = E.dust_motes(fr, t, 111, n=70, amt=0.5, drift=(60, -10), depth=True)
        return bloom(fr, 0.75, 0.6)

    def text(self, fr, t):
        return self.l2.draw(self.l1.draw(fr, t), t)


class E02(Shot):   # S112 LS 低机位背影举刀指日：刀尖正对日心；「刀不放」朱印盖在日轮上
    def __init__(self):
        self.sky = E.dawn_sky(112, horizon=0.86, mid=(0.36, 0.25, 0.14), cloud=0.8, sun=(960, 300))
        self.rd, self.ra = E.ridges(n=2, base=0.84, step=0.05, seed=12, amp=0.06, far=0.25, near=0.02)
        self.l = Lyr(row('刀不放', cts(48, [3, 4, 5]), 1640, 520, size=72, vertical=True, red='刀', track=0.45), CT(48, 5) + 0.25, seed=3)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        tip = (960, 300)
        z = np.zeros((H, W, 3), np.float32)
        z, info = put(z, t, 'S112/sword_sky|flip', (960, 1180 + 30 * u), h=980, anc=(0.5, 1.0), rim=GOLDW, rim_amt=3.0, grade=G_SIL, wind=8)
        tip = info['top']
        sun = (tip[0], tip[1] - 10 - 40 * eio(u))                          # 日从刀尖升起
        fr = E.sun_disc(fr, sun, 150, amt=1.15)
        fr = fx2.god_rays(fr, sun, strength=0.9, length=1.0, t=t, occluder=info['A'])
        fr = E.place_layer(fr, self.rd, self.ra, oy=40 * u)
        a = info['A'][..., None]
        fr = fr * (1 - a) + z * a
        p = (t - (CT(48, 5) - 0.12)) / 0.5
        if p > 0: fr = T.stamp_seal(fr, '刀不放', sun[0] + 6, sun[1] + 4, 170, min(1, p), style='yin', rot=-6, seed=4, alpha=0.92)
        return bloom(fr, 0.8, 0.45)

    def text(self, fr, t): return self.l.draw(fr, t)


class E03(Shot):   # S113 ELS 金色云海快速横移，字悬在云层不同深度
    def __init__(self):
        self.sky = E.dawn_sky(113, horizon=0.62, sun=(420, 560), mid=(0.40, 0.28, 0.16), cloud=0.4)
        self.cl = [E.cloud_layer(130 + i, W * 2, H, density=0.35 + 0.15 * i) for i in range(3)]
        tm = cts(49, range(6))
        xs = [1500, 1270, 1000, 780, 540, 300]; ys = [430, 640, 380, 600, 470, 690]; ss = [92, 150, 70, 128, 84, 112]
        self.depth = [0.5, 1.0, 0.3, 0.8, 0.4, 0.9]
        self.ls = [Lyr([(c, xs[i], ys[i], ss[i], 'serif_sb', WHITE_T if i % 2 else GOLD_T, tm[i])], 198.2, seed=10 + i, glow=0.45)
                   for i, c in enumerate('世间从来没有')]

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (420, 560), 60, amt=1.0)
        for i, (lum, a) in enumerate(self.cl):
            sp = (0.35 + 0.5 * i) * 900
            ox = sp * (t - self.t0)
            col = (lum[..., None] * (GOLD * (0.55 + 0.3 * i)) + 0.03)
            yoff = 300 - 120 * i + 60 * eio(u)
            M = np.float32([[1, 0, -ox % W], [0, 1, yoff]])
            R = cv2.warpAffine(col, M, (W, H), borderMode=cv2.BORDER_WRAP); A = cv2.warpAffine(a, M, (W, H))
            A[:int(max(0, yoff))] = 0
            fr = fr * (1 - A[..., None]) + R * A[..., None]
        return bloom(fr, 0.8, 0.4)

    def text(self, fr, t):
        for l, d in zip(self.ls, self.depth):
            fr = l.draw(fr, t, dx=-(t - self.t0) * 900 * (0.35 + 0.5 * d) * 0.35 + 160)
        return fr


class E04(Shot):   # S114 CU 暗殿金匾「天理昭彰」：随唱逐字亮起；高窗光柱
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 140, base=3, octaves=5)
        slat = (np.sin(np.linspace(0, 60, W // 2)) * 0.5 + 0.5)[None, :] ** 6
        self.bg = cv2.resize(((0.03 + 0.05 * n + 0.025 * slat)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.pl = PR.Plaque('天理昭彰', seed=3)
        self.l = Lyr(row('什么', cts(49, [6, 7]), 250, 330, size=66, vertical=True, base=GOLD_T, track=0.45), 198.3, seed=20)
        self.lit_t = cts(49, [8, 9, 10, 11])

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.light_beam(fr, (300, 80), (1100, 1000), 60, 420, t=t, strength=0.55, seed=4)
        fr = fx2.light_beam(fr, (700, 60), (1500, 1000), 30, 260, t=t, strength=0.35, seed=5)
        fr = plaque_lit(self.pl, fr, t, 960, 520, 1300, self.lit_t, crack=0.0, dim=0.35)
        fr = E.dust_motes(fr, t, 114, n=120, amt=0.45, drift=(12, 8), depth=True)
        return fr

    def text(self, fr, t): return self.l.draw(fr, t)


def plaque_lit(pl, fr, t, cx, cy, w, lit_t, crack=0.0, dim=0.35, tilt=0.0, sheen_t=None):
    """金匾：底子压暗，按字头逐字点亮（字形发金光 + 扫光）。"""
    base = fr.copy()
    fr, _ = pl.draw(fr, t, cx, cy, w, crack=crack, tilt=tilt)
    fr = base + (fr - base) * (dim + (1 - dim) * 0.25)
    n = len(pl.text); h = w * 0.36; fs = min(h * 0.62, (w * 0.86) / n)
    L = T.Layer()
    for i, ch in enumerate(pl.text):
        e = (t - (lit_t[i] - 0.12)) / 0.18
        if e <= 0: continue
        x = cx + (i - (n - 1) / 2) * (w * 0.86 / n)
        if tilt:
            c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
            dx, dy = x - cx, 0
            x, y = cx + dx * c - dy * s, cy + dx * s + dy * c
        else: y = cy
        a = T.glyph(ch, 'song', fs, False)
        if tilt:
            g = a.shape[0]; M = cv2.getRotationMatrix2D((g / 2, g / 2), -tilt, 1.0); a = cv2.warpAffine(a, M, (g, g))
        fl = 1.0 + 1.2 * math.exp(-max(0, t - lit_t[i]) / 0.15)
        L.paste(a, x, y, (1.0, 0.82, 0.48), min(1, e) * fl * 0.9)
    return T.compose(fr, L, shadow=0, glow=0.9, glow_sigma=22, glow_color=(1.0, 0.6, 0.25))


class E05(Shot):   # S114b ECU 低角度荷兰角：昭、彰亮起即开裂，金光从裂缝透出，碎屑剥落
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 141, base=3, octaves=5)
        self.bg = cv2.resize(((0.02 + 0.04 * n)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.pl = PR.Plaque('天理昭彰', seed=8)
        self.lit_t = [0, 0] + cts(49, [10, 11])

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.light_beam(fr, (1700, 60), (900, 1000), 40, 380, t=t, strength=0.5, seed=7)
        ta, tb = CT(49, 10), CT(49, 11)
        crack = 0.0 if t < ta else (0.35 * eout((t - ta) / 0.3) if t < tb else 0.35 + 0.65 * eout((t - tb) / 0.35))
        fr = plaque_lit(self.pl, fr, t, 520, 560, 2700, [ta - 9, ta - 9, ta, tb], crack=crack, dim=0.45, tilt=-9)
        if t > tb:   # 金光爆出
            q = t - tb
            fr = E.flare_streak(fr, (1250, 470), amt=1.3 * math.exp(-q / 0.25), angle=-9, length=1.4)
            fr = RS.fx_sparks(fr, t, tb, (1250, 480), 1141, n=80, spd=1100, dirv=(0.3, -1), spread=1.6, amt=1.2)
        return fr


class E06(Shot):   # S115 MS 规则纸帘背光翻飞，推轨穿帘；「若规则是」烙在近处纸条上
    def __init__(self):
        self.sky = E.dawn_sky(115, horizon=0.7, sun=(960, 520), hor=(1.0, 0.86, 0.6), mid=(0.62, 0.48, 0.30), cloud=0.3)
        self.far = PA.PaperCurtain(seed=15, n=17, x_span=(60, 1860), top=90, length=(800, 950), width=(60, 95), text=True)
        self.near = PA.PaperCurtain(seed=16, n=5, x_span=(-100, 2000), top=60, length=(900, 1100), width=(190, 250), text=False)
        tm = cts(50, [0, 1, 2, 3])
        self.l = Lyr(row('若规则是', tm, 1395, 520, size=78, vertical=True, red='规则', base=(0.08, 0.06, 0.05), track=0.5),
                     CT(50, 4) - 0.05, seed=30, glow=0.0, shadow=0.0)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (960, 520), 80, amt=1.0)
        fr = self.far.draw(fr, t, wind=1.0, light_color=GOLD, backlit=1.1, depth_blur=1.0)
        fr = fx2.god_rays(fr, (960, 520), strength=0.5, length=0.9, t=t)
        fr = self.near.draw(fr, t * 1.3, wind=0.8, light_color=GOLD, backlit=0.7, depth_blur=2.5)
        # 题字纸条（固定在 x≈1395）
        q = np.zeros((H, W), np.float32)
        sw = 18 * math.sin(t * 1.7)
        pts = np.array([[1300 + sw, 120], [1490 + sw, 120], [1495 + sw * 1.6, 960], [1305 + sw * 1.6, 960]], np.int32)
        cv2.fillPoly(q, [pts], 1.0, cv2.LINE_AA)
        tex = E.paper_tex(150, (0.95, 0.88, 0.72))
        fr = fr * (1 - q[..., None]) + (tex * 1.05 + GOLD * 0.12) * q[..., None]
        return bloom(fr, 0.85, 0.35)

    def text(self, fr, t):
        sw = 18 * math.sin(t * 1.7)
        return self.l.draw(fr, t, dx=sw * 1.3)


class E07(Shot):   # S116 LS 纸帘缝后太上皇巨影 → 环绕揭示：原来是纸糊的
    def __init__(self):
        self.sky = E.dawn_sky(116, horizon=0.75, sun=(980, 420), hor=(1.0, 0.85, 0.58), mid=(0.70, 0.52, 0.30), cloud=0.2)
        self.cur = PA.PaperCurtain(seed=17, n=8, x_span=(-60, 1980), top=70, length=(900, 1000), width=(130, 190), text=True)
        self.l = Lyr(row('谎言', cts(50, [4, 5]), 960, 820, size=150, red='谎言', track=0.15), CT(50, 6) - 0.1, seed=40)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (980, 420), 120, amt=0.9)
        rv = eio((t - 200.95) / 0.5)
        dx = 90 - 170 * eio(u)
        if rv < 1:
            fr0, _ = put(fr.copy(), t, 'S116/emperor', (960 + dx, 1110), h=1040, rim=GOLDW, rim_amt=2.0, grade=G_SIL, look='sil', wind=2)
        if rv > 0:
            fr1, info = put(fr.copy(), t, 'S116/emperor', (960 + dx, 1110), h=1040, rim=GOLDW, rim_amt=3.0, grade=G_DAWN, look='paper', wind=2)
            fr1 = paper_frame(fr1, info['A'], t)
        fr = fr1 if rv >= 1 else (fr0 if rv <= 0 else fr0 * (1 - rv) + fr1 * rv)
        fr = fx2.god_rays(fr, (980, 420), strength=0.45, length=0.8, t=t)
        # 前景纸帘从两侧掠过（环绕视差）
        sh = -260 * eio(u) + 130
        M = np.float32([[1, 0, sh], [0, 1, 0]])
        z = np.zeros_like(fr); z = self.cur.draw(z, t, wind=1.0, light_color=GOLD, backlit=1.0, depth_blur=3.0)
        o = np.ones_like(fr); o = self.cur.draw(o, t, wind=1.0, light_color=GOLD, backlit=1.0, depth_blur=3.0)
        a = np.clip(1 - (o - z).mean(-1), 0, 1)
        side = np.clip(np.abs(np.arange(W, dtype=np.float32) - 960) / 260 - 1.2, 0, 1)[None, :]
        a = cv2.warpAffine(a, M, (W, H)) * side; z = cv2.warpAffine(z, M, (W, H))
        fr = fr * (1 - a[..., None]) + z * a[..., None] * 0.9
        return bloom(fr, 0.85, 0.35)

    def text(self, fr, t): return self.l.draw(fr, t)


def paper_frame(fr, A, t, seed=0):
    """纸糊揭示：剪影内叠竹骨架暗格 + 纸缝。"""
    yy, xx = E.grid()
    g = (np.abs(np.sin(xx / 61.0 + 0.3)) < 0.035) | (np.abs(np.sin(yy / 47.0)) < 0.03)
    d = (np.abs(np.sin((xx + yy) / 140.0)) < 0.02)
    m = (g | d).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.2) * A
    return fr * (1 - 0.55 * m[..., None])


class E08(Shot):   # S117 CU 纸糊巨脸自中心烧穿，背后空无一物（白金虚空）
    def __init__(self):
        r = np.random.default_rng(117)
        self.nz = cv2.resize(r.random((H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        self.nz2 = cv2.resize(r.random((H // 6, W // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        self.void = E.dawn_sky(118, horizon=0.4, top=(0.75, 0.62, 0.42), mid=(1.0, 0.86, 0.62), hor=(1.0, 0.95, 0.85), cloud=0.0)
        self.l = Lyr(row('伪装', cts(50, [6, 7]), 960, 860, size=150, red='伪装', track=0.15), CT(51, 0) - 0.12, seed=50)

    def render(self, t):
        u = self.u(t)
        bg = E.dawn_sky(119, horizon=0.8, mid=(0.5, 0.36, 0.2), cloud=0.0) if not hasattr(self, 'bg') else self.bg
        self.bg = bg
        fr, info = put(bg.copy(), t, 'S117/emperor_face', (960, 1190), h=1250 * (1 + 0.08 * u), rim=GOLDW, rim_amt=2.4, look='paper', wind=1)
        fr = paper_frame(fr, info['A'], t)
        tb = CT(50, 6) - 0.05
        prog = c01((t - tb) / 1.1)
        if prog > 0:
            yy, xx = E.grid()
            c = (960, 470)
            d = np.hypot(xx - c[0], (yy - c[1]) * 1.15) / 900
            field = d + (self.nz - 0.5) * 0.35 + (self.nz2 - 0.5) * 0.08
            R = eout(prog, 2) * 1.25
            keep = np.clip((field - R) / 0.015, 0, 1)
            ember = np.exp(-((field - R) / 0.02) ** 2)
            char = np.exp(-np.maximum(field - R, 0) / 0.05) * (field > R)
            A = info['A']
            burned = A * (1 - keep)
            fr = fr * (1 - 0.85 * char * A)[..., None]
            fr = fr * (1 - burned[..., None]) + self.void * 1.15 * burned[..., None]
            em = ember * A
            fr = fr + (em * 2.2 + cv2.GaussianBlur(em, (0, 0), 10) * 2.0)[..., None] * np.array([1.0, 0.55, 0.2], np.float32)
            fr = E.dust_motes(fr, t, 1170, n=120, amt=0.9 * prog, drift=(20, -160), col=(1.0, 0.6, 0.3), size=(1, 4))
        return bloom(fr, 0.8, 0.5)

    def text(self, fr, t): return self.l.draw(fr, t)


class E09(Shot):   # S118 MS 交叉斩：一刀斜劈，纸帘沿刀线断开；「那就撕碎」沿刀线排、被一分为二
    def __init__(self):
        self.sky = E.dawn_sky(1180, horizon=0.72, sun=(1500, 500), hor=(1.0, 0.84, 0.55), mid=(0.55, 0.40, 0.24), cloud=0.3)
        self.cur = PA.PaperCurtain(seed=18, n=13, x_span=(60, 1860), top=60, length=(900, 1000), width=(90, 130), text=True)
        ts = cts(51, [0, 1, 2, 3])
        self.ts = CT(51, 2)
        p0, p1 = np.array([300.0, 300.0]), np.array([1700.0, 780.0])
        self.p0, self.p1 = p0, p1
        pos = [p0 + (p1 - p0) * f for f in (0.18, 0.36, 0.60, 0.80)]
        nrm = np.array([0.32, -0.95])
        cols = [WHITE_T, WHITE_T, REDT, REDT]
        self.la = Lyr([(c, *(pos[i] + nrm * 70), 120, 'serif_black', cols[i], ts[i]) for i, c in enumerate('那就撕碎')], CT(51, 4) - 0.1, seed=60)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (1500, 500), 70, amt=1.0)
        ts = self.ts
        cuts = [(tuple(self.p0 - (self.p1 - self.p0)), tuple(self.p1 + (self.p1 - self.p0)), ts)]
        fr = self.cur.draw(fr, t, wind=1.2, light_color=GOLD, backlit=1.0, cuts=cuts, depth_blur=1.5)
        pose_mb = (60, -30) if t < ts + 0.05 else (0, 0)
        fr, info = put(fr, t, 'S118/xslash', (900 - 40 * u, 1060), h=820, rim=GOLDW, rim_amt=2.8, wind=10, mblur=pose_mb)
        if t >= ts - 0.03:
            q = t - ts
            c = (900, 560)
            fr = RS.fx_arc(fr, c, 760, 30, 120, min(1, (q + 0.03) / 0.07), max(0, 1 - q / 0.35), w=16, col=(1.0, 0.95, 0.85), edge=(0.85, 0.5, 0.15), squash=0.55)
            if q < 0.1: fr = fr + (0.35 * (1 - q / 0.1))
        return bloom(fr, 0.8, 0.45)

    def text(self, fr, t):
        tb = CT(51, 3)
        if t < tb: return self.la.draw(fr, t)
        # 「碎」：沿刀线把整句一分为二，上半往左上、下半往右下错开
        q = eout((t - tb) / 0.25)
        d = (self.p1 - self.p0) / np.linalg.norm(self.p1 - self.p0)
        F1 = self.la.draw(fr.copy(), t, dx=-d[0] * 50 * q - 10 * q, dy=-d[1] * 50 * q - 30 * q)
        F2 = self.la.draw(fr.copy(), t, dx=d[0] * 50 * q + 10 * q, dy=d[1] * 50 * q + 30 * q)
        yy, xx = E.grid()
        side = ((xx - self.p0[0]) * d[1] - (yy - self.p0[1]) * d[0] + 70 > 0)[..., None]
        out = np.where(side, F2, F1)
        g = np.exp(-(((xx - self.p0[0]) * d[1] - (yy - self.p0[1]) * d[0] + 70) / 3) ** 2) * (1 - q * 0.8)
        return out + g[..., None] * GOLDW * 1.2


class E10(Shot):   # S119 碎纸慢放扑向镜头，日前剪影；字写在飞纸上
    def __init__(self):
        self.sky = E.dawn_sky(1190, horizon=0.78, sun=(960, 560), hor=(1.0, 0.86, 0.6), mid=(0.5, 0.36, 0.2), cloud=0.5)
        self.sh = PA.PaperShreds(119, n=110, size=(26, 90))
        self.l = Lyr([('它', 560, 380, 170, 'serif_black', WHITE_T, CT(51, 4)), ('自', 1380, 330, 120, 'serif_black', GOLD_T, CT(51, 5)),
                      ('己', 1420, 520, 120, 'serif_black', GOLD_T, CT(51, 6))], CT(51, 7) - 0.12, seed=70, glow=0.5)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (960, 560), 210, amt=1.15)
        fr = fx2.god_rays(fr, (960, 560), strength=0.8, length=1.0, t=t)
        fr, info = put(fr, t, 'S120/plant', (960, 1000), h=560, rim=GOLDW, rim_amt=3.5, grade=G_SIL, look='sil', wind=4)
        ts = (t - self.t0) * 0.35                                          # 慢放
        fr = self.sh.draw(fr, ts + 0.5, t0=0.0, origin=(960, 500), spread=(1100, 600), to_camera=1.0, gravity=60, wind=(30, 0), burn=0.15)
        return bloom(fr, 0.8, 0.5)

    def text(self, fr, t):
        z = 1 + 0.25 * self.u(t)
        return self.l.draw(fr, t, dx=0, dy=0)


class E11(Shot):   # S120a 低机位 MS：刀插大地，尘环冲击；巨字「定」
    def __init__(self):
        self.sky = E.dawn_sky(1200, horizon=0.80, sun=(1500, 700), mid=(0.40, 0.28, 0.15), cloud=0.6)
        n = K.fbm(H // 2, W // 2, 1201, base=8, octaves=6)
        yy = np.linspace(0, 1, H // 2, dtype=np.float32)[:, None]
        g = (0.04 + 0.10 * n) * (0.6 + 0.8 * yy)
        self.ground = cv2.resize((g[..., None] * np.array([1.0, 0.82, 0.62], np.float32)).astype(np.float32), (W, H))
        self.l1 = Lyr([('定', 430, 520, 440, 'serif_black', REDT, CT(51, 7))], CT(51, 9) + 0.25, seed=80)
        self.l2 = Lyr([('规', 430, 820, 72, 'serif_sb', GOLD_T, CT(51, 8))], CT(51, 9) + 0.25, seed=81)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        hy = 820
        m = sstep(hy - 4, hy + 4, np.arange(H, dtype=np.float32))[:, None, None]
        fr = fr * (1 - m) + self.ground * m
        fr = E.sun_disc(fr, (1500, hy - 20), 90, amt=1.0)
        fr, info = put(fr, t, 'S120/plant', (1180, 1010), h=860, rim=GOLDW, rim_amt=2.6, rim_dir=(0.8, -0.6), wind=8)
        q = t - self.t0
        c = (1180, 1000)
        r = 60 + 900 * eout(q / 0.7)
        yy, xx = E.grid()
        ring = np.exp(-((np.hypot(xx - c[0], (yy - c[1]) * 4) - r) / 30) ** 2) * (1 - c01(q / 0.8))
        fr = fr + ring[..., None] * np.array([0.9, 0.7, 0.45], np.float32) * 0.5
        fr = RS.fx_sparks(fr, t, self.t0, c, 1202, n=70, spd=800, dirv=(0, -1), spread=1.3, amt=1.0)
        fr = E.dust_motes(fr, t, 1203, n=60, amt=0.4, drift=(80, -20))
        return bloom(fr, 0.8, 0.4)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E12(Shot):   # S120b ELS 正俯拍：大地即印面，「鬼屠」朱印以刀为中心盖下（摇臂升）
    def __init__(self):
        n = K.fbm(H, W, 1210, base=6, octaves=6)
        cr = K.fbm(H, W, 1211, base=22, octaves=3)
        crack = np.clip(1 - np.abs(cr - 0.5) * 40, 0, 1)
        g = 0.10 + 0.16 * n - crack * 0.06
        self.ground = (g[..., None] * np.array([1.0, 0.80, 0.58], np.float32)).astype(np.float32)
        self.face = PR.seal_face('鬼屠', 700, style='bai', wear=0.25, seed=12)
        self.cap = Lyr(row('自己定规章', [CT(51, 5), CT(51, 6), CT(51, 7), CT(51, 8), CT(51, 9)], 960, 900, size=62, red='自己'),
                       CT(52, 0) - 0.15, seed=90)

    def render(self, t):
        u = self.u(t)
        fr = self.ground.copy()
        yy, xx = E.grid()
        fr = fr * (0.55 + 0.6 * np.exp(-((xx - 1300) ** 2 + (yy - 200) ** 2) / (2 * 700 ** 2)))[..., None]   # 晨光斜照
        p = (t - (CT(51, 9) - 0.18)) / 0.7
        if p > 0: fr = PR.stamp_imprint(fr, self.face, 960, 540, 700, min(1, p), angle=-4)
        # 刀与人（俯视极小）：人影 + 拉长的晨影
        cv2.line(fr, (960, 540), (760, 690), (0.02, 0.02, 0.02), 14, cv2.LINE_AA)
        cv2.circle(fr, (960, 540), 13, (0.03, 0.025, 0.02), -1, cv2.LINE_AA)
        cv2.line(fr, (948, 536), (975, 548), (0.9, 0.85, 0.75), 2, cv2.LINE_AA)
        if p > 0.3:
            q = (p - 0.3) / 0.7
            r = 380 + 600 * eout(q)
            ring = np.exp(-((np.hypot(xx - 960, yy - 540) - r) / 40) ** 2) * (1 - q)
            fr = fr + ring[..., None] * np.array([0.8, 0.5, 0.3], np.float32) * 0.6
        return fr

    def text(self, fr, t): return self.cap.draw(fr, t)


class E13(Shot):   # S121 MCU 晨风中马尾与暗红腰带甩动，背景日光虚焦；巨字「血」
    def __init__(self):
        sky = E.dawn_sky(1210, horizon=0.75, sun=(1500, 380), mid=(0.45, 0.32, 0.18), cloud=0.7)
        sky = E.sun_disc(sky, (1500, 380), 120, amt=1.0)
        self.bg = cv2.GaussianBlur(sky, (0, 0), 18)
        r = np.random.default_rng(121)
        self.bok = [(r.uniform(900, 1900), r.uniform(150, 900), r.uniform(25, 70), r.uniform(0.2, 0.6)) for _ in range(18)]
        self.l1 = Lyr([('血', 440, 520, 440, 'serif_black', REDT, CT(52, 0))], CT(52, 3) - 0.1, seed=100)
        self.l2 = Lyr(row('未冷', cts(52, [1, 2]), 740, 610, size=66, vertical=True, base=GOLD_T, track=0.4), CT(52, 3) - 0.1, seed=101)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        L = np.zeros((H // 2, W // 2), np.float32)
        for (x, y, r, a) in self.bok:
            cv2.circle(L, (int((x - 40 * u) / 2), int(y / 2)), int(r / 2), a, -1, cv2.LINE_AA)
        fr = fr + cv2.resize(cv2.GaussianBlur(L, (0, 0), 2), (W, H))[..., None] * GOLDW * 0.6
        fr, info = put(fr, t, 'S121/wind_sash', (1180 + 30 * u, 1160), h=1080, rim=GOLDW, rim_amt=3.2, rim_dir=(0.8, -0.6), wind=14)
        fr = RS.fx_embers(fr, t, 1213, amt=0.8, n=50)
        return bloom(fr, 0.8, 0.4)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E14(Shot):   # S122 LS 他逆光朝镜头走来，灰烬与焦纸在晨风中飘；拉轨
    def __init__(self):
        self.sky = E.dawn_sky(1220, horizon=0.66, sun=(960, 300), hor=(1.0, 0.84, 0.55), mid=(0.6, 0.42, 0.24), cloud=0.4)
        self.rd, self.ra = E.ridges(n=3, base=0.60, step=0.05, seed=122, amp=0.05, far=0.5, near=0.08)
        n = K.fbm(H // 2, W // 2, 1222, base=10, octaves=5)
        self.ground = cv2.resize(((0.06 + 0.10 * n)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.sh = PA.PaperShreds(122, n=40, size=(14, 40))

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (960, 300), 110, amt=1.1)
        fr = E.place_layer(fr, self.rd, self.ra)
        hy = 760
        m = sstep(hy - 3, hy + 40, np.arange(H, dtype=np.float32))[:, None, None]
        fr = fr * (1 - m) + self.ground * m
        fr = fx2.god_rays(fr, (960, 300), strength=0.7, length=1.0, t=t)
        bob = 6 * abs(math.sin(t * 5.5))
        fr, info = put(fr, t, 'S122/walk_toward', (960, 1000 - bob), h=740, rim=GOLDW, rim_amt=3.0, wind=10)
        sh = info['A']   # 长影朝镜头
        fr = self.sh.draw(fr, t, t0=self.t0 - 1.5, origin=(960, 300), spread=(1200, 500), to_camera=0.4, gravity=40, wind=(-40, 0), burn=0.6)
        fr = RS.fx_embers(fr, t, 1223, amt=0.7, n=60)
        return bloom(fr, 0.8, 0.45)


L_GU = None
def lyr_gu():
    global L_GU
    if L_GU is None:
        L_GU = Lyr(row('骨未凉', cts(52, [3, 4, 5]), 960, 860, size=78, red='骨', track=0.3), CT(53, 0) - 0.15, seed=110)
    return L_GU


class E14T(E14):
    def text(self, fr, t): return lyr_gu().draw(fr, t)


class E15(Shot):   # S122b ECU 刀尖拖过焦土，划出一道朱金火线（他自己画的线）
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1230, base=12, octaves=6)
        cr = K.fbm(H // 2, W // 2, 1231, base=30, octaves=3)
        g = 0.05 + 0.13 * n - np.clip(1 - np.abs(cr - 0.5) * 30, 0, 1) * 0.04
        yy = np.linspace(0, 1, H // 2, dtype=np.float32)[:, None]
        g = g * (0.5 + 0.9 * yy)
        self.ground = cv2.resize((g[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))

    def render(self, t):
        u = self.u(t)
        fr = self.ground.copy()
        yy, xx = E.grid()
        fr = fr * (0.7 + 0.6 * np.exp(-((yy - 150) / 500) ** 2))[..., None]
        x = -100 + 1900 * eio(0.15 + u * 0.85)
        ly = 640 + (xx - x) * 0.04
        on = (xx < x)
        line = np.exp(-((yy - ly) / 4) ** 2) * on
        glow = np.exp(-((yy - ly) / 26) ** 2) * on * np.exp(-np.maximum(x - xx, 0) / 900)
        fr = fr * (1 - 0.6 * np.exp(-((yy - ly) / 7) ** 2) * on)[..., None]
        fr = fr + (line * 1.5)[..., None] * np.array([1.0, 0.45, 0.15], np.float32) + glow[..., None] * np.array([1.0, 0.35, 0.1], np.float32) * 0.9
        fr, info = put(fr, t, 'S122/walk_toward|crop=0.26,0.62,0.40,0.92', (x + 20, 645), h=760, anc=(0.06, 0.96), rot=62, rim=GOLDW, rim_amt=2.5, wind=0)
        fr = RS.fx_sparks(fr, t, t - 0.3, (x, 642), int(t * 30) % 50 + 1230, n=40, spd=700, dirv=(-1, -0.6), spread=0.8, grav=1200, amt=1.2)
        return bloom(fr, 0.75, 0.5)

    def text(self, fr, t): return lyr_gu().draw(fr, t)


# ---- 对 / 错：竖分屏镜像（左金右墨 → 左墨右朱）
def half_field(seed, col, sun=None):
    sky = E.dawn_sky(seed, horizon=0.82, top=tuple(np.array(col) * 0.12), mid=tuple(np.array(col) * 0.5), hor=tuple(col), cloud=0.5, sun=sun)
    return sky


class E16(Shot):   # S123 MCU 左半金光中正面；右半墨底金色巨字「对」
    def __init__(self):
        self.gold = half_field(1230, (1.0, 0.80, 0.48), sun=(480, 700))
        n = K.fbm(H // 2, W // 2, 1232, base=5, octaves=5)
        self.ink = cv2.resize(((0.02 + 0.05 * n)[..., None] * np.array([1.0, 0.85, 0.7], np.float32)).astype(np.float32), (W, H))
        self.l1 = Lyr([('对', 1440, 500, 470, 'serif_black', GOLD_T, CT(53, 0))], CT(53, 4) - 0.1, seed=120, glow=0.6)
        self.l2 = Lyr(row('就是对', cts(53, [1, 2, 3]), 1440, 860, size=62, base=WHITE_T, track=0.3), CT(53, 4) - 0.1, seed=121)

    def render(self, t):
        u = self.u(t)
        left = self.gold.copy()
        left = fx2.god_rays(left, (480, 700), strength=0.6, length=1.0, t=t)
        left, info = put(left, t, 'S123/front', (500 + 60 * u, 1140), h=900, rim=GOLDW, rim_amt=2.6, grade=G_DAWN, wind=5)
        x = np.arange(W)[None, :, None]
        fr = np.where(x < 960, left, self.ink)
        g = np.exp(-((np.arange(W, dtype=np.float32) - 960) / 3) ** 2)[None, :, None]
        return fr + g * GOLDW * 1.0

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E17(Shot):   # S124 MCU 镜像：右半朱光中回头冷视；左半墨底朱红巨字「错」
    def __init__(self):
        self.red = half_field(1240, (0.85, 0.22, 0.12), sun=(1440, 720))
        n = K.fbm(H // 2, W // 2, 1242, base=5, octaves=5)
        self.ink = cv2.resize(((0.02 + 0.05 * n)[..., None] * np.array([1.0, 0.8, 0.7], np.float32)).astype(np.float32), (W, H))
        self.l1 = Lyr([('错', 480, 500, 470, 'serif_black', REDT, CT(53, 4))], CT(53, 8) - 0.1, seed=130, glow=0.5)
        self.l2 = Lyr(row('就是错', cts(53, [5, 6, 7]), 480, 860, size=62, base=WHITE_T, track=0.3), CT(53, 8) - 0.1, seed=131)

    def render(self, t):
        u = self.u(t)
        right = self.red.copy()
        right, info = put(right, t, 'S124/over_shoulder', (1430 - 60 * u, 1150), h=920, rim=(1.0, 0.45, 0.3), rim_amt=2.6,
                          grade=dict(expo=0.8, sat=0.6, tint=(1.08, 0.93, 0.88), lift=0.0, gamma=1.12), wind=5)
        x = np.arange(W)[None, :, None]
        fr = np.where(x >= 960, right, self.ink)
        g = np.exp(-((np.arange(W, dtype=np.float32) - 960) / 3) ** 2)[None, :, None]
        return fr + g * np.array([1.0, 0.5, 0.35], np.float32)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E18(Shot):   # S124b 对/错 同框：两半同时在画，中缝锯齿裂开、金光喷出（下一镜碎裂）
    def __init__(self):
        self.a = E16(); self.b = E17()
        self.a.t0, self.a.t1 = 210.33, 211.27; self.b.t0, self.b.t1 = 211.27, 212.10
        r = np.random.default_rng(124)
        ys = np.linspace(BAR, H - BAR, 14); self.zz = np.stack([960 + r.normal(0, 26, 14), ys], 1)

    def render(self, t):
        u = self.u(t)
        A = self.a.text(self.a.render(211.2), 211.2)
        B = self.b.text(self.b.render(212.05), 212.05)
        A = K.cam(A, 1.0, ox=480 - 40 * u); B = K.cam(B, 1.0, ox=-480 + 40 * u)
        x = np.arange(W)[None, :, None]
        fr = np.where(x < 960, A, B)
        m = np.zeros((H, W), np.float32)
        pts = self.zz.copy(); pts[:, 0] = 960 + (pts[:, 0] - 960) * (0.4 + 2.5 * u)
        cv2.polylines(m, [pts.astype(np.int32)], False, 1.0, 3 + int(14 * u), cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 2)
        fr = fr * (1 - m[..., None]) + m[..., None] * GOLDW * 2.0
        return fr + cv2.GaussianBlur(m, (0, 0), 25)[..., None] * GOLD * 1.5 * u


class E19(Shot):   # S125 CU 直视镜头；金匾碎片在他周围漂浮；「何须伪装」灰飞
    def __init__(self):
        sky = E.dawn_sky(1250, horizon=0.9, mid=(0.30, 0.20, 0.11), hor=(0.9, 0.66, 0.36), cloud=0.6, sun=(1500, 900))
        self.bg = cv2.GaussianBlur(sky, (0, 0), 10)
        r = np.random.default_rng(125)
        self.chips = [(r.uniform(100, 1820), r.uniform(150, 930), r.uniform(10, 34), r.uniform(0, 6.28), r.uniform(-1, 1), r.uniform(0.3, 1)) for _ in range(40)]
        self.l = Lyr(row('何须伪装', cts(53, [8, 9, 10, 11]), 1700, 520, size=76, vertical=True, red='伪装', track=0.45), CT(53, 11) + 0.25, seed=140)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr, info = put(fr, t, 'S125/stare', (960, 1170), h=1180 * (1 + 0.06 * u), rim=GOLDW, rim_amt=2.4, wind=6)
        L = np.zeros((H, W), np.float32); Lb = np.zeros((H, W), np.float32)
        for (x, y, s, a, v, d) in self.chips:
            px = x + v * 60 * u; py = y - 40 * u * d
            ang = a + v * 2 * (t - self.t0)
            pts = np.array([[math.cos(ang + k * 2.1) * s * (1 + 0.4 * (k % 2)), math.sin(ang + k * 2.1) * s * 0.7] for k in range(3)]) + [px, py]
            cv2.fillPoly(L if d > 0.6 else Lb, [pts.astype(np.int32)], float(0.4 + 0.6 * abs(math.sin(ang * 3))), cv2.LINE_AA)
        Lb = cv2.GaussianBlur(Lb, (0, 0), 5)
        fr = fr + (L + Lb * 0.8)[..., None] * GOLD * 0.9
        return bloom(fr, 0.8, 0.4)

    def text(self, fr, t): return self.l.draw(fr, t)


class E20(Shot):   # S126 ELS 巨大日轮占满画面；「天」字在日中被烧化
    def __init__(self):
        self.tex = K.fbm(H // 2, W // 2, 1260, base=24, octaves=5)
        self.tex2 = K.fbm(H // 2, W // 2, 1261, base=60, octaves=3)
        self.sky = E.dawn_sky(1262, horizon=0.95, mid=(0.5, 0.33, 0.16), hor=(1.0, 0.7, 0.35), cloud=0.7)
        self.l = Lyr(row('不', [CT(54, 1)], 960, 905, size=62, base=GOLD_T), CT(54, 2) + 0.1, seed=150)
        self.tt = CT(54, 0)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        c = (960, 520); R = 360 * (1 + 0.06 * u)
        yy, xx = E.grid()
        d = np.hypot(xx - c[0], yy - c[1])
        disc = np.clip((R - d) / 2.5, 0, 1)
        k = cv2.resize(self.tex * 0.6 + self.tex2 * 0.4, (W, H))
        k = np.roll(k, int(30 * (t - self.t0)), 1)
        limb = np.clip(1 - (d / R) ** 2, 0, 1) ** 0.35
        surf = (0.75 + 0.5 * k) * limb
        col = np.stack([surf * 1.25, surf * 0.95, surf * 0.55], -1)
        fr = fr * (1 - disc[..., None]) + col * disc[..., None]
        halo = np.exp(-np.maximum(d - R, 0) / 160) * (d > R)
        fr = fr + halo[..., None] * GOLD * 0.9
        fr = fx2.god_rays(fr, c, strength=0.6, length=0.9, t=t, src_r=R)
        # 天：日面上的墨色剪影字 → 被烧化（热扭曲 + 自下而上烧穿 + 熔滴）
        e = c01((t - (self.tt - 0.2)) / 0.2)
        if e > 0:
            a = T.glyph('天', 'serif_black', 520, False).copy()
            g = a.shape[0]
            q = c01((t - (self.tt + 0.25)) / 0.55)
            if q > 0:
                ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
                wob = np.sin(ys / 9 + t * 25) * 6 * q
                drip = (np.sin(xs / 23) * 0.5 + 0.5) ** 4 * 60 * q
                a = cv2.remap(a, xs + wob, ys - drip, cv2.INTER_LINEAR)
                nz = E._nz(g, 7)
                a = a * np.clip(((1 - ys / g) * 0.7 + nz * 0.3 - q * 1.1) * 6, 0, 1)
            L = T.Layer(); L.paste(a, c[0], c[1] + 10, (0.06, 0.03, 0.02), e)
            ember = cv2.GaussianBlur(L.A, (0, 0), 3) - L.A
            fr = fr * (1 - L.A[..., None] * 0.92) + np.clip(ember, 0, 1)[..., None] * np.array([1.0, 0.4, 0.1], np.float32) * 2.0 * (0.4 + q)
        return bloom(fr, 0.85, 0.55)

    def text(self, fr, t): return self.l.draw(fr, t)


class E21(Shot):   # S127 CU 仰望侧脸，晨光勾金边（呼应片头月下侧脸）；「判」「我」
    def __init__(self):
        sky = E.dawn_sky(1270, horizon=0.7, sun=(1700, 260), mid=(0.42, 0.30, 0.17), cloud=0.6)
        sky = E.sun_disc(sky, (1700, 260), 90, amt=1.0)
        self.bg = cv2.GaussianBlur(sky, (0, 0), 14)
        self.l1 = Lyr([('判', 1460, 300, 150, 'serif_black', REDT, CT(54, 2))], CT(54, 4) - 0.1, seed=160, glow=0.4)
        self.l2 = Lyr([('我', 1460, 640, 420, 'serif_black', WHITE_T, CT(54, 3))], CT(54, 4) - 0.1, seed=161, glow=0.45)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.god_rays(fr, (1700, 260), strength=0.6, length=1.0, t=t)
        fr, info = put(fr, t, 'S127/profile_up', (560 + 40 * u, 1180), h=1120, rim=GOLDW, rim_amt=3.4, rim_dir=(1.0, -0.4), wind=10)
        fr = E.dust_motes(fr, t, 1271, n=80, amt=0.5, drift=(-20, -30), depth=True)
        return bloom(fr, 0.8, 0.45)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E22(Shot):   # S128 LS 横刀于胸，太阳恰落在刀身正中——他自己成了秤；「来量」刻在刀上
    def __init__(self):
        self.sky = E.dawn_sky(1280, horizon=0.86, mid=(0.42, 0.30, 0.16), cloud=0.6, sun=(960, 360))
        self.rd, self.ra = E.ridges(n=2, base=0.82, step=0.05, seed=128, amp=0.05, far=0.35, near=0.03)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        z = np.zeros((H, W, 3), np.float32)
        z, info = put(z, t, 'S128/blade_level', (960, 1060 - 20 * u), h=900, rim=GOLDW, rim_amt=3.0, grade=dict(expo=0.55, sat=0.55, tint=(1.04, 0.97, 0.9), lift=0.0, gamma=1.2), wind=6)
        bx0, by0 = info['box'][0]; bx1, by1 = info['box'][1]
        self.blade_y = by0 + 0.205 * (by1 - by0)
        sun = (960, self.blade_y - 4)
        fr = E.sun_disc(fr, sun, 120, amt=1.15)
        fr = fx2.god_rays(fr, sun, strength=1.0, length=1.0, t=t, occluder=info['A'])
        fr = E.place_layer(fr, self.rd, self.ra)
        a = info['A'][..., None]
        fr = fr * (1 - a) + z * a
        return bloom(fr, 0.8, 0.5)

    def text(self, fr, t):
        y = getattr(self, 'blade_y', 300)
        l = Lyr([('来', 760, y + 2, 46, 'serif_b', GOLD_T, CT(54, 4)), ('量', 1160, y + 2, 46, 'serif_b', REDT, CT(54, 5))], CT(55, 0) - 0.1, seed=170, glow=0.8, shadow=0.3)
        fr = l.draw(fr, t)
        big = Lyr(row('我来量', [CT(54, 3) - 5, CT(54, 4), CT(54, 5)], 960, 880, size=66, red='量', track=0.3), CT(55, 0) - 0.1, seed=171)
        return big.draw(fr, t)


class E23(Shot):   # S129 MCU 后撤蓄力，速度坡度推近（吸气）
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1290, base=4, octaves=5)
        self.bg = cv2.resize(((0.10 + 0.18 * n)[..., None] * np.array([1.0, 0.75, 0.45], np.float32)).astype(np.float32), (W, H))
        self.l = Lyr(row('这一', cts(55, [0, 1]), 1700, 420, size=70, vertical=True, base=WHITE_T, track=0.45), CT(55, 2) - 0.08, seed=180)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        yy, xx = E.grid()
        st = (np.sin(yy / 7 + xx / 300 + t * 40) > 0.97).astype(np.float32)          # 风线
        fr = fr + cv2.GaussianBlur(st, (21, 1), 0)[..., None] * GOLD * 0.15
        fr, info = put(fr, t, 'S129/coil', (1000, 1140), h=1080, rim=GOLDW, rim_amt=2.8, wind=12)
        return bloom(fr, 0.8, 0.35)

    def text(self, fr, t): return self.l.draw(fr, t)


class E24(Shot):   # S129b ECU 刃身满屏，寒光自柄扫到尖；巨字「刀」朱红
    def __init__(self):
        self.bg = E.dawn_sky(1291, horizon=0.5, top=(0.06, 0.05, 0.04), mid=(0.25, 0.18, 0.1), hor=(0.6, 0.42, 0.22), cloud=0.3)
        self.l1 = Lyr([('刀', 1440, 520, 440, 'serif_black', REDT, CT(55, 2))], CT(55, 6) - 0.05, seed=190)
        self.l2 = Lyr(row('就是我', cts(55, [3, 4, 5]), 620, 860, size=66, base=WHITE_T, track=0.3), CT(55, 6) - 0.05, seed=191)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr, info = put(fr, t, 'S129/coil|crop=0,0,0.46,0.42', (900 - 60 * u, 470), w=2300, anc=(0.5, 0.5), rot=-12, rim=GOLDW, rim_amt=3.0, wind=0, grade=G_HOT)
        sx = -300 + 2600 * eio(u)
        fr = E.flare_streak(fr, (sx, 420 + (sx - 960) * 0.18), amt=1.1, angle=10)
        return bloom(fr, 0.78, 0.5)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E25(Shot):   # S130 LS 终极横斩：整个画面沿刀线裂开，金光喷涌；「的」「正」
    def __init__(self):
        self.sky = E.dawn_sky(1300, horizon=0.80, sun=(1500, 640), mid=(0.40, 0.28, 0.15), cloud=0.7)
        self.rd, self.ra = E.ridges(n=3, base=0.74, step=0.05, seed=130, amp=0.07, far=0.45, near=0.03)
        self.l = Lyr([('的', 560, 300, 70, 'serif_sb', WHITE_T, CT(55, 6)), ('正', 960, 520, 300, 'serif_black', GOLD_T, CT(55, 7))], CT(55, 8) - 0.12, seed=200, glow=0.7)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (1500, 640), 80, amt=1.0)
        fr = E.place_layer(fr, self.rd, self.ra)
        ts = CT(55, 6)
        mb = (90, 0) if t < ts + 0.04 else (0, 0)
        fr, info = put(fr, t, 'S130/hslash', (1000, 1020), h=760, rim=GOLDW, rim_amt=2.8, wind=12, mblur=mb)
        self.y = by = info['tip'][1] if info['tip'] else 520
        q = t - ts
        if q > 0:
            e = eout(q / 0.25)
            gap = 70 * e * (1 - 0.3 * c01((q - 0.6) / 0.5))
            top = K.cam(fr, 1.0, oy=-gap / 2); bot = K.cam(fr, 1.0, oy=gap / 2)
            yy = np.arange(H, dtype=np.float32)[:, None, None]
            fr = np.where(yy < by, top, bot)
            band = np.exp(-((yy[..., 0] - by) / max(2, gap * 0.6)) ** 2)
            fr = fr * (1 - band[..., None] * 0.9) + band[..., None] * GOLDW * 2.2
            fr = fr + np.exp(-((yy - by) / (gap + 40)) ** 2) * GOLD * 0.9 * (1 + 2 * math.exp(-q / 0.12))
            if q < 0.12: fr = fr + 0.5 * (1 - q / 0.12)
            fr = RS.fx_sparks(fr, t, ts, (300, by), 1301, n=90, spd=1400, dirv=(1, -0.1), spread=0.35, grav=300, amt=1.2)
        return bloom(fr, 0.8, 0.6)

    def text(self, fr, t): return self.l.draw(fr, t)


class E26(Shot):   # S131 ELS 刀线化为地平线，朝阳从他画的线上升起；「正道沧桑」
    def __init__(self):
        self.sky = E.dawn_sky(1310, horizon=0.60, mid=(0.45, 0.31, 0.17), hor=(1.0, 0.82, 0.5), cloud=0.8, sun=(960, 640))
        n = K.fbm(H // 2, W // 2, 1311, base=8, octaves=5)
        self.ground = cv2.resize(((0.03 + 0.06 * n)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.l = Lyr(row('正道沧桑', [CT(55, 7), CT(55, 8), CT(55, 9), CT(55, 10)], 960, 300, size=150, gold='正道', track=0.25),
                     None, seed=210, glow=0.6)

    def render(self, t):
        u = self.u(t)
        hy = 640
        fr = self.sky.copy()
        sun = (960, hy + 120 - 200 * eout(u, 2))
        fr = E.sun_disc(fr, sun, 150, amt=1.2)
        fr = fx2.god_rays(fr, sun, strength=0.7, length=1.0, t=t)
        m = sstep(hy - 1, hy + 2, np.arange(H, dtype=np.float32))[:, None, None]
        fr = fr * (1 - m) + self.ground * m
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        fr = fr + np.exp(-((yy - hy) / 2.5) ** 2) * GOLDW * 2.0 + np.exp(-((yy - hy) / 30) ** 2) * GOLD * 0.6
        fr, info = put(fr, t, 'S130/hslash', (640, hy + 2), h=150, rim=GOLDW, rim_amt=3.5, grade=G_SIL, look='sil', wind=8)
        return bloom(fr, 0.85, 0.5)

    def text(self, fr, t): return self.l.draw(fr, t)


# =========================================================================================== 尾奏
class E27(Shot):   # S132 LS 背影沿山脊走向朝阳（呼应片头构图，已是白天）
    def __init__(self):
        self.sky = E.dawn_sky(1320, horizon=0.66, mid=(0.48, 0.34, 0.19), hor=(1.0, 0.84, 0.55), cloud=0.7, sun=(1450, 520))
        self.rd, self.ra = E.ridges(n=4, base=0.56, step=0.06, seed=132, amp=0.09, far=0.6, near=0.04)
        r = np.random.default_rng(132)
        self.grass = [(r.uniform(0, W), r.uniform(0.6, 1.4), r.uniform(-0.4, 0.4)) for _ in range(40)]

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (1450, 520 - 30 * u), 100, amt=1.1)
        fr = E.place_layer(fr, self.rd, self.ra, oy=-30 * u)
        fr = fx2.god_rays(fr, (1450, 520), strength=0.6, length=1.0, t=t)
        bob = 4 * abs(math.sin(t * 5))
        fr, info = put(fr, t, 'S132/back_walk', (1080 + 90 * u, 930 - 40 * u - bob), h=360 * (1 - 0.1 * u), rim=GOLDW, rim_amt=3.0, wind=8)
        G = np.zeros((H // 2, W // 2), np.float32)
        for (x, s, lean) in self.grass:
            for k in range(5):
                x0 = (x + k * 9 - 120 * u) / 2
                cv2.line(G, (int(x0), H // 2), (int(x0 + (lean + 0.1 * math.sin(t * 2 + x)) * 80 * s), int(H // 2 - 110 * s)), 1.0, 2, cv2.LINE_AA)
        G = cv2.resize(cv2.GaussianBlur(G, (0, 0), 6), (W, H))
        fr = fr * (1 - np.clip(G * 1.4, 0, 1)[..., None] * 0.9)
        fr = E.dust_motes(fr, t, 1321, n=60, amt=0.4)
        return bloom(fr, 0.85, 0.4)


class E28(Shot):   # S132b ECU 暗红腰带与马尾在逆光里翻飞（红绳母题收束）
    def __init__(self):
        sky = E.dawn_sky(1330, horizon=0.6, sun=(700, 480), mid=(0.5, 0.36, 0.2), cloud=0.6)
        self.bg = cv2.GaussianBlur(E.sun_disc(sky, (700, 480), 140, amt=1.0), (0, 0), 22)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr, info = put(fr, t, 'S121/wind_sash|crop=0,0.55,0.52,1', (1040 - 80 * u, 560), w=2400, anc=(0.5, 0.5), rot=-6,
                       rim=GOLDW, rim_amt=3.5, wind=22, grade=G_DAWN)
        fr = E.dust_motes(fr, t, 1331, n=90, amt=0.6, drift=(-80, -10), depth=True)
        return bloom(fr, 0.8, 0.45)


class Title(Shot):
    def __init__(self):
        self.sky = E.dawn_sky(1340, horizon=0.66, mid=(0.36, 0.26, 0.15), hor=(1.0, 0.8, 0.5), cloud=0.5, sun=(960, 700))
        self.cl = [E.cloud_layer(1341 + i, W * 2, H, density=0.4 + 0.2 * i) for i in range(2)]

    def base(self, t):
        fr = E.sun_disc(self.sky.copy(), (960, 690), 110, amt=1.1)
        for i, (lum, a) in enumerate(self.cl):
            ox = (t - 224.0) * (40 + 60 * i)
            col = lum[..., None] * (GOLD * (0.5 + 0.3 * i)) + 0.03
            M = np.float32([[1, 0, -ox % W], [0, 1, 420 - 100 * i]])
            R = cv2.warpAffine(col, M, (W, H), borderMode=cv2.BORDER_WRAP); A = cv2.warpAffine(a, M, (W, H))
            A[:420 - 100 * i] = 0
            fr = fr * (1 - A[..., None]) + R * A[..., None]
        return fr


class E29(Title):   # S133 片名《谁定正邪》金色火烧显字（志莽行书，邪 朱红）
    def render(self, t):
        u = self.u(t)
        fr = self.base(t)
        p = c01((t - 224.0) / 1.1)
        fr = T.burn_in(fr, '谁定正邪', p, size=230, fname='xing', color='gold', emph='邪', emph_color='red', anchor=(0.5, 0.45), seed=3)
        return bloom(fr, 0.85, 0.45)


TS = 225.42 - 1 / 30
class E30(Title):   # S133b 刀光斩过片名 → 两半滑开，露出「鬼屠 · 第三季」+「主题曲《谁定正邪》」+ 鬼屠朱印
    def render(self, t):
        fr = self.base(t)
        q = t - TS
        title = T.burn_in(np.zeros_like(fr), '谁定正邪', 1.0, size=230, fname='xing', color='gold', emph='邪', emph_color='red', anchor=(0.5, 0.45), seed=3)
        if q < 0.9:   # 斩开的旧片名：上下两半错开 + 淡出
            e = eout(q / 0.5)
            yy, xx = E.grid()
            ang = math.radians(-12); s = (yy - 470) * math.cos(ang) - (xx - 960) * math.sin(ang)
            up = K.cam(title, 1.0, ox=-60 * e, oy=-50 * e); dn = K.cam(title, 1.0, ox=60 * e, oy=50 * e)
            tt = np.where((s < 0)[..., None], up, dn) * (1 - c01(q / 0.9))
            fr = fr + tt
            line = np.exp(-(s / 3) ** 2) * math.exp(-q / 0.15)
            fr = fr + line[..., None] * GOLDW * 2.5
            if q < 0.1: fr = fr + 0.4 * (1 - q / 0.1)
        p = c01((q - 0.25) / 0.6)
        if p > 0:
            L = T.Layer()
            for ch, x in (('鬼', 600, ), ('屠', 790)), (('第', 1110), ('三', 1300), ('季', 1490)):
                pass
            for i, (ch, x) in enumerate((('鬼', 590), ('屠', 790), ('第', 1100), ('三', 1300), ('季', 1500))):
                e = c01(p * 1.6 - i * 0.12)
                T.place(L, ch, x, 470 - 20 * (1 - eout(e)), 200, GOLD_T, 'kai', alpha=eout(e), texture=True)
            fr = T.compose(fr, L, shadow=0.75, shadow_sigma=24, glow=0.5, glow_sigma=18, glow_color=(1.0, 0.6, 0.25))
            cv2.circle(fr, (945, 475), int(13 * eout(p)), tuple(float(v) for v in REDT), -1, cv2.LINE_AA)
            s2 = c01((q - 0.75) / 0.5)
            if s2 > 0:
                L2 = T.Layer()
                for i, ch in enumerate('主题曲《谁定正邪》'):
                    T.place(L2, ch, 960 + (i - 4) * 58, 650, 46, WHITE_T, 'serif_m', alpha=eout(s2), texture=False)
                fr = T.compose(fr, L2, shadow=0.7, shadow_sigma=16)
            sp = c01((q - 0.55) / 0.5)
            if sp > 0: fr = T.stamp_seal(fr, '鬼屠', 1680, 470, 130, sp, style='yin', rot=-5, seed=9)
        return bloom(fr, 0.85, 0.4)


class Credits(Shot):
    blocks = []      # (role, names, x, y_top, t_start, dur)
    def credits(self, fr, t):
        for (role, names, x, y, ts, dur) in self.blocks:
            p = c01((t - ts) / dur)
            out = c01((t - (self.t1 - 0.35)) / 0.35)
            if p <= 0: continue
            g = T.credit_block(fr.copy(), role, names, x, y, p, role_size=32, name_size=66, seal_text=names[0] if isinstance(names, list) else names[0],
                               seal_size=46, color='white', name_font='kai', seed=hash(role) % 50)
            fr = g * (1 - out) + fr * out
        return fr

    def text(self, fr, t): return self.credits(fr, t)


class E31(Credits):   # S134 ELS 晨雾群山缓移；署名①：出品人 / 编剧·导演
    def __init__(self):
        self.sky = E.dawn_sky(1350, horizon=0.62, mid=(0.42, 0.30, 0.17), hor=(1.0, 0.83, 0.55), cloud=0.5, sun=(560, 560))
        self.rd, self.ra = E.ridges(n=5, base=0.50, step=0.07, seed=134, amp=0.11, far=0.65, near=0.05)
        self.blocks = [('出品人', ['西几王传媒'], 1660, 230, 227.85, 0.9), ('编剧 / 导演', ['苏打'], 1430, 230, 228.45, 0.8)]
        self.fog = K.fbm(H // 4, W // 2, 1351, base=5, octaves=5)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (560, 560), 80, amt=1.0)
        fr = E.place_layer(fr, self.rd, self.ra, ox=-120 * u, z=1.05)
        f = cv2.resize(np.roll(self.fog, int(60 * u), 1), (W, H))
        m = np.clip((f - 0.45) * 2.5, 0, 1) * sstep(0.45, 0.85, np.arange(H, dtype=np.float32)[:, None] / H)
        fr = fr * (1 - m[..., None] * 0.7) + m[..., None] * GOLDW * 0.55
        return bloom(fr, 0.85, 0.35)


class E32(Credits):   # S135 CU 手握刀柄，沿刀身滑移；署名②：美术设计 / 后期总监
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1360, base=4, octaves=5)
        self.bg = cv2.resize(((0.03 + 0.06 * n)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.blocks = [('美术设计', ['Cait'], 1700, 230, 230.15, 0.8), ('后期总监', ['沉溺'], 1470, 230, 230.75, 0.8)]

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.light_beam(fr, (200, 120), (900, 960), 50, 500, t=t, strength=0.4, seed=11)
        fr, info = put(fr, t, 'S135/sheath_hand|crop=0.25,0.08,0.80,0.62', (700 - 140 * u, 540), w=1900, anc=(0.5, 0.5), rim=GOLDW, rim_amt=2.6, wind=3)
        sx = -200 + 1600 * eio(u)
        fr = E.flare_streak(fr, (sx, 520 + (sx - 700) * 0.6), amt=0.6, angle=58, length=0.6)
        return bloom(fr, 0.8, 0.35)


class E33(Credits):   # S135b 高空云海，鸟群掠过朝阳；署名③：AI画师
    def __init__(self):
        self.sky = E.dawn_sky(1370, horizon=0.58, mid=(0.45, 0.32, 0.18), hor=(1.0, 0.84, 0.56), cloud=0.6, sun=(700, 470))
        self.cl = [E.cloud_layer(1371 + i, W * 2, H, density=0.45 + 0.2 * i) for i in range(2)]
        self.blocks = [('AI画师', ['沉溺', '雪晴', '索恩'], 1640, 230, 232.45, 1.3)]
        r = np.random.default_rng(137)
        self.birds = [(r.uniform(-300, 200), r.uniform(330, 560), r.uniform(0.6, 1.2), r.uniform(0, 6.28)) for _ in range(11)]

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (700, 470), 95, amt=1.1)
        fr = fx2.god_rays(fr, (700, 470), strength=0.5, length=0.9, t=t)
        for i, (lum, a) in enumerate(self.cl):
            ox = (t - self.t0) * (50 + 80 * i)
            col = lum[..., None] * (GOLD * (0.55 + 0.3 * i)) + 0.03
            M = np.float32([[1, 0, -ox % W], [0, 1, 520 - 110 * i]])
            R = cv2.warpAffine(col, M, (W, H), borderMode=cv2.BORDER_WRAP); A = cv2.warpAffine(a, M, (W, H)); A[:520 - 110 * i] = 0
            fr = fr * (1 - A[..., None]) + R * A[..., None]
        for (x0, y0, s, ph) in self.birds:
            x = x0 + 520 * (t - self.t0) * s; y = y0 - 30 * (t - self.t0) + 6 * math.sin(t * 3 + ph)
            fl = math.sin(t * 11 * s + ph); w = 16 * s
            pts = np.array([[x - w, y - fl * 7 * s], [x - w * 0.3, y], [x, y + 2], [x + w * 0.3, y], [x + w, y - fl * 7 * s]], np.int32)
            cv2.polylines(fr, [pts], False, (0.03, 0.025, 0.02), max(2, int(3 * s)), cv2.LINE_AA)
        return bloom(fr, 0.85, 0.4)


class E34(Credits):   # S136 ELS 日出全景，远处一粒走向太阳的人影；署名④：音乐
    def __init__(self):
        self.sky = E.dawn_sky(1380, horizon=0.68, mid=(0.48, 0.34, 0.19), hor=(1.0, 0.86, 0.6), cloud=0.7, sun=(960, 640))
        self.rd, self.ra = E.ridges(n=3, base=0.66, step=0.07, seed=136, amp=0.05, far=0.55, near=0.04)
        self.blocks = [('音乐', ['西几王'], 1660, 260, 234.75, 0.9)]

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        z = 1.05 - 0.05 * eio(u)
        sun = (960, 690 - 60 * u)
        fr = E.sun_disc(fr, sun, 130, amt=1.15)
        fr = fx2.god_rays(fr, sun, strength=0.8, length=1.1, t=t)
        fr = E.place_layer(fr, self.rd, self.ra, z=z)
        fr, info = put(fr, t, 'S132/back_walk', (900 + 25 * u, 752), h=92, rim=GOLDW, rim_amt=3.0, grade=G_SIL, look='sil', wind=4)
        return bloom(fr, 0.85, 0.45)


class E35(Shot):   # S137 ECU 刀完全入鞘「咔」，朱印「鬼屠」盖在画面中心
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1390, base=4, octaves=5)
        self.bg = cv2.resize(((0.02 + 0.04 * n)[..., None] * np.array([1.0, 0.82, 0.62], np.float32)).astype(np.float32), (W, H))
        self.tc = 238.24 - 1 / 30

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.light_beam(fr, (1500, 100), (900, 960), 40, 600, t=t, strength=0.35, seed=13)
        slide = 0 if t >= self.tc else 70 * (1 - eio((t - self.t0) / (self.tc - self.t0)))
        dim = 0.55 if t >= self.tc + 0.4 else 1.0 - 0.45 * c01((t - self.tc) / 0.4) if t >= self.tc else 1.0
        fr, info = put(fr, t, 'S137/sheath_close|crop=0,0.40,1,1', (960 - slide, 640), w=2200, anc=(0.5, 0.5), rim=GOLDW, rim_amt=2.4, wind=2,
                       grade=dict(expo=0.8 * dim, sat=0.6, tint=(1.05, 0.98, 0.9), lift=0.0, gamma=1.12))
        if t >= self.tc:
            q = t - self.tc
            fr = E.flare_streak(fr, (1010, 640), amt=1.4 * math.exp(-q / 0.12), length=0.8)
            fr = RS.fx_sparks(fr, t, self.tc, (1010, 640), 1391, n=40, spd=500, dirv=(0, -1), spread=1.6, amt=0.9)
            p = c01((q - 0.08) / 0.55)
            if p > 0: fr = T.stamp_seal(fr, '鬼屠', 960, 470, 260, p, style='yin', rot=-4, seed=21)
        return bloom(fr, 0.8, 0.35)


class E36(Shot):   # S138 墨从四角收拢，只剩中心一点朱红 → 黑（干净落版）
    def __init__(self):
        self.src = None

    def render(self, t):
        if self.src is None:
            s = E35(); s.t0, s.t1 = 237.30, 239.13
            self.src = s.render(239.10)
        p = c01((t - self.t0) / 1.35)
        if p < 1:
            fr = TR.transition('iris_moon', self.src, np.zeros_like(self.src), p, mode='collapse', center=(960, 470), dot_color=RED)
        else:
            fr = np.zeros((H, W, 3), np.float32)
        return fr


# =========================================================================================== 剪辑表
#   切点 = 卡点（小节拍 / 人声字头）最近帧再提前 1 帧
SHOTS = [  # (类, 起帧, 原镜号, 景别)
    (E01, 5797, 'S111', 'ECU'), (E02, 5823, 'S112', 'LS'), (E03, 5868, 'S113', 'ELS'), (E04, 5910, 'S114', 'CU'),
    (E05, 5951, 'S114b', 'ECU'), (E06, 5979, 'S115', 'MS'), (E07, 6020, 'S116', 'LS'), (E08, 6061, 'S117', 'CU'),
    (E09, 6102, 'S118', 'MS'), (E10, 6130, 'S119', 'ELS'), (E11, 6157, 'S120a', 'MS'), (E12, 6182, 'S120b', 'ELS'),
    (E13, 6218, 'S121', 'MCU'), (E14T, 6263, 'S122', 'LS'), (E15, 6295, 'S122b', 'ECU'), (E16, 6310, 'S123', 'MCU'),
    (E17, 6338, 'S124', 'MCU'), (E18, 6363, 'S124b', 'MS'), (E19, 6377, 'S125', 'CU'), (E20, 6418, 'S126', 'ELS'),
    (E21, 6459, 'S127', 'CU'), (E22, 6501, 'S128', 'LS'), (E23, 6534, 'S129', 'MCU'), (E24, 6548, 'S129b', 'ECU'),
    (E25, 6569, 'S130', 'LS'), (E26, 6610, 'S131', 'ELS'),
    (E27, 6638, 'S132', 'LS'), (E28, 6693, 'S132b', 'ECU'), (E29, 6720, 'S133', 'MS'), (E30, 6762, 'S133b', 'MS'),
    (E31, 6830, 'S134', 'ELS'), (E32, 6900, 'S135', 'CU'), (E33, 6968, 'S135b', 'ELS'), (E34, 7037, 'S136', 'ELS'),
    (E35, 7119, 'S137', 'ECU'), (E36, 7174, 'S138', '—'),
]
IDS = [s[0].__name__ for s in SHOTS]
BOUNDS = [s[1] for s in SHOTS] + [F1]
CHORUS_END = 6638

# 入点转场：(名称, pre 秒, post 秒, kw)；'hard' = 切后 2 帧冲击（推近+提亮）
CUTS = {
    'E01': ('entry_flash', 0, 0.22, {}),
    'E02': ('hard', 0, 0, {}),
    'E03': ('smoke_through', 0.10, 0.20, dict(color=(0.95, 0.80, 0.55), direction=-1, seed=3)),
    'E04': ('hard', 0, 0, {}),
    'E05': ('whip_pan_blur', 0.05, 0.10, dict(direction='right')),
    'E06': ('occlusion', 0.08, 0.16, dict(kind='pillar', direction=-1, color=(0.92, 0.84, 0.64))),
    'E07': ('hard', 0, 0, {}),
    'E08': ('hard', 0, 0, {}),
    'E09': ('paper_tear', 0.06, 0.18, dict(seed=5)),
    'E10': ('zoom_through', 0.08, 0.14, dict(variant='paper', center=(960, 520))),
    'E11': ('hard', 0, 0, {}),
    'E12': ('whip_pan_blur', 0.05, 0.10, dict(direction='up')),
    'E13': ('flash_white', 0.0, 0.20, dict(color=(1.0, 0.9, 0.7))),
    'E14T': ('hard', 0, 0, {}),
    'E15': ('hard', 0, 0, {}),
    'E16': ('slash_wipe', 0.04, 0.14, dict(angle=90)),
    'E17': ('split_slide', 0.06, 0.14, dict(angle=90)),
    'E18': ('hard', 0, 0, {}),
    'E19': ('shatter', 0.04, 0.22, dict(center=(960, 540))),
    'E20': ('iris_moon', 0.0, 0.30, dict(center=(960, 520), radius=120, sun=True)),
    'E21': ('flash_red', 0.0, 0.16, {}),
    'E22': ('match_cut', 0.0, 0.18, dict(shape='circle', center=(960, 330), radius=120)),
    'E23': ('hard', 0, 0, {}),
    'E24': ('whip_pan_blur', 0.04, 0.08, dict(direction='left')),
    'E25': ('slash_wipe', 0.03, 0.12, dict(angle=-8)),
    'E26': ('match_cut', 0.0, 0.22, dict(shape='line', center=(960, 640), angle=0)),
    'E27': ('ink_bleed', 0.0, 0.40, dict(center=(960, 640), white=True, seed=7)),
    'E28': ('whip_pan_blur', 0.05, 0.10, dict(direction='right')),
    'E29': ('flash_white', 0.0, 0.18, dict(color=(1.0, 0.9, 0.7))),
    'E30': ('hard', 0, 0, {}),
    'E31': ('smoke_through', 0.18, 0.30, dict(color=(0.96, 0.86, 0.66), seed=9)),
    'E32': ('occlusion', 0.10, 0.18, dict(kind='blade', direction=1)),
    'E33': ('ink_bleed', 0.0, 0.40, dict(center=(700, 470), white=True, seed=8)),
    'E34': ('zoom_through', 0.10, 0.18, dict(center=(960, 640))),
    'E35': ('hard', 0, 0, {}),
    'E36': ('cut', 0, 0, {}),
}

# 运镜：(缓动, z0, z1, x0, x1, y0, y1, rot0, rot1)
MOVE = {
    'E01': ('expo_out', 1.08, 1.0, 40, -40, 0, 0, -1.5, 0),
    'E02': ('sine', 1.0, 1.06, 0, 0, 60, -40, 0, 0),           # 摇臂升
    'E03': ('sine', 1.03, 1.0, 0, 0, -20, 30, 0, 0),
    'E04': ('expo_in', 1.0, 1.12, 0, 0, 0, 0, 0, 0),           # 推进匾
    'E05': ('sine', 1.04, 1.10, -30, 30, 0, 0, -2, 2),
    'E06': ('expo_in', 1.0, 1.16, 0, 0, 0, 0, 0, 0),           # 穿帘推轨
    'E07': ('sine', 1.04, 1.0, 0, 0, 0, 0, 1.5, -1.5),        # 环绕感（配合视差）
    'E08': ('sine', 1.0, 1.08, 0, 0, 20, -20, 0, 0),
    'E09': ('expo_out', 1.08, 1.0, -60, 40, 0, 0, 2, -1),
    'E10': ('sine', 1.0, 1.10, 0, 0, 0, 0, 0, 0),
    'E11': ('expo_out', 1.06, 1.0, 0, 0, -30, 0, 0, 0),
    'E12': ('expo_out', 1.28, 1.0, 0, 0, 0, 0, -6, 0),          # 摇臂升（拉高俯拍）
    'E13': ('sine', 1.0, 1.06, -20, 20, 0, 0, 0, 0),
    'E14T': ('sine', 1.08, 1.0, 0, 0, 0, 0, 0, 0),             # 拉轨
    'E15': ('sine', 1.04, 1.04, 60, -60, 0, 0, 0, 0),
    'E16': ('sine', 1.0, 1.03, -40, 40, 0, 0, 0, 0),
    'E17': ('sine', 1.0, 1.03, 40, -40, 0, 0, 0, 0),
    'E18': ('sine', 1.0, 1.04, 0, 0, 0, 0, 0, 0),
    'E19': ('expo_in', 1.0, 1.10, 0, 0, 0, 0, 0, 0),
    'E20': ('sine', 1.0, 1.08, 0, 0, 30, -20, 0, 0),
    'E21': ('sine', 1.0, 1.06, 30, -10, 0, 0, 0, 0),
    'E22': ('sine', 1.0, 1.04, 0, 0, -30, 0, 0, 0),           # 摇臂降对齐
    'E23': ('expo_in', 1.0, 1.18, 0, 0, 0, 0, 0, 0),           # 吸气
    'E24': ('sine', 1.06, 1.0, 0, 0, 0, 0, 0, 0),
    'E25': ('expo_out', 1.12, 1.0, 0, 0, 0, 0, 0, 0),
    'E26': ('expo_out', 1.25, 1.0, 0, 0, -40, 0, 0, 0),        # 拉远：刀线 → 地平线
    'E27': ('sine', 1.0, 1.04, 0, 0, 40, -30, 0, 0),
    'E28': ('sine', 1.0, 1.05, 0, 0, 0, 0, 0, 0),
    'E29': ('sine', 1.0, 1.04, 0, 0, 0, 0, 0, 0),
    'E30': ('sine', 1.04, 1.07, 0, 0, 0, 0, 0, 0),
    'E31': ('sine', 1.0, 1.02, 0, 0, 0, 0, 0, 0),
    'E32': ('sine', 1.0, 1.04, 0, 0, 0, 0, 0, 0),
    'E33': ('sine', 1.0, 1.03, 0, 0, 20, -20, 0, 0),
    'E34': ('sine', 1.0, 1.0, 0, 0, 30, -10, 0, 0),
    'E35': ('sine', 1.0, 1.04, 0, 0, 0, 0, 0, 0),
    'E36': ('sine', 1.0, 1.02, 0, 0, 0, 0, 0, 0),
}
SHAKE = {   # 冲击抖：(时刻, 幅度 px)
    'E05': [(CT(49, 11), 10)], 'E09': [(CT(51, 2), 12)], 'E11': [(6157 / 30, 12)], 'E12': [(CT(51, 9), 8)],
    'E25': [(CT(55, 6), 18)], 'E30': [(TS, 8)], 'E35': [(238.24 - 1 / 30, 5)], 'E20': [(CT(54, 2), 10)],
}


def ease(kind, u):
    u = c01(u)
    if kind == 'expo_out': return 1 - 2 ** (-8 * u)
    if kind == 'expo_in': return (2 ** (8 * u) - 1) / 255
    return 0.5 - 0.5 * math.cos(math.pi * u)


class _Sec:
    def __init__(self):
        self.objs = {}

    def obj(self, i):
        if i not in self.objs:
            cls = SHOTS[i][0]; o = cls()
            o.t0, o.t1 = BOUNDS[i] / FPS, BOUNDS[i + 1] / FPS; o.id = IDS[i]
            self.objs[i] = o
            if len(self.objs) > 6: self.objs.pop(next(k for k in self.objs if k != i))
        return self.objs[i]

    def full(self, i, t):
        """镜头 i 在时刻 t 的完整画面（场景 + 运镜 + 屏幕字）。"""
        o = self.obj(i); sid = IDS[i]
        fr = o.render(t)
        mv = MOVE.get(sid)
        if mv:
            e = ease(mv[0], o.u(t))
            z = mv[1] + (mv[2] - mv[1]) * e; x = mv[3] + (mv[4] - mv[3]) * e; y = mv[5] + (mv[6] - mv[5]) * e; r = mv[7] + (mv[8] - mv[7]) * e
            z *= 1 + 0.010 * kick_punch(t)
            for (ts, amp) in SHAKE.get(sid, []):
                q = t - ts
                if 0 <= q < 0.5:
                    a = amp * math.exp(-q / 0.12)
                    x += a * math.sin(q * 90); y += a * math.cos(q * 73)
            fr = K.cam(np.clip(fr, 0, 4).astype(np.float32), z, x, y, r)
        return o.text(fr, t)

    def frame_f(self, f):
        t = f / FPS
        i = max(k for k in range(len(SHOTS)) if BOUNDS[k] <= f)
        for j in (i, i + 1):
            if j <= 0 or j >= len(SHOTS): continue
            name, pre, post, kw = CUTS.get(IDS[j], ('hard', 0, 0, {}))
            c = BOUNDS[j] / FPS
            if name in ('hard', 'cut', 'entry_flash') or not (c - pre - 1e-6 <= t < c + post - 1e-6): continue
            B = self.full(j, t)
            A = self.full(j - 1, min(t, c - 1 / FPS) if pre == 0 else t)
            p = (t - (c - pre)) / (pre + post)
            if pre == 0: p = 0.5 + 0.5 * p
            return TR.transition(name, np.clip(A, 0, 4).astype(np.float32), np.clip(B, 0, 4).astype(np.float32), p, **kw), 2
        fr = self.full(i, t)
        name = CUTS.get(IDS[i], ('hard',))[0]
        k = f - BOUNDS[i]
        ca = 0
        if name == 'entry_flash':                              # 段落入口：自带金白闪（接 D 组任意尾帧）
            p = c01(k / (0.22 * FPS))
            fr = fr * (0.4 + 0.6 * p) + np.array([1.0, 0.92, 0.78], np.float32) * (1 - p) ** 2 * 1.2
            ca = 3 if k < 3 else 0
        elif name == 'hard' and k < 2 and i > 0:
            a = 1.0 if k == 0 else 0.45
            fr = K.cam(fr, 1 + 0.025 * a) * (1 + 0.15 * a) + 0.04 * a
            ca = 3 if k == 0 else 1
        return fr, ca


_SEC = {}
def frame(f, cache):
    s = cache.get('sec')
    if s is None: s = cache['sec'] = _Sec()
    fr, ca = s.frame_f(int(f))
    t = f / FPS
    if f >= 7174:     # 收墨：不加颗粒/暗角，干净落版
        out = np.clip(fr, 0, 1); out[:BAR] = 0; out[H - BAR:] = 0
        return K.to8(out)
    fr = K.finish(fr, t, ca=ca, grain=0.035, vig=0.38, bloom_amt=0.25, red_keep=1.0)
    return K.to8(fr)


if __name__ == '__main__':
    import time
    from PIL import Image
    f = int(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/e.png'
    a = time.time(); im = frame(f, {}); print('%.2fs' % (time.time() - a))
    Image.fromarray(im).save(out)
