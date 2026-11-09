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
import e_r3 as R3
from e_r3 import Sub, LyrO
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
        self.l2 = Sub('未冷', cts(48, [1, 2]), 'left', CT(48, 2) + 0.15, seed=2)   # 统一副句

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
        fr = R3.blood_rub(fr, '血', 470, 500, 440, c01((t - (CT(48, 0) - 0.1)) / 0.22), seed=11)   # 「血」碑拓材质
        return self.l2.draw(fr, t)


class E02(Shot):   # S112 低机位大仰拍（第三棒重做）：人物占画面 3/4 高、偏右，冷灰蓝黎明天（次色调）；刀尖指日，「刀不放」朱印盖在日上；前景虚化草叶/飞灰
    GR = dict(expo=0.78, sat=0.55, tint=(0.94, 0.98, 1.06), lift=0.0, gamma=1.12)

    def __init__(self):
        self.sky = E.dawn_sky(112, horizon=0.98, top=(0.025, 0.03, 0.055), mid=(0.16, 0.20, 0.28), hor=(0.62, 0.56, 0.52), cloud=0.9, sun=None)
        r = np.random.default_rng(112)
        self.grass = [(r.uniform(-80, 520) if i % 2 else r.uniform(1500, 2000), r.uniform(0.7, 1.5), r.uniform(-0.5, 0.5)) for i in range(30)]

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        z, info = put(np.zeros((H, W, 3), np.float32), t, 'S112/sword_sky|flip', (1150, 1460 + 20 * u), h=1220, anc=(0.5, 1.0),
                      rim=(1.0, 0.86, 0.62), rim_amt=3.4, rim_dir=(-0.2, -1.0), grade=self.GR, wind=10)
        tip = info['top']
        sun = (tip[0], tip[1] - 6 - 24 * eio(u))
        fr = E.sun_disc(fr, sun, 46, amt=1.0, core=(1.0, 0.98, 0.94))
        fr = fx2.god_rays(fr, sun, strength=0.7, length=1.0, t=t, occluder=info['A'])
        a = info['A'][..., None]
        fr = fr * (1 - a) + z * a
        p = (t - (CT(48, 5) - 0.12)) / 0.5
        if p > 0: fr = T.stamp_seal(fr, '刀不放', sun[0] - 6, sun[1] + 40, 210, min(1, p), style='yin', rot=-6, seed=4, alpha=0.95)
        G = np.zeros((H // 2, W // 2), np.float32)                         # 前景：贴镜头的草叶（强虚化、压暗）
        for (x, s_, lean) in self.grass:
            x0 = (x + 60 * u) / 2
            cv2.line(G, (int(x0), H // 2 - BAR // 2 + 30), (int(x0 + (lean + 0.12 * math.sin(t * 2.3 + x)) * 120 * s_), int(H // 2 - BAR // 2 - 95 * s_)), 1.0, 6, cv2.LINE_AA)
        G = cv2.resize(cv2.GaussianBlur(G, (0, 0), 7), (W, H))
        fr = fr * (1 - np.clip(G * 1.3, 0, 1)[..., None] * 0.92)
        fr = E.dust_motes(fr, t, 1121, n=36, amt=0.5, drift=(30, -40), size=(3, 9), blur=4, col=(0.9, 0.85, 0.8))
        return bloom(fr, 0.8, 0.45)


class E03(Shot):   # S113（第三棒重做）一点透视碑林：两列刻满律令的「天理」碑沿纵深排开，冷灰蓝晨雾，尽头门洞白光；推轨前进 + 前景碑擦镜虚化。
    #  「世间从来没有」一行横排、按唱的顺序逐字烧出（阅读顺序 = 出现顺序），压在画面下三分之一的暗衬上。
    def __init__(self):
        self.hall = R3.SteleHall(seed=31, n=7, hy=455)
        tm = cts(49, range(6))
        self.l = LyrO(row('世间从来没有', tm, 960, 800, size=100, gold='世间', track=0.18, font='serif_b'), CT(49, 6) - 0.12, seed=10,
                      glow=0.3, outline=3, plate_amt=0.45)

    def render(self, t):
        u = self.u(t)
        fr = self.hall.render(t, dz=0.2 + 1.5 * eio(u), sway=-30 + 60 * u)
        fr = fx2.god_rays(fr, (960, 455), strength=0.55, length=1.0, t=t)
        fr = E.dust_motes(fr, t, 113, n=90, amt=0.45, drift=(10, 6), col=(0.85, 0.9, 1.0), depth=True)
        return bloom(fr, 0.85, 0.4)

    def text(self, fr, t): return self.l.draw(fr, t)


def cloud_blit(fr, lum, a, ox, yoff, tone, amb=0.03, fade=140):
    """云海层横移贴图（层宽 2W，取 [ox%W, ox%W+W] 窗口；顶边软过渡，不留硬切线）。"""
    x0 = int(ox % W)
    L = lum[:, x0:x0 + W]; A = a[:, x0:x0 + W]
    M = np.float32([[1, 0, 0], [0, 1, yoff]])
    L = cv2.warpAffine(L, M, (W, H)); A = cv2.warpAffine(A, M, (W, H))
    A = A * sstep(yoff, yoff + fade, np.arange(H, dtype=np.float32))[:, None]
    R = L[..., None] * np.asarray(tone, np.float32) + amb
    return fr * (1 - A[..., None]) + R * A[..., None]


class E04(Shot):   # S114 CU 暗殿金匾「天理昭彰」：随唱逐字亮起；高窗光柱
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 140, base=3, octaves=5)
        slat = (np.sin(np.linspace(0, 60, W // 2)) * 0.5 + 0.5)[None, :] ** 6
        self.bg = cv2.resize(((0.03 + 0.05 * n + 0.025 * slat)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        self.pl = PR.Plaque('天理昭彰', seed=3)
        self.l = Sub('什么', cts(49, [6, 7]), 'left', 198.3, seed=20)
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


class E06(Shot):   # S115 MS（第三棒重做）律令挂轴林：远中近三层直垂挂轴（刚体，只绕挂点微摆，不再拉伸变形），背光透纸；
    #  推轨穿帘：近层大轴虚化擦镜；「若规则是」烧在正中偏右的一幅题字轴上（大字竖排，规则 朱红）。
    def __init__(self):
        self.sky = E.dawn_sky(115, horizon=0.7, sun=(900, 520), hor=(1.0, 0.86, 0.6), mid=(0.62, 0.48, 0.30), cloud=0.3)
        self.far = R3.Scrolls(15, 15, (40, 1880), 150, (640, 720), (78, 104), text_size=12)
        self.mid = R3.Scrolls(16, 5, (120, 1100), 110, (820, 880), (150, 180), text_size=17)
        self.title = R3.Scrolls(19, 1, (1400, 1400), 96, (860, 860), (270, 270), tone=(0.95, 0.90, 0.78), blank=True)
        self.title.items[0]['amp'] = 0.25
        self.near = R3.Scrolls(17, 3, (-120, 2040), 40, (1150, 1250), (300, 360), text_size=26)
        tm = cts(50, [0, 1, 2, 3])
        self.l = Lyr(row('若规则是', tm, 1400, 560, size=150, vertical=True, red='规则', base=(0.06, 0.045, 0.04), track=0.12, font='serif_black'),
                     CT(50, 4) - 0.05, seed=30, glow=0.0, shadow=0.0)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (900, 520), 80, amt=1.0)
        C, A = self.far.layer(t, wind=0.8); C, A = R3.Scrolls.light(C, A, fr, 1.0, blur=1.6)
        fr = R3.comp(fr, C, A)
        fr = fx2.god_rays(fr, (900, 520), strength=0.55, length=0.9, t=t)
        C, A = self.mid.layer(t, wind=1.0); C, A = R3.Scrolls.light(C, A, fr, 0.8)
        fr = R3.comp(fr, C, A)
        C, A = self.title.layer(t, wind=1.0); C, A = R3.Scrolls.light(C, A, fr, 0.55)
        fr = R3.comp(fr, C, A)
        sc = 1.0 + 0.35 * eio(u)
        C, A = self.near.layer(t, wind=1.2, scale=sc, dx=-80 * u); C, A = R3.Scrolls.light(C, A, fr * 0.6, 0.4, blur=9)
        fr = R3.comp(fr, C * 0.55, A)
        return bloom(fr, 0.85, 0.35)

    def text(self, fr, t):
        return self.l.draw(fr, t)


class E07(Shot):   # S116 LS（第三棒重做）挂轴缝后，太上皇本人（实体、正面受光、五官清楚）→ 环绕揭示：原来是纸扎（实体纸扎，不透明）
    def __init__(self):
        self.sky = E.dawn_sky(116, horizon=0.75, sun=(1480, 330), hor=(0.62, 0.46, 0.30), mid=(0.20, 0.15, 0.10), top=(0.03, 0.025, 0.02), cloud=0.3)
        self.cur = R3.Scrolls(17, 6, (-40, 1960), 50, (980, 1050), (190, 240), text_size=22)
        self.l = LyrO(row('谎言', cts(50, [4, 5]), 960, 800, size=230, red='谎言', track=0.08, font='serif_black'), CT(50, 6) - 0.1, seed=40,
                      outline=5, plate_amt=0.6, glow=0.3)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()                                               # 第四棒：背景压暗、日偏右后方（只留光晕），纸扎在暗底上成实体
        fr = fx2.god_rays(fr, (1480, 330), strength=0.35, length=0.8, t=t)
        rv = eio((t - 201.2) / 0.4)
        dx = 70 - 140 * eio(u)
        GE = dict(expo=1.0, sat=0.62, tint=(1.06, 0.99, 0.9), lift=0.02, gamma=1.0)
        if rv < 1:      # 帘后：太上皇本人，前侧受光，五官清楚
            fr0, _ = put(fr.copy(), t, 'S116/emperor', (960 + dx, 1590), h=1500, rim=GOLDW, rim_amt=2.6, grade=GE, wind=2)
        if rv > 0:      # 揭示：纸扎（宣纸蒙竹骨），实体不透明，只在边缘透光
            fr1, info = put(fr.copy() * 0.45, t, 'E_r2/paper_effigy', (960 + dx, 1590), h=1500, rim=GOLDW, rim_amt=3.4, rim_dir=(0.9, -0.4), wind=6,
                            grade=dict(expo=1.05, sat=0.5, tint=(1.06, 1.0, 0.9), lift=0.0, gamma=1.0))   # 第四棒：暗底上亮宣纸，读作实体
            (bx0, by0), (bx1, by1) = info['box']                              # 第四棒：身体部分（头以下）糊上不透光的厚宣纸：抹平透出的灯笼网格，只留竹骨压痕
            yv = np.arange(H, dtype=np.float32)[:, None]
            bm = info['A'] * sstep(by0 + 0.20 * (by1 - by0), by0 + 0.30 * (by1 - by0), yv)
            sm = cv2.GaussianBlur(fr1, (0, 0), 7)
            lum = sm.mean(-1, keepdims=True)
            solid = (0.6 * lum + 0.4 * sm) * np.array([1.08, 0.98, 0.82], np.float32) * 1.05
            fr1 = fr1 * (1 - 0.75 * bm[..., None]) + solid * 0.75 * bm[..., None]
            fr1 = paper_frame(fr1, info['A'], t, amt=0.32)
            A_ = info['A']                                                   # 体积：边缘向内压暗（纸扎是实心的，不是灯笼）+ 左侧冷面
            occ = np.clip(A_ - cv2.GaussianBlur(A_, (0, 0), 28) * 0.0, 0, 1) * (1 - cv2.GaussianBlur((A_ > 0.5).astype(np.float32), (0, 0), 30))
            xx_ = np.arange(W, dtype=np.float32)[None, :]
            side_ = np.clip((1100 + dx - xx_) / 500, 0, 1)
            fr1 = fr1 * (1 - (0.55 * occ + 0.35 * side_ * A_)[..., None])
        if rv >= 1: fr = fr1
        elif rv <= 0: fr = fr0
        else:            # 第四棒：揭示不再叠化（双重曝光读成「半透明」），改为一道纸面反光斜扫 = 硬擦除边界：光带左边已是纸扎、右边还是真人
            yy, xx = E.grid(); sx = xx - (300 + 1400 * rv) + (yy - 540) * 0.4
            m = sstep(8, -8, sx)[..., None]
            fr = fr1 * m + fr0 * (1 - m)
            band = np.exp(-(sx / 50) ** 2)
            fr = fr + band[..., None] * GOLDW * 0.6
        sh = -300 * eio(u) + 150                                          # 前景挂轴从两侧掠过（环绕视差，强虚化）
        C, A = self.cur.layer(t, wind=1.0, dx=sh)
        side = np.clip(np.abs(np.arange(W, dtype=np.float32) - 960) / 160 - 3.6, 0, 1)[None, :]   # 第四棒：前景挂轴只在两侧画边、实体不透明，不再叠在纸扎身上
        C, A = R3.Scrolls.light(C, A, fr * 0.5, 0.5, blur=10)
        fr = R3.comp(fr, C * 0.35, A * side)
        return bloom(fr, 0.85, 0.35)

    def text(self, fr, t): return self.l.draw(fr, t)


_PF = {}
def paper_frame(fr, A, t, seed=0, amt=0.55):
    """纸扎质感：皱纸纤维（明暗 ±）+ 不规则竹骨（少量歪斜竹篾，透光处亮、骨处暗）+ 糊纸接缝。只在精灵 alpha 内。"""
    if seed not in _PF:
        r = np.random.default_rng(900 + seed)
        yy, xx = E.grid()
        m = np.zeros((H, W), np.float32)
        for k in range(6):                                  # 竖篾：微弯、粗细不一
            x0 = 300 + k * 260 + r.uniform(-60, 60); a = r.uniform(-0.08, 0.08); w = r.uniform(2.5, 6)
            m = np.maximum(m, np.exp(-((xx - (x0 + a * (yy - 540) + 14 * np.sin(yy / r.uniform(90, 160)))) / w) ** 2))
        for k in range(5):                                  # 横篾
            y0 = 200 + k * 180 + r.uniform(-40, 40); a = r.uniform(-0.12, 0.12); w = r.uniform(2.5, 5)
            m = np.maximum(m, np.exp(-((yy - (y0 + a * (xx - 960) + 10 * np.sin(xx / r.uniform(120, 220)))) / w) ** 2))
        crumple = K.fbm(H // 2, W // 2, 910 + seed, base=26, octaves=5)
        crumple = cv2.resize(crumple, (W, H))
        fib = cv2.resize(K.fbm(H // 2, W // 2, 920 + seed, base=180, octaves=2), (W, H))
        seam = np.exp(-((np.sin((xx * 0.8 + yy * 0.6) / 210.0)) / 0.025) ** 2) * 0.5   # 糊纸接缝（斜向）
        _PF[seed] = (cv2.GaussianBlur(m, (0, 0), 1.0), crumple, fib, seam)
    m, cr, fib, seam = _PF[seed]
    tex = 0.86 + 0.22 * cr + 0.08 * (fib - 0.5)
    k = A[..., None]
    out = fr * (1 - k) + fr * k * tex[..., None]
    out = out * (1 - (amt * m + 0.25 * seam * amt)[..., None] * k)
    halo = cv2.GaussianBlur(m, (0, 0), 6) - m * 0.5                     # 骨两侧纸被绷紧 → 微亮
    return out + np.clip(halo, 0, 1)[..., None] * k * np.array([1.0, 0.85, 0.6], np.float32) * 0.08 * (amt / 0.55)


class E08(Shot):   # S117 CU（第三棒重做）纸糊巨脸自中心烧穿 → 洞后不是白光，是暗处他的双眼（S123 眼部裁切，火光从下照亮）
    def __init__(self):
        r = np.random.default_rng(117)
        self.nz = cv2.resize(r.random((H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        self.nz2 = cv2.resize(r.random((H // 6, W // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        n = K.fbm(H // 2, W // 2, 1171, base=4, octaves=5)
        self.void = cv2.resize(((0.015 + 0.035 * n)[..., None] * np.array([1.0, 0.8, 0.62], np.float32)).astype(np.float32), (W, H))
        self.l = LyrO(row('伪装', cts(50, [6, 7]), 960, 820, size=230, red='伪装', track=0.08, font='serif_black'), CT(51, 0) - 0.12, seed=50,
                      outline=5, plate_amt=0.6, glow=0.3)
        self.bg = E.dawn_sky(119, horizon=0.8, mid=(0.5, 0.36, 0.2), cloud=0.0)

    def render(self, t):
        u = self.u(t)
        fr, info = put(self.bg.copy(), t, 'S117/emperor_face', (960, 1190), h=1250 * (1 + 0.08 * u), rim=GOLDW, rim_amt=2.4, wind=1,
                       grade=dict(expo=0.8, sat=0.45, tint=(1.08, 1.0, 0.86), lift=0.03, gamma=1.08))
        fr = paper_frame(fr, info["A"], t, seed=1, amt=0.28)
        tb = CT(50, 6) - 0.05
        prog = c01((t - tb) / 1.3)
        if prog > 0:
            yy, xx = E.grid()
            c = (960, 470)
            d = np.hypot(xx - c[0], (yy - c[1]) * 1.15) / 900
            field = d + (self.nz - 0.5) * 0.35 + (self.nz2 - 0.5) * 0.08
            R = eout(prog, 2) * 0.62
            keep = np.clip((field - R) / 0.015, 0, 1)
            ember = np.exp(-((field - R) / 0.02) ** 2)
            char = np.exp(-np.maximum(field - R, 0) / 0.05) * (field > R)
            A = info['A']
            burned = A * (1 - keep)
            # 洞后：暗处横刀（S128 刀身裁切），刃口受光
            v = self.void.copy()
            v, bi = put(v, t, 'S123/front|crop=0.36,0.25,0.64,0.45', (960, 350 - 20 * u), w=2100, anc=(0.5, 0.5),
                        rim=(1.0, 0.6, 0.3), rim_amt=2.0, rim_dir=(0.0, 1.0), grade=dict(expo=0.95, sat=0.6, tint=(1.1, 0.96, 0.84), lift=0.0, gamma=1.08), wind=3)
            yy2 = yy - 470; glow = np.exp(-(((xx - 960) / 700) ** 2 + (yy2 / 330) ** 2))                 # 火光从洞口照进去：中心亮、四周沉入黑
            v = v * (0.25 + 0.85 * glow)[..., None] + (np.exp(-((yy - 900) / 260) ** 2) * 0.12)[..., None] * np.array([1.0, 0.4, 0.1], np.float32)
            fr = fr * (1 - 0.85 * char * A)[..., None]
            fr = fr * (1 - burned[..., None]) + v * burned[..., None]
            em = ember * A
            fr = fr + (em * 2.2 + cv2.GaussianBlur(em, (0, 0), 10) * 2.0)[..., None] * np.array([1.0, 0.5, 0.18], np.float32)
            fr = E.dust_motes(fr, t, 1170, n=120, amt=0.9 * prog, drift=(20, -160), col=(1.0, 0.6, 0.3), size=(1, 4))
        return bloom(fr, 0.85, 0.45)

    def text(self, fr, t): return self.l.draw(fr, t)


class E09(Shot):   # S118 MS 交叉斩：一刀斜劈，纸帘沿刀线断开；「那就撕碎」沿刀线排、被一分为二
    def __init__(self):
        self.sky = E.dawn_sky(1180, horizon=0.72, sun=(1500, 500), hor=(1.0, 0.84, 0.55), mid=(0.55, 0.40, 0.24), cloud=0.3)
        self.cur = R3.Scrolls(18, 11, (60, 1860), 70, (880, 960), (120, 160), text_size=17)
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
        C, A = self.cur.layer(t, wind=1.1)
        C, A = R3.Scrolls.light(C, A, fr, 0.9, blur=1.2)
        if t >= ts:                                                          # 挂轴沿刀线断开：线下半截下坠，断口烧红
            C, A, em = R3.cut_layer(C, A, tuple(self.p0), tuple(self.p1), t - ts, fall=1.0, side=-1)
            fr = R3.comp(fr, C, A)
            fr = fr + (em * 1.6 + cv2.GaussianBlur(em, (0, 0), 8) * 1.5)[..., None] * np.array([1.0, 0.5, 0.15], np.float32)
        else:
            fr = R3.comp(fr, C, A)
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


class E10(Shot):   # S119（第三棒重做）碎纸扑面：被斩碎的律令纸片 + 金箔，有厚度、3D 翻转、焦边余烬，前景大片强虚化擦镜；
    #  墨黑底、残存一点暗红火光（次色调），不再用日前黑剪影；「它 / 自己」大字描边落在纸片之间
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1190, base=4, octaves=5)
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        glow = np.exp(-(((xx - 480) / 420) ** 2 + ((yy - 300) / 260) ** 2))
        bg = (0.015 + 0.04 * n)[..., None] * np.array([1.0, 0.85, 0.75], np.float32) + glow[..., None] * np.array([0.45, 0.12, 0.04], np.float32) * (0.6 + 0.5 * n[..., None])
        self.bg = cv2.resize(bg.astype(np.float32), (W, H))
        self.sh = R3.Shreds(119, n=70, gold=0.3, size=(60, 200), spread=(1500, 700), zr=(0.5, 3.4))
        self.l = LyrO([('它', 520, 430, 240, 'serif_black', WHITE_T, CT(51, 4)), ('自', 1330, 380, 190, 'serif_black', GOLD_T, CT(51, 5)),
                       ('己', 1520, 560, 190, 'serif_black', GOLD_T, CT(51, 6))], CT(51, 7) - 0.12, seed=70, glow=0.4, outline=4)

    def render(self, t):
        fr = self.bg.copy()
        q = (t - self.t0) * 0.45 + 0.6                                          # 慢放
        fr = self.sh.draw(fr, q, light=(-0.5, -0.4, -0.75), dof=(2.5, 14.0), zfocus=1.3, emb=1.2, paper=(1.0, 0.84, 0.62), znear=0.55)
        fr = E.dust_motes(fr, t, 1191, n=80, amt=0.8, drift=(30, -80), col=(1.0, 0.55, 0.25), size=(1, 4))
        return bloom(fr, 0.8, 0.5)

    def text(self, fr, t):
        return self.l.draw(fr, t)


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


class E12(Shot):   # S120b ELS 正俯拍（第三棒返工）：大地即印面，「鬼屠」朱印盖下（摇臂升）；人物立在印旁（不压字），
    #  精灵只裁人（去掉插地刀土堆那团圆形物），按地面光调色 + 接触阴影 + 晨光长影
    KEY = 'E_r2/top_down|crop=0,0,0.56,1'

    def __init__(self):
        n = K.fbm(H, W, 1210, base=6, octaves=6)
        cr = K.fbm(H, W, 1211, base=22, octaves=3)
        crack = np.clip(1 - np.abs(cr - 0.5) * 40, 0, 1)
        pb = K.fbm(H // 2, W // 2, 1212, base=220, octaves=2)
        peb = cv2.resize(np.clip((pb - 0.6) * 5, 0, 1), (W, H))
        g = 0.10 + 0.16 * n - crack * 0.06 + peb * 0.05
        self.ground = (g[..., None] * np.array([1.0, 0.80, 0.58], np.float32)).astype(np.float32)
        self.face = PR.seal_face('鬼屠', 640, style='bai', wear=0.25, seed=12)
        self.cap = Sub('自己定规章', [CT(51, 5), CT(51, 6), CT(51, 7), CT(51, 8), CT(51, 9)], 'center', CT(52, 0) - 0.15, red='自己', seed=90)

    def render(self, t):
        u = self.u(t)
        fr = self.ground.copy()
        yy, xx = E.grid()
        fr = fr * (0.55 + 0.6 * np.exp(-((xx - 1300) ** 2 + (yy - 200) ** 2) / (2 * 700 ** 2)))[..., None]   # 晨光斜照
        p = (t - (CT(51, 9) - 0.18)) / 0.7
        c = (820, 500)
        if p > 0: fr = PR.stamp_imprint(fr, self.face, c[0], c[1], 640, min(1, p), angle=-4)
        at = (1390, 600)
        z, info = put(np.zeros((H, W, 3), np.float32), t, self.KEY, at, h=330, anc=(0.5, 0.5), rim=GOLDW, rim_amt=1.6, rim_dir=(0.7, -0.7),
                      grade=dict(expo=0.95, sat=0.6, tint=(1.1, 0.98, 0.84), lift=0.0, gamma=1.05), wind=4)
        A = info['A']
        Sh = np.zeros_like(A)                                              # 晨光自右上 → 长影拖向左下（印面上也压影）
        for k in range(1, 9):
            M = np.float32([[1, 0, -34 * k], [0, 1, 26 * k]]); Sh = np.maximum(Sh, cv2.warpAffine(A, M, (W, H)) * (1 - 0.07 * k))
        Sh = cv2.GaussianBlur(Sh, (0, 0), 6)
        con = cv2.GaussianBlur(cv2.dilate(A, np.ones((9, 9), np.uint8)), (0, 0), 14)        # 接触阴影
        fr = fr * (1 - 0.6 * Sh[..., None]) * (1 - 0.5 * con[..., None])
        fr = fr * (1 - A[..., None]) + z * A[..., None]
        if p > 0.3:
            q = (p - 0.3) / 0.7
            r = 330 + 600 * eout(q)
            ring = np.exp(-((np.hypot(xx - c[0], yy - c[1]) - r) / 40) ** 2) * (1 - q)
            fr = fr + ring[..., None] * np.array([0.8, 0.5, 0.3], np.float32) * 0.5
            fr = E.dust_motes(fr, t, 1213, n=60, amt=0.5 * (1 - q), drift=(40, 30), col=(0.9, 0.7, 0.5), size=(2, 5))
        return fr

    def text(self, fr, t): return self.cap.draw(fr, t)


class E13(Shot):   # S121 MCU 晨风中马尾与暗红腰带甩动，背景日光虚焦；巨字「血」
    def __init__(self):
        sky = E.dawn_sky(1210, horizon=0.75, sun=(1500, 380), mid=(0.45, 0.32, 0.18), cloud=0.7)
        sky = E.sun_disc(sky, (1500, 380), 120, amt=1.0)
        self.bg = cv2.GaussianBlur(sky, (0, 0), 18)
        r = np.random.default_rng(121)
        self.bok = [(r.uniform(900, 1900), r.uniform(150, 900), r.uniform(25, 70), r.uniform(0.2, 0.6)) for _ in range(18)]
        self.l2 = Sub('未冷', cts(52, [1, 2]), 'left', CT(52, 3) - 0.1, seed=101)

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

    def text(self, fr, t):
        if t < CT(52, 3) - 0.1: fr = R3.blood_drip(fr, '血', 470, 470, 470, t, CT(52, 0), seed=13)   # 「血」滴血笔触（与 E01 碑拓不同）
        return self.l2.draw(fr, t)


class E14(Shot):   # S122（第三棒重做）低机位仰拍：他从烟尘里朝镜头走来，人物偏右占 4/5 高；墨灰冷调烟幕（次色调），
    #  左前景大片焦纸强虚化擦过镜头（前景遮挡），火星斜飞。不再是「居中小剪影 + 日落」。
    def __init__(self):
        self.sky = E.dawn_sky(1220, horizon=0.95, top=(0.02, 0.022, 0.03), mid=(0.12, 0.13, 0.16), hor=(0.42, 0.36, 0.32), cloud=0.9, sun=None)
        n = K.fbm(H // 2, W // 2, 1222, base=10, octaves=5)
        self.ground = cv2.resize(((0.03 + 0.06 * n)[..., None] * np.array([0.9, 0.85, 0.82], np.float32)).astype(np.float32), (W, H))
        self.smoke = K.fbm(H // 4, W // 2, 1224, base=4, octaves=5)
        self.sh = R3.Shreds(122, n=12, gold=0.0, size=(120, 240), spread=(380, 360), zr=(0.35, 1.0))

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        hy = 905
        m = sstep(hy - 3, hy + 30, np.arange(H, dtype=np.float32))[:, None, None]
        fr = fr * (1 - m) + self.ground * m
        sm = cv2.resize(R3.slide(self.smoke, -40 * (t - self.t0)), (W, H))
        fr = fr + (np.clip(sm - 0.4, 0, 1) * 0.5)[..., None] * np.array([0.55, 0.55, 0.6], np.float32) * sstep(200, 900, np.arange(H, dtype=np.float32))[:, None, None]
        yy, xx = E.grid()
        fr = fr + (np.exp(-(((xx - 1700) / 500) ** 2 + ((yy - 800) / 260) ** 2)) * 0.35)[..., None] * np.array([1.0, 0.45, 0.15], np.float32)  # 右后方余火
        bob = 6 * abs(math.sin(t * 5.5))
        fr, info = put(fr, t, 'E_r4/low_stride', (1180, 1210 - bob + 40 * u), h=960 * (1 + 0.06 * u), rim=(1.0, 0.62, 0.32), rim_amt=3.2,
                       rim_dir=(0.8, -0.3), grade=dict(expo=0.84, sat=0.5, tint=(0.95, 0.97, 1.03), lift=0.0, gamma=1.1), wind=12)
        fr = RS.fx_embers(fr, t, 1223, amt=0.9, n=70)
        fr = self.sh.draw(fr, (t - self.t0) * 0.6 + 0.3, c=(420, 620), dof=(4.0, 18.0), zfocus=2.5, emb=0.8, paper=(0.55, 0.45, 0.36), amt=0.95)
        return bloom(fr, 0.8, 0.45)


L_GU = None
def lyr_gu():
    global L_GU
    if L_GU is None:
        L_GU = Sub('骨未凉', cts(52, [3, 4, 5]), 'left', CT(53, 0) - 0.15, red='骨', seed=110)
    return L_GU


class E14T(E14):
    def text(self, fr, t): return lyr_gu().draw(fr, t)


class E15(Shot):   # S122b ECU 刀尖拖过焦土，划出一道朱金火线（他自己画的线）；刀尖由精灵 alpha 标定，火线头 = 刀尖
    KEY = 'S122/walk_toward|crop=0.26,0.62,0.40,0.92'

    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1230, base=12, octaves=6)
        cr = K.fbm(H // 2, W // 2, 1231, base=30, octaves=3)
        g = 0.05 + 0.13 * n - np.clip(1 - np.abs(cr - 0.5) * 30, 0, 1) * 0.04
        yy = np.linspace(0, 1, H // 2, dtype=np.float32)[:, None]
        g = g * (0.5 + 0.9 * yy)
        self.ground = cv2.resize((g[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32), (W, H))
        pb = K.fbm(H // 2, W // 2, 1232, base=200, octaves=2)                       # 碎石/焦土颗粒高光
        self.peb = cv2.resize(np.clip((pb - 0.62) * 6, 0, 1), (W, H))
        _, info = put(np.zeros((H, W, 3), np.float32), 0.0, self.KEY, (960, 540), h=760, anc=(0.06, 0.96), rot=62, wind=0)
        ys, xs = np.where(info['A'] > 0.4)
        k = np.argmax(ys)                                                          # 最低点 = 刀尖
        self.off = (float(xs[k]) - 960, float(ys[k]) - 540)
        # 爆点：刀尖拖过时斩断一幅「天理」律令轴（第三棒新增：砍中东西）
        self.scr = R3.Scrolls(150, 1, (1180, 1180), 100, (760, 760), (250, 250), big=['天理'], tone=(0.92, 0.86, 0.72))
        self.scr.items[0]['amp'] = 0.3
        us = np.linspace(0, 1, 400); xs_ = -100 + 1900 * np.array([eio(0.3 + v * 0.7) for v in us])
        self.tcut = 209.83 + float(us[np.argmax(xs_ >= 1180)]) * (210.33 - 209.83)

    @staticmethod
    def ly(x): return 640 + (x - 960) * 0.03

    def render(self, t):
        u = self.u(t)
        fr = self.ground.copy()
        yy, xx = E.grid()
        x = -100 + 1900 * eio(0.3 + u * 0.7)
        tip = (x, self.ly(x))
        lit = np.exp(-((xx - x) ** 2 + ((yy - tip[1]) * 2.2) ** 2) / (2 * 260 ** 2))       # 火线照亮刀尖周围焦土
        fr = fr * (0.7 + 0.6 * np.exp(-((yy - 150) / 500) ** 2))[..., None] + (self.peb * lit)[..., None] * np.array([1.0, 0.55, 0.25], np.float32) * 0.22
        fr = fr + lit[..., None] * np.array([0.35, 0.12, 0.04], np.float32)
        ly = self.ly(xx)
        on = 1 - sstep(x - 40, x, xx)
        cool = np.exp(-np.maximum(x - xx, 0) / 700)                                     # 远处冷却成暗红
        line = np.exp(-((yy - ly) / 3.5) ** 2) * on
        glow = np.exp(-((yy - ly) / 24) ** 2) * on * cool
        fr = fr * (1 - 0.6 * np.exp(-((yy - ly) / 8) ** 2) * on)[..., None]
        hot = np.array([1.0, 0.85, 0.5], np.float32) * cool[..., None] + np.array([0.8, 0.12, 0.04], np.float32) * (1 - cool[..., None])
        fr = fr + (line * 1.6)[..., None] * hot + glow[..., None] * np.array([1.0, 0.35, 0.1], np.float32) * 0.9
        C, A = self.scr.layer(t, wind=1.0)
        C = C * (0.55 + 0.5 * lit[..., None]) + lit[..., None] * np.array([0.5, 0.2, 0.05], np.float32)
        q = t - self.tcut
        if q >= 0:
            p0 = (900.0, self.ly(900.0) - 60); p1 = (1460.0, self.ly(1460.0) + 30)
            C, A, em = R3.cut_layer(C, A, p0, p1, q, fall=1.3, side=-1)
            fr = R3.comp(fr, C, A)
            fr = fr + (em * 2.0 + cv2.GaussianBlur(em, (0, 0), 10) * 2.0)[..., None] * np.array([1.0, 0.5, 0.15], np.float32)
            if q < 2.5 / 30: fr = fr * 1.35 + np.array([0.35, 0.14, 0.04], np.float32) * (1 - q * 12)     # 2 帧冲击（橙闪，不发白）
            fr = RS.fx_sparks(fr, t, self.tcut, (1180, self.ly(1180)), 1501, n=90, spd=1300, dirv=(0.6, -0.8), spread=1.4, grav=1500, amt=1.4)
        else:
            fr = R3.comp(fr, C, A)
        fr, info = put(fr, t, self.KEY, (tip[0] - self.off[0], tip[1] - self.off[1]), h=760, anc=(0.06, 0.96), rot=62,
                       rim=(1.0, 0.6, 0.3), rim_amt=2.5, rim_dir=(-0.6, 0.8), wind=0)
        fr = RS.fx_sparks(fr, t, t - 0.3, tip, int(t * 30) % 50 + 1230, n=50, spd=700, dirv=(-1, -0.6), spread=0.8, grav=1200, amt=1.2)
        fr = E.dust_motes(fr, t, 1233, n=50, amt=0.5, drift=(-30, -60), col=(1.0, 0.55, 0.25), size=(1, 3))
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
        self.l2 = Sub('就是对', cts(53, [1, 2, 3]), 'right', CT(53, 4) - 0.1, seed=121)

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
        self.l2 = Sub('就是错', cts(53, [5, 6, 7]), 'left', CT(53, 8) - 0.1, seed=131)

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
        if not hasattr(self, 'AB'):   # 两半是定格（各取上一镜字落定的一帧），只算一次
            self.AB = (self.a.text(self.a.render(211.05), 211.05), self.b.text(self.b.render(212.0), 212.0))
        A, B = self.AB
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
        self.chips = R3.Shreds(125, n=46, gold=0.85, size=(14, 46), spread=(1100, 520), zr=(0.6, 2.6))   # 碎金匾箔片（有厚度、翻转、高光）
        self.l = Sub('何须伪装', cts(53, [8, 9, 10, 11]), 'right', CT(53, 11) + 0.25, red='伪装', seed=140)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr, info = put(fr, t, 'S125/stare', (960, 1170), h=1180 * (1 + 0.06 * u), rim=GOLDW, rim_amt=2.4, wind=6)
        fr = self.chips.draw(fr, (t - self.t0) * 0.5 + 0.2, to_cam=0.3, light=(0.6, -0.5, -0.6), dof=(3.0, 10.0), zfocus=1.4, emb=0.3)
        return bloom(fr, 0.8, 0.4)

    def text(self, fr, t): return self.l.draw(fr, t)


class E20(Shot):   # S126 ELS 巨大日轮占满画面；「天」字在日中被烧化
    def __init__(self):
        self.tex = K.fbm(H // 2, W // 2, 1260, base=24, octaves=5)
        self.tex2 = K.fbm(H // 2, W // 2, 1261, base=60, octaves=3)
        self.sky = E.dawn_sky(1262, horizon=0.95, mid=(0.5, 0.33, 0.16), hor=(1.0, 0.7, 0.35), cloud=0.7)
        self.l = Sub('不', [CT(54, 1)], 'center', CT(54, 2) + 0.1, seed=150)
        self.tt = CT(54, 0)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        c = (960, 520); R = 360 * (1 + 0.06 * u)
        yy, xx = E.grid()
        d = np.hypot(xx - c[0], yy - c[1])
        disc = np.clip((R - d) / 2.5, 0, 1)
        k = cv2.resize(self.tex * 0.6 + self.tex2 * 0.4, (W, H))
        k = R3.slide(k, 30 * (t - self.t0))
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


class E21(Shot):   # S127 CU（第三棒重做）逆光侧脸：低调光，脸沉在暗部、只留晨光勾出的额-鼻-唇-下颌金边与发丝（不再给写实五官，避免和 E16/E19 对不上）；
    #  「判」「我」同一套材质：红色拓印 + 金描边（不再一红一白混用），在天空侧、离脸
    GR = dict(expo=0.72, sat=0.55, tint=(1.06, 0.97, 0.88), lift=0.0, gamma=1.15)   # 第四棒：新精灵与 E16 同脸，不再藏五官

    def __init__(self):
        sky = E.dawn_sky(1270, horizon=0.7, sun=(1600, 300), mid=(0.42, 0.30, 0.17), cloud=0.6)
        sky = E.sun_disc(sky, (1600, 300), 90, amt=1.0)
        self.bg = cv2.GaussianBlur(sky, (0, 0), 14)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fx2.god_rays(fr, (1600, 300), strength=0.7, length=1.0, t=t)
        fr, info = put(fr, t, 'E_r4/profile_up', (560 + 40 * u, 1170), h=1080, rim=(1.0, 0.86, 0.6), rim_amt=4.0, rim_dir=(1.0, -0.35),
                       grade=self.GR, wind=12)
        fr = E.dust_motes(fr, t, 1271, n=80, amt=0.5, drift=(-20, -30), depth=True)
        return bloom(fr, 0.75, 0.55)

    def text(self, fr, t):
        to = CT(54, 4) - 0.1
        return R3.rub_text(fr, [('判', 1230, 400, 230, CT(54, 2), to, t), ('我', 1560, 600, 330, CT(54, 3), to, t)], seed=21)


class E22(Shot):   # S128 LS 横刀于胸，太阳恰落在刀身正中——他自己成了秤；「我来量」刻在刀身上（逐字烧亮）
    GR = dict(expo=0.55, sat=0.55, tint=(1.04, 0.97, 0.9), lift=0.0, gamma=1.2)

    def __init__(self):
        self.sky = E.dawn_sky(1280, horizon=0.86, mid=(0.42, 0.30, 0.16), cloud=0.6, sun=None)
        self.rd, self.ra = E.ridges(n=2, base=0.82, step=0.05, seed=128, amp=0.05, far=0.35, near=0.03)
        self.cal = None

    def _calib(self, t, dy):
        """刀身行：精灵 alpha 在 y∈[200,620] 内横向最宽的一行；刀身水平范围取该行 alpha>0.5 的首末列。"""
        z, info = put(np.zeros((H, W, 3), np.float32), t, 'S128/blade_level', (960, 1060 + dy), h=900, grade=self.GR, wind=0)
        A = info['A'] > 0.5
        wid = A[200:620].sum(1); r = 200 + int(np.argmax(wid))
        xs = np.where(A[r])[0]
        self.cal = dict(y=r - dy, x0=int(xs[0]), x1=int(xs[-1]))
        ch = '我来量'; ts = [CT(54, 3), CT(54, 4), CT(54, 5)]
        cx = [960 - 130, 960, 960 + 130]   # 双手之间的刀身段
        self.eng = Lyr([(c, cx[i], self.cal['y'] + 1, 30, 'serif_b', GOLD_T if i < 2 else REDT, ts[i]) for i, c in enumerate(ch)],
                       CT(55, 0) - 0.1, seed=170, glow=1.1, shadow=0.0, direction=(1.0, 0.0))
        self.big = Sub('我来量', ts, 'right', CT(55, 0) - 0.1, red='量', seed=171)

    def render(self, t):
        u = self.u(t); dy = -20 * u
        if self.cal is None: self._calib(self.t0, 0.0)
        by = self.cal['y'] + dy
        fr = self.sky.copy()
        z, info = put(np.zeros((H, W, 3), np.float32), t, 'S128/blade_level', (960, 1060 + dy), h=900, rim=GOLDW, rim_amt=3.0, grade=self.GR, wind=6)
        sun = (960, by)                                                  # 日心 = 刀身中点
        fr = E.sun_disc(fr, sun, 130, amt=1.15)
        fr = fx2.god_rays(fr, sun, strength=1.0, length=1.0, t=t, occluder=info['A'])
        fr = E.place_layer(fr, self.rd, self.ra)
        a = info['A'][..., None]
        fr = fr * (1 - a) + z * a
        # 日光沿刃线溢出（刀身受光的一条金线）
        yy = np.arange(H, dtype=np.float32)[:, None]; xx = np.arange(W, dtype=np.float32)[None, :]
        span = sstep(self.cal['x0'] + 20, self.cal['x0'] + 120, xx) * (1 - sstep(self.cal['x1'] - 120, self.cal['x1'] - 10, xx))
        edge = np.exp(-((yy - (by - 9)) / 1.6) ** 2) * span * (0.5 + 0.5 * np.exp(-((xx - 960) / 260) ** 2))
        fr = fr + edge[..., None] * GOLDW * 1.4
        self.by = by
        return bloom(fr, 0.8, 0.5)

    def text(self, fr, t):
        fr = self.eng.draw(fr, t, dy=getattr(self, 'by', self.cal['y']) - self.cal['y'] if self.cal else 0)
        return self.big.draw(fr, t)


class E23(Shot):   # S129 MCU 后撤蓄力，速度坡度推近（吸气）
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1290, base=4, octaves=5)
        self.bg = cv2.resize(((0.10 + 0.18 * n)[..., None] * np.array([1.0, 0.75, 0.45], np.float32)).astype(np.float32), (W, H))
        self.l = Sub('这一', cts(55, [0, 1]), 'left', CT(55, 2) - 0.08, seed=180)
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        self.ang = np.arctan2(yy - 300, xx - 480); self.rad = np.hypot(xx - 480, yy - 300) / 540
        r = np.random.default_rng(129); self.ph = r.uniform(0, 6.28, 7); self.fq = r.integers(40, 160, 7)

    def zoom_lines(self, t, u):
        """径向速度线（以人物胸口为灭点），强度随速度坡度上升 → 吸气蓄力。"""
        v = np.zeros_like(self.ang)
        for p, f in zip(self.ph, self.fq): v += np.sin(self.ang * f + p + t * 0.7)
        v = np.clip((v / 3.2) - 0.55, 0, 1) ** 2 * sstep(0.25, 0.9, self.rad)
        L = cv2.resize(v, (W, H))[..., None]
        return L * np.array([1.0, 0.8, 0.5], np.float32) * (0.10 + 0.45 * ein(u, 2))

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = fr + self.zoom_lines(t, u)
        fr, info = put(fr, t, 'S129/coil', (1000, 1140), h=1080, rim=GOLDW, rim_amt=2.8, wind=12)
        return bloom(fr, 0.8, 0.35)

    def text(self, fr, t): return self.l.draw(fr, t)


class E24(Shot):   # S129b ECU 刃身满屏，寒光自柄扫到尖；巨字「刀」朱红
    def __init__(self):
        self.bg = E.dawn_sky(1291, horizon=0.5, top=(0.06, 0.05, 0.04), mid=(0.25, 0.18, 0.1), hor=(0.6, 0.42, 0.22), cloud=0.3)
        self.l1 = Lyr([('刀', 1440, 520, 440, 'serif_black', REDT, CT(55, 2))], CT(55, 6) - 0.05, seed=190)
        self.l2 = Sub('就是我', cts(55, [3, 4, 5]), 330, CT(55, 6) - 0.05, seed=191)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr, info = put(fr, t, 'S129/coil|crop=0,0,0.46,0.42', (900 - 60 * u, 470), w=2300, anc=(0.5, 0.5), rot=-12, rim=GOLDW, rim_amt=3.0, wind=0, grade=G_HOT)
        sx = -300 + 2600 * eio(u)
        fr = E.flare_streak(fr, (sx, 420 + (sx - 960) * 0.18), amt=1.1, angle=10)
        return bloom(fr, 0.78, 0.5)

    def text(self, fr, t): return self.l2.draw(self.l1.draw(fr, t), t)


class E25(Shot):   # S130（第三棒重做）低机位终极横斩：墨黑红天（次色调），人物受刀光照亮、占满画面下 3/4；整个画面沿刀线裂开、金光喷涌
    def __init__(self):
        self.sky = E.dawn_sky(1300, horizon=0.92, top=(0.02, 0.012, 0.012), mid=(0.17, 0.05, 0.03), hor=(0.62, 0.20, 0.08), cloud=0.9, sun=None)
        n = K.fbm(H // 2, W // 2, 1302, base=10, octaves=5)
        self.ground = cv2.resize(((0.02 + 0.05 * n)[..., None] * np.array([1.0, 0.7, 0.6], np.float32)).astype(np.float32), (W, H))
        self.s1 = Sub('的', [CT(55, 6)], 450, CT(55, 8) - 0.12, seed=200)
        self.l = LyrO([('正', 450, 470, 330, 'serif_black', GOLD_T, CT(55, 7))], CT(55, 8) - 0.12, seed=201, glow=0.6, outline=4)

    def render(self, t):
        u = self.u(t)
        fr = self.sky.copy()
        hy = 930
        m = sstep(hy - 2, hy + 20, np.arange(H, dtype=np.float32))[:, None, None]
        fr = fr * (1 - m) + self.ground * m
        ts = CT(55, 6)
        mb = (90, 0) if t < ts + 0.04 else (0, 0)
        fr, info = put(fr, t, 'S130/hslash', (930, 1200), h=1040, rim=(1.0, 0.7, 0.4), rim_amt=3.4, rim_dir=(-0.9, -0.3), wind=12, mblur=mb,
                       grade=dict(expo=1.15, sat=0.62, tint=(1.1, 0.95, 0.86), lift=0.02, gamma=1.0))
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
            if q < 0.08: fr = fr * 1.3 + np.array([0.3, 0.12, 0.04], np.float32) * (1 - q / 0.08)
            fr = RS.fx_sparks(fr, t, ts, (300, by), 1301, n=90, spd=1400, dirv=(1, -0.1), spread=0.35, grav=300, amt=1.2)
        fr = RS.fx_embers(fr, t, 1303, amt=0.7, n=50)
        return bloom(fr, 0.8, 0.6)

    def text(self, fr, t): return self.l.draw(self.s1.draw(fr, t), t)


class E26(Shot):   # S131（第三棒重做）一点透视：他划下的那道火线贴地一路延伸到地平线（纵深），朝阳正从线的尽头升起；
    #  画面里没有人（不再是日落剪影），「正道沧桑」大字描边升在天上
    def __init__(self):
        self.sky = E.dawn_sky(1310, horizon=0.55, mid=(0.45, 0.31, 0.17), hor=(1.0, 0.82, 0.5), cloud=0.8, sun=(960, 590))
        self.g = R3.Ground(261)
        self.l = LyrO(row('正道沧桑', [CT(55, 7), CT(55, 8), CT(55, 9), CT(55, 10)], 960, 300, size=170, gold='正道', track=0.22, font='serif_black'),
                      None, seed=210, glow=0.5, outline=4)

    def render(self, t):
        u = self.u(t)
        hy = 600
        fr = self.sky.copy()
        sun = (960, hy + 90 - 170 * eout(u, 2))
        fr = E.sun_disc(fr, sun, 120, amt=1.2)
        fr = fx2.god_rays(fr, sun, strength=0.8, length=1.0, t=t)
        fr, z = self.g.render(fr, hy, f=900, camh=1.0, dz=-6 * u, line=1.0, fog=(1.0, 0.8, 0.5))
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        fr = fr + np.exp(-((yy - hy) / 2.0) ** 2) * GOLDW * 1.4 + np.exp(-((yy - hy) / 24) ** 2) * GOLD * 0.5
        fr = RS.fx_embers(fr, t, 1312, amt=0.6, n=40)
        return bloom(fr, 0.85, 0.5)

    def text(self, fr, t): return self.l.draw(fr, t)


# ---- 高潮碎切（221.27–223.10，按鼓点 221.28 / 221.97 / 222.65 切）：眼 → 斩 → 匾碎，每镜 0.5–0.67s，取代原 1.83s 慢走
class E27a(Shot):   # 眼：S125 眼部 ECU（第二次用，换景别），墨黑红底，红色轮廓光，急推
    def render(self, t):
        u = self.u(t)
        fr = np.zeros((H, W, 3), np.float32) + np.array([0.03, 0.01, 0.01], np.float32)
        fr, info = put(fr, t, 'S125/stare|crop=0.30,0.31,0.70,0.51', (960, 540), w=2500 * (1 + 0.12 * eout(u)), anc=(0.5, 0.5),
                       rim=(1.0, 0.35, 0.2), rim_amt=2.5, rim_dir=(1.0, 0.0), wind=4,
                       grade=dict(expo=0.85, sat=0.55, tint=(1.12, 0.92, 0.86), lift=0.0, gamma=1.15))
        yy, xx = E.grid()
        fr = fr * (0.35 + 0.8 * np.exp(-((yy - 560) / 260) ** 2))[..., None]                      # 只留眼睛这条光带
        sx = 300 + 1400 * eio(u * 1.4)
        fr = E.flare_streak(fr, (sx, 560), amt=0.6, angle=0, length=0.9, col=(1.0, 0.55, 0.35))
        return bloom(fr, 0.8, 0.4)


class E27b(Shot):   # 斩：纯刀光——一道巨大的金白弧光自左上劈到右下，拖出燃烧的刀痕、火星四溅，暗底震屏（不用精灵）
    def render(self, t):
        u = self.u(t)
        fr = np.zeros((H, W, 3), np.float32) + np.array([0.03, 0.018, 0.015], np.float32)
        fr = fx2.light_beam(fr, (1500, 60), (800, 1000), 40, 500, t=t, strength=0.25, seed=27)
        pr = eout(c01(0.3 + u / 0.45), 4)
        yy, xx = E.grid()
        c = (2300, -900); R = 1900
        d = np.hypot(xx - c[0], yy - c[1])
        ang = np.arctan2(yy - c[1], xx - c[0])
        a0, a1 = math.radians(175), math.radians(175 - 70 * pr)
        on = ((ang <= a0) & (ang >= a1)).astype(np.float32)
        head = np.exp(-((ang - a1) / 0.05) ** 2)
        core = np.exp(-((d - R) / (4 + 10 * head)) ** 2) * on
        glow = np.exp(-((d - R) / 60) ** 2) * on
        trail = np.exp(-((d - R - 30) / 22) ** 2) * on * (0.6 + 0.4 * np.sin(ang * 90 + t * 40)) * math.exp(-u * 1.5)
        fr = fr + core[..., None] * np.array([1.0, 0.96, 0.88], np.float32) * 2.2 + glow[..., None] * GOLD * 0.9 * (1 - 0.5 * u) \
             + trail[..., None] * np.array([1.0, 0.35, 0.08], np.float32) * 1.4
        tipx = c[0] + R * math.cos(a1); tipy = c[1] + R * math.sin(a1)
        fr = RS.fx_sparks(fr, t, self.t0 + 0.03, (tipx, tipy), 2701, n=120, spd=1500, dirv=(0.5, 0.8), spread=1.2, grav=1400, amt=1.4)
        if 0 <= tipx <= W and 0 <= tipy <= H: fr = E.flare_streak(fr, (tipx, tipy), amt=1.0 * (1 - u), angle=-35, length=0.7)
        if u < 0.1: fr = fr * 1.2 + np.array([0.35, 0.16, 0.05], np.float32) * (1 - u / 0.1)
        fr = RS.fx_embers(fr, t, 2702, amt=0.8, n=60)
        # 第四棒：刀光不再无主 —— 他（低机位跨步精灵翻转、第二次用：墨黑剪影 + 朱金轮廓光）随刀光落地，刀尖正是弧光的尽头
        fr, info = put(fr, t, 'E_r4/low_stride|flip', (900 - 60 * pr, 1290 - 40 * pr), h=1080, rim=(1.0, 0.55, 0.25), rim_amt=4.5, rim_dir=(0.7, -0.6),
                       grade=dict(expo=0.30, sat=0.5, tint=(1.05, 0.9, 0.85), lift=0.0, gamma=1.3), wind=16, mblur=(int(30 * (1 - pr)), 0))
        return bloom(fr, 0.7, 0.7)


class E27c(Shot):   # 匾碎：E04/E05 那块「天理昭彰」金匾终于被斩碎——裂开、金箔碎片扑面，匾后是黑
    def __init__(self):
        self.pl = PR.Plaque('天理昭彰', seed=3)
        self.burst = R3.Shreds(271, n=60, gold=0.9, size=(30, 110), spread=(900, 360), zr=(0.6, 2.0))

    def render(self, t):
        u = self.u(t)
        fr = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.016, 0.014], np.float32)
        base = fr.copy()
        sc = 1.0 + 0.25 * eout(u)
        fr = plaque_lit(self.pl, fr, t, 960, 520, 1500 * sc, [t - 9] * 4, crack=1.0, dim=0.7, tilt=-4 + 8 * u)
        fr = base + (fr - base) * (1 - 0.7 * eio(u))                       # 匾碎后沉入黑
        fr = self.burst.draw(fr, 0.2 + u * 1.1, to_cam=1.3, light=(0.2, -0.5, -0.8), dof=(1.5, 10.0), zfocus=1.0, emb=0.2)
        if u < 0.15: fr = fr + np.array([0.5, 0.3, 0.1], np.float32) * (1 - u / 0.15)
        return bloom(fr, 0.75, 0.6)


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


class Title(Shot):   # 片名底板（第三棒改）：不再是金色云海空镜，改墨黑底 + 低处缓缓涌动的暖金烟 + 余烬上飘（墨黑红次色调）
    def __init__(self):
        n = K.fbm(H // 2, W // 2, 1340, base=4, octaves=5)
        self.ink = cv2.resize(((0.015 + 0.03 * n)[..., None] * np.array([1.0, 0.85, 0.75], np.float32)).astype(np.float32), (W, H))
        self.sm = [K.fbm(H // 4, W // 2, 1341 + i, base=3 + 2 * i, octaves=5) for i in range(2)]

    def base(self, t):
        fr = self.ink.copy()
        yv = np.arange(H, dtype=np.float32)[:, None, None] / H
        for i, sm in enumerate(self.sm):
            m = cv2.resize(R3.slide(sm, (t - 224.0) * (8 + 10 * i)), (W, H))
            m = np.clip((m - 0.45) * 2.2, 0, 1)[..., None] * sstep(0.35, 0.95, yv)
            fr = fr + m * np.array([0.55, 0.28, 0.10], np.float32) * (0.35 + 0.25 * i)
        fr = RS.fx_embers(fr, t, 1342, amt=0.7, n=60)
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
            for i, (ch, x) in enumerate((('鬼', 505), ('屠', 705), ('第', 1015), ('三', 1215), ('季', 1415))):
                e = c01(p * 1.6 - i * 0.12)
                T.place(L, ch, x, 470 - 20 * (1 - eout(e)), 200, GOLD_T, 'kai', alpha=eout(e), texture=True)
            fr = T.compose(fr, L, shadow=0.75, shadow_sigma=24, glow=0.5, glow_sigma=18, glow_color=(1.0, 0.6, 0.25))
            cv2.circle(fr, (860, 475), int(13 * eout(p)), tuple(float(v) for v in REDT), -1, cv2.LINE_AA)
            s2 = c01((q - 0.75) / 0.5)
            if s2 > 0:
                L2 = T.Layer()
                for i, ch in enumerate('主题曲《谁定正邪》'):
                    T.place(L2, ch, 960 + (i - 4) * 58, 650, 46, WHITE_T, 'serif_m', alpha=eout(s2), texture=False)
                fr = T.compose(fr, L2, shadow=1.0, shadow_sigma=8, glow=0.2, glow_sigma=10)
            sp = c01((q - 0.55) / 0.5)
            if sp > 0: fr = T.stamp_seal(fr, '鬼屠', 1610, 470, 130, sp, style='yin', rot=-5, seed=9)
        return bloom(fr, 0.85, 0.4)


class Credits(Shot):
    blocks = []      # (role, names, x, y_top, t_start, dur)
    def credits(self, fr, t):
        for (role, names, x, y, ts, dur) in self.blocks:
            p = c01((t - ts) / dur)
            out = c01((t - (self.t1 - 0.35)) / 0.35)
            if p <= 0: continue
            g = R3.credit_block_up(fr.copy(), role, names, x, y, p, role_size=32, name_size=66, seal_text=(role[0] if names[0][0].isascii() else names[0][0]),
                               seal_size=46, color='white', name_font='kai', seed=hash(role) % 50)
            fr = g * (1 - out) + fr * out
        return fr

    def text(self, fr, t): return self.credits(fr, t)


class E31(Credits):   # S134 ELS 晨雾群山缓移；署名①：出品人 / 编剧·导演
    def __init__(self):
        self.sky = E.dawn_sky(1350, horizon=0.62, top=(0.05, 0.06, 0.08), mid=(0.30, 0.33, 0.38), hor=(0.86, 0.80, 0.72), cloud=0.5, sun=(560, 560))
        self.rd, self.ra = E.ridges(n=5, base=0.50, step=0.07, seed=134, amp=0.11, haze=(0.62, 0.68, 0.76), far=0.75, near=0.05)
        self.blocks = [('出品人', ['西几王传媒'], 1660, 230, 227.85, 0.9), ('编剧 / 导演', ['苏打'], 1430, 230, 228.45, 0.8)]
        self.fog = K.fbm(H // 4, W // 2, 1351, base=5, octaves=5)

    def render(self, t):
        u = self.u(t)
        fr = E.sun_disc(self.sky.copy(), (560, 600), 60, amt=0.55)
        fr = E.place_layer(fr, self.rd, self.ra, ox=-120 * u, z=1.05)
        f = cv2.resize(R3.slide(self.fog, 30 * u), (W, H))
        m = np.clip((f - 0.45) * 2.5, 0, 1) * sstep(0.45, 0.85, np.arange(H, dtype=np.float32)[:, None] / H)
        fr = fr * (1 - m[..., None] * 0.7) + m[..., None] * np.array([0.80, 0.82, 0.84], np.float32) * 0.6
        fr = E.dust_motes(fr, t, 1352, n=40, amt=0.25, drift=(20, -6), col=(0.9, 0.9, 0.95))
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
        fr, info = put(fr, t, 'S135/sheath_hand|crop=0.28,0.42,0.95,1', (760 - 140 * u, 560), w=1900, anc=(0.5, 0.5), rim=GOLDW, rim_amt=2.6, wind=3)
        sx = -200 + 1600 * eio(u)
        fr = E.flare_streak(fr, (sx, 520 + (sx - 700) * 0.6), amt=0.6, angle=58, length=0.6)
        return bloom(fr, 0.8, 0.35)


class E33(Credits):   # S135b（第三棒重做）晨后冷调：贴地低机位，他划下的那道线已冷成一道暗红焦痕伸向远方，晨雾贴地，远处一线淡日；署名③：AI画师
    def __init__(self):
        self.sky = E.dawn_sky(1370, horizon=0.52, top=(0.10, 0.12, 0.15), mid=(0.42, 0.46, 0.52), hor=(0.88, 0.86, 0.82), cloud=0.4, sun=(760, 560))
        self.g = R3.Ground(331, tone=(0.85, 0.88, 0.95))
        self.blocks = [('AI画师', ['沉溺', '雪晴', '索恩'], 1640, 230, 232.45, 1.3)]
        self.fog = K.fbm(H // 4, W // 2, 1372, base=5, octaves=5)
        # 第四棒：地平线不再是一刀硬线 —— 远山两层（被晨雾吃到只剩轮廓）+ 贴地平线的雾带 + 一排远处残碑剪影作焦点
        self.rd, self.ra = E.ridges(n=2, base=0.455, step=0.035, seed=337, amp=0.045, haze=(0.80, 0.82, 0.86), far=0.85, near=0.35)
        r = np.random.default_rng(338); self.stl = [(r.uniform(-1, 1), r.uniform(0.7, 1.3)) for _ in range(9)]

    def render(self, t):
        u = self.u(t)
        hy = 565
        fr = E.sun_disc(self.sky.copy(), (760, hy - 10), 40, amt=0.45, core=(1.0, 0.98, 0.95))
        fr = E.place_layer(fr, self.rd, self.ra, ox=-40 * u, z=1.0)
        fr, z = self.g.render(fr, hy, f=900, camh=0.6, dz=-3 * u, cx=760, line=0.6, line_col=(0.55, 0.08, 0.04), fog=(0.85, 0.87, 0.9))
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        for k, (xo, sc) in enumerate(self.stl):                       # 远处残碑剪影（线的两侧、地平线上）
            x = 760 + (xo * 520 if abs(xo) > 0.25 else xo * 520 + np.sign(xo) * 200) - 25 * u
            wv, hv = 9 * sc, 34 * sc * (1.3 if k % 3 == 0 else 1.0)
            cv2.rectangle(fr, (int(x - wv / 2), int(hy - hv)), (int(x + wv / 2), hy + 2), (0.36, 0.38, 0.42), -1, cv2.LINE_AA)
        band = np.exp(-((yy - hy + 6) / 38.0) ** 2)                    # 地平线雾带：软化天/地交界
        fr = fr * (1 - 0.75 * band) + band * 0.75 * np.array([0.84, 0.86, 0.89], np.float32)
        f = cv2.resize(R3.slide(self.fog, 25 * u), (W, H))
        m = np.clip((f - 0.4) * 2.0, 0, 1) * sstep(0.45, 0.62, np.arange(H, dtype=np.float32)[:, None] / H) * (1 - sstep(0.7, 0.95, np.arange(H, dtype=np.float32)[:, None] / H))
        fr = fr * (1 - m[..., None] * 0.6) + m[..., None] * np.array([0.82, 0.84, 0.88], np.float32) * 0.55
        fr = E.dust_motes(fr, t, 1373, n=40, amt=0.25, drift=(15, -5), col=(0.9, 0.92, 0.96))
        return bloom(fr, 0.85, 0.35)


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
        fr, info = put(fr, t, 'S132/back_walk', (900 + 25 * u, 756), h=135, rim=GOLDW, rim_amt=4.0, grade=G_SIL, look='sil', wind=4)
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
        fr, info = put(fr, t, 'S137/sheath_close|crop=0,0.58,1,1', (960 - slide, 600), w=2200, anc=(0.5, 0.5), rim=GOLDW, rim_amt=2.4, wind=2,
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
    (E01, 5797, 'S111', 'ECU'), (E02, 5823, 'S112', 'LS'), (E03, 5868, 'S113', 'ELS'), (E04, 5916, 'S114', 'CU'),
    (E05, 5951, 'S114b', 'ECU'), (E06, 5979, 'S115', 'MS'), (E07, 6020, 'S116', 'LS'), (E08, 6061, 'S117', 'CU'),
    (E09, 6102, 'S118', 'MS'), (E10, 6130, 'S119', 'ELS'), (E11, 6157, 'S120a', 'MS'), (E12, 6182, 'S120b', 'ELS'),
    (E13, 6218, 'S121', 'MCU'), (E14T, 6263, 'S122', 'LS'), (E15, 6295, 'S122b', 'ECU'), (E16, 6310, 'S123', 'MCU'),
    (E17, 6338, 'S124', 'MCU'), (E18, 6363, 'S124b', 'MS'), (E19, 6377, 'S125', 'CU'), (E20, 6418, 'S126', 'ELS'),
    (E21, 6459, 'S127', 'CU'), (E22, 6501, 'S128', 'LS'), (E23, 6534, 'S129', 'MCU'), (E24, 6548, 'S129b', 'ECU'),
    (E25, 6569, 'S130', 'LS'), (E26, 6610, 'S131', 'ELS'),
    (E27a, 6638, 'S132', 'ECU'), (E27b, 6658, 'S132x', 'ECU'), (E27c, 6678, 'S132y', 'CU'), (E28, 6693, 'S132b', 'ECU'), (E29, 6720, 'S133', 'MS'), (E30, 6762, 'S133b', 'MS'),
    (E31, 6830, 'S134', 'ELS'), (E32, 6900, 'S135', 'CU'), (E33, 6968, 'S135b', 'ELS'), (E34, 7037, 'S136', 'ELS'),
    (E35, 7119, 'S137', 'ECU'), (E36, 7174, 'S138', '—'),
]
IDS = [s[0].__name__ for s in SHOTS]
BOUNDS = [s[1] for s in SHOTS] + [F1]
CHORUS_END = 6638

# 入点转场：(名称, pre 秒, post 秒, kw)；'hard' = 切后 2 帧冲击（推近+提亮）
CUTS = {   # 第三棒：亮底上不再用白色系转场（flash_white 0 处、whip 1 处、zoom_through 0 处），换成暗色系：墨晕吞屏 / 刀光黑切 / 印章盖屏
    'E01': ('entry_flash', 0, 0.22, {}),
    'E02': ('hard', 0, 0, {}),
    'E03': ('ink_swallow', 0.10, 0.22, dict(center=(1150, 360), seed=3)),
    'E04': ('hard', 0, 0, {}),
    'E05': ('whip_pan_blur', 0.05, 0.10, dict(direction='right')),
    'E06': ('occlusion', 0.08, 0.16, dict(kind='pillar', direction=-1, color=(0.03, 0.02, 0.015))),
    'E07': ('hard', 0, 0, {}),
    'E08': ('hard', 0, 0, {}),
    'E09': ('paper_tear', 0.06, 0.18, dict(seed=5)),
    'E10': ('blade_black', 0.06, 0.12, dict(angle=-25, width=200)),
    'E11': ('hard', 0, 0, {}),
    'E12': ('seal_cover', 0.08, 0.16, dict(center=(820, 500), rot=-4)),
    'E13': ('hard', 0, 0, {}),
    'E14T': ('hard', 0, 0, {}),
    'E15': ('hard', 0, 0, {}),
    'E16': ('door_open', 0.0, 0.20, dict(cx=960)),             # 第四棒：E15 沿竖缝裂开两扇滑出 → E16 自己的中缝（图形匹配）
    'E17': ('push_h', 0.07, 0.13, {}),                         # 第四棒：横推，中间帧 = 金脸 | 朱缝 | 朱脸 正面相对
    'E18': ('hard', 0, 0, {}),
    'E19': ('shatter', 0.04, 0.22, dict(center=(960, 540))),
    'E20': ('iris_moon', 0.0, 0.30, dict(center=(960, 520), radius=120, sun=True)),
    'E21': ('flash_red', 0.0, 0.16, {}),
    'E22': ('match_cut', 0.0, 0.18, dict(shape='circle', center=(960, 330), radius=120)),
    'E23': ('hard', 0, 0, {}),
    'E24': ('hard', 0, 0, {}),
    'E25': ('slash_wipe', 0.03, 0.12, dict(angle=-8)),
    'E26': ('match_cut', 0.0, 0.22, dict(shape='line', center=(960, 600), angle=0)),
    'E27a': ('hard', 0, 0, {}),
    'E27b': ('hard', 0, 0, {}),
    'E27c': ('hard', 0, 0, {}),
    'E28': ('blade_black', 0.04, 0.10, dict(angle=35, width=170)),
    'E29': ('ink_swallow', 0.18, 0.18, dict(center=(1000, 500), seed=9)),
    'E30': ('hard', 0, 0, {}),
    'E31': ('ink_bleed', 0.0, 0.40, dict(center=(960, 540), white=False, seed=9)),
    'E32': ('occlusion', 0.10, 0.18, dict(kind='blade', direction=1)),
    'E33': ('occlusion', 0.10, 0.20, dict(kind='branch', direction=-1, color=(0.03, 0.035, 0.04))),
    'E34': ('ink_bleed', 0.0, 0.45, dict(center=(960, 620), white=False, seed=8)),
    'E35': ('hard', 0, 0, {}),
    'E36': ('cut', 0, 0, {}),
}
CUSTOM = {'blade_black': R3.blade_black, 'seal_cover': R3.seal_cover, 'ink_swallow': R3.ink_swallow, 'push_h': R3.push_h, 'door_open': R3.door_open}

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
    'E27a': ('expo_out', 1.15, 1.0, 0, 0, 0, 0, 0, 0),
    'E27b': ('expo_out', 1.08, 1.0, 40, -40, 0, 0, -2, 1),
    'E27c': ('expo_out', 1.0, 1.1, 0, 0, 0, 0, 0, 0),
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
    'E25': [(CT(55, 6), 18)], 'E27b': [(6658 / 30, 14)], 'E27c': [(6678 / 30, 18)], 'E15': [(210.058, 16)], 'E30': [(TS, 8)], 'E35': [(238.24 - 1 / 30, 5)], 'E20': [(CT(54, 2), 10)],
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
            A = np.clip(A, 0, 4).astype(np.float32); B = np.clip(B, 0, 4).astype(np.float32)
            if name in CUSTOM: return CUSTOM[name](A, B, p, **kw), 2
            return TR.transition(name, A, B, p, **kw), 2
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
