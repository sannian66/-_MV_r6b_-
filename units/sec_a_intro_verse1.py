"""units/sec_a_intro_verse1.py — A 组：前奏 + 主歌一（帧 0–1321，0–44.03s），第四轮重做。
设计说明：crew/round4/A_storyboard.md（29 镜，平均 1.52s；主歌 21 镜平均 1.37s）。
接口：READY / frame(f, cache) → uint8 (1080,1920,3) 成品帧（含黑边、颗粒、暗角）。
结构仿 units/chorus1.py：每镜一个类 render(t)->(线性帧, ca)；CUTS 表驱动转场；VCam3 主运镜 + 卡点脉冲；屏幕层 = 署名 / 歌词竖排。"""
import os, sys, math, json
ROOT = '/workspace/work/mv'
for p in (ROOT, ROOT + '/lib', ROOT + '/units'):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault('SPRITE_MODE', '1')
import numpy as np, cv2
import c1kit as K
import typo as T
import trans as TR
import fx2
import props as PR
import vcam3 as V3
import vcam as VC
import a_kit as A
from a_kit import clamp01, eio, eout, ein, COLD, GOLD, REDR, WARM

READY = False
FPS = 30
W, H, BAR = 1920, 1080, 120
F0, F1 = 0, 1321
HMP = ROOT + '/crew/round4/A_hitmap.json'
HM = json.load(open(HMP))
WT = {(w['line'], w['k']): w['t'] for w in HM['words']}
WT[(0, 1)] = 16.08                     # whisper 把「长夜」压在同一时刻，「夜」顺延
RED = np.array([0.72, 0.06, 0.05], np.float32)
GOLDV = np.array([1.0, 0.76, 0.40], np.float32)
COLDV = np.array([0.62, 0.74, 1.0], np.float32)
INK = np.array([0.02, 0.022, 0.03], np.float32)


def V(t): return (round(t * FPS) - 1) / FPS            # 声音 → 视觉帧（早 1 帧）
def CT(li, k): return V(WT[(li, k)])
def cprog(li, ks, t, dur=0.22):
    return [clamp01((t - (CT(li, k) - dur)) / dur) for k in ks]


def S(key, mode='cut', **kw): return A.src(key, mode, **kw)


class Shot:
    t0 = t1 = 0.0
    def u(self, t): return clamp01((t - self.t0) / (self.t1 - self.t0))


def night(top=(0.010, 0.013, 0.024), bot=(0.06, 0.07, 0.10), y1=H):
    return A.vgrad(top, bot, 0, y1, 1.3)


def pupil_glint(fr, p, g, size=1.0, col=(1.0, 0.96, 0.9)):
    if g <= 0.01: return fr
    q = 2; yy, xx = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
    X, Y = xx * q - p[0], yy * q - p[1]
    st = np.exp(-(np.abs(X) / (3 * size) + np.abs(Y) / (60 * size))) + np.exp(-(np.abs(Y) / (2.5 * size) + np.abs(X) / (160 * size)))
    st += np.exp(-(X * X + Y * Y) / (200 * size * size)) * 1.5
    return fr + cv2.resize(st, (W, H))[..., None] * g * np.array(col, np.float32)


def streak(fr, y, x0, x1, q, col=(1, 0.97, 0.92), w=3, ang=0.0, amt=1.0):
    """刀光直线：q 0→1 由 x0 扫到 x1，带辉光尾迹。"""
    if q <= 0 or amt <= 0: return fr
    L = np.zeros((H // 2, W // 2), np.float32)
    xe = x0 + (x1 - x0) * min(1, q); xs = x0 + (x1 - x0) * max(0, q - 0.45)
    th = math.radians(ang)
    def P(x): return (int(x / 2), int((y + (x - W / 2) * math.tan(th)) / 2))
    cv2.line(L, P(xs), P(xe), 1.0, max(1, w // 2), cv2.LINE_AA)
    L = cv2.resize(L, (W, H))
    fade = 1.0 if q < 1 else max(0, 1 - (q - 1) * 3)
    return fr + (L * 1.6 + cv2.GaussianBlur(L, (0, 0), 14) * 1.4 + cv2.GaussianBlur(L, (0, 0), 50) * 0.8)[..., None] * np.array(col, np.float32) * fade * amt


def fog(fr, t, seed, y0, y1, amt, col=(0.45, 0.5, 0.6), speed=(30, 0), light=None, cam=(0, 0)):
    return FOG[seed % 3].draw(fr, t, density=amt, y0=y0, y1=y1, color=col, speed=speed, light=light, cam=cam)


FOG = [fx2.Fog(1), fx2.Fog(2), fx2.Fog(3)]
RAIN = fx2.RainSheet(5, density=0.8)
EMB = fx2.Embers(11, n=160, mode='fire')
ASH = fx2.Embers(12, n=140, mode='ash', rise=60)
GDUST = fx2.Embers(13, n=120, mode='gold', rise=40)


# =====================================================================================
# 前奏
# A01 0.00–2.21 ELS 墨滴开天（第四棒：去月）：墨晕开出云海，光藏在云后——三道体积光束扫过云海，云层快速流动 + 三层山脊视差
class A01(Shot):
    def __init__(self):
        self.bg = night((0.012, 0.016, 0.03), (0.09, 0.10, 0.13))
    def scene(self, t):
        fr = self.bg.copy()
        lx = 1180 + 120 * t                                              # 云后光源缓慢横移（光束随之扫）
        fr = A.glow(fr, (lx, 300), 520, 0.30, (0.55, 0.62, 0.82))
        fr = fx2.god_rays(fr, (lx, 280), strength=0.28, length=1.0, color=(0.7, 0.78, 0.95), t=t * 2.5, src_r=160)
        fr = A.clouds(fr, t, 3, speed=(70, 0), y0=0, y1=640, amt=1.0, col=(0.16, 0.18, 0.23), light=(lx, 300), light_r=480, thr=0.40)
        for i, (by, amp, tone) in enumerate(((720, 160, (0.05, 0.06, 0.08)), (800, 120, (0.03, 0.035, 0.05)), (900, 90, (0.012, 0.014, 0.02)))):
            fr, _ = A.ridge(fr, by, amp, 40 + i, tone=tone, rim=(0.5, 0.58, 0.75), rim_amt=0.35 - 0.1 * i, dx=-t * (20 + 40 * i))
            fr = fog(fr, t, i, by - 80, by + 60, 0.35, (0.30, 0.34, 0.42), speed=(40 + 30 * i, 0))
        return fr
    def render(self, t):
        sc = self.scene(t)
        T_DROP = V(0.37)
        if t < T_DROP:                                  # 墨滴：自上而下一道细亮线落向画面中心
            fr = np.zeros_like(sc) + 0.004
            q = clamp01(t / T_DROP)
            y = 120 + (560 - 120) * q ** 2
            fr = A.glow(fr, (960, y), 10, 1.2, (0.9, 0.92, 1.0))
            fr = A.glow(fr, (960, y - 40), 6, 0.6, (0.6, 0.7, 0.9), aspect=0.15)
            return fr, 0
        p = eout((t - T_DROP) / 1.55, 2)
        blk = np.zeros_like(sc) + 0.004
        fr = TR.ink_bleed(blk, sc, p, seed=4, center=(960, 560), edge=0.85)
        k = math.exp(-(t - T_DROP) / 0.12)
        fr = A.glow(fr, (960, 560), 60 + 300 * (1 - k), 0.8 * k, (0.8, 0.85, 1.0))
        return fr, (2 if k > 0.5 else 0)


# A02 2.21–4.06 LS 侧逆光山脊（第四棒：人物移到左 1/3，巨月只露右侧一角出框；避开副歌一「满月前居中剪影」）
class A02(Shot):
    def __init__(self):
        self.bg = night((0.02, 0.025, 0.045), (0.07, 0.08, 0.11))
    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = A.moon(fr, 1830, 400, 360, 0.95, halo=0.7)
        fr = A.clouds(fr, t, 7, speed=(-60, 0), y0=300, y1=760, amt=0.65, col=(0.22, 0.24, 0.30), light=(1830, 400), light_r=420, thr=0.47)
        fr, _ = A.ridge(fr, 940, 260, 52, tone=(0.035, 0.04, 0.055), rim=(0.6, 0.68, 0.85), rim_amt=0.25, dx=-u * 60)
        fr, rl = A.ridge(fr, 1010, 150, 57, tone=(0.006, 0.007, 0.01), rim=(0.75, 0.82, 1.0), rim_amt=0.5, peaks=2, dx=u * 70)
        fy = float(rl[640]) + 6
        fr, info = A.place(fr, t, S('S002/back_wind'), (640 + 30 * u, fy), h=500, rim=COLD, rim_amt=3.6, rim_dir=(0.9, -0.4), wind=13,
                           grade=dict(expo=0.55, sat=0.5, gamma=1.25), wrap=0.6)
        fr = fog(fr, t, 1, fy - 60, 1000, 0.5, (0.25, 0.28, 0.35), speed=(120, 0))
        fr = ASH.draw(fr, t, 0.7, wind=(220, 10))
        return fr, 0


# A03 4.06–5.97 ECU 拇指推刀出鞘（第四棒：4.53 拍上刀身真实滑出 40px + 鞘口迸尘、手指收紧微颤、扫光）
class A03(Shot):
    def __init__(self):
        self.path = S('S003/thumb_push', 'dm')
        self.bg = night((0.01, 0.012, 0.02), (0.03, 0.035, 0.05))
    def render(self, t):
        u = self.u(t)
        fr = A.glow(self.bg.copy(), (1500 - 200 * u, 300), 420, 0.22, (0.5, 0.58, 0.8))
        push = eout((t - V(4.53)) / 0.35) * 40 + 8 * eout((t - V(5.46)) / 0.25)
        jit = 1.5 * math.sin(t * 47) * clamp01((t - V(4.53)) / 0.1) * math.exp(-max(0, t - V(4.53)) / 0.4)
        fr, info = A.place(fr, t, self.path, (860 - 30 * eio(u) - push, 610 + jit), w=2150 * (1 + 0.04 * u), anc=(0.5, 0.42), rim=COLD,
                           rim_amt=1.4, wind=0.0, grade=dict(expo=0.78, sat=0.6, gamma=1.15), wrap=0.2, breathe=0.0)
        q = clamp01((t - V(4.53)) / 0.95)
        if 0 < q < 1:
            x = 200 + 1500 * eio(q)
            lum = fr.mean(-1); sat = fr.max(-1) - fr.min(-1)
            steel = np.clip((lum - 0.30) / 0.2, 0, 1) * np.clip((0.12 - sat) / 0.08, 0, 1) * info['A']
            xx = np.arange(W, dtype=np.float32)[None]
            band = np.exp(-((xx - x) / 70) ** 2)
            fr = fr + (steel * band)[..., None] * 1.4 * np.array([0.9, 0.95, 1.0], np.float32)
            fr = A.glow(fr, (x, 470), 60, 0.5 * math.sin(math.pi * q), (0.8, 0.88, 1.0), aspect=3)
        mouth = info['to_scr']((0.47, 0.42))
        fr = K.sparks(fr, t, 33, mouth, n=70, life=0.9, speed=(40, 260), dirc=(0, -1), spread=2.4, grav=-60, area=(90, 30), amt=0.55,
                      t0=V(4.53), burst=True, color=(0.75, 0.8, 0.9))
        fr = GDUST.draw(fr, t, 0.35, wind=(-60, -30))
        return fr, (2 if 0 <= t - V(4.53) < 0.07 else 0)


# A04 5.97–7.38 CU 侧脸（第四棒：微转头 + 三层视差：远月晕反向漂、中景尘、近景虚焦光斑快速横移）
class A04(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.01, 0.012, 0.022), (0.025, 0.03, 0.045))
        fr = A.glow(fr, (1560 - 90 * u, 380), 330, 0.55, (0.55, 0.62, 0.82))
        fr = A.glow(fr, (1560 - 90 * u, 380), 110, 0.35, (0.85, 0.9, 1.0))
        fr = fog(fr, t, 2, 300, 1000, 0.25, (0.18, 0.2, 0.26), speed=(-90, 0))
        turn = eio((t - V(6.48)) / 0.5)
        fr, info = A.place(fr, t, S('S004/profile'), (1270 + 70 * u, 1000), h=880 * (1 + 0.03 * u), flip=True, rim=COLD, rim_amt=4.0,
                           rim_dir=(0.8, -0.6), wind=14, rot=-2.5 + 3.5 * turn, grade=dict(expo=0.82, sat=0.5, gamma=1.12))
        for i, (x, y, r) in enumerate(((300, 260, 70), (520, 820, 50), (1750, 760, 90), (980, 200, 40), (160, 600, 60))):
            fr = A.glow(fr, ((x + 420 * u + 30 * i) % 2100 - 90, y - 25 * t), r, 0.10, (0.6, 0.7, 0.9))
        fr = ASH.draw(fr, t, 0.45, wind=(-200, 0))
        return fr, 0


# A05 7.38–7.85 ECU 双眼睁开 → 7.60 眼中寒光 → 末 4 帧压暗（为片名蓄力），7.73 刀光预光条
class A05(Shot):
    def __init__(self):
        self.path = S('A/eyes', 'dm', crop=(0.16, 0.0, 0.84, 1.0))
    def render(self, t):
        u = self.u(t)
        fr = night((0.005, 0.006, 0.01), (0.012, 0.014, 0.02))
        fr, info = A.place(fr, t, self.path, (960, 470), w=2500, anc=(0.5, 0.80), rim=COLD, rim_amt=0.8, wind=3,
                           grade=dict(expo=0.9, sat=0.6, gamma=1.18), wrap=0.1, breathe=0.0)
        yy = np.arange(H, dtype=np.float32)[:, None, None]       # 顶光：额头/眉骨亮，眼下暗
        fr = fr * (0.55 + 0.6 * np.exp(-((yy - 430) / 260) ** 2))
        ex = [info['to_scr']((0.405, 0.80)), info['to_scr']((0.595, 0.80))]
        g = math.exp(-max(0, t - V(7.60)) / 0.10) * (t >= V(7.60))
        for p in ex: fr = pupil_glint(fr, (p[0], p[1] - 8), 0.9 * g, 1.2)
        k = clamp01((t - (self.t1 - 4 / FPS)) / (4 / FPS))
        fr = fr * (1 - 0.75 * k)
        fr = streak(fr, 520, -100, 2100, (t - V(7.73)) / 0.10, amt=0.7)
        return fr, (3 if g > 0.4 else 0)


# A06 7.85–10.61 片名重击：刀光横斩，「鬼屠 · 第三季」志莽行书炸出，飞火 + 墨溅 + 闪白；8.31 鬼屠印；8.35 副标题
TITLE_HIT = V(7.85)
TITLE_Y = 470
class Title:
    """片名层（A06 屏幕位置 / A07 贴到刀面），保持用户 05:52 定的层级：主标题 + 副标题 + 印。"""
    def __init__(self):
        T.warm('鬼屠第三季', 'xing', 210)
        T.warm('主题曲《谁定正邪》', 'song', 50)
    def layer(self, t, out=0.0, sheen=None):
        k = t - TITLE_HIT
        L = T.Layer()
        xs = [520, 735, 1080, 1290, 1500]
        for i, (c, x) in enumerate(zip('鬼屠第三季', xs)):
            e = eout((k - i * 0.035) / 0.22)
            if e <= 0: continue
            sc = 1 + 0.35 * (1 - e)
            T.place(L, c, x, TITLE_Y, 210, (0.96, 0.94, 0.90), 'xing', alpha=e, scale=sc, blur=6 * (1 - e) + 10 * out)
        L2 = T.Layer()
        se = eout((k - 0.06) / 0.25)
        if se > 0:
            dot = np.zeros((40, 40), np.float32); cv2.circle(dot, (20, 20), 11, 1.0, -1, cv2.LINE_AA)
            L2.paste(dot, 908, TITLE_Y + 4, (0.80, 0.08, 0.06), se)
        sr = eout((k - 0.5) / 0.6)
        if sr > 0:
            sub = '主题曲《谁定正邪》'
            for i, c in enumerate(sub):                                   # 第四棒：38→50px + 两侧细金线
                T.place(L2, c, 960 - (len(sub) - 1) * 34 + i * 68, 648, 50, (0.90, 0.86, 0.78), 'song', alpha=sr, blur=4 * (1 - sr))
            ln = np.zeros((4, int(260 * sr)), np.float32) + 1.0
            if ln.shape[1] > 2:
                L2.paste(ln, 960 - 360 - ln.shape[1] / 2, 650, (1.0, 0.78, 0.40), 0.9 * sr)
                L2.paste(ln, 960 + 360 + ln.shape[1] / 2, 650, (1.0, 0.78, 0.40), 0.9 * sr)
        return L, L2

    def draw(self, fr, t, out=0.0, sheen=None, glow=0.0):
        if t < TITLE_HIT: return fr
        L, L2 = self.layer(t, out)
        if out > 0:
            for LL in (L, L2):
                nz = cv2.resize(A.ink_noise(77, 256), (W, H))
                LL.A *= np.clip((nz - out * 1.15) * 5 + 0.5, 0, 1)
        if sheen is not None and 0 <= sheen <= 1:
            xx = np.arange(W, dtype=np.float32)[None]
            band = np.exp(-((xx - (300 + 1350 * sheen)) / 45) ** 2)
            fr = fr + (L.A * band)[..., None] * 1.3 * np.array([1.0, 0.93, 0.8], np.float32)
        fr = T.compose(fr, L, shadow=0.8, shadow_sigma=26, glow=0.22 + glow, glow_sigma=18)
        fr = T.compose(fr, L2, shadow=0.6, glow=0.1)
        ks = t - V(8.31)
        if ks > 0:
            fr = T.stamp_seal(fr, '鬼屠', 1665, 395, 78, min(1, ks / 0.5), style='yin', seed=3, alpha=0.95 * (1 - out))
        return fr


TITLE = Title()


class A06(Shot):
    def __init__(self):
        self.bg = night((0.015, 0.012, 0.016), (0.10, 0.07, 0.07))
    def render(self, t):
        u = self.u(t); k = t - TITLE_HIT
        fr = self.bg.copy()
        fr = A.moon(fr, 1480, 330, 150, 0.95, col=(1.0, 0.42, 0.32), halo=0.6)          # 血月：被云腰斩
        fr = A.glow(fr, (1480, 330), 520, 0.22, (0.6, 0.12, 0.08))
        fr = A.clouds(fr, t, 21, speed=(-45, 0), y0=150, y1=620, amt=0.9, col=(0.10, 0.08, 0.09), light=(1480, 330), light_r=360,
                      light_col=(1.0, 0.45, 0.32), thr=0.46)
        for i, (by, amp, tone) in enumerate(((760, 230, (0.05, 0.035, 0.04)), (860, 170, (0.025, 0.02, 0.025)), (960, 120, (0.006, 0.005, 0.007)))):
            fr, _ = A.ridge(fr, by, amp, 60 + i, tone=tone, rim=(0.9, 0.35, 0.25), rim_amt=0.25 - 0.06 * i, dx=-t * (8 + 14 * i))
            fr = fog(fr, t, i, by - 70, by + 50, 0.3, (0.25, 0.17, 0.17), speed=(25 + 20 * i, 0))
        if k >= 0:
            fr = EMB.draw(fr, t, 1.6 * max(0.35, 1 - k / 2.5), wind=(60, -30), burst=(960, TITLE_Y, 900) if k < 0.6 else None, t0=TITLE_HIT if k < 0.6 else None)
        if 0 <= k < 0.14:                                                               # 闪白（全段唯一一处，HANDOFF §1；第三棒压短压轻，片名不被洗掉）
            fr = fr * (1 + 1.1 * (1 - k / 0.14)) + 0.22 * (1 - k / 0.14)
        fr = streak(fr, TITLE_Y + 10, -100, 2100, (t - (TITLE_HIT - 0.06)) / 0.12, ang=-3, amt=1.2)
        ca = 6 if 0 <= k < 0.15 else (2 if 0 <= k < 0.4 else 0)
        return fr, ca


# A08 13.35–15.21 ELS 月下城墙绵延；14.28 云内闪电亮，照出云上太上皇巨影（一拍即逝）
class _A08_OLD(Shot):
    """（第四棒：只借 _wall 给 A07b）第二棒重做：城墙不再是一条平涂垛口——透视纵深的长城（近大远小、垛口按深度缩放、砖缝纹理、月光顶边轮廓、
    三座敌楼带飞檐与灯火），远端爬上山脊蜿蜒入云；前景城墙走道雾；闪电时照出云中太上皇巨影。"""
    VP = (1820, 640)
    def __init__(self):
        self.bg = night((0.012, 0.014, 0.024), (0.08, 0.085, 0.10))
        self.wall, self.face_lit, self.edge, self.towers = self._wall()
    def _wall(self):
        q = 2; h, w = H // q, W // q
        vx, vy = self.VP[0] / q, self.VP[1] / q
        Aw, Ht, Hb = 2300 / q, 300 / q, 300 / q
        xx = np.arange(w, dtype=np.float32)
        z = np.clip(Aw / np.maximum(vx - xx, 1e-3), 1.0, 60)            # 每列深度
        top = vy - Ht / z; bot = vy + Hb / z
        yy = np.arange(h, dtype=np.float32)[:, None]
        m = ((yy >= top[None]) & (yy <= bot[None])).astype(np.float32) * (xx[None] < vx - 2)
        # 垛口：世界间距 0.18（深度单位），高 0.06*Ht
        s_ = z - 1
        mer = (np.mod(s_ * 5.5, 1.0) < 0.55)
        mt = top - (Ht * 0.10) / z
        m = np.maximum(m, ((yy >= mt[None]) & (yy < top[None]) & mer[None]).astype(np.float32) * (xx[None] < vx - 2))
        # 砖缝：横缝 22 行，竖缝随行错位
        v = np.clip((yy - top[None]) / np.maximum(bot - top, 1e-3)[None], 0, 1)
        row = np.floor(v * 22)
        jx = np.mod(s_[None] * 14 + 0.5 * np.mod(row, 2), 1.0)
        mort = ((np.mod(v * 22, 1.0) < 0.10) | (jx < 0.07)).astype(np.float32)
        fade = np.clip((z - 1) / 10, 0, 1)[None]                           # 远处砖纹平均掉
        nz = cv2.resize(A.tex(81, 512, 6, 6), (w, h))
        stain = 0.75 + 0.5 * nz * (1 - v * 0.5)
        face = (1 - 0.55 * mort * (1 - fade)) * stain * (1 - 0.6 * v) * m    # 月光自上而下衰减
        # 顶边月光轮廓（含垛口）
        e = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
        e = e * (yy < (top[None] + 6))
        # 敌楼
        towers = []
        for zt in (2.4, 5.2, 11.0):
            xt = vx - Aw / zt; sc = 1 / zt
            towers.append((xt * q, top[int(np.clip(xt, 0, w - 1))] * q, sc))
        m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 0.6), (W, H)); face = cv2.resize(face, (W, H)); e = cv2.resize(e, (W, H))
        Tm = np.zeros((H, W), np.float32); Te = np.zeros((H, W), np.float32); wins = []
        for (xt, yt, sc) in towers:
            S_ = 2300 * sc
            bw, bh = 0.30 * S_, 0.20 * S_
            x0, x1 = int(xt - bw / 2), int(xt + bw / 2)
            cv2.rectangle(Tm, (x0, int(yt - bh)), (x1, int(yt + 4)), 1.0, -1)
            for k, (rw, ry, rh) in enumerate(((0.44, bh, 0.05), (0.34, bh + 0.11 * S_, 0.045))):
                y0 = int(yt - ry); W2 = rw * S_ / 2
                pts = np.array([(xt - W2 - 0.03 * S_, y0 + 0.012 * S_), (xt - W2 * 0.82, y0 - rh * S_ * 0.35), (xt - W2 * 0.55, y0 - rh * S_),
                                (xt + W2 * 0.55, y0 - rh * S_), (xt + W2 * 0.82, y0 - rh * S_ * 0.35), (xt + W2 + 0.03 * S_, y0 + 0.012 * S_)], np.int32)
                cv2.fillPoly(Tm, [pts], 1.0, cv2.LINE_AA)
                cv2.polylines(Te, [pts[:4]], False, 1.0, max(1, int(3 * sc * 1.5)), cv2.LINE_AA)
                if k == 0:
                    cv2.rectangle(Tm, (int(xt - 0.11 * S_), int(y0 - 0.11 * S_)), (int(xt + 0.11 * S_), y0), 1.0, -1)
            for j in (-1, 0, 1):                                            # 箭窗（灯火位置）
                wins.append((xt + j * bw * 0.28, yt - bh * 0.55, sc))
        return (m, Tm), face, (e, Te), towers + wins
    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        env = 0.0
        for t0, a in ((V(14.28), 1.0), (V(14.28) + 0.12, 0.6), (V(14.28) + 0.3, 0.4)):
            if t >= t0: env = max(env, a * math.exp(-(t - t0) / 0.09))
        if env > 0.02 or (V(14.28) - 0.05 < t < V(14.28) + 0.9):            # 巨影：闪电时从云里显出
            lit = np.clip(env * 2, 0, 1)
            fr = A.glow(fr, (1000, 380), 560, 0.9 * lit, (0.75, 0.8, 0.95))
            # 第二棒：改用 tsh_raise 举手剪影（预告 V16 召印；S011/backlit 只留给 V07）
            fr, _ = A.place(fr, t, S('A/tsh_raise', 'dm'), (980, 980), h=980, look='sil', rim=(0.85, 0.9, 1.0), rim_amt=5.0, wind=5,
                            alpha=0.12 + 0.8 * lit, defocus=2.0)
        fr = A.moon(fr, 1500, 230, 62, 0.9, halo=0.5)
        fr = A.clouds(fr, t, 31, speed=(25, 0), y0=120, y1=600, amt=0.9, col=(0.13, 0.14, 0.18), light=(1000, 330), light_r=600,
                      light_col=(0.9, 0.92, 1.0), thr=0.44)
        if env > 0: fr = fr + env * 0.18 * np.array([0.8, 0.85, 1.0], np.float32)
        # 远山 + 山脊上蜿蜒的城墙（细线 + 垛点）
        fr, rl = A.ridge(fr, 700, 150, 71, tone=(0.035, 0.04, 0.055), rim=(0.5, 0.56, 0.72), rim_amt=0.3, dx=-u * 15, fogcol=(0.10, 0.11, 0.14), fog=0.6)
        Lw = np.zeros((H, W), np.float32)
        xs = np.arange(1100, W, 3); ys = rl[xs] - 3
        cv2.polylines(Lw, [np.stack([xs, ys], 1).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
        for x in range(1100, W, 9): cv2.line(Lw, (x, int(rl[x] - 4)), (x, int(rl[x] - 7)), 1.0, 2)
        fr = fr * (1 - Lw[..., None] * 0.9) + Lw[..., None] * np.array([0.30, 0.34, 0.45], np.float32) * (0.5 + env)
        fr = fog(fr, t, 1, 560, 760, 0.45, (0.20, 0.22, 0.28), speed=(-20, 0))
        # 近处城墙
        (m, Tm), face, (e, Te), towers = self.wall, self.face_lit, self.edge, self.towers
        k = 1 + 1.5 * env
        col = np.array([0.050, 0.056, 0.075], np.float32) * k
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        fr = fr * (1 - np.clip((yy - 700) / 120, 0, 1) * 0.85)                 # 墙下大地压暗
        fr = fr * (1 - m[..., None]) + (face[..., None] * col + 0.006) * m[..., None]
        fr = fr * (1 - Tm[..., None]) + Tm[..., None] * (0.008 + 0.03 * env)
        fr = fr + (e + Te)[..., None] * np.array([0.55, 0.62, 0.82], np.float32) * (0.45 + 1.4 * env)
        for (x, y, sc) in towers[3:]:                                       # 敌楼箭窗灯火（微颤）
            fl = 0.8 + 0.2 * math.sin(t * 17 + x)
            fr = A.glow(fr, (x, y), max(3, 26 * sc * 1.2), 1.2 * fl, (1.0, 0.58, 0.25))
        fr = A.glow(fr, towers[3][:2], 220, 0.10, (1.0, 0.55, 0.25))
        fr = fog(fr, t, 0, 820, 1000, 0.3, (0.16, 0.18, 0.23), speed=(-45, 0))
        fr = ASH.draw(fr, t, 0.5 * clamp01(1 - u * 1.5), wind=(220, -60))
        return fr, (3 if env > 0.6 else 0)


# A07a 10.61–11.98 雨夜城楼大仰角（第四棒新镜，替换原 A07 刀面片名）：前景旗杆 + 大旗擦过左侧（虚焦），城楼占上半——两重飞檐、
#      檐下红灯笼、垛口月光边；城门洞下主角极小的背影；推轨。片名是 A06 延续下来的屏幕层（11.52 扫光）。
def _tower_masks():
    m = np.zeros((H, W), np.float32); roof = np.zeros((H, W), np.float32); arch = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [np.array([(300, 560), (1620, 560), (2250, H), (-330, H)], np.int32)], 1.0, cv2.LINE_AA)
    for x in range(300, 1620, 66): cv2.rectangle(m, (x, 522), (x + 36, 562), 1.0, -1)
    cv2.rectangle(m, (560, 330), (1360, 560), 1.0, -1)
    cv2.rectangle(m, (690, 210), (1230, 262), 1.0, -1)
    for pts in ([(360, 362), (450, 318), (590, 262), (1330, 262), (1470, 318), (1560, 362), (1540, 376), (380, 376)],
                [(540, 236), (620, 196), (720, 150), (1200, 150), (1300, 196), (1380, 236), (1360, 248), (560, 248)]):
        cv2.fillPoly(roof, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
    cv2.ellipse(arch, (960, 880), (170, 200), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
    cv2.rectangle(arch, (790, 880), (1130, H), 1.0, -1)
    return m, roof, arch


class A07a(Shot):
    def __init__(self):
        self.bg = night((0.014, 0.017, 0.03), (0.07, 0.075, 0.095))
        self.m, self.roof, self.arch = _tower_masks()
        self.tex = cv2.resize(A.tex(131, 512, 6, 6), (W, H))
        r = np.random.default_rng(3)
        self.wins = [(x, 450) for x in range(620, 1340, 90)] + [(x, 236) for x in range(740, 1220, 96)]
        self.lanterns = [(450, 410), (1470, 410), (640, 280), (1280, 280)]
    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = A.glow(fr, (960, 120), 700, 0.22, (0.45, 0.52, 0.7))
        fr = A.glow(fr, (960, 200), 900, 0.18, (0.4, 0.46, 0.62), aspect=0.5)            # 城楼后天光：让飞檐轮廓从夜空里分出来
        fr = A.clouds(fr, t, 91, speed=(80, 10), y0=0, y1=420, amt=0.9, col=(0.10, 0.11, 0.14), dark=0.7, thr=0.42,
                      light=(960, 120), light_r=800, light_col=(0.6, 0.66, 0.85))
        fr = fog(fr, t, 0, 260, 600, 0.35, (0.16, 0.18, 0.23), speed=(50, 0))
        yy = np.arange(H, dtype=np.float32)[:, None]
        face = (0.75 + 0.5 * self.tex) * (1 - 0.55 * np.clip((yy - 560) / 520, 0, 1))          # 墙面：顶亮下暗 + 湿渍
        m, roof, arch = self.m, self.roof, self.arch
        fr = fr * (1 - m[..., None]) + (face * m)[..., None] * np.array([0.045, 0.05, 0.068], np.float32)
        fr = fr * (1 - roof[..., None]) + roof[..., None] * 0.006
        e = np.clip(np.maximum(m, roof) - cv2.erode(np.maximum(m, roof), np.ones((3, 3), np.uint8)), 0, 1) * (yy < 600)
        fr = fr + cv2.GaussianBlur(e, (0, 0), 1.0)[..., None] * np.array([0.55, 0.62, 0.82], np.float32) * 0.7
        fr = fr * (1 - arch[..., None]) + arch[..., None] * np.array([0.03, 0.025, 0.02], np.float32)
        fr = A.glow(fr, (960, 940), 260, 0.25, (1.0, 0.6, 0.3), aspect=0.7)                       # 门洞深处火光
        for i, (x, y) in enumerate(self.wins):
            fl = 0.75 + 0.25 * math.sin(t * 13 + i * 1.7)
            fr = A.glow(fr, (x, y), 9, 0.9 * fl, (1.0, 0.6, 0.28), aspect=0.8)
        for i, (x, y) in enumerate(self.lanterns):                                                   # 檐下红灯笼（随风摆）
            sw = 10 * math.sin(t * 2.6 + i)
            cv2.line(fr, (int(x), int(y - 46)), (int(x + sw * 0.5), int(y - 26)), (0.02, 0.02, 0.02), 2, cv2.LINE_AA)
            fr = A.glow(fr, (x + sw, y), 26, 1.4, (1.0, 0.22, 0.10), aspect=1.25)
            fr = A.glow(fr, (x + sw, y), 90, 0.25, (1.0, 0.3, 0.12))
        for k, px in enumerate((360, 1560)):                                                        # 城头旗：杆 + 风中旗面
            top = 330; cv2.line(fr, (px, 540), (px, top), (0.01, 0.01, 0.012), 5, cv2.LINE_AA)
            d = 1 if k == 0 else -1
            pts = [(px + d * 180 * s, top + 8 + 14 * math.sin(t * 7 + s * 5 + k)) for s in np.linspace(0, 1, 9)]
            pts += [(px + d * 180 * s, top + 100 + 14 * math.sin(t * 7 + s * 5 + k + 0.6)) for s in np.linspace(1, 0, 9)]
            Lf = np.zeros((H, W), np.float32); cv2.fillPoly(Lf, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
            fr = fr * (1 - Lf[..., None]) + Lf[..., None] * np.array([0.16, 0.02, 0.02], np.float32)
        fr, _ = A.place(fr, t, S('A/back_walk'), (960, 1010), h=190, look='sil', rim=(1.0, 0.65, 0.35), rim_amt=3.0, wind=6)
        fr = RAIN.draw(fr, t, 1.0, angle=10, light=(960, 120), layers=(0, 1))
        fr = fog(fr, t, 1, 820, 1060, 0.4, (0.18, 0.2, 0.25), speed=(-60, 0))
        # 前景：虚焦旗杆 + 大旗擦过左侧（推轨视差）
        Lp = np.zeros((H // 2, W // 2), np.float32)
        px = (260 - 420 * u) / 2
        cv2.rectangle(Lp, (int(px - 16), 0), (int(px + 16), H // 2), 1.0, -1)
        pts = [(px + 20 + 330 * s, 40 + 60 * s + 26 * math.sin(t * 5 + s * 4)) for s in np.linspace(0, 1, 10)]
        pts += [(px + 20 + 330 * s, 300 + 30 * s + 26 * math.sin(t * 5 + s * 4 + 0.8)) for s in np.linspace(1, 0, 10)]
        cv2.fillPoly(Lp, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        Lp = cv2.resize(cv2.GaussianBlur(Lp, (0, 0), 7), (W, H))[..., None]
        fr = fr * (1 - Lp * 0.96) + Lp * np.array([0.05, 0.008, 0.008], np.float32)
        fr = RAIN.draw(fr, t, 0.8, angle=10, layers=(2,))
        return fr, 0


# A07b 11.98–13.35 ELS 透视城墙（第四棒新镜，原 A08 城墙移来，去月）：长城近大远小蜿蜒入雨雾，敌楼灯火；斜雨 + 三层雾；13.35 片名化灰
class A07b(Shot):
    VP = (1820, 640)
    def __init__(self):
        self.bg = night((0.012, 0.014, 0.024), (0.075, 0.08, 0.095))
        self.wall, self.face_lit, self.edge, self.towers = _A08_OLD._wall(self)
    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = A.glow(fr, (1300, 260), 700, 0.25, (0.5, 0.56, 0.72))
        fr = A.clouds(fr, t, 31, speed=(60, 0), y0=60, y1=600, amt=0.95, col=(0.12, 0.13, 0.17), light=(1300, 260), light_r=700,
                      light_col=(0.75, 0.8, 0.95), thr=0.42)
        fr, rl = A.ridge(fr, 700, 150, 71, tone=(0.035, 0.04, 0.055), rim=(0.5, 0.56, 0.72), rim_amt=0.25, dx=-u * 40, fogcol=(0.10, 0.11, 0.14), fog=0.6)
        fr = fog(fr, t, 1, 520, 760, 0.55, (0.20, 0.22, 0.28), speed=(-40, 0))
        (m, Tm), face, (e, Te), towers = self.wall, self.face_lit, self.edge, self.towers
        col = np.array([0.050, 0.056, 0.075], np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        fr = fr * (1 - np.clip((yy - 700) / 120, 0, 1) * 0.85)
        fr = fr * (1 - m[..., None]) + (face[..., None] * col + 0.006) * m[..., None]
        fr = fr * (1 - Tm[..., None]) + Tm[..., None] * 0.01
        fr = fr + (e + Te)[..., None] * np.array([0.55, 0.62, 0.82], np.float32) * 0.5
        for (x, y, sc) in towers[3:]:
            fl = 0.8 + 0.2 * math.sin(t * 17 + x)
            fr = A.glow(fr, (x, y), max(3, 26 * sc * 1.2), 1.2 * fl, (1.0, 0.58, 0.25))
            fr = A.glow(fr, (x, y + 40 * sc), max(8, 120 * sc), 0.12 * fl, (1.0, 0.5, 0.2))
        fr = fog(fr, t, 2, 640, 900, 0.4, (0.17, 0.19, 0.24), speed=(70, 0))
        fr = RAIN.draw(fr, t, 1.0, angle=14, light=(1300, 260), layers=(0, 1, 2))
        fr = fog(fr, t, 0, 820, 1060, 0.45, (0.16, 0.18, 0.23), speed=(-110, 0))
        return fr, 0


# A08 13.35–15.21 MS 背影推进 + 城门合上（第四棒重做）：门洞深处冷光，两扇包铁城门向中合拢；鬼屠背影走向门前、越走越远；
#      14.28 闪电从门缝外劈亮；15.05 合死只剩一线门缝 → a_seam 门缝里开出 V01
class A08(Shot):
    def __init__(self):
        self.tex = cv2.resize(A.tex(141, 512, 5, 6), (W, H)); self.tex2 = cv2.resize(A.tex(142, 512, 30, 3), (W, H))
    def close(self, t): return eio((t - self.t0) / (V(15.05) - self.t0))
    def render(self, t):
        u = self.u(t)
        env = 0.0
        for t0, a in ((V(14.28), 1.0), (V(14.28) + 0.12, 0.6), (V(14.28) + 0.3, 0.4)):
            if t >= t0: env = max(env, a * math.exp(-(t - t0) / 0.09))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # 门洞外：冷雾光 + 远处街道
        fr = A.vgrad((0.10, 0.12, 0.16), (0.04, 0.045, 0.06)) * (1 + 3 * env)
        fr = A.glow(fr, (960, 500), 420, 0.6 + 2.0 * env, (0.6, 0.7, 0.95))
        fr = RAIN.draw(fr, t, 0.8, angle=6, light=(960, 400), layers=(0,))
        # 城门两扇（门洞 x 520–1400）
        c = self.close(t); gx0, gx1 = 520, 1400; half = (gx1 - gx0) / 2
        wL = half * (0.18 + 0.82 * c)
        door = ((xx < gx0 + wL) & (xx >= gx0)) | ((xx > gx1 - wL) & (xx <= gx1))
        door = door.astype(np.float32)
        plank = 0.75 + 0.35 * self.tex2 + 0.15 * (np.mod(xx, 70) < 3)
        studs = ((np.mod(xx - gx0, 88) - 44) ** 2 + (np.mod(yy, 96) - 48) ** 2 < 49).astype(np.float32)
        dcol = (plank * (0.5 + 0.5 * self.tex))[..., None] * np.array([0.05, 0.035, 0.025], np.float32) * (1 + 2 * env)
        dcol = dcol + studs[..., None] * np.array([0.10, 0.09, 0.08], np.float32) * (1 + 3 * env)
        fr = fr * (1 - door[..., None]) + dcol * door[..., None]
        seamL = np.exp(-((xx - (gx0 + wL)) / 4) ** 2); seamR = np.exp(-((xx - (gx1 - wL)) / 4) ** 2)
        fr = fr + ((seamL + seamR) * (yy > 200))[..., None] * np.array([0.6, 0.7, 1.0], np.float32) * (0.8 + 3 * env)
        # 城墙 + 拱门（门洞外沿）
        wall = np.ones((H, W), np.float32)
        hole = ((((xx - 960) / 440) ** 2 + ((yy - 470) / 300) ** 2 < 1) & (yy < 470)) | ((xx > gx0) & (xx < gx1) & (yy >= 470))
        wall[hole] = 0; wall = cv2.GaussianBlur(wall, (0, 0), 1.2)
        brick = (0.7 + 0.5 * self.tex) * (1 - 0.4 * ((np.mod(yy, 60) < 3) | (np.mod(xx + 60 * (np.mod(yy // 60, 2)), 120) < 3)))
        wcol = brick[..., None] * np.array([0.035, 0.038, 0.05], np.float32) * (1 + 2.5 * env)
        fr = fr * (1 - wall[..., None]) + wcol * wall[..., None]
        ed = np.clip(cv2.dilate(1 - wall, np.ones((5, 5), np.uint8)) - (1 - wall), 0, 1)
        fr = fr + ed[..., None] * np.array([0.5, 0.6, 0.85], np.float32) * (0.5 + 2 * env)
        # 湿地面倒影：门缝光的竖条
        g = np.clip((yy - 900) / 30, 0, 1)
        fr = fr * (1 - g[..., None] * 0.5) + (g * np.exp(-((xx - 960) / (40 + 300 * (1 - c))) ** 2))[..., None] * np.array([0.25, 0.3, 0.45], np.float32) * (1 + 2 * env)
        # 主角背影：走向门前（越走越小）
        hh = 760 - 170 * eio(u); by = 1040 - 70 * eio(u)
        fr, _ = A.place(fr, t, S('A/back_walk'), (960 + 25 * math.sin(t * 3.0), by), h=hh, rim=COLD, rim_amt=3.0 + 6 * env,
                        rim_dir=(0, -1), grade=dict(expo=0.6, sat=0.5, gamma=1.2), wind=9)
        fr = RAIN.draw(fr, t, 0.9, angle=6, light=(960, 400), layers=(1, 2))
        fr = fog(fr, t, 0, 860, 1060, 0.35, (0.16, 0.18, 0.23), speed=(-45, 0))
        return fr, (3 if env > 0.6 else 0)


_SILM = {}
def _sil_mask(t):
    k = int(t * 10)
    if k not in _SILM:
        _, info = A.place(np.zeros((H, W, 3), np.float32), t, S('S011/backlit'), (1200, 1040), h=1050, wind=5)
        _SILM.clear(); _SILM[k] = cv2.GaussianBlur(info['A'], (0, 0), 4)[..., None]
    return _SILM[k]


# =====================================================================================
# 主歌一
# V01 15.21–17.00 ECU 雨打刃口微距（第四棒：程序横刀 → blade_rain 精灵刃身，有钢纹/水珠；斜透视指向画面深处），乌云压低逼近刃口；
#     雨点落刃溅起；16.86「刀」一线刀光沿刃走
class V01(Shot):
    def __init__(self):
        self.path = S('A/blade_rain', 'dm', crop=(0.22, 0.69, 1.0, 1.0))
        r = np.random.default_rng(15); self.drops = [(r.uniform(0.15, 0.95), r.uniform(0, 1.8)) for _ in range(26)]
    def render(self, t):
        u = self.u(t)
        fr = night((0.006, 0.007, 0.012), (0.04, 0.045, 0.06))
        fr = A.glow(fr, (1500, 860), 520, 0.25, (0.45, 0.52, 0.7))
        cy = 330 + 230 * eio(u)
        fr = A.clouds(fr, t, 41, speed=(30, 50), y0=100, y1=int(cy + 160), amt=1.0, col=(0.05, 0.055, 0.07), dark=0.85, thr=0.38,
                      light=(1500, 860), light_r=700, light_col=(0.5, 0.55, 0.7))
        fr = RAIN.draw(fr, t, 0.8, angle=8, light=(1500, 300), layers=(0, 1))
        fr, info = A.place(fr, t, self.path, (1080 - 90 * u, 700), w=2500 * (1 + 0.07 * u), anc=(0.5, 0.5), rot=-8, rim=COLD, rim_amt=1.6,
                           rim_dir=(0.3, -1.0), grade=dict(expo=0.95, sat=0.35, gamma=1.08), wind=0, breathe=0.0)
        P0 = info['to_scr']((0.10, 0.80)); P1 = info['to_scr']((0.98, 0.80))
        for i, (s, ph) in enumerate(self.drops):                         # 雨点打刃：小冠状溅射
            k = (t * 1.3 + ph) % 1.8
            if k < 0.18:
                x = P0[0] + (P1[0] - P0[0]) * s; y = P0[1] + (P1[1] - P0[1]) * s - 40
                fr = A.glow(fr, (x, y), 5 + 30 * k, 0.9 * (1 - k / 0.18), (0.8, 0.88, 1.0), aspect=0.45)
        fr = ASH.draw(fr, t, 0.4, wind=(-30, 90))
        fr = RAIN.draw(fr, t, 0.7, angle=8, layers=(2,))
        a = t - CT(0, 5)
        if a >= 0:
            fr = streak(fr, P0[1] - 40, P0[0], P1[0], a / 0.12, amt=1.3, ang=math.degrees(math.atan2(P1[1] - P0[1], P1[0] - P0[0])))
        return fr, (4 if 0 <= a < 0.1 else 0)


# V02 17.00–17.95 CU 低机位：靴子踏进水洼，倒影碎开，雨（「之上」收尾，独行）
class V02(Shot):
    def __init__(self):
        self.spl = fx2.Splashes(4) if hasattr(fx2, 'Splashes') else None
    def render(self, t):
        u = self.u(t)
        fr = night((0.02, 0.022, 0.03), (0.035, 0.04, 0.05), y1=700)
        fr = A.glow(fr, (1300, 300), 400, 0.35, (0.5, 0.58, 0.78))
        fr = RAIN.draw(fr, t, 0.9, angle=8, light=(1300, 300), layers=(0, 1))
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        ground = np.clip((yy - 760) / 30, 0, 1)
        fr = fr * (1 - ground * 0.85)
        dy = -20 * math.sin(math.pi * clamp01((t - self.t0) / 0.5))
        fr, info = A.place(fr, t, S('A/boots'), (960 + 140 * u, 830 + dy), h=900, anc=(0.5, 0.92), rim=COLD, rim_amt=3.0,
                           grade=dict(expo=0.7, sat=0.6), wind=7)
        refl = cv2.flip(fr[460:760], 0)                                                 # 水洼倒影（波纹扰动）
        ph = np.sin(np.arange(300, dtype=np.float32)[:, None] / 6 + t * 12) * 6
        mx = (np.arange(W, dtype=np.float32)[None] + ph).astype(np.float32); my = np.repeat(np.arange(300, dtype=np.float32)[:, None], W, 1)
        refl = cv2.remap(refl, mx, my, cv2.INTER_LINEAR)
        m = np.clip((yy[760:1060] - 760) / 40, 0, 1) * 0.45
        fr[760:1060] = fr[760:1060] * (1 - m) + refl[:300] * m * np.array([0.8, 0.88, 1.0], np.float32)
        ts = V(17.46)
        if t >= ts:
            fr = fx2.ripples(fr, t, [(960 + 140 * u, 900)], [ts], speed=520, amt=1.0, life=0.8)
            fr = K.sparks(fr, t, 17, (960 + 140 * u, 900), n=40, life=0.5, speed=(200, 600), dirc=(0, -1), spread=1.2, grav=1400,
                          area=(120, 10), amt=0.5, t0=ts, burst=True, color=(0.7, 0.8, 1.0))
        fr = RAIN.draw(fr, t, 0.7, angle=8, layers=(2,))
        return fr, 0


# V03 17.95–19.78 ELS 俯瞰五百重甲方阵，火把金点，雾中推进；巨字「五百」半透明压在方阵上，「重甲」随后
class V03(Shot):
    def __init__(self):
        import crowd
        self.army = crowd.Army(seed=3, rows=20, cols=25, torches=50)
        T.warm('五百', 'kai', 330); T.warm('重甲', 'kai', 120)
    def render(self, t):
        u = self.u(t)
        fr = night((0.02, 0.022, 0.03), (0.05, 0.055, 0.065))
        # 第三棒：原俯角 40° 灰白小人方块（评审口径「白模几何体」）→ 俯 30°、前排压到画框下沿铺满画面；
        # 甲胄压暗 + 冷月轮廓光 1.6 + 红缨；火把暖光池 + 行间低雾给纵深
        fr, _ = self.army.draw(fr, t, cam=dict(x=-3 + 5 * u, y=16, z=-8 + 2.5 * u, pitch=30, f=1400), advance=1.2 * u, march=1.0,
                               rim_color=COLDV, rim=1.6, fog_color=(0.04, 0.045, 0.06), fog_dist=70, torch=1.6, origin=(0, 0, 3))
        for (gx, gy, gr) in ((380, 760, 260), (1250, 560, 220), (1650, 900, 300)):
            fr = A.glow(fr, (gx + 40 * u, gy), gr, 0.10, (1.0, 0.55, 0.22))
        fr = fog(fr, t, 1, 300, 960, 0.28, (0.12, 0.13, 0.17), speed=(60, 10))
        fr = A.glow(fr, (1500, 120), 600, 0.15, (0.5, 0.55, 0.7))
        # 巨字
        L = T.Layer()
        for i, (c, x, y, s, k) in enumerate((('五', 380, 360, 330, 0), ('百', 690, 360, 330, 1))):   # 第三棒：右移守左侧安全边
            p = cprog(1, [k], t, 0.3)[0]
            if p > 0: L.paste(T.ink_bleed_glyph(c, 'kai', s, p, seed=i, bleed=0.9), x, y + 20 * (1 - p), (0.92, 0.90, 0.86))
        for i, (c, k) in enumerate((('重', 2), ('甲', 3))):
            p = cprog(1, [k], t, 0.2)[0]
            if p > 0: L.paste(T.brush_glyph(c, 'kai', 120, p, seed=i), 960, 330 + i * 130, (0.86, 0.12, 0.08) if c == '甲' else (0.92, 0.9, 0.86))
        fr = T.compose(fr, L, shadow=0.5, glow=0.15, opacity=0.82)
        return fr, 0


# V04 19.78–20.69 LS 方阵前排（虚焦枪林）前，鬼屠逆向独行入画（前景枪尖横移遮挡）
class V04(Shot):
    def __init__(self):
        import crowd
        self.army = crowd.Army(seed=5, rows=10, cols=30, torches=30)
    def render(self, t):
        u = self.u(t)
        fr = night((0.015, 0.017, 0.025), (0.06, 0.065, 0.08), y1=900)
        fr, _ = self.army.draw(fr, t, cam=dict(x=1.5 * u, y=1.9, z=-4, pitch=2, f=1500), march=0.0, rim_color=COLDV, dof=1.6, fog_dist=60)
        fr = K.blur_ds(fr, 3) if hasattr(K, 'blur_ds') and fr.ndim == 2 else cv2.GaussianBlur(fr, (0, 0), 2.5)
        fr = fog(fr, t, 2, 600, 960, 0.5, (0.28, 0.3, 0.36), speed=(-50, 0))
        fr = A.glow(fr, (700, 500), 300, 0.18, (0.55, 0.62, 0.8))
        fr, info = A.place(fr, t, S('S010/walk'), (1180 + 160 * u, 985), h=700, rim=COLD, rim_amt=3.0, wind=8,
                           grade=dict(expo=0.85, sat=0.55))
        fr = RAIN.draw(fr, t, 0.6, angle=10, layers=(1, 2))
        return fr, 0


# V05 20.69–21.20 ECU 手 + 刀特写（第四棒：脸完全出画，hand_draw_cu 精灵）：右手握柄、左拇指推开护手，刃口一线寒光，急推
class V05(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.01, 0.012, 0.02), (0.03, 0.035, 0.05))
        fr = A.glow(fr, (1300, 420), 420, 0.3, (0.5, 0.58, 0.8))
        fr, info = A.place(fr, t, S('A/hand_draw_cu'), (1040 - 40 * u, 560), w=2300 * (1 + 0.10 * u), anc=(0.5, 0.5), rim=COLD, rim_amt=2.2,
                           wind=0, grade=dict(expo=0.85, sat=0.5, gamma=1.1), breathe=0.0)
        g = math.exp(-max(0, t - V(20.82)) / 0.14) * (t >= V(20.82))
        p = info['to_scr']((0.49, 0.63))
        fr = pupil_glint(fr, p, 1.0 * g, 1.6)
        fr = A.glow(fr, p, 140, 0.5 * g, (0.75, 0.85, 1.0), aspect=0.35)
        fr = RAIN.draw(fr, t, 0.5, angle=10, layers=(2,))
        return fr, (3 if g > 0.5 else 0)


# V06 21.20–22.52 LS 横斩：21.24 一道横向寒光贯穿画面，前排盔甲裂开红线；「寒光」被斩开错位
class V06(Shot):
    def __init__(self):
        import crowd
        self.army = crowd.Army(seed=6, rows=8, cols=30, torches=20)
        T.warm('寒光', 'kai', 230)
    def render(self, t):
        u = self.u(t); ts = CT(1, 10); ts2 = V(21.86)                     # 第四棒：二次刀光斩字（寒光先完整可读 ≥0.55s）
        fr = night((0.015, 0.017, 0.025), (0.06, 0.065, 0.08), y1=900)
        fr, _ = self.army.draw(fr, t, cam=dict(x=-1.0 * u, y=1.8, z=-2.5, pitch=1, f=1500), march=0.0, rim_color=COLDV, dof=1.4, fog_dist=50,
                               level=1.0)
        fr = cv2.GaussianBlur(fr, (0, 0), 2.0)
        a = t - ts
        if a >= 0:                                                                     # 前排被斩：一道红热裂线
            L = np.zeros((H, W), np.float32)
            cv2.line(L, (0, 610), (W, 590), 1.0, 3, cv2.LINE_AA)
            fr = fr + (cv2.GaussianBlur(L, (0, 0), 2) * 1.5 + cv2.GaussianBlur(L, (0, 0), 14))[..., None] * RED * 1.5 * math.exp(-a / 0.8)
        fr = fog(fr, t, 0, 640, 960, 0.5, (0.28, 0.3, 0.36), speed=(80, 0))
        fr, info = A.place(fr, t, S('S010/slash_h'), (900, 1000), h=720, rim=COLD, rim_amt=3.0, wind=10, grade=dict(expo=0.9, sat=0.55),
                           mblur=(30 * math.exp(-max(0, a) / 0.08), 0) if a >= 0 else (0, 0))
        if a >= 0:
            fr = streak(fr, 600, -100, 2100, a / 0.08, w=5, amt=1.6, ang=-0.6)
        if 0 <= a < 0.07: fr = fr * 1.55 + 0.06 * COLDV          # 第三棒：冲击改冷色曝光脉冲（不再白闪）
        # 「寒光」斩开：字在屏幕下方，被同一刀切开两半错位
        p = cprog(1, [9], t, 0.25)[0]
        if p > 0:
            L = T.Layer()
            L.paste(T.brush_glyph('寒', 'kai', 230, p, seed=1), 800, 760, (0.94, 0.92, 0.88))
            L.paste(T.brush_glyph('光', 'kai', 230, cprog(1, [10], t, 0.12)[0], seed=2), 1080, 760, (0.86, 0.12, 0.08))
            a2 = t - ts2
            if a2 < 0:
                fr = T.compose(fr, L, shadow=0.7, glow=0.2)
            else:
                fr = T.slash_layer(fr, L, clamp01(a2 / 0.66), (940, 760), -4, slide=0.25, sz=230, length=900, sparks=50,
                                   style_kw=dict(shadow=0.7, glow=0.3))
                fr = streak(fr, 760, 300, 1700, a2 / 0.08, w=4, amt=1.2, ang=-4)
        return fr, (5 if 0 <= a < 0.1 else 0)


# V07 22.52–23.45 LS 仰拍宫阶（第四棒：去月 → 宫灯逆光）：殿檐压顶、两列红宫灯摇晃，太上皇立于丹陛顶端的背光剪影；
#     丹陛顶边 y=600 与 V06 寒光横线匹配剪辑；「谁在高处」由上压下
class V07(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.03, 0.015, 0.012), (0.10, 0.06, 0.04), y1=700)
        fr = A.glow(fr, (960, 420), 520, 0.55, (1.0, 0.55, 0.25))                     # 殿门内烛火大逆光（非月）
        fr = A.glow(fr, (960, 440), 180, 0.6, (1.0, 0.75, 0.45), aspect=0.8)
        fr = fx2.god_rays(fr, (960, 430), strength=0.3, length=1.0, color=(1.0, 0.7, 0.4), t=t, src_r=160)
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [np.array([(-50, 0), (1970, 0), (1970, 120), (1700, 190), (960, 150), (220, 190), (-50, 120)], np.int32)], 1.0, cv2.LINE_AA)   # 殿檐
        cv2.fillPoly(m, [np.array([(0, 600), (W, 600), (W, H), (0, H)], np.int32)], 1.0)
        for k in range(1, 6):                                                          # 丹陛台阶
            y = 600 + k * 60; cv2.line(m, (0, y), (W, y), 0.6, 2)
        fr = fr * (1 - m[..., None]) + m[..., None] * 0.008
        e = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
        fr = fr + e[..., None] * np.array([1.0, 0.7, 0.45], np.float32) * 0.7
        for i, x in enumerate((180, 420, 1500, 1740)):                                 # 宫灯
            y = 270 + (40 if i in (1, 2) else 0); sw = 12 * math.sin(t * 2.2 + i)
            cv2.line(fr, (x, 160), (int(x + sw * 0.6), y - 50), (0.01, 0.01, 0.01), 2, cv2.LINE_AA)
            fr = A.glow(fr, (x + sw, y), 34, 1.6, (1.0, 0.22, 0.08), aspect=1.3)
            fr = A.glow(fr, (x + sw, y), 120, 0.25, (1.0, 0.3, 0.1))
        fr, info = A.place(fr, t, S('S011/backlit'), (960, 602), h=400 * (1 + 0.04 * u), look='sil', rim=GOLD, rim_amt=5.0, wind=7)
        fr = GDUST.draw(fr, t, 0.5, wind=(30, -20))
        return fr, 0


# V08 23.45–25.29 CU 太上皇的手压下玉玺；24.48 落印 → 宣纸上朱印「罪状」
class V08(Shot):
    def __init__(self):
        self.path = S('A/tsh_hand', 'dm')
        self.paper = K.paper_light(21) if hasattr(K, 'paper_light') else None
    def render(self, t):
        u = self.u(t); th = CT(2, 7)
        fr = night((0.02, 0.016, 0.014), (0.05, 0.04, 0.035))
        fr = A.glow(fr, (1500, 300), 500, 0.35, (1.0, 0.72, 0.4))                         # 烛光暖侧光
        down = eio(clamp01((t - self.t0) / (th - self.t0)))
        fr, info = A.place(fr, t, self.path, (980, 470 + 120 * down), w=2200, anc=(0.5, 0.5), rim=GOLD, rim_amt=2.2, rim_dir=(0.8, -0.5),
                           grade=dict(expo=0.75, sat=0.5, tint=(1.04, 0.98, 0.9)), wind=3)
        a = t - th
        if a >= 0:
            fr = fr * (1 + 0.5 * math.exp(-a / 0.08))
            fr = T.stamp_seal(fr, '罪状', 560, 700, 250, clamp01(0.3 + a / 0.8), style='yin', seed=5)
        else:
            fr = T.stamp_seal(fr, '罪状', 560, 700, 250, clamp01(0.3 * (1 - (th - t) / 0.3)), style='yin', seed=5) if th - t < 0.3 else fr
        fr = GDUST.draw(fr, t, 0.35, wind=(20, -30))
        return fr, (4 if 0 <= a < 0.08 else 0)


PAPER_BODY = np.array([(0, -1.0), (-0.07, -0.97), (-0.09, -0.86), (-0.06, -0.80), (-0.22, -0.74), (-0.30, -0.40), (-0.24, -0.38),
                       (-0.20, -0.62), (-0.18, -0.30), (-0.30, 0), (0.30, 0), (0.18, -0.30), (0.20, -0.62), (0.24, -0.38), (0.30, -0.40),
                       (0.22, -0.74), (0.06, -0.80), (0.09, -0.86), (0.07, -0.97)], np.float32)


def paper_crowd(fr, t, cx, cy, wpx, rows=5, per=11, hpx=70, seed=0, light=(0, -1)):
    """第三棒：秤盘里的「千人」= 层层叠叠的剪纸人（宣纸纹理 + 顶光金边 + 每人轻微晃动），后排更小更暗、堆成丘，
    替换原来的白色小方块（评审口径：程序道具不能是白模几何体）。cx,cy = 盘心（脚底线），wpx = 盘宽。"""
    r = np.random.default_rng(seed)
    Lp = np.zeros((H, W), np.float32); Sh = np.zeros((H, W), np.float32)
    for row in range(rows - 1, -1, -1):
        k = row / max(1, rows - 1)
        n = per - row; sc = 1 - 0.12 * row
        span = wpx * (0.92 - 0.12 * row)
        for i in range(n):
            x = cx - span / 2 + span * (i + 0.5 + r.uniform(-0.3, 0.3)) / n
            y = cy - row * hpx * 0.42 - (1 - ((x - cx) / (wpx / 2)) ** 2) * hpx * 0.25 * row
            ang = r.uniform(-8, 8) + 4 * math.sin(t * 2.2 + i + row)
            th = math.radians(ang); Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
            P = (PAPER_BODY * hpx * sc * r.uniform(0.85, 1.1) @ Rm.T + [x, y]).astype(np.int32)
            tmp = np.zeros_like(Lp); cv2.fillPoly(tmp, [P], 1.0, cv2.LINE_AA)
            tone = 1.0 - 0.45 * k
            Sh = np.maximum(Sh * (1 - tmp), tmp * (1 - tone))          # 近排盖住远排
            Lp = np.maximum(Lp, tmp)
    Lp = cv2.GaussianBlur(Lp, (0, 0), 0.7)
    paper = (0.72 + 0.28 * cv2.resize(A.tex(seed + 102, 256, 10, 4), (W, H)))[..., None] * np.array([0.90, 0.74, 0.50], np.float32)
    shade = 1 - Sh[..., None]
    top = np.clip(Lp - cv2.warpAffine(Lp, np.float32([[1, 0, -light[0] * 3], [0, 1, -light[1] * 3]]), (W, H)), 0, 1)[..., None]
    edge = np.clip(Lp - cv2.erode(Lp, np.ones((3, 3), np.uint8)), 0, 1)[..., None]
    col = paper * 0.80 * shade + top * GOLDV * 0.9 - edge * 0.10
    return fr * (1 - Lp[..., None]) + np.clip(col, 0, 2) * Lp[..., None]


# V09 25.29–27.10 MS 悬空巨天平：左盘千人、右盘一枚金印仍压下；环绕；字沿横梁排列，「人命」朱红
class V09(Shot):
    def __init__(self):
        self.bal = PR.Balance(seed=4, people=90)
    def render(self, t):
        u = self.u(t)
        fr = night((0.012, 0.012, 0.02), (0.06, 0.055, 0.06))
        fr = A.glow(fr, (960, 260), 650, 0.3, (0.6, 0.55, 0.45))
        fr = fx2.god_rays(fr, (960, 160), strength=0.5, length=1.0, color=(0.9, 0.8, 0.6), t=t, src_r=160)
        tilt = -9 + 2 * math.sin(t * 1.3)
        fr, info = self.bal.draw(fr, t, 960, 262, 1120, tilt=tilt, left_people=0.0, right='seal', rim_color=GOLDV, drip=0.5)
        pl = info.get('pan_L')
        if pl is not None: fr = paper_crowd(fr, t, pl[0], pl[1] - 6, 290, hpx=62, seed=9)
        fr = fog(fr, t, 1, 700, 960, 0.4, (0.25, 0.24, 0.26), speed=(-30, 0))
        # 沿横梁排字（屏幕层随梁倾斜）
        txt = '谁把人命装进天平里'; ks = list(range(9))
        L = T.Layer(); ang = math.radians(tilt)
        for i, (c, p) in enumerate(zip(txt, cprog(3, ks, t, 0.2))):
            if p <= 0: continue
            d = (i - 4.5) * 120 + (60 if i >= 5 else -0)                       # 第三棒：字挂在梁下（让开支柱），守安全区
            x = 960 + d * math.cos(ang); y = 262 + 100 - d * math.sin(ang) + 30 * (1 - p)
            col = (0.86, 0.12, 0.08) if c in '人命' else (0.95, 0.92, 0.86)
            T.place(L, c, x, y, 88, col, 'kai', alpha=eout(p), rot=-tilt, blur=4 * (1 - p))
        fr = T.compose(fr, L, shadow=0.7, glow=0.25)
        return fr, 0


# V10 27.10–28.93 ECU 左秤盘边缘：小人滑落，红绳垂下滴朱墨；28.48 墨滴落地 →「称量」晕开
class V10(Shot):
    """ECU 左秤盘边缘（金色盘沿弧线占画面上部）：纸人小剪影沿盘沿滑落，盘下红绳垂直滴朱墨；28.48 落入墨池 →「称量」晕开"""
    def __init__(self):
        T.warm('称量', 'kai', 190)
        r = np.random.default_rng(10)
        self.ppl = [(r.uniform(300, 1500), r.uniform(0.7, 1.2), r.uniform(-20, 20)) for _ in range(26)]
    def render(self, t):
        u = self.u(t); td = CT(3, 10)
        fr = night((0.008, 0.008, 0.012), (0.025, 0.022, 0.024))
        fr = A.glow(fr, (1350, 260), 520, 0.22, (0.9, 0.75, 0.5))
        fr = fog(fr, t, 2, 500, 860, 0.35, (0.16, 0.12, 0.10), speed=(-25, 0))
        # 盘：青铜盘底（噪声铜锈 + 同心刻纹 + 盘沿金色高光随运镜滑动）
        q = 2; Lm = np.zeros((H // q, W // q), np.float32); Le = np.zeros_like(Lm)
        c = (int(900 / q), int(-60 / q)); ax = (int(1250 / q), int(560 / q))
        cv2.ellipse(Lm, c, ax, 0, 0, 180, 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(Le, c, ax, 0, 0, 180, 1.0, 4, cv2.LINE_AA)
        yy2, xx2 = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
        rr = np.hypot((xx2 - c[0]) / ax[0], (yy2 - c[1]) / ax[1])
        rings = 0.5 + 0.5 * np.sin(rr * 90) ** 8
        nz = cv2.resize(A.tex(101, 512, 5, 6), (W // q, H // q))
        spec = np.exp(-((xx2 * q - (700 + 500 * u)) / 260) ** 2) * np.clip(rr - 0.7, 0, 1) * 3
        bronze = (0.35 + 0.9 * nz)[..., None] * np.array([0.22, 0.15, 0.07], np.float32) * (0.7 + 0.3 * rings[..., None]) \
                 + spec[..., None] * np.array([0.5, 0.36, 0.16], np.float32)
        verd = np.clip((nz - 0.62) * 4, 0, 1)[..., None] * np.array([0.02, 0.06, 0.05], np.float32)
        Lm3 = cv2.resize(Lm, (W, H))[..., None]
        fr = fr * (1 - Lm3) + cv2.resize(bronze + verd, (W, H)) * Lm3
        Le = cv2.resize(Le, (W, H))
        fr = fr + (Le * 1.0 + cv2.GaussianBlur(Le, (0, 0), 10) * 0.8)[..., None] * GOLDV * (0.6 + 0.6 * cv2.resize(spec, (W, H))[..., None] + 0.3)
        # 纸人：剪纸人形（头、肩、垂袖、袍摆），沿盘沿排开，每三个一个失足滑落翻滚（带拖影）
        Lp = np.zeros((H, W), np.float32)
        body = np.array([(0, -1.0), (-0.07, -0.97), (-0.09, -0.86), (-0.06, -0.80), (-0.22, -0.74), (-0.30, -0.40), (-0.24, -0.38),
                         (-0.20, -0.62), (-0.18, -0.30), (-0.30, 0), (0.30, 0), (0.18, -0.30), (0.20, -0.62), (0.24, -0.38), (0.30, -0.40),
                         (0.22, -0.74), (0.06, -0.80), (0.09, -0.86), (0.07, -0.97)])
        for i, (x, s_, a0) in enumerate(self.ppl):
            yr = -60 + 560 * math.sqrt(max(0.0, 1 - ((x - 900) / 1250) ** 2))
            tf = self.t0 + 0.10 * i
            falling = i % 3 == 0
            for gst in ((0.0, 1.0), (0.04, 0.35), (0.08, 0.15)) if falling else ((0.0, 1.0),):
                fall = max(0.0, t - tf - gst[0]) if falling else 0.0
                y = yr + 520 * fall ** 2; ang = a0 + 260 * fall
                hh = 92 * s_
                th = math.radians(ang); Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                P = (body * hh @ Rm.T + [x + 60 * fall, y + 2]).astype(np.int32)
                tmp = np.zeros_like(Lp); cv2.fillPoly(tmp, [P], gst[1], cv2.LINE_AA)
                Lp = np.maximum(Lp, tmp)
        Lp = cv2.GaussianBlur(Lp, (0, 0), 0.8)
        paper = (0.70 + 0.3 * cv2.resize(A.tex(102, 256, 10, 4), (W, H)))[..., None] * np.array([0.80, 0.74, 0.62], np.float32)
        lit = np.clip(Lp - cv2.warpAffine(Lp, np.float32([[1, 0, -3], [0, 1, 3]]), (W, H)), 0, 1)[..., None]
        fr = fr * (1 - Lp[..., None]) + Lp[..., None] * paper * 0.45 + lit * GOLDV * 0.5
        # 红绳 + 朱墨滴（先在绳端胀大，再拉长坠落）
        rx = 960 - 50 * u
        cv2.line(fr, (int(rx), 480), (int(rx), 640), (0.55, 0.05, 0.04), 4, cv2.LINE_AA)
        fr = A.glow(fr, (rx, 570), 30, 0.15, (0.8, 0.1, 0.05), aspect=0.2)
        qd = clamp01((t - (td - 0.40)) / 0.40)
        if t < td:
            grow = clamp01((t - self.t0) / max(0.1, td - 0.4 - self.t0))
            yd = 648 + (870 - 648) * qd ** 2
            fr = A.glow(fr, (rx, yd), 7 + 6 * grow, 1.6, (0.80, 0.06, 0.04), aspect=1 - 0.5 * qd)
            fr = A.glow(fr, (rx - 3, yd - 4), 3, 0.8, (1.0, 0.7, 0.6))
        # 墨池：镜面倒影（盘沿金线 + 纸人倒影，水波扰动）
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        pool = np.clip((yy - 860) / 14, 0, 1)
        n = H - 860
        refl = cv2.flip(fr[860 - n:860], 0)
        ph = np.sin(np.arange(n, dtype=np.float32)[:, None] / 5 + t * 10) * 5
        mx = (np.arange(W, dtype=np.float32)[None] + ph).astype(np.float32); my = np.repeat(np.arange(n, dtype=np.float32)[:, None], W, 1)
        refl = cv2.remap(refl, mx, my, cv2.INTER_LINEAR)
        fr[860:] = fr[860:] * (1 - pool[860:]) + (refl * 0.5 * np.array([1.0, 0.8, 0.75], np.float32) + 0.004) * pool[860:]
        a = t - td
        if a >= 0:
            fr = fx2.ripples(fr, t, [(rx, 890)], [td], color=(0.75, 0.12, 0.07), speed=420, amt=1.0, life=1.4)
            fr = fr * (1 + 0.3 * math.exp(-a / 0.1))
        L = T.Layer()
        for i, (ch, k) in enumerate((('称', 9), ('量', 10))):
            p = clamp01((t - (CT(3, k) - 0.12)) / 0.4)
            if p > 0: L.paste(T.ink_bleed_glyph(ch, 'kai', 190, p, seed=9 + i, bleed=1.0), rx - 160 + 320 * i, 750, (0.82, 0.08, 0.06))
        fr = T.compose(fr, L, shadow=0.6, glow=0.4, glow_color=(0.9, 0.12, 0.06))
        return fr, (3 if 0 <= a < 0.08 else 0)


# V11 28.93–29.86 ECU 刀身当面（blade_face 精灵，第二棒替换单眼重复）：竖刀把脸劈成两半，两侧各一只眼；
#     左侧朱红轮廓光（承接 V10 朱墨），29.40「我」起一线红光沿刃口自下而上爬升 —— 「我杀过」之前的一口气
class V11(Shot):
    def __init__(self):
        self.path = A.defringe('A/blade_face')
    def render(self, t):
        u = self.u(t)
        fr = night((0.012, 0.005, 0.005), (0.035, 0.012, 0.012))
        fr = A.glow(fr, (300, 500), 600, 0.25, (0.7, 0.10, 0.06))
        fr, info = A.place(fr, t, self.path, (960 + 30 * u, 560), w=2350 * (1 + 0.05 * eio(u)), anc=(0.5, 0.40), rim=REDR, rim_amt=1.3,
                           rim_dir=(-0.8, -0.3), grade=dict(expo=0.70, sat=0.45, tint=(1.0, 0.95, 0.95), gamma=1.2), wind=4, breathe=0.0)
        xx = np.arange(W, dtype=np.float32)[None, :, None]
        fr = fr * (0.45 + 0.65 * np.clip(1 - np.abs(xx - 960) / 1100, 0, 1))
        # 刃口红光：沿刀身（屏幕上 x≈刀中线）自下而上
        bx0, by0 = info['to_scr']((0.5, 1.0)); bx1, by1 = info['to_scr']((0.5, 0.0))
        q = clamp01((t - (self.t0 + 0.22)) / 0.5)
        if q > 0:
            y = by0 + (by1 - by0) * eio(q)
            yy = np.arange(H, dtype=np.float32)[:, None]
            xs = np.exp(-((np.arange(W, dtype=np.float32)[None] - bx0 - 30 * u) / 8) ** 2)
            trail = np.clip((yy - y) / 600, 0, 1) * (yy > y) * np.exp(-np.clip(yy - y, 0, None) / 500)
            head = np.exp(-((yy - y) / 40) ** 2)
            band = (head * 2.0 + trail * 0.8) * xs
            fr = fr + (band * 4.0 + cv2.GaussianBlur(band, (0, 0), 16) * 6.0)[..., None] * np.array([1.0, 0.28, 0.18], np.float32)
        fr = ASH.draw(fr, t, 0.3, wind=(-60, -20))
        return fr, 0


# V12 29.86–30.77 MS 斜斩：左半画面，鬼屠斩倒恶徒（甲兵剪影倒下）；巨字「杀」朱红
class V12(Shot):
    def __init__(self):
        import crowd
        self.vill = crowd.soldier_sprite(None, arm=0.6, view='front', seed=2) if hasattr(crowd, 'soldier_sprite') else None
        T.warm('杀', 'kai', 360)
    def render(self, t):
        u = self.u(t); ts = CT(4, 1)
        fr = night((0.04, 0.008, 0.008), (0.12, 0.02, 0.02))
        fr = A.glow(fr, (1400, 420), 500, 0.35, (0.9, 0.2, 0.12))
        fr = _bamboo(fr, t, 0.10 * u)
        a = t - ts
        # 恶徒（第二棒：villain_fall 精灵替换代码剪影）：被斩后沿斩线一刀两断、上半滑落
        fr = _villain_spr(fr, t, a, (1330, 1000), 640)
        fr, info = A.place(fr, t, S('S014/slash_diag'), (720, 990), h=640, rim=REDR, rim_amt=3.0, wind=10, grade=dict(expo=0.85, sat=0.6))
        if a >= 0:
            fr = streak(fr, 480, 1700, 300, a / 0.09, w=6, amt=1.5, ang=32, col=(1.0, 0.85, 0.8))
            fr = K.sparks(fr, t, 31, (1200, 620), n=60, life=0.6, speed=(200, 800), dirc=(1, -0.4), spread=1.0, grav=1200, area=(60, 60), amt=0.8, t0=ts, burst=True)
        L = T.Layer(); p = cprog(4, [1], t, 0.16)[0]
        if p > 0: L.paste(T.brush_glyph('杀', 'kai', 360, p, seed=4), 330, 400, (0.88, 0.10, 0.07))
        fr = T.compose(fr, L, shadow=0.6, glow=0.45, glow_color=(0.9, 0.1, 0.06))
        return fr, (5 if 0 <= a < 0.1 else 0)


def _bamboo(fr, t, dx):
    """月下竹林竖条纹（后景）"""
    r = np.random.default_rng(91)
    L = np.zeros((H // 2, W // 2), np.float32)
    for i in range(34):
        x = (r.uniform(0, W) + dx * 800 * r.uniform(0.5, 1.5)) / 2
        w = r.uniform(4, 14) / 2; tone = r.uniform(0.3, 1)
        cv2.rectangle(L, (int(x), 0), (int(x + w), H // 2), tone, -1)
        for y in range(0, H // 2, int(r.uniform(40, 80))): cv2.line(L, (int(x - 1), y), (int(x + w + 1), y), tone * 0.5, 1)
    L = cv2.resize(cv2.GaussianBlur(L, (0, 0), 3), (W, H))
    return fr * (1 - 0.65 * L[..., None])


def _villain_spr(fr, t, a, at, h, deg=32):
    """恶徒精灵：先在黑底单独合成（得到预乘色 + alpha），斩后沿 deg 斜线分成两半，上半沿斩线滑开下坠；断口朱红。"""
    z = np.zeros((H, W, 3), np.float32)
    back = 0.0 if a < 0 else 40 * eout(a / 0.4)
    P, info = A.place(z, t, A.bg_matte('A/villain_fall'), (at[0] + back, at[1]), h=h, anc=(0.5, 1.0), rim=REDR, rim_amt=2.4,
                      rim_dir=(-0.8, -0.4), grade=dict(expo=0.75, sat=0.12, tint=(1.06, 0.94, 0.90)), wind=8, rot=-6 * eout(max(0, a) / 0.5))   # 第三棒：sat 0.5→0.12 去腿部青色溢色
    Al = info['A'][..., None]
    col = P / np.maximum(Al, 1e-3); Al2 = np.clip((Al - 0.25) / 0.75, 0, 1)                         # 第三棒：收 matte 灰边
    Al2 = np.minimum(Al2, cv2.erode(Al2[..., 0], np.ones((3, 3), np.uint8))[..., None] + 0.0)
    P = col * Al2; Al = Al2
    if a > 0:
        th = math.radians(deg); cy = at[1] - 0.55 * h
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        n = -(xx - at[0]) * math.sin(th) + (yy - cy) * math.cos(th)
        side = (n < 0).astype(np.float32)[..., None]
        sl = 240 * a ** 1.2
        M = np.float32([[1, 0, sl * math.cos(th)], [0, 1, sl * math.sin(th) + 260 * a * a]])
        Pu = cv2.warpAffine(P * side, M, (W, H)); Au = cv2.warpAffine(Al * side, M, (W, H))[..., None]
        P = P * (1 - side) * (1 - Au) + Pu; Al = Al * (1 - side) * (1 - Au) + Au     # 第三棒：上半盖在下半上（原相加 Al>1 → 背景被减成青色）
        seam = np.exp(-(n / 3.0) ** 2) * (np.abs(xx - at[0]) < h * 0.4)
        fr = fr + seam[..., None] * RED * 2.2 * math.exp(-a / 0.5) * cv2.GaussianBlur(info['A'], (0, 0), 6)[..., None] * 3
    fr = fr * (1 - Al) + P
    return fr


def _villain(fr, x, y, h, ang, red=False, cut=None):
    """恶徒剪影：宽肩甲兵（代码）；cut=(斩后秒, 角度)：沿斜线一刀两断，上半沿斩线滑落。"""
    m = np.zeros((H, W), np.float32)
    s = h / 100
    pts = [(-16, 0), (-12, -45), (-24, -62), (-20, -80), (-8, -86), (-7, -94), (7, -94), (8, -86), (20, -80), (24, -62), (12, -45), (16, 0)]
    th = math.radians(ang)
    P = [(x + (px * math.cos(th) - py * math.sin(th)) * s, y + (px * math.sin(th) + py * math.cos(th)) * s) for px, py in pts]
    cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
    hc = (x + (0 * math.cos(th) + 100 * math.sin(th)) * s, y + (0 * math.sin(th) - 100 * math.cos(th)) * s)
    cv2.circle(m, (int(hc[0]), int(hc[1])), int(9 * s), 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    if cut is not None and cut[0] > 0:
        a, deg = cut; th2 = math.radians(deg)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        side = (-(xx - x) * math.sin(th2) + (yy - (y - 0.55 * h)) * math.cos(th2)) < 0
        sl = 260 * a ** 1.3
        up = cv2.warpAffine(m * side, np.float32([[1, 0, -sl * math.cos(th2)], [0, 1, -sl * math.sin(th2) + 300 * a * a]]), (W, H))
        m = np.maximum(m * (~side), up)
        seam = np.exp(-((-(xx - x) * math.sin(th2) + (yy - (y - 0.55 * h)) * math.cos(th2)) / 3) ** 2) * (np.abs(xx - x) < h * 0.35)
        fr = fr + seam[..., None] * RED * 2.0 * math.exp(-a / 0.5)
    fr = fr * (1 - m[..., None]) + m[..., None] * 0.01
    e = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
    return fr + e[..., None] * (np.array([1.0, 0.3, 0.2], np.float32) if red else COLDV) * 0.6


# V13 30.77–32.60 MS 收刀：右半，冷光；跪地求饶的混账剪影在左前景；巨字「放」墨灰
class V13(Shot):
    def __init__(self):
        T.warm('放', 'kai', 360)
    def render(self, t):
        u = self.u(t)
        fr = night((0.012, 0.016, 0.03), (0.05, 0.06, 0.08))
        fr = A.glow(fr, (500, 380), 500, 0.3, (0.55, 0.62, 0.85))
        fr = _bamboo(fr, t, -0.08 * u)
        fr = fog(fr, t, 2, 700, 960, 0.45, (0.25, 0.28, 0.34), speed=(-40, 0))
        fr, info = A.place(fr, t, S('S014/sheath'), (1250, 990), h=700, rim=COLD, rim_amt=3.2, wind=8, grade=dict(expo=0.85, sat=0.5))
        # 跪地者（第二棒：kneel_beg 精灵替换代码糊团）：左前景，面朝鬼屠磕头求饶；前景虚焦 + 冷轮廓
        bow = 0.5 + 0.5 * math.sin(t * 6.0)
        fr, _ = A.place(fr, t, A.bg_matte('A/kneel_beg'), (560, 1010), h=560, anc=(0.55, 1.0), flip=True, rim=COLD, rim_amt=2.6,
                        rim_dir=(0.7, -0.6), grade=dict(expo=0.55, sat=0.4, gamma=1.3), wind=3, rot=2.5 * bow, defocus=1.2)
        L = T.Layer(); p = cprog(4, [6], t, 0.25)[0]
        if p > 0: L.paste(T.ink_bleed_glyph('放', 'kai', 330, p, seed=6, bleed=0.8), 640, 330, (0.62, 0.65, 0.72))   # 第三棒：悬在被放过的人头顶
        fr = T.compose(fr, L, shadow=0.6, glow=0.1)
        return fr, 0


# V14 32.60–34.44 CU 手握刀柄慢拔：刀身阴刻歌词随拔出依次亮起；34.10「刀」朱红 + 白刃闪
class V14(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.01, 0.01, 0.014), (0.03, 0.03, 0.035))
        fr = A.glow(fr, (1500, 300), 450, 0.25, (1.0, 0.6, 0.3))
        fr, info = A.place(fr, t, S('S015/hand_draw'), (960 - 50 * u, 560), w=2350 * (1 + 0.06 * u), anc=(0.5, 0.5), rim=COLD, rim_amt=2.0,
                           grade=dict(expo=0.88, sat=0.55), wind=4)
        P0 = info['to_scr']((0.335, 0.585)); P1 = info['to_scr']((0.585, 0.825))                    # 刀身（护手 → 刀尖方向）
        txt = '每一次拔刀都是我的'
        prog = cprog(5, list(range(9)), t, 0.25)
        if prog[0] > 0:
            fr = T.carved_on_blade(fr, txt, [P0, P1], height=66, p=min(1.0, sum(prog) / 9 + 0.02), fname='kai',
                                   finish='steel', taper=(1.0, 0.9), fill=True)
        a = t - CT(5, 4)
        if a >= 0:
            fr = streak(fr, (P0[1] + P1[1]) / 2, P0[0], P1[0], a / 0.12, amt=1.2, ang=math.degrees(math.atan2(P1[1] - P0[1], P1[0] - P0[0])))
        fr = EMB.draw(fr, t, 0.6, wind=(80, -60))
        return fr, (4 if 0 <= a < 0.1 else 0)


# V15 34.44–36.27 过肩大前景 + 城门石碑（第四棒：去月）：右 1/3 鬼屠肩背黑剪影，越肩看左侧雨中巨碑；「主张」逐笔凿进碑面
#     （阴刻：凹槽暗 + 下缘反光 + 石屑），推镜；入：V14 刀身刻字 ↔ 碑面刻字 match_cut
class V15(Shot):
    def __init__(self):
        T.warm('主张', 'kai', 250)
        self.tex = cv2.resize(A.tex(151, 512, 8, 6), (W, H)); self.tex2 = cv2.resize(A.tex(152, 512, 40, 3), (W, H))
    def render(self, t):
        u = self.u(t)
        fr = night((0.01, 0.012, 0.022), (0.04, 0.045, 0.06))
        fr = A.glow(fr, (300, 150), 700, 0.3, (0.5, 0.58, 0.78))
        fr = RAIN.draw(fr, t, 0.7, angle=8, light=(300, 150), layers=(0,))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        x0, x1, y0 = 330, 1210, 170
        st = ((xx > x0) & (xx < x1) & (yy > y0)).astype(np.float32); st = cv2.GaussianBlur(st, (0, 0), 1.5)
        lit = np.clip(1.15 - (xx - x0) / 1400 - (yy - y0) / 1800, 0.25, 1)
        stone = ((0.6 + 0.45 * self.tex) * (0.85 + 0.25 * self.tex2) * lit)[..., None] * np.array([0.13, 0.135, 0.15], np.float32)
        wet = np.clip(np.sin(xx / 23 + self.tex * 6) * 0.5 + 0.5, 0, 1) ** 8 * (yy > y0)          # 雨水竖流痕
        stone = stone + (wet * 0.06)[..., None] * np.array([0.6, 0.7, 0.9], np.float32)
        fr = fr * (1 - st[..., None]) + stone * st[..., None]
        ed = np.exp(-((xx - x0) / 3) ** 2) * (yy > y0) + np.exp(-((yy - y0) / 3) ** 2) * (xx > x0) * (xx < x1)
        fr = fr + ed[..., None] * np.array([0.5, 0.58, 0.8], np.float32) * 0.6
        G = np.zeros((H, W), np.float32)
        for i, (c, k) in enumerate((('主', 9), ('张', 10))):
            p = clamp01((t - (CT(5, k) - 0.25)) / 0.5)
            if p > 0:
                g = T.ink_bleed_glyph(c, 'kai', 300, p, seed=20 + i, bleed=0.4)
                h_, w_ = g.shape; cx, cy = 770, 420 + 330 * i
                ys, xs = int(cy - h_ / 2), int(cx - w_ / 2)
                G[max(0, ys):ys + h_, max(0, xs):xs + w_] = np.maximum(G[max(0, ys):ys + h_, max(0, xs):xs + w_], g[max(0, -ys):, max(0, -xs):][:H - max(0, ys), :W - max(0, xs)])
        if G.max() > 0:
            Gs = cv2.GaussianBlur(G, (0, 0), 1.2)
            sh = np.clip(Gs - cv2.warpAffine(Gs, np.float32([[1, 0, 5], [0, 1, 6]]), (W, H)), 0, 1)   # 凹槽上缘阴影
            hl = np.clip(Gs - cv2.warpAffine(Gs, np.float32([[1, 0, -4], [0, 1, -5]]), (W, H)), 0, 1)  # 下缘反光
            fr = fr * (1 - 0.85 * Gs[..., None]) + Gs[..., None] * np.array([0.012, 0.012, 0.016], np.float32)
            fr = fr * (1 - 0.6 * sh[..., None]) + hl[..., None] * np.array([0.55, 0.62, 0.85], np.float32) * 0.9
            for i, k in enumerate((9, 10)):
                a = t - CT(5, k)
                if 0 <= a < 0.6:
                    fr = K.sparks(fr, t, 150 + i, (770, 420 + 330 * i), n=40, life=0.5, speed=(150, 500), dirc=(0.6, -0.4), spread=1.5, grav=900,
                                  area=(90, 90), amt=0.6, t0=CT(5, k), burst=True, color=(0.8, 0.82, 0.9))
        fr = fog(fr, t, 0, 780, 1000, 0.4, (0.22, 0.25, 0.32), speed=(-50, 0))
        fr, info = A.place(fr, t, S('S002/back_wind'), (1520 - 40 * u, 1760), h=1580, flip=True, look='sil', rim=COLD, rim_amt=4.5,
                           rim_dir=(-0.9, -0.3), wind=14, defocus=1.0)
        fr = RAIN.draw(fr, t, 0.8, angle=8, layers=(1, 2))
        return fr, 0


# V16 36.27–38.10 LS 仰拍太上皇举手召印：头顶云涡旋开，金光中一方巨印的阴影成形
class V16(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.02, 0.016, 0.012), (0.08, 0.07, 0.06), y1=900)
        c = (720, 230)
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
        ang = np.arctan2(yy * 4 - c[1], xx * 4 - c[0]); r = np.hypot(xx * 4 - c[0], yy * 4 - c[1])
        sw = np.sin(ang * 3 + r / 60 - t * 2.5) * 0.5 + 0.5
        vort = cv2.resize((sw * np.exp(-r / 600)).astype(np.float32), (W, H))
        fr = fr + vort[..., None] * np.array([0.25, 0.2, 0.12], np.float32) * (0.5 + u)
        fr = fx2.god_rays(fr, c, strength=0.6 + 0.8 * u, length=1.2, color=(1.0, 0.8, 0.5), t=t, src_r=140)
        # 巨印轮廓（尚在云中，半透明）
        k = eio(clamp01((t - V(37.3)) / 0.8))
        if k > 0:
            m = np.zeros((H, W), np.float32)
            cv2.rectangle(m, (int(c[0] - 230), int(c[1] - 200 - 40 * k)), (int(c[0] + 230), int(c[1] + 60)), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 30 * (1 - k) + 4)
            fr = fr * (1 - m[..., None] * 0.6 * k)
        fr, info = A.place(fr, t, S('A/tsh_raise', 'dm'), (1250, 1040), h=860, look=None, rim=GOLD, rim_amt=4.0, rim_dir=(-0.6, -0.8),
                           grade=dict(expo=0.55, sat=0.45, tint=(1.05, 0.97, 0.9)), wind=7)
        fr = fog(fr, t, 1, 820, 1000, 0.5, (0.25, 0.22, 0.2), speed=(40, 0))
        return fr, 0


# V17 38.10–39.47 ELS 巨印「正」自天砸下，38.72 落地：尘浪环形爆开、全屏震
class V17(Shot):
    """仰视：巨印「正」的印面自天压下、越来越大（38.10–38.72）→ 落地：画面压黑 + 震屏 → 俯拍：大地上压出朱红「正」字印痕、裂纹、尘浪环"""
    def __init__(self):
        self.face = PR.seal_face('正', 900, style='bai', wear=0.25, seed=6)
        r = np.random.default_rng(17)
        self.cracks = [K.crack_path((960, 600), (960 + math.cos(a) * 1300, 600 + math.sin(a) * 700), 17 + i) for i, a in
                       enumerate(r.uniform(0, 2 * math.pi, 9))] if hasattr(K, 'crack_path') else []
    def render(self, t):
        th = V(38.72); a = t - th
        if a < 0:
            q = clamp01((t - self.t0) / (th - self.t0))
            fr = night((0.03, 0.025, 0.025), (0.10, 0.08, 0.07))
            fr = A.clouds(fr, t, 71, speed=(0, -60), y0=120, y1=960, amt=0.9, col=(0.13, 0.11, 0.10), thr=0.42, light=(960, 520),
                          light_r=700, light_col=(1.0, 0.75, 0.45))
            sz = int(260 + 2200 * q ** 2.2)
            sh = A.glow(np.zeros_like(fr), (960, 540), sz * 0.7, 1.0, (1, 1, 1))[..., :1]
            fr = fr * (1 - np.clip(sh * 0.8, 0, 0.85))
            img = cv2.resize(self.face, (sz, sz), interpolation=cv2.INTER_AREA)
            stone = np.zeros((sz + 40, sz + 40, 4), np.float32); stone[..., :3] = (0.05, 0.05, 0.05); stone[..., 3] = 1.0
            fr = K.paste_sprite(fr, stone[..., :3], stone[..., 3], 960, 540, alpha=1.0)
            fr = K.paste_sprite(fr, img[..., :3] * 1.1, img[..., 3], 960, 540)
            fr = fx2.god_rays(fr, (960, 540), strength=0.5 * (1 - q), length=1.3, color=(1.0, 0.8, 0.5), t=t, src_r=sz * 0.6)
            return fr, 0
        # 第二棒：夯土石板地面（大块石板缝 + 细噪声），印坑下陷（偏移投影 + 斜面高光），裂缝朱红余热
        fr = A.vgrad((0.11, 0.09, 0.075), (0.05, 0.04, 0.034))
        if not hasattr(self, 'ground'):
            g = 0.75 + 0.5 * cv2.resize(A.tex(5, 512, 8, 5), (W, H)) + 0.25 * cv2.resize(A.tex(6, 512, 40, 3), (W, H))
            sl = np.zeros((H, W), np.float32)
            r = np.random.default_rng(5); y = 140
            while y < H:
                cv2.line(sl, (0, y), (W, y + int(r.uniform(-8, 8))), 1.0, 2, cv2.LINE_AA); x = int(r.uniform(0, 200))
                while x < W:
                    cv2.line(sl, (x, y), (x + int(r.uniform(-6, 6)), y + 150), 1.0, 2, cv2.LINE_AA); x += int(r.uniform(220, 380))
                y += int(r.uniform(130, 170))
            self.ground = (g * (1 - 0.6 * cv2.GaussianBlur(sl, (0, 0), 1.2)))[..., None]
            sq = np.zeros((H, W), np.float32); cv2.rectangle(sq, (960 - 290, 600 - 290), (960 + 290, 600 + 290), 1.0, -1)
            self.pit = cv2.GaussianBlur(cv2.warpAffine(sq, np.float32([[1, 0, 18], [0, 1, 24]]), (W, H)), (0, 0), 14)
            e = cv2.GaussianBlur(sq, (0, 0), 3); self.bevel = np.clip(e - cv2.warpAffine(e, np.float32([[1, 0, 6], [0, 1, 6]]), (W, H)), 0, 1)
        fr = fr * self.ground
        for c in self.cracks:
            m = K.draw_crack((H, W), c, 4, branches=4, seed=3, prog=clamp01(a / 0.5))
            mg = cv2.GaussianBlur(m, (0, 0), 6)
            fr = fr * (1 - m[..., None] * 0.9) + (m[..., None] * 0.9 + mg[..., None] * 1.2) * RED * (0.25 + 0.6 * math.exp(-a / 0.6))
        fr = fr * (1 - 0.75 * self.pit[..., None])
        fr = PR.stamp_imprint(fr, self.face, 960, 600, 560, prog=1.0, alpha=0.95)
        fr = fr + self.bevel[..., None] * np.array([0.9, 0.7, 0.5], np.float32) * 0.5
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        d = np.hypot(xx * 2 - 960, (yy * 2 - 600) * 1.6)
        R = 320 + 1400 * eout(a / 0.9)
        ring = np.exp(-((d - R) / (90 + 260 * a)) ** 2) * math.exp(-a / 0.8)
        rg = cv2.resize(ring, (W, H))[..., None] * (0.6 + 0.4 * cv2.resize(A.tex(9, 256, 6, 4), (W, H))[..., None])
        rg = rg * np.clip(cv2.resize(A.ink_noise(19, 256), (W, H))[..., None] * 1.6 - 0.3, 0, 1)        # 第三棒：尘浪打碎成团，不再是整圈椭圆光环
        fr = fr * (1 - rg * 0.35) + rg * np.array([0.20, 0.16, 0.12], np.float32)
        fr = K.sparks(fr, t, 71, (960, 600), n=90, life=0.9, speed=(300, 1300), dirc=(0, -1), spread=3.0, grav=0, area=(260, 260), amt=0.6,
                      t0=th, burst=True, color=(0.8, 0.7, 0.55))
        if a < 0.06: fr = fr * 0.35
        elif a < 0.14: fr = fr * 1.45 + 0.08 * GOLDV          # 第三棒：暖色曝光脉冲，不再灰白
        return fr, (6 if 0 <= a < 0.2 else 0)


# V18 39.47–40.84 HA 高机位大远景（第四棒：原 MCU 回头 → 俯拍）：印坑边缘的石板地上鬼屠一人极小、长影拖向画外，尘浪掠过，
#     斜雨冷线；降饱和（破晓留给后段）
class V18(Shot):
    def __init__(self):
        g = 0.75 + 0.5 * cv2.resize(A.tex(5, 512, 8, 5), (W, H)) + 0.25 * cv2.resize(A.tex(6, 512, 40, 3), (W, H))
        sl = np.zeros((H, W), np.float32); r = np.random.default_rng(8)
        for y in range(-200, H + 200, 150):
            cv2.line(sl, (0, y), (W, y + 260), 1.0, 2, cv2.LINE_AA)
        for x in range(-400, W + 400, 260):
            cv2.line(sl, (x, 0), (x - 200, H), 1.0, 2, cv2.LINE_AA)
        self.ground = (g * (1 - 0.55 * cv2.GaussianBlur(sl, (0, 0), 1.2)))[..., None]
        pit = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(pit, np.array([(-300, -200), (820, -60), (640, 420), (-420, 260)], np.int32), 1.0)
        self.pit = cv2.GaussianBlur(pit, (0, 0), 10)
    def render(self, t):
        u = self.u(t)
        fr = A.vgrad((0.07, 0.065, 0.06), (0.045, 0.045, 0.05)) * self.ground
        fr = fr * (1 - 0.8 * self.pit[..., None])
        e = np.clip(self.pit - cv2.warpAffine(self.pit, np.float32([[1, 0, -8], [0, 1, -8]]), (W, H)), 0, 1)
        fr = fr + e[..., None] * np.array([0.6, 0.25, 0.15], np.float32) * 0.8                  # 印坑边缘余热
        fr = A.glow(fr, (300, 150), 600, 0.3, (0.9, 0.7, 0.5))
        at = (1180 + 40 * u, 820)
        sh = np.zeros((H // 2, W // 2), np.float32)                                              # 长影
        cv2.line(sh, (int(at[0] / 2), int(at[1] / 2)), (int((at[0] + 520) / 2), int((at[1] + 300) / 2)), 1.0, 22, cv2.LINE_AA)
        sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 6), (W, H))[..., None]
        fr = fr * (1 - 0.6 * sh)
        fr, info = A.place(fr, t, S('A/stand_ha'), at, h=430, rim=COLD, rim_amt=2.6, rim_dir=(-0.8, -0.5), wind=13,
                           grade=dict(expo=0.75, sat=0.35, tint=(0.97, 0.98, 1.02)))
        fr = fog(fr, t, 1, 300, 1000, 0.5, (0.30, 0.28, 0.26), speed=(380, 60), light=(300, 150))
        fr = RAIN.draw(fr, t, 0.7, angle=20, light=(300, 150), layers=(0, 1, 2))
        fr = ASH.draw(fr, t, 0.8, wind=(420, 40))
        return fr, 0


# V19 40.84–41.77 MCU 仰望巨印（第四棒：人物缩小留出头顶，推镜改为有方向的竖向拖影，眼中寒光；冷雨边光）
class V19(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.02, 0.02, 0.024), (0.06, 0.06, 0.065))
        fr = A.glow(fr, (960, 100), 600, 0.4, (1.0, 0.82, 0.55))
        fr = RAIN.draw(fr, t, 0.6, angle=4, light=(960, 100), layers=(0, 1))
        hh = 780 * (1 + 0.08 * eio(u))
        fr, info = A.place(fr, t, S('S018/look_up'), (1220, 1110 + 30 * u), h=hh, anc=(0.5, 1.0), rim=COLD, rim_amt=3.0,
                           rim_dir=(0.0, -1.0), wind=10, grade=dict(expo=0.82, sat=0.38, tint=(0.97, 0.99, 1.03)),
                           mblur=(0, 10 * eio(u)))
        g = math.exp(-max(0, t - V(41.40)) / 0.12) * (t >= V(41.40))
        for p in ((0.455, 0.235), (0.545, 0.235)):
            fr = pupil_glint(fr, info['to_scr'](p), 0.6 * g, 1.0)
        fr = RAIN.draw(fr, t, 0.6, angle=4, layers=(2,))
        return fr, (3 if g > 0.5 else 0)


# V20 41.77–42.68 LS 极低机位：鬼屠自下而上斜撩一刀，刀光弧线劈向天上的巨印
class V20(Shot):
    def render(self, t):
        u = self.u(t)
        fr = night((0.04, 0.035, 0.03), (0.12, 0.10, 0.09), y1=800)
        fr = A.glow(fr, (1300, 200), 500, 0.5, (1.0, 0.8, 0.5))
        fr = A.clouds(fr, t, 83, speed=(0, 60), y0=120, y1=520, amt=0.8, col=(0.12, 0.10, 0.09), light=(1300, 200), light_r=500,
                      light_col=(1.0, 0.8, 0.5), thr=0.44)
        fr = RAIN.draw(fr, t, 0.6, angle=-6, light=(1300, 200), layers=(0, 1))
        fr, info = A.place(fr, t, S('A/slash_up', 'dm'), (900, 1020), h=820, rim=COLD, rim_amt=3.2, rim_dir=(0.6, -0.8), wind=12,
                           grade=dict(expo=0.82, sat=0.38))                       # 第四棒：降饱和 + 冷雨边光
        a = t - V(42.0)
        if a >= 0:
            fr = K.sparks(fr, t, 81, (1300, 330), n=70, life=0.6, speed=(200, 900), dirc=(0, 1), spread=1.3, grav=900, area=(300, 10), amt=0.8, t0=V(42.0), burst=True)
            L = np.zeros((H // 2, W // 2), np.float32)
            q = min(1, a / 0.1)
            cv2.ellipse(L, (int(700 / 2), int(1000 / 2)), (int(900 / 2), int(750 / 2)), 0, -90, -90 + 85 * q, 1.0, 5, cv2.LINE_AA)
            L = cv2.resize(L, (W, H)) * max(0, 1 - a / 0.5)
            fr = fr + (L * 1.4 + cv2.GaussianBlur(L, (0, 0), 16) * 1.5)[..., None] * np.array([1.0, 0.92, 0.8], np.float32)
            if a < 0.06: fr = fr * 1.4 + 0.08 * GOLDV
        return fr, (5 if 0 <= a < 0.12 else 0)


# V21 42.68–44.03 ECU 巨字「锋」压满屏，与巨印「正」一同被斩成两半错开，碎块飞散；末帧墨缝 → 交给 B 组（墨渗成纸）
class V21(Shot):
    def __init__(self):
        T.warm('锋', 'xing', 560); T.warm('芒', 'xing', 150)
        self.face = PR.seal_face('正', 760, style='bai', wear=0.35, seed=4)
    def render(self, t):
        u = self.u(t); ts = V(43.30)
        fr = night((0.03, 0.025, 0.025), (0.06, 0.05, 0.05))
        a = t - ts
        L = T.Layer()                                                                   # 印面
        L.paste_rgba(self.face[..., :3], self.face[..., 3] * 0.9, 960 - 380, 540 - 380)
        p = cprog(7, [8], t, 0.12)[0]
        if p > 0: T.place(L, '锋', 900, 520, 560, (0.97, 0.94, 0.88), 'xing', alpha=eout(p), scale=1 + 0.3 * (1 - eout(p)))
        p2 = cprog(7, [9], t, 0.15)[0]
        if p2 > 0: T.place(L, '芒', 1350, 760, 150, (0.86, 0.12, 0.08), 'xing', alpha=eout(p2))
        if a < 0:
            fr = T.compose(fr, L, shadow=0.7, glow=0.25)
        else:
            fr = T.slash_layer(fr, L, clamp01(a / 0.73), (960, 540), -62, slide=0.35, sz=560, length=1500, sparks=90,
                               style_kw=dict(shadow=0.7, glow=0.3))
            if a < 0.07: fr = fr * 1.5 + 0.07 * RED
            # 墨缝：斩线处渗出墨（给 B 组 ink_bleed 入口）
            k = clamp01((a - 0.35) / 0.38)
            if k > 0:
                yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
                th = math.radians(-62); n = (-(xx * 2 - 960) * math.sin(th) + (yy * 2 - 540) * math.cos(th))
                nz = cv2.resize(A.ink_noise(5, 256), (W // 2, H // 2))
                ink = np.clip((60 + 900 * k ** 2 + nz * 200 - np.abs(n)) / 40, 0, 1)
                ink = cv2.resize(ink, (W, H))[..., None]
                fr = fr * (1 - ink) + INK * ink
                seam = cv2.resize(np.exp(-(n / 4) ** 2), (W, H))[..., None]
                fr = fr + seam * RED * 1.2 * (1 - k * 0.5)
        return fr, (5 if 0 <= a < 0.1 else 0)


# =====================================================================================
# 剪辑表：切点帧（=拍点最近帧 - 1，见 A_storyboard.md）
SHOTS = [A01, A02, A03, A04, A05, A06, A07a, A07b, A08,
         V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21]
BEAT_T = [0.0, 2.21, 4.06, 5.97, 7.38, 7.85, 10.61, 11.98, 13.35,
          15.21, 17.00, 17.95, 19.78, 20.69, 21.20, 22.52, 23.45, 25.29, 27.10, 28.93, 29.86, 30.77, 32.60, 34.44, 36.27, 38.10,
          39.47, 40.84, 41.77, 42.68, 44.034]
CUT_F = [0] + [round(b * FPS) - 1 for b in BEAT_T[1:-1]] + [F1]
for _S, _a, _b in zip(SHOTS, CUT_F, CUT_F[1:]):
    _S.t0, _S.t1 = _a / FPS, _b / FPS

# (名称, pre, post, kw)；'hard' = 硬切 + 切后 2 帧冲击；第四棒：whip 0、occlusion 3→2、slash_wipe 3→2、match_cut 3→2，
# 新增 a_cloud / a_zoom2 / a_seam（中间帧都带前后镜元素）
CUTS = {
    'A02': ('a_cloud', 0.20, 0.24, dict(seed=2)),                                     # 云墙横扫：A01 云海 → A02 山脊
    'A03': ('a_zoom2', 0.12, 0.16, dict(center=(640, 700), seed=3)),                 # 推进人物腰间 → 握刀手
    'A04': ('ink_bleed', 0.12, 0.18, dict(seed=7, center=(1300, 400))),
    'A05': ('hard', 0, 0, {}),
    'A06': ('hard', 0, 0, {}),                                                        # 片名刀光在镜内（A05 末帧预光条）
    'A07a': ('occlusion', 0.14, 0.16, dict(kind='pillar', direction=-1, seed=6)),     # 前景旗杆擦过（片名屏幕层不动）
    'A07b': ('zoom_through', 0.16, 0.20, dict(center=(960, 620), variant='raindrop')),  # 穿过一滴雨：滴内倒映城墙
    'A08': ('smoke_through', 0.06, 0.40, dict(seed=8, direction=1, density=0.65)),    # 片名化灰成烟，烟后露出城门
    'V01': ('a_seam', 0.10, 0.20, {}),                                               # 城门合缝，门缝里开出雨刃
    'V02': ('hard', 0, 0, {}),
    'V03': ('shatter', 0.04, 0.20, dict(seed=3, center=(1060, 880), n=60)),
    'V04': ('occlusion', 0.14, 0.14, dict(kind='spear', direction=-1, seed=4)),
    'V05': ('hard', 0, 0, {}),
    'V06': ('hard', 0, 0, {}),
    'V07': ('match_cut', 0.08, 0.10, dict(shape='line', center=(960, 600), angle=0, mode='cut')),   # 寒光横线 ↔ 丹陛顶边
    'V08': ('paper_tear', 0.10, 0.20, dict(seed=8, angle=6)),
    'V09': ('a_zoom2', 0.12, 0.16, dict(center=(560, 700), seed=9)),                 # 推进「罪状」朱印 → 天平
    'V10': ('hard', 0, 0, {}),
    'V11': ('flash_red', 0.03, 0.10, {}),
    'V12': ('slash_wipe', 0.0, 0.14, dict(angle=58)),
    'V13': ('split_slide', 0.10, 0.18, dict(angle=90)),
    'V14': ('hard', 0, 0, {}),
    'V15': ('match_cut', 0.08, 0.12, dict(shape='line', center=(770, 560), angle=90, mode='cut')),   # 刀身刻字 ↔ 碑面刻字
    'V16': ('ink_bleed', 0.06, 0.26, dict(seed=16, center=(770, 420))),               # 「主」字墨槽晕开成召印金光
    'V17': ('hard', 0, 0, {}),
    'V18': ('flash_red', 0.03, 0.10, {}),
    'V19': ('hard', 0, 0, {}),
    'V20': ('hard', 0, 0, {}),
    'V21': ('slash_wipe', 0.0, 0.14, dict(angle=-62)),
}

M3 = V3.Move3
MOVES3 = {
    'A01': M3('sine_io', None, z=(1.0, 1.05), y=(-120, 0), note='摇臂降 crane down + 缓推'),
    'A02': M3('sine_io', 'beat', x=(-70, 70), z=(1.02, 1.05), note='横移 truck right（前景松枝视差）'),
    'A03': M3('sine_io', 'beat', x=(40, -60), z=(1.0, 1.06), note='沿刀身滑移 + 推'),
    'A04': M3('sine_io', 'beat', z=(1.0, 1.09), x=(20, -20), note='推 + 手持'),
    'A05': M3('expo_in', None, z=(1.0, 1.10), note='急推（蓄力）'),
    'A06': M3('expo_out', None, z=(1.10, 1.02), rot=(-1.2, 0), note='punch 拉开后缓推'),
    'A07a': M3('sine_io', 'beat', z=(1.0, 1.09), x=(30, -40), pitch=(-4, 1), note='低机位推轨 + 仰'),
    'A07b': M3('sine_io', 'beat', x=(60, -60), z=(1.02, 1.06), note='横移（城墙纵深视差）'),
    'A08': M3('sine_io', 'beat', z=(1.0, 1.10), note='背影推进'),
    'V01': M3('sine_io', 'beat', z=(1.0, 1.08), pitch=(2, -1), note='推 + 下摇'),
    'V02': M3('expo_out', None, z=(1.06, 1.0), x=(-40, 30), note='低机位跟步'),
    'V03': M3('sine_io', 'beat', x=(-60, 60), pitch=(6, 0), note='高机位横移 + 俯仰'),
    'V04': M3('sine_io', 'beat', x=(-50, 60), note='横移跟走'),
    'V05': M3('expo_in', None, z=(1.0, 1.12), note='急推'),
    'V06': M3('expo_out', None, z=(1.10, 1.0), x=(30, -20), note='冲击后拉'),
    'V07': M3('sine_io', 'beat', y=(60, -40), pitch=(-6, 0), note='摇臂升 crane up'),
    'V08': M3('sine_io', 'beat', z=(1.0, 1.06), note='推'),
    'V09': M3('sine_io', 'beat', yaw=(-12, 8), z=(1.0, 1.06), note='环绕 orbit'),
    'V10': M3('sine_io', 'beat', y=(-40, 50), z=(1.0, 1.05), note='下摇跟墨滴'),
    'V11': M3('sine_io', None, z=(1.0, 1.06), note='极慢推'),
    'V12': M3('sine_io', 'beat', x=(-50, 40), rot=(-2, 1), note='手持 + 横移'),
    'V13': M3('sine_io', 'beat', x=(40, -40), z=(1.04, 1.0), note='反向横移 + 拉'),
    'V14': M3('sine_io', 'beat', x=(-50, 50), z=(1.0, 1.05), note='沿刀身滑移'),
    'V15': M3('expo_out', None, z=(1.15, 1.0), note='后拉 crane back'),
    'V16': M3('sine_io', 'beat', y=(40, -30), pitch=(-5, 0), z=(1.0, 1.05), note='仰拍升'),
    'V17': M3('sine_io', 'beat', y=(-60, 40), note='跟印下摇'),
    'V18': M3('sine_io', 'beat', z=(1.08, 1.0), rot=(-2, 0), note='高机位拉 + 回旋'),
    'V19': M3('sine_io', None, z=(1.0, 1.05), y=(30, -10), note='缓推 + 升'),
    'V20': M3('expo_out', None, y=(60, -20), pitch=(-5, 0), z=(1.06, 1.0), note='硬切冲击后仰摇'),
    'V21': M3('expo_out', None, z=(1.08, 1.0), rot=(2, 0), note='punch 后稳'),
}
FG = {'A02': ('ash', 0.3), 'A04': ('ash', 0.25), 'A06': ('ember', 0.5), 'A07a': ('rain', 0.4), 'A07b': ('rain', 0.35), 'A08': ('rain', 0.3), 'V01': ('ash', 0.3), 'V02': ('rain', 0.45),
      'V04': ('rain', 0.4), 'V06': ('rain', 0.35), 'V07': ('gold', 0.3), 'V09': ('gold', 0.3), 'V12': ('ember', 0.4), 'V13': ('ash', 0.3),
      'V14': ('ember', 0.4), 'V16': ('gold', 0.4), 'V17': ('ash', 0.5), 'V18': ('rain', 0.4), 'V19': ('rain', 0.3), 'V20': ('ember', 0.4), 'V21': ('ember', 0.5)}
EXPO = {}
EXPO_ALL = 1.15

# 段落能量包络（前奏轻 → 片名重击 → 主歌压低 → 「杀」「正义」「锋芒」加码）
ENERGY = [(0, 0.3), (7.8, 0.4), (7.85, 1.3), (9.0, 0.8), (13.3, 0.5), (15.2, 0.7), (21.2, 1.1), (22.5, 0.75), (29.8, 1.0),
          (32.6, 0.8), (38.6, 1.3), (39.5, 0.9), (41.7, 1.2), (44.1, 1.3)]
_ET = np.array([e[0] for e in ENERGY]); _EV = np.array([e[1] for e in ENERGY])
def energy(t): return float(np.interp(t, _ET, _EV))


# ---------------- 屏幕层：署名（前奏）——思源宋职务 + 马善政人名竖排 + 名后小朱印，墨晕逐字显出
CREDITS = [  # (显出起点, 退场起点, 职务, 人名, x, y_top, seal)
    (0.45, 1.95, '出品人', '西几王传媒', 560, 250, '西几王'),
    (0.60, 1.95, '后期总监', '沉溺', 360, 250, None),               # HANDOFF §1：0.45 出品人 + 0.60 后期总监
    (2.21, 3.85, '编剧 / 导演', '苏打', 420, 300, '苏打'),
    (4.06, 5.80, '美术设计', 'Cait', 1640, 250, None),
    (4.21, 5.80, '音乐', '西几王', 1440, 250, '西几王'),
    (5.97, 7.30, 'AI画师', '沉溺　雪晴　索恩', 640, 250, None),
]


def credits(fr, t):
    for (a, b, role, names, x, y, seal) in CREDITS:
        a_, b_ = V(a), V(b)
        if not (a_ - 0.01 <= t < b_ + 0.35): continue
        p = clamp01((t - a_) / 0.55)                     # 第二棒：0.55s 显全（A01 1.2s 时两条署名都已完整）
        out = clamp01((t - b_) / 0.3)
        if names.isascii():                                   # 第四棒：拉丁名横排（不再旋转竖排）——职务竖排，人名横排在职务下方
            g = T.credit_block(fr, role, '', x, y, p, role_size=30, name_size=62, seal_text=None, seed=len(role))
            L = T.Layer(); f = T._font('song', 54)
            from PIL import Image as _I, ImageDraw as _D
            w_ = int(f.getlength(names)) + 10; im = _I.new('L', (w_, 70), 0); _D.Draw(im).text((w_ / 2, 35), names, font=f, fill=255, anchor='mm')
            L.paste(np.asarray(im).astype(np.float32) / 255, x + 30, y + 30 * 4 * 1.08 + 70, (0.95, 0.94, 0.92), eout(clamp01(p * 1.6 - 0.4)))
            g = T.compose(g, L, shadow=0.7, glow=0.1)
        else:
            g = T.credit_block(fr, role, names, x, y, p, role_size=30, name_size=62, seal_text=seal, seed=len(role))
        fr = fr + (g - fr) * (1 - out)
    return fr


LY_TXT = {0: '长夜压在我刀锋之上', 1: '五百重甲拦不住一道寒光', 2: '谁在高处定对错罪状', 4: '我杀过恶徒也放过混账',
          6: '若正义由权势执掌', 7: '那我宁愿做乱世的锋芒'}


# ---------------- 屏幕层：主歌一歌词（第四棒：停用 LY 双列竖排）
# 每句一个主词 150–240px 与动作同帧落下（笔触书写 + 落字微震），余字 70–96px 同一基线横排、暗衬条托底；y 180–900 安全区；
# 主词为朱/金；镜内互动大字（五百重甲、寒光、罪状、称量、杀、放、刀身刻字、主张、正、锋芒）仍由镜头自己画。
# (行, [(字序, 字号, 颜色)], 中心 x, 基线 y, 退场秒, 点亮扫光)
_Wt, _R, _G = (0.95, 0.93, 0.88), (0.86, 0.12, 0.08), (1.0, 0.78, 0.40)
LY2 = [
    (0, [(0, 230, _Wt), (1, 230, _Wt)], 560, 470, 17.00, False),                                   # 「长夜」压在乌云下
    (0, [(k, 84, _R if k == 5 else _Wt) for k in range(2, 9)], 1300, 330, 17.92, False),             # 压在我刀锋之上
    (1, [(4, 190, _Wt), (5, 190, _Wt), (6, 190, _R), (7, 90, _Wt), (8, 90, _Wt)], 560, 820, 21.15, False),   # 拦不住·一道（V04–V05 左下暗区）
    (2, [(0, 190, _Wt), (1, 190, _Wt), (2, 190, _G), (3, 190, _G)], 960, 900, 23.44, False),        # 谁在高处（自殿檐压下）
    (2, [(4, 96, _Wt), (5, 96, _Wt), (6, 96, _Wt)], 1480, 330, 25.05, False),                       # 定对错（→ 罪状朱印）
    (4, [(k, 80, _R if k == 1 else _Wt) for k in range(0, 5)], 1360, 880, 30.74, False),            # 我杀过恶徒
    (4, [(k, 80, _Wt) for k in range(5, 10)], 640, 880, 32.55, False),                              # 也放过混账
    (6, [(0, 120, _Wt), (1, 220, _R), (2, 220, _R)], 560, 880, 38.08, False),                       # 若「正义」
    (6, [(3, 90, _Wt), (4, 165, _G), (5, 165, _G), (6, 90, _Wt), (7, 90, _Wt)], 960, 300, 39.85, False),  # 由「权势」执掌
    (7, [(0, 96, _Wt), (1, 96, _Wt), (2, 180, _G), (3, 180, _G), (4, 96, _Wt), (5, 180, _R), (6, 180, _R), (7, 96, _Wt)],
     900, 300, 42.66, True),                                                                        # 那我「宁愿」做「乱世」的（随刀光逐字点亮）
]


def lyrics(fr, t):
    for (li, chars, cx, by, t_out, sweep) in LY2:
        t_on = CT(li, chars[0][0]) - 0.3
        if not (t_on <= t < t_out + 0.4): continue
        out = clamp01((t - t_out) / 0.35)
        adv = [s * 1.04 for (_, s, _) in chars]
        x = cx - sum(adv) / 2
        L = T.Layer(); pmax = 0.0; big = max(s for (_, s, _) in chars)
        for (k, s, c), a in zip(chars, adv):
            p = clamp01((t - (CT(li, k) - 0.16)) / 0.26); pmax = max(pmax, p)
            if p > 0:
                cxx = x + a / 2; cyy = by - s / 2
                if s >= 150:
                    pop = 1 + 0.12 * math.exp(-max(0, t - CT(li, k)) / 0.09) * (t >= CT(li, k))
                    g = T.brush_glyph(LY_TXT[li][k], 'kai', s, p, seed=li * 13 + k)
                    if pop > 1.001: g = cv2.resize(g, None, fx=pop, fy=pop)
                    L.paste(g, cxx, cyy, c)
                else:
                    T.place(L, LY_TXT[li][k], cxx, cyy + 14 * (1 - eout(p)), s, c, 'kai', alpha=eout(p), blur=3 * (1 - p))
                if sweep:
                    a_ = t - CT(li, k)
                    if 0 <= a_ < 0.5: fr = A.glow(fr, (cxx, cyy), s * 0.9, 0.5 * math.exp(-a_ / 0.15), (1.0, 0.8, 0.5))
            x += a
        if pmax <= 0: continue
        if out > 0:
            nz = cv2.resize(A.ink_noise(li * 3 + 7, 256), (W, H))
            L.A *= np.clip((nz - out * 1.15) * 5 + 0.5, 0, 1)
        # 暗衬条：基线下沿到字顶的柔边暗带（只压字背后，让字在任何底上都可读）
        w_ = sum(adv) + 80; bx0 = int(cx - w_ / 2); bx1 = int(cx + w_ / 2)
        band = np.zeros((H // 4, W // 4), np.float32)
        cv2.rectangle(band, (bx0 // 4, int(by - big * 1.05) // 4), (bx1 // 4, int(by + 18) // 4), 1.0, -1)
        band = cv2.resize(cv2.GaussianBlur(band, (0, 0), 10), (W, H))[..., None]
        fr = fr * (1 - 0.55 * band * min(1, pmax * 2) * (1 - out))
        fr = T.compose(fr, L, shadow=0.75, shadow_sigma=14, glow=0.18)
    return fr


def a_zoom2(A_, B_, p, center=(960, 540), seed=0):
    """第四棒：推镜穿越（替换叠化版 a_zoom）。A 加速推近 + 径向模糊；B 从中心一个墨边圆窗里放大开出，窗沿冷光；
    中间帧 = 外圈 A 的推镜拖影 + 内圈清晰的 B，两镜元素都可读，无灰糊、无白场。"""
    e = TR.ease_in(min(1, p / 0.8), 2.2)
    A2 = TR._radial_blur(TR._zoom(A_, 1 + 0.8 * e, center), center, 0.10 * e)
    q = 1 - TR.ease_out(p, 3); B2 = TR._radial_blur(TR._zoom(B_, 1 + 0.35 * q, center), center, 0.05 * q)
    yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    d = np.hypot(xx * 2 - center[0], yy * 2 - center[1])
    nz = cv2.resize(A.ink_noise(seed + 31, 256), (W // 2, H // 2)) * 120
    R = 2300 * TR.ease_in(p, 1.8)
    m = cv2.resize(np.clip((R + nz - d) / 30, 0, 1), (W, H))[..., None]
    rim = cv2.resize(np.exp(-((R + nz - d) / 14) ** 2), (W, H))[..., None] * (p < 0.95)
    return A2 * (1 - m) + B2 * m + rim * np.array([0.5, 0.6, 0.85], np.float32) * 0.8


def a_cloud(A_, B_, p, seed=0):
    """第四棒：云层横扫（A01 云海 → A02）：一堵带体积的云墙自左向右推过，云墙前沿后露出 B；中间帧同时有 A、云墙、B。"""
    yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    nz = cv2.resize(A.tex(seed + 61, 256, 4, 5), (W // 2, H // 2))
    edge = (-0.35 + 1.7 * TR.ease_io(p)) * W
    d = (xx * 2 - edge) + (nz - 0.5) * 520 + 80 * np.sin(yy / 40)
    cloud = np.exp(-(d / 260) ** 2)
    m = cv2.resize(np.clip(-d / 120 + 0.5, 0, 1), (W, H))[..., None]
    c = cv2.resize(cloud * (0.6 + 0.6 * nz), (W, H))[..., None]
    out = A_ * (1 - m) + B_ * m
    return out * (1 - 0.8 * c) + c * np.array([0.14, 0.16, 0.21], np.float32)


def a_seam(A_, B_, p):
    """第四棒：城门合缝 → 门缝里开出 V01（A08 → V01）：前半 A（门正合拢）；后半门缝冷光一线从中缝向两侧拉开成 B。"""
    if p < 0.4: return A_
    q = TR.ease_in((p - 0.4) / 0.6, 1.6)
    xx = np.arange(W, dtype=np.float32)[None, :, None]
    half = 6 + 980 * q
    m = np.clip((half - np.abs(xx - 960)) / 6, 0, 1)
    rim = np.exp(-((np.abs(xx - 960) - half) / 5) ** 2)
    return A_ * (1 - m) + B_ * m + rim * np.array([0.7, 0.8, 1.0], np.float32) * (1.4 - q)


def a_zoom(A_, B_, p, center=(960, 540)):
    """第三棒：推镜穿越（无闪白版）。A 加速推近 + 径向模糊，B 从放大态减速回位；0.35–0.65 两画面按亮度叠化，
    中间帧始终同时带着前后镜的画面元素（不出现白场/灰雾）。"""
    e = TR.ease_in(min(1, p / 0.65), 2.4); A2 = TR._radial_blur(TR._zoom(A_, 1 + 0.7 * e, center), center, 0.09 * e)
    q = 1 - TR.ease_out(max(0, (p - 0.35) / 0.65), 3); B2 = TR._radial_blur(TR._zoom(B_, 1 + 0.4 * q, center), center, 0.08 * q)
    m = TR.ease_io((p - 0.35) / 0.3)
    return np.maximum(A2 * (1 - m), B2 * m) * 0.6 + (A2 * (1 - m) + B2 * m) * 0.4


class _Lazy:
    def __init__(self, cls): self.cls = cls; self.t0 = cls.t0; self.t1 = cls.t1; self.obj = None; self.id = cls.__name__
    def render(self, t):
        if self.obj is None: self.obj = self.cls()
        return self.obj.render(t)


def build_cam():
    cam = V3.VCam3(HMP, energy=energy)
    for S_ in SHOTS:
        cam.shot(S_.t0, S_.t1, MOVES3.get(S_.__name__))
        c = CUTS.get(S_.__name__)
        if c and c[0] == 'whip': cam.whip(S_.t0, c[3]['dir'], c[3].get('L', 400))
        else: cam.cut(S_.t0)
    return cam


class Section:
    def __init__(self):
        self.shots = [_Lazy(S_) for S_ in SHOTS]
        self.ids = [S_.__name__ for S_ in SHOTS]
        self.cam = build_cam()
        self.fg = VC.FGLayer(17, n=24)

    def idx(self, t):
        for i, s in enumerate(self.shots):
            if t < s.t1 - 1e-6: return i
        return len(self.shots) - 1

    def raw(self, t):
        i = self.idx(t); s = self.shots[i]
        for j in (i, i + 1):
            if j <= 0 or j >= len(self.shots): continue
            b = self.shots[j]; name, pre, post, kw = CUTS.get(self.ids[j], ('hard', 0, 0, {}))
            c = b.t0
            if name in ('hard', 'whip') or not (c - pre - 1e-6 <= t < c + post - 1e-6): continue
            a = self.shots[j - 1]
            B, ca2 = b.render(t)
            Af, ca1 = a.render(t) if (pre > 0 or t < c) else (B, 0)
            p = (t - (c - pre)) / (pre + post)
            if pre == 0: p = 0.5 + 0.5 * p
            Af = np.clip(Af, 0, 4).astype(np.float32); B = np.clip(B, 0, 4).astype(np.float32)
            if name == 'a_zoom': return a_zoom(Af, B, p, **kw), max(ca1, ca2)
            if name == 'a_zoom2': return a_zoom2(Af, B, p, **kw), max(ca1, ca2)
            if name == 'a_cloud': return a_cloud(Af, B, p, **kw), max(ca1, ca2)
            if name == 'a_seam': return a_seam(Af, B, p), max(ca1, ca2)
            return TR.transition(name, Af, B, p, **kw), max(ca1, ca2)
        fr, ca = s.render(t)
        name = CUTS.get(self.ids[i], ('hard',))[0]
        if name == 'hard' and i > 0:
            k = int((t - s.t0) * FPS + 1e-6)
            if k < 2:
                a_ = 1.0 if k == 0 else 0.45
                fr = K.cam(fr, 1 + 0.03 * a_) * (1 + 0.22 * a_) + 0.05 * a_
                ca = max(ca, 3 if k == 0 else 1)
        return fr, ca

    def frame(self, f):
        t = f / FPS
        fr, ca = self.raw(t)
        sid = self.ids[self.idx(t)]
        fr, st = self.cam.apply(np.clip(fr, 0, 4).astype(np.float32), t)
        kind, amt = FG.get(sid, (None, 0))
        if kind: fr = self.fg.draw(fr, t, st, kind, amt)
        fr = fr * (EXPO_ALL * EXPO.get(sid, 1.0))
        fr = fr + 0.012 * (1 - np.clip(fr, 0, 1))
        if TITLE_HIT <= t < V(13.35) + 0.5:                                     # 第四棒：片名 = 屏幕层（A06 → A07a → A07b 不动，13.35 化灰）
            k = t - TITLE_HIT
            fr = TITLE.draw(fr, t, out=clamp01((t - V(13.35)) / 0.45), sheen=clamp01((t - V(11.52)) / 0.7) if t >= V(11.52) else None,
                            glow=0.5 * math.exp(-max(0, k) / 0.3))
        if t < 7.9: fr = credits(fr, t)
        fr = lyrics(fr, t)
        ca = max(ca, int(round(4 * min(1.0, st['punch']))))
        out = K.finish(fr, t, ca=ca, vig=0.45, red_keep=1.0)
        return K.to8(out)


def frame(f, cache):
    if 'sec' not in cache: cache['sec'] = Section()
    return cache['sec'].frame(int(f))


if __name__ == '__main__':
    # 自测：python3 units/sec_a_intro_verse1.py <秒> [...] → /tmp/a_<秒>.png
    from PIL import Image
    c = {}
    for a in sys.argv[1:]:
        f = int(round(float(a) * FPS))
        Image.fromarray(frame(f, c)).save('/tmp/a_%s.png' % a); print('saved', a, flush=True)
