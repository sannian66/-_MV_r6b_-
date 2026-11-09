"""units/sec_d_bridge.py — [D组] Bridge「风暴之眼」（帧 4839–5797，161.31–193.23s）第四轮重做。
设计说明见 crew/round4/D_storyboard.md。30 镜，平均 1.06s；每镜一个类（render(t) -> (float 帧, paper_mode)），
切点转场由 CUTS 表驱动；收尾 d_kit.d_finish（白纸段保留暖白）。
叙事：朱墨入水 → 白纸审判庭（宋体字 = 权力的话，从天而降、堆成雪、立成碑、盖成印）→ 剪纸里的被牺牲者被火烧穿，
白纸世界烧出黑洞 → 余烬中睁眼、拔刀（字体切回手写 = 他的话）→ 撞碎字墙 →「天」在风暴之眼里 → 刀光在天上刻「判」→
子弹时间 → 雨滴炸成金 →「答」在裂开的云里金色爆开，白金出画接副歌终。"""
import sys, math, os
ROOT = '/workspace/work/mv'
for p in (ROOT, ROOT + '/lib', ROOT + '/units'):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, cv2
from PIL import Image
import c1kit as K
import typo as T
import trans as TR
import d_kit as D
from d_kit import V, cp, INK, CINNABAR, GOLD, RED, W, H, BAR

READY = False
F0, F1 = 4839, 5797
T0, T1 = F0 / 30, F1 / 30
O = {i: D.onsets(i) for i in range(40, 48)}
WHITE = (0.97, 0.94, 0.88)
EMBER = np.array([1.0, 0.5, 0.16], np.float32)
COLD = (0.62, 0.74, 1.0)
GOLDR = (1.0, 0.78, 0.45)


def eo(x, k=3): return K.eout(x, k)
def ei(x, k=3): return K.ein(x, k)
def cl(x): return K.clamp01(x)
def lerp(a, b, u): return a + (b - a) * u


def impact(t, t_hit, decay=0.12):
    return 0.0 if t < t_hit else math.exp(-(t - t_hit) / decay)


def shake_cam(fr, t, hits, amp=10, z=1.0, ox=0.0, oy=0.0, rot=0.0, center=None):
    a = sum(amp * impact(t, h) for h in hits)
    dx, dy, dr = K.shake(t, a, 14, int(t * 3) % 7)
    return K.cam(fr, z * (1 + 0.012 * a / max(1, amp)), ox + dx, oy + dy, rot + dr, center)


def vcol(fr, text, x, y0, size, onsets_, t, color='ink', emph=None, fname='serif_sb', adv=1.2, ink=True, horiz=False, dur=0.2):
    prog = [cp(t, o, dur) for o in onsets_]
    L = D.text_layer(D.column(text, x, y0, size, prog, adv, color, emph, fname, drop=size * 0.4, horiz=horiz), fname=fname)
    return D.ink_text(fr, L, bleed=0.18) if ink else D.light_text(fr, L)


class Shot:
    t0 = t1 = 0.0
    paper = 0.0
    def u(self, t): return cl((t - self.t0) / (self.t1 - self.t0))


# ===================================================================================== 第一幕：白纸审判庭
class S01_Drop(Shot):
    """ECU 朱墨滴入黑水：两滴（落拍 161.77 / 162.21），涟漪 + 水下朱墨云翻涌。黑场接 C 组。"""
    t0, t1 = 161.31, 162.58
    def __init__(self):
        self.c = np.array([960., 560.])
        x, y = K.grid(); self.x, self.y = x, y
        self.base = (0.012 + 0.03 * cv2.resize(K.fbm(H // 8, W // 8, 93, base=3, octaves=4), (W, H)))[..., None] * np.array([0.8, 0.9, 1.1], np.float32)
        self.tex = D._tile(91)
        self.hits = [(V(161.77), (960, 560), 1.0), (V(162.21), (1010, 585), 0.6)]

    def render(self, t):
        u = self.u(t)
        x, y = self.x, self.y
        fr = self.base.copy()
        h = np.zeros((H, W), np.float32)
        for th, c, s in self.hits:
            age = t - th
            if age <= 0: continue
            r = np.sqrt((x - c[0]) ** 2 + ((y - c[1]) * 2.4) ** 2)
            for k in range(3):
                R = (160 + 120 * k) * age ** 0.8 * 2.2
                h += s * np.sin((r - R) / 9.0) * np.exp(-((r - R) / (28 + 30 * age)) ** 2) * math.exp(-age * 1.6) * (1 - 0.25 * k)
        gy, gx = np.gradient(h)
        spec = np.clip(-gx * 0.6 - gy * 1.4, 0, None) ** 1.5 * 6
        fr = fr + spec[..., None] * np.array([0.8, 0.85, 0.95], np.float32) * 0.7
        # 水下朱墨云
        for th, c, s in self.hits:
            age = t - th
            if age <= 0: continue
            Rr = (40 + 330 * eo(age / 1.2, 2)) * s
            n = cv2.remap(self.tex, ((x / 3 + t * 20) % 512).astype(np.float32), ((y / 3 * 1.0 + age * 40) % 512).astype(np.float32), cv2.INTER_LINEAR)
            d = np.sqrt((x - c[0]) ** 2 + ((y - c[1]) * 2.0) ** 2) / max(1, Rr)
            m = np.clip((1.15 - d - (n - 0.5) * 1.2) * 2.2, 0, 1) * min(1, age * 6) * (0.8 + 0.2 * s)
            fr = fr * (1 - m[..., None] * 0.6) + m[..., None] * np.array([0.42, 0.03, 0.02], np.float32) * (0.6 + 0.6 * n[..., None])
        # 下落的墨滴（落前）
        for th, c, s in self.hits:
            if th - 0.28 <= t < th:
                q = (t - (th - 0.28)) / 0.28
                yy = 120 + (c[1] - 120) * q ** 1.8
                L = np.zeros((H, W), np.float32)
                cv2.ellipse(L, (int(c[0]), int(yy)), (int(9 * s + 4), int(16 * s + 6)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
                L = cv2.GaussianBlur(L, (0, 0), 1.2)
                fr = fr * (1 - L[..., None]) + L[..., None] * np.array([0.65, 0.05, 0.03], np.float32)
                hl = np.zeros((H, W), np.float32); cv2.circle(hl, (int(c[0] - 3), int(yy - 6)), 3, 1.0, -1, cv2.LINE_AA)
                fr += cv2.GaussianBlur(hl, (0, 0), 1.5)[..., None] * 1.5
        # 冠状水花
        for th, c, s in self.hits:
            age = t - th
            if 0 <= age < 0.35:
                fr = K.sparks(fr, t, 300 + int(s * 10), c, n=int(40 * s), life=0.35, speed=(200, 600), dirc=(0, -1), spread=0.9,
                              grav=2200, burst=True, t0=th, color=(0.75, 0.15, 0.1), amt=0.9)
        fr = fr * min(1, (t - self.t0) / 0.18 + 0.0)
        return K.cam(fr, 1.0 + 0.05 * eo(u, 2), center=(960, 560)), 0.0


class S02_Steles(Shot):
    """ELS 白纸虚空：小小的水墨鬼屠背影前行；宋体巨字「什」「么」「叫」从天砸下立成碑，墨点飞溅"""
    t0, t1 = 162.58, 164.05
    paper = 1.0
    CH = [('什', 520, 560, 430, O[40][0]), ('么', 1420, 520, 400, O[40][1]), ('叫', 960, 330, 300, O[40][2])]
    def __init__(self):
        self.sp = D.spr('walk_back')
        for c in self.CH: T.glyph(c[0], 'serif_black', c[3], False)

    def render(self, t):
        u = self.u(t)
        fr = D.paper(11).copy()
        fr = D.wash(fr, 0.10, 0.0, 0.5, seed=21)
        fr = D.fall_chars(fr, t, '什么叫大局牺牲忍让规矩正义', 31, density=7, sizes=(18, 46), alpha=0.55, near_blur=3)
        hits = []
        for ch, x, y, s, on in self.CH:
            th = V(on)
            if t < th - 0.24: continue
            q = cl((t - (th - 0.24)) / 0.24)
            hits.append(th)
            yy = lerp(y - 900, y, ei(q, 2))
            sq = impact(t, th, 0.07)
            # 落地阴影
            if q > 0.2:
                sh = D.text_layer([(ch, x + 30, y + s * 0.46, s, 0.35 * q, 1.0, 0.0)], fname='serif_black')
                sh.A = cv2.GaussianBlur(cv2.resize(cv2.resize(sh.A, (W, H // 6)), (W, H)), (0, 0), 9) * 0.0 + cv2.GaussianBlur(sh.A, (0, 0), 18) * 0.5
                fr = fr * (1 - sh.A[..., None] * 0.4)
            L = D.text_layer([(ch, x, yy, s, min(1, q * 2), 1.0 + 0.0, 0.0)], fname='serif_black')
            if q < 1: L.A = cv2.GaussianBlur(L.A, (0, 0), 0.1 + 6 * (1 - q))
            if sq > 0.01:
                L = D.text_layer([(ch, x, yy + s * 0.05 * sq, s, 1.0, 1.0, 0.0)], fname='serif_black')
            fr = D.ink_text(fr, L, bleed=0.3, sig=7)
            fr = D.ink_spray(fr, t, th, (x, y + s * 0.42), n=46, seed=ord(ch), power=420)
        # 鬼屠：远离镜头慢走
        z = 300 - 18 * u
        fr, _ = D.ink_figure(fr, self.sp, t, D.feet(self.sp), (960, 830 - 14 * u + 3 * math.sin(t * 7.3)), D.fit_scale(self.sp, z), shadow=0.55)
        fr = shake_cam(fr, t, hits, 7, z=1.0 + 0.05 * eo(u, 2), oy=-30 * (1 - eo(u, 2)))
        return fr, 1.0


class S03_TypeSlam(Shot):
    """仰拍地平：「大」「局」两枚宋体铅字块（带厚度）砸进纸面，墨尘翻起"""
    t0, t1 = 164.05, 164.98
    paper = 1.0
    CH = [('大', 700, 560, 520, O[40][3]), ('局', 1290, 620, 540, O[40][4])]

    def render(self, t):
        u = self.u(t)
        fr = D.paper(12).copy()
        g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        fr = fr * (1 - 0.10 * K.sstep(0.55, 0.62, g) * (1 - K.sstep(0.62, 0.9, g)))  # 地平线淡墨
        fr = D.wash(fr, 0.16, 0.6, 1.0, seed=33)
        fr = D.fall_chars(fr, t, '什么叫大局', 32, density=5, sizes=(16, 40), alpha=0.5, near_blur=2)
        hits = []
        for ch, x, y, s, on in self.CH:
            th = V(on)
            if t < th - 0.2: continue
            q = cl((t - (th - 0.2)) / 0.2); hits.append(th)
            yy = lerp(y - 1000, y, ei(q, 2))
            sc = 1.25 - 0.25 * q
            # 厚度（铅字块侧面）
            depth = [(ch, x + 2.2 * k, yy - 1.4 * k, s, 1.0, sc, 0.0, (0.28 + 0.01 * k,) * 3) for k in range(16, 0, -1)]
            Ld = D.text_layer(depth, fname='serif_black')
            # 地面投影
            if q > 0.3:
                Ls = D.text_layer([(ch, x + 60, y + s * 0.5, s, 1.0, 1.0)], fname='serif_black')
                A = Ls.A; M = np.float32([[1, -1.2, 0], [0, 0.22, 0]]); M[1, 2] = (y + s * 0.5) * (1 - 0.22); M[0, 2] = 1.2 * (y + s * 0.5)
                A = cv2.GaussianBlur(cv2.warpAffine(A, M, (W, H)), (0, 0), 10) * 0.45 * q
                fr = fr * (1 - A[..., None])
            fr = D.ink_text(fr, Ld, bleed=0.0)
            L = D.text_layer([(ch, x, yy, s, 1.0, sc, 0.0)], fname='serif_black')
            fr = D.ink_text(fr, L, bleed=0.25)
            fr = D.ink_spray(fr, t, th, (x, y + s * 0.45), n=60, seed=ord(ch) + 3, power=700)
            # 墨尘
            a = cl((t - th) / 0.5) * (1 - cl((t - th - 0.3) / 0.6))
            if a > 0:
                fr = K.fog(fr, t, 40 + ord(ch) % 9, amt=0.55 * a, speed=(80, -30), y0=(y + s * 0.2) / H, y1=(y + s * 0.6) / H, color=(0.2, 0.19, 0.18))
        fr = shake_cam(fr, t, hits, 12, z=1.02 + 0.04 * u, oy=10 * u)
        return fr, 1.0


class S04_Snow(Shot):
    """贴地 MS：字落如雪，地上堆起宋体字堆；鬼屠的靴子和拖地的刀尖走进字堆；朱红「牺牲」落在镜头前"""
    t0, t1 = 164.98, 165.88
    paper = 1.0
    def __init__(self):
        self.sp = D.spr('walk_back')
        r = np.random.default_rng(44)
        pool = '什么叫大局牺牲忍让理所应当规矩正义'
        items = []
        for i in range(520):
            yy = r.uniform(0, 1) ** 0.6
            y = 760 + yy * 220; s = 12 + yy * 58
            items.append((pool[i % len(pool)], r.uniform(-40, W + 40), y, s, 0.85, 1.0, r.uniform(-60, 60)))
        items.sort(key=lambda c: c[2])
        self.pile = D.text_layer(items, fname='serif_m')
        near = [c for c in items if c[2] > 930]
        self.near = D.text_layer(near, fname='serif_m'); self.near.A = cv2.GaussianBlur(self.near.A, (0, 0), 4)

    def render(self, t):
        u = self.u(t)
        fr = D.paper(13).copy()
        fr = D.wash(fr, 0.08, 0.0, 0.4, seed=8)
        fr = D.fall_chars(fr, t, '什么叫大局牺牲忍让', 41, density=9, sizes=(20, 60), alpha=0.6, near_blur=5)
        fr = D.ink_text(fr, self.pile, bleed=0.08, dens=0.75)
        fr, _ = D.ink_figure(fr, self.sp, t, D.feet(self.sp), (1060 - 40 * u, 950 + 6 * math.sin(t * 7)), D.fit_scale(self.sp, 1450), wind=8)
        fr = D.ink_text(fr, self.near, bleed=0.0, dens=0.8)
        # 朱红「牺」「牲」近景坠落
        for ch, x, on in (('牺', 560, O[40][8]), ('牲', 1420, O[40][9])):
            th = V(on)
            if t < th - 0.35: continue
            q = (t - (th - 0.35)) / 0.35
            y = lerp(-100, 560, ei(min(q, 1), 1.6)) + max(0, t - th) * 260
            L = D.text_layer([(ch, x, y, 300, 1.0, 1.0, -10 + 25 * q, CINNABAR)], fname='serif_black')
            L.A = cv2.GaussianBlur(L.A, (0, 0), 0.5 + 2.0 * max(0, 1 - q))
            fr = D.ink_text(fr, L, bleed=0.2, sig=8)
        fr = vcol(fr, '什么叫牺牲', 1752, 262, 58, O[40][5:10], t, 'ink', {'牺': CINNABAR, '牲': CINNABAR})
        return K.cam(fr, 1.04, -30 + 60 * u, 0), 1.0


class S05_Ren(Shot):
    """ECU 纸面：「什么叫」墨洇显出 →「忍」宋体巨字踩拍砸下（先大后定、墨洇开），「刃」上那一点变朱墨、
    凝成一滴沿纸滑落到左下（→ 下一镜秤盘左盘那枚朱红「人」，圆形匹配剪辑）"""
    t0, t1 = 165.88, 167.25
    paper = 1.0
    REN = (960, 540, 760)
    DROP = (680, 600)

    def __init__(self):
        x, y, s = self.REN
        self.pap = cv2.resize(D.paper(14)[300:780, 540:1380], (W, H))
        a = T.ink_bleed_glyph('忍', 'serif_black', s, 1.0, seed=5); G = a.shape[0]
        n, lab, st, cen = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8))
        # 「刃」的那一点：左上方的小连通域
        cand = [i for i in range(1, n) if st[i, 4] < st[1:, 4].max() * 0.4 and cen[i][1] < G / 2 and cen[i][0] < G / 2]
        k = cand[0] if cand else 2
        dm = cv2.dilate((lab == k).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        L = T.Layer(); L.paste(cv2.GaussianBlur(dm, (0, 0), 1.5), x, y, CINNABAR); self.dotA = L.A
        self.dot = (int(x - G / 2 + cen[k][0]), int(y - G / 2 + cen[k][1]))

    def render(self, t):
        u = self.u(t)
        fr = self.pap.copy()
        tr = V(O[40][13]); tin = tr - 0.1
        x, y, s = self.REN
        if t < tr:
            Lq = T.Layer()
            for k, (ch, on) in enumerate(zip('什么叫', O[40][10:13])):
                p = cl((t - V(on) + 0.12) / 0.4)
                if p > 0:
                    Lq.paste(T.ink_bleed_glyph(ch, 'serif_black', 220, p, seed=k), 960, 300 + k * 250, INK)
            fr = D.ink_text(fr, Lq, bleed=0.1)
        else:
            fr = vcol(fr, '什么叫', 300, 330, 110, [self.t0] * 3, t, 'ink', fname='serif_black', adv=1.15)
        if t >= tin:
            q = cl((t - tin) / 0.1)
            p = cl((t - tin) / 0.32)
            a = T.ink_bleed_glyph('忍', 'serif_black', s, 0.15 + 0.85 * p, seed=5)
            sc = 1.0 + 0.6 * (1 - eo(q, 2))
            if sc > 1.01: a = cv2.resize(a, None, fx=sc, fy=sc)
            L = T.Layer(); L.paste(a, x, y, INK, alpha=0.4 + 0.6 * q)
            if q < 1: L.A = cv2.GaussianBlur(L.A, (0, 0), 0.5 + 8 * (1 - q))
            fr = D.ink_text(fr, L, bleed=0.25, sig=10)
            if t >= tr:
                fr = D.ink_spray(fr, t, tr, (x, y + 300), n=70, seed=55, power=900)
                # 刃上那一点：洇成朱墨
                c = cl((t - tr - 0.05) / 0.3)
                m = self.dotA * c
                fr = fr * (1 - m[..., None]) + m[..., None] * CINNABAR * 0.95
                if t > tr + 0.35:                      # 凝成一滴，滑落到秤盘位置
                    d = cl((t - tr - 0.35) / (self.t1 - tr - 0.38))
                    xx = lerp(self.dot[0], self.DROP[0], ei(d, 2)); yy = lerp(self.dot[1], self.DROP[1], ei(d, 2))
                    m2 = np.zeros((H, W), np.float32)
                    cv2.line(m2, self.dot, (int(xx), int(yy)), 0.55, 9, cv2.LINE_AA)
                    cv2.ellipse(m2, (int(xx), int(yy)), (17, int(20 + 10 * d)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
                    m2 = cv2.GaussianBlur(m2, (0, 0), 1.6)
                    fr = fr * (1 - m2[..., None]) + m2[..., None] * CINNABAR * 0.95
                    hl = np.zeros((H, W), np.float32); cv2.circle(hl, (int(xx - 5), int(yy - 8)), 4, 1.0, -1, cv2.LINE_AA)
                    fr = fr + cv2.GaussianBlur(hl, (0, 0), 2)[..., None] * 0.5
            p2 = cp(t, O[40][14], 0.2)
            if p2 > 0:
                L2 = D.text_layer([('让', 1640, 800, 170, p2, 1 + 0.3 * (1 - p2))], fname='serif_black')
                fr = D.ink_text(fr, L2)
        hit = [tr] if t >= tr else []
        fr = shake_cam(fr, t, hit, 14, z=1.0 + 0.05 * u, rot=1.2 * u)
        return fr, 1.0


def brush_mask(shape, p0, p1, w, seed, dry=0.5, pts=None, closed=False):
    """毛笔干湿笔触蒙版（墨的边缘不规则 + 顺笔方向的飞白丝）；pts 给折线/弧线时按折线画"""
    m = np.zeros(shape, np.float32)
    P = np.array(pts if pts is not None else [p0, p1], np.float32)
    # 笔压：起笔重、行笔略轻、收笔提
    for i in range(len(P) - 1):
        q = i / max(1, len(P) - 2)
        ww = w * (1.0 - 0.25 * math.sin(math.pi * q) + 0.15 * (q < 0.08))
        cv2.line(m, tuple(map(int, P[i])), tuple(map(int, P[i + 1])), 1.0, max(1, int(ww)), cv2.LINE_AA)
    if closed: cv2.line(m, tuple(map(int, P[-1])), tuple(map(int, P[0])), 1.0, int(w), cv2.LINE_AA)
    r = np.random.default_rng(seed)
    n = cv2.resize(r.random((shape[0] // 16 + 1, shape[1] // 16 + 1)).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(m, (0, 0), max(0.8, w * 0.18))
    m = np.clip((m - 0.35 - (n - 0.5) * 0.35) * 4, 0, 1)
    d = P[-1] - P[0]
    ang = math.degrees(math.atan2(d[1], d[0]))
    st = r.random((shape[0] // 2, shape[1] // 2)).astype(np.float32)
    k = np.zeros((31, 31), np.float32); cv2.line(k, (0, 15), (30, 15), 1.0, 1)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((15, 15), -ang, 1.0), (31, 31)); k /= k.sum() + 1e-6
    st = cv2.resize(cv2.filter2D(st, -1, k), (shape[1], shape[0]))
    return m * np.clip(1 - dry * np.clip((st - 0.52) * 6, 0, 1), 0, 1)


def arc_pts(c, ax, ay, a0, a1, n=40):
    return [(c[0] + ax * math.cos(math.radians(a)), c[1] + ay * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]


class S06_Scale(Shot):
    """LS 白纸上的毛笔天平（大写意：干湿笔触、飞白、墨池、淡墨投影）：左盘一枚朱红「人」，右盘堆起上千个墨「人」——
    竟然平衡；句子「什么叫以一人换千人」坐在横梁上逐字落下，「一人」朱红。镜头缓推 + 微仰。"""
    t0, t1 = 167.25, 168.62
    paper = 1.0
    PIV = (960, 300)
    LP, RP = (650, 640), (1270, 640)

    def __init__(self):
        sh = (H, W)
        post = brush_mask(sh, (962, 895), (958, 300), 36, 1, 0.65)
        foot = brush_mask(sh, None, None, 30, 2, 0.55, pts=[(770, 905), (860, 892), (960, 888), (1060, 892), (1150, 902)])
        self.post = np.maximum(post, foot)
        # 墨池：底座洇开的淡墨
        x, y = K.grid()
        nn = cv2.resize(D._tile(61)[:H // 4, :W // 4], (W, H))
        self.pool = np.clip(1 - np.sqrt(((x - 960) / 330) ** 2 + ((y - 905) / 44) ** 2) - (nn - 0.5) * 0.7, 0, 1) ** 0.7 * 0.5
        # 背后一轮淡墨晕（像月，也像秤砣的影）
        n2 = cv2.resize(D._tile(62)[:H // 4, :W // 4], (W, H))
        self.moon = np.clip((1 - np.sqrt(((x - 960) / 420) ** 2 + ((y - 470) / 400) ** 2)) * 3 - (n2 - 0.5) * 2.2, 0, 1) * 0.13
        beam = brush_mask(sh, None, None, 24, 3, 0.6, pts=[(560, 308), (760, 300), (960, 298), (1160, 300), (1370, 306)])
        cap = np.zeros(sh, np.float32); cv2.circle(cap, self.PIV, 22, 1.0, -1, cv2.LINE_AA)
        ch = np.zeros(sh, np.float32)
        for i, (cx, cy) in enumerate((self.LP, self.RP)):
            for dx in (-160, 0, 160):
                ch = np.maximum(ch, brush_mask(sh, (cx, 310), (cx + dx, cy - 4), 4, 10 + i * 3 + dx // 80, 0.35))
            bowl = brush_mask(sh, None, None, 15, 20 + i, 0.5, pts=arc_pts((cx, cy), 182, 48, 4, 176))
            lip = brush_mask(sh, None, None, 6, 30 + i, 0.6, pts=arc_pts((cx, cy), 182, 12, 186, 354))
            ch = np.maximum(ch, np.maximum(bowl, lip))
        self.beam = np.maximum(np.maximum(beam, cv2.GaussianBlur(cap, (0, 0), 1.5)), ch)
        r = np.random.default_rng(66)
        items = []
        for i in range(950):
            a = r.uniform(-1, 1); hgt = (1 - a * a) ** 0.75
            items.append(('人', self.RP[0] + a * 168, self.RP[1] - 8 - r.uniform(0, 1) ** 0.7 * hgt * 200, r.uniform(12, 28), 0.95, 1.0, r.uniform(-50, 50)))
        items.sort(key=lambda c: -c[2])
        self.heap = D.text_layer(items, fname='serif_b')
        self.one = D.text_layer([('人', self.LP[0], self.LP[1] - 50, 112, 1.0, 1.0, 0.0, CINNABAR)], fname='serif_black')
        T.warm('什么叫以一人换千人', 'serif_sb', 60, False)

    def _rot(self, L, M):
        L2 = T.Layer(); L2.A = cv2.warpAffine(L.A, M, (W, H)); L2.C = cv2.warpAffine(L.C, M, (W, H)); L2.bb = [0, 0, W, H]
        return L2

    def scene(self, t):
        fr = D.paper(15).copy() * D.fiber(5)[..., None]
        fr = D.wash(fr, 0.07, 0.0, 0.5, seed=9)
        fr = D.fall_chars(fr, t, '人', 61, density=5, sizes=(14, 34), alpha=0.45, near_blur=3)
        ang = 0.9 * math.sin(t * 2.1)
        M = cv2.getRotationMatrix2D(self.PIV, ang, 1.0)
        bm = cv2.warpAffine(self.beam, M, (W, H))
        # 淡墨投影（光从左上）
        shadow = cv2.GaussianBlur(np.maximum(self.post, bm), (0, 0), 9)
        shadow = np.roll(np.roll(shadow, 34, 1), 26, 0) * 0.14
        fr = fr * (1 - shadow[..., None])
        fr = fr * (1 - self.moon[..., None])
        fr = fr * (1 - self.pool[..., None] * 0.55) + self.pool[..., None] * INK * 0.55 * (fr / 0.93)
        fr = fr * (1 - self.post[..., None] * 0.93) + self.post[..., None] * INK * 0.93 * (fr / 0.93)
        fr = fr * (1 - bm[..., None] * 0.93) + bm[..., None] * INK * 0.93 * (fr / 0.93)
        fr = D.ink_text(fr, self._rot(self.heap, M), bleed=0.1)
        fr = D.ink_text(fr, self._rot(self.one, M), bleed=0.2, sig=6)
        fr = D.ink_spray(fr, 200.0, 199.0, (960, 890), n=50, seed=606, power=500, size=(2, 7))
        prog = [cp(t, o, 0.2) for o in O[41][0:9]]
        items = []
        for i, (ch, p) in enumerate(zip('什么叫以一人换千人', prog)):
            if p <= 0: continue
            xx = 610 + i * 87.5; yy = 240 - 40 * (1 - eo(p))
            c = CINNABAR if i in (4, 5) else 'ink'
            items.append((ch, xx, yy, 60, min(1, p * 1.5), 1 + 0.35 * (1 - eo(p)), 0.0, c))
        L = D.text_layer(items, fname='serif_sb')
        fr = D.ink_text(fr, self._rot(L, M), bleed=0.12)
        return fr

    def render(self, t):
        u = self.u(t)
        fr = self.scene(t)
        return K.cam(fr, 1.0 + 0.07 * eo(u, 2), 0, 30 * (1 - u), rot=-0.6 * u, center=(960, 600)), 1.0


class S07_Heap(Shot):
    """CU 右盘：千人字堆被不断加码（「人」从天倾倒），左盘朱红「人」虚焦在远处；竖排「换千人」"""
    t0, t1 = 168.62, 169.55
    paper = 1.0
    def __init__(self):
        self.base = None

    def render(self, t):
        u = self.u(t)
        if self.base is None: self.base = S06_Scale()
        fr = self.base.scene(t)
        fr = K.cam(fr, 2.3 + 0.25 * u, 1270 - 960 + 20 * u, 540 - 540, center=(960, 540))
        x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        bl = cv2.GaussianBlur(fr, (0, 0), 9)
        m = (1 - K.sstep(0.18, 0.42, x))[..., None]
        fr = fr * (1 - m) + bl * m
        # 倾倒的「人」
        r = np.random.default_rng(71)
        L = T.Layer()
        for i in range(70):
            ph = r.random(); x0 = 1000 + r.normal(0, 220); s = r.uniform(30, 70)
            y = ((t - 168.0) * r.uniform(500, 900) + ph * 900) % 900 - 40
            T.place(L, '人', x0, y, s, 'ink', 'serif_b', alpha=0.9, rot=r.uniform(-60, 60) + t * 40, texture=False)
        fr = D.ink_text(fr, L, bleed=0.08)
        fr = vcol(fr, '换千人', 1752, 300, 96, O[41][6:9], t, 'ink', fname='serif_black')
        return shake_cam(fr, t, [V(O[41][7])], 5), 1.0


class S08_Seals(Shot):
    """俯拍：「理」「所」「应」四方朱印一枚枚盖下，把那枚朱红「人」盖没；「当」最大最狠一印"""
    t0, t1 = 169.55, 170.92
    paper = 1.0
    SEALS = [('理', 760, 400, 300, -7), ('所', 1170, 430, 300, 5), ('应', 790, 690, 300, 9), ('当', 960, 540, 560, -3)]

    def render(self, t):
        u = self.u(t)
        fr = D.paper(16).copy()
        fr = D.wash(fr, 0.06, 0.0, 1.0, seed=17)
        L = D.text_layer([('人', 960, 540, 120, 1.0, 1.0, 0.0, CINNABAR)], fname='serif_black')
        fr = D.ink_text(fr, L)
        hits = []
        for k, ((ch, x, y, s, rot), on) in enumerate(zip(self.SEALS, O[41][9:13])):
            th = V(on)
            if t < th - 0.12: continue
            p = 0.3 * cl((t - (th - 0.12)) / 0.12) if t < th else 0.3 + (t - th) / 1.4
            hits.append(th)
            fr = T.stamp_seal(fr, ch, x, y, s, min(1, p), style='yin', rot=rot, seed=11 + k, color='red' if k == 3 else (0.62, 0.07, 0.05))
        fr = shake_cam(fr, t, hits, 9, z=1.0 + 0.06 * u, rot=-1.0 * u)
        return fr, 1.0


class S09_Brush(Shot):
    """ECU 太上皇的巨笔：笔锋落纸，宋体「正义」被他一笔一笔写出来（笔尖跟着笔序走）"""
    t0, t1 = 170.92, 171.83
    paper = 1.0
    CH = [('正', 720, 520, 520, 170.95, 171.42), ('义', 1260, 540, 520, 171.40, 171.83)]

    def __init__(self):
        import copy
        base = D.spr('tsh_brush')
        A = base.A; ys, xs = np.nonzero(A > 0.5)
        sel = xs > base.w * 0.5; j = np.argmax(ys * sel)
        self.tip = (xs[j] / base.w, ys[j] / base.h)
        # 只留「握笔的手 + 笔杆 + 笔锋」：沿笔杆的一条带（人和袍子不进 ECU）
        self.sp = copy.copy(base)
        yy, xx = np.mgrid[0:base.h, 0:base.w].astype(np.float32)
        p0 = np.array([0.40 * base.w, 0.22 * base.h]); p1 = np.array([xs[j], ys[j]], np.float32)
        d = p1 - p0; L2 = float(d @ d)
        tt = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L2, 0, 1)
        dist = np.sqrt((xx - p0[0] - tt * d[0]) ** 2 + (yy - p0[1] - tt * d[1]) ** 2)
        band = np.clip((0.07 * base.w - dist) / (0.02 * base.w), 0, 1)
        self.sp.A = (A * band).astype(np.float32); self.sp.ring = base.ring * band
        for ch, *_ in self.CH: T._brush_prep(ch, 'serif_black', 520, 0)

    def head(self, ch, x, y, s, p):
        B = T._brush_prep(ch, 'serif_black', s, 0)
        O_, a = B['O'], B['a']; G = a.shape[0]
        fy, fx = np.nonzero((np.abs(O_ - p) < 0.015) & (a > 0.5))
        if not len(fy): return None
        return (x - G / 2 + fx.mean(), y - G / 2 + fy.mean())

    def render(self, t):
        u = self.u(t)
        fr = D.paper(17).copy()
        L = T.Layer(); hd = None
        for ch, x, y, s, a0, a1 in self.CH:
            p = cl((t - a0) / (a1 - a0))
            if p <= 0: continue
            A = T.brush_glyph(ch, 'serif_black', s, p, 0, 0.15, 0.35)
            L.paste(A, x, y, INK)
            if p < 1: hd = self.head(ch, x, y, s, p) or hd
            else: hd = hd
        fr = D.ink_text(fr, L, bleed=0.3, sig=8)
        if hd is None: hd = (1500, 760)
        # 巨笔（笔尖压在笔序前沿）
        fr, _ = D.ink_figure(fr, self.sp, t, self.tip, (hd[0] + 14, hd[1] + 18), 1.5, rot=-6 + 4 * math.sin(t * 9), wind=0, k=1.4, dens=0.98)
        fr = vcol(fr, '若正义由你定义', 1752, 262, 58, O[42][0:7], t, 'ink')
        return K.cam(fr, 1.05 + 0.04 * u, (hd[0] - 960) * 0.05, (hd[1] - 540) * 0.05), 1.0


class S10_Giant(Shot):
    """仰拍 LS：太上皇水墨巨像握着巨笔压下来，天幕被他的墨染黑；身后刚写好的「正义」湿墨未干"""
    t0, t1 = 171.83, 172.56
    paper = 1.0
    def __init__(self):
        self.sp = D.spr('tsh_brush')
        self.zy = D.text_layer([('正', 420, 330, 330, 0.85, 1.0, 0.0), ('义', 420, 680, 330, 0.85, 1.0, 0.0)], fname='serif_black')

    def render(self, t):
        u = self.u(t)
        fr = D.paper(18).copy()
        fr = D.wash(fr, 0.55, -0.2, 0.55, seed=19)
        fr = D.ink_text(fr, self.zy, bleed=0.35, sig=12)
        fr = D.fall_chars(fr, t, '正义由你定义', 62, density=5, sizes=(16, 40), alpha=0.5, near_blur=3)
        fr, _ = D.ink_figure(fr, self.sp, t, D.feet(self.sp), (1150, 1500 - 40 * u), D.fit_scale(self.sp, 1500), wind=4, shadow=0)
        # 笔尖滴墨
        fr = D.ink_spray(fr, t, 171.9, (1330, 960), n=30, seed=5, power=300)
        fr = vcol(fr, '若正义由你定义', 1752, 262, 58, O[42][0:7], t, 'ink')
        return K.cam(fr, 1.0 + 0.05 * eo(u, 2), 0, -20 * u), 1.0


class S11_Answer(Shot):
    """反打 MCU 仰拍：鬼屠举刀向天（水墨），反问「谁来定义你的」第一次用手写体（行书）——字体 = 立场"""
    t0, t1 = 172.56, 173.66
    paper = 0.8
    def __init__(self):
        self.sp = D.spr('S099/point_up')

    def render(self, t):
        u = self.u(t)
        fr = D.paper(19).copy()
        fr = D.wash(fr, 0.62, -0.1, 0.75, seed=23)
        fr = D.fall_chars(fr, t, '谁来定义罪状', 63, density=5, sizes=(16, 44), alpha=0.5, near_blur=3)
        fr, _ = D.ink_figure(fr, self.sp, t, (0.5, 0.62), (1120, 640 - 20 * u), 0.78 + 0.05 * u, wind=6, k=1.35, dens=0.97)
        items = []
        for i, (ch, on) in enumerate(zip('谁来定义你的', O[42][7:13])):
            p = cp(t, on, 0.22)
            if p <= 0: continue
            items.append((ch, 330 + i * 18, 230 + i * 118 - 30 * (1 - eo(p)), 118, min(1, p * 1.5), 1 + 0.4 * (1 - eo(p)), -6.0))
        L = D.text_layer(items, fname='xing')
        fr = D.ink_text(fr, L, bleed=0.2)
        return K.cam(fr, 1.0 + 0.1 * eo(u, 2), 0, 30 * u, center=(1200, 300)), 0.8


class S12_Pierce(Shot):
    """ECU 一柄真刀（实拍质感，唯一不是墨的东西）从纸底刺上来，刀尖顶进朱红行书「罪状」；「状」落字那一刻
    一刀两断（字沿刀线裂开滑走），刀继续上冲出画（→ 下一镜纸被撕开）。"""
    t0, t1 = 173.66, 174.59
    paper = 0.8
    TIP = (0.498, 0.03)
    def __init__(self):
        self.sp = D.spr('blade_up_cu')
        self.P = [dict(ch='罪', x=790, y=380, size=330, rot=0.0, font='xing', idx=0, emph=False),
                  dict(ch='状', x=1140, y=400, size=330, rot=0.0, font='xing', idx=1, emph=False)]

    def render(self, t):
        u = self.u(t)
        fr = D.paper(20).copy()
        fr = D.wash(fr, 0.45, 0.2, 1.1, seed=29)
        tc = V(O[42][14])
        tipy = lerp(930, 520, eo(cl((t - self.t0) / (tc - self.t0)), 2)) - (520 * ei(cl((t - tc) / 0.35), 2) if t > tc else 0)
        # 刀在纸上的投影（暖光从左上）
        fr = D.fall_chars(fr, t, '罪状', 120, density=4, sizes=(16, 40), alpha=0.35, near_blur=3, emph={'罪': tuple(CINNABAR)})
        Fw, Aw = D.warp(self.sp, t, self.TIP, (975 + 40, tipy + 30), 1.75, rot=-3, wind=0)
        fr = fr * (1 - cv2.GaussianBlur(Aw, (0, 0), 14)[..., None] * 0.35)
        fr, Aw = D.real_figure(fr, self.sp, t, self.TIP, (975, tipy), 1.75, rot=-3, expo=0.82, sat=0.55, tint=(1.04, 0.98, 0.9),
                               rim=(1.0, 0.92, 0.8), rim_amt=2.5, rim_dir=(-0.7, -0.7), wind=0, breathe=0.0,
                               mblur=(0, 10 * impact(t, self.t0 + 0.02, 0.2) + (14 if t > tc else 0)))
        # 刃口一线寒光沿刃上行
        Fw, A2 = D.warp(self.sp, t, self.TIP, (975, tipy), 1.75, rot=-3, wind=0)
        lum = Fw.mean(-1)
        fr = D.metal_sheen(fr, A2 * ((Fw.max(-1) - Fw.min(-1)) < 0.12), lum, t, (975, 0), 90, speed=lerp(1100, 200, u), width=50, amt=0.7)
        if t < tc:
            items = []
            for p_, on in zip(self.P, O[42][13:15]):
                p = cp(t, on, 0.16)
                if p > 0: items.append((p_['ch'], p_['x'], p_['y'], 330, min(1, p * 1.5), 1 + 0.35 * (1 - eo(p)), 0.0, CINNABAR))
            L = D.text_layer(items, fname='xing')
            fr = D.ink_text(fr, L, bleed=0.3, sig=8)
        else:
            q = cl((t - tc) / 0.45)
            fr = T.slash_cut(fr, '罪状', 0.02 + 0.98 * q, places=self.P, fname='xing', color=tuple(CINNABAR), angle=-87, cut_at=0.02,
                             offset=-10, slide=0.35, appear=0.0, style_kw=dict(shadow=0.0, outline=0, glow=0.35, glow_color=(0.85, 0.12, 0.05)))
            fr = D.ink_spray(fr, t, tc, (975, 400), n=60, seed=121, power=800, col=CINNABAR)
        return shake_cam(fr, t, [tc], 12, z=1.0 + 0.04 * u), 0.8


# ===================================================================================== 第二幕：被牺牲者 / 白纸烧穿
def papercut(sp, seed=0):
    """把同伴群像转成朱红剪纸：亮处镂空、轮廓线刻空"""
    F, A = sp.F, sp.A
    lum = (F * np.array([0.3, 0.59, 0.11], np.float32)).sum(-1)
    inside = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    e = cv2.Canny((np.clip(lum, 0, 1) * 255).astype(np.uint8), 40, 110)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    holes = np.clip((lum - 0.55) * 6, 0, 1) * (inside > 10)
    holes = np.maximum(holes, e * (inside > 8) * 0.95)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    Ac = np.clip(A * 1.3 - 0.15, 0, 1) * (1 - holes)
    return cv2.GaussianBlur(Ac.astype(np.float32), (0, 0), 0.7)


class Companions:
    _A = None
    @classmethod
    def mask(cls):
        if cls._A is None:
            sp = D.spr('companions'); cls._A = (papercut(sp), sp)
        return cls._A


def place_mask(A, sp_w, sp_h, M):
    return cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR)


def burn(A_shape_y, t, t_start, speed, seed, x=None, y=None, base=H - BAR):
    """自下而上烧：返回 (intact, glow, char)"""
    x, y = K.grid() if x is None else (x, y)
    n = cv2.resize(D._tile(seed)[:H // 4, :W // 4], (W, H))
    yf = base - (t - t_start) * speed
    f = (yf - y) + (n - 0.5) * 160
    intact = np.clip(f / 6, 0, 1)
    glow = np.exp(-(f / 16) ** 2) * (t > t_start)
    char = np.clip(1 - np.abs(f + 30) / 40, 0, 1) * (f < 0)
    return intact, glow, char


class S13_PaperCut(Shot):
    """LS 剪纸追忆：师兄、师姐、殷若、徒弟四个朱红剪纸剪影立在背光宣纸上；马善政楷竖排「谁又来为 / 被牺牲的」；脚下起火"""
    t0, t1 = 174.59, 175.96
    paper = 0.85
    def render(self, t):
        u = self.u(t)
        Ac, sp = Companions.mask()
        x0, y0, x1, y1 = sp.bbox
        s = 700 / ((y1 - y0) * sp.h)
        cx = (x0 + x1) / 2 * sp.w; by = y1 * sp.h
        ox = -40 + 80 * u
        M = np.float32([[s, 0, 960 + ox - cx * s], [0, s, 905 - by * s]])
        A = cv2.warpAffine(Ac, M, (W, H))
        bg = D.paper(21) * 0.92
        x, y = K.grid()
        glow = np.exp(-(((x - 960) / 900) ** 2 + ((y - 520) / 420) ** 2))
        bg = bg * (0.72 + 0.4 * glow[..., None]) * np.array([1.0, 0.93, 0.8], np.float32)
        sh = cv2.GaussianBlur(np.roll(np.roll(A, 14, 1), 16, 0), (0, 0), 7)
        fr = bg * (1 - sh[..., None] * 0.35)
        red = CINNABAR * (0.85 + 0.25 * D.paper(22)[..., :1] / 0.93)
        intact, gl, ch = burn(None, t, 175.5, 120, 77, x, y)
        Ab = A * intact
        fr = fr * (1 - Ab[..., None]) + red * Ab[..., None]
        fr = fr + (gl * A)[..., None] * EMBER * 2.2 + gl[..., None] * EMBER * 0.25 * cl((t - 175.5) / 0.3)
        fr = fr * (1 - (ch * 0.6)[..., None])
        if t > 175.5: fr = D.embers(fr, t, 13, n=40, area=(0, 700, W, H - BAR), amt=0.8)
        for (txt, xx, ons) in (('谁又来为', 250 + ox, O[43][0:4]), ('被牺牲的', 1680 + ox, O[43][4:8])):
            fr = vcol(fr, txt, xx, 250, 72, ons, t, 'ink', {'牺': CINNABAR, '牲': CINNABAR}, fname='kai', adv=1.12)
        return K.cam(fr, 1.0 + 0.03 * u), 0.85


class S14_Burn(Shot):
    """CU 剪纸殷若被火从脚下烧上来：焦黑卷边、火线、余烬上飘；楷书朱红「牺牲」"""
    t0, t1 = 175.96, 176.87
    paper = 0.4
    def render(self, t):
        u = self.u(t)
        Ac, sp = Companions.mask()
        x0, y0, x1, y1 = sp.bbox
        fx = x0 + (x1 - x0) * 0.635; s = 1500 / ((y1 - y0) * sp.h)
        M = np.float32([[s, 0, 760 - fx * sp.w * s], [0, s, 140 + 60 * u - y0 * sp.h * s]])
        A = cv2.warpAffine(Ac, M, (W, H))
        x, y = K.grid()
        bg = D.paper(23) * (0.55 - 0.25 * u) * np.array([1.0, 0.85, 0.7], np.float32)
        fr = bg.copy()
        intact, gl, ch = burn(None, t, 175.85, 330, 78, x, y)
        Ab = A * intact
        red = CINNABAR * (0.85 + 0.25 * D.paper(24)[..., :1] / 0.93)
        fr = fr * (1 - Ab[..., None]) + red * Ab[..., None]
        fr = fr + (gl * np.maximum(A, 0.25))[..., None] * EMBER * 2.4
        fr = fr * (1 - ch[..., None] * 0.8)
        fr = D.embers(fr, t, 14, n=110, amt=1.1, rise=(80, 260))
        fr = vcol(fr, '牺牲', 1700, 360, 200, [O[43][5], O[43][6]], t, (0.95, 0.85, 0.7), fname='kai', adv=1.1, ink=False)
        return K.cam(fr, 1.0 + 0.06 * u, 0, -20 * u), 0.4


class S15_Hole(Shot):
    """LS 整张白纸世界从中心烧穿：黑洞里余烬中鬼屠单膝跪地；行书「陪葬」火烧显字后成灰飞散，镜头随灰上摇"""
    t0, t1 = 176.87, 178.26
    paper = 0.0
    def __init__(self):
        self.sp = D.spr('kneel_bow')

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        dark = 0.02 + 0.03 * cv2.resize(K.fbm(H // 8, W // 8, 151, base=3, octaves=4), (W, H))
        fr = np.stack([dark * 1.4, dark * 0.9, dark * 0.7], -1).astype(np.float32)
        fr = fr + np.exp(-(((x - 960) / 600) ** 2 + ((y - 980) / 260) ** 2))[..., None] * EMBER * 0.25
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (960, 935), D.fit_scale(self.sp, 430), expo=0.7, sat=0.7,
                              tint=(1.08, 0.88, 0.72), rim=(1.0, 0.55, 0.2), rim_amt=3.5, rim_dir=(0.0, 1.0), wind=4)
        fr = D.embers(fr, t, 15, n=120, amt=1.0, rise=(60, 200))
        # 纸：中心烧开
        R = 40 + 1500 * eo(cl((t - self.t0) / 1.1), 1.6)
        n = cv2.resize(D._tile(79)[:H // 4, :W // 4], (W, H))
        d = np.sqrt((x - 960) ** 2 + ((y - 560) * 1.4) ** 2) + (n - 0.5) * 260
        f = d - R
        intact = np.clip(f / 6, 0, 1); gl = np.exp(-(f / 18) ** 2); chr_ = np.clip(1 - np.abs(f - 25) / 35, 0, 1) * (f > 0)
        pap = D.paper(25) * np.array([1.0, 0.95, 0.86], np.float32)
        fr = fr * (1 - intact[..., None]) + pap * intact[..., None]
        fr = fr * (1 - chr_[..., None] * 0.85) + gl[..., None] * EMBER * 2.5
        p = cl((t - (V(O[43][8]) - 0.32)) / 1.25)
        if p > 0:
            fr = T.burn_in(fr, '陪葬', p, places=[dict(ch='陪', x=760, y=420, size=300, rot=0.0, font='xing', idx=0, emph=True),
                                                  dict(ch='葬', x=1160, y=430, size=300, rot=0.0, font='xing', idx=1, emph=True)],
                           fname='xing', color='red', mode='ash', split=0.55, seed=3, emph='陪葬', emph_color='red')
        if t > 176.96: fr = vcol(fr, '的', 1752, 262, 58, [O[43][7]], t, WHITE, ink=False)
        fr = fr + impact(t, V(O[43][8]), 0.2) * 0.25
        return K.cam(fr, 1.0 + 0.04 * u, 0, 40 - 90 * eo(u, 2)), 0.0


# ===================================================================================== 第三幕：觉醒 / 闯
def ember_bg(seed, warm=1.0):
    x, y = K.grid()
    d = 0.015 + 0.03 * cv2.resize(K.fbm(H // 8, W // 8, seed, base=3, octaves=4), (W, H))
    fr = np.stack([d * 1.3, d * 0.95, d * 0.8], -1).astype(np.float32)
    return fr + np.exp(-(((y - 1100) / 380) ** 2))[..., None] * EMBER * 0.18 * warm


class S16_Closed(Shot):
    """CU 余烬里低头闭目的侧脸：灰从上方落，火光从下方打上来；行书竖排「从今往后」在他视线方向"""
    t0, t1 = 178.26, 179.17
    def __init__(self):
        self.sp = D.spr('eyes_closed'); self.bg = ember_bg(161)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = D.embers(fr, t, 160, n=40, amt=0.5, size=(2, 6), blur=4)
        fr, _ = D.real_figure(fr, self.sp, t, (0.5, 0.5), (1120, 560), 1.02, expo=0.78, sat=0.65, tint=(1.06, 0.9, 0.78),
                              rim=(1.0, 0.55, 0.22), rim_amt=3.0, rim_dir=(0.2, 1.0), wind=4)
        fr = D.ash(fr, t, 16, n=70, amt=0.55, col=(0.6, 0.57, 0.53))
        # 一粒余烬飘向画面中心（→ 瞳孔 匹配）
        q = cl((t - self.t0) / (self.t1 - self.t0))
        ex, ey = lerp(380, 0.29 * W, eo(q, 2)), lerp(900, 0.555 * H, eo(q, 2))
        m = np.zeros((H, W), np.float32); cv2.circle(m, (int(ex), int(ey)), 5, 1.0, -1, cv2.LINE_AA)
        fr = fr + (cv2.GaussianBlur(m, (0, 0), 2) * 2 + cv2.GaussianBlur(m, (0, 0), 14) * 3)[..., None] * EMBER
        fr = vcol(fr, '从今往后', 300, 280, 112, O[44][0:4], t, WHITE, fname='xing', ink=False, adv=1.1)
        return K.cam(fr, 1.0 + 0.06 * u), 0.0


class S17_Eye(Shot):
    """ECU 睁眼：瞳孔里一点火光；「我的」行书落在画面右侧，「刀」字朱红巨字伴一道白刃寒光横扫"""
    t0, t1 = 179.17, 180.09
    def __init__(self):
        im = Image.open(ROOT + '/assets/gen/S102/eye_open_raw.png').convert('RGB').resize((W, H), Image.LANCZOS)
        F = np.asarray(im).astype(np.float32) / 255
        lum = F.mean(-1, keepdims=True)
        F = lum + (F - lum) * 0.75
        x, y = K.grid()
        light = 0.35 + 0.75 * np.exp(-(((x - 640) / 700) ** 2 + ((y - 600) / 420) ** 2))
        self.F = (F ** 1.2 * light[..., None] * np.array([1.05, 0.92, 0.82], np.float32)).astype(np.float32)
        self.pupil = (0.29 * W, 0.555 * H)

    def render(self, t):
        u = self.u(t)
        fr = self.F * (0.55 + 0.45 * eo(cl((t - self.t0) / 0.3)))
        x, y = K.grid()
        g = np.exp(-(((x - self.pupil[0]) ** 2 + (y - self.pupil[1]) ** 2) / (2 * 14 ** 2)))
        fr = fr + g[..., None] * EMBER * (1.2 + 0.5 * math.sin(t * 20))
        fr = D.ash(fr, t, 17, n=30, amt=0.4, size=(4, 12))
        tk = V(O[44][6])
        fr = vcol(fr, '我的', 1500, 300, 120, O[44][4:6], t, WHITE, fname='xing', ink=False)
        p = cp(t, O[44][6], 0.12)
        if p > 0:
            L = D.text_layer([('刀', 1500, 640, 330, min(1, p * 1.5), 1 + 0.5 * (1 - eo(p)), 0.0, RED)], fname='xing')
            fr = T.compose(fr, L, **T.auto_style('red'))
        if tk - 0.02 <= t < tk + 0.2:   # 白刃寒光横扫
            q = (t - tk + 0.02) / 0.22
            fr = TR._slash_streak(fr, (lerp(-200, 2100, eo(q, 2)), 520), np.array((math.cos(math.radians(-12)), math.sin(math.radians(-12)))), q, length=2600, width=4)
            fr = fr + 0.25 * (1 - q)
        z = 1.0 + 0.12 * eo(u, 2)
        return shake_cam(fr, t, [tk], 10, z=z, center=self.pupil), 0.0


class S18_Draw(Shot):
    """MS 风暴将起：鬼屠横刀出鞘，寒光沿刃走过；行书竖排「只听我」"""
    t0, t1 = 180.09, 181.0
    def __init__(self):
        self.sp = D.spr('S103/slow_draw')

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 181, c=(960, 200), spin=0.03, base=(0.02, 0.022, 0.03), lit=(0.2, 0.22, 0.28), eye=260)
        fr = K.fog(fr, t, 44, amt=0.3, speed=(60, 0), y0=0.75, y1=1.0, color=(0.25, 0.27, 0.32))
        cx = 1040 + 40 * u
        fr, Aw = D.real_figure(fr, self.sp, t, D.feet(self.sp), (cx, 1010), D.fit_scale(self.sp, 860), expo=0.85, sat=0.6,
                               rim=COLD, rim_amt=3.0, rim_dir=(-0.7, -0.7), wind=7)
        Fw, A2 = D.warp(self.sp, t, D.feet(self.sp), (cx, 1010), D.fit_scale(self.sp, 860), wind=0)
        lum = Fw.mean(-1)
        sat = Fw.max(-1) - Fw.min(-1)
        metal = A2 * np.clip((lum - 0.28) * 3, 0, 1) * (sat < 0.1)
        fr = D.metal_sheen(fr, metal, np.ones_like(lum), t, (600, 0), 0, speed=lerp(500, 1500, eo(u, 2)), width=90, amt=2.0)
        fr = vcol(fr, '只听我', 300, 300, 120, O[44][7:10], t, WHITE, fname='xing', ink=False)
        return K.cam(fr, 1.03, -20 + 40 * u, 0), 0.0


def stroke_head(ch, fname, x, y, s, p):
    """笔序前沿在画面里的位置（brush_glyph 进度 p 时）"""
    B = T._brush_prep(ch, fname, s, 0)
    O_, a = B['O'], B['a']; G = a.shape[0]
    fy, fx = np.nonzero((np.abs(O_ - p) < 0.02) & (a > 0.5))
    if not len(fy): return None
    return (x - G / 2 + fx.mean(), y - G / 2 + fy.mean())


class S19_Forge(Shot):
    """ECU 刃面斜贯全屏、继续出鞘滑动；刀刃擦出的火星在刀上方的空气里「写」出行书「主张」（火星就是笔锋，
    写完的笔画从白金冷却成暗红余烬）。——字不刻在刀上（副歌一 S046 已用 carved_on_blade），而是刀「写」出来的。"""
    t0, t1 = 181.0, 181.91
    CH = [('主', 700, 330, 300), ('张', 1230, 300, 320)]
    def __init__(self):
        self.sp = D.spr('S103/slow_draw')
        for ch, x, y, sz in self.CH: T._brush_prep(ch, 'xing', sz, 0)

    def render(self, t):
        u = self.u(t)
        fr = ember_bg(190, 0.6)
        fr = D.embers(fr, t, 191, n=50, amt=0.6, size=(1, 2.5))
        bx = 980 - 160 * u
        fr, Aw = D.real_figure(fr, self.sp, t, (0.46, 0.27), (bx, 700), 4.0, rot=14, expo=0.95, sat=0.5,
                               rim=COLD, rim_amt=2.0, wind=0, breathe=0.0)
        Fw, A2 = D.warp(self.sp, t, (0.46, 0.27), (bx, 700), 4.0, rot=14, wind=0)
        metal = A2 * ((Fw.max(-1) - Fw.min(-1)) < 0.1)
        fr = D.metal_sheen(fr, metal, Fw.mean(-1), t, (0, 0), -14, speed=lerp(200, 2000, eo(u, 2)), width=80, amt=1.4)
        hits = []
        for k, ((ch, x, y, sz), on) in enumerate(zip(self.CH, O[44][10:12])):
            tv = V(on); a0 = tv - 0.26; a1 = tv + 0.06
            p = cl((t - a0) / (a1 - a0))
            if p <= 0: continue
            if t >= tv: hits.append(tv)
            A = T.brush_glyph(ch, 'xing', sz, p, 0, 0.35, 0.6)
            L = T.Layer(); L.paste(A, x, y, (1, 1, 1))
            heat = math.exp(-max(0, t - a1) / 0.5)
            hot = np.array([1.0, 0.86, 0.6], np.float32) * heat + np.array([0.95, 0.22, 0.06], np.float32) * (1 - heat)
            Ab = cv2.GaussianBlur(L.A, (0, 0), 14)
            fr = fr * (1 - L.A[..., None] * 0.6) + L.A[..., None] * hot * (1.3 + 0.6 * heat) + Ab[..., None] * np.array([1.0, 0.4, 0.1], np.float32) * (0.5 + 0.6 * heat)
            hd = stroke_head(ch, 'xing', x, y, sz, p) if p < 1 else None
            if hd is not None:      # 笔锋 = 一团火星
                g = np.zeros((H, W), np.float32); cv2.circle(g, (int(hd[0]), int(hd[1])), 10, 1.0, -1, cv2.LINE_AA)
                fr = fr + (cv2.GaussianBlur(g, (0, 0), 4) * 2.5 + cv2.GaussianBlur(g, (0, 0), 30) * 1.5)[..., None] * np.array([1.0, 0.7, 0.35], np.float32)
                fr = K.sparks(fr, t, 1900 + k * 7 + int(p * 20), hd, n=26, life=0.35, speed=(150, 600), dirc=(0, 1), spread=1.4, grav=900, amt=1.0)
        # 刃口持续擦出的火星（向上飞向字）
        fr = K.sparks(fr, t, 1950, (bx - 200, 640), n=70, life=0.5, speed=(400, 1100), dirc=(0.3, -1), spread=0.5, grav=600, area=(300, 30), amt=0.9)
        fr = vcol(fr, '我', 1752, 262, 72, [O[44][9]], t, WHITE, fname='xing', ink=False)
        return shake_cam(fr, t, hits, 7, z=1.0 + 0.05 * u), 0.0


class S20_Wall(Shot):
    """仰拍 LS：宋体字砖墙由下而上砌起、透视压顶（墙顶收窄、像压下来）；湿石质感、雨、雷光从右上扫过；正中「规矩」两砖金光渗出；
    前景：鬼屠背影仰望（新精灵 wall_stand_back）。行书竖排「谁定的规矩」= 他的质问"""
    t0, t1 = 181.91, 182.81
    def __init__(self):
        self.B = T.wall_bricks('律令天条正邪大局理所应当忍让规矩牺牲本分名分上下尊卑', highlight='规矩', hl_color='gold', cols=15, rows=9,
                               brick=(150, 92), center=(960, 500), seed=4)
        self.sp = D.spr('wall_stand_back')
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[330, -60], [W - 330, -60], [W + 260, H - 40], [-260, H - 40]])
        self.Mp = cv2.getPerspectiveTransform(src, dst)
        x, y = K.grid()
        st = cv2.resize(D._tile(205)[:H // 2, :W // 2], (W, H))
        self.stone = (0.75 + 0.5 * st) * (0.55 + 0.45 * (1 - y / H))
        self.hl = None

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 201, c=(960, -80), spin=0.04, base=(0.02, 0.022, 0.03), lit=(0.22, 0.24, 0.3), eye=200)
        p = eo(cl((t - self.t0) / 0.7), 2)
        wl = T.text_wall(np.zeros((H, W, 3), np.float32), p, mode='build', bricks=self.B, brick=(150, 92), seed=4)
        wl = cv2.warpPerspective(wl, self.Mp, (W, H))
        m = np.clip(wl.max(-1) * 5, 0, 1)
        gold = np.clip((wl[..., 0] - wl[..., 2]) * 3, 0, 1) * m           # 金砖
        flash = 0.6 * impact(t, 182.2, 0.14) + 0.9 * impact(t, V(O[45][3]), 0.1)
        x, y = K.grid()
        light = 0.55 + 0.9 * flash * np.exp(-(((x - 1500) / 900) ** 2 + ((y - 100) / 600) ** 2))
        wl = wl * self.stone[..., None] * light[..., None] * (1 - gold[..., None]) + wl * gold[..., None] * (1.4 + 0.6 * math.sin(t * 13))
        fr = fr * (1 - m[..., None]) + wl
        fr = D.godrays(fr, gold, (960, 480), amt=0.5 * p)
        fr = K.rain(fr, t, 20, n=600, amt=0.35, angle=12)
        fr = K.fog(fr, t, 45, amt=0.5, speed=(90, -10), y0=0.75, y1=1.0, color=(0.22, 0.23, 0.27))
        h = lerp(560, 600, u)
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (980, 1030), D.fit_scale(self.sp, h), expo=0.55 + 0.4 * flash, sat=0.5,
                              rim=(1.0, 0.8, 0.5), rim_amt=3.0 + 4 * flash, rim_dir=(0.0, -1.0), wind=9)
        fr = vcol(fr, '谁定的规矩', 230, 250, 104, O[45][0:5], t, WHITE, {'规': GOLD, '矩': GOLD}, fname='xing', ink=False, adv=1.12)
        return shake_cam(fr, t, [V(O[45][3]), V(O[45][4])], 5, z=1.0 + 0.06 * eo(u, 2), oy=-14 * u), 0.0


class S21_Charge(Shot):
    """MS 鬼屠正面冲向镜头；两侧字砖在前景合拢成门；行书「我偏」"""
    t0, t1 = 182.81, 183.69
    def __init__(self):
        self.sp = D.spr('S105/charge_front')
        self.B = T.wall_bricks('律令天条正邪大局理所应当规矩名分尊卑', cols=4, rows=10, brick=(150, 92), center=(0, 540), seed=8)
        x, y = K.grid()
        self.stone = (0.7 + 0.6 * cv2.resize(D._tile(215)[:H // 2, :W // 2], (W, H))) * (0.5 + 0.5 * np.abs(x - 960) / 960)

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 211, c=(960, 300), spin=0.06, base=(0.025, 0.027, 0.035), lit=(0.25, 0.27, 0.33), eye=160)
        fr = D.lightning(fr, t, 183.05, (1500, 120), (1350, 520), seed=3, amt=1.0)
        h = lerp(560, 760, ei(u, 1.5))
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (960, 1000 + 40 * u), D.fit_scale(self.sp, h), expo=0.85, sat=0.6,
                              rim=COLD, rim_amt=3.2, rim_dir=(0.6, -0.8), wind=10, mblur=(0, 6 * u))
        fr = K.rain(fr, t, 21, n=500, amt=0.3, angle=8)
        fr = K.radial_blur(fr, (960, 520), 0.03 + 0.05 * u)
        g = cl((t - self.t0) / (self.t1 - self.t0))
        for side in (-1, 1):
            x = 960 + side * lerp(1250, 900, eo(g, 2))
            fr2 = T.text_wall(np.zeros_like(fr), 1.0, mode='stand', bricks=[dict(b, x=b['x'] + x) for b in self.B], brick=(150, 92), seed=8)
            m = np.clip(fr2.max(-1) * 4, 0, 1)
            m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
            fl = 0.5 + 1.2 * impact(t, 183.05, 0.12)
            fr = fr * (1 - m) + cv2.GaussianBlur(fr2, (0, 0), 3) * self.stone[..., None] * fl
        fr = vcol(fr, '我偏', 300, 300, 130, O[45][5:7], t, WHITE, fname='xing', ink=False)
        return K.cam(fr, 1.0 + 0.04 * u, *K.shake(t, 6, 3, 21)[:2]), 0.0


class S22_Chuang(Shot):
    """「闯」朱红行书巨字立在镜头前，他从字后冲来，184.18 一头撞碎：字裂成碎片朝镜头炸飞"""
    t0, t1 = 183.69, 184.67
    def __init__(self):
        self.sp = D.spr('S105/charge_front')
        self.P = [dict(ch='闯', x=960, y=520, size=700, rot=0.0, font='xing', idx=0, emph=True)]

    def render(self, t):
        u = self.u(t)
        th = V(O[45][8])
        fr = D.storm(t, 221, c=(960, 300), spin=0.08, base=(0.025, 0.027, 0.035), lit=(0.25, 0.27, 0.33), eye=140)
        q = cl((t - self.t0) / (th - self.t0))
        h = lerp(820, 1250, ei(q, 2)) + (600 * eo(cl((t - th) / 0.4)) if t > th else 0)
        fr, _ = D.real_figure(fr, self.sp, t, (0.5, 0.42), (960, 600), D.fit_scale(self.sp, h), expo=0.85, sat=0.6,
                              rim=COLD, rim_amt=3.5, rim_dir=(0.6, -0.8), wind=12)
        fr = K.radial_blur(fr, (960, 520), 0.05 + 0.1 * q)
        p_in = cp(t, O[45][5], 0.3)
        if t < th:
            L = D.text_layer([('闯', 960, 520, 700, min(1, p_in * 2), 1.0 + 0.04 * q, 0.0, RED)], fname='xing')
            fr = T.compose(fr, L, **T.auto_style('red'))
        else:
            fr = T.shatter_out(fr, '闯', cl((t - th) / 0.5), places=self.P, fname='xing', color='red', impact=(960, 560), power=1.6, seed=7)
        fr = vcol(fr, '要', 300, 300, 130, [O[45][7]], t, WHITE, fname='xing', ink=False)
        fr = fr + impact(t, th, 0.08) * 0.8
        return shake_cam(fr, t, [th], 14), 0.0


# ===================================================================================== 第四幕：天判 / 刀答
class S23_Eye(Shot):
    """ELS 仰拍风暴之眼：云涡旋转，闪电游走；185.26 一道闪电照亮云眼里的宋体「天」"""
    t0, t1 = 184.67, 185.57
    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 231, c=(960, 480), spin=0.09, twist=0.8, base=(0.02, 0.022, 0.032), lit=(0.30, 0.33, 0.42), eye=170, bright=1.0,
                     glow=(1300, 300, 0.6 * impact(t, V(O[46][0]), 0.15) + 0.4 * impact(t, 184.85, 0.12)))
        th = V(O[46][0])
        fr = D.lightning(fr, t, 184.85, (300, 120), (760, 500), seed=11, amt=1.2)
        fr = D.lightning(fr, t, th, (1500, 120), (1050, 470), seed=12, amt=1.8, life=0.25)
        p = cl((t - th + 0.03) / 0.12)
        if p > 0:
            L = D.text_layer([('天', 960, 480, 470, p * (0.85 + 0.15 * math.sin(t * 30)), 1.0 + 0.15 * (1 - p))], fname='serif_black', color=(0.86, 0.9, 1.0))
            fr = T.compose(fr, L, shadow=0.0, glow=0.9, glow_color=(0.6, 0.7, 1.0), glow_sigma=30)
        fr = fr + impact(t, th, 0.06) * 0.15
        return K.cam(fr, 1.0 + 0.08 * u, 0, -30 * u, rot=6 * u, center=(960, 480)), 0.0


class S24_Defy(Shot):
    """MCU 仰拍：鬼屠举刀（雷光冷轮廓、雨），右侧行书巨字竖排「天不判」，「判」朱红 punch"""
    t0, t1 = 185.57, 186.5
    def __init__(self):
        self.sp = D.spr('S099/point_up')

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 241, c=(1300, 150), spin=0.08, base=(0.03, 0.032, 0.045), lit=(0.35, 0.38, 0.48), eye=200)
        fr = D.lightning(fr, t, 185.75, (1700, 120), (1500, 600), seed=13, amt=1.0)
        fl = impact(t, 185.75, 0.1)
        fr, _ = D.real_figure(fr, self.sp, t, (0.5, 0.62), (760, 700 - 15 * u), 0.62, expo=0.85 + 0.4 * fl, sat=0.6,
                              rim=(0.75, 0.82, 1.0), rim_amt=3.5 + 4 * fl, rim_dir=(0.8, -0.6), wind=10)
        fr = K.rain(fr, t, 24, n=500, amt=0.35, angle=10)
        tj = V(O[46][2])
        items = []
        for k, (ch, on) in enumerate(zip('天不判', O[46][0:3])):
            p = 1.0 if on < self.t0 else cp(t, on, 0.12)
            if p <= 0: continue
            items.append((ch, 1560, 280 + k * 230, 220, min(1, p * 1.5), 1 + 0.4 * (1 - eo(p)), 0.0, RED if ch == '判' else WHITE))
        L = D.text_layer(items, fname='xing')
        fr = T.compose(fr, L, shadow=0.7, glow=0.3, glow_sigma=16)
        fr = fr + impact(t, tj, 0.08) * RED * 0.5
        z = 1.0 + 0.04 * u + 0.08 * impact(t, tj, 0.1)
        return shake_cam(fr, t, [tj], 12, z=z), 0.0


JUDGE = [dict(ch='判', x=1000, y=470, size=660, rot=0.0, font='xing', idx=0, emph=True)]
J0, J1 = V(O[46][3]) - 0.1, V(O[46][5])


def judge(fr, t, places=JUDGE, glow=1.0):
    p = cl((t - J0) / (J1 - J0))
    if p <= 0: return fr
    out = T.blade_write(fr, '判', p, places=places, fname='xing', color='red', sparks=200, seed=9)
    if t > J1:
        L = D.text_layer([(c['ch'], c['x'], c['y'], c['size'], 1.0, 1.0, 0.0, RED) for c in places], fname='xing')
        out = T.compose(out, L, shadow=0.0, glow=0.9 * glow * (0.6 + 0.4 * impact(t, J1, 0.4)), glow_color=(1.0, 0.15, 0.05), glow_sigma=40)
    return out


class S25_Leap(Shot):
    """LS 背面仰拍：鬼屠跃起向天斜斩，刀光拖出朱白弧线——天幕上「判」的第一笔开始被刻出来"""
    t0, t1 = 186.5, 187.41
    def __init__(self):
        self.sp = D.spr('slash_sky_back')

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 251, c=(1000, 300), spin=0.1, base=(0.03, 0.03, 0.042), lit=(0.38, 0.4, 0.5), eye=220)
        fr = judge(fr, t, [dict(JUDGE[0], x=1080, y=420, size=620)])
        y = 760 - 120 * eo(u, 2)
        fr, Aw = D.real_figure(fr, self.sp, t, (0.5, 0.6), (900, y), D.fit_scale(self.sp, 640) , expo=0.8, sat=0.6,
                               rim=(1.0, 0.45, 0.35), rim_amt=3.5, rim_dir=(0.7, -0.7), wind=12)
        q = cl((t - self.t0) / 0.45)
        if q < 1:
            fr = TR._slash_streak(fr, (lerp(500, 1500, eo(q)), lerp(900, 200, eo(q))), np.array((0.8, -0.6)), q, length=1400, width=5, color=(1.0, 0.75, 0.65))
        fr = vcol(fr, '我', 300, 300, 130, [O[46][3]], t, WHITE, fname='xing', ink=False)
        return K.cam(fr, 1.02 + 0.05 * u, 0, -15 * u), 0.0


class S26_Judge(Shot):
    """ELS：风暴天幕上朱红「判」被刀光一笔笔刻完（187.94 落字：红闪 + 定格辉光），地平线上一个小小的跃斩剪影"""
    t0, t1 = 187.41, 188.34
    def __init__(self):
        self.sp = D.spr('slash_sky_back')

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 261, c=(960, 460), spin=0.05, base=(0.02, 0.02, 0.03), lit=(0.24, 0.25, 0.32), eye=300)
        x, y = K.grid()
        fr = fr * (1 - 0.6 * K.sstep(860, 960, y)[..., None])
        fr = judge(fr, t)
        fr, _ = D.real_figure(fr, self.sp, t, (0.5, 0.6), (960, 830), D.fit_scale(self.sp, 200), expo=0.4, sat=0.3,
                              rim=(1.0, 0.4, 0.3), rim_amt=4.0, wind=6, look='sil')
        fr = vcol(fr, '我来判', 260, 300, 96, [O[46][3], O[46][4], O[46][5]], t, WHITE, {'判': RED}, fname='xing', ink=False)
        fl = impact(t, J1, 0.12)
        fr = fr + fl * RED * 0.3
        return shake_cam(fr, t, [J1], 14, z=1.0 + 0.03 * u + 0.08 * fl), 0.0


class S27_Bullet(Shot):
    """CU 子弹时间：雨滴悬停，插地的刀与握刀的手静立其中，镜头缓慢环绕；「刀不答」竖排，「刀」朱红寒光一闪"""
    t0, t1 = 188.34, 189.71
    def __init__(self):
        self.sp = D.spr('kneel_bow'); self.bg = None

    def render(self, t):
        u = self.u(t)
        if self.bg is None:
            x, y = K.grid()
            self.bg = (0.02 + 0.035 * np.exp(-(((x - 700) / 700) ** 2 + ((y - 400) / 400) ** 2)))[..., None] * np.array([0.8, 0.9, 1.15], np.float32)
        fr = self.bg.copy()
        fr = D.droplets(fr, t, 271, n=110, orbit=-0.3 + 0.6 * u, focus=0.9)
        fr, _ = D.real_figure(fr, self.sp, t, (0.485, 0.2), (760 + 30 * u, 390), 1.45, expo=0.95, sat=0.55,
                              rim=COLD, rim_amt=3.5, rim_dir=(-0.8, -0.5), wind=0, breathe=0.0)
        fr = D.droplets(fr, t, 272, n=40, orbit=-0.8 + 1.6 * u, focus=0.2)
        tk = V(O[47][0])
        fr = vcol(fr, '刀不答', 1500, 300, 150, O[47][0:3], t, WHITE, {'刀': RED}, fname='xing', ink=False)
        if tk <= t < tk + 0.2:
            q = (t - tk) / 0.2
            fr = TR._slash_streak(fr, (lerp(500, 1300, q), lerp(200, 900, q)), np.array((0.6, 0.8)), q, length=1800, width=1, amt=0.6)
        return shake_cam(fr, t, [tk], 6, z=1.0 + 0.04 * u, ox=-30 + 60 * u), 0.0


class S28_Spin(Shot):
    """MS 回旋斩：悬停的雨滴被刀圈全部炸成金点；「我」"""
    t0, t1 = 189.71, 190.61
    def __init__(self):
        self.sp = D.spr('spin_front')

    def render(self, t):
        u = self.u(t)
        x_, y_ = K.grid()
        fr = (0.015 + 0.05 * np.exp(-(((x_ - 960) / 800) ** 2 + ((y_ - 560) / 420) ** 2)))[..., None] * np.array([1.1, 0.95, 0.8], np.float32)
        fr = K.fog(fr, t, 282, amt=0.35, speed=(140, 0), y0=0.55, y1=1.0, color=(0.2, 0.18, 0.16))
        fr = D.droplets(fr, t, 281, n=150, burst=(V(O[47][2]), (900, 560)), focus=0.6)
        rb = 26 * math.exp(-(t - self.t0) / 0.25)
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (1000, 1000), D.fit_scale(self.sp, 700), expo=1.15, sat=0.65,
                              tint=(1.05, 0.98, 0.9), rim=GOLDR, rim_amt=3.5, rim_dir=(0.0, -1.0), wind=12, rotblur=rb)
        fr = K.sparks(fr, t, 2801, (900, 560), n=140, life=0.8, speed=(400, 1400), dirc=(0, -1), spread=3.0, grav=200, burst=True,
                      t0=V(O[47][2]), color=tuple(GOLD), amt=1.4)
        fr = vcol(fr, '我', 300, 330, 200, [O[47][3]], t, WHITE, fname='xing', ink=False)
        return K.cam(fr, 1.04 - 0.04 * u, 0, 0, rot=-3 + 3 * eo(u)), 0.0


class S29_Thrust(Shot):
    """LS 背面：他把刀直刺入云，云层从刀尖处裂出一道金缝；「来」"""
    t0, t1 = 190.61, 191.54
    def __init__(self):
        self.sp = D.spr('thrust_sky_back')

    def render(self, t):
        u = self.u(t)
        fr = D.storm(t, 291, c=(960, 250), spin=0.05, base=(0.035, 0.03, 0.03), lit=(0.38, 0.34, 0.3), eye=180)
        x, y = K.grid()
        n = cv2.resize(D._tile(293)[:H // 4, :W // 4], (W, H))
        crack = np.exp(-(((x - 960 - (n - 0.5) * 80) / (4 + 30 * u)) ** 2)) * K.sstep(560 - 450 * eo(u), 560, y) * (y < 600)
        fr = fr + crack[..., None] * GOLD * 1.8
        fr = D.godrays(fr, crack, (960, 140), amt=0.6 * u)
        tip_y = 300
        sc = D.fit_scale(self.sp, 760)
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (960, 1050), sc, expo=0.8, sat=0.6, tint=(1.04, 0.98, 0.9),
                              rim=GOLDR, rim_amt=3.0 + 2 * u, rim_dir=(0.0, -1.0), wind=12)
        fr = vcol(fr, '来', 300, 330, 200, [O[47][4]], t, WHITE, fname='xing', ink=False)
        return K.cam(fr, 1.0 + 0.05 * u, 0, -20 * u), 0.0


class S30_Da(Shot):
    """ELS 镜头升起：云层沿刀线裂开，金光倾泻；「答」金色行书巨字在裂缝里爆开；最后两拍亮度拉满，白金出画接副歌终"""
    t0, t1 = 191.54, 193.24
    def __init__(self):
        self.sp = D.spr('thrust_sky_back')

    def render(self, t):
        u = self.u(t)
        td = V(O[47][5])
        op = eo(cl((t - self.t0) / 1.4), 2)
        fr = D.storm(t, 301, c=(960, 300), spin=0.05, base=(0.035, 0.03, 0.03), lit=(0.4, 0.36, 0.3), eye=200 + 300 * op)
        x, y = K.grid()
        n = cv2.resize(D._tile(303)[:H // 4, :W // 4], (W, H))
        gap = 20 + 520 * op
        open_ = np.clip((gap - np.abs(x - 960 - (n - 0.5) * 120)) / 60, 0, 1) * (1 - K.sstep(700, 1000, y))
        sky = np.array([1.2, 0.95, 0.6], np.float32) * (0.8 + 0.5 * (1 - y / H))[..., None]
        fr = fr * (1 - open_[..., None]) + sky * open_[..., None]
        edge = np.clip(1 - np.abs(np.abs(x - 960 - (n - 0.5) * 120) - gap) / 30, 0, 1) * (1 - K.sstep(700, 1000, y))
        fr = fr + edge[..., None] * GOLD * 1.5
        fr = D.godrays(fr, open_ + edge, (960, 200), amt=0.9)
        fr, _ = D.real_figure(fr, self.sp, t, D.feet(self.sp), (960, 1000 + 160 * eo(u, 2)), D.fit_scale(self.sp, 420), expo=0.5, sat=0.5,
                              tint=(1.05, 0.95, 0.85), rim=GOLDR, rim_amt=5.0, rim_dir=(0.0, -1.0), wind=12, look='sil')
        p = cp(t, O[47][5], 0.1)
        if p > 0:
            L = D.text_layer([('答', 960, 470, 680, min(1, p * 2), 1 + 0.35 * (1 - eo(p)) + 0.05 * u, 0.0, GOLD)], fname='xing')
            fr = T.compose(fr, L, **T.auto_style('gold'))
            fr = K.sparks(fr, t, 3001, (960, 470), n=220, life=1.0, speed=(500, 1600), dirc=(0, -1), spread=3.1, grav=150, burst=True,
                          t0=td, color=tuple(GOLD), amt=1.5)
        fr = fr + impact(t, td, 0.12) * GOLD * 0.8
        wo = ei(cl((t - (self.t1 - 0.5)) / 0.5), 2)        # 白金出画（接 E 组 flash_white）
        fr = fr * (1 + 1.5 * wo) + wo * np.array([1.0, 0.92, 0.75], np.float32) * 0.9
        z = 1.0 + 0.04 * u + 0.1 * impact(t, td, 0.15)
        return shake_cam(fr, t, [td], 10, z=z, oy=-60 * eo(u, 2)), 0.0


SHOTS = [S01_Drop, S02_Steles, S03_TypeSlam, S04_Snow, S05_Ren, S06_Scale, S07_Heap, S08_Seals, S09_Brush, S10_Giant,
         S11_Answer, S12_Pierce, S13_PaperCut, S14_Burn, S15_Hole, S16_Closed, S17_Eye, S18_Draw, S19_Forge, S20_Wall,
         S21_Charge, S22_Chuang, S23_Eye, S24_Defy, S25_Leap, S26_Judge, S27_Bullet, S28_Spin, S29_Thrust, S30_Da]
# 入点转场：镜头名 → (转场, 切前秒, 切后秒, 参数)
CUTS = {
    'S02_Steles': ('match_cut', 0, 0.35, dict(shape='circle', center=(960, 560), radius=60, glow=np.array([0.9, 0.2, 0.1], np.float32))),
    'S03_TypeSlam': ('hard_cut_on_beat', 0, 0.07, dict(frames=2, impact=0.06, flash=0.0)),
    'S04_Snow': ('smoke_through', 0.12, 0.2, dict(color=(0.93, 0.91, 0.86), seed=4)),
    'S05_Ren': ('ink_bleed', 0.1, 0.2, dict(seed=5, center=(1420, 560))),
    'S06_Scale': ('match_cut', 0, 0.3, dict(shape='circle', center=(680, 600), radius=40, glow=np.array([0.74, 0.1, 0.06], np.float32))),
    'S07_Heap': ('zoom_through', 0.1, 0.15, dict(center=(1270, 580))),
    'S08_Seals': ('whip_pan_blur', 0.08, 0.12, dict(direction='down')),
    'S09_Brush': ('ink_bleed', 0.1, 0.25, dict(seed=9, center=(960, 540))),
    'S10_Giant': ('hard_cut_on_beat', 0, 0.07, dict(frames=2)),
    'S11_Answer': ('match_cut', 0, 0.2, dict(shape='line', center=(1300, 760), angle=-35, mode='cut')),
    'S12_Pierce': ('hard_cut_on_beat', 0, 0.07, dict(frames=2)),
    'S13_PaperCut': ('paper_tear', 0.1, 0.3, dict(seed=12, angle=6)),
    'S14_Burn': ('hard_cut_on_beat', 0, 0.07, dict(frames=2, flash=0.3)),
    'S15_Hole': ('flash_red', 0.04, 0.1, dict(one_frame=True)),
    'S16_Closed': ('smoke_through', 0.1, 0.25, dict(seed=16, direction=-1)),
    'S17_Eye': ('match_cut', 0, 0.3, dict(shape='circle', center=(int(0.29 * W), int(0.555 * H)), radius=8, glow=EMBER)),
    'S18_Draw': ('slash_wipe', 0.05, 0.2, dict(angle=-12)),
    'S19_Forge': ('hard_cut_on_beat', 0, 0.07, dict(frames=2, flash=0.3)),
    'S20_Wall': ('occlusion', 0.1, 0.15, dict(kind='pillar', direction=1)),
    'S21_Charge': ('hard_cut_on_beat', 0, 0.07, dict(frames=2)),
    'S22_Chuang': ('zoom_through', 0.08, 0.12, dict()),
    'S23_Eye': ('whip_pan_blur', 0.1, 0.15, dict(direction='up')),
    'S24_Defy': ('flash_white', 0.03, 0.1, dict(peak=0.4)),
    'S25_Leap': ('slash_wipe', 0.05, 0.2, dict(angle=-40)),
    'S26_Judge': ('match_cut', 0, 0.25, dict(shape='line', center=(1080, 420), angle=-37, mode='grow', glow=np.array([1.0, 0.3, 0.2], np.float32))),
    'S27_Bullet': ('freeze_glitch', 0.06, 0.08, dict(seed=27)),
    'S28_Spin': ('flash_white', 0.03, 0.12, dict(color=(1.0, 0.9, 0.7))),
    'S29_Thrust': ('whip_pan_blur', 0.08, 0.12, dict(direction='up')),
    'S30_Da': ('zoom_through', 0.06, 0.12, dict(center=(960, 300), variant='glint')),
}


class Bridge:
    def __init__(self):
        self.objs = {}
        self.ids = [S.__name__ for S in SHOTS]

    def shot(self, i):
        S = SHOTS[i]
        if S.__name__ not in self.objs: self.objs[S.__name__] = S()
        return self.objs[S.__name__]

    def idx(self, t):
        for i, S in enumerate(SHOTS):
            if t < S.t1 - 1e-6: return i
        return len(SHOTS) - 1

    def raw(self, t):
        i = self.idx(t)
        for j in (i, i + 1):
            if j <= 0 or j >= len(SHOTS): continue
            name, pre, post, kw = CUTS.get(self.ids[j], ('hard', 0, 0, {}))
            c = SHOTS[j].t0
            if name == 'hard' or not (c - pre - 1e-6 <= t < c + post - 1e-6): continue
            B, pb = self.shot(j).render(max(t, c))
            A, pa = self.shot(j - 1).render(min(t, SHOTS[j - 1].t1 - 1e-3)) if (pre > 0 or t < c) else (B, pb)
            p = (t - (c - pre)) / (pre + post)
            if pre == 0: p = 0.5 + 0.5 * p
            kw2 = dict(kw)
            out = TR.transition(name, np.clip(A, 0, 4).astype(np.float32), np.clip(B, 0, 4).astype(np.float32), p, **kw2)
            return out, (pa if p < 0.5 else pb), 3
        fr, pm = self.shot(i).render(t)
        return fr, pm, 0

    def frame(self, t):
        fr, pm, ca = self.raw(t)
        return D.d_finish(fr, t, ca=ca, vig=0.45 - 0.3 * pm, grain=0.04, paper_mode=pm)


_B = [None]


def frame(f, cache):
    if _B[0] is None: _B[0] = Bridge()
    t = f / 30.0
    return K.to8(_B[0].frame(t))


if __name__ == '__main__':
    import sys as _s
    from PIL import Image as _I
    for a in _s.argv[1:]:
        tt = float(a)
        _I.fromarray(frame(int(round(tt * 30)), {})).save('/tmp/d/f_%.2f.png' % tt)
        print('ok', tt)
