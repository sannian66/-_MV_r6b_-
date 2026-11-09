"""units/sec_b_pre1.py — B 组 · 预副歌一「谁配」（帧 1321–1784，44.048–59.47s）
分镜：crew/round4/B_storyboard.md 第 1 节。14 镜 + 副歌一 S028 开头 5 帧（直接调用 chorus1，接缝逐像素一致）。
接口：READY / frame(f, cache) -> uint8 (1080,1920,3)"""
import sys, math, os
ROOT = '/workspace/work/mv'
for p in (ROOT, ROOT + '/lib', ROOT + '/units'):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, cv2
from b_kit import *
import b_kit as BK
import c1kit as K
import typo as T
import trans as TR

READY = False
F0, F1 = 1321, 1784
C1_FROM = 1779                    # 59.30 起交给副歌一（它的 S028 =「血」字头硬切）

# 切点（秒，落拍/字头）→ 帧（视觉帧 = 最近帧 - 1）
CUT_T = [44.048, 44.95, 45.88, 46.79, 47.72, 49.09, 50.46, 51.85, 52.76, 53.68, 54.59, 55.50, 56.40, 57.33, 59.30]
CUT_F = [F0] + [int(round(t * FPS)) - 1 for t in CUT_T[1:-1]] + [C1_FROM]


# =====================================================================================
# P01 ECU 俯拍生死簿：太上皇执朱笔（tsh_brush_hand），「谁配」流江毛草朱笔逐笔写在纸上
NAMES = '张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文'


def ledger(seed, cols=13, size=46, x0=150, x1=1770, y0=150, y1=930, rule=(0.55, 0.12, 0.08), ink=(0.12, 0.10, 0.09), tone=(0.80, 0.74, 0.62)):
    """生死簿纸面：竖格朱丝栏 + 宋体人名竖排（右起）"""
    fr = BK.paper(seed, tone)
    r = np.random.default_rng(seed)
    L = T.Layer()
    cw = (x1 - x0) / cols
    for c in range(cols + 1):
        x = int(x0 + c * cw)
        cv2.line(fr, (x, y0 - 30), (x, y1 + 30), tuple(float(v) * 0.9 for v in rule), 2, cv2.LINE_AA)
    for c in range(cols):
        cx = x1 - (c + 0.5) * cw
        y = y0 + r.uniform(0, size)
        while y < y1 - size:
            n = int(r.integers(2, 4))
            for k in range(n):
                T.place(L, NAMES[int(r.integers(len(NAMES)))], cx, y, size, color=ink, fname='serif_m', alpha=r.uniform(0.6, 0.9))
                y += size * 1.12
            y += size * r.uniform(0.6, 1.4)
    A = L.A
    fr = fr * (1 - A[..., None] * 0.85) + L.C * A[..., None] * 0.85
    return fr.astype(np.float32)


class P01(Shot):
    def __init__(self):
        self.bg = ledger(201)
        self.hand = BK.spr_feather('tsh_brush_hand', left=0.10, right=0.18, bottom=0.32, keep_col=(0.60, 0.68))
        T.prepare_brush('谁配', 'cao', 250)
        self.P = [dict(ch='谁', x=640, y=420, size=260, rot=-4, font='cao', idx=0, emph=False),
                  dict(ch='配', x=760, y=700, size=260, rot=3, font='cao', idx=1, emph=False)]

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        # 烛光：右上暖光斑 + 左下压暗
        x, y = K.grid()
        light = np.exp(-(((x - 1400) / 900) ** 2 + ((y - 250) / 600) ** 2))
        fr = fr * (0.35 + 0.75 * light[..., None]) * np.array([1.0, 0.92, 0.80], np.float32)
        p0 = clamp01((t - 44.0) / 0.42); p1 = clamp01((t - (44.78 - 0.22)) / 0.42)
        # 笔尖跟随正在写的字（笔锋在字心附近画小圈）；手在字的右上方，袖子渐隐进暗部
        k = 1 if p1 > 0 else 0
        pc = p1 if k else p0
        pp = self.P[k]
        ang = pc * 9.0
        tip = (pp['x'] + 40 + 90 * pc + 30 * math.sin(ang), pp['y'] - 70 + 150 * pc + 25 * math.cos(ang * 1.3))
        if pc >= 1 and k == 1: tip = (pp['x'] + 130, pp['y'] + 90 - 60 * (t - 45.0))
        sh = np.zeros((H // 4, W // 4), np.float32)
        cv2.line(sh, (int((tip[0] + 40) / 4), int((tip[1] + 30) / 4)), (int((tip[0] + 420) / 4), int((tip[1] - 900) / 4)), 1.0, 9, cv2.LINE_AA)
        sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 6), (W, H))
        fr = fr * (1 - 0.45 * sh[..., None])
        # 手：笔尖在精灵内约 (0.638, 0.88)；镜像（手从右上入画，不挡字）
        fr, _ = put_at(fr, self.hand, t, (0.638, 0.88), tip, 1500, grade=GR_WARM, rim=RIM_WARM, wind_amp=0, breathe=0.0,
                       shade=None, wrap=0.2)
        fr = T.brush_write(fr, '谁配', [p0, p1], places=self.P, fname='cao', color=(0.62, 0.06, 0.04), splatter=True,
                           style_kw=dict(shadow=0.15))
        z = 1.0 + 0.08 * eio(u)
        fr = cam(fr, z, 0, 0, 2.0 * eio(u) - 1.0)
        # 入口：墨从黑中洇开成纸（接 A 组）
        if t < self.t0 + 0.30:
            m = K.ink_blot(eout((t - self.t0) / 0.28, 2), seed=201, center=(930, 520), reverse=True)
            fr = over(fr, m, np.array([0.015, 0.012, 0.012], np.float32))
        return fr, 0


# =====================================================================================
# P02 微距：朱砂滴落在名字上炸开；「写生死簿」宋体竖排印出，「生死」被朱砂浸红
class P02(Shot):
    T_DROP = V(45.38)
    def __init__(self):
        self.bg = ledger(202, cols=6, size=110, x0=40, x1=1880, y0=130, y1=980)
        T.warm('写生死簿', 'serif_black', 130)
        self.P = [dict(ch=c, x=1010, y=250 + i * 170, size=130, rot=0, font='serif_black', idx=i, emph=c in '生死')
                  for i, c in enumerate('写生死簿')]

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        fr = fr * (0.45 + 0.6 * np.exp(-(((x - 1100) / 800) ** 2 + ((y - 400) / 500) ** 2)))[..., None] * np.array([1.0, 0.93, 0.82], np.float32)
        # 朱砂晕开（落点 = 「生」「死」之间）
        cx, cy = 1010, 505
        if t >= self.T_DROP:
            q = eout((t - self.T_DROP) / 0.45, 2)
            nz = cv2.resize(K.fbm(H // 4, W // 4, 22, base=6, octaves=5), (W, H))
            d = np.sqrt((x - cx) ** 2 + ((y - cy) * 0.8) ** 2) / 330
            m = np.clip((q * 1.0 - d - (nz - 0.5) * 0.5) * 6, 0, 1)
            rim = np.clip(1 - np.abs(q - d - (nz - 0.5) * 0.5) * 12, 0, 1) * 0.5
            fr = fr * (1 - m[..., None] * np.array([0.25, 0.85, 0.9], np.float32)) * (1 - rim[..., None] * 0.3)
            # 飞溅
            r = np.random.default_rng(23)
            for i in range(26):
                a = r.uniform(0, 6.28); dd = r.uniform(120, 520) * eout((t - self.T_DROP) / 0.12)
                rr = int(r.uniform(4, 16))
                cv2.circle(fr, (int(cx + math.cos(a) * dd), int(cy + math.sin(a) * dd * 0.8)), rr, (0.5, 0.05, 0.03), -1, cv2.LINE_AA)
        pr = tp(8, [2, 3, 4, 5], t, 0.16)
        cols = [(0.08, 0.06, 0.05) if not p['emph'] else (0.62, 0.05, 0.04) for p in self.P]
        L = T.Layer()
        for p, pc, c in zip(self.P, pr, cols):
            if pc <= 0: continue
            e = eout(pc, 3)
            T.place(L, p['ch'], p['x'], p['y'], p['size'], color=c, fname='serif_black', alpha=min(1, pc * 2), scale=1 + 0.8 * (1 - e))
        fr = T.compose(fr, L, shadow=0.25)
        # 下落的朱砂滴（前景、虚焦、慢）
        if t < self.T_DROP:
            q = clamp01((t - (self.T_DROP - 0.35)) / 0.35)
            yy = -40 + (cy + 40) * q ** 1.7
            dl = np.zeros((H, W), np.float32)
            cv2.ellipse(dl, (cx, int(yy)), (16, 24), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            dl = cv2.GaussianBlur(dl, (0, 0), 2)
            fr = over(fr, dl, np.array([0.55, 0.03, 0.02], np.float32))
            fr += cv2.GaussianBlur(dl, (0, 0), 8)[..., None] * np.array([0.3, 0.05, 0.03], np.float32)
        k = pulse(t, self.T_DROP, 0.1) * (t >= self.T_DROP)
        z = 1.12 - 0.08 * eio(u) + 0.04 * k
        fr = cam(fr, z, 0, 0, -1.5 + 1.5 * u, center=(cx, cy))
        return fr, (2 if k > 0.5 else 0)


# =====================================================================================
# P03 MS 俯拍：朱笔沿四边画框，把纸面中央的小纸人框住；「谁配画」毛草错落
import paper as PA
BOX_C = (960, 560)


def brush_box(fr, c, half, prog, seed=0, width=26, color=(0.62, 0.06, 0.04)):
    """干笔朱框：四条边依次画出（prog 0→1），飞白 + 起笔顿点"""
    pts = [(c[0] - half, c[1] - half), (c[0] + half, c[1] - half), (c[0] + half, c[1] + half), (c[0] - half, c[1] + half), (c[0] - half, c[1] - half)]
    m = np.zeros((H // 2, W // 2), np.float32)
    for i in range(4):
        q = clamp01(prog * 4 - i)
        if q <= 0: break
        a = np.array(pts[i]); b = np.array(pts[i + 1])
        e = a + (b - a) * q
        n = max(2, int(q * 30))
        for j in range(n):
            s0 = a + (b - a) * q * j / n; s1 = a + (b - a) * q * (j + 1) / n
            w = width * (1.15 - 0.35 * j / n)
            cv2.line(m, tuple((s0 / 2).astype(int)), tuple((s1 / 2).astype(int)), 1.0, max(1, int(w / 2)), cv2.LINE_AA)
        cv2.circle(m, tuple((a / 2).astype(int)), int(width * 0.75 / 2), 1.0, -1, cv2.LINE_AA)
    m = cv2.resize(m, (W, H))
    nz = cv2.resize(K.fbm(H // 2, W // 8, seed, base=8, octaves=4), (W, H))
    dry = np.clip((nz - 0.32) * 4, 0, 1)
    m = m * (0.55 + 0.45 * dry)
    return over(fr, np.clip(m * 1.2, 0, 1), np.array(color, np.float32))


class P03(Shot):
    HALF = 230
    def __init__(self):
        self.bg = BK.paper(203, (0.78, 0.72, 0.60))
        self.fig = PA.PaperFigure(31, view='front')
        T.warm('谁配画', 'cao', 200)

    def scene(self, t, prog, half, c):
        fr = self.bg.copy()
        x, y = K.grid()
        fr = fr * (0.42 + 0.65 * np.exp(-(((x - c[0]) / 900) ** 2 + ((y - c[1] + 100) / 600) ** 2)))[..., None] * np.array([1.0, 0.93, 0.82], np.float32)
        hh = half * 1.25
        # 纸人投影 + 纸人
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(c[0] + hh * 0.12), int(c[1] + hh * 0.48)), (int(hh * 0.28), int(hh * 0.07)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        fr = fr * (1 - 0.5 * cv2.GaussianBlur(sh, (0, 0), 10)[..., None])
        fr, _ = self.fig.draw(fr, c[0], c[1] + hh * 0.48, hh, t, sway=0.6, light='front', light_color=(1.0, 0.8, 0.5), rim=0.6)
        fr = brush_box(fr, c, half, prog, seed=33, width=max(10, half * 0.11))
        return fr

    def render(self, t):
        u = self.u(t)
        prog = clamp01((t - self.t0 - 0.05) / 0.75)
        fr = self.scene(t, prog, self.HALF, BOX_C)
        pr = tp(8, [6, 7, 8], t, 0.18)
        fr = T.scatter_stack(fr, '谁配画', pr, places=[dict(ch='谁', x=1480, y=300, size=150, rot=-6, font='cao', idx=0, emph=False),
                                                         dict(ch='配', x=1600, y=470, size=170, rot=5, font='cao', idx=1, emph=False),
                                                         dict(ch='画', x=1470, y=680, size=230, rot=-3, font='cao', idx=2, emph=True)],
                             color=(0.10, 0.08, 0.07), emph_color=(0.62, 0.06, 0.04), t_local=u, style_kw=dict(shadow=0.2))
        fr = cam(fr, 1.0 + 0.03 * eio(u), -20 * eio(u), 0, 0.8 * u)
        return fr, 0


# =====================================================================================
# P04 LS 后拉：朱框在太上皇巨大指尖下（图形匹配：P03 的框 = 指下的框）；「善恶框」，「框」字被真框住
class P04(Shot):
    HALF = 170
    C = (960, 610)
    T_KUANG = V(47.70)
    def __init__(self):
        self.p3 = P03()
        self.finger = spr('S021/finger')
        T.warm('善恶', 'cao', 170); T.warm('框', 'cao', 230)

    def render(self, t):
        u = self.u(t)
        fr = self.p3.scene(t, 1.0, self.HALF, self.C)
        # 指尖：从上方压下按住框上沿（finger 精灵：指尖在精灵内约 (0.50, 0.97)）
        k = eout(clamp01((t - self.t0) / 0.5), 3)
        tip = (self.C[0] - 40, self.C[1] - self.HALF - 6 - 220 * (1 - k))
        fr = fr * (1 - 0.35 * cv2.GaussianBlur((np.abs(K.grid()[0] - tip[0] - 60) < 220).astype(np.float32) * (K.grid()[1] < tip[1] + 30), (0, 0), 60)[..., None])
        fr, _ = put_at(fr, self.finger, t, (0.50, 0.96), tip, 1500, grade=dict(expo=0.55, sat=0.4, tint=(1.04, 0.96, 0.86), lift=0.0, gamma=1.2),
                       rim=RIM_WARM, wind_amp=0, breathe=0.0, shade=None, wrap=0.25)
        # 「善恶框」：善恶毛草；「框」外画朱红真框，字头时合拢
        pr = tp(8, [9, 10, 11], t, 0.18)
        L = T.Layer()
        for ch, xx, yy, s, pc in (('善', 1450, 330, 150, pr[0]), ('恶', 1590, 500, 150, pr[1])):
            if pc > 0: T.place(L, ch, xx, yy, s, color=(0.10, 0.08, 0.07), fname='cao', alpha=min(1, pc * 2), scale=1 + (1 - eout(pc)) * 1.2)
        if pr[2] > 0: T.place(L, '框', 1520, 720, 200, color=(0.62, 0.06, 0.04), fname='cao', alpha=min(1, pr[2] * 2), scale=1 + (1 - eout(pr[2])) * 1.2)
        fr = T.compose(fr, L, shadow=0.2)
        kq = clamp01((t - (self.T_KUANG - 0.25)) / 0.25)
        if kq > 0:
            fr = brush_box(fr, (1520, 720), 150, kq, seed=44, width=16)
        # 后拉：开头 z=230/170（与 P03 框同大同位），拉到 1.0
        z0 = P03.HALF / self.HALF
        z = z0 + (1.0 - z0) * eout(u, 2.4)
        cx = BOX_C[0] + (self.C[0] - BOX_C[0]); cy = self.C[1]
        pk = pulse(t, self.T_KUANG, 0.1) * (t >= self.T_KUANG)
        fr = cam(fr, z * (1 + 0.05 * pk), (BOX_C[0] - cx) * 0, (BOX_C[1] - cy) * (1 - eout(u, 2.4)), 0.8 * (1 - u), center=(cx, cy))
        return fr, (3 if pk > 0.5 else 0)


# =====================================================================================
# P05 LS 仰：夜色高墙 + 黑漆金字匾「规矩」，「谁的规矩」凹刻进砖面、随砖缝断开
class P05(Shot):
    def __init__(self):
        wall, seam = K.brick_wall(2600, 1700, 150, 64, 7, 905)
        q = np.float32([[-260, 120], [2180, 120], [1960, 1080 + 260], [-40, 1080 + 260]])   # 仰拍梯形（上宽下窄）
        self.wall = K.warp_quad(wall, q, (H + 300, W)); self.seam = K.warp_quad(seam[..., None].repeat(3, -1), q, (H + 300, W))[..., 0]
        self.sky = K.sky(seed=905, cloud=0.8, horizon=0.2, stars=0.2)
        self.plq = __import__('props').Plaque('规矩', seed=5, fname='song')
        T.warm('谁的规矩', 'serif_black', 150)

    def render(self, t):
        u = self.u(t)
        oy = 170 - 260 * eio(u)            # 下摇：匾 → 墙脚
        fr = np.zeros((H + 300, W, 3), np.float32)
        fr[:] = cv2.resize(self.sky, (W, H + 300))
        m = np.zeros((H + 300, W), np.float32); cv2.fillConvexPoly(m, np.int32([[-260, 120], [2180, 120], [1960, 1380], [-40, 1380]]), 1.0)
        wl = self.wall * 1.15
        # 冷月顶光 + 匾下暖反光
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H + 300, dtype=np.float32))
        wl = wl * (0.55 + 0.6 * np.exp(-((y - 200) / 600) ** 2))[..., None] + np.exp(-(((x - 960) / 420) ** 2 + ((y - 420) / 160) ** 2))[..., None] * GOLD * 0.10
        fr = fr * (1 - m[..., None]) + wl * m[..., None]
        off = int(150 - oy * 0.5)
        fr = np.ascontiguousarray(fr[off:off + H]); seam = self.seam[off:off + H]
        fr, _ = self.plq.draw(fr, t, 960, 300 - off, 720, sheen=clamp01((t - self.t0) / 1.2) * 1.4 - 0.2)
        # 刻字：每字切成砖块碎片（砖缝处断开）
        pr = tp(9, [0, 1, 2, 3], t, 0.2)
        L = T.Layer()
        for i, (ch, pc) in enumerate(zip('谁的规矩', pr)):
            if pc > 0: T.place(L, ch, 600 + i * 240, 760 - off, 190, color='white', fname='serif_black', alpha=min(1, pc * 2.5))
        A = L.A * (1 - np.clip(seam * 1.6, 0, 1))
        inner = A * 0.85
        hl = np.clip(np.roll(A, 3, 0) - A, 0, 1)
        sh = np.clip(np.roll(A, -3, 0) - A, 0, 1)
        fr = fr * (1 - inner[..., None] * 0.75) + inner[..., None] * np.array([0.03, 0.025, 0.025], np.float32)
        fr = fr + hl[..., None] * 0.35 * np.array([0.8, 0.85, 1.0], np.float32) - sh[..., None] * 0.05
        glow = sum(pulse(t, V(CT(9, k)), 0.18) * (t >= V(CT(9, k))) for k in range(4))
        fr = fr + inner[..., None] * GOLD * (0.15 + 0.4 * glow)
        fr = BK.rain(fr, t, 951, amt=0.3, n=400, angle=6)
        fr = K.fog(fr, t, 952, amt=0.22, y0=0.6, y1=1.0)
        fr = cam(fr, 1.0 + 0.06 * eio(u), 0, 0)
        return fr, 0


# =====================================================================================
# P06 CU：匾下涌出朱红血水，沿砖缝网格往下流；「血」巨字从砖缝渗出并下流；「用…来装潢」宋体竖排
class P06(Shot):
    T_XUE = V(49.62)
    def __init__(self):
        wall, seam = K.brick_wall(2200, 1300, 240, 100, 10, 906)
        self.wall = wall[:H, :W] * 1.1; self.seam = seam[:H, :W]
        T.warm('血', 'cao', 520); T.warm('用来装潢', 'serif_sb', 72)
        self.flow_n = cv2.resize(K.fbm(H // 8, W // 8, 906, base=4, octaves=4), (W, H))

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        fr = self.wall * (0.5 + 0.55 * np.exp(-((x - 900) / 900) ** 2))[..., None]
        fr = fr * np.array([0.95, 0.98, 1.05], np.float32)
        # 血流前沿：从顶部沿砖缝向下推进（竖缝快、横缝慢）
        front = 120 + (H + 200) * ein(clamp01((t - self.t0) / 1.25), 1.4) - self.flow_n * 260
        vs = self.seam * (y < front)
        blood = np.clip(vs * 1.3, 0, 1)
        top = np.clip((260 - y) / 140, 0, 1) * clamp01((t - self.t0) / 0.3)       # 顶部匾底一片血
        blood = np.maximum(blood, top * (0.6 + 0.4 * self.flow_n))
        wet = cv2.GaussianBlur(blood, (0, 0), 6)
        fr = fr * (1 - wet[..., None] * 0.5)
        fr = over(fr, blood, np.array([0.42, 0.02, 0.015], np.float32))
        spec = np.clip(cv2.Sobel(cv2.GaussianBlur(blood, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3), 0, 1)
        fr = fr + spec[..., None] * np.array([0.9, 0.4, 0.35], np.float32) * 0.6
        # 「血」巨字：49.62 字头墨洇显出，之后下流
        pb = clamp01((t - (self.T_XUE - 0.12)) / 0.3)
        if pb > 0:
            a = T.ink_bleed_glyph('血', 'cao', 520, max(0.02, pb), seed=6, bleed=1.0)
            L = T.Layer(); L.paste(a, 760, 520, (0.70, 0.04, 0.03))
            L = T.drip_layer(L, clamp01((t - self.T_XUE) / 0.8), n=7, seed=6, length=0.5)
            fr = T.compose(fr, L, shadow=0.55, glow=0.25, glow_color=(0.6, 0.05, 0.03))
        # 用 血 来 装 潢（宋体竖排，血位留红点）
        pr = tp(9, [4, 5, 6, 7, 8], t, 0.18)
        L = T.Layer()
        for i, (ch, pc) in enumerate(zip('用血来装潢', pr)):
            if pc <= 0: continue
            col = (0.84, 0.10, 0.07) if ch == '血' else (0.95, 0.93, 0.88)
            T.place(L, ch, 1560, 300 + i * 105, 76, color=col, fname='serif_sb', alpha=min(1, pc * 2), scale=1 + 0.4 * (1 - eout(pc)))
        fr = T.compose(fr, L, shadow=0.75, glow=0.12)
        k = pulse(t, self.T_XUE, 0.12) * (t >= self.T_XUE)
        fr = cam(fr, 1.0 + 0.04 * eio(u) + 0.06 * k, 0, 40 - 80 * eio(u))
        return fr, (3 if k > 0.5 else 0)


# =====================================================================================
# P07 MS 低机位仰拍：鬼屠仰头，雷云旋转、云缝预亮；「我」「不信」毛草错落
class P07(Shot):
    def __init__(self):
        self.sp = spr('S023/look_up_low')
        self.tex = K.tile(1024, 907, 4, 6)
        T.warm('我不信', 'cao', 200); T.warm('不', 'xing', 150)
        import fx2; self.fx2 = fx2

    def sky(self, t, rot=0.0, seed_off=0):
        x, y = K.grid()
        cx, cy = 960, -200
        ang = np.arctan2(y - cy, x - cx) + t * 0.08 + rot
        rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        u_ = (ang * 160) % 1024; v_ = (rr * 0.6 - t * 30) % 1024
        n = self.tex[v_.astype(np.int32) % 1024, u_.astype(np.int32) % 1024]
        n = cv2.GaussianBlur(n, (0, 0), 2)
        l = 0.03 + 0.16 * n ** 1.6
        return np.stack([l * 0.95, l * 0.98, l * 1.08], -1).astype(np.float32)

    def render(self, t):
        u = self.u(t)
        fr = self.sky(t)
        fl = self.fx2.cloud_flash(fr, t, t0=50.95, center=(1350, 250), radius=600, seed=7, intensity=0.8)
        fr = fl if isinstance(fl, np.ndarray) else fl[0]
        crane = -60 + 90 * eio(u)
        fr, _ = put(fr, self.sp, t, 1700, (820, 1080 + 1050 + crane), rim=dict(col=(0.75, 0.82, 1.0), amt=2.2, dir=(0.4, -1.0)),
                    grade=GR_COLD, wind_amp=6, shade=(400, 1500, 0.2, 700, 0.3))
        fr = BK.rain(fr, t, 971, amt=0.42, n=650, angle=-8, speed=2500)
        pr = tp(10, [0, 1, 2], t, 0.18)
        fr = T.scatter_stack(fr, '我不信', pr, places=[dict(ch='我', x=1400, y=330, size=190, rot=-5, font='cao', idx=0, emph=False),
                                                         dict(ch='不', x=1560, y=540, size=150, rot=6, font='xing', idx=1, emph=True),
                                                         dict(ch='信', x=1690, y=720, size=170, rot=-4, font='cao', idx=2, emph=True)],
                             color='white', emph_color=(0.84, 0.10, 0.07), t_local=u, style_kw=dict(shadow=0.75, glow=0.12))
        fr = cam(fr, 1.04 - 0.04 * eio(u), 0, 0, -4 + 4 * eio(u))
        return fr, 0


# =====================================================================================
# P08 ELS 仰天：云隙负形构成「天」（宋体），52.14 闪电劈亮；朱红干笔划掉
class P08(Shot):
    T_TIAN = V(52.14)
    def __init__(self):
        self.p7 = P07()
        self.m = T.glyph_mask('天', 'serif_black', 760, 960, 520) if hasattr(T, 'glyph_mask') else None
        import fx2; self.fx2 = fx2
        self.bolt = fx2.Lightning(52, p0=(1300, 120), p1=(980, 560), branches=6)

    def render(self, t):
        u = self.u(t)
        fr = self.p7.sky(t * 1.3, rot=1.2)
        m = cv2.GaussianBlur(self.m.astype(np.float32), (0, 0), 7)
        nz = cv2.resize(K.fbm(H // 4, W // 4, 81, base=6, octaves=4), (W, H))
        m = np.clip(m * (0.8 + 0.4 * nz), 0, 1)
        lit = 0.25 + 0.75 * clamp01((t - (self.T_TIAN - 0.05)) / 0.06) * (0.6 + 0.4 * math.exp(-(t - self.T_TIAN) / 0.4) if t > self.T_TIAN else 1.0)
        glowc = np.array([0.85, 0.88, 1.0], np.float32)
        fr = fr * (1 - m[..., None] * 0.7) + m[..., None] * glowc * 0.75 * lit
        fr += cv2.GaussianBlur(m, (0, 0), 40)[..., None] * glowc * 0.25 * lit
        if t >= self.T_TIAN - 0.05:
            fr = self.fx2.Lightning.draw(self.bolt, fr, t, self.T_TIAN, intensity=1.0)
        # 朱红一笔划掉
        ps = clamp01((t - (self.T_TIAN + 0.22)) / 0.16)
        if ps > 0:
            P = [dict(ch='天', x=960, y=520, size=760, rot=0, font='serif_black', idx=0, emph=False)]
            fr = T.strike(fr, P, [0], ps, color=(0.84, 0.10, 0.07), width=0.12, seed=8)
        z = 1.0 + 0.06 * eio(u)
        fr = cam(fr, z, 0, 30 - 60 * eio(u), 3 * eio(u))
        return fr, (3 if abs(t - self.T_TIAN) < 0.04 else 0)


# =====================================================================================
# P09 MCU：鬼屠握拳（fist）；「命」朱印从背后逼近；53.17 刀柄回撞（pommel_thrust），印碎
class P09(Shot):
    T_HIT = V(53.17)
    def __init__(self):
        self.a = spr('S024/fist'); self.b = spr('S024/pommel_thrust')
        self.bg = BK.dark_bg(909, 0.02, 0.06, tint=(1.1, 1.0, 1.0))
        self.seal = T.seal_layer('命', 960, 470, 520, style='yin', rot=-4, seed=9)
        T.warm('不信', 'cao', 120); T.warm('不', 'xing', 130)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        fr += np.exp(-(((x - 960) / 700) ** 2 + ((y - 450) / 400) ** 2))[..., None] * RED * 0.25
        # 「命」印：逼近（缩放 0.6→1.0）→ 撞碎
        pc = tp(10, [6], t, 0.3)[0]
        if t < self.T_HIT:
            s = 0.55 + 0.45 * ein(clamp01((t - self.t0) / (self.T_HIT - self.t0)), 1.5)
            L = T.Layer(); a = cv2.resize(self.seal.A, None, fx=s, fy=s); c = cv2.resize(self.seal.C, None, fx=s, fy=s)
            L.paste_rgba(c, a, int(960 - 960 * s), int(470 - 470 * s))
            L.A = cv2.GaussianBlur(L.A, (0, 0), 2.5 * (1 - s) + 0.5)
            fr = T.compose(fr, L, shadow=0.5, opacity=min(1, pc + 0.4))
        else:
            q = clamp01((t - self.T_HIT) / 0.5)
            L = T.shatter_layer(self.seal, q, impact=(960, 520), n=80, seed=9, power=1.4)
            fr = T.compose(fr, L, shadow=0.4)
        if t < self.T_HIT:
            fr, _ = put_at(fr, self.a, t, (0.5, 0.42), (930, 520), 2150, grade=GR_COLD, rim=dict(col=(1.0, 0.45, 0.35), amt=1.8, dir=(0.0, -1.0)),
                           wind_amp=4, shade=(700, 1800, 0.25, 700, 0.35))
        else:
            k = clamp01((t - self.T_HIT) / 0.08)
            fr, _ = put_at(fr, self.b, t, (0.5, 0.45), (980, 560), 2250 * (1.05 - 0.05 * k), grade=GR_COLD,
                           rim=dict(col=(1.0, 0.45, 0.35), amt=2.0, dir=(0.0, -1.0)), wind_amp=6, mblur=(-50 * (1 - k), 0), shade=(700, 1800, 0.2, 700, 0.3))
            fr = K.sparks(fr, t, 991, (720, 380), n=60, life=0.5, speed=(400, 1200), dirc=(-0.4, -1), spread=1.2, grav=900, area=(80, 80), amt=1.0)
        if self.T_HIT <= t < self.T_HIT + 1 / 30 + 1e-3: fr = fr * 0.4 + 0.75
        pr = tp(10, [7, 8], t, 0.18)
        fr = T.scatter_stack(fr, '不信', pr, places=[dict(ch='不', x=300, y=330, size=130, rot=-6, font='xing', idx=0, emph=False),
                                                       dict(ch='信', x=380, y=500, size=150, rot=5, font='cao', idx=1, emph=False)],
                             color='white', t_local=u, style_kw=dict(shadow=0.75, glow=0.1))
        k2 = pulse(t, V(52.84), 0.1) * (t >= V(52.84)) + pulse(t, self.T_HIT, 0.12) * (t >= self.T_HIT)
        dx, dy, dr = shake(t, 6 + 14 * k2, 8, 9)
        fr = cam(fr, 1.0 + 0.05 * k2 + 0.03 * eio(u), dx, dy, dr)
        return fr, (4 if k2 > 0.5 else 0)


# =====================================================================================
# 铅字块（P10/P11 共用）：金属活字，宋体浮凸
def type_block(ch, size=220, seed=0, face=(0.42, 0.42, 0.45), gold=0.0):
    """返回 (rgb, A)：正方形铅字顶面（含倒角、磨损、浮凸字面）"""
    s = size
    rgb = np.zeros((s, s, 3), np.float32); A = np.zeros((s, s), np.float32)
    r = np.random.default_rng(seed)
    nz = cv2.resize(r.random((s // 6, s // 6)).astype(np.float32), (s, s))
    base = np.array(face, np.float32) * (0.75 + 0.3 * nz[..., None])
    rgb[:] = base * 0.45
    A[4:-4, 4:-4] = 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    ed = np.minimum(np.minimum(xx, s - xx), np.minimum(yy, s - yy))
    bev = np.clip(1 - ed / 14, 0, 1)
    rgb += (bev * ((xx < yy) * 0.25 - 0.1))[..., None]
    g = T.glyph(ch, 'serif_black', int(s * 0.74), False)      # 方形画布 ≈1.5×size、字心居中 → 居中裁 s×s
    gh, gw = g.shape
    gm = np.zeros((s, s), np.float32)
    if gh >= s:
        o = (gh - s) // 2; gm[:] = g[o:o + s, o:o + s]
    else:
        o = (s - gh) // 2; gm[o:o + gh, o:o + gw] = g
    hl = np.clip(gm - np.roll(np.roll(gm, 3, 0), 3, 1), 0, 1)
    sd = np.clip(gm - np.roll(np.roll(gm, -3, 0), -3, 1), 0, 1)
    col = np.array(face, np.float32) * 1.6 * (1 - gold) + GOLD * 1.1 * gold
    rgb = rgb * (1 - gm[..., None]) + gm[..., None] * col * (0.8 + 0.25 * nz[..., None])
    rgb = rgb + hl[..., None] * 0.45 - sd[..., None] * 0.25
    return np.clip(rgb, 0, 2).astype(np.float32), A


def tip_blade():
    """真实刀身横置（刀尖在右），尖端按三角形切出"""
    F, A = BK.blade_strip()
    h, w = A.shape
    A = A.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    L = 260
    tipm = np.clip(((w - xx) / L) * h - np.abs(yy - h * 0.05) * 1.0, 0, 1)   # 背直、刃弧（单刃刀尖）
    A = A * np.where(xx > w - L, np.clip(tipm, 0, 1), 1.0)
    return F, A


class P10(Shot):
    T_LI, T_SUO = V(54.0), V(54.18)
    def __init__(self):
        r = np.random.default_rng(910)
        S = 230
        canvas = np.zeros((1500, 2800, 3), np.float32) + 0.02
        self.pos = {}
        for row in range(6):
            for col in range(11):
                cx, cy = 140 + col * (S + 10), 160 + row * (S + 10)
                main = (row == 3 and 3 <= col <= 6)
                ch = '理所应当'[col - 3] if main else NAMES[int(r.integers(len(NAMES)))]
                if main: self.pos[ch] = (cx, cy)
                rgb, A = type_block(ch, S, seed=row * 20 + col, gold=0.35 if main else 0.0)
                canvas[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2] = rgb * (1.0 if main else 0.75)
        self.canvas = canvas; self.S = S
        self.q = np.float32([[-300, -60], [2350, -160], [2050, 1240], [-80, 1180]])     # 斜俯透视
        self.Mq = cv2.getPerspectiveTransform(np.float32([[0, 0], [2800, 0], [2800, 1500], [0, 1500]]), self.q)
        self.blk = {c: type_block(c, S, seed=99 + i, gold=0.35) for i, c in enumerate('理所')}
        self.blade = tip_blade()
        T.warm('不信谁的', 'cao', 110)

    def proj(self, p):
        v = self.Mq @ np.array([p[0], p[1], 1.0]); return v[:2] / v[2]

    def render(self, t):
        u = self.u(t)
        cv_ = self.canvas.copy()
        S = self.S
        flying = []
        for ch, th in (('理', self.T_LI), ('所', self.T_SUO)):
            if t >= th - 0.02:
                cx, cy = self.pos[ch]
                cv_[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2] = 0.01      # 空槽
                flying.append((ch, th))
        fr = cv2.warpPerspective(cv_, self.Mq, (W, H), flags=cv2.INTER_LINEAR)
        x, y = K.grid()
        fr = fr * (0.35 + 0.9 * np.exp(-(((x - 1100) / 800) ** 2 + ((y - 450) / 450) ** 2)))[..., None] * np.array([0.95, 0.98, 1.06], np.float32)
        # 飞起的铅字：抛物线 + 旋转 + 动态模糊
        for ch, th in flying:
            q = t - th
            p0 = self.proj(self.pos[ch])
            px = p0[0] - 900 * q; py = p0[1] - 1400 * q + 1800 * q * q
            rgb, A = self.blk[ch]
            sc = 0.82 * (1 + 0.8 * q)
            M = cv2.getRotationMatrix2D((S / 2, S / 2), -q * 720, sc); M[0, 2] += px - S / 2; M[1, 2] += py - S / 2
            fr, _ = BK.paste_rgba(fr, rgb * 1.3, A, M)
        # 刀尖：从右侧探入，挑「理」「所」
        F, A = self.blade
        tgt = self.proj(self.pos['理']) if t < self.T_LI + 0.05 else self.proj(self.pos['所'])
        e = eout(clamp01((t - self.t0) / 0.25), 3)
        flick = pulse(t, self.T_LI, 0.08) * (t >= self.T_LI) + pulse(t, self.T_SUO, 0.08) * (t >= self.T_SUO)
        tip = np.array([tgt[0] + 40, tgt[1] + 30 - 120 * flick]) + np.array([900, 0]) * (1 - e)
        base = tip + np.array([1900, 280 + 200 * flick])
        M = BK.M_line(A.shape[1], A.shape[0], base, tip, 150)
        fr, Aw = BK.paste_rgba(fr, F * 1.15, A, M)
        fr += cv2.GaussianBlur(Aw, (0, 0), 20)[..., None] * np.array([0.6, 0.7, 0.9], np.float32) * 0.15 * (1 + 2 * flick)
        if flick > 0.3:
            fr = K.sparks(fr, t, 1010, tuple(tip.astype(int)), n=40, life=0.35, speed=(300, 900), dirc=(-1, -0.6), spread=0.9, grav=900, area=(20, 20), amt=1.0)
        pr = tp(10, [9, 10], t, 0.18)
        fr = T.scatter_stack(fr, '谁的', pr, places=[dict(ch='谁', x=250, y=260, size=130, rot=-6, font='cao', idx=0, emph=False),
                                                       dict(ch='的', x=380, y=400, size=110, rot=4, font='cao', idx=1, emph=False)],
                             color='white', t_local=u, style_kw=dict(shadow=0.8))
        fr = cam(fr, 1.04, 60 - 160 * eio(u), 0, -1.5)
        return fr, (3 if flick > 0.6 else 0)


# =====================================================================================
# P11 微距慢放：「应」「当」两块铅字在空中翻滚，55.1「当」落地弹起溅火星
class P11(Shot):
    T_DANG = V(55.10)
    def __init__(self):
        self.blk = {c: type_block(c, 300, seed=120 + i, gold=0.35) for i, c in enumerate('应当')}
        self.bg = BK.dark_bg(911, 0.01, 0.04)
        self.bok = np.random.default_rng(911).uniform(0, 1, (40, 4))

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        floor = sstep(780, 800, y)
        fr = fr * (1 - floor[..., None]) + floor[..., None] * (0.03 + 0.03 * cv2.resize(K.fbm(H // 4, W // 4, 9, 6, 4), (W, H)))[..., None]
        # 背景虚焦光斑（冷/暖）
        Lb = np.zeros((H // 4, W // 4), np.float32)
        for bx, by, br, ph in self.bok:
            cv2.circle(Lb, (int(bx * W / 4), int(by * 180 + 20)), int(6 + br * 14), 0.25 + 0.2 * math.sin(t * 2 + ph * 6), -1, cv2.LINE_AA)
        fr += cv2.resize(cv2.GaussianBlur(Lb, (0, 0), 3), (W, H))[..., None] * np.array([0.6, 0.65, 0.8], np.float32)
        ts = self.t0 + (t - self.t0) * 0.45                   # 慢放
        for ch, (x0, y0, vx, vy, spin, land) in (('应', (700, 300, 160, -120, 260, None)), ('当', (1150, 260, -90, -60, -320, self.T_DANG))):
            rgb, A = self.blk[ch]
            q = ts - self.t0
            if land is None:
                px, py = x0 + vx * q * 2, y0 + vy * q * 2 + 260 * q * q * 4
                ang = spin * q * 2; sy = abs(math.cos(q * 5.0)) * 0.9 + 0.1
            else:
                tl = (t - self.t0) / (self.T_DANG - self.t0)
                if t < self.T_DANG:
                    px = x0 + vx * tl; py = y0 + (640 - y0) * tl ** 2; ang = spin * tl; sy = abs(math.cos(tl * 4.2)) * 0.9 + 0.1
                else:
                    b = t - self.T_DANG
                    px = x0 + vx + 60 * b; py = 640 - 140 * math.sin(min(math.pi, b * 9)) * math.exp(-b * 3); ang = spin + 20 * math.sin(b * 12) * math.exp(-b * 4); sy = 0.9
            M = cv2.getRotationMatrix2D((150, 150), ang, 0.9); M[1] *= sy
            M[0, 2] += px - 150; M[1, 2] += py - 150 * sy
            Mb = M.copy()
            fr, Aw = BK.paste_rgba(fr, rgb * 1.35, A, Mb)
            fr += (cv2.GaussianBlur(Aw, (0, 0), 3) - Aw).clip(0, 1)[..., None] * np.array([0.8, 0.85, 1.0], np.float32) * 0.6
        if t >= self.T_DANG:
            fr = K.sparks(fr, t, 1111, (1060, 760), n=70, life=0.5, speed=(300, 900), dirc=(0, -1), spread=1.3, grav=1200, area=(120, 10), amt=1.1)
        # 刃光掠过（前景斜线）
        q = clamp01((t - (self.t0 + 0.15)) / 0.25)
        if 0 < q < 1:
            fr = T._slash_light(fr, (960, 420), np.array([0.94, -0.34]), q, 2400)
        k = pulse(t, self.T_DANG, 0.1) * (t >= self.T_DANG)
        dx, dy, dr = shake(t, 10 * k, 12, 11)
        fr = cam(fr, 1.0 + 0.06 * eio(u) + 0.04 * k, dx - 40 * eio(u), dy, dr)
        return fr, (3 if k > 0.5 else 0)


# =====================================================================================
# P12 MCU：鬼屠竖刀当面，刀身一分为二的脸（blade_face）；「只信这柄」马善政竖排，逐字光洇
class P12(Shot):
    def __init__(self):
        self.sp = spr('blade_face')
        self.bg = BK.dark_bg(912, 0.015, 0.05)
        T.warm('只信这柄', 'kai', 96)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        fr += np.exp(-(((x - 960) / 500) ** 2 + ((y - 380) / 300) ** 2))[..., None] * np.array([0.10, 0.11, 0.14], np.float32)
        fr, _ = put_at(fr, self.sp, t, (0.5, 0.36), (960, 470), 2350, grade=dict(expo=0.82, sat=0.6, tint=(0.92, 0.96, 1.05), lift=0.0, gamma=1.12),
                       rim=dict(col=(0.75, 0.82, 1.0), amt=2.2, dir=(-0.9, -0.4)), wind_amp=5, shade=(1100, 1900, 0.25, 760, 0.4))
        # 刀刃上一线寒光自上而下
        q = clamp01((t - self.t0 - 0.1) / 0.6)
        yy = 120 + 600 * eio(q)
        g = np.exp(-(((x - 960) / 26) ** 2 + ((y - yy) / 60) ** 2))
        fr += g[..., None] * np.array([0.9, 0.95, 1.0], np.float32) * 0.8 * (1 - q * 0.5)
        fr = K.sparks(fr, t, 1212, (960, 980), n=30, life=1.2, speed=(60, 200), dirc=(0, -1), spread=0.8, grav=-60, area=(900, 40), amt=0.6)
        pr = tp(11, [0, 1, 2, 3], t, 0.2)
        fr = T.ink_bleed_in(fr, '只信这柄', pr, places=[dict(ch=c, x=1420, y=300 + i * 130, size=104, rot=0, font='kai', idx=i, emph=False)
                                                        for i, c in enumerate('只信这柄')],
                            color='white', style_kw=dict(shadow=0.85, glow=0.12))
        fr = cam(fr, 1.0 + 0.08 * eio(u), 0, 0)
        return fr, 0


# =====================================================================================
# P13 CU：真实刀身横贯，56.5「刀」字头白刃光扫过，刀身内部开始透出炭火红；「刀」志莽行书被刀光斩出
def heat_blade(fr, t, p0, p1, width, heat, shine=0.0, seed=0, haze=0.0):
    F, A = tip_blade()
    M = BK.M_line(A.shape[1], A.shape[0], p0, p1, width)
    Fw = cv2.warpAffine(F, M, (W, H), flags=cv2.INTER_LINEAR)
    Aw = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR)
    lum = Fw.mean(-1, keepdims=True)
    # 炭火：内部透红（亮处更红），刃口一侧更热
    x, y = K.grid()
    d = np.array(p1, float) - np.array(p0, float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
    sd = ((x - p0[0]) * n[0] + (y - p0[1]) * n[1]) / (width / 2)           # -1 背 … +1 刃
    nz = cv2.resize(K.fbm(H // 8, W // 8, seed, base=6, octaves=4), (W, H))
    hz = np.clip(heat * (0.55 + 0.45 * np.clip(sd, -1, 1)) * (0.7 + 0.6 * nz) * (0.9 + 0.1 * math.sin(t * 13)), 0, 1.4)[..., None]
    col = Fw * (1 - 0.6 * hz.clip(0, 1)) + hz * (RED * 1.3 + GOLD * 0.55 * hz) * (0.5 + lum)
    if shine > 0:
        col = col + shine
    fr = fr * (1 - Aw[..., None]) + col * Aw[..., None]
    fr += cv2.GaussianBlur(Aw * hz[..., 0], (0, 0), 30)[..., None] * RED * 0.6 * heat
    return fr, Aw, M


class P13(Shot):
    T_DAO = V(56.50)
    def __init__(self):
        self.bg = BK.dark_bg(913, 0.01, 0.04, tint=(1.15, 1.0, 0.95))
        T.warm('刀', 'xing', 340); T.warm('还', 'kai', 90)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        heat = 0.25 * clamp01((t - self.T_DAO) / 0.8)
        slide = -120 * eio(u)
        fr, Aw, M = heat_blade(fr, t, (-200 + slide, 700), (2150 + slide, 560), 210, heat, seed=13)
        # 白刃光沿刃扫过（字头）
        q = clamp01((t - (self.T_DAO - 0.05)) / 0.3)
        if 0 < q < 1:
            x, y = K.grid()
            px = -200 + 2400 * q
            g = np.exp(-((x - px) / 120) ** 2) * Aw
            fr += g[..., None] * 1.4
            fr = T._slash_light(fr, (960, 630), np.array([0.998, -0.06]), q, 2600)
        # 「刀」被刀光斩出（左上，朱红）
        pd = clamp01((t - (self.T_DAO - 0.08)) / 0.2)
        if pd > 0:
            P = [dict(ch='刀', x=520, y=330, size=330, rot=-6, font='xing', idx=0, emph=True)]
            fr = T.slash_cut(fr, '刀', clamp01(0.2 + pd * 0.25 + (t - self.T_DAO) * 0.3), places=P, color='white', emph=('刀',),
                             emph_color=(0.84, 0.10, 0.07), cut_at=0.25, slide=0.12, angle=-20, appear=0.15)
        ph = tp(11, [5], t, 0.18)[0]
        if ph > 0:
            fr = T.ink_bleed_in(fr, '还', [ph], places=[dict(ch='还', x=1500, y=820, size=96, rot=0, font='kai', idx=0, emph=False)], color='white',
                                style_kw=dict(shadow=0.8))
        fr = K.sparks(fr, t, 1313, (1300, 640), n=40, life=0.8, speed=(80, 260), dirc=(0.2, -1), spread=0.7, grav=-50, area=(800, 60), amt=0.5 + heat)
        k = pulse(t, self.T_DAO, 0.1) * (t >= self.T_DAO)
        dx, dy, dr = shake(t, 14 * k, 10, 13)
        fr = cam(fr, 1.02 + 0.04 * k, dx, dy, dr)
        return fr, (4 if k > 0.5 else 0)


# =====================================================================================
# P14 ECU：刀身通体烧红、火星剥落、热浪；「滚烫」火烧显字在刀面；59.13 鼓停定格压暗；收成副歌一 S028 的横刀位置
class P14(Shot):
    T_HOLD = V(59.13)
    def __init__(self):
        self.bg = BK.dark_bg(914, 0.01, 0.035, tint=(1.2, 1.0, 0.95))
        self.tex = K.tile(512, 914, 6, 4)
        T.warm('滚烫', 'xing', 240)

    def render(self, t):
        tr = t
        if t > self.T_HOLD: t = self.T_HOLD + (t - self.T_HOLD) * 0.05      # 定格憋住
        u = clamp01((t - self.t0) / (self.T_HOLD - self.t0))
        fr = self.bg.copy()
        e = eio(u)
        # 机位：从斜向 ECU（刀身占满）→ 收成 S028 的横刀（(-200,760)→(2200,640) 宽 250）
        p0 = (np.array([-500, 1080]) * (1 - e) + np.array([-200, 760]) * e)
        p1 = (np.array([2500, 120]) * (1 - e) + np.array([2200, 640]) * e)
        wd = 520 * (1 - e) + 250 * e
        heat = 0.55 + 0.45 * clamp01((t - self.t0) / 0.9)
        fr, Aw, M = heat_blade(fr, t, tuple(p0), tuple(p1), wd, heat, seed=14)
        # 「滚烫」烧在刀面（随刀走：在刀面坐标里排字）
        pb = clamp01((t - (V(CT(11, 6)) - 0.25)) / 0.75)
        if pb > 0:
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            c = p0 + (p1 - p0) * 0.42
            ang = math.degrees(math.atan2(d[1], d[0]))
            s = wd * 0.62
            P = [dict(ch=ch, x=c[0] + d[0] * (i - 0.5) * s * 1.15, y=c[1] + d[1] * (i - 0.5) * s * 1.15, size=s, rot=-ang, font='xing', idx=i, emph=False)
                 for i, ch in enumerate('滚烫')]
            fr = T.burn_in(fr, '滚烫', pb, places=P, color='gold', embers=120, seed=14, style_kw=dict(shadow=0.3, glow=0.5))
        fr = K.sparks(fr, t, 1414, tuple(((p0 + p1) / 2).astype(int)), n=80, life=0.9, speed=(120, 420), dirc=(0.3, -1), spread=0.6, grav=-80,
                      area=(900, 120), amt=0.9)
        # 热浪扭曲
        x, y = K.grid()
        n = cv2.resize(K.tile_crop(self.tex, 0, -t * 160, H // 8, W // 8), (W, H))
        amp = 5 + 9 * heat
        fr = cv2.remap(fr, (x + (n - 0.5) * amp).astype(np.float32), (y + (n - 0.5) * amp * 1.6).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dx, dy, dr = shake(t, 6 * u, 13, 14)
        fr = cam(fr, 1.0, dx, dy, dr)
        if tr > self.T_HOLD:                     # 鼓停：压暗憋住
            fr = fr * (1 - 0.55 * clamp01((tr - self.T_HOLD) / 0.1))
        return fr, 0


SHOTS = [P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14]
for _S, _a, _b in zip(SHOTS, CUT_F, CUT_F[1:]):
    _S.t0, _S.t1 = _a / FPS, _b / FPS
P01.t0 = 44.048

CUTS = {
    'P02': ('flash', 0, 0, {}),                                     # 落墨帧闪白
    'P03': ('paper_tear', 0.06, 0.20, dict(seed=3, angle=10)),
    'P04': ('none', 0, 0, {}),                                      # 同一机位连续后拉（图形匹配）
    'P05': ('whip_pan_blur', 0.06, 0.12, dict(direction='right')),
    'P06': ('flashr', 0, 0, {}),
    'P07': ('flash_white', 0.0, 0.17, {}),                          # 雷闪
    'P08': ('whip_pan_blur', 0.05, 0.10, dict(direction='up')),
    'P09': ('glitch_cut', 0.03, 0.12, dict(seed=9)),
    'P10': ('slash_wipe', 0.05, 0.14, dict(angle=-28)),
    'P11': ('flash', 0, 0, {}),
    'P12': ('match_cut', 0.0, 0.12, dict(shape='rect', center=(960, 470), radius=140)),
    'P13': ('flash_red', 0.03, 0.12, {}),
    'P14': ('hard', 0, 0, {}),
}
FG = {'P05': ('rain', 0.35), 'P07': ('rain', 0.45), 'P08': ('ash', 0.25), 'P09': ('ember', 0.35), 'P10': ('ash', 0.25),
      'P12': ('ember', 0.35), 'P13': ('ember', 0.5), 'P14': ('ember', 0.7)}
EXPO = {'P01': 1.05, 'P02': 1.0, 'P03': 1.0, 'P04': 1.0, 'P07': 1.1, 'P12': 1.1}

_SEC = [None]
_C1 = [None]


def section():
    if _SEC[0] is None: _SEC[0] = Section(SHOTS, CUTS, FG, EXPO, expo_all=1.12)
    return _SEC[0]


def frame(f, cache=None):
    if f >= C1_FROM:                            # 59.30 起 = 副歌一自身（S028「血」字头硬切）
        if _C1[0] is None:
            os.environ.setdefault('SPRITE_MODE', '1')
            import chorus1 as C; _C1[0] = (C, C.Chorus1())
        C, X = _C1[0]
        return C.to8(X.frame(f / FPS)[0])
    return section().frame(f)
