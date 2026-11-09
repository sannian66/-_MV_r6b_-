"""units/sec_c_pre2_chorus2.py — 第四轮 C 组：预副歌二 + 副歌二「破」（帧 3534–4839，117.795–161.309s）
设计说明：crew/round4/C_storyboard.md（逐镜表 + 接缝）。41 镜，平均 1.06s/镜（预副歌二 15 镜 1.11s，副歌二 26 镜 1.03s）。
接口：READY / frame(f, cache) -> uint8 (1080,1920,3)（已收尾：调色、辉光、色差、暗角、颗粒、2.39 黑边）。
组内自测：V4_FORCE=c_pre2_chorus2 python3 render_sprite_full.py frame 140.0 /tmp/x.png
视觉语言（与副歌一「月夜·砖墙·单色」区分）：预副歌二 = 烛光暗房里的「剧本 / 提线 / 暗箱」（暖金、宋体铅字打字机）→
「我不低头」起切进雨夜；副歌二 = 雷暴（电白 + 朱红血雨），字墙 = 宋体律令砌成，巨印、巨掌、天平，不出现月亮与金裂砖墙。"""
import os, sys, math
ROOT = '/workspace/work/mv'
for p in (ROOT, ROOT + '/lib', ROOT + '/units'):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, cv2
import c_kit as K
from c_kit import (W, H, FPS, BAR, RED, GOLD, COLD, WARM, clamp, sm, eout, ein, eio, lerp, V, pulse, CT, camxf, handheld,
                   dirblur, radial, storm_sky, wet_ground, rain, fog, bolt, bolt_env, vol_light, put, Sent, draw_sent,
                   paste_glyph, glyph_layer, RED_T, GOLD_T, WHITE_T, CREAM)
import typo as T, trans as TR, fx2
import props as PR
from c1kit import finish, to8, sparks, dust, crack_path, draw_crack, paper_dark, fbm

READY = False
F0, F1 = 3534, 4839
T0, T1 = F0 / FPS, F1 / FPS


# =====================================================================================
# 共用：烛光暗房（预副歌二前半）
def candle_room(t, seed=0, glow_c=(1300, 300), amt=1.0):
    fr = paper_dark(31).copy() * 0.55
    fr = K.vol_light(fr, glow_c, 900, WARM * 0.5, 0.35 * amt * (1 + 0.08 * math.sin(t * 23) + 0.05 * math.sin(t * 9.1)))
    return fr


def storm(t, seed=0, horizon=780, flash=0.0, fc=(960, 160), ox=0.0, oy=0.0, refl=0.5, cloud=1.0, ground=True, rip=1.0):
    fr = storm_sky(t, seed, horizon, flash, fc, ox=ox, oy=oy, cloud=cloud)
    if ground: fr = wet_ground(fr, t, horizon, seed, refl=refl * (1 + flash), ox=ox, ripples=rip)
    return fr


# ---------------- 纸人（提线木偶）
_PUP = {}
def puppet_sprite(seed, h=300):
    k = (seed, h)
    if k in _PUP: return _PUP[k]
    r = np.random.default_rng(seed)
    w = int(h * 0.6); A = np.zeros((h, w), np.float32); cx = w // 2
    hr = int(h * 0.075)
    cv2.circle(A, (cx, int(h * 0.12)), hr, 1.0, -1, cv2.LINE_AA)
    body = np.array([[cx - h * 0.09, h * 0.2], [cx + h * 0.09, h * 0.2], [cx + h * 0.2, h * 0.78], [cx - h * 0.2, h * 0.78]], np.int32)
    cv2.fillPoly(A, [body], 1.0, cv2.LINE_AA)
    sw = r.uniform(-0.15, 0.15)
    for s in (-1, 1):
        cv2.line(A, (int(cx + s * h * 0.08), int(h * 0.24)), (int(cx + s * h * (0.22 + sw * s)), int(h * 0.52)), 1.0, max(2, int(h * 0.035)), cv2.LINE_AA)
        cv2.line(A, (int(cx + s * h * 0.07), int(h * 0.76)), (int(cx + s * h * 0.1), int(h * 0.98)), 1.0, max(2, int(h * 0.04)), cv2.LINE_AA)
    # 纸纹 + 毛边
    nz = cv2.resize(r.random((h // 6 + 1, w // 6 + 1)).astype(np.float32), (w, h))
    A = np.clip(A * (0.85 + 0.25 * nz), 0, 1)
    A = cv2.GaussianBlur(A, (0, 0), 0.8)
    fib = cv2.GaussianBlur(r.random((h, w)).astype(np.float32), (0, 0), 1.0)
    _PUP[k] = (A, fib)
    return _PUP[k]


def draw_puppet(fr, A, fib, cx, top, ang, col=(0.92, 0.80, 0.58), back=1.0, alpha=1.0):
    h, w = A.shape
    M = cv2.getRotationMatrix2D((w / 2, 0), ang, 1.0); M[0, 2] += cx - w / 2; M[1, 2] += top
    a = cv2.warpAffine(A, M, (W, H)) * alpha
    f = cv2.warpAffine(fib, M, (W, H))
    c = np.asarray(col, np.float32) * (0.45 + 0.55 * f[..., None]) * back
    return fr * (1 - a[..., None]) + c * a[..., None]


def red_line(fr, p0, p1, w=2, amt=1.0, glow=0.6):
    L = np.zeros((H // 2, W // 2), np.float32)
    cv2.line(L, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)), 1.0, max(1, int(w / 2)), cv2.LINE_AA, 3)
    g = cv2.GaussianBlur(L, (0, 0), 4)
    lay = cv2.resize(L + g * glow, (W, H))
    return fr * (1 - np.clip(cv2.resize(L, (W, H)), 0, 1)[..., None] * 0.9 * amt) + lay[..., None] * RED * 1.2 * amt


def slash_streak(fr, p0, p1, q, width=6, col=(1.0, 0.97, 0.92), amt=1.6):
    """刀光（白刃 + 辉光），q 0..1 生长，之后衰减"""
    if q <= 0: return fr
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    grow = min(1.0, q * 3); fade = 1 - sm(0.35, 1.0, q)
    e = p0 + (p1 - p0) * grow
    L = np.zeros((H // 2, W // 2), np.float32)
    cv2.line(L, tuple((p0 * 4).astype(int)), tuple((e * 4).astype(int)), 1.0, max(1, int(width / 2)), cv2.LINE_AA, 3)
    g1 = cv2.GaussianBlur(L, (0, 0), 3); g2 = cv2.GaussianBlur(L, (0, 0), 14)
    lay = cv2.resize(L * 1.5 + g1 * 1.2 + g2 * 1.4, (W, H))
    return fr + lay[..., None] * np.asarray(col, np.float32) * amt * fade


def arc_streak(fr, c, rx, ry, a0, a1, q, width=8, col=(1, 0.97, 0.92), amt=1.6, rot=0):
    if q <= 0: return fr
    fade = 1 - sm(0.4, 1.0, q); ae = a0 + (a1 - a0) * min(1, q * 2.5)
    L = np.zeros((H // 2, W // 2), np.float32)
    n = 40; th = np.radians(np.linspace(a0, ae, n))
    pts = np.stack([c[0] + rx * np.cos(th), c[1] + ry * np.sin(th)], 1)
    if rot:
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        d = pts - c; pts = c + np.stack([d[:, 0] * cr - d[:, 1] * sr, d[:, 0] * sr + d[:, 1] * cr], 1)
    P = (pts / 2 * 8).astype(np.int32)
    for i in range(n - 1):
        wv = max(1, int(width / 2 * (0.3 + 0.7 * i / n)))
        cv2.line(L, tuple(P[i]), tuple(P[i + 1]), 1.0, wv, cv2.LINE_AA, 3)
    g = cv2.GaussianBlur(L, (0, 0), 3); g2 = cv2.GaussianBlur(L, (0, 0), 16)
    lay = cv2.resize(L * 1.4 + g + g2 * 1.3, (W, H))
    return fr + lay[..., None] * np.asarray(col, np.float32) * amt * fade


def shock_ring(fr, c, q, rx=900, flat=0.18, col=COLD, amt=1.0, n=3):
    if q <= 0 or q >= 1: return fr
    L = np.zeros((H // 2, W // 2), np.float32)
    for i in range(n):
        qq = q - i * 0.12
        if qq <= 0: continue
        r = rx * eout(qq, 2.2)
        cv2.ellipse(L, (int(c[0] / 2), int(c[1] / 2)), (int(r / 2), max(1, int(r * flat / 2))), 0, 0, 360, (1 - qq) * (1 - 0.3 * i), 3, cv2.LINE_AA)
    g = cv2.GaussianBlur(L, (0, 0), 3)
    return fr + cv2.resize(L + g * 1.5, (W, H))[..., None] * np.asarray(col, np.float32) * amt


def splash(fr, t, t0, c, seed, n=140, spread=520, up=900, col=COLD, amt=0.9):
    return sparks(fr, t, seed, c, n=n, life=0.9, speed=(up * 0.4, up), dirc=(0, -1), spread=0.9, grav=2200, size=1, amt=amt,
                  t0=t0, burst=True, color=tuple(col), area=(spread * 0.5, 6))


def flash_frame(fr, amt, col=(0.92, 0.95, 1.0)):
    return fr * (1 + amt * 0.6) + amt * np.asarray(col, np.float32) * 0.5


def ink_vignette(fr, amt):
    yy, xx = np.ogrid[0:H // 4, 0:W // 4]
    d = np.sqrt(((xx * 4 - W / 2) / (W / 2)) ** 2 + ((yy * 4 - H / 2) / (H / 2.3)) ** 2)
    m = np.clip((d - (1.3 - amt * 1.3)) * 2.5, 0, 1).astype(np.float32)
    nz = cv2.resize(K.tex('inkv', seed=5, base=8), (W // 4, H // 4))
    m = np.clip(m * (0.7 + 0.6 * nz), 0, 1)
    return fr * (1 - cv2.resize(m, (W, H))[..., None])


# =====================================================================================
# 预副歌二
class Shot:
    t0 = t1 = 0.0; ca = 0
    def u(self, t): return clamp((t - self.t0) / (self.t1 - self.t0))


class C01(Shot):
    """CU 剧本：烛光下一页剧本，宋体铅字打出标题「谁定好规则」（字在纸上），太上皇执朱笔圈出「规则」。"""
    def __init__(self):
        r = np.random.default_rng(1)
        pw, ph = 1500, 980; self.pw, self.ph = pw, ph
        n = fbm(ph, pw, 77, base=60, octaves=4)
        fib = cv2.GaussianBlur(r.random((ph, pw)).astype(np.float32), (0, 0), 0.7)
        l = 0.80 + 0.08 * n + 0.05 * fib
        self.paper = np.stack([l * 0.98, l * 0.90, l * 0.74], -1).astype(np.float32)
        # 剧本正文（小号宋体，已印好，暗）
        L = T.Layer(ph, pw)
        body = ['第一回　大局', '众人各安其位　不得僭越', '凡逆命者　以罪论处', '一人之死　可换千人之安', '此乃天理　理所应当',
                '钦定　剧本提纲　不得改动', '违者　斩']
        for j, line in enumerate(body):
            for i, c in enumerate(line):
                if c.strip(): T.place(L, c, 1300 - j * 120, 330 + i * 44, 34, (0.10, 0.08, 0.07), 'serif_r', texture=False)
        self.body = L
        for c in '谁定好规则': T.glyph(c, 'serif_b', 104, False)

    def render(self, t):
        u = self.u(t)
        P = self.paper.copy()
        P = T.compose(P, self.body, shadow=0, glow=0)
        # 标题打字机（铅字敲下 + 光标）
        s = Sent(28, [0, 1, 2, 3, 4], 640, 200, size=104, mode='type', red='规则', glitch='', font='serif_b', out=99)
        L = T.Layer(self.ph, self.pw)
        n = 0
        for i, (c, (px, py)) in enumerate(zip(s.text, s.P)):
            if t < s.tin[i]: continue
            n += 1
            k = t - s.tin[i]
            col = (0.62, 0.06, 0.04) if c in '规则' else (0.07, 0.06, 0.05)
            T.place(L, c, px, py + 3 * pulse(t, s.tin[i], 0.04), 104, col, 'serif_b', scale=1 + 0.08 * pulse(t, s.tin[i], 0.05), texture=False)
        if n < 5 or int(t * 6) % 2 == 0:
            j = max(0, n - 1); px, py = s.P[j]
            L.paste(np.ones((92, 12), np.float32), px + (70 if n else -60), py, (0.7, 0.05, 0.03), 0.9)
        P = T.compose(P, L, shadow=0.0, glow=0.0)
        # 朱笔圈「规则」：118.62–118.95 一笔椭圆
        q = sm(118.58, 118.95, t)
        cx, cy = (s.P[3][0] + s.P[4][0]) / 2, s.P[3][1]
        if q > 0:
            M = np.zeros((self.ph, self.pw), np.float32)
            th = np.radians(np.linspace(-200, -200 + 380 * q, 60))
            pts = np.stack([cx + 165 * np.cos(th), cy + 78 * np.sin(th) + np.linspace(0, 14, 60)], 1).astype(np.int32)
            cv2.polylines(M, [pts], False, 1.0, 9, cv2.LINE_AA)
            M = cv2.GaussianBlur(M, (0, 0), 1.2)
            P = P * (1 - M[..., None] * 0.92) + M[..., None] * np.array([0.60, 0.05, 0.03], np.float32)
            tip = pts[-1]
        else:
            tip = np.array([cx + 260 - 120 * sm(117.8, 118.58, t), cy - 140 + 60 * sm(117.8, 118.58, t)])
        # 纸面 → 画面（透视，俯 30°），慢推 + 右移
        z = 1.0 + 0.10 * eio(u); sx = -40 + 90 * eio(u)
        q0 = np.float32([[0, 0], [self.pw, 0], [self.pw, self.ph], [0, self.ph]])
        dst = np.float32([[230, 140], [1770, 170], [1960, 1010], [-60, 990]])
        dst = ((dst - [960, 540]) * z + [960 - sx, 540]).astype(np.float32)
        Mh = cv2.getPerspectiveTransform(q0, dst)
        fr = candle_room(t) * 0.6
        pw_ = cv2.warpPerspective(P, Mh, (W, H), flags=cv2.INTER_LINEAR)
        A = cv2.warpPerspective(np.ones((self.ph, self.pw), np.float32), Mh, (W, H))
        # 烛光：右上暖光，左下暗
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
        light = np.exp(-(((xx - 1500) ** 2 + (yy - 150) ** 2) / 1100 ** 2)) * (1.0 + 0.06 * math.sin(t * 21)) + 0.12
        light = cv2.resize(light.astype(np.float32), (W, H))[..., None]
        fr = fr * (1 - A[..., None]) + pw_ * A[..., None] * light * np.array([1.0, 0.93, 0.80], np.float32)
        # 朱笔手：笔尖对准 tip（纸面坐标 → 画面）
        tp = cv2.perspectiveTransform(np.float32([[tip]]).reshape(1, 1, 2), Mh)[0, 0]
        # 手的投影
        sh = np.zeros((H // 4, W // 4), np.float32)
        cv2.ellipse(sh, (int((tp[0] + 160) / 4), int((tp[1] + 150) / 4)), (110, 60), -30, 0, 360, 1.0, -1)
        fr = fr * (1 - cv2.resize(cv2.GaussianBlur(sh, (0, 0), 12), (W, H))[..., None] * 0.55)
        fr, _ = put(fr, t, 'C/tsh_brush', tp[0], tp[1], 900, anchor='src', src=(0.635, 0.83), rot=-8, rim=WARM, rim_amt=1.4,
                    rim_dir=(0.8, -0.6), expo=0.95, sat=0.55, tint=(1.05, 0.96, 0.86), gamma=1.05, wind=0, breathe=0.0)
        fr = sparks(fr, t, 11, (1500, 200), n=30, life=2.5, speed=(10, 40), dirc=(0, -1), spread=1.2, grav=-10, amt=0.25, area=(500, 200))
        return fr


class C02(Shot):
    """MS 提线：太上皇的手在画顶，朱红丝线垂下；歌词「谁写好剧本提」一字一线挂在丝线上随线摆（字即提线木偶）。"""
    XS = [520, 760, 980, 1190, 1400, 1610]
    def render(self, t):
        u = self.u(t)
        oy = lerp(70, -40, eio(u))
        fr = candle_room(t, glow_c=(960, 120), amt=1.3) * 0.8
        fr = fog(fr, t, 3, density=0.35, y0=200, y1=900, color=(0.30, 0.22, 0.15), speed=(12, -4))
        hand_y = 70 - oy * 0.6
        tips = [(830, hand_y + 300), (900, hand_y + 330), (980, hand_y + 340), (1060, hand_y + 320), (1120, hand_y + 280), (790, hand_y + 250)]
        s = Sent(28, [5, 6, 7, 8, 9, 10], 0, 0, size=70, mode='type', red='', glitch='剧本')
        for i, x in enumerate(self.XS):
            ti = s.tin[i]
            sw = 9 * math.sin(t * 2.2 + i * 1.3) + 22 * pulse(t, ti, 0.25) * (1 if i % 2 else -1)
            ly = 560 + 90 * math.sin(i * 1.7) - oy * 0.4
            jerk = -40 * pulse(t, ti, 0.12)
            end = (x + sw, ly + jerk - 50)
            fr = red_line(fr, tips[i], end, w=2, amt=0.9)
            if t >= ti:
                c = s.text[i]
                L = glyph_layer(c, end[0], end[1] + 60, 72, CREAM if c not in '剧本' else (1.0, 0.86, 0.6), font='serif_sb',
                                rot=-sw * 0.6, scale=1 + 0.15 * pulse(t, ti, 0.06))
                fr = T.compose(fr, L, shadow=0.6, shadow_sigma=14, glow=0.25, glow_sigma=10)
                if c in '剧本':
                    off = 7 * pulse(t, ti, 0.3) + 3
                    for dx, cc in ((-off, (1, 0.1, 0.1)), (off, (0.1, 0.6, 1.0))):
                        L2 = glyph_layer(c, end[0] + dx, end[1] + 60, 72, cc, font='serif_sb', rot=-sw * 0.6, alpha=0.5)
                        fr = T.compose(fr, L2, shadow=0, glow=0)
        fr, _ = put(fr, t, 'S062/puppeteer', 960, hand_y, 640, anchor='src', src=(0.5, 0.06), rim=WARM, rim_amt=2.2, rim_dir=(0, -1),
                    expo=0.8, sat=0.4, tint=(1.0, 0.92, 0.82), wind=2)
        return fr


class C03(Shot):
    """LS 仰拍：一排纸人木偶被红线吊着，在「纲」字上同时被拽起；背后是皮影幕上巨手的影子。"""
    def render(self, t):
        u = self.u(t)
        fr = candle_room(t, glow_c=(960, 420), amt=1.6) * 0.9
        # 皮影幕（暖光纸幕）+ 巨手影
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
        scr = np.exp(-(((xx - 960) / 1100) ** 2 + ((yy - 420) / 420) ** 2)) * 0.55
        fr = fr + cv2.resize(scr.astype(np.float32), (W, H))[..., None] * np.array([1.0, 0.72, 0.42], np.float32)
        fr, _ = put(fr, t, 'S062/puppeteer', 960 + 40 * math.sin(t), 120 - 60 * pulse(t, V(120.96), 0.2), 760, anchor='src', src=(0.5, 0.1),
                    look='shadow', alpha=0.85, wind=0)
        tj = V(120.96); tj2 = V(121.31)
        lift = 70 * pulse(t, tj, 0.22) + 40 * pulse(t, tj2, 0.18)
        r = np.random.default_rng(3)
        for i in range(9):
            x = 170 + i * 200 + r.uniform(-30, 30); hgt = 300 + r.uniform(-20, 20)
            top = 470 - lift * (1 + 0.2 * math.sin(i)) + 10 * math.sin(t * 3 + i)
            ang = 6 * math.sin(t * 2.5 + i) + 10 * pulse(t, tj, 0.3) * (1 if i % 2 else -1)
            A, fib = puppet_sprite(i, int(hgt))
            fr = red_line(fr, (x, BAR), (x, top + 4), w=2, amt=0.85)
            fr = draw_puppet(fr, A, fib, x, top, ang, back=1.1 - 0.4 * abs(i - 4) / 4)
        # 「纲」：宋体铅字压在纸人群上方，纲常之「纲」，朱红块光标
        if t >= V(120.96) - 0.0:
            k = t - V(120.96)
            fr = paste_glyph(fr, '纲', 960, 330, 300, (0.95, 0.85, 0.62), font='serif_b', glow=0.35, glow_c=(1, 0.6, 0.3),
                             scale=1 + 0.12 * pulse(t, V(120.96), 0.06))
        fr = fog(fr, t, 4, density=0.4, y0=750, y1=980, color=(0.25, 0.17, 0.10), speed=(30, 0))
        return fr


class C04(Shot):
    """CU 天平：金印落进右盘（121.72），左盘纸人被翘起抛飞；只有烛光金边，深景深。"""
    def __init__(self): self.B = PR.Balance(seed=5, people=50)
    def render(self, t):
        u = self.u(t)
        fr = candle_room(t, glow_c=(1500, 250), amt=1.2) * 0.7
        drop = sm(V(121.72) - 0.18, V(121.72), t)
        tilt = -14 * eout((t - V(121.72)) / 0.35) if t > V(121.72) else 0
        tilt += 2.5 * math.sin(t * 7) * pulse(t, V(121.72), 0.4)
        fr, info = self.B.draw(fr, t, 960, 330, 1350, tilt=tilt, sway=0.5, left_people=1.0 - 0.6 * sm(V(121.72), V(122.3), t),
                               right='seal', rim_color=GOLD, rim=1.6, light=(0.5, -1))
        # 金印坠落（落进右盘前的残影）
        if t < V(121.72):
            y = lerp(-80, 560, ein(drop, 2))
            fr = K.vol_light(fr, (1620, y), 70, GOLD, 0.8)
        fr = sparks(fr, t, 21, (1640, 640), n=60, life=0.7, speed=(150, 600), dirc=(0, -1), spread=1.0, grav=900, amt=0.9,
                    t0=V(121.72), burst=True, color=(1.0, 0.75, 0.35))
        # 抛飞的纸人（剪影碎纸）
        if t > V(121.72):
            k = t - V(121.72); r = np.random.default_rng(9)
            for i in range(14):
                A, fib = puppet_sprite(20 + i, 70)
                x = 300 + r.uniform(-120, 140) + r.uniform(-300, 200) * k; y = 560 - (700 * k - 900 * k * k) * r.uniform(0.5, 1.2)
                fr = draw_puppet(fr, A, fib, x, y, r.uniform(-180, 180) * k * 3, back=0.9)
        fr = cv2.GaussianBlur(fr, (0, 0), 0.6)
        fr = K.vol_light(fr, (960, 330), 120, GOLD, 0.25 + 0.6 * pulse(t, V(121.98), 0.12))   # 「正」金闪
        return fr


class C05(Shot):
    """ECU 暗箱：黑漆金边方箱张着口，金印、铜钱、纸人倾泻而入，镜头推向箱口（交易）。"""
    def render(self, t):
        u = self.u(t)
        fr = candle_room(t, glow_c=(960, 160), amt=1.0) * 0.55
        z = 1.0 + 0.22 * ein(u, 2)
        cx, cy = 960, 640
        def P(x, y): return (int(cx + (x - cx) * z), int(cy + (y - cy) * z))
        fl, fr_, br, bl = P(560, 600), P(1360, 600), P(1250, 470), P(670, 470)        # 箱口（顶面）
        dl, dr = P(560, 1000), P(1360, 1000)
        L = np.zeros((H, W), np.float32)
        box = fr.copy()
        cv2.fillPoly(box, [np.array([fl, fr_, dr, dl])], (0.035, 0.03, 0.03), cv2.LINE_AA)               # 前面
        cv2.fillPoly(box, [np.array([fl, fr_, br, bl])], (0.0, 0.0, 0.0), cv2.LINE_AA)                     # 箱口（黑洞）
        g = np.zeros((H, W), np.float32)
        for a, b in ((fl, fr_), (fr_, br), (br, bl), (bl, fl), (fl, dl), (fr_, dr)):
            cv2.line(g, a, b, 1.0, 3, cv2.LINE_AA)
        fr = box + (g + cv2.GaussianBlur(g, (0, 0), 6) * 0.8)[..., None] * GOLD * 0.9
        # 箱面漆光：烛光高光条
        hl = np.zeros((H, W), np.float32); cv2.line(hl, P(600, 640), P(1320, 640), 1.0, 14, cv2.LINE_AA)
        fr = fr + cv2.GaussianBlur(hl, (0, 0), 18)[..., None] * WARM * 0.12
        # 倾泻物：金币（椭圆金片）+ 纸人
        r = np.random.default_rng(13)
        for i in range(70):
            ph = r.random(); sp = r.uniform(0.6, 1.0)
            k = ((t * sp * 1.4 + ph) % 1.0)
            x = 960 + r.uniform(-260, 260) * (1 - k * 0.6); y = lerp(150, 560, k ** 1.6)
            x, y = P(x, y)
            if r.random() < 0.75:
                sz = int(r.uniform(6, 13) * z); ang = (t * 300 * sp + i * 40) % 180
                cv2.ellipse(fr, (x, y), (sz, max(1, int(sz * abs(math.cos(math.radians(ang)))))), i * 20, 0, 360, tuple(float(v) for v in GOLD * 0.9), -1, cv2.LINE_AA)
            else:
                A, fib = puppet_sprite(40 + i % 6, int(60 * z))
                fr = draw_puppet(fr, A, fib, x, y, (t * 200 + i * 50) % 360, back=0.8)
        fr = K.vol_light(fr, (960, 560), 260, GOLD, 0.25)
        return fr


class C06(Shot):
    """ECU 俯拍：箱盖砸下合上（124.22「箱」），黑屏故障帧；「暗箱」二字被关进黑暗后碎裂。"""
    def render(self, t):
        tc = V(124.22)
        fr = np.zeros((H, W, 3), np.float32) + 0.012
        if t < tc:
            q = ein(sm(self.t0, tc, t), 2.4)
            fr = candle_room(t, glow_c=(960, 540), amt=1.0) * 0.5
            # 俯拍箱口：黑色方洞 + 金边，箱盖从上方压下（遮挡比例 q）
            cv2.rectangle(fr, (420, 230), (1500, 960), (0.0, 0.0, 0.0), -1)
            g = np.zeros((H, W), np.float32); cv2.rectangle(g, (420, 230), (1500, 960), 1.0, 4, cv2.LINE_AA)
            fr = fr + (g + cv2.GaussianBlur(g, (0, 0), 8))[..., None] * GOLD * 0.8
            yl = int(lerp(-500, 960, q))
            lid = np.zeros((H, W), np.float32); cv2.rectangle(lid, (380, -800), (1540, yl), 1.0, -1)
            fr = fr * (1 - lid[..., None]) + lid[..., None] * 0.03
            gl = np.zeros((H, W), np.float32); cv2.line(gl, (380, yl), (1540, yl), 1.0, 5, cv2.LINE_AA)
            fr = fr + (gl + cv2.GaussianBlur(gl, (0, 0), 10))[..., None] * GOLD
            # 「暗箱」金字在箱口里（等着被关进去）
            fr = paste_glyph(fr, '暗', 760, 600, 260, GOLD_T, font='serif_b', glow=0.6, glow_c=(1, 0.6, 0.25))
            fr = paste_glyph(fr, '箱', 1160, 600, 260, GOLD_T, font='serif_b', glow=0.6, glow_c=(1, 0.6, 0.25),
                             alpha=sm(V(123.86) - 0.08, V(123.86) + 0.02, t) * 0 + 1.0)
            self.ca = 0
            return fr
        k = t - tc
        # 合盖后：黑 + 故障条 + 「暗箱」在黑里碎裂
        L = T.Layer()
        T.place(L, '暗', 760, 600, 260, GOLD_T, 'serif_b', texture=False); T.place(L, '箱', 1160, 600, 260, GOLD_T, 'serif_b', texture=False)
        q = sm(0.12, 0.6, k)
        L2 = T.shatter_layer(L, q, (960, 600), 60, seed=3, power=1.2, gravity=200) if q > 0 else L
        out = T.compose(fr, L2, shadow=0, glow=0.5, glow_sigma=14, glow_color=(1, 0.5, 0.2))
        if k < 0.1:
            out = TR.glitch_cut(out, np.zeros_like(out), 0.4, seed=int(t * 30))
            out = flash_frame(out, 0.6, (1, 0.8, 0.5))
        r = np.random.default_rng(int(t * 30))
        if r.random() < 0.35 * (1 - q):
            yb = int(r.uniform(200, 900)); out[yb:yb + int(r.uniform(4, 30))] += 0.08
        self.ca = 6 if k < 0.15 else 0
        return out


class C07(Shot):
    """MCU 雨夜：鬼屠在雨里抬起下颌（S065），闪电从背后勾边——「我不低头」从这里起，字体从铅字换成他自己的重字。"""
    def render(self, t):
        u = self.u(t)
        tf = V(124.90)
        fl = bolt_env(t, tf) * 0.9 + 0.4 * bolt_env(t, V(125.46))
        fr = storm(t, 7, horizon=1200, flash=0.5 + fl, fc=(1500, 120), ox=30 * u, ground=False)
        fr = bolt(fr, t, tf, 71, (1650, 40), (1500, 700), inten=0.8, sky=0.3, width=2.5)
        fr = rain(fr, t, 70, inten=0.6, angle=14, layers=(0, 1), light=(1500, 300))
        fr, _ = put(fr, t, 'S065/chin_up', 1280, 1100, 1080, rim=COLD, rim_amt=2.6, rim_dir=(0.6, -0.8), flash=fl, expo=0.95, wind=8,
                    tint=(0.92, 0.97, 1.06))
        fr = rain(fr, t, 72, inten=0.9, angle=14, layers=(2, 3), light=(1500, 300))
        return fr


class C08(Shot):
    """ECU 同轴跳切进眼睛：雨水顺睫毛，闪电在瞳孔里一亮（「命」126.14 推）。"""
    def render(self, t):
        u = self.u(t)
        tf = V(126.14)
        fl = bolt_env(t, tf)
        fr = storm(t, 8, horizon=1200, flash=0.3 + fl * 0.7, ground=False)
        z = 1.0 + 0.05 * pulse(t, tf, 0.15)
        fr, _ = put(fr, t, 'S065/chin_up', 980, 470, 4200 * z, anchor='src', src=(0.535, 0.27), rim=COLD, rim_amt=2.0,
                    rim_dir=(0.6, -0.8), flash=fl * 0.7, expo=1.0, wind=3, breathe=0.0)
        fr = K.vol_light(fr, (1040, 470), 40, COLD, 1.6 * fl)                # 瞳中电光
        fr = rain(fr, t, 80, inten=1.0, angle=14, layers=(2, 3))
        return fr


class C09(Shot):
    """LS 仰拍：鬼屠单手上撩（新精灵 slash_strings），头顶数十根红线在每个「不」上被斩断回弹。"""
    def render(self, t):
        u = self.u(t)
        cuts = [V(126.32), V(127.00)]
        fl = bolt_env(t, cuts[0]) * 0.8 + bolt_env(t, cuts[1]) * 0.5
        fr = storm(t, 9, horizon=930, flash=0.4 + fl, fc=(700, 100), oy=-40 * u, refl=0.6)
        fr = bolt(fr, t, cuts[0], 91, (600, 30), (420, 760), inten=0.8)
        r = np.random.default_rng(5)
        for i in range(26):
            x = 150 + i * 65 + r.uniform(-20, 20)
            ci = 0 if x < 960 else 1
            tc = cuts[ci]
            y_end = 520 + r.uniform(-80, 140)
            if t < tc:
                fr = red_line(fr, (x + 30 * math.sin(i), BAR), (x + 6 * math.sin(t * 3 + i), y_end), w=2, amt=0.8)
            else:
                k = t - tc; rec = 300 * eout(k / 0.4) + 400 * k
                fr = red_line(fr, (x + 30 * math.sin(i), BAR), (x + 30 * math.sin(i * 2) * k, max(BAR + 2, y_end - 160 - rec)), w=2, amt=0.8)
                yb = y_end + 900 * k * k
                fr = red_line(fr, (x - 40 * k, y_end - 140 + 900 * k * k), (x - 80 * k, yb + 120), w=2, amt=0.7 * (1 - sm(0.3, 0.9, k)))
        fr, _ = put(fr, t, 'C/slash_strings', 980, 1015, 760, rim=COLD, rim_amt=2.4, rim_dir=(-0.5, -0.9), flash=fl, expo=0.95, wind=7)
        for tc, (a, b) in zip(cuts, (((520, 620), (1050, 260)), ((900, 280), (1500, 560)))):
            fr = slash_streak(fr, a, b, (t - tc) / 0.35, width=8)
            fr = sparks(fr, t, int(tc * 10), b, n=50, life=0.6, speed=(200, 700), dirc=(0, 1), spread=1.4, grav=900, t0=tc, burst=True,
                        color=(1.0, 0.35, 0.2))
        fr = rain(fr, t, 90, inten=0.9, angle=-8)
        return fr


class C10(Shot):
    """CU：宋体「大局为上」四块铅字牌挂在红线上（字即物体）；127.86 一道横刀光斩断，四字坠入积水。"""
    def render(self, t):
        u = self.u(t)
        tc = V(127.86)
        fl = bolt_env(t, tc) * 0.6
        fr = storm(t, 10, horizon=840, flash=0.35 + fl, ox=-40 * u, refl=0.7, rip=1.5)
        xs = [600, 840, 1080, 1320]; tins = [V(CT(30, k)) for k in (11, 12, 13, 14)]
        for i, (x, ti) in enumerate(zip(xs, tins)):
            if t < ti - 0.02 and t < tc: continue
            sw = 6 * math.sin(t * 2.4 + i)
            if t < tc:
                y = 430 + 30 * math.sin(i * 2.1) - 30 * pulse(t, ti, 0.1)
                fr = red_line(fr, (x, BAR), (x + sw, y - 105), w=2, amt=0.9)
                ang = sw * 0.5
            else:
                k = t - tc
                y = 430 + 30 * math.sin(i * 2.1) + 1300 * k * k * (0.8 + 0.1 * i)
                ang = sw * 0.5 + (40 + 30 * i) * k * (1 if i % 2 else -1)
                fr = red_line(fr, (x, BAR), (x + sw * 3 * k, BAR + 240 - 400 * k), w=2, amt=0.9 * (1 - sm(0.2, 0.6, k)))
            if y > 800: 
                if t > tc: fr = splash(fr, t, tc + math.sqrt(max(0, (800 - 430) / 1300 / (0.8 + 0.1 * i))), (x, 800), 40 + i, n=60, spread=150, up=500)
                continue
            # 铅字牌：金属块（暗）+ 金字
            L = T.Layer(); a = np.ones((190, 190), np.float32)
            T.place(L, ' ', x, y, 10, CREAM)
            M = cv2.getRotationMatrix2D((95, 95), ang, 1.0); a = cv2.warpAffine(a, M, (190, 190))
            L.paste(a * 0.92, x, y, (0.06, 0.055, 0.05))
            fr = T.compose(fr, L, shadow=0.7, shadow_sigma=18, glow=0)
            fr = paste_glyph(fr, '大局为上'[i], x, y, 150, GOLD_T, font='serif_b', glow=0.35, glow_c=(1, 0.6, 0.25), rot=ang, shadow=0)
        fr = slash_streak(fr, (300, 330), (1700, 300), (t - tc) / 0.35, width=7)
        fr = rain(fr, t, 100, inten=0.9, angle=8)
        return fr


class C11(Shot):
    """CU 横刀：刀身烧得通红（生图 blade_hot），雨点打上去瞬间化作白汽；镜头沿刀身横移。"""
    def render(self, t):
        u = self.u(t)
        fr = storm(t, 11, horizon=1300, flash=0.25, ground=False) * 0.8
        x = lerp(1060, 860, eio(u))
        fr, A = put(fr, t, 'C/blade_hot', x, 560, 2100, anchor='src', src=(0.5, 0.5), rim=COLD, rim_amt=1.2, expo=1.05, sat=1.0,
                    tint=(1.04, 0.98, 0.95), gamma=1.0, wind=0, breathe=0.0)
        heat = 0.8 + 0.2 * math.sin(t * 7)
        fr = K.vol_light(fr, (x + 200, 560), 700, (1.0, 0.35, 0.12), 0.18 * heat)
        fr = fog(fr, t, 12, density=0.7, y0=330, y1=560, color=(0.75, 0.75, 0.78), speed=(-60, -90), scale=0.6)
        fr = sparks(fr, t, 111, (x + 300, 560), n=80, life=0.9, speed=(40, 220), dirc=(0, -1), spread=0.8, grav=-60, amt=0.8, area=(700, 10))
        fr = rain(fr, t, 110, inten=0.7, angle=6, layers=(1, 2, 3))
        return fr


class C12(Shot):
    """ECU 刃口：通红的刃口占满画面，白汽翻滚、铁屑火星剥落（同一张生图换成刃口大特写）。"""
    def render(self, t):
        u = self.u(t)
        fr = np.zeros((H, W, 3), np.float32) + 0.015
        fr, A = put(fr, t, 'C/blade_hot', 1100 - 120 * u, 600, 2100 * 2.8, anchor='src', src=(0.66, 0.5), rot=-6, rim=COLD, rim_amt=0.8,
                    expo=1.1, sat=1.1, gamma=0.95, tint=(1.06, 0.97, 0.93), wind=0, breathe=0.0)
        fr = fr + cv2.GaussianBlur(fr * A[..., None], (0, 0), 30) * np.array([1.0, 0.35, 0.12], np.float32) * 0.6
        fr = fx2.heat_haze(fr, t, amt=5.0)
        fr = fog(fr, t, 13, density=0.85, y0=200, y1=700, color=(0.8, 0.8, 0.82), speed=(-120, -160), scale=0.5)
        fr = sparks(fr, t, 121, (900, 640), n=140, life=0.8, speed=(80, 420), dirc=(0.3, -1), spread=0.7, grav=-100, amt=1.2, area=(800, 20))
        return fr


class C13(Shot):
    """riser：镜头冲进翻滚的雷云，电光在云腹里蓄能，越推越快，鼓点上闪。"""
    def render(self, t):
        u = self.u(t)
        hits = [V(x) for x in (131.42, 131.61, 131.96, 132.07)]
        fl = sum(0.7 * pulse(t, h, 0.07) for h in hits) + 0.6 * ein(u, 2)
        fr = storm_sky(t * 3, 13, 2000, flash=fl, flash_c=(960, 480), speed=(30, -40), cloud=1.3)
        z = 1.0 + 0.5 * ein(u, 2.5)
        fr = camxf(fr, z, rot=12 * ein(u, 2))
        fr = radial(fr, (960, 500), 0.05 + 0.25 * ein(u, 2))
        for i, h in enumerate(hits):
            fr = bolt(fr, t, h, 130 + i, (300 + 400 * i, 150), (800 + 120 * i, 980), inten=0.6, sky=0.0, width=2)
        return fr


class C14(Shot):
    """LS 低机位：雷暴旷野，鬼屠立在积水里仰面看天（新精灵 look_up_back），132.31 背后一道落地闪电。"""
    def render(self, t):
        u = self.u(t)
        tb = V(132.31)
        fl = bolt_env(t, tb)
        fr = storm(t, 14, horizon=800, flash=0.4 + fl * 1.2, fc=(700, 100), refl=0.7)
        fr = bolt(fr, t, tb, 141, (640, 130), (560, 800), inten=1.2, width=4)
        fr = fog(fr, t, 14, density=0.5, y0=650, y1=900, color=(0.4, 0.43, 0.5), speed=(40, 0), light=(600, 600), light_r=500)
        fr, _ = put(fr, t, 'C/look_up_back', 1180, 905, 600, rim=COLD, rim_amt=2.2, rim_dir=(-0.8, -0.6), flash=fl, expo=0.9, wind=6)
        # 倒影
        fr = rain(fr, t, 140, inten=1.1, angle=12, light=(600, 400))
        return fr


class C15(Shot):
    """ELS：鬼屠腾空（新精灵 leap_arc），133.22 电光炸开，速度坡度 1→0.1 定格在空中，雨滴悬停。"""
    def render(self, t):
        tb = V(133.22); tb2 = V(133.79)
        k = t - self.t0
        ramp = 1 - 0.9 * sm(tb + 0.05, tb + 0.35, t)
        fl = bolt_env(t, tb) * 1.2 + bolt_env(t, tb2) * 0.9
        fr = storm(t, 15, horizon=930, flash=0.6 + fl, fc=(960, 200), refl=0.5)
        fr = bolt(fr, t, tb, 151, (1100, 60), (1300, 920), inten=1.3, width=4.5)
        fr = bolt(fr, t, tb2, 152, (700, 60), (520, 900), inten=1.0, width=3)
        # 起跳轨迹：前 0.4s 快速升起，之后近乎悬停
        y = lerp(980, 520, eout(k / 0.5, 3)) - 15 * (k - 0.5) * (k > 0.5)
        x = 760 + 260 * eout(k / 0.6, 2) + 20 * k
        fr, _ = put(fr, t, 'C/leap_arc', x, y, 420, look='sil', rim=(0.9, 0.93, 1.0), rim_amt=3.6, rim_dir=(0.3, -1), flash=fl,
                    mblur=(0, 30 * (1 - sm(0.15, 0.45, k))), wind=10 * ramp)
        fr = rain(fr, self.t0 + 0.45 + (t - self.t0 - 0.45) * (0.08 if k > 0.45 else 1), 150, inten=1.1, angle=10,
                  freeze=sm(0.35, 0.6, k) * 0.85)
        return fr


# =====================================================================================
# 副歌二「破」
class C16(Shot):
    """ECU 血雨：雨在空中瞬间变红（慢放），巨字「血」砸进雨幕，「冷」上碎成血珠落下。"""
    def render(self, t):
        u = self.u(t)
        tb = V(134.50); ts = V(135.06)
        fl = bolt_env(t, tb) * 1.3
        fr = storm(t, 16, horizon=1300, flash=0.5 + fl, ground=False, fc=(960, 300))
        fr = fr * np.array([1.05, 0.9, 0.9], np.float32)
        tr = 134.0 + (t - 134.0) * 0.3
        fr = rain(fr, tr, 160, inten=1.3, angle=6, red=sm(tb - 0.05, tb + 0.25, t), freeze=0.35)
        L = T.Layer()
        sc = 1 + 0.35 * (1 - eout((t - tb) / 0.22)) if t >= tb else 1.4
        T.place(L, '血', 960, 520, 640, RED_T, 'serif_black', scale=sc, texture=False)
        q = sm(ts, ts + 0.55, t)
        if q > 0: L = T.shatter_layer(L, q, (960, 470), 140, seed=16, power=0.7, gravity=1600, spin=200)
        fr = T.compose(fr, L, shadow=0.6, shadow_sigma=30, glow=0.9, glow_sigma=30, glow_color=(0.9, 0.08, 0.04))
        if 0 < t - ts < 0.12: self.ca = 6
        else: self.ca = 0
        return fr


class C17(Shot):
    """MS 落地：鬼屠从天砸进积水单膝跪地（S071），135.71 冲击波水环扩散、水花弹起。"""
    def render(self, t):
        u = self.u(t)
        tl = V(135.71)
        fl = 0.5 * bolt_env(t, tl)
        fr = storm(t, 17, horizon=760, flash=0.45 + fl, refl=0.7, rip=1.2)
        fr = fog(fr, t, 17, density=0.4, y0=680, y1=940, color=(0.4, 0.42, 0.48), speed=(50, 0))
        y = 960 if t >= tl else lerp(200, 960, ein(sm(self.t0, tl, t), 2))
        fr = shock_ring(fr, (960, 975), (t - tl) / 0.7, rx=1300, flat=0.16, amt=1.2)
        fr, _ = put(fr, t, 'S071/land_kneel', 960, y, 560, rim=COLD, rim_amt=2.2, flash=fl + 0.3 * pulse(t, tl, 0.2), expo=0.95,
                    mblur=(0, 40) if t < tl else (0, 0), wind=8)
        fr = splash(fr, t, tl, (960, 975), 171, n=220, spread=900, up=1100)
        fr = rain(fr, t, 170, inten=1.0, angle=10)
        self.ca = 5 if 0 <= t - tl < 0.1 else 0
        return fr


class C18(Shot):
    """LS 两翼冲锋：雷光下两侧重甲军阵平枪冲来，中间鬼屠起身拔刀（S072）。"""
    def __init__(self): self.army = None
    def render(self, t):
        import crowd
        if self.army is None: self.army = crowd.Army(seed=18, rows=14, cols=24, banners=8, torches=18)
        u = self.u(t)
        fl = 0.8 * bolt_env(t, V(136.64))
        fr = storm(t, 18, horizon=600, flash=0.5 + fl, fc=(960, 120), refl=0.4)
        # 方阵直接画进雷暴（自带大气透视/景深/甲光），中线分开让出通道，鬼屠在通道里起身
        fr, _info = self.army.draw(fr, t, cam=dict(x=0, y=2.2, z=0, pitch=4, f=1500), advance=1.2 + 2.4 * u, march=1.0, level=1.0,
                                   part=0.55, rim_color=COLD, rim=1.0 + 1.5 * fl, fog_color=(0.06, 0.07, 0.09), fog_dist=70,
                                   torch=0.5, glint=1.0 + fl, origin=(0.0, 0.0, 9.0))
        fr = fog(fr, t, 18, density=0.55, y0=560, y1=900, color=(0.35, 0.38, 0.45), speed=(0, 0), light=(960, 300), light_r=700)
        fr, _ = put(fr, t, 'S072/rise_draw', 960, 1000 - 30 * eout(u), 640, rim=COLD, rim_amt=2.4, flash=fl, expo=0.95, wind=7)
        fr = rain(fr, t, 180, inten=1.1, angle=12)
        return fr


class C19(Shot):
    """MS 甩镜旋斩：S073 起势 → 137.92 旋身（角向模糊）+ 白刃环形刀光，三名甲士剪影被斩开。"""
    def render(self, t):
        u = self.u(t)
        ts = V(137.92)
        fl = 0.6 * bolt_env(t, ts)
        fr = storm(t, 19, horizon=820, flash=0.4 + fl, ox=200 * u, refl=0.6)
        import crowd
        r = np.random.default_rng(19)
        for i, (x, sc) in enumerate(((360, 1.0), (1560, 1.05), (1240, 0.8))):
            sp = crowd.soldier_sprite(phase=0.3 * i, arm=0.6, weapon='spear', view='front', seed=i)
            A = sp[0] if isinstance(sp, tuple) else sp['A'] if isinstance(sp, dict) else None
            if A is None: break
            A = cv2.resize(A.astype(np.float32), None, fx=sc * 1.6, fy=sc * 1.6)
            h, w = A.shape[:2]
            k = max(0.0, t - ts - 0.05 * i)
            for half, sgn in ((A[:h // 2], -1), (A[h // 2:], 1)):
                hh = half.shape[0]
                dy = sgn * 220 * k * k * 3 + (0 if sgn < 0 else 0)
                dx = sgn * 90 * k * (1 if x < 960 else -1)
                y0 = int(960 - h + (0 if sgn < 0 else h // 2) + dy); x0 = int(x - w / 2 + dx)
                ys, xs = max(BAR, y0), max(0, x0); ye, xe = min(H - BAR, y0 + hh), min(W, x0 + w)
                if ye <= ys or xe <= xs: continue
                a = half[ys - y0:ye - y0, xs - x0:xe - x0]
                a = a[..., 0] if a.ndim == 3 else a
                a = np.clip(a, 0, 1) * (1 - sm(0.2, 0.5, k))
                fr[ys:ye, xs:xe] = fr[ys:ye, xs:xe] * (1 - a[..., None] * 0.95) + a[..., None] * 0.02
        pose = 'S073/spin1' if t < ts else 'S073/spin2'
        fr, _ = put(fr, t, pose, 960, 990, 660, rim=COLD, rim_amt=2.3, flash=fl, expo=0.95, rotblur=-25 * pulse(t, ts, 0.1), wind=9)
        fr = arc_streak(fr, (960, 640), 760, 170, 160, -200, (t - ts) / 0.4, width=10)
        fr = rain(fr, t, 190, inten=1.0, angle=-14)
        return fr


class C20(Shot):
    """CU 太上皇：电光一亮他缓缓侧头（S074），身后雷云里一个金色巨字「定」被闪电照出来。"""
    def render(self, t):
        u = self.u(t)
        tb = V(138.42); td = V(138.56); tz = V(138.88)
        fl = bolt_env(t, tb) + 0.8 * bolt_env(t, tz)
        fr = storm(t, 20, horizon=1300, flash=0.3 + fl, fc=(600, 300), ground=False)
        a = sm(td - 0.06, td + 0.1, t) * (0.35 + 0.65 * min(1, fl + 0.3))
        fr = paste_glyph(fr, '定', 640, 500, 760, GOLD_T, alpha=a, glow=0.9, glow_c=(1.0, 0.6, 0.2), shadow=0, scale=1 + 0.06 * u)
        fr = bolt(fr, t, tb, 201, (300, 80), (500, 900), inten=0.9)
        fr, _ = put(fr, t, 'S074/head_turn', 1330, 1100, 1100, anchor='foot', rim=(0.9, 0.92, 1.0), rim_amt=3.0, rim_dir=(-0.9, -0.4),
                    flash=fl, expo=0.75, look=None, wind=6, sat=0.5)
        fr = K.vol_light(fr, (640, 500), 300, GOLD, 0.5 * pulse(t, tz, 0.15))
        fr = rain(fr, t, 200, inten=0.8, angle=8)
        return fr


class C21(Shot):
    """对角分屏：左鬼屠（朱红边光）右太上皇（金边光），斜刀线隔开、两半反向滑；「正」在金侧、「邪」在红侧。"""
    def render(self, t):
        u = self.u(t)
        tx = V(139.48)
        A = storm(t, 211, horizon=1300, flash=0.3, ground=False) * np.array([1.1, 0.75, 0.72], np.float32)
        A, _ = put(A, t, 'S072/rise_draw', 640 - 60 * u, 1500, 1500, rim=RED * 1.3, rim_amt=3.2, rim_dir=(0.9, -0.4), expo=0.85, wind=7,
                   flash=0.6 * pulse(t, tx, 0.2))
        A = rain(A, t, 211, inten=0.7, angle=10, color=np.array([1.0, 0.5, 0.45], np.float32))
        B = storm(t, 212, horizon=1300, flash=0.3, ground=False) * np.array([1.08, 0.98, 0.82], np.float32)
        B, _ = put(B, t, 'S074/head_turn', 1450 + 60 * u, 1100, 1100, flip=True, rim=GOLD, rim_amt=3.0, rim_dir=(-0.9, -0.4), expo=0.7, wind=5)
        B = rain(B, t, 212, inten=0.7, angle=10, color=np.array([1.0, 0.85, 0.6], np.float32))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = (xx - 960) * 0.88 + (yy - 540) * 0.47              # 斜线（约 62°）
        m = np.clip(d / 2 + 0.5, 0, 1)[..., None]
        fr = A * (1 - m) + B * m
        line = np.exp(-(d / 3) ** 2) + 0.4 * np.exp(-(d / 30) ** 2)
        fr = fr + line[..., None] * np.array([1.0, 0.95, 0.9], np.float32) * (0.9 + 0.8 * pulse(t, tx, 0.2))
        fr = paste_glyph(fr, '正', 1380, 390, 300, GOLD_T, glow=0.7, glow_c=(1, 0.6, 0.2), alpha=sm(V(138.88) - 0.2, V(138.88), t) if t < self.t0 + 0.05 else 1.0)
        if t >= tx - 0.12:
            e = eout((t - (tx - 0.12)) / 0.12)
            fr = paste_glyph(fr, '邪', 560, 420, 300, RED_T, glow=0.9, glow_c=(0.9, 0.08, 0.04), scale=lerp(1.8, 1, e), alpha=min(1, e * 2))
        self.ca = 4 if 0 <= t - tx < 0.1 else 0
        return fr


class C22(Shot):
    """ELS 字墙入云：宋体律令砌成的墙从积水升到雷云里，仰拍上摇；脚下一个小小的人（look_up_back）。"""
    def __init__(self): self.wall = PR.CharWall(seed=22, cols=22, rows=26, brick=(150, 64))
    def render(self, t):
        u = self.u(t)
        tb = V(141.32)
        fl = bolt_env(t, tb) + 0.7 * bolt_env(t, V(141.94))
        oy = lerp(120, -80, eio(u))
        fr = storm(t, 22, horizon=930 - oy * 0.3, flash=0.5 + fl, fc=(960, 60), refl=0.5)
        fr, info = self.wall.draw(fr, t, x0=360, y0=-900 - oy, x1=1560, y1=900 - oy * 0.3, tilt=0.55, light_color=COLD, rim=0.8, fog=0.4)
        fr = bolt(fr, t, tb, 221, (1400, 40), (1250, 500), inten=1.0)
        # 律令里被点亮的「若规则是」：竖排在墙面中轴
        for i, k in enumerate((0, 1, 2, 3)):
            ti = V(CT(34, k))
            if t < ti - 0.1: continue
            e = eout((t - ti + 0.1) / 0.18)
            fr = paste_glyph(fr, '若规则是'[i], 960, 220 + i * 120 - oy * 0.6, 96, (0.95, 0.97, 1.0) if i != 1 else RED_T,
                             glow=0.8, glow_c=(0.7, 0.8, 1.0) if i != 1 else (0.9, 0.1, 0.05), scale=lerp(1.5, 1, e), alpha=e, font='serif_b')
        fr = fog(fr, t, 22, density=0.6, y0=760, y1=950, color=(0.4, 0.43, 0.5), speed=(40, 0))
        fr, _ = put(fr, t, 'C/look_up_back', 960, 915 - oy * 0.3, 140, rim=COLD, rim_amt=3.0, flash=fl, expo=0.8, wind=3, look='sil')
        fr = rain(fr, t, 220, inten=1.0, angle=8)
        return fr


class C23(Shot):
    """CU 仰拍字墙局部（同墙 match cut）：「铁」「壁」「铜」三块字砖逐拍从墙里凸出、逼近镜头。"""
    def __init__(self): self.wall = PR.CharWall(seed=23, cols=12, rows=14, brick=(150, 64))
    def render(self, t):
        u = self.u(t)
        fl = sum(0.6 * bolt_env(t, V(CT(34, k))) for k in (4, 5, 6))
        fr = storm(t, 23, horizon=1300, flash=0.3 + fl, ground=False)
        fr, info = self.wall.draw(fr, t, x0=-300, y0=-500, x1=2200, y1=1400, tilt=0.35, light_color=COLD, rim=1.0)
        fr = fr * 1.25
        pos = [(560, 470), (960, 690), (1380, 420)]
        for i, k in enumerate((4, 5, 6)):
            ti = V(CT(34, k))
            if t < ti - 0.15: continue
            e = eout((t - ti + 0.15) / 0.22)
            x, y = pos[i]
            sc = lerp(1.0, 1.35 + 0.08 * (t - ti), e)
            # 砖体：厚度阴影 + 砖面
            bw, bh = 300 * sc, 128 * sc
            sh = np.zeros((H, W), np.float32)
            cv2.rectangle(sh, (int(x - bw / 2 + 30 * e), int(y - bh / 2 + 40 * e)), (int(x + bw / 2 + 30 * e), int(y + bh / 2 + 40 * e)), 1.0, -1)
            fr = fr * (1 - cv2.GaussianBlur(sh, (0, 0), 16)[..., None] * 0.7 * e)
            face = np.zeros((H, W), np.float32)
            cv2.rectangle(face, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), 1.0, -1)
            fr = fr * (1 - face[..., None] * e) + face[..., None] * e * np.array([0.12, 0.12, 0.13], np.float32)
            rim = cv2.Canny((face * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255
            fr = fr + cv2.GaussianBlur(rim, (0, 0), 2)[..., None] * COLD * 1.5 * e
            fr = paste_glyph(fr, '铁壁铜'[i], x, y, 110 * sc, (0.92, 0.94, 1.0), glow=0.6, glow_c=(0.6, 0.7, 1.0), alpha=e, font='serif_b', shadow=0.3)
            fr = dust(fr, t, 230 + i, n=40, amt=0.5 * e * (1 - sm(0.2, 0.8, t - ti)), drift=(0, 120), region=(int(x - bw / 2), int(y), int(x + bw / 2), int(y + 300)))
        fr = rain(fr, t, 230, inten=0.7, angle=4)
        return fr


class C24(Shot):
    """MS 横移跟拍：鬼屠沿墙根疾奔（S078），刀尖在字墙上拖出一路火星长线；143.94「墙」低频震。"""
    def __init__(self): self.wall = PR.CharWall(seed=24, cols=26, rows=10, brick=(150, 64))
    def render(self, t):
        u = self.u(t)
        ox = 900 * u
        fr = storm(t, 24, horizon=950, flash=0.35 + 0.5 * bolt_env(t, V(143.94)), refl=0.5, ox=ox)
        fr, _ = self.wall.draw(fr, t, x0=-400 - ox * 0.9, y0=BAR - 40, x1=3600 - ox * 0.9, y1=960, tilt=0.0, light_color=WARM, rim=0.5)
        fr = fr * 0.7
        x = 1250 - 80 * u; y = 990
        tip = (x - 330, y - 140)
        # 火星长线（墙面上灼痕：从屏幕左侧到刀尖）
        G = np.zeros((H // 2, W // 2), np.float32)
        cv2.line(G, (0, int((tip[1] + 30) / 2)), (int(tip[0] / 2), int(tip[1] / 2)), 1.0, 2, cv2.LINE_AA)
        Gs = cv2.resize(G + cv2.GaussianBlur(G, (0, 0), 6) * 2.5, (W, H))
        fr = fr + Gs[..., None] * np.array([1.0, 0.5, 0.18], np.float32) * 0.9
        fr = sparks(fr, t, 241, tip, n=160, life=0.5, speed=(300, 1100), dirc=(-1, -0.4), spread=0.5, grav=1500, amt=1.4)
        fr, _ = put(fr, t, 'S078/run', x, y + 8 * abs(math.sin(t * 9)), 640, rim=WARM, rim_amt=2.0, rim_dir=(-1, -0.3), expo=0.9, wind=10,
                    mblur=(-18, 0))
        fr = K.vol_light(fr, tip, 160, (1.0, 0.5, 0.2), 0.6)
        fr = rain(fr, t, 240, inten=0.9, angle=-6, cam=(ox, 0))
        return fr


class C25(Shot):
    """LS 跃起上撩劈墙（S079）：145.14 一道白金刀光自下而上贯穿整面字墙，「劈」字被竖向斩开。"""
    def __init__(self): self.wall = PR.CharWall(seed=25, cols=16, rows=20, brick=(150, 64))
    def render(self, t):
        u = self.u(t)
        ts = V(145.14)
        fl = 1.2 * pulse(t, ts, 0.12)
        fr = storm(t, 25, horizon=1000, flash=0.3 + fl, refl=0.4)
        fr, _ = self.wall.draw(fr, t, x0=-200, y0=-300, x1=2120, y1=1000, tilt=0.25, light_color=COLD, rim=0.6)
        fr = fr * (0.8 + fl)
        k = t - self.t0
        y = lerp(1040, 820, eout(k / 0.4))
        fr, _ = put(fr, t, 'S079/leap_slash_up', 900, y, 700, rim=COLD, rim_amt=2.4, flash=fl, expo=0.95, wind=9)
        fr = slash_streak(fr, (1010, 1000), (1120, BAR - 20), (t - ts) / 0.45, width=12, col=(1.0, 0.92, 0.75), amt=2.0)
        if t > ts:
            g = np.zeros((H, W), np.float32); cv2.line(g, (1010, 1000), (1120, BAR), 1.0, 3, cv2.LINE_AA)
            fr = fr + cv2.GaussianBlur(g, (0, 0), 10)[..., None] * GOLD * 0.8 * (1 - sm(0.3, 0.8, t - ts))
            fr = sparks(fr, t, 251, (1065, 560), n=200, life=0.7, speed=(300, 1200), dirc=(1, 0), spread=1.5, grav=900, t0=ts, burst=True,
                        area=(30, 420), color=(1.0, 0.85, 0.5))
        # 「劈」被斩开
        if t >= ts - 0.12:
            L = T.Layer(); e = eout((t - ts + 0.12) / 0.12)
            T.place(L, '劈', 1480, 520, 420, RED_T, 'serif_black', scale=lerp(1.6, 1, e), alpha=min(1, e * 2), texture=False)
            q = sm(ts + 0.05, ts + 0.4, t)
            if q > 0:
                A2, C2 = L.A.copy(), L.C.copy()
                L.A[:] = 0
                for sgn in (-1, 1):
                    M = np.float32([[1, 0, sgn * 60 * q], [0, 1, -sgn * 20 * q]])
                    msk = np.zeros((H, W), np.float32); xx = np.arange(W)[None, :]
                    msk[:] = ((xx < 1480) if sgn < 0 else (xx >= 1480))
                    L.A[:] = np.maximum(L.A, cv2.warpAffine(A2 * msk, M, (W, H)))
                L.C[:] = C2
            fr = T.compose(fr, L, shadow=0.7, shadow_sigma=24, glow=0.8, glow_sigma=22, glow_color=(0.9, 0.08, 0.04))
        self.ca = 6 if 0 <= t - ts < 0.1 else 0
        return fr


class C26(Shot):
    """崩塌：字墙从中间炸开，字砖朝镜头飞来（慢放），缺口后面是惨白的电光天。"""
    def __init__(self): self.wall = PR.CharWall(seed=26, cols=16, rows=16, brick=(150, 64))
    def render(self, t):
        u = self.u(t)
        tr = self.t0 + (t - self.t0) * 0.4
        fl = 0.8 + 0.6 * math.sin(t * 13) ** 2
        fr = storm(t, 26, horizon=1100, flash=fl, fc=(960, 500), ground=False)
        fr = fr + 0.25
        fr, _ = self.wall.draw(fr, tr, x0=-250, y0=-200, x1=2170, y1=1150, tilt=0.15, collapse=0.08 + 0.75 * eout(u, 2),
                               collapse_center=(0.5, 0.55), light_color=COLD, rim=1.0, dust=1.5)
        fr = dust(fr, t, 261, n=200, amt=0.6, drift=(0, 160), region=(0, BAR, W, H - BAR))
        fr = rain(fr, tr, 260, inten=0.8, angle=6)
        return fr


class C27(Shot):
    """ELS 缺口：墙的两半剪影之间，天被闪电照得惨白如昼（伪天亮），缺口里站着一个小小的人。"""
    def render(self, t):
        u = self.u(t)
        tt = V(147.28)
        fl = 0.4 + 1.6 * bolt_env(t, tt) + 0.6 * pulse(t, V(146.5), 0.1)
        fr = storm(t, 27, horizon=900, flash=fl, fc=(960, 400), refl=0.9)
        fr = fr + sm(tt - 0.05, tt + 0.1, t) * 0.5 * (1 - sm(tt + 0.3, tt + 0.8, t))
        # 墙两半剪影（锯齿边）
        r = np.random.default_rng(27)
        ys = np.linspace(BAR - 20, 910, 30)
        lx = 820 + np.cumsum(r.normal(0, 14, 30)) - 40 * u; rx = 1100 + np.cumsum(r.normal(0, 14, 30)) + 40 * u
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [np.array([[0, BAR - 20]] + [[int(a), int(b)] for a, b in zip(lx, ys)] + [[0, 910]], np.int32)], 1.0, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([[W, BAR - 20]] + [[int(a), int(b)] for a, b in zip(rx, ys)] + [[W, 910]], np.int32)], 1.0, cv2.LINE_AA)
        edge = cv2.GaussianBlur(m, (0, 0), 3) - cv2.GaussianBlur(m, (0, 0), 12)
        fr = fr * (1 - m[..., None]) + m[..., None] * 0.02 + np.clip(-edge, 0, 1)[..., None] * COLD * 0.8 * fl
        fr, _ = put(fr, t, 'S082/walk_front', 960, 910, 170, look='sil', rim=(1, 1, 1), rim_amt=3.0, wind=3, flash=fl)
        fr = rain(fr, t, 270, inten=1.0, angle=8, light=(960, 400))
        return fr


class C28(Shot):
    """字即光：惨白天光里金色巨字「亮」墨晕显出（147.74），随即四周黑墨涌来把它吞掉——只是闪电的伪天亮。"""
    def render(self, t):
        u = self.u(t)
        tl = V(147.74)
        base = 1.15 - 0.95 * sm(tl + 0.25, self.t1, t)
        fr = np.ones((H, W, 3), np.float32) * np.array([0.93, 0.94, 0.97], np.float32) * base
        fr = storm_sky(t, 28, 2000, flash=0.6, cloud=0.6) * 0.3 + fr
        if t >= tl - 0.15:
            e = eout((t - tl + 0.15) / 0.2)
            fr = paste_glyph(fr, '亮', 960, 520, 620, GOLD_T, glow=1.0, glow_c=(1.0, 0.65, 0.25), alpha=e, scale=lerp(1.25, 1.0, e) + 0.04 * u,
                             shadow=0.2)
        fr = ink_vignette(fr, sm(tl + 0.15, self.t1, t))
        return fr


class C29(Shot):
    """MS 后退跟拍：鬼屠从缺口走出（S082），碎石在身侧坠落，余电在背后。"""
    def render(self, t):
        u = self.u(t)
        fr = storm(t, 29, horizon=860, flash=0.3 + 0.8 * bolt_env(t, V(148.54)), fc=(960, 300), refl=0.6)
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [np.array([[0, BAR], [700, BAR], [760, 860], [0, 860]])], 1.0)
        cv2.fillPoly(m, [np.array([[W, BAR], [1220, BAR], [1160, 860], [W, 860]])], 1.0)
        fr = fr * (1 - m[..., None] * 0.92)
        fr, _ = put(fr, t, 'S082/walk_front', 960, 1010, 640 * (1 + 0.1 * u), rim=COLD, rim_amt=2.6, rim_dir=(0, -1), expo=0.9, wind=6,
                    flash=0.6 * bolt_env(t, V(148.54)))
        r = np.random.default_rng(29)
        for i in range(10):
            x = r.choice([r.uniform(200, 700), r.uniform(1250, 1750)]); ph = r.random()
            y = BAR + ((t * r.uniform(500, 900) + ph * 800) % 900); s = int(r.uniform(10, 40))
            cv2.rectangle(fr, (int(x), int(y)), (int(x + s), int(y + s * 0.6)), (0.02, 0.02, 0.025), -1)
        fr = dust(fr, t, 291, n=120, amt=0.5, drift=(0, 80))
        fr = rain(fr, t, 290, inten=0.9, angle=6)
        return fr


class C30(Shot):
    """CU 腰带：暗红腰带在暴风里狂舞（S083，红绳母题）；「血」写成竖向长条朱红，像被风扯开的布。"""
    def render(self, t):
        u = self.u(t)
        fr = storm(t, 30, horizon=1300, flash=0.25 + 0.6 * bolt_env(t, V(149.70)), ground=False)
        fr, _ = put(fr, t, 'S083/sash', 900 - 80 * u, 600, 1500, anchor='src', src=(0.5, 0.62), rim=COLD, rim_amt=1.8, expo=1.0, sat=0.9,
                    wind=22, flash=0.4 * bolt_env(t, V(149.70)))
        fr = rain(fr, t, 300, inten=1.0, angle=20)
        self.ca = 3 if 0 <= t - V(149.70) < 0.1 else 0
        return fr


class C31(Shot):
    """MS 巨掌：天上一只巨手的阴影一掌拍下（150.77），鬼屠被震飞后撤（S084）。"""
    def render(self, t):
        u = self.u(t)
        ti = V(150.77)
        fr = storm(t, 31, horizon=860, flash=0.4 + 0.9 * pulse(t, ti, 0.15), refl=0.5)
        q = ein(sm(self.t0, ti, t), 2.5)
        hy = lerp(-700, 80, q) if t < ti else 80 + 20 * math.sin((t - ti) * 30) * math.exp(-(t - ti) * 6)
        fr, A = put(fr, t, 'S062/puppeteer', 1150, hy, 1400, anchor='src', src=(0.5, 0.0), look='shadow', alpha=0.9, wind=0, breathe=0)
        fr = fr * (1 - 0.35 * q)
        k = max(0.0, t - ti)
        x = 1180 - 420 * eout(k / 0.5)
        fr, _ = put(fr, t, 'S084/knock_back', x, 1000, 620, rim=COLD, rim_amt=2.3, expo=0.95, wind=10, mblur=(-60 * pulse(t, ti, 0.2), 0),
                    flash=0.8 * pulse(t, ti, 0.15))
        fr = shock_ring(fr, (1150, 960), k / 0.6, rx=1400, flat=0.14, amt=1.0)
        fr = dust(fr, t, 311, n=220, amt=0.9 * min(1, k * 5) * (1 - sm(0.4, 1.2, k)), drift=(-300, -60), region=(300, 700, W, 1000))
        fr = rain(fr, t, 310, inten=1.0, angle=10)
        self.ca = 6 if 0 <= k < 0.1 else 0
        return fr


class C32(Shot):
    """ELS 低机位：巨印「是」从雷云里砸下（152.21），鬼屠（S078 反向）在下面疾跑闪避。"""
    def __init__(self): self.seal = PR.GiantSeal('是', seed=32)
    def render(self, t):
        u = self.u(t)
        ti = V(152.21)
        fr = storm(t, 32, horizon=880, flash=0.5 + 1.0 * pulse(t, ti, 0.15), fc=(1200, 150), refl=0.6)
        fall = sm(self.t0, ti, t) ** 2
        fr, _ = self.seal.draw(fr, t, 1250, 900, 520, fall=fall, tilt=-8, rim_color=COLD, rim=1.4)
        k = max(0.0, t - ti)
        fr = dust(fr, t, 321, n=240, amt=1.0 * min(1, k * 6) * (1 - sm(0.3, 1.0, k)), drift=(0, -200), region=(800, 600, 1800, 960))
        fr = shock_ring(fr, (1250, 900), k / 0.6, rx=1500, flat=0.12)
        fr, _ = put(fr, t, 'S078/run', 420 + 700 * u, 920 + 6 * abs(math.sin(t * 10)), 300, flip=True, rim=COLD, rim_amt=2.6, expo=0.9, wind=9,
                    mblur=(16, 0))
        fr = rain(fr, t, 320, inten=1.0, angle=12)
        self.ca = 5 if 0 <= k < 0.1 else 0
        return fr


class C33(Shot):
    """LS 近距：第二方巨印「非」紧贴镜头前砸下（152.70），泥水与碎石扑向镜头。"""
    def __init__(self): self.seal = PR.GiantSeal('非', seed=33)
    def render(self, t):
        u = self.u(t)
        ti = V(152.70)
        fr = storm(t, 33, horizon=800, flash=0.4 + 1.0 * pulse(t, ti, 0.12), fc=(500, 200), refl=0.6)
        fall = sm(self.t0 - 0.2, ti, t) ** 2
        fr, _ = self.seal.draw(fr, t, 760, 1080, 1000, fall=fall, tilt=10, rim_color=COLD, rim=1.4)
        k = max(0.0, t - ti)
        fr = splash(fr, t, ti, (760, 960), 331, n=260, spread=1500, up=1400, col=(0.5, 0.5, 0.55))
        fr = dust(fr, t, 332, n=240, amt=1.0 * min(1, k * 6), drift=(100, -150), size=(2, 6))
        fr = rain(fr, t, 330, inten=1.0, angle=12)
        self.ca = 6 if 0 <= k < 0.1 else 0
        return fr


class C34(Shot):
    """MS 斜劈（新精灵 slash_diag_down）：身后「对」「错」两方印被一刀斩成两半错开。"""
    def __init__(self): self.s1 = PR.GiantSeal('对', seed=341); self.s2 = PR.GiantSeal('错', seed=342)
    def render(self, t):
        u = self.u(t)
        ts = V(153.12)
        fr = storm(t, 34, horizon=860, flash=0.4 + 0.9 * pulse(t, ts, 0.14), fc=(960, 150), refl=0.5)
        sp = sm(ts, ts + 0.5, t)
        fr, _ = self.s1.draw(fr, t, 560, 760, 440, fall=1.0, tilt=-6, split=sp, split_angle=-62, rim_color=GOLD, rim=1.2)
        fr, _ = self.s2.draw(fr, t, 1380, 740, 440, fall=1.0, tilt=-6, split=sp, split_angle=-62, rim_color=RED, rim=1.4)
        fr = slash_streak(fr, (1650, 200), (300, 980), (t - ts) / 0.4, width=12, amt=2.0)
        fr, _ = put(fr, t, 'C/slash_diag_down', 980, 1010, 680, rim=COLD, rim_amt=2.4, rim_dir=(0.4, -1), flash=0.8 * pulse(t, ts, 0.15),
                    expo=0.95, wind=9)
        fr = rain(fr, t, 340, inten=1.0, angle=-10)
        self.ca = 5 if 0 <= t - ts < 0.1 else 0
        return fr


class C35(Shot):
    """LS 硬扛：最大的一方巨印压下，鬼屠举刀过顶硬扛（S087），地面龟裂、积水被压得往外喷；「扛」朱红竖排。"""
    def __init__(self): self.seal = PR.GiantSeal('令', seed=35)
    def render(self, t):
        u = self.u(t)
        tk = V(154.70)
        sink = 18 * eout(u)
        fr = storm(t, 35, horizon=880, flash=0.35 + 0.8 * pulse(t, tk, 0.2), refl=0.5)
        oy = lerp(-60, 20, eio(u))
        fr, _ = self.seal.draw(fr, t, 960, 330 + sink - oy * 0.3, 1500, fall=1.0, tilt=12, rim_color=GOLD, rim=1.0)
        cr = np.zeros((H // 4, W // 4), np.float32)
        for i in range(6):
            p = crack_path((960, 965), (960 + (i - 2.5) * 330, 1060), 350 + i, n=24, jag=0.06)
            cr = np.maximum(cr, draw_crack(cr.shape, p / 4, 2, branches=3, seed=i, prog=sm(self.t0, self.t0 + 0.8, t)))
        crm = cv2.resize(cr, (W, H))
        fr = fr * (1 - crm[..., None] * 0.8) + crm[..., None] * np.array([1.0, 0.5, 0.2], np.float32) * 0.5
        fr, _ = put(fr, t, 'S087/brace_overhead', 960, 1000 + sink, 720, rim=GOLD, rim_amt=1.8, rim_dir=(0, -1), expo=0.95, wind=6,
                    flash=0.5 * pulse(t, tk, 0.2))
        fr = dust(fr, t, 351, n=160, amt=0.7, drift=(0, -40), region=(300, 820, 1620, 1000))
        fr = rain(fr, t, 350, inten=0.9, angle=6)
        return fr


class C36(Shot):
    """ELS 雷暴全景：天幕翻滚，155.96「天」上一道横贯天际的闪电。"""
    def render(self, t):
        u = self.u(t)
        tt = V(155.96)
        fl = bolt_env(t, tt) * 1.3 + 0.5 * bolt_env(t, V(155.40))
        fr = storm(t, 36, horizon=900, flash=0.4 + fl, fc=(960, 250), refl=0.6, oy=-60 * u)
        fr = bolt(fr, t, tt, 361, (80, 260), (1840, 330), inten=1.3, branches=12, width=4)
        fr = bolt(fr, t, V(155.40), 362, (1400, 100), (1500, 880), inten=0.8)
        fr = fog(fr, t, 36, density=0.4, y0=780, y1=920, color=(0.4, 0.42, 0.5), speed=(60, 0))
        fr = rain(fr, t, 360, inten=0.8, angle=10)
        return fr


class C37(Shot):
    """闪电写字：巨字「判」由电光逐笔亮起（156.70），随即一道朱红横刀把它当中斩断。"""
    def render(self, t):
        u = self.u(t)
        tp = V(156.70); ts = tp + 0.24
        fl = bolt_env(t, tp) * 1.2
        fr = storm(t, 37, horizon=1300, flash=0.25 + fl, fc=(960, 400), ground=False)
        if t >= tp - 0.1:
            a = T.glyph('判', 'serif_black', 680, False)
            g = a.shape[0]
            edge = np.clip(a - cv2.erode(a, np.ones((9, 9), np.uint8)), 0, 1)
            nz = cv2.resize(np.random.default_rng(int(t * 30)).random((g // 12, g // 12)).astype(np.float32), (g, g))
            core = np.clip(edge * (0.6 + 0.8 * nz), 0, 1) + a * 0.12
            prog = sm(tp - 0.1, tp + 0.04, t)
            yy = np.linspace(0, 1, g, dtype=np.float32)[:, None]
            core = core * (yy < prog * 1.05)
            L = T.Layer(); L.paste(core, 960, 530, (0.92, 0.95, 1.0))
            if t > ts:
                q = sm(ts, ts + 0.35, t)
                A2 = L.A.copy(); L.A[:] = 0
                yyf = np.arange(H)[:, None]
                for sgn in (-1, 1):
                    msk = ((yyf < 530) if sgn < 0 else (yyf >= 530)).astype(np.float32)
                    M = np.float32([[1, 0, -sgn * 40 * q], [0, 1, sgn * 70 * q]])
                    L.A[:] = np.maximum(L.A, cv2.warpAffine(A2 * msk, M, (W, H)))
            fr = T.compose(fr, L, shadow=0, glow=1.4, glow_sigma=18, glow_color=(0.6, 0.75, 1.0))
            fr = T.compose(fr, L, shadow=0, glow=0.6, glow_sigma=60, glow_color=(0.5, 0.6, 1.0))
        fr = slash_streak(fr, (160, 560), (1760, 500), (t - ts) / 0.4, width=10, col=(1.0, 0.25, 0.15), amt=2.2)
        fr = rain(fr, t, 370, inten=0.8, angle=10)
        self.ca = 6 if 0 <= t - tp < 0.12 else 0
        return fr


class C38(Shot):
    """CU 刀即秤：双手把刀端平如秤杆（生图 blade_level），雨珠沿刃面等距停成刻度；「我来量」刻在刻度上。"""
    def render(self, t):
        u = self.u(t)
        fr = storm(t, 38, horizon=1300, flash=0.25, ground=False) * 0.9
        x = lerp(1010, 910, eio(u))
        fr, A = put(fr, t, 'C/blade_level', x, 560, 2000, anchor='src', src=(0.5, 0.5), rim=COLD, rim_amt=1.5, expo=1.05, sat=0.6,
                    wind=0, breathe=0.0)
        # 刻度雨珠：沿刀身等距（按 alpha 求刀身横线）
        ys, xs = np.nonzero(A[400:720] > 0.5)
        if len(xs):
            y0 = 400 + np.median(ys)
            xa, xb = np.percentile(xs, 8), np.percentile(xs, 96)
            for i in range(13):
                px = xa + (xb - xa) * (i + 0.5) / 13; ph = (t * 2.15 * 1 - i * 0.08) % 1
                rr = 5 + 2 * (i % 2 == 0)
                fr = K.vol_light(fr, (px, y0 - 6), 10, (0.9, 0.95, 1.0), 0.9 + 0.6 * math.exp(-ph * 6))
                tick = np.zeros((H, W), np.float32); cv2.line(tick, (int(px), int(y0 + 30)), (int(px), int(y0 + 30 + (24 if i % 3 == 0 else 12))), 1.0, 2, cv2.LINE_AA)
                fr = fr + tick[..., None] * COLD * 0.6
            for j, k in enumerate((3, 4, 5)):
                ti = V(CT(38, k))
                if t < ti - 0.1: continue
                e = eout((t - ti + 0.1) / 0.16)
                px = xa + (xb - xa) * (0.3 + 0.2 * j)
                c = '我来量'[j]
                fr = paste_glyph(fr, c, px, y0 + 130, 96, RED_T if c == '量' else (0.95, 0.96, 1.0), glow=0.7,
                                 glow_c=(0.9, 0.1, 0.05) if c == '量' else (0.6, 0.7, 1.0), alpha=e, scale=lerp(1.6, 1, e), font='serif_b')
        fr = rain(fr, t, 380, inten=0.6, angle=4, layers=(0, 1, 2))
        return fr


class C39(Shot):
    """MCU 蓄力：鬼屠把刀拉到身后（S090），镜头后拉、慢放，159.01 一记底鼓暗角收紧。"""
    def render(self, t):
        u = self.u(t)
        fr = storm(t, 39, horizon=1300, flash=0.3 + 0.5 * pulse(t, V(159.01), 0.2), ground=False, fc=(500, 200))
        fr, _ = put(fr, t, 'S090/wind_up', 1150, 1110, 900 * (1.06 - 0.06 * eout(u)), rim=COLD, rim_amt=2.6, rim_dir=(-0.8, -0.6), expo=1.0,
                    wind=7, flash=0.4 * pulse(t, V(159.01), 0.2))
        fr = rain(fr, self.t0 + (t - self.t0) * 0.5, 390, inten=0.9, angle=8)
        return fr


class C40(Shot):
    """LS 背身横斩（新精灵 slash_back）：前方雷暴里一架高过云的巨大天平，159.95 一刀横断天平横梁。"""
    def __init__(self): self.B = PR.Balance(seed=40, people=80)
    def render(self, t):
        u = self.u(t)
        ts = V(159.95)
        fl = 1.0 * pulse(t, ts, 0.15) + 0.6 * bolt_env(t, V(159.53))
        fr = storm(t, 40, horizon=860, flash=0.4 + fl, fc=(960, 250), refl=0.5)
        br = sm(ts, ts + 0.45, t)
        fr, _ = self.B.draw(fr, t, 960, 330, 1500, tilt=4 * math.sin(t * 2), sway=1, left_people=1.0, right='seal', break_=br,
                            rim_color=GOLD, rim=1.8, light=(0, -1))
        fr = fog(fr, t, 40, density=0.5, y0=600, y1=900, color=(0.38, 0.4, 0.46), speed=(30, 0))
        pose_t = ts - 0.25
        fr, _ = put(fr, t, 'C/slash_back', 760, 1050, 760, rim=COLD, rim_amt=2.6, rim_dir=(0, -1), flash=fl, expo=0.85, wind=10)
        fr = arc_streak(fr, (960, 360), 1100, 140, 200, -20, (t - ts) / 0.4, width=12, amt=2.2)
        fr = sparks(fr, t, 401, (960, 330), n=180, life=0.8, speed=(200, 900), dirc=(0, 1), spread=2.5, grav=600, t0=ts, burst=True,
                    color=(1.0, 0.8, 0.4))
        fr = rain(fr, t, 400, inten=1.0, angle=10)
        self.ca = 6 if 0 <= t - ts < 0.1 else 0
        return fr


class C41(Shot):
    """MCU 秤盘坠落：断梁和秤盘砸向镜头 → 定格 → 故障 → 「答」金闪 → 切黑。"""
    def __init__(self): self.B = PR.Balance(seed=41, people=80)
    def render(self, t):
        u = self.u(t)
        tf = min(t, V(160.72))
        fr = storm(t, 41, horizon=1300, flash=0.4, ground=False)
        fr, _ = self.B.draw(fr, tf, 960, 260 + 500 * ein(sm(self.t0, V(160.72), tf), 1.5), 2600, tilt=0, break_=1.0, rim_color=GOLD, rim=2.0)
        fr = rain(fr, tf, 410, inten=0.9, angle=10, freeze=0.6 if t > V(160.72) else None)
        return fr


SHOTS = [C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15,
         C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C30, C31, C32, C33, C34, C35, C36, C37, C38,
         C39, C40, C41]
# 切点（声音时刻，秒）：拍 / 人声首字；帧 = 视觉帧（最近帧再提前 1 帧）。首尾 = 段落边界。
CUT_T = [117.80, 118.95, 120.53, 121.46, 122.83, 123.74, 124.88, 125.70, 126.27, 127.41, 128.78, 130.17, 131.08, 132.30, 133.22,
         134.50, 135.60, 136.56, 137.49, 138.39, 139.32, 140.69, 142.06, 143.43, 144.82, 145.73, 146.63, 147.56, 148.49, 149.30,
         150.30, 151.70, 152.60, 153.07, 153.97, 155.36, 156.27, 157.18, 158.62, 159.47, 160.38]
CUT_F = [F0] + [round(x * FPS) - 1 for x in CUT_T[1:]] + [F1]
for _S, _a, _b in zip(SHOTS, CUT_F, CUT_F[1:]):
    _S.t0, _S.t1 = _a / FPS, _b / FPS

# 入切转场：(名称, pre 秒, post 秒, kw)。'hard' = 切后 2 帧冲击（推近 + 提亮）；其余走 lib/trans.py。
CUTS = {
    'C02': ('match_line', 0.0, 0.12, {}),                                # 朱笔竖笔 ↔ 红丝线（图形匹配）
    'C03': ('whip_pan_blur', 0.10, 0.12, dict(direction='down')),        # 摇臂沿丝线下摇
    'C04': ('glitch_cut', 0.07, 0.10, dict(seed=4)),
    'C05': ('zoom_through', 0.10, 0.12, dict(center=(1620, 640))),        # 秤盘 → 箱口（推进穿越）
    'C06': ('hard', 0, 0, {}),
    'C07': ('flash_white', 0.0, 0.16, dict(color=(0.85, 0.9, 1.0))),     # 黑箱 → 闪电照亮的脸
    'C08': ('zoom_through', 0.06, 0.08, dict(center=(1050, 400), strength=0.6)),   # 同轴跳进眼睛
    'C09': ('slash_wipe', 0.05, 0.12, dict(angle=-55)),
    'C10': ('hard', 0, 0, {}),
    'C11': ('slash_wipe', 0.05, 0.12, dict(angle=-2)),                   # 横刀光 ↔ 横刀身（图形匹配）
    'C12': ('whip_pan_blur', 0.08, 0.10, dict(direction='right')),
    'C13': ('zoom_through', 0.10, 0.14, dict(center=(1000, 600), variant='glint')),   # 钻进刃口反光
    'C14': ('flash_white', 0.0, 0.18, dict(color=(0.85, 0.9, 1.0))),
    'C15': ('hard', 0, 0, {}),
    'C16': ('flash_white', 0.0, 0.20, {}),                               # 副歌入口：白闪
    'C17': ('hard', 0, 0, {}),
    'C18': ('occlusion', 0.12, 0.12, dict(kind='spear', direction=1)),  # 枪林掠过遮挡
    'C19': ('whip_pan_blur', 0.08, 0.10, dict(direction='left')),
    'C20': ('flash_white', 0.0, 0.14, dict(color=(0.85, 0.9, 1.0))),
    'C21': ('slash_wipe', 0.05, 0.14, dict(angle=-62)),                  # 对角刀线 = 分屏线
    'C22': ('hard', 0, 0, {}),
    'C23': ('zoom_through', 0.08, 0.12, dict(center=(960, 400))),        # 墙远景 → 墙面（同墙推进）
    'C24': ('whip_pan_blur', 0.08, 0.10, dict(direction='right')),
    'C25': ('hard', 0, 0, {}),
    'C26': ('shatter', 0.0, 0.22, dict(center=(1065, 560))),
    'C27': ('flash_white', 0.0, 0.14, {}),
    'C28': ('hard', 0, 0, {}),
    'C29': ('occlusion', 0.10, 0.10, dict(kind='pillar', direction=-1)),   # 坠石遮挡
    'C30': ('flash_red', 0.0, 0.14, {}),
    'C31': ('hard', 0, 0, {}),
    'C32': ('whip_pan_blur', 0.08, 0.10, dict(direction='up')),
    'C33': ('hard', 0, 0, {}),
    'C34': ('slash_wipe', 0.05, 0.12, dict(angle=-62)),
    'C35': ('hard', 0, 0, {}),
    'C36': ('flash_white', 0.0, 0.14, dict(color=(0.85, 0.9, 1.0))),
    'C37': ('hard', 0, 0, {}),
    'C38': ('slash_wipe', 0.05, 0.12, dict(angle=-4)),                   # 斩字的横线 ↔ 横刀（图形匹配）
    'C39': ('hard', 0, 0, {}),
    'C40': ('whip_pan_blur', 0.08, 0.10, dict(direction='left')),
    'C41': ('hard', 0, 0, {}),
}

# 每镜运镜（叠加在镜头内运动上）：z 推拉、x/y 平移、rot 滚转、hh 手持幅度、ease
MOV = {
    'C01': dict(z=(1.0, 1.04)), 'C02': dict(z=(1.02, 1.0), hh=3), 'C03': dict(z=(1.0, 1.06), rot=(-2, 1), hh=3),
    'C04': dict(z=(1.0, 1.08), x=(-30, 30)), 'C05': dict(), 'C06': dict(z=(1.0, 1.05)),
    'C07': dict(z=(1.0, 1.08), hh=5), 'C08': dict(z=(1.0, 1.05), hh=3), 'C09': dict(z=(1.04, 1.0), y=(30, -30), rot=(3, 1), hh=6),
    'C10': dict(z=(1.0, 1.05), x=(0, 40)), 'C11': dict(z=(1.0, 1.03)), 'C12': dict(z=(1.0, 1.08), rot=(0, -2)),
    'C13': dict(), 'C14': dict(z=(1.0, 1.06), hh=4), 'C15': dict(z=(1.0, 1.04), y=(40, -20)),
    'C16': dict(z=(1.0, 1.08)), 'C17': dict(z=(1.03, 1.0), y=(-40, 0), hh=6), 'C18': dict(z=(1.0, 1.12), hh=8),
    'C19': dict(z=(1.0, 1.04), rot=(-4, 4), hh=6), 'C20': dict(z=(1.0, 1.1)), 'C21': dict(z=(1.0, 1.03)),
    'C22': dict(z=(1.0, 1.05), rot=(-1.5, 0.5)), 'C23': dict(z=(1.0, 1.12)), 'C24': dict(hh=7),
    'C25': dict(z=(1.0, 1.08), y=(30, -30), hh=6), 'C26': dict(z=(1.1, 1.0)), 'C27': dict(z=(1.0, 1.06)),
    'C28': dict(z=(1.0, 1.04)), 'C29': dict(z=(1.06, 1.0), hh=4), 'C30': dict(z=(1.0, 1.08), hh=6),
    'C31': dict(z=(1.0, 0.95), hh=6), 'C32': dict(z=(1.0, 1.04), hh=5), 'C33': dict(z=(1.04, 1.0), hh=8),
    'C34': dict(z=(1.0, 1.06), rot=(3, -2), hh=6), 'C35': dict(z=(1.0, 1.06), y=(-30, 20), hh=4),
    'C36': dict(z=(1.0, 1.04), y=(30, -20)), 'C37': dict(z=(1.0, 1.08)), 'C38': dict(z=(1.0, 1.04)),
    'C39': dict(z=(1.08, 1.0), hh=3), 'C40': dict(z=(1.0, 1.1), hh=6), 'C41': dict(z=(1.0, 1.12)),
}
EXPO = {'C13': 1.0, 'C28': 1.0, 'C06': 1.0}
EXPO_ALL = 1.22

# 段内重拍冲击（副歌二）：底鼓强拍 → 推镜 1.5% + 抖 3px（克制，按副歌一 v3 用户意见）
KICKS = [h['t'] for h in K.HM['hits'] if h['t'] >= 134.4 and 'kick' in h['type'] and h['strength'] >= 0.72]

# 插入闪切（帧, 帧数, 类型）：副歌里鼓加花处
INSERTS = {}
for _t, _n, _k in ((131.42, 1, 'white'), (131.96, 1, 'neg'), (132.07, 1, 'white'), (133.79, 2, 'neg'),
                   (140.205, 1, 'white'), (140.408, 2, 'neg'), (148.54, 1, 'white'), (155.385, 2, 'neg'),
                   (160.238, 1, 'white'), (160.679, 2, 'neg')):
    for _i in range(_n): INSERTS[round(_t * FPS) - 1 + _i] = _k


# ---------------- 歌词（屏幕层，字随镜头轻漂）。C01–C03、C06、C10、C20–C23、C25、C28、C37、C38 的字是画内物体，不在此表。
S = Sent
SENTS = [
    # 预副歌二：铅字打字机（权力的话）
    S(29, [0, 1, 2, 3], 1440, 860, size=58, mode='type', glitch='正义', exit='glitch', out=V(122.79)),                 # 谁的正义（C04）
    S(29, [4, 5, 6, 7, 8], 960, 880, size=58, mode='type', glitch='交易', gold='交易', exit='glitch', out=V(123.72)),  # 成了交易的（C05）
    # 「我不低头」起：他自己的重字（Black，朱红「不」逐拍落下）
    S(30, [0, 1, 2, 3], 420, 330, size=96, mode='fly', layout='v', red='不', exit='shatter', out=V(125.62), track=0.12),
    S(30, [4, 5, 6], 960, 900, size=110, mode='fly', red='不命', exit='shatter', out=V(126.22), track=0.15),
    S(30, [7, 8, 9, 10], 1560, 300, size=86, mode='fly', layout='v', red='不', exit='shatter', out=V(127.36), track=0.12),
    S(31, [0, 1, 2, 3], 960, 250, size=78, mode='fly', red='刀', exit='shatter', out=V(130.10)),
    S(31, [4, 5, 6, 7], 960, 860, size=110, mode='fly', red='发烫', exit='shatter', out=V(131.02)),
    # 副歌二：kinetic 飞入 + 碎裂
    S(32, [1, 2], 1500, 860, size=70, mode='fly', exit='shatter', out=V(135.55)),                                    # 未冷（C16）
    S(32, [3, 4, 5], 960, 260, size=120, mode='fly', red='刀', exit='shatter', out=V(137.40), track=0.25),           # 刀不放
    S(33, [0, 1, 2], 560, 300, size=100, mode='fly', exit='shatter', out=V(138.33), offs=[(0, 0), (40, 60), (80, 120)]),   # 谁人来（斜）
    S(33, [6, 7], 960, 880, size=76, mode='fly', exit='shatter', out=V(140.62), track=0.4),                        # 一方
    S(34, [7], 1600, 470, size=260, mode='fly', exit='shatter', out=V(144.70)),                                      # 墙（C24 低频震）
    S(35, [0, 1], 460, 300, size=84, mode='fly', exit='shatter', out=V(145.60), layout='v'),                        # 那就（C25）
    S(35, [3, 4], 1480, 830, size=84, mode='fly', exit='shatter', out=V(146.40)),                                    # 开它
    S(35, [5, 6, 7], 960, 860, size=96, mode='fly', exit='cut', out=V(147.52), track=0.3),                          # 看个天
    S(36, [0, 1, 2], 1600, 230, size=150, mode='fly', layout='v', red='血', exit='shatter', out=V(150.22), track=0.05,
      sizes=[220, 110, 110]),                                                                                          # 血未冷（C30）
    S(36, [3, 4, 5], 420, 260, size=110, mode='fly', red='骨', exit='shatter', out=V(151.62)),                       # 骨未凉（C31）
    S(37, [0, 2, 3], 960, 250, size=70, mode='fly', exit='shatter', out=V(153.00), track=0.6, text='是对错', red='错'),   # 是/对/错（随印）
    S(37, [4, 5, 6, 7], 1560, 260, size=110, mode='fly', layout='v', red='扛', exit='shatter', out=V(155.25), sizes=[110, 110, 110, 200]),   # 我自己扛
    S(38, [0, 1], 960, 830, size=130, mode='fly', exit='shatter', out=V(156.62), track=0.3),                       # 天不
    S(39, [0, 1], 420, 860, size=100, mode='fly', exit='shatter', out=V(159.40)),                                    # 这一
    S(39, [2, 3, 4, 5], 960, 230, size=110, mode='fly', red='刀', exit='shatter', out=V(160.30), track=0.2),        # 刀就是我
    S(39, [6, 7, 8], 960, 860, size=120, mode='fly', red='答', exit='glitch', out=V(161.20), track=0.25),           # 的回答
]


class _Lazy:
    def __init__(self, cls): self.cls = cls; self.t0 = cls.t0; self.t1 = cls.t1; self.obj = None; self.id = cls.__name__
    def render(self, t):
        if self.obj is None: self.obj = self.cls()
        fr = self.obj.render(t)
        return np.clip(fr, 0, 6).astype(np.float32), getattr(self.obj, 'ca', 0)


class Sec:
    def __init__(self):
        self.shots = [_Lazy(s) for s in SHOTS]; self.ids = [s.__name__ for s in SHOTS]

    def idx(self, t):
        for i, s in enumerate(self.shots):
            if t < s.t1 - 1e-6: return i
        return len(self.shots) - 1

    def move(self, i, t, fr):
        s = self.shots[i]; m = MOV.get(self.ids[i], {}); u = clamp((t - s.t0) / (s.t1 - s.t0)); e = eio(u)
        z = lerp(*m.get('z', (1, 1)), e); x = lerp(*m.get('x', (0, 0)), e); y = lerp(*m.get('y', (0, 0)), e)
        rot = lerp(*m.get('rot', (0, 0)), e)
        hx, hy = handheld(t, m.get('hh', 0), seed=i)
        # 重拍冲击
        pk = sum(pulse(t, V(k), 0.10) for k in KICKS if 0 <= t - V(k) < 0.4)
        z *= 1 + 0.015 * min(1.5, pk)
        sx, sy = handheld(t * 7, 3 * min(1.5, pk), seed=99)
        return camxf(fr, z, x + hx + sx, y + hy + sy, rot), (x + hx, y + hy, z)

    def shot_frame(self, i, t):
        fr, ca = self.shots[i].render(t)
        fr, cm = self.move(i, t, fr)
        return fr, ca, cm

    def raw(self, t):
        i = self.idx(t); s = self.shots[i]
        for j in (i, i + 1):
            if j <= 0 or j >= len(self.shots): continue
            b = self.shots[j]; name, pre, post, kw = CUTS.get(self.ids[j], ('hard', 0, 0, {}))
            c = b.t0
            if name == 'hard' or not (c - pre - 1e-6 <= t < c + post - 1e-6): continue
            B, cb, cm = self.shot_frame(j, t)
            A, caa, _ = self.shot_frame(j - 1, t) if (pre > 0 or t < c) else (B, 0, cm)
            p = (t - (c - pre)) / (pre + post)
            if pre == 0: p = 0.5 + 0.5 * p
            if name == 'match_line':
                out = B.copy() if t >= c else A
                k = (t - c) / max(post, 1e-3)
                out = out + np.exp(-((np.arange(W)[None, :] - 900) / 6.0) ** 2)[..., None] * RED * 1.5 * (1 - k)
                return out, max(caa, cb), cm
            out = TR.transition(name, np.clip(A, 0, 4), np.clip(B, 0, 4), p, **kw)
            return out, max(caa, cb, 2), cm
        fr, ca, cm = self.shot_frame(i, t)
        name = CUTS.get(self.ids[i], ('hard',))[0]
        if name == 'hard' and i > 0:
            k = int((t - s.t0) * FPS + 1e-6)
            if k < 2:
                a = 1.0 if k == 0 else 0.45
                fr = camxf(fr, 1 + 0.03 * a) * (1 + 0.18 * a) + 0.04 * a
                ca = max(ca, 4 if k == 0 else 2)
        return fr, ca, cm

    def frame(self, f):
        t = f / FPS
        fr, ca, cm = self.raw(t)
        ins = INSERTS.get(f)
        if ins == 'white': fr = fr * 0.3 + 0.95; ca = max(ca, 3)
        elif ins == 'neg':
            l = fr.mean(-1, keepdims=True); fr = np.clip(1.0 - l * 1.3, 0, 1) * np.array([0.95, 0.97, 1.0], np.float32); ca = max(ca, 5)
        sid = self.ids[self.idx(t)]
        fr = fr * (EXPO_ALL * EXPO.get(sid, 1.0))
        fr = fr + 0.014 * (1 - np.clip(fr, 0, 1))
        dx, dy, z = cm
        for s in SENTS:
            if s.tin[0] - s.lead - 0.05 <= t < s.out + 0.32:
                fr = draw_sent(fr, s, t, cam=(dx * 0.35, dy * 0.35, 1 + (z - 1) * 0.35))
        # 段尾：最后 4 帧切黑（接 Bridge 白纸虚空，由 D 组白闪入）
        if f >= F1 - 4:
            fr = TR.freeze_glitch(fr, None, 0.6 + 0.1 * (f - (F1 - 4)), seed=f) * (0.3 if f >= F1 - 2 else 1.0)
        out = finish(fr, t, ca=ca, vig=0.40, bloom_amt=0.42, red_keep=1.0)
        return out


_SEC = [None]
def frame(f, cache):
    if _SEC[0] is None: _SEC[0] = Sec()
    return to8(_SEC[0].frame(int(f)))


if __name__ == '__main__':
    from PIL import Image
    ts = [float(x) for x in sys.argv[2:]]
    for t in ts:
        Image.fromarray(frame(round(t * FPS), {})).save(f'{sys.argv[1]}_{t:.2f}.png')
        print(t, flush=True)
