"""units/sec_b_verse2.py — B 组 · 间奏 + 主歌二「大局」（帧 2586–3534，86.192–117.8s）
分镜：crew/round4/B_storyboard.md 第 2 节（25 镜，切点按 align.json 字头修正，见 storyboard「第二棒改动」）。
接口：READY / frame(f, cache) -> uint8 (1080,1920,3)
入口：帧 2586 复刻副歌一 S047 墨退（K.ink_blot seed=47，中心 (960,600)，起点 86.192，同一遮罩/墨色）。
出口：V25 末拍推进 + 压暗到 ~30%，刀痕红线留在画面中线，供 C 组硬切。
色：墨 88 / 朱 6 / 金 6。字体即立场：权力 = 思源宋（金/白）；鬼屠 = 马善政 / 志莽行书 / 流江毛草（朱）。"""
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
import fx2
import paper as PA

READY = False
F0, F1 = 2586, 3534
T_IN = 86.192                       # 副歌一 S047 的起点（墨退时间基准）

# 切点（秒）：落拍 / 字头（第二棒按 align.json 逐字时间修正，避免字头跨镜）
CUT_T = [86.192, 87.53, 89.40, 90.77, 91.67, 93.34, 93.97, 94.48, 96.10, 97.52, 98.48, 99.47, 100.58, 102.14,
         103.14, 104.30, 105.88, 107.55, 108.34, 109.55, 110.92, 112.29, 113.60, 115.03, 116.42, 117.80]
CUT_F = [F0] + [int(round(t * FPS)) - 1 for t in CUT_T[1:-1]] + [F1]

INK_C = np.array([0.015, 0.014, 0.016], np.float32)
COLDC = np.array([0.80, 0.86, 0.96], np.float32)


def prs(li, ks, t, dur=0.18):
    return tp(li, ks, t, dur)


def hitp(t, tt, decay=0.12):
    """视觉帧 tt 起的冲击包络"""
    return pulse(t, V(tt), decay) * (t >= V(tt))


def lum3(v, tint=(1.0, 1.0, 1.0)):
    return np.stack([v * tint[0], v * tint[1], v * tint[2]], -1).astype(np.float32)


# ------------------------------------------------------------------ 共用场景件
def city_wall_far(seed, y_top=520, y_bot=760, tone=0.10, crenel=True):
    """远景城墙剪影（砖纹 + 城垛 + 城楼），返回 (rgb, mask)"""
    wall, seam = K.brick_wall(2200, 600, 70, 30, 4, seed)
    h = y_bot - y_top
    wl = cv2.resize(wall[:400, :2200], (W, h)) * 0.55
    m = np.zeros((H, W), np.float32); m[y_top:y_bot] = 1
    rgb = np.zeros((H, W, 3), np.float32); rgb[y_top:y_bot] = wl * (tone / 0.12)
    if crenel:
        for xx in range(-20, W + 40, 58):
            cv2.rectangle(m, (xx, y_top - 22), (xx + 30, y_top), 1.0, -1)
            cv2.rectangle(rgb, (xx, y_top - 22), (xx + 30, y_top), (tone * 0.8,) * 3, -1)
        # 城楼（重檐）
        cx = int(W * 0.62)
        for k, (wd, hh, yy) in enumerate(((420, 40, y_top - 150), (300, 34, y_top - 230))):
            pts = np.int32([[cx - wd // 2 - 40, yy + hh], [cx + wd // 2 + 40, yy + hh], [cx + wd // 2, yy], [cx - wd // 2, yy]])
            cv2.fillConvexPoly(m, pts, 1.0); cv2.fillConvexPoly(rgb, pts, (tone * 0.7,) * 3)
            cv2.rectangle(m, (cx - wd // 2 + 30, yy + hh), (cx + wd // 2 - 30, yy + hh + 80 - k * 10), 1.0, -1)
            cv2.rectangle(rgb, (cx - wd // 2 + 30, yy + hh), (cx + wd // 2 - 30, yy + hh + 80 - k * 10), (tone * 0.85,) * 3, -1)
    return rgb, m


def night_sky(seed, top=(0.035, 0.04, 0.05), bot=(0.16, 0.17, 0.19), horizon=0.7, cloud=0.9):
    return K.sky(seed=seed, cloud=cloud, horizon=horizon, top=top, bot=bot, stars=0.0).copy()


def sword_stuck():
    """斜插泥地的真刀（blade_face 生图刀身 + 护手，刀尖朝下）+ 代码缠绳刀柄。返回 (F, A, grip_y, guard_y)"""
    if 'stuck' in BK._BL: return BK._BL['stuck']
    F0_, A0_ = BK.blade_prop()                       # 刃尖朝上，底部护手，约 2690×800
    F0_ = F0_[::-1].copy(); A0_ = A0_[::-1].copy()   # 翻转：护手在上，刃尖朝下
    hb, wb = A0_.shape
    HL = 640                                         # 刀柄长（源像素）
    F = np.zeros((hb + HL, wb, 3), np.float32); A = np.zeros((hb + HL, wb), np.float32)
    F[HL:] = F0_; A[HL:] = A0_
    cx = 398; hw = 46
    yy, xx = np.mgrid[0:HL, 0:wb].astype(np.float32)
    grip = ((np.abs(xx - cx) < hw * (1 - 0.05 * (1 - yy / HL))) & (yy > 40)).astype(np.float32)
    # 缠绳菱格
    ph = ((yy + (xx - cx) * 0.9) % 52 < 26).astype(np.float32); ph2 = ((yy - (xx - cx) * 0.9) % 52 < 26).astype(np.float32)
    dia = ph * ph2
    shade = np.clip(1 - ((xx - cx) / hw) ** 2, 0, 1) ** 0.5
    gcol = (0.05 + 0.05 * dia + 0.07 * shade)[..., None] * np.array([1.0, 0.88, 0.80], np.float32)
    # 柄头（金属）
    pom = (((xx - cx) / (hw * 1.15)) ** 2 + ((yy - 50) / 40) ** 2 < 1).astype(np.float32)
    gcol = gcol * (1 - pom[..., None]) + pom[..., None] * np.array([0.30, 0.26, 0.18], np.float32) * (0.6 + 0.6 * shade[..., None])
    gm = np.maximum(grip, pom)
    F[:HL] = gcol; A[:HL] = cv2.GaussianBlur(gm, (0, 0), 1.0)
    BK._BL['stuck'] = (F, A, HL, HL + 120)
    return BK._BL['stuck']


def paste_sword(fr, base, ang, length, alpha=1.0, wet=1.0, t=0.0, blur=0.0):
    """把 sword_stuck 贴到屏幕：base=刀身入泥点（屏幕），ang=偏离竖直角度（度，正=向右倾），length=刀从柄头到入泥点屏幕长度"""
    F, A, HL, gy = sword_stuck()
    hs, ws = A.shape
    y_in = HL + int((hs - HL) * 0.62)                # 刀身 62% 处入泥
    s = length / y_in
    M = cv2.getRotationMatrix2D((398.0, float(y_in)), -ang, s)
    M[0, 2] += base[0] - 398.0; M[1, 2] += base[1] - y_in
    Fw = cv2.warpAffine(F, M, (W, H)); Aw = cv2.warpAffine(A, M, (W, H))
    if blur > 0.6:
        Fw = K.blur_ds(Fw * Aw[..., None], blur, 2); Aw = K.blur_ds(Aw, blur, 2); Fw = Fw / np.clip(Aw[..., None], 1e-3, None)
    Aw = Aw * alpha
    # 冷月顶光：刃面高光带 + 雨水湿反光
    x, y = K.grid()
    Fw = Fw * (0.55 + 0.15 * wet) + np.exp(-((y - base[1] + length * 0.55) / (length * 0.35)) ** 2)[..., None] * COLDC * 0.12 * wet
    out = fr * (1 - Aw[..., None]) + Fw * Aw[..., None]
    return out, Aw, M


def glitch_text(fr, L, k, seed=0, t=0.0):
    """RGB 错位故障字：k 0..1 强度"""
    out = T.compose(fr, L, shadow=0.7, glow=0.2, glow_color=(1.0, 0.6, 0.3))
    if k < 0.02: return out
    r = np.random.default_rng(int(seed + t * 30))
    A = L.A
    sh = int(10 + 30 * k)
    Ar = np.roll(A, sh, 1); Ab = np.roll(A, -sh, 1)
    # 横向条带撕裂
    for _ in range(int(3 + 6 * k)):
        y0 = int(r.uniform(120, 940)); hh = int(r.uniform(6, 40)); dx = int(r.uniform(-80, 80) * k)
        Ar[y0:y0 + hh] = np.roll(Ar[y0:y0 + hh], dx, 1); Ab[y0:y0 + hh] = np.roll(Ab[y0:y0 + hh], -dx, 1)
    out[..., 0] += Ar * 0.8 * k; out[..., 2] += Ab * 0.9 * k
    out[..., 1] -= (Ar + Ab) * 0.15 * k
    return out


def sil_over(fr, tmp, rim_col, rim_w=5, amt=1.0, blur=0.0):
    """把画在黑底上的纸人（tmp）变成逆光剪影：近黑实体 + 背光金边（不要卡通脸）"""
    m = np.clip(tmp.max(-1) * 10, 0, 1)
    if blur > 0: m = K.blur_ds(m, blur, 2)
    er = cv2.erode(m, np.ones((rim_w * 2 + 1, rim_w * 2 + 1), np.uint8))
    rim = np.clip(m - er, 0, 1)
    rim = rim * sstep(0.0, 1.0, np.clip(-cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=5) * 0.05 + 0.4, 0, 1))   # 上缘更亮（逆光在上）
    out = fr * (1 - m[..., None]) + m[..., None] * np.array([0.012, 0.011, 0.012], np.float32)
    return out + rim[..., None] * np.asarray(rim_col, np.float32) * amt + K.blur_ds(rim, 10)[..., None] * np.asarray(rim_col, np.float32) * 0.3 * amt


# =====================================================================================
# V01 86.19–87.53 ELS→MS：墨退，夜雨倾盆；真刀斜插泥里，雨顺刃流下、积水倒影；后景雨雾中城墙轮廓
#   前＝虚焦草叶（焦点由草→刀）；中＝斜插的刀；后＝雾中城墙/城楼。小款「鬼屠」+ 屠字朱印（与副歌一 S047 同款）
class V01(Shot):
    def __init__(self):
        sk = night_sky(4701, top=(0.03, 0.035, 0.045), bot=(0.20, 0.21, 0.235), horizon=0.72)
        wl, wm = city_wall_far(4702, 560, 790, tone=0.075)
        wl = K.blur_ds(wl, 5); wm = K.blur_ds(wm, 5)
        sk = sk * (1 - wm[..., None] * 0.8) + wl * wm[..., None] * 0.8
        x, y = K.grid()
        self.gm = sstep(828, 850, y)
        mud = (0.025 + 0.035 * cv2.resize(K.fbm(H // 2, W // 2, 4703, 10, 6), (W, H)))[..., None] * np.array([1.0, 0.98, 0.96], np.float32)
        self.bg = sk * (1 - self.gm[..., None]) + mud * self.gm[..., None]
        self.puddle = np.clip((cv2.resize(K.fbm(H // 8, W // 8, 4704, 4, 4), (W, H)) - 0.45) * 5, 0, 1) * self.gm
        self.grass = K.Grass(4705, n=40, y_base=1010, height=(160, 420), width=(5, 13))
        self.rain = fx2.RainSheet(4706, density=0.9)
        self.spl = fx2.Splashes(4707)
        self.fog = fx2.Fog(4708)
        T.warm('鬼屠', 'song', 40)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        fr = self.bg.copy()
        fr = self.fog.draw(fr, t, density=0.75, y0=480, y1=860, color=(0.36, 0.38, 0.42), speed=(40, 0), top_fade=300)
        # 刀：焦点转移（开头刀虚、草实 → 刀实、草虚）
        focus = sstep(0.15, 0.75, u)
        base = (1060, 905)
        fr, Aw, M = paste_sword(fr, base, -14, 760, wet=1.0, t=t, blur=7 * (1 - focus))
        # 积水倒影（上下翻 + 涟漪抖动）
        ref = fr[::-1].copy()
        ref = np.roll(ref, int(2 * (2 * 846 - H)) - 0, 0) if False else cv2.warpAffine(ref, np.float32([[1, 0, 0], [0, 1, 2 * 846 - H]]), (W, H))
        nz = cv2.resize(K.tile_crop(K.tile(256, 4709, 6, 3), t * 30, t * 70, H // 8, W // 8), (W, H))
        ref = cv2.remap(ref, *[a.astype(np.float32) for a in (K.grid()[0] + (nz - 0.5) * 14, K.grid()[1])], cv2.INTER_LINEAR)
        fr = fr + ref * (self.puddle * 0.55)[..., None]
        # 雨水沿刃下流（亮点）
        F, A, HL, gy = sword_stuck()
        for k_ in range(5):
            s = ((t * 0.8 + k_ * 0.21) % 1.0)
            src = np.array([398 + 30 * math.sin(k_ * 2), gy + 60 + (A.shape[0] * 0.62 - gy - 60) * s, 1.0])
            p = M @ src
            cv2.circle(fr, (int(p[0]), int(p[1])), 3, (0.85, 0.9, 1.0), -1, cv2.LINE_AA)
        fr = self.spl.draw(fr, t, y_far=840, y_near=1000, rate=140, size=1.0, amt=0.8)
        fr = fx2.ripples(fr, t, [(base[0] - 30, base[1] + 6), (base[0] + 40, base[1] + 10)], [86.3, 86.9], amt=0.5, flat=0.18)
        fr = self.rain.draw(fr, t, intensity=1.0, angle=10, light=(1300, 300), light_r=700)
        # 前景草叶（虚焦，随焦点转移由实到虚）
        gmask = self.grass.mask(t, wind=1.2, blur=2 + 22 * focus)
        fr = fr * (1 - gmask[..., None] * 0.92) + gmask[..., None] * np.array([0.012, 0.014, 0.016], np.float32)
        fr = fr * np.array([0.92, 0.97, 1.08], np.float32)
        fr = cam(fr, 1.08 - 0.06 * e, 0, 50 - 70 * e)               # 摇臂升 + 微拉
        # 墨退（与副歌一 S047 同一遮罩、同一时间、同一墨色）
        if t < T_IN + 0.3:
            m = K.ink_blot(0.93 + 0.07 * eout((t - T_IN) / 0.2, 2), seed=47, center=(960, 600), reverse=True)
            fr = over(fr, m, RED * 0.22 + 0.015)
        pk = clamp01((t - 86.6) / 0.6)
        if pk > 0:
            fr = T.ink_bleed_in(fr, '鬼屠', [pk, clamp01(pk * 1.2 - 0.1)], style='h', size=34, fname='serif_r', color='white',
                                box=(120, 830, 300, 900), style_kw=dict(shadow=0.6, glow=0.1))
            fr = T.stamp_seal(fr, '屠', 330, 865, 54, clamp01((t - 86.95) / 0.5), style='yin', seed=47)
        return fr, 0


# =====================================================================================
# V02 87.53–89.40 ECU：刀镡上悬着一滴雨，雨滴里倒映城墙与太上皇背光剪影（scepter 精灵 look=sil）；88.91 雷闪；镜头追进雨滴
class V02(Shot):
    T_THUNDER = 88.905
    def __init__(self):
        sc = night_sky(4801, top=(0.03, 0.035, 0.05), bot=(0.22, 0.23, 0.26), horizon=0.62, cloud=1.0)
        wl, wm = city_wall_far(4802, 600, 1080, tone=0.09)
        sc = sc * (1 - wm[..., None]) + wl * wm[..., None]
        self.scene0 = sc
        self.sp = spr('S050/scepter')
        self.bolt = fx2.Lightning(4803, p0=(1500, 0), p1=(1180, 600), branches=5)
        # 背景：刀镡/刀柄的微距虚焦（真刀贴图放大）
        bgc = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.022, 0.026], np.float32)
        bg, Aw, _ = paste_sword(bgc, (900, 2900), 4, 3600, wet=1.2)
        self.bg = K.blur_ds(bg, 14) * 0.38
        x, y = K.grid(); r = np.random.default_rng(4805)
        for _ in range(26):                          # 焦外雨珠光斑
            cx_, cy_, rr = r.uniform(0, W), r.uniform(120, 960), r.uniform(10, 46)
            self.bg += (np.exp(-(((x - cx_) ** 2 + (y - cy_) ** 2) / rr ** 2) ** 3) * r.uniform(0.03, 0.09))[..., None] * COLDC

    def render(self, t):
        u = self.u(t)
        tl = V(self.T_THUNDER)
        env = 0.0 if t < tl else math.exp(-(t - tl) / 0.25)
        # 倒影小场景（背光剪影：太上皇执杖，站在城头）
        sc = self.scene0.copy()
        if t >= tl - 0.05:
            sc = self.bolt.draw(sc, t, tl, intensity=1.0)
        sc, _ = put(sc, self.sp, t, 820, (900, 760), grade=GR_COLD, rim=dict(col=(0.85, 0.88, 1.0), amt=2.4 + 3 * env, dir=(0.6, -0.8)),
                    look='sil', wind_amp=3, shade=None, wrap=0.1)
        fr = self.bg.copy()
        fr = fr * (1 + 1.2 * env)
        fr = K.rain(fr, t, 4804, n=220, amt=0.22, speed=900, angle=4, near=0.5, length=(60, 160))
        z = 1.0 + 2.2 * ein(clamp01(u), 2.2)
        cx, cy = 960, 470 + 50 * u
        R = 150
        x, y = K.grid()
        Rz = R * z
        dist = np.sqrt((x - cx) ** 2 + ((y - cy) * 0.92) ** 2)
        rn = np.clip(dist / max(Rz, 1), 0, 1)
        fe = rn ** 0.65
        k = 1.0 / (0.9 + 0.6 * (1 - u))
        sx = (960 + (x - cx) / max(Rz, 1) * 960 * fe / np.maximum(rn, 1e-3) * k).astype(np.float32)
        sy = (540 - (y - cy) / max(Rz, 1) * 960 * fe / np.maximum(rn, 1e-3) * k).astype(np.float32)   # 倒置
        inner = cv2.remap(sc, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = np.clip((Rz - dist) / 2, 0, 1)
        rim = np.exp(-((dist - Rz) / (2 + 0.01 * Rz)) ** 2) * 0.6
        spec = np.exp(-(((x - cx + Rz * 0.35) ** 2 + (y - cy + Rz * 0.42) ** 2) / (Rz * 0.1 + 2) ** 2)) * 0.9
        # 水滴挂在刀镡下沿：上方一条细水颈
        neck = np.exp(-((x - cx) / (8 + 0.04 * Rz)) ** 2) * (y < cy - Rz * 0.9) * (y > cy - Rz * 1.6) * (1 - u)
        fr = fr * (1 - m[..., None]) + inner * m[..., None] * 0.92
        fr += (rim + spec * m + neck * 0.3)[..., None] * np.array([0.85, 0.9, 1.0], np.float32)
        if tl <= t < tl + 1 / 30 + 1e-3: fr = fr * 0.5 + 0.7
        elif t > tl: fr = fr * (1 + 0.5 * pulse(t, tl + 0.04, 0.15))
        return fr, (3 if abs(t - tl) < 0.04 else 0)


# =====================================================================================
# V03 89.40–90.77 LS：雨中纸人队列从队首到队尾依次鞠躬（顾全大局）；前景雨帘，后景雾中金光；
#   「世人教我顾全」思源宋 3 个深度的空间字，随横移视差
class V03(Shot):
    def __init__(self):
        self.crowd = PA.PaperCrowd(4903, rows=3, cols=10, view='side', x_span=(120, 1820), y_front=930, h_front=300)
        sk = night_sky(4901, top=(0.03, 0.03, 0.04), bot=(0.13, 0.12, 0.11), horizon=0.75)
        x, y = K.grid()
        sk += np.exp(-(((x - 1450) / 520) ** 2 + ((y - 560) / 260) ** 2))[..., None] * GOLD * 0.22
        self.bg = sk
        self.fog = fx2.Fog(4902)
        self.rain = fx2.RainSheet(4904, density=0.8)
        T.warm('世人教我顾全', 'serif_sb', 120)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        pan = -180 * e
        fr = self.bg.copy()
        fr = cam(fr, 1.0, -pan * 0.2, 0)
        fr = self.fog.draw(fr, t, density=0.5, y0=480, y1=900, color=(0.34, 0.33, 0.32), speed=(30, 0), light=(1450, 560),
                           light_r=600, light_color=GOLD * 0.8, cam=(pan * 0.3, 0))
        bow = clamp01((t - (V(CT(20, 4)) - 0.4)) / 1.0)
        fr, _ = self.crowd.draw(fr, t, bow=eio(bow), wave=0.9, wet=0.6, light='back', light_color=GOLD, dof=1.2, x_off=pan * 0.7)
        pr = prs(20, [0, 1, 2, 3, 4, 5], t)
        items = [dict(text='世人', x=-520, y=-170, z=1700, size=150, color='white', font='serif_sb', style='h', prog=pr[0:2]),
                 dict(text='教我', x=80, y=-60, z=1250, size=150, color='white', font='serif_sb', style='h', prog=pr[2:4]),
                 dict(text='顾全', x=420, y=110, z=850, size=160, color='gold', font='serif_black', style='h', prog=pr[4:6])]
        fr = T.kinetic_track(fr, items, cam=(-pan * 1.4, 0, 0), focal=1000, focus_z=1250, dof=0.004)
        fr = self.rain.draw(fr, t, intensity=1.1, angle=12, light=(1450, 560), light_r=600, cam=(pan, 0))
        fr = cam(fr, 1.02 + 0.02 * e, 0, 0, 0.6 * e)
        return fr, 0


# =====================================================================================
# V04 90.77–91.67 仰 CU：鞠躬的纸人头顶剪影压在画面下沿；「大局」金色宋体巨字悬在雨里、雨打在字上溅开；「退让」小字逐字压低
class V04(Shot):
    T_DA = 90.62
    def __init__(self):
        sk = night_sky(4911, top=(0.02, 0.022, 0.03), bot=(0.09, 0.085, 0.08), horizon=0.9)
        self.bg = sk
        self.figs = [PA.PaperFigure(4912 + i, view='front') for i in range(5)]
        self.rain = fx2.RainSheet(4913, density=1.0)
        T.warm('大局', 'serif_black', 300); T.warm('退让', 'serif_sb', 90)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        fr += np.exp(-(((x - 960) / 700) ** 2 + ((y - 380) / 300) ** 2))[..., None] * GOLD * 0.12
        # 「大局」金字（已在上一镜字头出现 → 这里落定 + 被雨打）
        L = T.Layer()
        k1 = clamp01((t - self.t0) / 0.18)
        for ch, xx in (('大', 760), ('局', 1160)):
            T.place(L, ch, xx, 400 - 30 * eio(u), 330, color='gold', fname='serif_black', alpha=1.0, scale=1.12 - 0.12 * eout(k1))
        fr = T.compose(fr, L, shadow=0.6, glow=0.55, glow_color=(1.0, 0.55, 0.2))
        # 雨打字面：字的上缘溅起细小水花
        edge = np.clip(L.A - np.roll(L.A, 6, 0), 0, 1)
        ys, xs = np.nonzero(edge[::8, ::8] > 0.5)
        r = np.random.default_rng(int(t * 30))
        if len(xs):
            idx = r.choice(len(xs), min(26, len(xs)), replace=False)
            for i in idx:
                px, py = xs[i] * 8, ys[i] * 8
                for _ in range(3):
                    a = r.uniform(-2.6, -0.5); d = r.uniform(6, 22)
                    cv2.line(fr, (px, py), (int(px + math.cos(a) * d), int(py + math.sin(a) * d)), (0.75, 0.78, 0.85), 1, cv2.LINE_AA)
        # 「退让」：逐字从上压低（宋体小字，像被按下去的头）
        pr = prs(20, [8, 9], t)
        L2 = T.Layer()
        for i, (ch, pc) in enumerate(zip('退让', pr)):
            if pc <= 0: continue
            T.place(L2, ch, 1460 + i * 120, 600 + 90 * eout(pc, 3) + i * 40, 96, color='white', fname='serif_sb', alpha=min(1, pc * 2))
        fr = T.compose(fr, L2, shadow=0.8)
        fr = self.rain.draw(fr, t, intensity=1.2, angle=4, light=(960, 380), light_r=700)
        # 纸人头顶剪影（下沿，前景，逆光金边，随鞠躬下压）
        bowk = eio(clamp01((t - V(CT(20, 8)) + 0.2) / 0.8))
        tmp = np.zeros_like(fr)
        for i, f_ in enumerate(self.figs):
            xx = 180 + i * 390
            tmp, _ = f_.draw(tmp, xx, 1080 + 560 + 60 * bowk + 30 * (i % 2), 900, t, bow=0.25 * bowk, light='back', light_color=GOLD, rim=0.0)
        fr = sil_over(fr, tmp, GOLD, rim_w=5, amt=1.3)
        k = hitp(t, 91.314, 0.12)
        fr = cam(fr, 1.04 + 0.04 * eio(u) + 0.03 * k, 0, -20 * eio(u))
        return fr, (2 if k > 0.5 else 0)


# =====================================================================================
# V05 91.67–93.34 MS：越过人群背影，烛光照亮的纸屏风，屏风上投着太上皇执权杖的巨大影子；「可大局背后」宋体竖排印在屏风上
class V05(Shot):
    def __init__(self):
        r = np.random.default_rng(5001)
        pap = BK.paper(5001, (0.86, 0.72, 0.50))
        x, y = K.grid()
        # 屏风格栅（木框 + 竖格）
        lat = np.zeros((H, W), np.float32)
        for xx in range(160, W, 400): cv2.rectangle(lat, (xx - 14, 0), (xx + 14, H), 1.0, -1)
        for yy in (150, 560, 940): cv2.rectangle(lat, (0, yy - 10), (W, yy + 10), 1.0, -1)
        self.lat = cv2.GaussianBlur(lat, (0, 0), 1.2)
        self.pap = pap
        self.sp = spr('S050/scepter')
        self.figs = [PA.PaperFigure(5010 + i, view='front') for i in range(6)]
        T.warm('可大局背后', 'serif_black', 110)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        x, y = K.grid()
        fl = 0.9 + 0.1 * math.sin(t * 17) * math.sin(t * 7.3)
        glow = np.exp(-(((x - 900) / 900) ** 2 + ((y - 1000) / 520) ** 2))
        fr = self.pap * (0.18 + 0.95 * glow[..., None] * fl) * np.array([1.0, 0.8, 0.55], np.float32)
        # 太上皇执杖的投影（精灵 look=shadow，柔边，随烛焰轻晃）
        sh = np.zeros_like(fr)
        _, Aw = put(sh, self.sp, t, 1150, (930 + 12 * math.sin(t * 3), 1060), look='shadow', wind_amp=4, shade=None, wrap=0.0)
        Aw = K.blur_ds(Aw, 7 + 2 * math.sin(t * 5), 2)
        fr = fr * (1 - 0.78 * Aw[..., None])
        # 宋体竖排印在屏风上（乘法墨色，透纸纹）
        pr = prs(21, [0, 1, 2, 3, 4], t)
        L = T.Layer()
        for i, (ch, pc) in enumerate(zip('可大局背后', pr)):
            if pc <= 0: continue
            T.place(L, ch, 1480, 230 + i * 128, 112, color='ink', fname='serif_black', alpha=min(1, pc * 2) * 0.9, scale=1 + 0.25 * (1 - eout(pc)))
        fr = fr * (1 - L.A[..., None] * 0.95) + L.A[..., None] * np.array([0.05, 0.012, 0.008], np.float32)
        # 格栅木框（暗）
        fr = fr * (1 - self.lat[..., None] * 0.88) + self.lat[..., None] * np.array([0.03, 0.02, 0.015], np.float32)
        # 前景：人群背影（纸人背面，深色剪影、景深虚化），dolly 向前 → 背影下沉出画
        sink = 140 * e
        tmp = np.zeros_like(fr)
        for i, f_ in enumerate(self.figs):
            xx = -60 + i * 400 + (i % 2) * 60
            tmp, _ = f_.draw(tmp, xx, 1080 + 420 + sink + 50 * (i % 3), 820, t, bow=0.35, light='back', light_color=GOLD, rim=0.0)
        fr = sil_over(fr, tmp, GOLD * 0.8, rim_w=4, amt=0.9, blur=5)
        fr = fr + 0.03 * glow[..., None] * GOLD
        fr = cam(fr, 1.0 + 0.07 * e, 0, -30 * e)
        return fr, 0


# =====================================================================================
# V06 93.34–93.97 CU：屏风后真相——太上皇半张脸被烛光从下照亮，冷笑（tsh_smirk）。图形匹配：影子的头 → 真人头同位置
class V06(Shot):
    def __init__(self):
        self.sp = spr('tsh_smirk')
        self.bg = BK.dark_bg(5101, 0.01, 0.035, tint=(1.15, 1.0, 0.9))
        T.warm('是谁的', 'serif_sb', 84)

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        fl = 0.85 + 0.15 * math.sin(t * 19) * math.sin(t * 6.1)
        fr = self.bg + np.exp(-(((x - 960) / 600) ** 2 + ((y - 1100) / 400) ** 2))[..., None] * np.array([0.35, 0.16, 0.05], np.float32) * fl
        fr, _ = put_at(fr, self.sp, t, (0.42, 0.30), (930, 400), 2100,
                       grade=dict(expo=0.78 * fl, sat=0.7, tint=(1.08, 0.94, 0.80), lift=0.0, gamma=1.15),
                       rim=dict(col=(1.0, 0.62, 0.30), amt=2.0, dir=(0.0, 1.0)), wind_amp=2, breathe=0.004,
                       shade=(1300, 1900, 0.35, 300, -0.15), wrap=0.25)
        # 底光：脸下半更亮（烛焰），上半沉入暗部
        fr = fr * (0.55 + 0.55 * sstep(150, 760, y))[..., None]
        pr = prs(21, [5, 6, 7], t)
        L = T.Layer()
        for i, (ch, pc) in enumerate(zip('是谁的', pr)):
            if pc <= 0: continue
            T.place(L, ch, 260, 330 + i * 110, 92, color='gold' if ch == '谁' else 'white', fname='serif_sb', alpha=min(1, pc * 2))
        fr = T.compose(fr, L, shadow=0.85, glow=0.2, glow_color=(1.0, 0.55, 0.2))
        fr = cam(fr, 1.0 + 0.05 * eio(u), 0, 0)
        return fr, 0


# =====================================================================================
# V07 93.97–94.48 CU：太上皇的手攥紧权杖（grip_scepter），「权」金色巨字在烛焰热浪里扭曲
class V07(Shot):
    T_QUAN = 94.02
    def __init__(self):
        self.sp = spr('S051/grip_scepter')
        self.bg = BK.dark_bg(5201, 0.012, 0.04, tint=(1.2, 1.0, 0.85))
        T.warm('权', 'serif_black', 520)

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        fr = self.bg + np.exp(-(((x - 1250) / 700) ** 2 + ((y - 700) / 500) ** 2))[..., None] * np.array([0.22, 0.10, 0.03], np.float32)
        # 「权」在人物背后（后景巨字，热浪扭曲）
        L = T.Layer()
        pq = clamp01((t - (V(self.T_QUAN) - 0.1)) / 0.2)
        T.place(L, '权', 1340, 520, 560, color='gold', fname='serif_black', alpha=pq, scale=1.4 - 0.4 * eout(pq, 3))
        Lf = T.compose(np.zeros_like(fr), L, shadow=0.0, glow=0.7, glow_color=(1.0, 0.5, 0.15))
        Lf = fx2.heat_haze(Lf, t, amt=9.0, scale=36, speed=1.6, seed=52)
        fr = fr + Lf
        k = eout(clamp01((t - self.t0) / 0.25), 3)                 # 攥紧：手从略松到紧（推进 + 微抖）
        fr, _ = put_at(fr, self.sp, t, (0.20, 0.22), (520, 420), 2600 * (1.0 + 0.04 * k),
                       grade=dict(expo=0.75, sat=0.65, tint=(1.06, 0.95, 0.82), lift=0.0, gamma=1.15),
                       rim=dict(col=(1.0, 0.66, 0.32), amt=2.2, dir=(0.9, -0.3)), wind_amp=2, shade=(900, 1900, 0.3, 700, 0.3), wrap=0.3)
        hk = hitp(t, self.T_QUAN, 0.1)
        dx, dy, dr = shake(t, 3 + 10 * hk, 11, 52)
        fr = cam(fr, 1.0 + 0.12 * eio(u) + 0.04 * hk, dx, dy, dr)
        return fr, (3 if hk > 0.5 else 0)


# =====================================================================================
# V08 94.48–96.10 ECU 下摇：沿权杖杖身下摇（grip_scepter 局部，与 V07 不同景别/方向）；「柄在」小字，「嚣张」RGB 错位故障
class V08(Shot):
    def __init__(self):
        self.sp = spr('S051/grip_scepter')
        self.bg = BK.dark_bg(5301, 0.01, 0.03, tint=(1.1, 1.0, 0.9))
        T.warm('柄在', 'serif_sb', 100); T.warm('嚣张', 'serif_black', 300)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        fr = self.bg.copy()
        # 杖身：精灵内杖在 x≈0.17，从顶端 (0.17,0.0) 下摇到手 (0.2,0.3)；镜像（与 V07 方向相反）+ 斜 12°
        sy = 0.02 + 0.30 * e
        fr, _ = put_at(fr, self.sp, t, (0.17, sy), (1100, 520), 5200, rot=-12,
                       grade=dict(expo=0.8, sat=0.7, tint=(1.08, 0.95, 0.80), lift=0.0, gamma=1.1),
                       rim=dict(col=(1.0, 0.7, 0.35), amt=2.6, dir=(-0.9, -0.2)), wind_amp=0, breathe=0.0, shade=None, wrap=0.2,
                       mblur=(0, 18 * math.sin(math.pi * u)))
        x, y = K.grid()
        fr += np.exp(-((x - 1100) / 160) ** 2)[..., None] * np.array([0.12, 0.07, 0.03], np.float32) * (0.6 + 0.4 * math.sin(t * 9))
        pr = prs(21, [9, 10, 11, 12], t)
        L = T.Layer()
        for i, ch in enumerate('柄在'):
            if pr[i] > 0: T.place(L, ch, 420, 330 + i * 120, 104, color='white', fname='serif_sb', alpha=min(1, pr[i] * 2))
        fr = T.compose(fr, L, shadow=0.85)
        L2 = T.Layer()
        for i, ch in enumerate('嚣张'):
            if pr[2 + i] > 0: T.place(L2, ch, 560 + i * 330, 700, 300, color='gold', fname='serif_black', alpha=min(1, pr[2 + i] * 2),
                                      scale=1.3 - 0.3 * eout(pr[2 + i]))
        gk = max(hitp(t, CT(21, 11), 0.18), hitp(t, CT(21, 12), 0.25), 0.25 * (0.5 + 0.5 * math.sin(t * 23)) * (pr[3] > 0.99))
        fr = glitch_text(fr, L2, gk, seed=53, t=t)
        fr = cam(fr, 1.04, 0, 0, 0)
        return fr, (3 if gk > 0.6 else 0)


# =====================================================================================
# V09 96.10–97.52 MS：雨中天平：左盘纸人不断落下越堆越高，横梁纹丝不动（右盘金印）。匹配：权杖竖线 → 天平立柱
#   「牺牲多少才」宋体字随纸人一起落进秤盘，「牺牲」朱
import props as PR


class V09(Shot):
    def __init__(self):
        self.bal = PR.Balance(5401, people=70)
        sk = night_sky(5402, top=(0.02, 0.022, 0.028), bot=(0.10, 0.10, 0.11), horizon=0.8)
        x, y = K.grid()
        sk += np.exp(-(((x - 960) / 520) ** 2 + ((y - 300) / 300) ** 2))[..., None] * COLDC * 0.08
        self.bg = sk
        self.rain = fx2.RainSheet(5403, density=0.8)
        self.figs = [PA.PaperFigure(5410 + i, view='side') for i in range(6)]
        self.fog = fx2.Fog(5404)
        T.warm('牺牲多少才', 'serif_black', 120)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        fr = self.bg.copy()
        fr = self.fog.draw(fr, t, density=0.4, y0=600, y1=960, color=(0.3, 0.31, 0.33))
        pile = 0.25 + 0.75 * clamp01((t - self.t0) / 1.3)
        fr, info = self.bal.draw(fr, t, 960, 360, 1180, tilt=0.0, sway=0.15, left_people=pile, right='seal', rim_color=GOLD, rim=1.2)
        pl = np.array(info.get('pan_L', (420, 640)), float)
        # 落下的纸人（不同时刻、旋转下坠）
        for i, f_ in enumerate(self.figs):
            ti = self.t0 + 0.05 + i * 0.22
            q = (t - ti) / 0.55
            if 0 < q < 1:
                yy = -150 + (pl[1] - 40 + 150) * q ** 2
                fr, _ = f_.draw(fr, pl[0] - 90 + 40 * i, yy, 120, t, tilt=200 * q * (1 if i % 2 else -1), light='back', light_color=GOLD, rim=1.0)
        # 字：从上落进左盘，落定后叠在纸人堆上
        pr = prs(22, [0, 1, 2, 3, 4], t, 0.3)
        L = T.Layer(); Lr = T.Layer()
        for i, (ch, pc) in enumerate(zip('牺牲多少才', pr)):
            if pc <= 0: continue
            q = eio(pc)
            tx, ty = pl[0] - 150 + (i % 3) * 120, pl[1] - 120 - (i // 3) * 110
            yy = 160 + (ty - 160) * q ** 1.6
            T.place(Lr if ch in '牺牲' else L, ch, tx, yy, 108, color='red' if ch in '牺牲' else 'white', fname='serif_black',
                    rot=(1 - q) * (30 if i % 2 else -30) + (6 if i % 2 else -5) * q)
        fr = T.compose(fr, L, shadow=0.8, glow=0.1)
        fr = T.compose(fr, Lr, **T.auto_style('red'))
        fr = self.rain.draw(fr, t, intensity=1.0, angle=8, light=(960, 300), light_r=600)
        fr = cam(fr, 1.0 + 0.08 * e, 0, -20 * e)
        return fr, 0


# =====================================================================================
# V10 97.52–98.48 ECU：天平指针与刻度盘：又一摞纸人砸下（98.37 重拍，画面震），指针仍直立不动；「算够」刻在刻度盘弧上
class V10(Shot):
    T_HIT = 98.373
    def __init__(self):
        x, y = K.grid()
        c = (960, 1020); self.c = c
        d = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2); ang = np.arctan2(y - c[1], x - c[0])
        n = cv2.resize(K.fbm(H // 4, W // 4, 5501, 8, 5), (W, H))
        brushed = cv2.resize(np.random.default_rng(5502).random((H // 2, 4)).astype(np.float32), (W, H))
        face = (d < 760).astype(np.float32)
        metal = (0.16 + 0.10 * n + 0.05 * np.sin(ang * 40) * 0.2 + 0.05 * brushed)[..., None] * np.array([1.0, 0.82, 0.55], np.float32)
        ring = ((d > 700) & (d < 760)).astype(np.float32)
        metal = metal * (1 - ring[..., None]) + ring[..., None] * np.array([0.45, 0.33, 0.16], np.float32) * (0.8 + 0.4 * np.cos(ang * 2 + 1))[..., None]
        bg = BK.dark_bg(5503, 0.01, 0.03)
        fr = bg * (1 - face[..., None]) + metal * face[..., None]
        # 刻度（刻线 + 阴刻）
        for k in range(-40, 41):
            a = math.radians(-90 + k * 2.0)
            r0 = 620 if k % 5 else 560
            p0 = (int(c[0] + math.cos(a) * r0), int(c[1] + math.sin(a) * r0)); p1 = (int(c[0] + math.cos(a) * 690), int(c[1] + math.sin(a) * 690))
            cv2.line(fr, p0, p1, (0.03, 0.025, 0.02), 4 if k % 5 == 0 else 2, cv2.LINE_AA)
            cv2.line(fr, (p0[0] + 2, p0[1] + 2), (p1[0] + 2, p1[1] + 2), (0.5, 0.40, 0.22), 1, cv2.LINE_AA)
        self.face = fr.astype(np.float32)
        self.d = d
        self.drops = np.random.default_rng(5504).uniform(0, 1, (60, 3))
        T.warm('算够', 'serif_black', 120)

    def render(self, t):
        u = self.u(t)
        fr = self.face.copy()
        x, y = K.grid()
        c = self.c
        # 玻璃罩上的雨珠（高光点 + 暗折射）
        for dx_, dy_, r_ in self.drops:
            px, py = int(dx_ * W), int(120 + dy_ * 840); rr = int(4 + r_ * 12)
            cv2.circle(fr, (px, py), rr, (0.02, 0.02, 0.025), -1, cv2.LINE_AA)
            cv2.circle(fr, (px - rr // 3, py - rr // 3), max(1, rr // 3), (0.85, 0.85, 0.9), -1, cv2.LINE_AA)
        # 「算够」刻在弧上（阴刻：暗字 + 下缘高光），逐字烫金
        pr = prs(22, [5, 6], t, 0.22)
        L = T.Layer()
        for i, (ch, pc) in enumerate(zip('算够', pr)):
            if pc <= 0: continue
            a = math.radians(-90 + (-14 if i == 0 else 14))
            T.place(L, ch, c[0] + math.cos(a) * 470, c[1] + math.sin(a) * 470, 130, color='white', fname='serif_black',
                    rot=-(math.degrees(a) + 90), alpha=min(1, pc * 2))
        A = L.A
        fr = fr * (1 - A[..., None] * 0.8) + A[..., None] * np.array([0.02, 0.015, 0.01], np.float32)
        fr += np.clip(np.roll(A, -3, 0) - A, 0, 1)[..., None] * np.array([0.6, 0.45, 0.25], np.float32)
        fr += A[..., None] * GOLD * 0.5 * sum(pulse(t, V(CT(22, k)), 0.25) * (t >= V(CT(22, k))) for k in (5, 6))
        # 指针：纹丝不动（冲击时只抖 0.3°，然后回正）
        hk = hitp(t, self.T_HIT, 0.15)
        a = math.radians(-90 + 0.6 * hk * math.sin(t * 60))
        tip = (c[0] + math.cos(a) * 680, c[1] + math.sin(a) * 680)
        pm = np.zeros((H, W), np.float32)
        cv2.line(pm, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])), 1.0, 16, cv2.LINE_AA)
        cv2.circle(pm, (int(c[0]), int(c[1]) - 10), 60, 1.0, -1, cv2.LINE_AA)
        sh = np.roll(np.roll(pm, 14, 1), 10, 0)
        fr = fr * (1 - 0.5 * K.blur_ds(sh, 8)[..., None])
        fr = over(fr, pm, np.array([0.55, 0.40, 0.18], np.float32))
        fr += (pm * np.exp(-((x - c[0] + 6) / 6) ** 2))[..., None] * 0.6
        # 冲击：上方一摞纸人砸下的阴影掠过 + 碎纸飞
        if t >= V(self.T_HIT) - 0.12:
            q = clamp01((t - (V(self.T_HIT) - 0.12)) / 0.12)
            shadow = sstep(-200 + 600 * q, 200 + 600 * q, -y + 600) * (1 - hitp(t, self.T_HIT, 0.25) * 0) * 0.5 * (1 - clamp01((t - V(self.T_HIT)) / 0.3))
            fr = fr * (1 - shadow[..., None])
        if t >= V(self.T_HIT):
            fr = K.sparks(fr, t, 5505, (960, 160), n=40, life=0.6, speed=(200, 700), dirc=(0, 1), spread=1.6, grav=600, size=3.0,
                          area=(700, 20), color=(0.85, 0.78, 0.62), amt=0.9, t0=V(self.T_HIT), burst=True)
        dx, dy, dr = shake(t, 26 * hk, 14, 55)
        fr = cam(fr, 1.0 + 0.04 * eio(u) + 0.05 * hk, dx, dy, dr)
        return fr, (4 if hk > 0.5 else 0)


# =====================================================================================
# V11 98.48–99.47 CU：右盘金印特写，99.0「量」作印文重重落下，雨水从印面溅开；「分」宋体小字
class V11(Shot):
    T_LIANG = 99.0
    def __init__(self):
        self.seal = PR.GiantSeal('量', 5601)
        self.bg = BK.dark_bg(5602, 0.012, 0.04)
        self.rain = fx2.RainSheet(5603, density=0.7)
        self.spl = fx2.Splashes(5604)
        T.warm('分', 'serif_black', 140)

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        fr = self.bg.copy()
        # 秤盘（金属浅盘，俯视椭圆）+ 积水
        pan = (((x - 960) / 820) ** 2 + ((y - 840) / 150) ** 2 < 1).astype(np.float32)
        lip = (((x - 960) / 820) ** 2 + ((y - 840) / 150) ** 2 < 1).astype(np.float32) - (((x - 960) / 790) ** 2 + ((y - 836) / 136) ** 2 < 1).astype(np.float32)
        fr = fr * (1 - pan[..., None]) + pan[..., None] * np.array([0.10, 0.08, 0.05], np.float32) * (0.7 + 0.5 * cv2.resize(K.fbm(H // 4, W // 4, 5605, 6, 4), (W, H)))[..., None]
        fr += np.clip(lip, 0, 1)[..., None] * GOLD * 0.35
        fall = clamp01((t - (V(self.T_LIANG) - 0.32)) / 0.32)
        fr, info = self.seal.draw(fr, t, 960, 820, 520, fall=ein(fall, 2) if fall < 1 else 1.0, tilt=8, rim_color=GOLD, rim=1.4)
        hk = hitp(t, self.T_LIANG, 0.14)
        if t >= V(self.T_LIANG):
            fr = K.sparks(fr, t, 5606, (960, 820), n=120, life=0.55, speed=(300, 1100), dirc=(0, -1), spread=1.4, grav=1400, size=1.4,
                          area=(500, 20), color=(0.75, 0.8, 0.9), amt=1.0, t0=V(self.T_LIANG), burst=True)
            fr = fx2.ripples(fr, t, [(960, 840)], [V(self.T_LIANG)], amt=0.9, flat=0.18, speed=700, life=0.8)
        fr = self.spl.draw(fr, t, y_far=700, y_near=990, rate=80, amt=0.6)
        pr = prs(22, [7], t)[0]
        if pr > 0:
            L = T.Layer(); T.place(L, '分', 360, 360, 150, color='white', fname='serif_black', alpha=min(1, pr * 2), scale=1.3 - 0.3 * eout(pr))
            fr = T.compose(fr, L, shadow=0.85)
        fr = self.rain.draw(fr, t, intensity=0.9, angle=6)
        dx, dy, dr = shake(t, 22 * hk, 13, 56)
        fr = cam(fr, 1.06 + 0.05 * eio(u) + 0.04 * hk, dx, dy - 20 * eio(u), dr)
        return fr, (4 if hk > 0.5 else 0)


# =====================================================================================
# V12 99.47–100.58 ECU 下摇：纸人从秤盘边被挤落、慢放坠入黑暗，雨打湿纸人、墨色化开；「被牺牲的人」马善政竖排随纸人下坠、遇雨晕开
class V12(Shot):
    def __init__(self):
        self.fig = PA.PaperFigure(5701, view='front')
        self.bg = BK.dark_bg(5702, 0.006, 0.025)
        self.rain = fx2.RainSheet(5703, density=0.6)
        T.warm('被牺牲的人', 'kai', 110)

    def pos(self, t):
        q = clamp01((t - self.t0) / (self.t1 - self.t0))
        return 820 + 60 * math.sin(q * 2.4), 300 + 420 * q, 220 * q     # x, y, 旋转

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        # 上方：秤盘边缘（金线）随下摇离开画面
        edge_y = 200 - 520 * eio(u)
        fr += np.exp(-((y - edge_y) / 8) ** 2)[..., None] * GOLD * 0.5 * (x < 1300)[..., None]
        fr += np.exp(-((y - edge_y - 30) / 60) ** 2)[..., None] * GOLD * 0.06
        px, py, rot = self.pos(t)
        wet = clamp01(u * 1.3)
        fr, _ = self.fig.draw(fr, px, py + 150, 300, t, tilt=rot, wet=wet, ink=0.5 * wet, light='back', light_color=GOLD, rim=0.8)
        # 墨在纸人周围化开（被雨打湿）
        blot = np.exp(-(((x - px) ** 2 + (y - py) ** 2) / (120 + 260 * wet) ** 2))
        nz = cv2.resize(K.fbm(H // 8, W // 8, 5704, 6, 4), (W, H))
        fr = fr * (1 - (blot * nz * 0.6 * wet)[..., None])
        pr = prs(23, [0, 1, 2, 3, 4], t, 0.22)
        P = [dict(ch=c, x=1280, y=260 + i * 128 + 60 * eio(u), size=112, rot=0, font='kai', idx=i, emph=c in '牺牲') for i, c in enumerate('被牺牲的人')]
        fr = T.ink_bleed_in(fr, '被牺牲的人', pr, places=P, color='white', emph_color=(0.84, 0.10, 0.07), bleed=1.4,
                            style_kw=dict(shadow=0.85, glow=0.1))
        fr = self.rain.draw(fr, t * 0.6 + self.t0 * 0.4, intensity=0.8, angle=4)           # 慢放雨
        fr = cam(fr, 1.04, 0, 0, 2 * u)
        return fr, 0


# =====================================================================================
# V13 100.58–102.14 MS：坠落的纸人剪影落地——匹配剪辑：同一位置同一姿态变成鬼屠，从泥水里起身（rise）；「可曾有机会」马善政右侧
class V13(Shot):
    def __init__(self):
        self.sp = spr('S054/rise')
        sk = night_sky(5801, top=(0.02, 0.022, 0.03), bot=(0.12, 0.125, 0.14), horizon=0.78)
        x, y = K.grid()
        self.gm = sstep(830, 850, y)
        mud = (0.02 + 0.035 * cv2.resize(K.fbm(H // 2, W // 2, 5802, 10, 6), (W, H)))[..., None] * np.ones(3, np.float32)
        self.bg = sk * (1 - self.gm[..., None]) + mud * self.gm[..., None]
        self.fig = PA.PaperFigure(5701, view='front')
        self.rain = fx2.RainSheet(5803, density=1.0)
        self.spl = fx2.Splashes(5804)
        self.fog = fx2.Fog(5805)
        T.warm('可曾有机会', 'kai', 110)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        fr = self.fog.draw(fr, t, density=0.5, y0=500, y1=860, color=(0.32, 0.34, 0.37))
        tl = self.t0 + 0.2
        if t < tl:                                    # 纸人落地（与 V12 同姿态、同屏幕位置）
            q = clamp01((t - self.t0) / 0.2)
            tmp = np.zeros_like(fr)
            tmp, _ = self.fig.draw(tmp, 820, 560 + 280 * q ** 2, 300, t, tilt=220 + 40 * q, light='back', rim=0.0)
            fr = sil_over(fr, tmp, COLDC, rim_w=3, amt=1.0)
        else:
            k = eout(clamp01((t - tl) / 1.1), 2)       # 起身：从半跪抬升
            fr, _ = put(fr, self.sp, t, 640 + 80 * k, (840, 900 + 80 * (1 - k)), rim=dict(col=(0.70, 0.80, 1.0), amt=2.4, dir=(0.5, -1.0)),
                        wind_amp=6, shade=(500, 1500, 0.25, 600, 0.3))
            if t < tl + 0.2:
                fr = K.sparks(fr, t, 5806, (840, 880), n=80, life=0.5, speed=(200, 600), dirc=(0, -1), spread=1.3, grav=1300,
                              area=(260, 10), color=(0.7, 0.75, 0.85), amt=1.0, t0=tl, burst=True)
        fr = self.spl.draw(fr, t, y_far=850, y_near=1000, rate=120, amt=0.7)
        fr = self.rain.draw(fr, t, intensity=1.1, angle=10)
        pr = prs(23, [5, 6, 7, 8, 9], t)
        P = [dict(ch=c, x=1480 + (i % 2) * 40, y=250 + i * 120, size=104, rot=0, font='kai', idx=i, emph=False) for i, c in enumerate('可曾有机会')]
        fr = T.ink_bleed_in(fr, '可曾有机会', pr, places=P, color='white', style_kw=dict(shadow=0.85, glow=0.12))
        if tl <= t < tl + 1 / 30 + 1e-3: fr = fr * 0.6 + 0.5
        e = eio(u)
        fr = cam(fr, 1.0 + 0.18 * e, 0, -60 * e)
        return fr, (3 if abs(t - tl) < 0.04 else 0)


# =====================================================================================
# V14 102.14–103.14 ECU：「反抗」二字：宋体被火从下往上烧成志莽行书（字体即立场），朱红火光，雨中蒸汽
class V14(Shot):
    def __init__(self):
        self.bg = BK.dark_bg(5901, 0.01, 0.03, tint=(1.15, 1.0, 0.95))
        self.emb = fx2.Embers(5902, n=220, box=(300, 300, 1620, 960), mode='fire', rise=260)
        self.fog = fx2.Fog(5903)
        self.rain = fx2.RainSheet(5904, density=0.6)
        T.warm('反抗', 'song', 400)

    def render(self, t):
        u = self.u(t)
        fr = self.bg.copy()
        x, y = K.grid()
        pr = prs(23, [10, 11], t, 0.25)
        m = [clamp01((t - (V(CT(23, k)) - 0.05)) / 0.45) for k in (10, 11)]
        P = [dict(ch='反', x=700, y=540, size=400, rot=0, font='xing', idx=0, emph=False),
             dict(ch='抗', x=1220, y=540, size=400, rot=0, font='xing', idx=1, emph=False)]
        heat = max(m)
        fr += np.exp(-(((x - 960) / 800) ** 2 + ((y - 760) / 260) ** 2))[..., None] * RED * 0.5 * heat
        if pr[0] > 0:
            fr = T.font_morph(fr, '反抗', [min(1.0, mm) for mm in m], fa='song', fb='xing', places=P, color='white',
                              colorB=(0.88, 0.12, 0.06), burst=True, style_kw=dict(shadow=0.6, glow=0.6, glow_color=(1.0, 0.3, 0.08)))
            # 火线：沿字自下而上（烧过处转为朱红行书）
            for pp, mm in zip(P, m):
                if 0 < mm < 1:
                    fy = pp['y'] + 220 - 440 * mm
                    band = np.exp(-((y - fy) / 18) ** 2) * (np.abs(x - pp['x']) < 230)
                    fr += band[..., None] * np.array([1.0, 0.5, 0.15], np.float32) * 0.9
        fr = self.emb.draw(fr, t, intensity=0.4 + heat, wind=(30, 0))
        fr = self.fog.draw(fr, t, density=0.3 * heat, y0=300, y1=900, color=(0.55, 0.5, 0.48), speed=(10, -60))
        fr = self.rain.draw(fr, t, intensity=0.6, angle=6)
        hk = hitp(t, CT(23, 11), 0.12)
        dx, dy, dr = shake(t, 4 + 14 * hk, 10, 59)
        fr = cam(fr, 1.0 + 0.08 * eio(u) + 0.04 * hk, dx, dy, dr)
        return fr, (3 if hk > 0.5 else 0)


# =====================================================================================
# V15 103.14–104.30 LS：鬼屠在暴雨中行走（walk_look_up），103.99 闪电照亮，他抬头；「不听天命」两列竖排，「天命」被朱红斜线划掉
class V15(Shot):
    T_BOLT = 103.987
    def __init__(self):
        self.sp = spr('S055/walk_look_up')
        sk = night_sky(6001, top=(0.02, 0.022, 0.03), bot=(0.10, 0.105, 0.12), horizon=0.8)
        wl, wm = city_wall_far(6002, 600, 820, tone=0.05, crenel=False)
        wl = K.blur_ds(wl, 4); wm = K.blur_ds(wm, 4)
        self.bg = sk * (1 - wm[..., None]) + wl * wm[..., None]
        x, y = K.grid()
        self.gm = sstep(820, 840, y)
        self.bg = self.bg * (1 - self.gm[..., None]) + self.gm[..., None] * 0.03
        self.bolt = fx2.Lightning(6003, p0=(1500, 120), p1=(1700, 700), branches=6)
        self.rain = fx2.RainSheet(6004, density=1.2)
        self.spl = fx2.Splashes(6005)
        T.warm('不听天命', 'kai', 120)

    def render(self, t):
        u = self.u(t)
        e = eio(u)
        tl = V(self.T_BOLT)
        env = 0.0 if t < tl - 0.02 else math.exp(-max(0, t - tl) / 0.3)
        fr = self.bg * (1 + 2.5 * env)
        if t >= tl - 0.06:
            fr = self.bolt.draw(fr, t, tl, intensity=1.1)
        walk = 1060 - 80 * e
        bob = 6 * abs(math.sin(t * 6.5))
        fr, _ = put(fr, self.sp, t, 640, (walk, 860 - bob), rim=dict(col=(0.75, 0.82, 1.0), amt=2.0 + 4 * env, dir=(0.7, -0.7)),
                    grade=dict(expo=0.62 + 0.4 * env, sat=0.5, tint=(0.9, 0.96, 1.06), lift=0.0, gamma=1.2), wind_amp=8,
                    shade=(500, 1500, 0.2, 600, 0.3))
        # 积水反光
        ref = fr[::-1].copy(); ref = cv2.warpAffine(ref, np.float32([[1, 0, 0], [0, 1, 2 * 860 - H]]), (W, H))
        fr = fr + ref * (self.gm * 0.25)[..., None]
        fr = self.spl.draw(fr, t, y_far=830, y_near=1000, rate=160, amt=0.8)
        fr = self.rain.draw(fr, t, intensity=1.2, angle=14, light=(1500, 150) if env > 0.1 else None, light_r=900, cam=(80 * e, 0))
        pr = prs(24, [0, 1, 2, 3], t)
        P = [dict(ch='不', x=520, y=300, size=120, rot=0, font='kai', idx=0, emph=False), dict(ch='听', x=520, y=430, size=120, rot=0, font='kai', idx=1, emph=False),
             dict(ch='天', x=380, y=380, size=150, rot=0, font='serif_black', idx=2, emph=False), dict(ch='命', x=380, y=540, size=150, rot=0, font='serif_black', idx=3, emph=False)]
        L = T.Layer()
        for pp, pc in zip(P, pr):
            if pc > 0: T.place(L, pp['ch'], pp['x'], pp['y'], pp['size'], color='gold' if pp['font'] == 'serif_black' else 'white',
                               fname=pp['font'], alpha=min(1, pc * 2), scale=1.2 - 0.2 * eout(pc))
        fr = T.compose(fr, L, shadow=0.85, glow=0.15)
        ps = clamp01((t - (V(CT(24, 3)) + 0.12)) / 0.16)
        if ps > 0: fr = T.strike(fr, P, [2, 3], ps, color='red', width=0.16, seed=60)
        fr = cam(fr, 1.0 + 0.05 * e, 30 * e, 0, (0.8 * math.sin(t * 1.7)))
        return fr, (3 if abs(t - tl) < 0.04 else 0)


# =====================================================================================
# V16 104.30–105.88 ELS 仰天：雨从天心直落向镜头（放射雨丝），云眼里金色宋体「上苍」，被暴雨冲刷溶解；「不问」毛草
class V16(Shot):
    def __init__(self):
        x, y = K.grid()
        self.c = (960, 470)
        d = np.sqrt((x - 960) ** 2 + (y - 470) ** 2); a = np.arctan2(y - 470, x - 960)
        n = cv2.resize(K.fbm(H // 4, W // 4, 6101, 5, 6), (W, H))
        swirl = cv2.resize(K.tile(512, 6102, 6, 5), (W, H))
        self.d, self.a, self.n, self.sw = d, a, n, swirl
        r = np.random.default_rng(6103)
        self.streaks = r.uniform(0, 1, (900, 3))
        T.warm('上苍', 'serif_black', 330); T.warm('不问', 'cao', 170)

    def render(self, t):
        u = self.u(t)
        x, y = K.grid()
        d, a = self.d, self.a
        rot = a + 0.25 * t + 600 / (d + 200)
        tex = self.sw[(np.clip(d * 0.4 + t * 40, 0, 1e6).astype(np.int32)) % H, ((rot % (2 * math.pi)) / (2 * math.pi) * W).astype(np.int32) % W]
        cloud = 0.02 + 0.13 * np.clip(tex * 0.7 + self.n * 0.5 - 0.2, 0, 1) ** 1.4
        eye = np.exp(-(d / 260) ** 2)
        fr = lum3(cloud, (0.95, 0.98, 1.08)) * (1 - eye[..., None]) + eye[..., None] * GOLD * 0.35
        # 「上苍」金宋体悬于云眼 → 被雨冲刷溶解（粒子化 + 下流）
        diss = clamp01((t - (V(CT(24, 7)) - 0.0)) / 0.7)
        pr = prs(24, [6, 7], t)
        L = T.Layer()
        for i, ch in enumerate('上苍'):
            if pr[i] > 0: T.place(L, ch, 790 + i * 340, 470, 330, color='gold', fname='serif_black', alpha=min(1, pr[i] * 2))
        if diss > 0:
            nz = cv2.resize(K.fbm(H // 4, W // 4, 6104, 12, 4), (W, H))
            L.A = L.A * np.clip((nz - diss * 1.1 + 0.15) * 6, 0, 1)
            run = cv2.resize(L.A[::4, ::4], (W, H))
            M = np.float32([[1, 0, 0], [0, 1, 160 * diss]])
            L.A = np.maximum(L.A, cv2.warpAffine(run, M, (W, H)) * 0.45 * diss * np.clip(nz * 2 - 0.6, 0, 1))
        fr = T.compose(fr, L, shadow=0.4, glow=0.6, glow_color=(1.0, 0.6, 0.25))
        # 放射雨丝（从天心落向镜头）
        buf = np.zeros((H // 2, W // 2), np.float32)
        for k, (s0, aa, sp) in enumerate(self.streaks):
            ph = (t * (0.6 + sp) + s0) % 1.0
            ang = aa * 2 * math.pi
            r0 = 40 + 900 * ph ** 2; r1 = r0 + 30 + 160 * ph ** 2
            p0 = (int((960 + math.cos(ang) * r0) / 2), int((470 + math.sin(ang) * r0) / 2)); p1 = (int((960 + math.cos(ang) * r1) / 2), int((470 + math.sin(ang) * r1) / 2))
            cv2.line(buf, p0, p1, 0.25 + 0.6 * ph, 1, cv2.LINE_AA)
        fr += cv2.resize(buf, (W, H))[..., None] * COLDC * 0.6
        pb = prs(24, [4, 5], t)
        fr = T.scatter_stack(fr, '不问', pb, places=[dict(ch='不', x=330, y=300, size=170, rot=-8, font='cao', idx=0, emph=False),
                                                       dict(ch='问', x=420, y=520, size=200, rot=6, font='cao', idx=1, emph=True)],
                             color='white', emph_color=(0.84, 0.10, 0.07), t_local=u, style_kw=dict(shadow=0.8))
        fr = cam(fr, 1.0 + 0.08 * eio(u), 0, 0, -8 * eio(u), center=self.c)
        return fr, 0


# ===================================================================================== 段落装配
SHOTS = [V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16]
CUTS = {
    'V02': ('zoom_through', 0.06, 0.14, dict(center=(960, 470))),
    'V03': ('flash_white', 0.0, 0.17, {}),
    'V04': ('gold', 0, 0, {}),
    'V05': ('occlusion', 0.08, 0.16, dict(kind='pillar', direction=1, seed=5)),
    'V06': ('match_cut', 0.0, 0.12, dict(shape='circle', center=(930, 400), radius=220)),
    'V07': ('gold', 0, 0, {}),
    'V08': ('whip_pan_blur', 0.05, 0.10, dict(direction='down')),
    'V09': ('match_cut', 0.0, 0.12, dict(shape='rect', center=(960, 520), radius=160)),
    'V10': ('hard', 0, 0, {}),
    'V11': ('slash_wipe', 0.05, 0.12, dict(angle=-70)),
    'V12': ('whip_pan_blur', 0.05, 0.10, dict(direction='down')),
    'V13': ('none', 0, 0, {}),
    'V14': ('flash_red', 0.03, 0.12, {}),
    'V15': ('ink_bleed', 0.06, 0.20, dict(seed=15)),
    'V16': ('whip_pan_blur', 0.05, 0.10, dict(direction='up')),
}
FG = {'V01': ('rain', 0.4), 'V03': ('rain', 0.3), 'V07': ('ember', 0.3), 'V08': ('ember', 0.25), 'V09': ('rain', 0.3), 'V13': ('rain', 0.35),
      'V14': ('ember', 0.5), 'V15': ('rain', 0.45)}
EXPO = {}


def _bind():
    for _S, _a, _b in zip(SHOTS, CUT_F, CUT_F[1:]):
        _S.t0, _S.t1 = _a / FPS, _b / FPS
    SHOTS[0].t0 = T_IN
_bind()

_SEC = [None]


def section():
    if _SEC[0] is None: _SEC[0] = Section(SHOTS, CUTS, FG, EXPO, expo_all=1.12)
    return _SEC[0]


_C1 = [None]
SEAM_N = 10                                    # 2586–2595：副歌一 S047 原帧 →（雨幕溶解）→ V01；墨退遮罩两边同一个


def frame(f, cache=None):
    if f < F0 + SEAM_N:
        if _C1[0] is None:
            os.environ.setdefault('SPRITE_MODE', '1')
            import chorus1 as C; _C1[0] = (C, C.Chorus1())
        C, X = _C1[0]
        A = X.frame(f / FPS)[0]
        B = section().frame_f(f / FPS)
        p = (f - F0 + 1) / (SEAM_N + 1)
        nz = _seam_noise()
        m = np.clip((eio(p) * 1.5 - nz) * 4, 0, 1)[..., None]
        return K.to8(A * (1 - m) + B * m)
    return section().frame(f)


_SN = []


def _seam_noise():
    if not _SN:
        x, y = K.grid()
        n = cv2.resize(K.fbm(H // 8, W // 8, 4799, 5, 5), (W, H))
        streak = cv2.resize(np.random.default_rng(4798).random((1, W // 6)).astype(np.float32), (W, H))   # 竖向雨痕
        _SN.append((0.55 * n + 0.45 * streak + 0.15 * (y / H)).astype(np.float32))
    return _SN[0]
