"""units/r6b_f_s035.py — r6b F 组 S035 重做（R6B_SHOT=1，默认开）：69.633–70.933（帧 2089–2127），歌词「那就劈开它」。
用户 2026-11-11：「1m10s 角色跳斩不好看，换其他呈现方式重做分镜」。
新分镜：不拍跳起。低机位正面仰对一整面律令砖墙（铁壁铜墙），主角背影立在墙前画面中轴（逆光剪影）：
  69.63–69.96 「那就」  墙前暗场，主角双手举刀过顶（生图 R6B_F/s35_raise_c，背面），刀尖一点冷光；镜头极慢推，墙缝落灰
  69.967       一帧竖向刀光自上而下贯穿墙面中轴（刀光一记）
  70.00 「劈」 命中：墙面沿刀线自上而下炸裂（4 帧传到底），金白光从裂口喷出，碎砖向镜头迸飞，暖白闪 + 震屏；
               主角换收刀姿势（R6B_F/s35_follow_a，背面低弓步、刀垂右下），成为裂口光里的剪影（金色轮廓光）
  70.23 「开」 两半墙再被推开一截（第二记顿挫）；70.53「它」第三记，裂口宽到 ≈330px，光柱 + 体积光，墙后是更亮的天光
切点不变（S034→S035 右甩、S035→S036 硬切），镜内不出现任何字（墙砖小字为底纹小字，保留）。"""
import math, os
import numpy as np, cv2


def make(G, B_old):
    K, T = G['K'], G['T']
    W, H, GOLD, RED = G['W'], G['H'], G['GOLD'], G['RED']
    cam, over, clamp01, eio, eout, pulse, sstep, shake = G['cam'], G['over'], G['clamp01'], G['eio'], G['eout'], G['pulse'], G['sstep'], G['shake']
    import sprite as SP
    GEN = G['ROOT'] + '/assets/gen/'
    B35 = G['S035']
    M3 = G['V3'].Move3
    G['MOVES3']['S035'] = M3('sine_io', 'beat', z=(1.0, 1.06), y=(20, -10), note='r6b：正面仰拍慢推（劈开墙），命中拍加速')
    G['MOVES']['S035'] = G['VC'].Move('sine_io', z=(1.0, 1.03))
    WHITE = np.array([1.0, 0.95, 0.86], np.float32)

    class S035Split(B35):
        T_STREAK = 69.96            # 帧 2099（69.9667）
        T_HIT = 70.0
        T_KAI = 70.233
        T_TA = 70.533
        HX, HY = 960.0, 1010.0               # 主角脚底（画面中轴，脚在下黑边里）
        HH = 640.0                            # 主角包围盒高（收刀姿势按身高比例换算）

        def __init__(self):
            super().__init__()                                    # 复用 S035 的律令砖墙纹理（暖色底光）
            src = np.ascontiguousarray(self.wall[60:1380, 40:2360])
            h, w = src.shape[:2]
            S = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            D = np.float32([[150, -330], [1770, -330], [2240, 1230], [-320, 1230]])   # 低机位仰拍：墙顶收窄
            self.wallP = cv2.warpPerspective(src, cv2.getPerspectiveTransform(S, D), (W, H), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            x, y = K.grid()
            lit = (0.30 + 0.85 * sstep(150, 1060, y) ** 1.4) * (0.75 + 0.35 * np.exp(-((x - 960) / 700) ** 2))
            self.wallP = self.wallP * lit[..., None] * np.array([1.0, 0.92, 0.84], np.float32)
            self.path = K.crack_path((972, -120), (946, 1200), 3561, n=80, jag=0.03)
            py, px = self.path[:, 1], self.path[:, 0]
            o = np.argsort(py)
            self.xc = np.interp(np.arange(H, dtype=np.float32), py[o], px[o]).astype(np.float32)   # 每行裂缝 x
            self.sd = (x - self.xc[:, None]).astype(np.float32)                                   # 到裂缝的水平有符号距离
            self.x, self.y = x, y
            self.sp_raise = SP.layer(GEN + 'R6B_F/s35_raise_c_cut.png')
            self.sp_follow = SP.layer(GEN + 'R6B_F/s35_follow_a_cut.png')
            # 墙后天光（裂口里看见的）：中心过曝金白 → 外圈暖金
            self.beyond = (WHITE * 1.6 * np.exp(-((y - 420) / 520) ** 2)[..., None] + GOLD * 0.9).astype(np.float32)

        # 两半墙的分开量（像素，总宽）：劈 / 开 / 它 三记顿挫 + 慢漂
        def gap(self, t):
            if t < self.T_HIT: return 0.0
            return (70 * eout((t - self.T_HIT) / 0.10, 3) + 90 * eout((t - self.T_KAI) / 0.14, 3) * (t >= self.T_KAI)
                    + 110 * eout((t - self.T_TA) / 0.20, 3) * (t >= self.T_TA) + 60 * (t - self.T_HIT))

        def hero(self, fr, t, lightk):
            if t < self.T_STREAK:
                sp, hh, rim = self.sp_raise, self.HH * 1.18, dict(col=(1.0, 0.72, 0.42), amt=1.6, dir=(0.0, 1.0))
                gr = dict(expo=0.55, sat=0.55, tint=(1.0, 0.94, 0.88), lift=0.0, gamma=1.2)
                rise = eout(clamp01((t - self.t0) / 0.30), 2)                # 举刀：0.3s 内刀再上提一点
                foot = (self.HX - 10, self.HY + 18 * (1 - rise))
                mb = (0, -6 * (1 - rise))
            else:
                sp, hh = self.sp_follow, self.HH * 0.93
                k = clamp01(lightk)
                rim = dict(col=(1.0, 0.86, 0.55), amt=2.5 + 3.5 * k, dir=(0.0, -1.0))
                gr = dict(expo=0.55 - 0.30 * k, sat=0.5, tint=(1.0, 0.93, 0.85), lift=0.0, gamma=1.25)
                a = t - self.T_STREAK
                foot = (self.HX + 15, self.HY + 22 * (1 - eout(clamp01(a / 0.12), 2)))   # 斩落：身子沉下去落定
                mb = (0, 10 * (1 - clamp01(a / 0.07)))
            anc, s = sp.fit(hh)
            fr, Am = sp.draw(fr, t, anc, foot, s, rim=rim, grade=gr, wrap=0.85, mblur=mb, shade=None, wind_amp=7.0)
            return fr, Am

        def render(self, t):
            x, y = self.x, self.y
            g = self.gap(t)
            # ---------------- 墙（两半错开）
            if g > 0.5:
                L = cam(self.wallP, 1.0, g / 2, 0)                  # 左半向左（视点右移 = 内容左移）
                R = cam(self.wallP, 1.0, -g / 2, 0)
                left = self.sd < -g / 2; right = self.sd > g / 2
                fr = np.where(left[..., None], L, np.where(right[..., None], R, 0.0)).astype(np.float32)
                hole = (~left & ~right).astype(np.float32)
                hole = cv2.GaussianBlur(hole, (0, 0), 1.2)
                q = t - self.T_HIT
                fr = fr * (1 - hole[..., None]) + self.beyond * hole[..., None] * (1.0 + 0.6 * math.exp(-q / 0.3))
                # 断面：裂口两侧墙沿被光吃亮（发热的金边）
                d = np.abs(self.sd) - g / 2
                edge = np.exp(-np.clip(d, 0, None) / 10) * (d > -1)
                fr = fr + edge[..., None] * GOLD * (0.9 + 0.8 * math.exp(-q / 0.25))
                # 光溢到墙面（越近裂口越亮，带底光）
                spill = np.exp(-np.clip(d, 0, None) / (160 + 2.2 * g)) * (d > 0)
                fr = fr + spill[..., None] * GOLD * 0.55 * fr.mean(-1, keepdims=True) * 4.0
            else:
                fr = self.wallP.copy()
            # ---------------- 裂纹：命中前一帧起自上而下传到底（4 帧），分叉向两侧
            if t >= self.T_STREAK:
                pc = eout((t - self.T_STREAK) / 0.16, 2)
                cm = K.draw_crack((H, W), self.path, 5, branches=28, seed=3562, prog=pc)
                if g > 0.5:
                    cmL = cam(cm, 1.0, g / 2, 0) * (self.sd < 0); cmR = cam(cm, 1.0, -g / 2, 0) * (self.sd > 0)
                    cm = np.maximum(cmL, cmR)
                hot = 0.5 + 1.0 * math.exp(-max(0.0, t - self.T_HIT) / 0.35)
                fr = over(fr, cm, (0.0, 0.0, 0.0))
                fr = fr + (K.blur_ds(cm, 5) * 1.8 + cm * 0.8)[..., None] * GOLD * hot
            else:                                                    # 「那就」：墙缝落灰 + 极暗
                fr = K.dust(fr, t, 3563, n=50, amt=0.30, size=(1, 3), drift=(0, 60), color=(0.9, 0.7, 0.5), blur=1.2)
            # ---------------- 刀光一记（69.967 一帧 + 命中后 3 帧余辉）：沿裂缝的竖向白刃光
            ts = t - self.T_STREAK
            if 0 <= ts < 0.15:
                k = 1.0 if ts < 1 / 30 + 1e-3 else math.exp(-(ts - 1 / 30) / 0.04)
                head = clamp01(ts / (1 / 30) * 1.0 + 0.35)           # 第一帧已划到画面 ⅔，第二帧到底
                yl = -120 + 1320 * head
                dd = np.abs(self.sd)
                band = (np.exp(-(dd / 2.5) ** 2) * 2.2 + np.exp(-(dd / 18) ** 2) * 0.7) * (y < yl)
                fr = fr + band[..., None] * WHITE * k + K.blur_ds(band, 40)[..., None] * RED * 0.5 * k
            # ---------------- 体积光（命中后）：从裂口向镜头放射
            lightk = 0.0
            if t >= self.T_HIT:
                q = t - self.T_HIT
                lightk = 0.6 + 0.4 * eout(q / 0.08, 2)
                core = np.exp(-(np.abs(self.sd) / (12 + g * 0.6)) ** 2) * sstep(-100, 300, y)
                src = np.repeat(np.clip(core, 0, 1)[..., None], 3, -1) * GOLD
                rays = K.radial_blur(src, (962, 470), 0.65, n=8, scale=4)
                fr = fr + rays * (0.9 + 0.6 * math.exp(-q / 0.3)) * lightk
            # ---------------- 主角（剪影）
            fr, Am = self.hero(fr, t, lightk)
            if t < self.T_STREAK and Am is not None:               # 举刀：刀尖一点冷光（预兆）
                ys_, xs_ = np.nonzero(Am > 0.5)
                if len(ys_):
                    i = int(np.argmin(ys_)); tx_, ty_ = float(xs_[i]), float(ys_[i])
                    tw = np.exp(-(((x - tx_) ** 2 + (y - ty_) ** 2) / (2 * 9.0 ** 2)))
                    st = np.exp(-((y - ty_) / 3.0) ** 2) * np.exp(-((x - tx_) / 70) ** 2) + np.exp(-((x - tx_) / 3.0) ** 2) * np.exp(-((y - ty_) / 50) ** 2)
                    kk = 0.4 + 0.8 * sstep(69.80, 69.95, t)
                    fr = fr + (tw * 1.2 + st * 0.6)[..., None] * np.array([0.85, 0.9, 1.0], np.float32) * kk
            # ---------------- 碎砖向镜头迸飞 + 火星（命中后）
            if t >= self.T_HIT:
                q = t - self.T_HIT
                r = np.random.default_rng(3564)
                Lb = np.zeros((H // 2, W // 2), np.float32); Le = np.zeros_like(Lb)
                fade = 1 - sstep(0.45, 0.80, q)
                for i in range(34):
                    oy = r.uniform(120, 900); ox = self.xc[int(np.clip(oy, 0, H - 1))] + r.normal(0, 25)
                    side = -1 if r.random() < 0.5 else 1
                    vx, vy = side * r.uniform(150, 900), r.normal(-150, 260)
                    grow = r.uniform(1.5, 4.5)                        # 冲向镜头：越飞越大
                    s0 = r.uniform(10, 34); rot0 = r.uniform(0, 180)
                    if fade <= 0.01: break
                    px = ox + vx * q; py = oy + vy * q + 900 * q * q
                    s = s0 * (1 + grow * q)
                    if py > 1000 or px < -100 or px > W + 100: continue
                    box = cv2.boxPoints(((px / 2, py / 2), (s / 2, s * 0.55 / 2), rot0 + 400 * q * side))
                    cv2.fillConvexPoly(Lb, box.astype(np.int32), fade, cv2.LINE_AA)
                    cv2.polylines(Le, [box.astype(np.int32)], True, fade, 1, cv2.LINE_AA)
                Lb = cv2.resize(cv2.GaussianBlur(Lb, (0, 0), 0.8), (W, H)); Le = cv2.resize(Le, (W, H))
                fr = over(fr, np.clip(Lb, 0, 1), (0.035, 0.03, 0.03))
                fr = fr + np.clip(Le - Lb * 0.5, 0, 1)[..., None] * GOLD * 0.8          # 碎块受裂口光的金边
                fr = K.sparks(fr, t, 3565, (962, 520), n=120, life=0.6, speed=(400, 1500), dirc=(1, 0), spread=1.6, grav=900,
                              burst=True, t0=self.T_HIT, area=(30, 360), color=(1.0, 0.75, 0.4), amt=1.2)
                fr = K.sparks(fr, t, 3566, (962, 520), n=60, life=0.5, speed=(300, 1100), dirc=(-1, 0), spread=1.2, grav=900,
                              burst=True, t0=self.T_KAI, area=(30, 300), color=(1.0, 0.7, 0.35), amt=0.9) if t >= self.T_KAI else fr
                fr = K.dust(fr, t, 3567, n=70, amt=0.45 * fade + 0.15, size=(1, 4), drift=(0, -40), color=(1.0, 0.8, 0.55), blur=1.5)
            # ---------------- 命中暖白闪（1 帧）+ 三记顿挫震屏
            if abs(t - self.T_HIT) < 0.5 / 30:
                fl = np.exp(-((x - 962) ** 2 + (y - 480) ** 2) / (2 * 380.0 ** 2)).astype(np.float32)
                fr = fr + (fl * 0.38)[..., None] * WHITE
            sk = max(pulse(t, self.T_HIT, 0.12) * (t >= self.T_HIT), 0.6 * pulse(t, self.T_KAI, 0.10) * (t >= self.T_KAI),
                     0.45 * pulse(t, self.T_TA, 0.10) * (t >= self.T_TA))
            dx, dy, dr = shake(t, 14 * sk, 16, 351)
            fr = cam(fr, 1.02 + 0.03 * sk, dx, dy, dr)
            return fr, (3 if sk > 0.5 else 0)

    return S035Split
