"""units/r6b_f_c1.py — r6b F 组：副歌一（units/chorus1.py，帧 1779–2586）去字 + S035 重做。
由 chorus1.py 末尾调用 patch(globals(), txt, shot)；R6B_TXT=0 且 R6B_SHOT=0 时不 import，画面回到第六轮。

R6B_TXT（去字，2026-11-11 用户：没点名保留的大字/中字/歌词字一律去掉，去字后画面不能空）：
  · S028 59.3–60.0  巨「血」去掉 → 横刀 + 血滴慢放撑满（推近 1.12、视点下移压到刀面）
  · S032 64.2–66.0  正/邪方印去掉 → 延长上一镜 S031 太上皇天幕巨像（推近到冕旒半身），竖斩把巨像劈成左右两半（原斩切代码不变）
  · S036 71.0–72.4  律令字墙 → 同一面巨砖墙、砖面无字（CharWall chars_ratio=0），金缝体积光照旧
  · S038 73.7–74.7  巨「血」去掉 → 刀面倒映眼睛整组左移到画面中轴、略推近（不留字的空位）
  · S040 76.5–78.4  「是非对错」四方印去掉 → 律令小字纸（小字保留）被四记刀光斜劈，逐刀裂开错位（卡 是/非/对/错 四个字头）
  · 镜内横排歌词句（血未冷 / 骨未凉 / 天不判 / 我来量，chorus1 lyr_h 层）整层不画
R6B_SHOT（S035 69.63–70.93「那就劈开它」重做，见 S035Split）。"""
import math, os
import numpy as np, cv2


def patch(G, txt=True, shot=True):
    K, T, PR = G['K'], G['T'], G['PR']
    W, H, GOLD, RED = G['W'], G['H'], G['GOLD'], G['RED']
    cam, over, clamp01, eio, eout, pulse, sstep = G['cam'], G['over'], G['clamp01'], G['eio'], G['eout'], G['pulse'], G['sstep']
    V, CT = G['V'], G['CT']
    SHOTS = G['SHOTS']
    idx = {S.__name__: i for i, S in enumerate(SHOTS)}

    def swap(name, cls):
        i = idx[name]; old = SHOTS[i]
        cls.__name__ = name; cls.t0, cls.t1 = old.t0, old.t1
        SHOTS[i] = cls

    if txt:
        G['HLYR'] = False                         # 镜内横排歌词句（血未冷/骨未凉/天不判/我来量）不画

        # ---------------- S028：去巨「血」
        B28 = SHOTS[idx['S028']]
        class S028(B28):
            def render(self, t):
                f0 = T.ink_bleed_glyph
                T.ink_bleed_glyph = lambda *a, **k: np.zeros_like(f0(*a, **k))
                try:
                    fr, ca = super().render(t)
                finally:
                    T.ink_bleed_glyph = f0
                return cam(fr, 1.12, 0, 55), ca
        swap('S028', S028)

        # ---------------- S032：去正/邪方印 → 延长 S031 太上皇巨像，竖斩劈开
        B32 = SHOTS[idx['S032']]; B31 = SHOTS[idx['S031']]
        class S032(B32):
            def __init__(self):
                super().__init__()
                self.emp = B31()
            def seals(self, t):
                u = self.u(t)
                te = B31.t1 - 1 / 30 + (t - self.t0) * 0.6        # 接着上一镜的时间往下走（冕旒余摆、云照流）
                fr, _ = self.emp.render(te)
                return cam(fr, 1.32 + 0.06 * eio(u), 0, -40, center=(960, 470))
        swap('S032', S032)

        # ---------------- S036：字砖 → 无字巨砖
        B36 = SHOTS[idx['S036']]
        class S036(B36):
            def __init__(self):
                cw = PR.CharWall(seed=36, cols=5, rows=7, brick=(520, 220), fname='song', chars_ratio=0.0, text='律令纲常尊卑规矩法度天理昭彰')
                xx = np.linspace(0, 1, 2300, dtype=np.float32)[None, :, None]
                self.wall = np.ascontiguousarray(cw.tex[:1300, :2300]) * (0.55 + 0.6 * xx) * np.array([1.05, 0.98, 0.9], np.float32)
                self.crack = K.crack_path((300, -100), (1900, 1300), 361, n=80, jag=0.025)
                self.cm = K.draw_crack((1300, 2300), self.crack, 7, branches=24, seed=362, prog=1.0)
                self.wall_d = self.wall * (1 - self.cm[..., None])
        swap('S036', S036)

        # ---------------- S038：去巨「血」，刀面倒影组移到中轴
        B38 = SHOTS[idx['S038']]
        class S038(B38):
            def render(self, t):
                g0 = T.giant_layer
                T.giant_layer = lambda *a, **k: T.Layer()
                try:
                    fr, ca = super().render(t)
                finally:
                    T.giant_layer = g0
                return cam(fr, 1.10, 330, 0, center=(1290, 540)), ca
        swap('S038', S038)

        # ---------------- S040：去「是非对错」→ 律令小字纸被四记刀光劈开
        B40 = SHOTS[idx['S040']]
        class S040(B40):
            CUTS = [((820, 430), -58.0), ((1120, 470), 62.0), ((900, 640), -18.0), ((1060, 600), 24.0)]   # (过点, 刀线角度°)
            def render(self, t):
                u = self.u(t)
                fr = self.pap.copy()
                x, y = K.grid()
                hot_all = np.zeros((H, W), np.float32)
                for k, ((px, py), ang) in enumerate(self.CUTS):
                    ts = self.TS[k]
                    q = t - ts
                    if q < -0.04: continue
                    th = math.radians(ang)
                    dx, dy = math.cos(th), math.sin(th)
                    sd = ((x - px) * dy - (y - py) * dx).astype(np.float32)     # 到刀线的有符号距离
                    if q >= 0:
                        e = eout(q / 0.30, 4)
                        off = 16 * e
                        A = cam(fr, 1.0, -dy * off + dx * off * 0.6, dx * off + dy * off * 0.6)
                        B = cam(fr, 1.0, dy * off - dx * off * 0.6, -dx * off - dy * off * 0.6)
                        fr = np.where((sd < 0)[..., None], A, B)
                        g = np.abs(sd)
                        gap = np.clip(1 - (g - 3.5 * e) / 2.5, 0, 1) * (e > 0.02)
                        fr = over(fr, gap, (0.02, 0.015, 0.015))                 # 纸缝里透出的暗底
                        col = np.array([1.0, 0.80, 0.55], np.float32) if k % 2 == 0 else np.array([1.0, 0.30, 0.12], np.float32)
                        hot = np.exp(-g / 4) * math.exp(-q / 0.30) * 1.6 + np.exp(-g / 30) * math.exp(-q / 0.18) * 0.5
                        fr = fr + hot[..., None] * col
                        cut_edge = np.exp(-g / 2.0) * (1 - math.exp(-q / 0.4))  # 冷却后留一道焦边
                        fr = fr * (1 - 0.5 * cut_edge[..., None] * (np.array([0.3, 0.9, 0.9], np.float32) if k % 2 else 1.0))
                        if q < 0.5:
                            fr = T._slash_light(fr, (px, py), np.array([dx, dy]) * (1 if k % 2 == 0 else -1), clamp01(q / 0.22), 1700)
                        fr = K.sparks(fr, t, 4040 + k, (px, py), n=50, life=0.45, speed=(250, 900), dirc=(-dy, dx), spread=1.4, grav=900,
                                      burst=True, t0=ts, area=(10, 260), color=(1.0, 0.7, 0.4), amt=0.9)
                z = 1.0 + 0.12 * eio(u)
                pk = sum(pulse(t, ts, 0.1) * (t >= ts) for ts in self.TS)
                z *= 1 + 0.035 * min(1, pk)
                fr = cam(fr, z, 0, 0)
                return fr, (2 if pk > 0.6 else 0)
        swap('S040', S040)

    if shot:
        import r6b_f_s035 as S5
        swap('S035', S5.make(G, SHOTS[idx['S035']]))
