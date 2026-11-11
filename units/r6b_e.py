"""r6b E 组（帧 5797–7240，3:13–4:01）—— 去字 + 3:26 鬼屠镜修复，以补丁形式挂到 sec_e_final_outro（在 e_r9 / R5_NOVLYR 之后）。

开关（环境变量，默认开）：
  R6B_TXT   去字。=0 回第六轮。
            去：E01 刀身阴刻「血」+ 副句「未冷」；E02「刀 不 放」朱印；E03「世间从来没有」；E04 副句「什么」（匾保留）；E07「谎言」；
                E09「那就撕碎」（含整句被斩开的金线）；E10「它 / 自…」；E12 副句「自己定规章」（鬼屠印保留）；E13 滴血「血」+「未冷」；
                E14T「骨未凉」；E19「何须伪装」+「？」；E20 熔化巨字「天」+「不」；E21「我」；E23/E24/E25/E26 屏幕层（含「正道沧桑」）。
            留：E04/E05/E27c 天理昭彰匾、E06 挂轴小字、E12 鬼屠印、E15 天理木牌、E16/E17/E18 ✓ ✗、片名、署名、E31 烧纸残字、E30/E35 鬼屠小朱印。
  R6B_SHOT  E12（3:26.1–3:27.3）鬼屠俯拍重做：人物换绿幕抠图整身（原 crop=0.56 把左臂/左靴截断、灰底抠图发丝描白边），
            印心那把代码画的刀（扁平描边镡 + 斜压过「屠」字的刀影）去掉，改用同一张生图里的真刀（抠出刀 + 红穗）插在人物右侧地面，
            刀不再压字；R6B_E/top_down_cut.png、R6B_E/sword_td_cut.png 由 tools/r6b_e_mkguitu_*.py 从 E_r2/top_down_gcut 生成。
"""
import os, math
import numpy as np, cv2
import c1kit as K
import rest_sprites as RS
import e_kit as E
from e_kit import W, H, GOLDW, eout

TXT = os.environ.get('R6B_TXT', '1') == '1'
SHOT = os.environ.get('R6B_SHOT', '1') == '1'
M = None


def _nil_text(self, fr, t): return fr


class _Nil:
    def draw(self, fr, *a, **k): return fr


def _wrap_init(cls, post):
    f0 = cls.__init__

    def init(self, *a, **k):
        f0(self, *a, **k); post(self)
    cls.__init__ = init


# ------------------------------------------------------------------ E12 鬼屠俯拍
BODY = 'R6B_E/top_down'
SWORD = 'R6B_E/sword_td'
_SW = {}


def _sword_anc(h):
    """刀身入土点（精灵最低处刃中心）在精灵包围盒里的归一化位置。"""
    if 'anc' not in _SW:
        sp = RS.layer(SWORD, 400)
        A = sp.A
        ys, xs = np.nonzero(A > 0.5)
        yb = ys.max(); sel = ys > yb - 12
        bx, by = xs[sel].mean() / sp.w, (yb - 4) / sp.h
        x0, y0, x1, y1 = sp.bbox
        _SW['anc'] = ((bx - x0) / (x1 - x0), (by - y0) / (y1 - y0))
    return _SW['anc']


def e12_render(self, t):
    u = self.u(t)
    fr = self.ground.copy()
    yy, xx = E.grid()
    fr = fr * (0.55 + 0.6 * np.exp(-((xx - 1300) ** 2 + (yy - 200) ** 2) / (2 * 700 ** 2)))[..., None]   # 晨光斜照
    p = (t - (M.CT(51, 9) - 0.18)) / 0.7
    c = (820, 500)
    if p > 0: fr = M.PR.stamp_imprint(fr, self.face, c[0], c[1], 640, min(1, p), angle=-4)
    fr = np.ascontiguousarray(fr, np.float32)
    gr = dict(expo=0.95, sat=0.6, tint=(1.1, 0.98, 0.84), lift=0.0, gamma=1.05)
    # 人物（整身，绿幕抠图）
    at = (1360, 600)
    z, info = M.put(np.zeros((H, W, 3), np.float32), t, BODY, at, h=495, anc=(0.5, 0.5), rim=GOLDW, rim_amt=2.4, rim_dir=(0.7, -0.7),
                    grade=gr, wind=4)
    A = info['A']
    # 刀：同一张生图里的真刀，插在人物右侧地面（刀身入土点 = bx,by），红穗向右飘
    bx, by = 1600, 790
    zs, si = M.put(np.zeros((H, W, 3), np.float32), t, SWORD, (bx, by), h=252, anc=_sword_anc(252), rim=GOLDW, rim_amt=2.0, rim_dir=(0.7, -0.7),
                   grade=gr, wind=3)
    As = si['A']
    z = np.clip(z + 0.9 * (z - cv2.GaussianBlur(z, (0, 0), 1.6)), 0, 4)          # 俯拍人物 USM 锐化（同第六轮）
    zs = np.clip(zs + 0.6 * (zs - cv2.GaussianBlur(zs, (0, 0), 1.4)), 0, 4)
    Aall = A
    Sh = np.zeros_like(Aall)                                          # 晨光自右上 → 长影拖向左下
    for k in range(1, 9):
        Mt = np.float32([[1, 0, -34 * k], [0, 1, 26 * k]]); Sh = np.maximum(Sh, cv2.warpAffine(Aall, Mt, (W, H)) * (1 - 0.07 * k))
    kk = 0.9                                                          # 刀影：从入土点投向左下（竖直物体在地面上的投影）
    Ms = np.float32([[1, kk, -kk * by], [0, -0.75 * kk, by * (1 + 0.75 * kk)]])
    Sh = np.maximum(Sh, 0.8 * cv2.warpAffine(As, Ms, (W, H)))
    Sh = cv2.GaussianBlur(Sh, (0, 0), 6)
    con = cv2.GaussianBlur(cv2.dilate(A, np.ones((9, 9), np.uint8)), (0, 0), 14)        # 接触阴影
    con2 = cv2.GaussianBlur(cv2.dilate(A, np.ones((5, 5), np.uint8)), (0, 0), 4)
    fr = fr * (1 - 0.6 * Sh[..., None]) * (1 - 0.5 * con[..., None]) * (1 - 0.35 * con2[..., None])
    # 刀入土处：刀身脚下一小团接触暗影 + 几粒翻起的碎土（不画裂纹线）
    dd = np.hypot((xx - bx) / 1.5, yy - by - 2)
    fr = fr * (1 - 0.55 * np.exp(-(dd / 11) ** 2)[..., None])
    rng = np.random.default_rng(1207)
    for k in range(10):
        a0 = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(9, 26)
        q0 = (int(bx + r0 * math.cos(a0)), int(by + 0.65 * r0 * math.sin(a0)))
        rr = int(rng.uniform(1, 3))
        cv2.circle(fr, (q0[0] - 1, q0[1] + 1), rr + 1, (0.03, 0.025, 0.02), -1, cv2.LINE_AA)
        cv2.circle(fr, q0, rr, (0.16, 0.12, 0.085), -1, cv2.LINE_AA)
    fr = fr * (1 - As[..., None]) + zs * As[..., None]
    fr = fr * (1 - A[..., None]) + z * A[..., None]
    ring_ = np.clip(A - cv2.GaussianBlur(cv2.erode(A, np.ones((5, 5), np.uint8)), (0, 0), 4), 0, 1)       # 右上轮廓光
    gy, gx = np.gradient(cv2.GaussianBlur(A, (0, 0), 4)); d2 = np.clip((-gx * 0.7 + gy * 0.7) / (np.hypot(gx, gy) + 1e-6), 0, 1)
    fr = fr + (ring_ * d2)[..., None] * np.array([1.0, 0.8, 0.5], np.float32) * 1.1
    if p > 0.3:
        q = (p - 0.3) / 0.7
        r = 330 + 600 * eout(q)
        ring = np.exp(-((np.hypot(xx - c[0], yy - c[1]) - r) / 40) ** 2) * (1 - q)
        fr = fr + ring[..., None] * np.array([0.8, 0.5, 0.3], np.float32) * 0.5
        fr = E.dust_motes(fr, t, 1213, n=60, amt=0.5 * (1 - q), drift=(40, 30), col=(0.9, 0.7, 0.5), size=(2, 5))
    return fr


def _cls_of(nm):
    for s in M.SHOTS:
        if s[0].__name__ == nm: return s[0]
    return getattr(M, nm)


def apply(mod):
    global M
    M = mod
    if getattr(M, '_r6b_applied', False): return
    M._r6b_applied = True
    if TXT:
        for nm in ('E01', 'E02', 'E03', 'E04', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14T', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26'):
            _cls_of(nm).text = _nil_text
        _wrap_init(_cls_of('E01'), lambda s: setattr(s.B, 'engrave', lambda col, *a, **k: col))   # 刀身阴刻「血」
        _wrap_init(_cls_of('E20'), lambda s: setattr(s, 'tt', 1e9))                               # 熔化巨字「天」（日出照常）
        e19 = _cls_of('E19')
        if hasattr(e19, '_text'):                                                                  # r5 版 E19：「？」+「何须伪装」画在 render 里
            e19._text = lambda self: setattr(self, 'lyr', _Nil())
        else:
            e19.text = _nil_text
    if SHOT:
        _cls_of('E12').render = e12_render
