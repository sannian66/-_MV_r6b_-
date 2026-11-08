"""sk.py · 美术·角色组 —— Skia 矢量绘制层

所有角色/道具的“形状”都经由这里用 skia 矢量路径光栅化（解析式抗锯齿、亚像素坐标、贝塞尔曲线、渐变、MaskFilter 模糊），
再以 float32 / uint8 numpy 数组的形式交还给原有的 numpy 合成管线（输出仍为 float32 RGB 0..1）。

核心思路：skia 只负责“覆盖率”(coverage)。每个图元先画进一张与画布同尺寸、包着 numpy 内存的 A8 覆盖率面，
然后在 numpy 里按覆盖率对各通道做 lerp（与旧版 cv2.fillPoly(LINE_AA) 的语义完全一致），因此：
  · 旧代码直接对 C/G/M/D 数组做的 cv2 写入仍然可以与 skia 绘制交错使用（同一块内存）；
  · 一个图元由多个子形状组成（胶囊=矩形+两圆）时，先在覆盖率面里并集，再一次性上色 → 没有接缝/双重 AA 暗边；
  · 条带（ribbon/发丝）按段画时相邻段做 0.6px 重叠，消除 cv2 版本可见的段间细缝。

环境：pip install skia-python ；系统库 sudo apt-get install -y libegl1 libgl1
"""
import math
import numpy as np, cv2

try:
    import skia
    HAVE_SKIA = True
except Exception:                       # 没装 skia 时整个库自动回退到 cv2 光栅化
    skia = None
    HAVE_SKIA = False


# ============================================================== 路径构造
def _pts(p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    return [skia.Point(float(x), float(y)) for x, y in p]

def path_poly(pts, closed=True, smooth=False):
    """多边形路径；smooth=True 用中点二次贝塞尔把折线磨圆（布料、发丝、袖口）。"""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if not smooth or len(p) < 3:
        return skia.Path.Polygon(_pts(p), closed)
    P = skia.Path()
    if closed:
        m = (p + np.roll(p, -1, 0)) / 2
        P.moveTo(*m[-1])
        for i in range(len(p)): P.quadTo(p[i][0], p[i][1], m[i][0], m[i][1])
        P.close()
    else:
        P.moveTo(*p[0])
        for i in range(1, len(p) - 1):
            m = (p[i] + p[i + 1]) / 2; P.quadTo(p[i][0], p[i][1], m[0], m[1])
        P.lineTo(*p[-1])
    return P

def path_circle(c, r):
    P = skia.Path(); P.addCircle(float(c[0]), float(c[1]), float(max(r, 0.15))); return P

def path_ellipse(c, ax, ang_deg=0.0, a0=None, a1=None):
    """椭圆（cv2.ellipse 约定：ax=(半轴x,半轴y)，ang 度，顺时针为正的屏幕坐标）。a0/a1 给定时为扇形弧带（闭合）。"""
    cx, cy = float(c[0]), float(c[1]); rx, ry = float(max(ax[0], 0.15)), float(max(ax[1], 0.15))
    P = skia.Path()
    if a0 is None:
        P.addOval(skia.Rect.MakeLTRB(cx - rx, cy - ry, cx + rx, cy + ry))
    else:
        P.addArc(skia.Rect.MakeLTRB(cx - rx, cy - ry, cx + rx, cy + ry), float(a0), float(a1 - a0)); P.close()
    if ang_deg:
        P.transform(skia.Matrix.RotateDeg(float(ang_deg), skia.Point(cx, cy)))
    return P

def path_capsule_union(a, b, r1, r2):
    """稳妥版胶囊：四边形 + 两圆的并集（Path.op）。"""
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; L = float(np.hypot(*d))
    P = path_circle(a, r1)
    if L > 1e-6:
        n = np.array([-d[1], d[0]]) / L
        q = np.array([a + n * r1, b + n * r2, b - n * r2, a - n * r1])
        ar = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
        if ar < 0: q = q[::-1]                      # 与 addCircle(kCW) 同向，避免非零环绕相消
        P.addPoly(_pts(q), True)
        P.addCircle(float(b[0]), float(b[1]), float(max(r2, 0.15)))
    P.setFillType(skia.PathFillType.kWinding)     # 非零环绕 → 自动并集（同向绘制）
    return P

def path_stroke(pts, width, cap='round', closed=False):
    """把折线描边转换成填充路径（用于需要覆盖率的细线）。"""
    P = path_poly(pts, closed=closed)
    paint = skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=float(width),
                       StrokeCap=skia.Paint.kRound_Cap if cap == 'round' else skia.Paint.kButt_Cap,
                       StrokeJoin=skia.Paint.kRound_Join)
    out = skia.Path(); paint.getFillPath(P, out)
    return out


# ============================================================== 覆盖率光栅器
class Coverage:
    """包着 numpy uint8 (h,w) 的 skia A8 面。render(path or [paths]) → (m float32 0..1, (y0,y1,x0,x1))。"""
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.buf = np.zeros((h, w), np.uint8)
        self.surf = skia.Surface(self.buf, colorType=skia.kAlpha_8_ColorType, alphaType=skia.kPremul_AlphaType)
        self.cv = self.surf.getCanvas()
        self.paint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK)
        self.spaint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK, Style=skia.Paint.kStroke_Style,
                                 StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)

    def render(self, paths, stroke=None, blur=0.0, alpha=1.0):
        if not isinstance(paths, (list, tuple)): paths = [paths]
        r = None
        for P in paths:
            b = P.computeTightBounds()
            r = b if r is None else skia.Rect.MakeLTRB(min(r.left(), b.left()), min(r.top(), b.top()), max(r.right(), b.right()), max(r.bottom(), b.bottom()))
        if r is None: return None, None
        pad = 2 + (stroke or 0) + blur * 3
        x0 = max(0, int(math.floor(r.left() - pad))); x1 = min(self.w, int(math.ceil(r.right() + pad)))
        y0 = max(0, int(math.floor(r.top() - pad))); y1 = min(self.h, int(math.ceil(r.bottom() + pad)))
        if x1 <= x0 or y1 <= y0: return None, None
        self.buf[y0:y1, x0:x1] = 0
        if stroke:
            p = self.spaint; p.setStrokeWidth(float(stroke))
        else:
            p = self.paint
        if blur > 0 or alpha != 1.0:
            p = skia.Paint(p)
            if blur > 0: p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, float(blur)))
            p.setAlphaf(float(alpha))
        for P in paths: self.cv.drawPath(P, p)
        return self.buf[y0:y1, x0:x1].astype(np.float32) * (1 / 255.), (y0, y1, x0, x1)


def lerp_into(arr, m, bb, val):
    """arr[bb] = arr*(1-m) + val*m（uint8 或 float32；arr 可为 2D 或 3D，val 标量或通道向量）。"""
    y0, y1, x0, x1 = bb
    reg = arr[y0:y1, x0:x1]
    if reg.ndim == 3:
        mm = m[..., None]; v = np.asarray(val, np.float32).reshape(1, 1, -1)
    else:
        mm = m; v = np.float32(val)
    out = reg.astype(np.float32) * (1 - mm) + v * mm
    if arr.dtype == np.uint8: out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    arr[y0:y1, x0:x1] = out


# ============================================================== rig._Canvas 的 skia 版（接口不变）
class SkCanvas:
    """与 rig._Canvas 完全同接口：C(h,w,4 uint8: A,tone,red,metal) / M(h,w,3) / G(h,w) / D(h,w float 深度) / dcur。
    poly / cap / circ / ell / line / strand / ribbon 与旧版签名一致；另加 fill(target,...) / stroke(target,...) 供刀身等
    直接写 M/G 通道的地方使用。"""
    backend = 'skia'

    def __init__(self, h, w):
        self.C = np.zeros((h, w, 4), np.uint8)
        self.M = np.zeros((h, w, 3), np.uint8)
        self.G = np.zeros((h, w), np.uint8)
        self.D = np.full((h, w), -1e4, np.float32)
        self.dcur = 0.0
        self.cov = Coverage(h, w)

    # ---- 底层：一条路径 → 覆盖率 → 各通道 lerp
    def _apply(self, paths, col, gold=0, stroke=None, depth=True, smooth_alpha=None):
        m, bb = self.cov.render(paths, stroke=stroke)
        if m is None: return
        lerp_into(self.C, m, bb, col)
        lerp_into(self.G, m, bb, gold)
        if depth:
            y0, y1, x0, x1 = bb
            d = self.D[y0:y1, x0:x1]; d[m >= 0.5] = self.dcur

    def poly(self, pts, col, gold=0):
        self._apply(path_poly(pts), col, gold)

    def cap(self, a, b, r1, r2, col, gold=0):
        self._apply(path_capsule_union(a, b, r1, r2), col, gold)

    def circ(self, c, r, col, gold=0):
        self._apply(path_circle(c, max(r, 0.3)), col, gold)

    def ell(self, c, ax, ang, col, gold=0):
        self._apply(path_ellipse(c, (max(ax[0], .3), max(ax[1], .3)), ang), col, gold)

    def line(self, a, b, w, col, gold=0):
        if w < 1.6:          # 细线：真·亚像素宽度描边（skia 对 <1px 线按覆盖率衰减，比旧版“1px+减色”更准）
            P = path_poly([a, b], closed=False)
            m, bb = self.cov.render(P, stroke=max(0.35, float(w)))
            if m is None: return
            lerp_into(self.C, m, bb, col); lerp_into(self.G, m, bb, gold)
        else:
            self.cap(a, b, w / 2, w / 2, col, gold)

    def strand(self, pts, w0, w1, col):
        """发丝/飘带：整根一条平滑闭合路径（贝塞尔磨圆），不再是逐段拼接。"""
        pts = np.asarray(pts, float)
        n = len(pts)
        if n < 2: return
        if w0 < 1.3:
            P = path_poly(pts, closed=False, smooth=n >= 3)
            m, bb = self.cov.render(P, stroke=max(0.35, float(w0)))
            if m is not None: lerp_into(self.C, m, bb, col)
            return
        if n >= 3: pts = _upsample(pts, 3 if w0 > 2.5 else 2); n = len(pts)
        d = np.gradient(pts, axis=0); nn = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        u = np.linspace(0, 1, n)
        prof = np.clip(0.55 + 2.2 * u, 0, 1) * (1 - u) ** 0.9 if w1 >= 0 else np.sin(np.pi * np.clip(u * 0.95 + 0.03, 0, 1)) ** 0.7
        w = (max(w1, 0) + (w0 - max(w1, 0)) * prof)[:, None] * 0.5
        poly = np.concatenate([pts + nn * w, (pts - nn * w)[::-1]])
        self._apply(path_poly(poly, smooth=True), col)
        if col[1] > 0 and n > 3 and w0 > 3.0:          # 发丝光泽线：细描边，渐细
            hl = (255, min(255, col[1] + 45), col[2], 0)
            m0, m1 = n // 5, max(n // 5 + 2, int(n * 0.7))
            off = pts[m0:m1] - nn[m0:m1] * w[m0:m1] * 0.35
            P = path_poly(off, closed=False, smooth=True)
            m, bb = self.cov.render(P, stroke=max(0.5, w0 * 0.12))
            if m is not None: lerp_into(self.C, m, bb, hl)

    def ribbon(self, pts, widths, normals, cols):
        """逐段着色的带子：每段沿切线前后各外扩 0.6px 与邻段重叠 → 无段间细缝。"""
        pts = np.asarray(pts, float)
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            t = b - a; L = np.hypot(*t)
            e = t / L * 0.6 if L > 1e-6 else t * 0
            q = np.array([a - e + normals[k] * widths[k], b + e + normals[k + 1] * widths[k + 1],
                          b + e - normals[k + 1] * widths[k + 1], a - e - normals[k] * widths[k]])
            self._apply(path_poly(q), cols[k], 0)

    # ---- 通用：对任意通道填充 / 描边（替代直接 cv2.fillPoly(cv_.M, ...)）
    def fill(self, target, pts, val, depth=False):
        m, bb = self.cov.render(path_poly(pts))
        if m is None: return
        lerp_into(getattr(self, target), m, bb, val)
        if depth:
            y0, y1, x0, x1 = bb; d = self.D[y0:y1, x0:x1]; d[m >= 0.5] = self.dcur

    def stroke(self, target, pts, val, width, closed=False):
        m, bb = self.cov.render(path_poly(pts, closed=closed), stroke=max(0.35, float(width)))
        if m is None: return
        lerp_into(getattr(self, target), m, bb, val)

    def fill_ell(self, target, c, ax, ang, val):
        m, bb = self.cov.render(path_ellipse(c, ax, ang))
        if m is None: return
        lerp_into(getattr(self, target), m, bb, val)


def _upsample(p, k):
    p = np.asarray(p, float); n = len(p)
    P = np.vstack([p[:1] * 2 - p[1:2], p, p[-1:] * 2 - p[-2:-1]])
    u = np.linspace(0, 1, k, endpoint=False)[:, None]
    out = []
    for i in range(n - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(p[-1:])
    return np.vstack(out)


# ============================================================== 浮点渲染面（渐变 / 网格 / 模糊直接出 float32）
class FloatSurface:
    """包着 numpy float32 (h,w,4) 的 skia RGBA_F32 面（预乘）。用于渐变填充、带顶点色的网格（刀光扫掠面）、
    MaskFilter 模糊辉光等“连续值”绘制。.rgb() / .alpha() 取回 numpy。"""
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.buf = np.zeros((h, w, 4), np.float32)
        self.surf = skia.Surface(self.buf, colorType=skia.kRGBA_F32_ColorType, alphaType=skia.kPremul_AlphaType)
        self.cv = self.surf.getCanvas()

    def clear(self): self.buf[:] = 0

    def paint(self, color=(1, 1, 1), alpha=1.0, blur=0.0, blend=None, shader=None, stroke=None, cap='round'):
        p = skia.Paint(AntiAlias=True)
        p.setColor4f(skia.Color4f(float(color[0]), float(color[1]), float(color[2]), float(alpha)))
        if blur > 0: p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, float(blur)))
        if blend == 'plus': p.setBlendMode(skia.BlendMode.kPlus)
        elif blend == 'src': p.setBlendMode(skia.BlendMode.kSrc)
        elif blend == 'screen': p.setBlendMode(skia.BlendMode.kScreen)
        if shader is not None: p.setShader(shader)
        if stroke:
            p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(float(stroke))
            p.setStrokeCap(skia.Paint.kRound_Cap if cap == 'round' else skia.Paint.kButt_Cap); p.setStrokeJoin(skia.Paint.kRound_Join)
        return p

    def path(self, P, **kw): self.cv.drawPath(P, self.paint(**kw))

    def mesh(self, pos, cols, idx, blend='plus'):
        """三角网格 + 逐顶点颜色（RGBA 0..1，非预乘），平滑插值，无缝。pos (n,2), cols (n,4), idx (m,) 三角形索引。"""
        c = np.clip(np.asarray(cols, np.float64), 0, 1)
        cc = (np.clip(c[:, 3] * 255 + 0.5, 0, 255).astype(np.uint32) << 24) | (np.clip(c[:, 0] * 255 + .5, 0, 255).astype(np.uint32) << 16) | \
             (np.clip(c[:, 1] * 255 + .5, 0, 255).astype(np.uint32) << 8) | np.clip(c[:, 2] * 255 + .5, 0, 255).astype(np.uint32)
        V = skia.Vertices(skia.Vertices.kTriangles_VertexMode, _pts(pos), None, [int(x) for x in cc], [int(i) for i in idx])
        p = skia.Paint(AntiAlias=True)
        bm = {'plus': skia.BlendMode.kPlus, 'src': skia.BlendMode.kSrc, 'over': skia.BlendMode.kSrcOver}[blend]
        p.setBlendMode(bm)
        self.cv.drawVertices(V, p, skia.BlendMode.kDst)

    def rgb(self): return self.buf[..., :3]
    def alpha(self): return self.buf[..., 3]


def linear_gradient(p0, p1, colors, stops=None):
    """线性渐变着色器。colors: [(r,g,b,a)...] 0..1。"""
    cs = [skia.Color4f(*map(float, c)).toColor() for c in colors]
    return skia.GradientShader.MakeLinear([skia.Point(*map(float, p0)), skia.Point(*map(float, p1))], cs,
                                          None if stops is None else [float(s) for s in stops])

def radial_gradient(c, r, colors, stops=None):
    cs = [skia.Color4f(*map(float, cc)).toColor() for cc in colors]
    return skia.GradientShader.MakeRadial(skia.Point(*map(float, c)), float(r), cs, None if stops is None else [float(s) for s in stops])


def mask(h, w, paths, blur=0.0, stroke=None):
    """一次性：把路径光栅化为 float32 (h,w) 覆盖率（可带 MaskFilter 高斯模糊）。"""
    cv_ = Coverage(h, w)
    m, bb = cv_.render(paths, stroke=stroke, blur=blur)
    out = np.zeros((h, w), np.float32)
    if m is not None: out[bb[0]:bb[1], bb[2]:bb[3]] = m
    return out
