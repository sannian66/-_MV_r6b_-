

# ---------------------------------------------------------------- 3D 头部（头部系：x 右, y 上, z 前；原点 Hc）
def _ellip_pts(c, r, na=18, bs=(-1.2, -0.75, -0.3, 0.15, 0.6, 1.05, 1.45), keep=None):
    out = []
    for b in bs:
        for a in np.linspace(0, 2 * math.pi, na, endpoint=False):
            p = [c[0] + r[0] * math.cos(b) * math.sin(a), c[1] + r[1] * math.sin(b), c[2] + r[2] * math.cos(b) * math.cos(a)]
            if keep is None or keep(p): out.append(p)
    return np.array(out)

def _mirror(pts):
    pts = np.asarray(pts, float); m = pts.copy(); m[:, 0] *= -1
    return np.vstack([pts, m])

HEAD2 = dict(
    cran=_ellip_pts((0, 0.9, -0.55), (4.85, 5.55, 5.75)),
    jaw=_mirror([[4.15, -0.6, 2.4], [3.95, -3.3, -0.2], [3.4, -4.6, 1.2], [2.4, -5.6, 3.3], [1.25, -6.25, 4.35],
                 [3.3, 1.6, 4.1], [3.55, -1.4, 3.6], [1.9, 2.15, 5.15], [1.2, 4.1, 4.9], [0.0, 2.25, 5.72], [0.0, 4.3, 5.1],
                 [0.0, -2.55, 5.25], [0.0, -4.55, 5.15], [0.0, -6.45, 4.1], [1.0, -2.6, 5.0]]),
    nose=_mirror([[0.0, 1.45, 5.55], [0.45, 1.1, 5.3], [0.0, -1.35, 6.85], [0.0, -1.85, 6.6], [1.0, -1.95, 5.55],
                  [0.0, -2.3, 5.85], [0.55, -2.3, 5.3]]),
    ulip=_mirror([[0.0, -2.3, 5.8], [0.0, -2.95, 6.02], [1.35, -3.15, 5.05], [0.0, -3.42, 5.72], [0.85, -2.4, 5.2]]),
    llip=_mirror([[0.0, -3.5, 5.72], [0.0, -3.85, 5.86], [1.05, -3.55, 4.95], [0.0, -4.3, 5.42]]),
    chin=_mirror([[0.0, -4.55, 5.3], [0.0, -5.25, 5.72], [0.0, -5.95, 5.45], [0.0, -6.45, 4.55], [1.35, -5.35, 4.95],
                  [1.3, -6.1, 4.3], [0.8, -4.6, 5.05]]),
)
# 发团（头盖）：略大的椭球，去掉脸部
HAIR_CAP = _ellip_pts((0, 1.25, -0.75), (5.35, 6.05, 6.3), na=22, bs=(-1.0, -0.6, -0.2, 0.2, 0.6, 0.95, 1.25, 1.5),
                      keep=lambda p: not (p[2] > 1.2 and p[1] < 3.6 - max(0, abs(p[0]) - 3.4) * 2.2) and p[1] > -3.2)


def _eye_pts(sg, gaze=(0.0, 0.0), open_=1.0):
    """单眼（sg=+1 角色右眼 / -1 左眼）上睑、下睑、虹膜 3D 点（头部系）。外眼角略上挑，冷峻。"""
    u = np.linspace(-1, 1, 15)            # -1 内眼角 → +1 外眼角
    cx, cy = 2.05, 0.55
    x = cx + 1.12 * u
    zf = lambda xx: 5.15 - 0.165 * xx ** 2
    yu = cy + 0.50 * open_ * (1 - u ** 2) ** 0.75 + 0.16 * u - 0.05 * (u < -0.6)
    yl = cy - 0.24 * (1 - u ** 2) ** 1.2 + 0.10 * u
    up = np.stack([sg * x, yu, zf(x) + 0.05], 1); lo = np.stack([sg * x, yl, zf(x)], 1)
    a = np.linspace(0, 2 * math.pi, 20, endpoint=False)
    ic = np.array([sg * (cx + gaze[0] * 0.5 - 0.06), cy + 0.06 + gaze[1] * 0.3, zf(cx) + 0.12])
    ex = _nrm(np.array([1.0, 0, -0.165 * 2 * cx * sg * sg]) * [sg, 1, 1]); ey = np.array([0, 1.0, 0])
    ir = ic + 0.5 * (np.cos(a)[:, None] * ex + np.sin(a)[:, None] * ey * 1.02)
    pu = ic + 0.22 * (np.cos(a)[:, None] * ex + np.sin(a)[:, None] * ey)
    return up, lo, ir, pu, ic


def _draw_head2(cv_, P2, DP, S, sc, vdir, L3, skin=165, hair_tone=34, detail=1.0, E=None, hd0=0.0):
    """头部 v2：多凸块并集（侧脸轮廓）+ 五官 + 耳 + 发盖。返回绘制函数列表 [(depth, fn, bias)]。"""
    Rh, Hc = S['Rh'], S['Hc']
    W = lambda p: Hc + np.asarray(p, float) @ Rh.T
    out = []
    face_n = Rh @ [0, 0, 1.0]
    facing = float(face_n @ vdir)                       # 1=正脸 0=侧脸 -1=后脑
    side = float((Rh @ [1.0, 0, 0]) @ vdir)              # >0 角色右侧朝镜头
    pieces = [cv2.convexHull(P2(W(np.vstack([HEAD2['cran'], HEAD2['jaw']]))).astype(np.float32)).reshape(-1, 2)]
    for k in ('nose', 'ulip', 'llip', 'chin'):
        pieces.append(cv2.convexHull(P2(W(HEAD2[k])).astype(np.float32)).reshape(-1, 2))
    big = sc > 5.5
    feat = sc > 7.0 and facing > -0.35
    neck = P2(np.array([S['N'] - S['Rc'] @ [0, 1.5, 0], W([0, -3.2, -0.8]), W([0, -4.6, 1.2])]))

    def headfn():
        # 颈：胸锁乳突肌斜线（侧面）
        cv_.path(_sk.path_limb(neck[:2], [3.05 * sc, 2.65 * sc], k=2)[0], _col(skin * 0.86))
        cv_.polys(pieces, _col(skin), smooth=False)
        if not big: return
        # 颧下/下颌阴影、眼窝（内绘，软）
        sk_ = _sk
        for sg in (-1, 1):
            vis = float((Rh @ _nrm(np.array([sg * 0.45, 0.05, 1.0]))) @ vdir)
            if vis < -0.15: continue
            ec = P2(W([sg * 2.0, 0.75, 4.7]))
            cv_.paint_in(sk_.path_ellipse(ec, (1.55 * sc, 1.05 * sc), 0), (skin * 0.62,), 0.55, blur=0.6 * sc)
            ch = P2(W([sg * 3.3, -2.6, 3.6]))
            cv_.paint_in(sk_.path_ellipse(ch, (0.9 * sc, 1.9 * sc), 15 * sg), (skin * 0.72,), 0.45 * min(1, sc / 10), blur=0.9 * sc)
        jaw_line = P2(W(np.array([[4.0, -3.4, -0.2], [3.3, -4.7, 1.3], [2.3, -5.7, 3.3], [1.1, -6.3, 4.4], [0, -6.5, 4.1]])))
        cv_.paint_in(sk_.path_poly(jaw_line, closed=False, smooth=True), (skin * 0.55,), 0.35, blur=0.35 * sc, stroke=0.5 * sc)
        if not feat: return
        # 耳
        for sg in (-1, 1):
            en = Rh @ [sg, 0, 0.15]
            if en @ vdir < 0.25: continue
            ring = W(np.array([[sg * 4.75, 0.2 + 1.35 * math.sin(a), -0.35 + 0.75 * math.cos(a) - 0.25 * (math.sin(a) < 0)]
                               for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]))
            q = P2(ring)
            cv_.paint_in(sk_.path_poly(q, smooth=True), (skin * 0.9,), 1.0)
            cv_.paint_in(sk_.path_poly(q, smooth=True), (skin * 0.45,), 0.8, stroke=0.25 * sc)
            inner = P2(W(np.array([[sg * 4.8, 0.2 + 0.8 * math.sin(a), -0.25 + 0.4 * math.cos(a)] for a in np.linspace(0.3, 5.0, 9)])))
            cv_.paint_in(sk_.path_poly(inner, closed=False, smooth=True), (skin * 0.5,), 0.7, stroke=0.18 * sc)
        # 眉（粗、内低外高，冷峻）
        for sg in (-1, 1):
            vis = float((Rh @ _nrm(np.array([sg * 0.5, 0.1, 1.0]))) @ vdir)
            if vis < -0.05: continue
            bw = np.array([[sg * (0.65 + 2.75 * u), 1.72 + 0.62 * u - 0.18 * u * u, 5.45 - 0.165 * (0.65 + 2.75 * u) ** 2 * 0.8 + 0.15]
                           for u in np.linspace(0, 1, 9)])
            q = P2(W(bw)); d = np.gradient(q, axis=0); nn = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
            wd = (0.34 - 0.2 * np.linspace(0, 1, 9) ** 1.5)[:, None] * sc
            cv_.paint_in(sk_.path_poly(np.vstack([q + nn * wd, (q - nn * wd)[::-1]]), smooth=True), (hair_tone * 0.7,), 0.95)
        # 眼
        for sg in (-1, 1):
            vis = float((Rh @ _nrm(np.array([sg * 0.42, 0.0, 1.0]))) @ vdir)
            if vis < 0.0: continue
            up, lo, ir, pu, ic = _eye_pts(sg)
            qu, ql = P2(W(up)), P2(W(lo))
            white = sk_.path_poly(np.vstack([qu, ql[::-1]]), smooth=True)
            cv_.paint_in(white, (skin * 1.18,), 1.0)
            irp = sk_.path_and(sk_.path_poly(P2(W(ir)), smooth=True), white)
            cv_.paint_in(irp, (34,), 1.0)
            cv_.paint_in(sk_.path_and(sk_.path_poly(P2(W(pu)), smooth=True), white), (6,), 1.0)
            # 上睑阴影落在眼球上
            cv_.paint_in(sk_.path_and(sk_.path_poly(qu + [0, 0.35 * sc], closed=False, smooth=True), white), (20,), 0.55,
                         blur=0.25 * sc, stroke=0.55 * sc)
            # 上睑线（粗，外眼角拖出眼尾）+ 双眼皮细线 + 下睑
            tail = qu[-1] + (qu[-1] - qu[-3]) * 0.9
            lid = np.vstack([qu, tail[None]])
            d = np.gradient(lid, axis=0); nn = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
            prof = np.sin(np.linspace(0.12, 1.0, len(lid)) * math.pi * 0.82) ** 0.6
            wl = (0.07 + 0.2 * prof)[:, None] * sc
            cv_.paint_in(sk_.path_poly(np.vstack([lid + nn * wl, (lid - nn * wl)[::-1]]), smooth=True), (8,), 1.0)
            crease = P2(W(up[3:13] + np.array([0, 0.36, 0.0])))
            cv_.paint_in(sk_.path_poly(crease, closed=False, smooth=True), (skin * 0.5,), 0.8, stroke=0.12 * sc)
            cv_.paint_in(sk_.path_poly(ql[5:], closed=False, smooth=True), (skin * 0.55,), 0.8, stroke=0.1 * sc)
        # 鼻：鼻梁暗侧 + 鼻翼 + 鼻孔
        nl = 1 if side < 0 else -1                       # 背光侧
        if facing > 0.15:
            br = P2(W(np.array([[nl * 0.55, 1.2, 5.3], [nl * 0.5, -0.4, 6.0], [nl * 0.6, -1.5, 6.3]])))
            cv_.paint_in(sk_.path_poly(br, closed=False, smooth=True), (skin * 0.62,), 0.55, blur=0.25 * sc, stroke=0.45 * sc)
        for sg in (-1, 1):
            nv = float((Rh @ _nrm(np.array([sg * 0.6, -0.4, 0.8]))) @ vdir)
            if nv < -0.05: continue
            wing = P2(W(np.array([[sg * 0.25, -2.2, 5.95], [sg * 0.85, -2.05, 5.6], [sg * 1.05, -1.65, 5.45]])))
            cv_.paint_in(sk_.path_poly(wing, closed=False, smooth=True), (skin * 0.45,), 0.85, stroke=0.13 * sc)
            if facing > 0.3:
                cv_.paint_in(sk_.path_ellipse(P2(W([sg * 0.5, -2.15, 5.75])), (0.22 * sc, 0.1 * sc), 0), (skin * 0.3,), 0.9)
        # 唇：口裂线 + 下唇阴影
        mouth = P2(W(np.array([[x_, -3.45 - 0.05 * abs(x_), 5.72 - 0.38 * x_ * x_] for x_ in np.linspace(-1.3, 1.3, 9)])))
        mz = np.array([float((Rh @ _nrm(np.array([x_ * 0.5, 0, 1.0]))) @ vdir) for x_ in np.linspace(-1.3, 1.3, 9)])
        ok = mz > -0.1
        if ok.sum() >= 2:
            cv_.paint_in(sk_.path_poly(mouth[ok], closed=False, smooth=True), (skin * 0.36,), 0.95, stroke=0.13 * sc)
            ll = P2(W(np.array([[x_, -4.3, 5.35 - 0.3 * x_ * x_] for x_ in np.linspace(-0.8, 0.8, 5)])))
            cv_.paint_in(sk_.path_poly(ll, closed=False, smooth=True), (skin * 0.6,), 0.5, blur=0.2 * sc, stroke=0.3 * sc)
    out.append((DP(Hc), headfn, 0.0))
    # 发盖（头顶 + 后脑 + 鬓）
    capq = P2(W(HAIR_CAP))
    cap = cv2.convexHull(capq.astype(np.float32)).reshape(-1, 2)
    ring = np.array([[5.2 * math.cos(b) * math.sin(a), 1.2 + 6.0 * math.sin(b), -0.75 + 6.2 * math.cos(b) * math.cos(a)]
                     for a, b in [(a_, 0.85) for a_ in np.linspace(-2.6, 2.6, 22)]])
    ringq = P2(W(ring)); rv = (W(ring) - Hc) @ vdir > -0.5
    part = P2(W(np.array([[0.4 * math.sin(u * 3), 1.2 + 6.05 * math.sin(b_), -0.75 + 6.3 * math.cos(b_)] for u, b_ in
                          zip(np.linspace(0, 1, 8), np.linspace(1.15, 0.2, 8))])))
    def capfn():
        cv_.poly(cap, _col(hair_tone))
        if big:
            # 高光环（天使环）：分段短束高光，随头转
            seg = np.nonzero(rv)[0]
            for i in seg[::2]:
                j = min(i + 2, len(ringq) - 1)
                if not rv[j]: continue
                cv_.paint_in(_sk.path_poly(ringq[i:j + 1], closed=False, smooth=True), (hair_tone + 70,), 0.75, blur=0.15 * sc,
                             stroke=0.55 * sc)
            cv_.paint_in(_sk.path_poly(part, closed=False, smooth=True), (hair_tone * 0.4,), 0.6, stroke=0.25 * sc)
    out.append((DP(Hc) + 0.4, capfn, 0.0))
    eyes = []
    for sg in (-1, 1):
        vis = float((Rh @ _nrm(np.array([sg * 0.42, 0.0, 1.0]))) @ vdir)
        if vis > 0.05:
            _, _, _, _, ic = _eye_pts(sg); eyes.append((vis, P2(W(ic + np.array([-0.12 * sg, 0.18, 0.1])))))
    return out, eyes, facing


def _clump(cv_, q, w0, w1, tone, sc, hl=40, hl_side=1, red=0, depth=True, prof='taper'):
    """一束头发：渐细填充 + 一侧高光线。q (n,2) 局部像素点，w0/w1 根/梢宽（像素）。"""
    q = np.asarray(q, float)
    if len(q) < 2: return
    if len(q) >= 3: q = _upsample(q, 3)
    n = len(q); u = np.linspace(0, 1, n)
    d = np.gradient(q, axis=0); nn = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    if prof == 'taper':
        pr = np.clip(0.75 + 1.2 * u, 0, 1) * (1 - u) ** 0.85
    else:
        pr = np.sin(np.pi * np.clip(u * 0.9 + 0.08, 0, 1)) ** 0.6
    w = (w1 + (w0 - w1) * pr)[:, None] * 0.5
    poly = np.vstack([q + nn * w, (q - nn * w)[::-1]])
    P = _sk.path_poly(poly, smooth=True)
    cv_.path(P, _col(tone, red), depth=depth)
    if hl and w0 > 2.2:
        m0, m1 = max(1, n // 6), max(n // 6 + 2, int(n * 0.62))
        hq = q[m0:m1] + nn[m0:m1] * w[m0:m1] * 0.35 * hl_side
        cv_.paint_in(_sk.path_poly(hq, closed=False, smooth=True), (tone + hl, red), 0.8, blur=max(0.3, w0 * 0.06),
                     stroke=max(0.6, w0 * 0.16))
        if w0 > 6:
            cv_.paint_in(_sk.path_poly(q[1:-2] - nn[1:-2] * w[1:-2] * 0.55 * hl_side, closed=False, smooth=True), (tone * 0.45, red),
                         0.6, blur=w0 * 0.08, stroke=w0 * 0.18)
