

# ##########################################################################################
# ============================================================== v2 渲染层（角色组第二轮：精致度）
# 接口不变：Swordsman.draw / profile_cu / HandCU.draw / Emperor.draw / _composite 签名与返回值完全一致。
# rig.set_quality('v1'|'v2') 可切回旧画法做前后对比（默认 v2；无 skia 时自动走 v1）。
# 新增：卡通体积明暗（伪法线：深度分区距离场 → 主光赛璐璐 + 天光环境 + 接触遮蔽 + 地面 AO）、
#       loft 躯干（真实腰线）、可变半径肢体（关节过渡）、分层衣摆（前开衩/里衬/褶皱网格渐变）、
#       3D 头部（多凸块并集侧脸轮廓 + 眼/眉/鼻/唇/耳特征，转头可读）、成束头发（发团 + 发束 + 高光环）、
#       护腕/腰封/刀鞘细节、出刀拖影形变（smear）。
QUALITY = 'v2'
def set_quality(q):
    """'v2'（默认，精致版）或 'v1'（第一轮画法，用于对比）。"""
    global QUALITY
    QUALITY = q


def _cel(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


def _shade2(A, D, rd, R, ground=None, key_soft=0.12, occ_k=None, depth_thr=0.9):
    """伪法线体积明暗。A 覆盖、D 深度（-1e4=空）、rd 屏幕光向（光来自 rd 一侧）、R 体积半径（像素，≈肢体半径）。
    返回 (key, amb, ao, nz)：key 主光(赛璐璐软边 0..1)、amb 天光(0.6..1.1)、ao 遮蔽(0..1)。"""
    h, w = A.shape
    Dm = np.where(A > 0.35, D, np.float32(-1e4)).astype(np.float32)
    E = A < 0.5
    dx = np.abs(np.diff(Dm, axis=1)) > depth_thr; dy = np.abs(np.diff(Dm, axis=0)) > depth_thr
    E[:, :-1] |= dx; E[:, 1:] |= dx; E[:-1] |= dy; E[1:] |= dy
    dist = cv2.distanceTransform((~E).astype(np.uint8), cv2.DIST_L2, 3)
    u = np.clip(dist / max(R, 1.0), 0, 1)
    hgt = np.sqrt(u * (2 - u))
    hgt = cv2.GaussianBlur(hgt, (0, 0), max(0.8, R * 0.22))
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) * (R / 8.0)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) * (R / 8.0)
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx * inv, -gy * inv, inv
    ndl = nx * rd[0] + ny * rd[1]
    key = _cel(ndl, 0.10 - key_soft, 0.10 + key_soft)
    amb = 0.62 + 0.45 * np.clip(-ny * 0.9 + 0.35, 0, 1)
    # 接触遮蔽：靠近更近部件处变暗（落影/AO）
    k = int(max(3, occ_k if occ_k is not None else R * 1.6)) | 1
    Dn = cv2.dilate(Dm, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    occ = np.clip((Dn - Dm - depth_thr) / 3.0, 0, 1) * (A > 0.02)
    occ = cv2.GaussianBlur(occ, (0, 0), k * 0.35) * (A > 0.02)
    ao = 1.0 - 0.55 * occ
    if ground is not None:
        yy = np.arange(h, dtype=np.float32)[:, None]
        g0, g1 = ground
        ao = ao * (0.62 + 0.38 * _cel(yy, g0, g1))
    return key, amb, ao, nz, dist


def _comp2(fr, cv_, x0, y0, rim_color, rim_dir, rim, rimw, lift, alpha, ink, R, ground=None, key=1.0,
           tex_scale=1.0, inner_thr=3.5, spec_tone=None, return_A=False, paper=0.0, paper_col=(0.92, 0.88, 0.78),
           burn=None, met_cb=None):
    """v2 合成：tone/red/gold/金属 + 体积明暗 + 轮廓光 → fr。met_cb(col, A, inv) 可选：刀身金属自行着色。"""
    C = cv_.C.astype(np.float32) * (1 / 255.)
    A = C[..., 0]; h_, w_ = A.shape
    if A.max() <= 0: return (fr, A, None) if return_A else (fr, A)
    inv = 1.0 / np.maximum(A, 1e-3)
    tone = C[..., 1] * inv; red = np.clip(C[..., 2] * inv, 0, 1); met = np.clip(C[..., 3] * inv, 0, 1)
    gold = np.clip(cv_.G.astype(np.float32) / 255. * inv, 0, 1)
    rd = np.array(rim_dir, float); rd /= max(np.linalg.norm(rd), 1e-6)
    Dm = cv_.D
    kk, amb, ao, nz, dist = _shade2(A, Dm, rd, R, ground)
    # 布纹（细，低幅）
    ts = max(1, int(round(tex_scale)))
    tt = _tile(noise_tex(512, seed=3, base=40, octaves=3), (x0 // ts) % 512, (y0 // ts) % 512, h_ // ts + 2, w_ // ts + 2)
    tt = cv2.resize(tt, ((w_ // ts + 2) * ts, (h_ // ts + 2) * ts), interpolation=cv2.INTER_LINEAR)[:h_, :w_]
    inkc = INK if ink is None else np.asarray(ink, np.float32)
    rc = np.asarray(rim_color, np.float32)
    shade = (0.50 * amb + 0.95 * key * kk) * ao * (0.93 + 0.14 * tt)
    lum = tone * shade * lift
    cool = np.array([0.255, 0.258, 0.285], np.float32)
    col = inkc[None, None] + lum[..., None] * cool
    col = col + (kk * ao * tone * 0.10 * key)[..., None] * rc                # 受光面染一点光色
    redc = np.array([0.60, 0.05, 0.042], np.float32)
    rshade = (0.30 + 0.85 * tone) * (0.55 * amb + 0.75 * key * kk) * ao * min(1.4, 0.6 + 0.4 * lift)
    col = col * (1 - red[..., None]) + (redc * rshade[..., None]) * red[..., None]
    col = col * (1 - gold[..., None] * 0.65) + GOLD[None, None] * (0.34 * (0.45 + tone) * (0.5 + 0.7 * kk) * gold * 0.65)[..., None]
    glow_add = None
    if burn is not None and burn.get('prog', 0) > 0:
        bp = burn['prog']; yb0, yb1 = burn['y0'], burn['y1']
        nzb = _tile(noise_tex(512, seed=burn.get('seed', 5), base=12, octaves=5), x0 % 512, y0 % 512, h_, w_)
        yy = (np.arange(h_, dtype=np.float32)[:, None] + y0 - yb0) / (yb1 - yb0)
        f = yy * 0.8 + nzb * 0.35 - 0.05
        th_ = bp * 1.15
        keep = np.clip((f - th_) / 0.012, 0, 1)
        charb = np.clip(1 - (f - th_) / 0.05, 0, 1) * keep
        glow_add = np.exp(-((f - th_) / 0.010) ** 2) * (A > 0.05)
        col = col * (1 - charb[..., None] * 0.85)
        A = A * keep
    if paper > 0:
        fib = _tile(noise_tex(512, seed=21, base=60, octaves=3), x0 % 512, y0 % 512, h_, w_)
        thick = np.clip(dist / (rimw * 12 + 1), 0, 1)
        trans = (0.55 + 0.45 * fib) * (1 - 0.55 * thick) * (1 - np.clip(tone - 0.3, 0, 1))
        pc = np.asarray(paper_col, np.float32) * (0.35 + 0.65 * rc / max(rc.max(), 1e-3))
        col = col * (1 - paper) + (pc[None, None] * trans[..., None] + col * 0.2) * paper
    if met_cb is not None and met.max() > 0.01:
        col = met_cb(col, A, inv, met)
    if rim > 0:
        dx, dy = rd
        r1 = np.clip(A - _shift(A, -dx * rimw, -dy * rimw), 0, 1)
        sw_ = rimw * 4
        r2 = _blur_ds(np.clip(A - _shift(A, -dx * sw_, -dy * sw_), 0, 1), sw_ * 0.7 + 0.5) * A
        Ds = _shift(Dm, -dx * rimw * 1.5, -dy * rimw * 1.5)
        inner = (((Dm - Ds) > inner_thr) * (Ds > -1e3)).astype(np.float32)
        inner = cv2.GaussianBlur(inner, (0, 0), 1.0); inner = np.clip((inner - 0.12) * 2.2, 0, 1) * A
        thin = np.clip(dist / max(1.0, rimw * 1.2), 0.35, 1.0)                  # 极细的东西（飞丝）少吃边缘光，避免满屏白线
        rl = (r1 * 0.95 * thin + inner * 0.38 * kk + r2 * 0.34 + met * r2 * 0.6) * rim
        col = col + rc[None, None] * (rl / np.maximum(A, 1e-3))[..., None] * (A > 0.02)[..., None]
    H_, W_ = fr.shape[:2]
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W_, x0 + w_), min(H_, y0 + h_)
    if xb <= xa or yb <= ya: return (fr, A, col) if return_A else (fr, A)
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    a3 = (A[sl] * alpha)[..., None]
    fr[ya:yb, xa:xb] = fr[ya:yb, xa:xb] * (1 - a3) + col[sl] * a3
    if glow_add is not None:
        g = glow_add[sl]
        fr[ya:yb, xa:xb] += (g[..., None] * (GOLD * 1.6 + 0.3) + _blur_ds(g, 6)[..., None] * (RED * 1.5 + GOLD * 0.4)) * alpha
    return (fr, A, col) if return_A else (fr, A)


_composite_v1 = _composite
def _composite(fr, cv_, x0, y0, rim_color=(0.85, 0.9, 1.0), rim_dir=(-0.7, -0.7), rim=1.0, rimw=2.0, lift=1.0, alpha=1.0,
               ink=None, tex_scale=1.0, soft=None, paper=0.0, paper_col=(0.92, 0.88, 0.78), inner_thr=3.5, burn=None, vol=None):
    """接口同 v1；v2 下加体积明暗（vol=体积半径像素，缺省按 rimw 推）。"""
    if QUALITY != 'v2' or BACKEND != 'skia':
        return _composite_v1(fr, cv_, x0, y0, rim_color, rim_dir, rim, rimw, lift, alpha, ink, tex_scale, soft, paper, paper_col,
                             inner_thr, burn)
    bdict = burn
    return _comp2(fr, cv_, x0, y0, rim_color, rim_dir, rim, rimw, lift, alpha, ink, vol if vol else rimw * 7, None,
                  tex_scale=tex_scale, inner_thr=inner_thr, paper=paper, paper_col=paper_col, burn=bdict)
