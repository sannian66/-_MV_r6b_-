"""units/hitmap.py — 副歌一 58–90s 鼓件分离级卡点图
python3 units/hitmap.py  → crew/hitmap_c1.json + crew/hitmap_c1.png

方法：HPSS 取打击乐分量 → STFT 频段拆分（底鼓 <150Hz、军鼓 150–4kHz 瞬态（以 1.2–4k 为主判据）、镲/hi-hat >6kHz）
→ 各频段谱通量 onset 包络 → 自适应峰值检测 → 同一时刻多频段合并分类 → 强度归一化
→ 结合 analysis.npz beats（拍网格、小节号、强拍）和 align.json 逐字时间（字头）。
所有时间同时给出最近帧号（30fps）与「视觉帧」= 最近帧 - 1（感知同步：画面早 1 帧）。"""
import json, sys, math
import numpy as np
ROOT = '/workspace/work/mv'
sys.path.insert(0, ROOT + '/lib'); sys.path.insert(0, ROOT)
FPS = 30
A0, A1 = 56.0, 92.0      # 分析窗（含前后余量）
W0, W1 = 58.0, 90.0      # 输出窗


def frame_of(t): return int(round(t * FPS))


def main():
    import librosa
    y, sr = librosa.load(ROOT + '/song.mp3', sr=44100, mono=True, offset=A0, duration=A1 - A0)
    D = librosa.stft(y, n_fft=2048, hop_length=256)
    H_, P_ = librosa.decompose.hpss(D, margin=(1.0, 2.0))
    S = np.abs(P_)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hop_t = 256 / sr
    tt = A0 + np.arange(S.shape[1]) * hop_t

    def band_flux(lo, hi, log=True):
        m = (f >= lo) & (f < hi)
        X = S[m]
        if log: X = np.log1p(X * 10)
        fl = np.maximum(0, np.diff(X, axis=1, prepend=X[:, :1])).sum(0)
        fl = np.convolve(fl, np.hanning(5) / np.hanning(5).sum(), 'same')
        return fl / (np.percentile(fl, 99.5) + 1e-9)

    def band_energy(lo, hi):
        m = (f >= lo) & (f < hi)
        e = (S[m] ** 2).sum(0)
        return e / (np.percentile(e, 99.5) + 1e-9)

    kick = band_flux(30, 150)
    snare = band_flux(150, 4000) * 0.4 + band_flux(1200, 4000) * 0.6
    hat = band_flux(6000, 16000)
    full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    full = full[:len(tt)] / (np.percentile(full, 99.5) + 1e-9)
    e_kick = band_energy(30, 150)

    def peaks(env, thr, wait_s, delta):
        p = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=12, post_avg=6, delta=delta,
                                   wait=int(wait_s / hop_t))
        return [int(i) for i in p if env[i] >= thr]

    pk = {'kick': peaks(kick, 0.28, 0.11, 0.08), 'snare': peaks(snare, 0.30, 0.09, 0.07), 'hat': peaks(hat, 0.25, 0.06, 0.05)}
    an = np.load(ROOT + '/analysis.npz')
    beats = an['beats']
    bw = float(np.median(np.diff(beats)))

    hits = []
    for typ, idx in pk.items():
        env = {'kick': kick, 'snare': snare, 'hat': hat}[typ]
        for i in idx:
            t = float(tt[i])
            if not (W0 <= t <= W1): continue
            # 回溯到瞬态起点（峰前包络上升沿 50%）
            j = i
            while j > 0 and env[j - 1] > env[i] * 0.5 and i - j < 6: j -= 1
            t_on = float(tt[j])
            hits.append(dict(t=round(t_on, 3), type=typ, raw=float(env[i])))
    # 各类型内强度归一（分位数）→ 0–1
    for typ in pk:
        v = np.array([h['raw'] for h in hits if h['type'] == typ])
        if len(v) == 0: continue
        lo, hi = np.percentile(v, 5), np.percentile(v, 98)
        for h in hits:
            if h['type'] == typ: h['strength'] = round(float(np.clip((h['raw'] - lo) / (hi - lo + 1e-9), 0, 1)) * 0.8 + 0.2, 3)
    # 军鼓 / 底鼓 去串扰：同一 35ms 内若底鼓更强，则军鼓标记为 kick+snare（合击）
    hits.sort(key=lambda h: h['t'])
    merged = []
    for h in hits:
        if merged and abs(h['t'] - merged[-1]['t']) < 0.035 and h['type'] != merged[-1]['type']:
            m = merged[-1]
            types = sorted(set(m['type'].split('+')) | {h['type']})
            m['type'] = '+'.join(t for t in ('kick', 'snare', 'hat') if t in types)
            m['strength'] = round(max(m['strength'], h['strength']), 3)
            m['t'] = round(min(m['t'], h['t']), 3)
            continue
        merged.append(dict(h))
    hits = merged
    # 拍网格信息
    bsel = beats[(beats >= W0 - 1) & (beats <= W1 + 1)]
    # 强拍相位：在每 4 拍相位上累加底鼓强度，取最大者为小节第 1 拍
    bidx = np.searchsorted(beats, W0 - 1)
    score = np.zeros(4)
    for k, b in enumerate(bsel):
        i = np.searchsorted(tt, b)
        score[(bidx + k) % 4] += kick[max(0, i - 3):i + 4].max() + e_kick[max(0, i - 3):i + 4].max()
    ph = int(np.argmax(score))
    for h in hits:
        bi = int(np.argmin(np.abs(beats - h['t'])))
        off = h['t'] - beats[bi]
        h['beat'] = int(bi)
        h['beat_pos'] = int((bi - ph) % 4) + 1           # 小节内第几拍
        h['on_beat'] = bool(abs(off) < 0.06)
        sub = (h['t'] - beats[bi]) / bw
        h['sub'] = float(round(sub * 4) / 4)                  # 16 分位置（-0.5..0.5）
        h['frame'] = frame_of(h['t'])
        h['vis_frame'] = h['frame'] - 1                # 画面早一帧
        del h['raw']
    # 歌词字头
    import typo as T
    L = T.load_lines()
    words = []
    for li in range(12, 20):
        txt = [c for c in L[li]['text'] if c.strip()]
        ts = [x for x in L[li]['times'] if x is not None]
        for k, (c, t) in enumerate(zip(txt, ts)):
            # 重音字：落在拍上（±70ms）或句首/句尾，或是关键字
            bi = int(np.argmin(np.abs(beats - t)))
            onb = abs(t - beats[bi]) < 0.07
            key = c in '血刀谁正邪墙劈亮骨扛天判量答'
            words.append(dict(line=li, k=k, ch=c, t=round(float(t), 3), frame=frame_of(t), vis_frame=frame_of(t) - 1,
                              on_beat=bool(onb), accent=bool(key or onb), beat_pos=int((bi - ph) % 4) + 1))
    # 乐句（按歌词句）
    phrases = [dict(line=li, text=L[li]['text'], start=round(float([x for x in L[li]['times'] if x][0]), 3),
                    end=round(float(L[li + 1]['times'][0]) if li < 19 else 86.19, 3)) for li in range(12, 20)]
    beat_list = [dict(i=int(i), t=round(float(beats[i]), 3), frame=frame_of(beats[i]), pos=int((i - ph) % 4) + 1)
                 for i in range(len(beats)) if W0 - 0.5 <= beats[i] <= W1 + 0.5]
    cnt = {}
    for h in hits:
        for ty in h['type'].split('+'): cnt[ty] = cnt.get(ty, 0) + 1
    out = dict(window=[W0, W1], fps=FPS, bpm=round(60 / bw, 2), beat_s=round(bw, 4), downbeat_phase=int(ph),
               counts=dict(total=len(hits), **cnt), method=__doc__.strip().splitlines()[2:5],
               hits=hits, beats=beat_list, words=words, phrases=phrases)
    json.dump(out, open(ROOT + '/crew/hitmap_c1.json', 'w'), ensure_ascii=False, indent=1)
    print('hits', len(hits), cnt, 'bpm', 60 / bw, 'downbeat phase', ph)
    plot(y, sr, tt, kick, snare, hat, out)


def plot(y, sr, tt, kick, snare, hat, out):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    Wd, Hh = 3600, 1200
    rows = 2
    img = Image.new('RGB', (Wd, Hh * rows // 2 * 1), (14, 14, 18))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(ROOT + '/fonts/NotoSerifSC.otf', 20)
    seg = [(58, 74), (74, 90)]
    rh = Hh // 2
    for r, (a, b) in enumerate(seg):
        y0 = r * rh
        X = lambda t: int(60 + (t - a) / (b - a) * (Wd - 80))
        # 波形
        s0, s1 = int((a - A0) * sr), int((b - A0) * sr)
        w = y[s0:s1]
        n = Wd - 80
        ch = w[:len(w) // n * n].reshape(n, -1)
        mx, mn = ch.max(1), ch.min(1)
        mid = y0 + 170
        for i in range(n):
            d.line([(60 + i, mid - mx[i] * 140), (60 + i, mid - mn[i] * 140)], fill=(90, 95, 110))
        # 包络
        for env, col, base in ((kick, (230, 60, 50), y0 + 420), (snare, (240, 200, 110), y0 + 500), (hat, (120, 180, 255), y0 + 570)):
            m = (tt >= a) & (tt <= b)
            pts = [(X(t), base - min(1.5, v) * 50) for t, v in zip(tt[m], env[m])]
            d.line(pts, fill=col, width=1)
        # 拍网格
        for bt in out['beats']:
            if a <= bt['t'] <= b:
                c = (200, 200, 200) if bt['pos'] == 1 else (70, 70, 80)
                d.line([(X(bt['t']), y0 + 20), (X(bt['t']), y0 + 590)], fill=c, width=2 if bt['pos'] == 1 else 1)
        # 击打
        for h in out['hits']:
            if not (a <= h['t'] <= b): continue
            x = X(h['t']); s = h['strength']
            for ty, col, yy in (('kick', (230, 60, 50), 300), ('snare', (240, 200, 110), 330), ('hat', (120, 180, 255), 355)):
                if ty in h['type']:
                    rr = 3 + 8 * s
                    d.ellipse([x - rr, y0 + yy - rr, x + rr, y0 + yy + rr], fill=col)
        # 字头
        for wd in out['words']:
            if a <= wd['t'] <= b:
                x = X(wd['t'])
                d.line([(x, y0 + 30), (x, y0 + 70)], fill=(255, 120, 100) if wd['accent'] else (180, 180, 180), width=2)
                d.text((x - 10, y0 + 2), wd['ch'], font=fnt, fill=(255, 140, 120) if wd['accent'] else (220, 220, 220))
        for k in range(int(a), int(b) + 1):
            d.text((X(k) - 10, y0 + 592 - 22), f'{k}', fill=(150, 150, 150))
        d.text((8, y0 + 290), 'K', fill=(230, 60, 50)); d.text((8, y0 + 320), 'S', fill=(240, 200, 110)); d.text((8, y0 + 345), 'H', fill=(120, 180, 255))
    img.save(ROOT + '/crew/hitmap_c1.png')
    print(ROOT + '/crew/hitmap_c1.png')


if __name__ == '__main__':
    main()
