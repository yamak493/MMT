"""口パク: 音素認識の結果から、口のモーフ（あ・い・う・え・お・ん）のキーを作る。

Allosaurus の時刻は音素が鳴り始める 1 点（30ms 刻み）で長さは分からないので、口を開けている長さと開き具合は
ボーカルの音量から決める。
1. 音素を口の形に分ける: 母音 → あいうえお（IPA の母音を近い日本語の母音へ）、両唇音 m・b・p と撥音 → ん、
   それ以外の子音は形を持たない（すぐ後の母音の形を子音の時刻から始める）
2. 声が出ている区間を音量のヒステリシスで決め、声が出ていない間はどのモーフも 0 にする
3. 母音のモーフの値は声の大きさで min_open〜1 × strength、ん は closed_weight
4. ガウシアンでならしてフレームごとに平均し、線形補間で済む所のキーを省く
"""
import unicodedata

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..core.filters import runs

SHAPES = ('a', 'i', 'u', 'e', 'o', 'n')
KANA = dict(zip(SHAPES, 'あいうえおん'))
CLOSED = SHAPES.index('n')
_VOWELS = {'a': 'aɑɐæʌɶəɜɝɚ', 'i': 'iɪɨ', 'u': 'uɯʊʉyʏ', 'e': 'eɛɘ', 'o': 'oɔɒɤɵøœɞ'}
VOWEL_SHAPE = {ch: s for s, chars in _VOWELS.items() for ch in chars}
CLOSED_PHONES = set('mbpɓʙɴ')
GLIDE_SHAPE = {'j': 'i', 'w': 'u', 'ɥ': 'u', 'ʍ': 'u'}
_MODIFIERS = set('ːˑʰʲʷˠˤʼ˞')


def _base(phone):
    s = unicodedata.normalize('NFD', str(phone))
    return ''.join(ch for ch in s if not unicodedata.combining(ch) and ch not in _MODIFIERS)


def phone_shape(phone):
    """音素（IPA）の口の形: a / i / u / e / o / n（口を閉じる）/ None（子音）。"""
    base = _base(phone)
    for ch in base:
        if ch in VOWEL_SHAPE:
            return VOWEL_SHAPE[ch]
    if base[:1] in CLOSED_PHONES:
        return 'n'
    return GLIDE_SHAPE.get(base[:1])


def parse_allosaurus(text):
    """Allosaurus の recognize(..., timestamp=True) の出力（'時刻 長さ 音素' の行）→ (時刻, 音素)。"""
    times, phones = [], []
    for line in str(text).splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        times.append(t)
        phones.append(parts[2])
    return np.asarray(times, np.float64), np.asarray(phones, dtype=str)


def vocal_level_db(samples, sample_rate, rate=100.0, window_sec=0.03):
    """音量 [dB]（フルスケール = 0 dB）を rate [Hz] ごとに求める。"""
    x = np.asarray(samples)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = np.asarray(x, np.float64).reshape(len(x), -1).mean(axis=1)
    n = int(np.ceil(len(x) / sample_rate * rate))
    centers = np.arange(n) / rate * sample_rate
    half = 0.5 * window_sec * sample_rate
    lo = np.clip(np.round(centers - half).astype(np.int64), 0, len(x))
    hi = np.clip(np.round(centers + half).astype(np.int64), 0, len(x))
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    ms = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return 10.0 * np.log10(np.maximum(ms, 1e-12))


def voice_levels(level_db, cfg):
    level = np.asarray(level_db, np.float64)
    if len(level) == 0:
        return dict(reference=float('-inf'), noise=float('-inf'), on=float('inf'), off=float('inf'))
    ref = float(np.percentile(level[level >= level.max() + cfg.voice_off_db], cfg.reference_percentile))
    noise = float(np.percentile(level, cfg.noise_percentile))
    on = min(max(ref + cfg.voice_on_db, noise + cfg.on_above_noise_db), ref + cfg.max_on_db)
    off = min(max(ref + cfg.voice_off_db, noise + cfg.off_above_noise_db), on)
    return dict(reference=ref, noise=noise, on=on, off=off)


def voice_activity(level_db, rate, cfg):
    level = np.asarray(level_db, np.float64)
    levels = voice_levels(level, cfg)
    voiced = np.zeros(len(level), bool)
    if len(level) == 0 or levels['reference'] < cfg.min_reference_dbfs:
        return voiced, levels
    loud = level >= levels['on']
    for s, e in runs(level >= levels['off']):
        if loud[s:e + 1].any():
            voiced[s:e + 1] = True
    gap = int(round(cfg.fill_gap_sec * rate))
    segs = runs(voiced)
    for (_, e0), (s1, _) in zip(segs[:-1], segs[1:]):
        if s1 - e0 - 1 <= gap:
            voiced[e0 + 1:s1] = True
    min_len = int(round(cfg.min_voice_sec * rate))
    for s, e in runs(voiced):
        if e - s + 1 < min_len:
            voiced[s:e + 1] = False
    return voiced, levels


def _run_events(events, t_end, fallback, cfg):
    out = []
    for k, (t, shape) in enumerate(events):
        if shape is None:
            nxt = next(((t2, s2) for t2, s2 in events[k + 1:] if s2 is not None), None)
            if nxt is None or nxt[0] - t > cfg.consonant_sec:
                continue
            shape = nxt[1]
        out.append((t, SHAPES.index(shape)))
    fixed = []
    for k, (t, idx) in enumerate(out):
        fixed.append((t, idx))
        t_next = out[k + 1][0] if k + 1 < len(out) else t_end
        if idx != CLOSED or t_next - t <= cfg.max_closed_sec:
            continue
        back = next((i for _, i in reversed(fixed) if i != CLOSED), None)
        ahead = next((i for _, i in out[k + 1:] if i != CLOSED), None)
        vowel = back if back is not None else (ahead if ahead is not None else fallback)
        if vowel >= 0:
            fixed.append((t + cfg.max_closed_sec, vowel))
    return fixed


def shape_timeline(times, shapes, voiced, rate, cfg):
    label = np.full(len(voiced), -1, np.int64)
    fallback = SHAPES.index(cfg.fallback_vowel) if cfg.fallback_vowel in SHAPES else -1
    order = np.argsort(times, kind='stable')
    times = np.asarray(times, np.float64)[order]
    shapes = [shapes[i] for i in order]
    margin, lead = float(cfg.event_margin_sec), float(cfg.lead_sec)
    for s, e in runs(voiced):
        t0, t1 = s / rate, (e + 1) / rate
        sel = np.flatnonzero((times >= t0 - margin) & (times <= t1 + margin))
        changes = _run_events([(times[i], shapes[i]) for i in sel], t1, fallback, cfg)
        if not changes:
            label[s:e + 1] = fallback
            continue
        starts = (np.array([t for t, _ in changes]) - lead) * rate
        k = np.searchsorted(starts, np.arange(s, e + 1), side='right') - 1
        label[s:e + 1] = np.array([i for _, i in changes])[np.maximum(k, 0)]
    return label


def shape_weights(label, level_db, reference_db, rate, cfg):
    amp = np.clip(1.0 + (np.asarray(level_db, np.float64) - reference_db) / cfg.open_range_db, 0.0, 1.0)
    vowel = cfg.strength * (cfg.min_open + (1.0 - cfg.min_open) * amp)
    w = np.zeros((len(SHAPES), len(label)))
    for i in range(len(SHAPES)):
        on = label == i
        w[i, on] = cfg.closed_weight if i == CLOSED else vowel[on]
    sigma = float(cfg.smooth_sec) * rate
    if sigma > 0 and w.shape[1] > 1:
        w = gaussian_filter1d(w, sigma, axis=1, mode='nearest')
    return w


def to_frames(values, rate, fps, num_frames):
    values = np.asarray(values, np.float64)
    M = values.shape[-1]
    idx = np.floor(np.arange(M) / rate * fps + 0.5).astype(np.int64)
    ok = idx < num_frames
    counts = np.bincount(idx[ok], minlength=num_frames).astype(np.float64)
    flat = values.reshape(int(np.prod(values.shape[:-1])), M)
    out = np.stack([np.bincount(idx[ok], weights=v[ok], minlength=num_frames) for v in flat])
    return (out / np.maximum(counts, 1.0)).reshape(values.shape[:-1] + (num_frames,))


def thin_weights(weights, tol):
    """線形補間したときの誤差が tol 以内なら、キーを省く（両端は残す）。残すフレームの bool。"""
    w = np.asarray(weights, np.float64)
    n = len(w)
    keep = np.zeros(n, bool)
    if n <= 2 or tol <= 0:
        keep[:] = True
        return keep
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (np.arange(a + 1, b) - a) / float(b - a)
        err = np.abs(w[a] + (w[b] - w[a]) * t - w[a + 1:b]) / tol
        i = int(np.argmax(err))
        if err[i] > 1.0:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def build_lipsync(analysis, cfg, fps, num_frames, available_morphs=None):
    """音素認識の結果から、モーフのキー [{name, shape, frames, weights}] と情報・警告。"""
    times = np.asarray(analysis['phone_times'], np.float64).reshape(-1)
    phones = np.asarray(analysis['phones'], dtype=str).reshape(-1)
    level = np.asarray(analysis['level_db'], np.float64).reshape(-1)
    rate = float(analysis['level_rate'])
    shapes = [phone_shape(p) for p in phones]
    warns = []
    voiced, levels = voice_activity(level, rate, cfg)
    if not voiced.any():
        warns.append('声が見つからなかったので、口パクは作りませんでした')
    label = shape_timeline(times, shapes, voiced, rate, cfg)
    weights = to_frames(shape_weights(label, level, levels['reference'], rate, cfg), rate, fps, num_frames)
    names = dict(cfg.morphs)
    available = None if available_morphs is None else set(available_morphs)
    frames = np.arange(num_frames)
    tracks = []
    for i, shape in enumerate(SHAPES):
        name = names.get(shape) or ''
        if not name:
            continue
        if available is not None and name not in available:
            warns.append(f'モデルにモーフ「{name}」が無いので、{KANA[shape]} の口の形は動かしません')
            continue
        keep = thin_weights(weights[i], float(cfg.thin_tolerance))
        tracks.append(dict(name=name, shape=shape, frames=frames[keep].tolist(),
                           weights=np.round(weights[i][keep], 4).tolist()))
    info = dict(phones=len(phones), voiced_sec=round(float(voiced.sum() / rate), 2),
                vowels={KANA[s]: sum(1 for x in shapes if x == s) for s in SHAPES})
    return tracks, weights, info, warns
