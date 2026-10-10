"""ステージ8: センターの安定化と「届く高さ」へのクランプ。

水平移動はセンター、上下移動はグルーブに分けて出力する。脚が届かない問題は、センターを下げる向きの補正だけで
解決する。届く高さは体を支えている脚（接地している脚と、足裏が床の近くにある脚）だけで判定する（宙にある
遊脚に届かせるために体全体を下げない）。ジャンプの滞空（ステージ6a）の上下は、平滑化で削らない。
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..core import filters
from ..rig.skeleton import SIDES


@dataclass
class ReachGeometry:
    """届く高さの判定に使う MMD モデルの寸法（内部座標・MMD 単位）。"""
    lower: np.ndarray        # (3,) 下半身の初期位置
    hips: np.ndarray         # (2, 3) 左右の足（股関節）の初期位置
    ik: np.ndarray           # (2, 3) 左右の足ＩＫの初期位置
    leg_length: np.ndarray   # (2,)

    @classmethod
    def from_skeleton(cls, skel):
        return cls(skel.internal('下半身'), np.stack([skel.internal(s + '足') for s in SIDES]),
                   np.stack([skel.internal(s + '足ＩＫ') for s in SIDES]),
                   np.array([skel.leg_length(0), skel.leg_length(1)]))

    def hip_positions(self, center_delta, lower_rot):
        rel = self.hips - self.lower
        return self.lower + center_delta[:, None] + np.einsum('tab,sb->tsa', lower_rot, rel)

    def distances(self, center_delta, lower_rot, ik_delta):
        hip = self.hip_positions(center_delta, lower_rot)
        return np.linalg.norm(hip - (self.ik + ik_delta), axis=-1)

    def overextended(self, center_delta, lower_rot, ik_delta, ratio, tol=1e-6, legs=None):
        d = self.distances(center_delta, lower_rot, ik_delta)
        over = d > ratio * self.leg_length + tol
        if legs is not None:
            over &= np.asarray(legs, bool)
        return over.any(axis=1)


def supporting_legs(flags, sole_heights, near, margin, flight=None):
    """(T, 2) bool 届く高さの判定に使う脚（接地とその前後 margin フレーム・足裏が near より低い脚。滞空は除く）。"""
    flags = np.asarray(flags, bool)
    if margin > 0 and len(flags):
        flags = maximum_filter1d(flags.astype(np.uint8), 2 * int(margin) + 1, axis=0,
                                 mode='constant').astype(bool)
    legs = flags | (np.asarray(sole_heights, np.float64) < float(near))
    if flight is not None:
        legs &= ~np.asarray(flight, bool)[:, None]
    return legs


def split_flight(y, flight):
    """上下の位置 y を、滞空の区間で「地上の動き」（前後をつなぐ直線）と「跳んだ高さ」（放物線）に分ける。"""
    y = np.asarray(y, np.float64)
    ground = y.copy()
    jump = np.zeros_like(y)
    if flight is None:
        return ground, jump
    T = len(y)
    for s, e in filters.runs(flight):
        a, b = s - 1, e + 1
        if a < 0 or b >= T:
            continue
        tau = np.arange(b - a + 1, dtype=np.float64)
        line = y[a] + (y[b] - y[a]) * tau / (b - a)
        shape = tau * (b - a - tau)
        c = max(0.0, float(shape @ (y[a:b + 1] - line)) / float(shape @ shape))
        ground[a:b + 1] = line
        jump[a:b + 1] = c * shape
    return ground, jump


@dataclass
class CenterResult:
    delta: np.ndarray            # (T, 3) センター＋グルーブの差分（内部座標）
    smoothed: np.ndarray         # (T, 3) クランプ前の差分
    raw: np.ndarray              # (T, 3) 処理前の差分
    correction_raw: np.ndarray
    correction: np.ndarray
    exceed_before: int
    exceed_after: int
    mode: str
    legs: np.ndarray = None

    def with_values(self, **kw):
        return replace(self, **kw)


def _lowpass_axes(out, p, fps, cfg):
    """cfg.lowpass_hz の軸（> 0）は、One Euro の代わりにゼロ位相の Butterworth でならす。"""
    out = np.array(out, np.float64, copy=True)
    for axis, name in enumerate('xyz'):
        hz = float(cfg.lowpass_hz.get(name, 0.0) or 0.0)
        if hz > 0:
            out[:, axis] = filters.lowpass(p[:, axis], fps, hz, int(cfg.lowpass_order))
    return out


def _axis_one_euro(p, fps, unit, cfg, depth_reconstructed):
    ax = cfg.axis_one_euro
    z = ax.z if depth_reconstructed else ax.z_raw
    mc = np.array([ax.x.min_cutoff, ax.y.min_cutoff, z.min_cutoff])
    beta = np.array([ax.x.beta, ax.y.beta, z.beta])
    out = filters.one_euro(p / unit, fps, mc, beta, cfg.d_cutoff, cfg.zero_phase) * unit
    return _lowpass_axes(out, p, fps, cfg)


def _hermite(p0, p1, v0, v1, n):
    s = np.arange(1, n + 1) / (n + 1.0)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    L = n + 1.0
    return h00[:, None] * p0 + h10[:, None] * v0 * L + h01[:, None] * p1 + h11[:, None] * v1 * L


def pelvis_from_contacts(pelvis, ankles, ik_target, contact, fps, unit, cfg, flight=None, keep_jump=True):
    """モード B: 接地中は「ロックした足ＩＫの位置 + 姿勢から求めた足首 → 骨盤」で骨盤を求める。"""
    T = len(pelvis)
    acc = np.zeros((T, 3))
    cnt = np.zeros(T)
    for foot in range(2):
        m = contact.flags[:, foot]
        acc[m] += ik_target[m, foot] + (pelvis[m] - ankles[m, foot])
        cnt[m] += 1
    known = cnt > 0
    if not known.any():
        return pelvis.copy()
    est = np.where(known[:, None], acc / np.maximum(cnt, 1)[:, None], np.nan)
    offset = est - pelvis
    vel = np.gradient(pelvis, axis=0)
    for s, e in filters.runs(~known):
        n = e - s + 1
        if s == 0 or e == T - 1:
            ref = e + 1 if s == 0 else s - 1
            est[s:e + 1] = pelvis[s:e + 1] + offset[ref]
            continue
        a, b = s - 1, e + 1
        est[s:e + 1, [0, 2]] = _hermite(est[a, [0, 2]], est[b, [0, 2]], vel[a, [0, 2]], vel[b, [0, 2]], n)
        w = np.arange(1, n + 1) / (n + 1.0)
        est[s:e + 1, 1] = pelvis[s:e + 1, 1] + (1 - w) * offset[a, 1] + w * offset[b, 1]
    fo = cfg.mode_b.final_one_euro
    est[:, 1], jump = split_flight(est[:, 1], flight)
    out = filters.one_euro(est / unit, fps, fo.min_cutoff, fo.beta, cfg.d_cutoff, cfg.zero_phase) * unit
    out = _lowpass_axes(out, est, fps, cfg)
    if keep_jump:
        out[:, 1] += jump
    return out


def reach_correction(geom, center_delta, lower_rot, ik_delta, ratio, max_drop, legs=None):
    """各フレームで、脚が ratio × 脚長 以内に届く最大の骨盤の高さへの補正量（≤ 0）。"""
    hip = geom.hip_positions(center_delta, lower_rot)
    v = hip - (geom.ik + ik_delta)
    limit = ratio * geom.leg_length
    rhs = limit ** 2 - v[..., 0] ** 2 - v[..., 2] ** 2
    allowed = np.where(rhs >= 0, -v[..., 1] + np.sqrt(np.maximum(rhs, 0)), -v[..., 1])
    need = np.minimum(0.0, allowed)
    if legs is not None:
        need = np.where(np.asarray(legs, bool), need, 0.0)
    return np.maximum(need.min(axis=1), -max_drop)


def apply_reach_clamp(geom, center_delta, lower_rot, ik_delta, cfg, unit, legs=None):
    """届く高さへのクランプ（下げる向きの補正だけ）。戻り値 (差分, 補正量, 掛けた補正量, 超過 前, 後)。"""
    ratio = float(cfg.reach_ratio)
    corr_raw = reach_correction(geom, center_delta, lower_rot, ik_delta, ratio,
                                float(cfg.reach_max_drop_m) * unit, legs)
    # 移動最小値で谷を広げてからガウシアン（カーネルを窓の半分で打ち切り、どのフレームでも必要量以上に下げる）
    window = int(cfg.reach_min_window)
    corr = filters.gaussian_time(filters.moving_min(corr_raw, window), float(cfg.reach_gaussian_sigma),
                                 radius=window // 2)
    corr = np.minimum(corr, 0.0)
    delta = np.array(center_delta, np.float64, copy=True)
    delta[:, 1] += corr
    before = int(geom.overextended(center_delta, lower_rot, ik_delta, ratio, legs=legs).sum())
    after = int(geom.overextended(delta, lower_rot, ik_delta, ratio, legs=legs).sum())
    return delta, corr_raw, corr, before, after


def stabilize_center(pelvis_raw, pelvis, pelvis_rest, ankles, ik, contact, geom, lower_rot, fps, unit, cfg,
                     depth_reconstructed=True, legs=None, flight=None, keep_jump=True):
    """ステージ8。pelvis_raw: 処理前の骨盤 / pelvis: 処理後の骨盤（どちらも (T, 3)）。"""
    mode = str(cfg.mode).upper()
    if mode == 'A':
        grounded = np.array(pelvis, np.float64, copy=True)
        grounded[:, 1], jump = split_flight(grounded[:, 1], flight)
        smooth = _axis_one_euro(grounded, fps, unit, cfg, depth_reconstructed)
        if keep_jump:
            smooth[:, 1] += jump
    elif mode == 'B':
        smooth = pelvis_from_contacts(pelvis, ankles, ik.target, contact, fps, unit, cfg, flight, keep_jump)
    else:
        raise ValueError(f'center.mode は A か B です: {cfg.mode}')
    smooth_delta = smooth - pelvis_rest
    delta, corr_raw, corr, before, after = apply_reach_clamp(geom, smooth_delta, lower_rot, ik.delta, cfg,
                                                             unit, legs)
    return CenterResult(delta, smooth_delta, pelvis_raw - pelvis_rest, corr_raw, corr, before, after, mode, legs)
