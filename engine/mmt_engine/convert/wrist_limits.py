"""ステージ9c: 手首の可動域（前腕の回内・回外と、手首の曲げを人の関節の範囲に収める）。

前腕（ひじ）に対する手の向きを、前腕のひねり（ひじ + 手首 の前腕の軸まわりのひねりの和）と手首の曲げ
（掌屈・背屈・橈屈・尺屈）に分けて、可動域（既定は日本整形外科学会の参考可動域）に収める。曲げは 4 つの
1/4 楕円の中に収める。変えるのは 手首 の回転だけ。範囲の外のひねりは、前のフレームに近いほうの端へ戻し
（上限と下限の間を行き来しない）、1 フレームの変化を max_speed_deg_per_s までにする。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..core import quat
from ..core.filters import rate_limit
from ..rig.skeleton import SIDES

FRONT = np.array([0.0, 0.0, 1.0])
JUMP_WEIGHT = 0.5
_EPS = 1e-9


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _wrap(x):
    return (np.asarray(x, np.float64) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class HandAxes:
    forearm: np.ndarray    # 前腕の軸（ひじ → 手首）
    palm: np.ndarray       # 手のひら側の法線
    thumb: np.ndarray      # 親指側
    sign: float            # 前腕の軸まわりの回転 → 回外が + になる符号

    @property
    def flexion(self):
        return np.cross(self.forearm, self.palm)

    @property
    def radial(self):
        return np.cross(self.forearm, self.thumb)


def hand_axes(elbow, wrist, side):
    """初期姿勢のひじ・手首の位置（内部座標）と腕（0 = 左 / 1 = 右）から HandAxes。

    親指は正面（前腕の軸に垂直にしたもの）、手のひらの法線は 左手 = 前腕 × 親指・右手 = 親指 × 前腕。
    """
    a = _unit(np.asarray(wrist, np.float64) - np.asarray(elbow, np.float64))
    thumb = FRONT - (FRONT @ a) * a
    if np.linalg.norm(thumb) < 1e-6:
        thumb = np.array([0.0, 1.0, 0.0]) - a[1] * a
    thumb = _unit(thumb)
    side_sign = -1.0 if side == 0 else 1.0
    palm = _unit(side_sign * np.cross(thumb, a))
    return HandAxes(a, palm, thumb, float(np.sign(np.cross(a, palm) @ thumb)))


def _split_twist(q, axis):
    q = quat.normalize(q)
    theta = _wrap(2.0 * np.arctan2(q[..., :3] @ axis, q[..., 3]))
    swing = quat.mul(q, quat.conj(quat.from_rotvec(theta[..., None] * axis)))
    return theta, swing


def _decompose(elbow_q, wrist_q, ax):
    """(ひじのひねり, 回外, 掌屈, 橈屈) [rad] それぞれ (T,)。"""
    a = ax.forearm
    t_elbow, _ = _split_twist(elbow_q, a)
    t_wrist, swing = _split_twist(wrist_q, a)
    v = quat.rotate(quat.from_rotvec(-t_wrist[:, None] * a), quat.to_rotvec(swing))
    return t_elbow, ax.sign * _wrap(t_elbow + t_wrist), v @ ax.flexion, v @ ax.radial


def wrist_angles(elbow_q, wrist_q, ax):
    """(T, 3) [度] 回外（+）/回内（−）・掌屈（+）/背屈（−）・橈屈（+）/尺屈（−）。"""
    _, sup, flex, rad = _decompose(np.asarray(elbow_q, np.float64), np.asarray(wrist_q, np.float64), ax)
    return np.rad2deg(np.stack([sup, flex, rad], axis=-1))


def _limits(cfg, margin_deg=0.0):
    names = ('pronation_deg', 'supination_deg', 'flexion_deg', 'extension_deg', 'radial_deg', 'ulnar_deg')
    return tuple(np.deg2rad(max(float(getattr(cfg, n)) + float(margin_deg), 1e-3)) for n in names)


def _swing_excess(flex, rad, lim):
    _, _, f_pos, f_neg, r_pos, r_neg = lim
    F = np.where(flex >= 0.0, f_pos, f_neg)
    R = np.where(rad >= 0.0, r_pos, r_neg)
    return np.hypot(flex / F, rad / R)


def within_limits(angles_deg, cfg, margin_deg=0.0):
    lim = _limits(cfg, margin_deg)
    sup, flex, rad = np.deg2rad(np.moveaxis(np.asarray(angles_deg, np.float64), -1, 0))
    tol = 1e-7
    return (sup >= -lim[0] - tol) & (sup <= lim[1] + tol) & (_swing_excess(flex, rad, lim) <= 1.0 + tol)


def _clamp_twist(sup, lo, hi):
    out = np.empty_like(sup)
    prev = None
    for t, x in enumerate(sup):
        if lo <= x <= hi:
            out[t] = prev = x
            continue
        to_hi = (x - hi) % (2.0 * np.pi)
        to_lo = (lo - x) % (2.0 * np.pi)
        if prev is not None:
            to_hi += JUMP_WEIGHT * abs(hi - prev)
            to_lo += JUMP_WEIGHT * abs(lo - prev)
        out[t] = prev = hi if to_hi <= to_lo else lo
    return out


def _changed(sup, flex, rad, sup_c, flex_c, rad_c):
    return (np.abs(_wrap(sup_c - sup)) > _EPS) | (np.abs(flex_c - flex) > _EPS) | (np.abs(rad_c - rad) > _EPS)


def clamp_wrist(elbow_q, wrist_q, ax, cfg, fps):
    """手首のローカル回転 (T, 4) を可動域に収めたものと、直したフレーム (T,)。"""
    elbow_q = np.asarray(elbow_q, np.float64)
    wrist_q = np.asarray(wrist_q, np.float64)
    t_elbow, sup, flex, rad = _decompose(elbow_q, wrist_q, ax)
    lim = _limits(cfg)
    sup_c = _clamp_twist(sup, -lim[0], lim[1])
    scale = 1.0 / np.maximum(_swing_excess(flex, rad, lim), 1.0)
    flex_c, rad_c = flex * scale, rad * scale
    changed = _changed(sup, flex, rad, sup_c, flex_c, rad_c)
    speed = float(cfg.max_speed_deg_per_s)
    if speed > 0 and len(sup) > 1 and changed.any():
        step = np.deg2rad(speed) / fps
        reach = int(np.ceil((lim[0] + lim[1]) / step)) + 1
        near = maximum_filter1d(changed.astype(np.uint8), 2 * reach + 1, mode='constant') > 0
        sup_c = np.where(near, rate_limit(sup_c[:, None], step)[:, 0], sup_c)
        bend = rate_limit(np.stack([flex_c, rad_c], axis=-1), step)
        flex_c, rad_c = np.where(near, bend[:, 0], flex_c), np.where(near, bend[:, 1], rad_c)
        changed = _changed(sup, flex, rad, sup_c, flex_c, rad_c)
    if not changed.any():
        return wrist_q.copy(), changed
    a = ax.forearm
    twist = quat.from_rotvec((ax.sign * sup_c - t_elbow)[:, None] * a)
    bend = quat.from_rotvec(flex_c[:, None] * ax.flexion + rad_c[:, None] * ax.radial)
    out = np.where(changed[:, None], quat.mul(twist, bend), wrist_q)
    return quat.make_continuous(out), changed


@dataclass
class WristLimitResult:
    enabled: bool
    changed: np.ndarray          # (T, 2)
    correction_deg: np.ndarray   # (T, 2)
    info: dict = field(default_factory=dict)


def limit_wrists(skel, local, cfg, fps):
    """local の 手首 を可動域に収めた dict と WristLimitResult。"""
    T = len(next(iter(local.values()))) if local else 0
    changed = np.zeros((T, 2), bool)
    corr = np.zeros((T, 2))
    out = dict(local)
    for side, s in enumerate(SIDES):
        e, w = s + 'ひじ', s + '手首'
        if e not in local or w not in local or not (skel.has(e) and skel.has(w)):
            continue
        ax = hand_axes(skel.internal(e), skel.internal(w), side)
        if cfg.enabled and T:
            out[w], changed[:, side] = clamp_wrist(local[e], local[w], ax, cfg, fps)
            corr[:, side] = np.rad2deg(quat.angle_between(out[w], local[w]))
    info = dict(enabled=bool(cfg.enabled), changed_frames=[int(changed[:, s].sum()) for s in range(2)],
                max_correction_deg=[round(float(corr[:, s].max(initial=0.0)), 2) for s in range(2)])
    return out, WristLimitResult(bool(cfg.enabled), changed, corr, info)
