"""ステージ6c: 前後・左右の傾きの補正（全身の重心が、接地している足の上に来るように体を起こす）。

単眼推定では、体がカメラに向かって前後に傾く角度がほとんど決まらない。左右（画面の横）の傾きも、床の傾きの
推定のずれなどで残ることがある。体の縦のラインを鉛直に直すとお辞儀など本当の傾きまで消えるので、
「全身の重心は、数秒の平均では接地している足の上にある」という条件を使う。
各フレームの重心と支持点のカメラの奥行き方向（次に左右方向）の差を、数秒の窓で最も小さくする補正角を格子の上で
探し、足首から下はそのままにして骨盤から上を傾ける。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..body.smpl import Kinematics
from ..core import filters, quat
from .com import HIPS, body_com, com_weights

KNEES = (4, 5)
FIXED = (7, 8, 10, 11)       # 足首・足先（足の位置と向きは変えない）
UPPER = tuple(j for j in range(24) if j not in HIPS + KNEES + FIXED)
UP = np.array([0.0, 1.0, 0.0])


def horizontal_depth_direction(axis):
    """奥行きの軸を水平面に投影した単位ベクトル（真下を向くカメラでは Z 軸）。"""
    d = np.array([axis[0], 0.0, axis[2]], np.float64)
    n = np.linalg.norm(d)
    return d / n if n > 1e-3 else np.array([0.0, 0.0, 1.0])


def rotation_about(axis, angles):
    return quat.to_matrix(quat.from_rotvec(np.asarray(angles, np.float64)[:, None] * axis))


def rest_offset(rest):
    """初期姿勢（直立）での、足の支持点の中心から全身の重心への水平のずれ (3,)。"""
    sup = np.asarray(rest.points, np.float64).mean(axis=(0, 1, 2))
    off = body_com(rest.joints) - sup
    off[1] = 0.0
    return off


def support(contact_points, flags, band):
    """接地している足の支持点（その足の最下点から band 以内の点）。戻り値 (使う点 (T,2,2), 足ごとの中心 (T,2,3))。"""
    pts = np.asarray(contact_points, np.float64)
    h = pts[..., 1]
    use = (h <= h.min(-1, keepdims=True) + band) & np.asarray(flags, bool)[..., None]
    cnt = np.maximum(use.sum(-1), 1)[..., None]
    center = (pts * use[..., None]).sum(2) / cnt
    return use, center


@dataclass
class LeanResult:
    angle: np.ndarray        # (T, 2) 補正角 [rad]（前後・左右）
    shift: np.ndarray        # (T, 3) 骨盤の平行移動
    directions: np.ndarray   # (2, 3) 傾ける向き（水平）: カメラの奥行き方向・左右方向
    before_deg: np.ndarray   # (T, 2)
    after_deg: np.ndarray    # (T, 2)
    supported: np.ndarray
    enabled: bool
    clamped: tuple = (False, False)

    def rotation(self):
        """(T, 3, 3) 前後に傾けてから左右に傾ける回転。"""
        R = np.broadcast_to(np.eye(3), (len(self.angle), 3, 3))
        for d, a in zip(self.directions, self.angle.T):
            R = np.einsum('tab,tbc->tac', rotation_about(np.cross(UP, d), a), R)
        return R

    def apply(self, kin):
        """骨盤から上を回して骨盤を平行移動する。足首から下は変えない（膝は股関節の移動の半分）。"""
        if not self.enabled:
            return kin
        R = self.rotation()
        G = np.array(kin.glob_rot, copy=True)
        P = np.array(kin.joints, copy=True)
        p0 = kin.joints[:, 0]
        moved = list(UPPER) + list(HIPS)
        P[:, moved] = (p0 + self.shift)[:, None] + np.einsum('tab,tjb->tja', R, kin.joints[:, moved] - p0[:, None])
        for knee, hip in zip(KNEES, HIPS):
            P[:, knee] += 0.5 * (P[:, hip] - kin.joints[:, hip])
        G[:, list(UPPER)] = np.einsum('tab,tjbc->tjac', R, kin.glob_rot[:, list(UPPER)])
        return Kinematics(G, P, kin.contact_points)

    def info(self):
        if not self.enabled:
            return dict(enabled=False)
        deg = np.rad2deg(self.angle)
        return dict(enabled=True, min_deg=round(float(deg[:, 0].min()), 2), max_deg=round(float(deg[:, 0].max()), 2),
                    side_min_deg=round(float(deg[:, 1].min()), 2), side_max_deg=round(float(deg[:, 1].max()), 2),
                    clamped=bool(self.clamped[0]), side_clamped=bool(self.clamped[1]))


def pivot_height(kin, fps):
    """(T,) 骨盤を平行移動する量に使う骨盤の高さ（σ 0.5 秒でならす。1 歩ごとの上下まで追わない）。"""
    return filters.gaussian_time(np.asarray(kin.joints[:, 0, 1], np.float64), 0.5 * fps)


def com_along(kin, d, height):
    """補正角 θ で apply したときの重心の奥行き方向の位置 = c0 + c1 tanθ + c2 cosθ + c3 sinθ の係数 (T, 4)。"""
    w = com_weights()
    P = np.asarray(kin.joints, np.float64)
    p0 = P[:, 0]
    x = P - p0[:, None]
    moved = list(UPPER) + list(HIPS)
    xd, xy = x @ d, x[..., 1]
    w_moved, w_knee = w[moved].sum(), w[list(KNEES)].sum()
    rot_d = xd[:, moved] @ w[moved] + 0.5 * sum(w[k] * xd[:, h] for k, h in zip(KNEES, HIPS))
    rot_y = xy[:, moved] @ w[moved] + 0.5 * sum(w[k] * xy[:, h] for k, h in zip(KNEES, HIPS))
    fixed = list(FIXED) + list(KNEES)
    c0 = (P[:, fixed] @ d) @ w[fixed] + w_moved * (p0 @ d) \
        - 0.5 * sum(w[k] * xd[:, h] for k, h in zip(KNEES, HIPS))
    c1 = (w_moved + 0.5 * w_knee) * np.asarray(height, np.float64)
    return np.stack([c0, c1, rot_d, rot_y], axis=1)


def _evaluate(coef, theta):
    basis = np.stack([np.ones_like(theta), np.tan(theta), np.cos(theta), np.sin(theta)])
    return coef @ basis


def _refine(cost, grid):
    """(T, G) の損失の最小を格子から放物線で補間する。戻り値 (角度 (T,), 端に当たったか (T,))。"""
    idx = np.argmin(cost, axis=1)
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    theta = grid[idx]
    inner = (idx > 0) & (idx < len(grid) - 1)
    t = np.flatnonzero(inner)
    if len(t):
        a, b, c = cost[t, idx[t] - 1], cost[t, idx[t]], cost[t, idx[t] + 1]
        den = a - 2.0 * b + c
        off = np.where(den > 1e-12, 0.5 * (a - c) / np.where(den > 1e-12, den, 1.0), 0.0)
        theta[t] += np.clip(off, -0.5, 0.5) * step
    return theta, ~inner & (len(grid) > 1)


def _fit(kin, d, flags, use, centers, used, rest, height, fps, unit, cfg):
    """向き d (3,) の補正角を求める。戻り値 (角度 (T,), 補正前・後の重心の傾き [deg] (T,), 上限に達したか)。"""
    T = len(kin.joints)
    n_feet = flags.sum(1)
    supported = n_feet > 0
    center = (centers @ d * flags).sum(1) / np.maximum(n_feet, 1)
    along = kin.contact_points @ d
    lo = np.where(use, along, np.inf).min(axis=(1, 2))
    hi = np.where(use, along, -np.inf).max(axis=(1, 2))
    t = np.arange(T)
    center = np.interp(t, t[supported], center[supported])
    fwd = kin.glob_rot[:, 0, :, 2] * [1.0, 0.0, 1.0]
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    left = np.cross(UP, fwd)
    off = rest_offset(rest) * unit
    target = (off[0] * left + off[2] * fwd) @ d
    center, lo, hi = center + target, lo + target, hi + target

    coef = com_along(kin, d, height)
    step = np.deg2rad(0.25)
    limit = np.deg2rad(float(cfg.max_deg))
    grid = np.linspace(-limit, limit, 2 * int(np.ceil(limit / step)) + 1)
    com = _evaluate(coef, grid)
    diff = com - center[:, None]
    double = (n_feet >= 2)[:, None]
    with np.errstate(invalid='ignore'):
        outside = np.maximum(np.maximum(lo[:, None] - com, com - hi[:, None]), 0.0)
    lam = float(cfg.double_support_center_weight)
    cost = np.where(double, outside ** 2 + lam * diff ** 2, diff ** 2)
    cost = np.minimum(cost, (float(cfg.outlier_m) * unit) ** 2) * used[:, None]

    total = cost[used].mean(0)
    sigma = float(cfg.window_sec) * fps
    if sigma > 0:
        local = gaussian_filter1d(cost, sigma, axis=0, mode='constant')
        angle, edge = _refine(local + 0.05 * total[None], grid)
        angle = filters.gaussian_time(angle, 0.5 * fps)
    else:
        a, e = _refine(total[None], grid)
        angle, edge = np.full(T, a[0]), np.full(T, e[0])
    angle = np.clip(angle, -limit, limit)

    com_height = body_com(kin.joints)[:, 1]
    com0 = coef @ np.array([1.0, 0.0, 1.0, 0.0])
    com1 = (coef * np.stack([np.ones(T), np.tan(angle), np.cos(angle), np.sin(angle)], 1)).sum(1)
    before = np.where(used, np.rad2deg(np.arctan2(com0 - center, com_height)), np.nan)
    after = np.where(used, np.rad2deg(np.arctan2(com1 - center, com_height)), np.nan)
    return angle, before, after, bool(edge[used].any())


def estimate_lean(kin, contact, depth_axis, rest, fps, unit, cfg, flight=None, valid=None):
    """ステージ6c。kin: ステージ6a の後（床 y = 0・MMD 単位）/ contact: 接地判定 / flight: 6a の滞空。

    前後（カメラの奥行き方向）の補正角を求めて傾けた後の姿勢で、左右（カメラの左右方向）の補正角を求める。
    """
    T = len(kin.joints)
    d = horizontal_depth_direction(depth_axis)
    directions = np.stack([d, np.cross(UP, d)])
    nan = np.full((T, 2), np.nan)

    flags = np.asarray(contact.flags, bool)
    n_feet = flags.sum(1)
    observed = np.ones(T, bool) if valid is None else np.asarray(valid, bool)
    supported = (n_feet > 0) & observed
    if not cfg.enabled or supported.sum() < max(3, int(round(fps))):
        return LeanResult(np.zeros((T, 2)), np.zeros((T, 3)), directions, nan, nan, supported, False)

    use, centers = support(kin.contact_points, flags, float(cfg.support_band_m) * unit)
    filled = np.zeros(T, bool)
    max_gap = float(cfg.fill_gap_sec) * fps
    for s, e in filters.runs(~supported):
        if 0 < s and e < T - 1 and e - s + 1 <= max_gap:
            filled[s:e + 1] = True
    if flight is not None:
        filled &= ~np.asarray(flight, bool)
    used = (supported | filled) & observed

    height = pivot_height(kin, fps)
    angle, before, after = np.zeros((T, 2)), nan.copy(), nan.copy()
    clamped = [False, False]
    result, current = None, kin
    for i in range(2 if cfg.side_enabled else 1):
        angle[:, i], before[:, i], after[:, i], clamped[i] = _fit(
            current, directions[i], flags, use, centers, used, rest, height, fps, unit, cfg)
        shift = height[:, None] * (np.tan(angle) @ directions)
        result = LeanResult(angle.copy(), shift, directions, before, after, used, True, tuple(clamped))
        current = result.apply(kin)
    return result
