"""ステージ9l: 脚（足・ひざ・足首）の回転のキー（膝の向き）。

MMD の脚は足ＩＫで決まり、膝がどちらを向くか（内股・がに股）は IK の解き方しだいになる。MMD の IK は
リンクボーン（足・ひざ）のその時の回転から解き始めるので、すでに足ＩＫに届いていて膝が SMPL の膝の向きを
向いた 足・ひざ・足首 の回転をキーにすれば、膝の向きがそのまま残る。股関節から足ＩＫのターゲットまでを
モデルの脚の寸法で 2 ボーンの IK として解く（ひざはローカルの X 軸まわりに後ろへ曲げる向きだけ）。
"""
from dataclasses import dataclass, field

import numpy as np

from ..body.smpl import LEG_JOINTS
from ..core import filters, quat
from ..rig.skeleton import SIDES
from .center import ReachGeometry

LEG_BONES = tuple(s + n for s in SIDES for n in ('足', 'ひざ', '足首'))
X_AXIS = np.array([1.0, 0.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])
MIN_BEND_DEG = 0.5
_EPS = 1e-9


@dataclass
class LegResult:
    local: dict                  # ボーン名 → ローカル回転 (T, 4)
    glob: dict                   # ボーン名 → 大域回転 (T, 3, 3)
    knee: np.ndarray             # (T, 2, 3)
    bend_deg: np.ndarray         # (T, 2)
    knee_out_deg: np.ndarray     # (T, 2)
    unreached: np.ndarray        # (T, 2)
    info: dict = field(default_factory=dict)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def _perpendicular(v, axis):
    return v - np.sum(v * axis, -1, keepdims=True) * axis


def _frames(up, lateral):
    ey = _unit(up)
    ex = _unit(_perpendicular(np.broadcast_to(lateral, ey.shape), ey))
    return np.stack([ex, ey, np.cross(ex, ey)], axis=-1)


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(np.shape(theta) + (3, 3))
    R[..., 0, 0] = 1.0
    R[..., 1, 1], R[..., 1, 2] = c, -s
    R[..., 2, 1], R[..., 2, 2] = s, c
    return R


def knee_poles(kin, fps=None, one_euro=None):
    """(T, 2, 3) SMPL の膝の向き（股関節 → 足首 の線に垂直な単位ベクトル）。

    SMPL の膝の曲げの軸（股関節の大域回転で回した X 軸）を股関節 → 足首の線に垂直にして、その軸で後ろへ曲げたときに
    膝が出る向き。骨盤の座標系でゼロ位相の One Euro をかけてならす（太もものひねりは単眼推定で最も決まりにくい）。
    """
    J = np.asarray(kin.joints, np.float64)
    R = np.asarray(kin.glob_rot, np.float64)
    out = np.empty((len(J), 2, 3))
    for side, (hip, _, ankle, _) in enumerate(LEG_JOINTS):
        u = _unit(J[:, ankle] - J[:, hip])
        axis = _unit(_perpendicular(R[:, hip] @ X_AXIS, u))
        out[:, side] = np.cross(u, axis)
    if fps is None or one_euro is None or float(one_euro.min_cutoff) <= 0:
        return out
    pelvis = R[:, 0]
    local = np.einsum('tba,tsb->tsa', pelvis, out)
    local = filters.one_euro(local, fps, float(one_euro.min_cutoff), float(one_euro.beta), zero_phase=True,
                             vector_axis=-1)
    return _unit(np.einsum('tab,tsb->tsa', pelvis, local))


def _global_of(skel, local, name, cache):
    if name not in cache:
        parent = skel.nearest_ancestor(name, set(local))
        R = quat.to_matrix(local[name])
        cache[name] = R if parent is None else _global_of(skel, local, parent, cache) @ R
    return cache[name]


def solve_legs(skel, center_delta, ik, local, poles, cfg):
    """センター・足ＩＫ・ローカル回転 local（下半身を含む）から、足・ひざ（・足首）の回転。cfg: 設定 leg_keys。"""
    center_delta = np.asarray(center_delta, np.float64)
    T = len(center_delta)
    cache = {}
    lower = _global_of(skel, local, '下半身', cache)
    geom = ReachGeometry.from_skeleton(skel)
    hips = geom.hip_positions(center_delta, lower)
    targets = geom.ik[None] + np.asarray(ik.delta, np.float64)
    forward = lower @ Z_AXIS
    min_bend = np.deg2rad(MIN_BEND_DEG)
    out_local, out_glob = {}, {}
    knee_pos = np.empty((T, 2, 3))
    bend_deg = np.empty((T, 2))
    knee_out = np.empty((T, 2))
    unreached = np.zeros((T, 2), bool)
    for side, s in enumerate(SIDES):
        names = [s + n for n in ('足', 'ひざ', '足首')]
        v1 = skel.internal(names[1]) - skel.internal(names[0])
        v2 = skel.internal(names[2]) - skel.internal(names[1])
        # ひざを X 軸まわりに θ 回したときの 足 → 足首 の距離: |v1|² + |v2|² + 2 v1x v2x + 2ρ cos(θ − ψ)
        a = v1[1] * v2[1] + v1[2] * v2[2]
        b = v1[2] * v2[1] - v1[1] * v2[2]
        rho, psi = np.hypot(a, b), np.arctan2(b, a)
        d = targets[:, side] - hips[:, side]
        dist = np.linalg.norm(d, axis=-1)
        c = (dist ** 2 - v1 @ v1 - v2 @ v2 - 2.0 * v1[0] * v2[0]) / max(2.0 * rho, _EPS)
        theta = psi + np.arccos(np.clip(c, -1.0, 1.0))
        unreached[:, side] = (np.abs(c) > 1.0) | (theta < min_bend)
        theta = np.maximum(theta, min_bend)
        u = _unit(d)
        p = _perpendicular(poles[:, side], u)
        weak = np.linalg.norm(p, axis=-1) < 1e-3
        p[weak] = _perpendicular(forward[weak], u[weak])
        p = _unit(p)
        n = np.cross(p, u)
        w = v1[None] + np.einsum('tab,b->ta', _rot_x(theta), v2)
        G1 = _frames(u, n) @ np.swapaxes(_frames(w, X_AXIS), -1, -2)
        G2 = G1 @ _rot_x(theta)
        out_glob[names[0]], out_glob[names[1]] = G1, G2
        knee_pos[:, side] = hips[:, side] + G1 @ v1
        bend_deg[:, side] = np.rad2deg(theta)
        outward = np.arctan2(np.cross(forward, p)[:, 1], np.sum(forward * p, -1))
        knee_out[:, side] = np.rad2deg(outward if side == 0 else -outward)
        if cfg.ankle:
            out_glob[names[2]] = quat.to_matrix(np.asarray(ik.rotation, np.float64)[:, side])
    keyed = set(local) | set(out_glob)
    for name, G in out_glob.items():
        parent = skel.nearest_ancestor(name, keyed)
        if parent is None:
            Gp = np.eye(3)
        elif parent in out_glob:
            Gp = out_glob[parent]
        else:
            Gp = _global_of(skel, local, parent, cache)
        out_local[name] = quat.make_continuous(quat.from_matrix(np.swapaxes(Gp, -1, -2) @ G))
    info = dict(bones=list(out_local), unreached_frames=unreached.sum(0).tolist(),
                knee_out_median_deg=np.median(knee_out, axis=0).round(1).tolist())
    return LegResult(out_local, out_glob, knee_pos, bend_deg, knee_out, unreached, info)
