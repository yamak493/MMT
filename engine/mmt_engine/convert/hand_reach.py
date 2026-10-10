"""ステージ9h: 胴に対する手の位置を保つ（体の比率の違いの補正）。

腕は回転のコピーなので、肩幅・腕の長さ・胴の厚みの比が SMPL と MMD で違うと、胴に対する手の位置が変わる
（腰に当てた手が胴に入り込む・胸の前で合わせた両手が交差する）。手首が胴の近くにあるフレームで、SMPL の手首の
位置を胸の座標系で表し、胴の寸法の比で MMD の胴へ移した所へ 2 ボーン IK で手首を置く（手首の向きは変えない）。
胴から遠い手・まっすぐな腕は回転のコピーのまま（間はなめらかに混ぜる）。
"""
from dataclasses import dataclass

import numpy as np

from ..core import filters, quat
from ..rig.skeleton import SIDES
from .body_contacts import _arm_bone, body_capsules, fallback_capsules
from .jitter import turn_towards
from .retarget import bone_positions

ARM_JOINTS = ((13, 16, 18, 20), (14, 17, 19, 21))
CHEST_JOINT = 9
HIP_JOINTS = (1, 2)
SPINE_JOINTS = (3, 6, 9)
SPINE_LINE = (3, 12)
MIN_VERTICES = 30
DEPTH_PERCENTILE = 95.0
STANDARD_HALF_DEPTH = 0.24
_EPS = 1e-9


@dataclass
class TorsoBox:
    """胴の寸法（胸の座標系。MMD の初期姿勢の向き・MMD 単位）。"""
    center_x: float
    hip_y: float
    height: float
    width: float
    center_z: float
    depth: float
    depth_source: str

    def map_to(self, other, u):
        u = np.asarray(u, np.float64)
        out = np.empty_like(u)
        out[..., 0] = other.center_x + (u[..., 0] - self.center_x) * (other.width / self.width)
        out[..., 1] = other.hip_y + (u[..., 1] - self.hip_y) * (other.height / self.height)
        out[..., 2] = other.center_z + (u[..., 2] - self.center_z) * (other.depth / self.depth)
        return out

    def distance(self, u):
        lo = np.array([self.center_x - 0.5 * self.width, self.hip_y, self.center_z - 0.5 * self.depth])
        hi = np.array([self.center_x + 0.5 * self.width, self.hip_y + self.height, self.center_z + 0.5 * self.depth])
        u = np.asarray(u, np.float64)
        return np.linalg.norm(np.maximum(lo - u, 0.0) + np.maximum(u - hi, 0.0), axis=-1)


def _box(shoulders, hips, depth_range, depth_source):
    width = max(abs(float(shoulders[0, 0] - shoulders[1, 0])), _EPS)
    hip_y = float(hips[:, 1].mean())
    height = max(float(shoulders[:, 1].mean()) - hip_y, _EPS)
    back, front = depth_range
    return TorsoBox(float(shoulders[:, 0].mean()), hip_y, height, width, 0.5 * (back + front),
                    max(front - back, _EPS), depth_source)


def smpl_torso(rest_joints, rest_vertices, vertex_weights, correction, unit):
    J = np.asarray(rest_joints, np.float64)

    def to_u(p):
        return (np.asarray(p, np.float64) - J[CHEST_JOINT]) @ correction * unit

    shoulders = to_u(J[[ARM_JOINTS[0][1], ARM_JOINTS[1][1]]])
    hips = to_u(J[list(HIP_JOINTS)])
    source = 'config'
    depth_range = None
    if rest_vertices is not None and vertex_weights is not None:
        owner = np.argmax(np.asarray(vertex_weights), axis=1)
        pts = np.asarray(rest_vertices, np.float64)[np.isin(owner, SPINE_JOINTS)]
        if len(pts) >= MIN_VERTICES:
            z = to_u(pts)[:, 2]
            depth_range = (float(np.percentile(z, 100.0 - DEPTH_PERCENTILE)), float(np.percentile(z, DEPTH_PERCENTILE)))
            source = 'mesh'
    if depth_range is None:
        width = abs(float(shoulders[0, 0] - shoulders[1, 0]))
        cz = float(to_u(J[list(SPINE_LINE)])[:, 2].mean())
        depth_range = (cz - STANDARD_HALF_DEPTH * width, cz + STANDARD_HALF_DEPTH * width)
    return _box(shoulders, hips, depth_range, source)


def _torso_bone(skel, name):
    return skel.is_descendant(name, '上半身') and not _arm_bone(skel, name) and not skel.is_descendant(name, '首')


def mmd_torso(skel, chest, contacts_cfg, unit):
    base = skel.internal(chest)

    def S(name):
        return skel.internal(name) - base

    shoulders = np.stack([S('左腕'), S('右腕')])
    hips = np.stack([S('左足'), S('右足')])
    caps, source = body_capsules(skel, contacts_cfg, unit)
    torso = [c for c in caps if c.weight >= 1.0 and _torso_bone(skel, c.bone)]
    if not torso:
        caps, source = fallback_capsules(skel, unit, contacts_cfg), 'config'
        torso = [c for c in caps if _torso_bone(skel, c.bone)]
    front = max(max(c.a[2], c.b[2]) + c.radius for c in torso) - base[2]
    back = min(min(c.a[2], c.b[2]) - c.radius for c in torso) - base[2]
    return _box(shoulders, hips, (float(back), float(front)), source)


@dataclass
class HandReachResult:
    enabled: bool
    weight: np.ndarray
    info: dict = None


def _angle_between(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, -1))


def reach_wrists(pos, glob, targets, max_deg):
    """2 ボーン IK で手首を targets へ寄せる 腕・ひじ の大域回転（手首の大域回転は変えない）。"""
    S, E, W = pos['S'], pos['E'], pos['W']
    upper, fore = E - S, W - E
    a = np.linalg.norm(upper, axis=-1)
    b = np.linalg.norm(fore, axis=-1)
    limit = np.deg2rad(float(max_deg))
    normal = np.cross(upper, fore)
    n_len = np.linalg.norm(normal, axis=-1)
    unit_normal = normal / np.maximum(n_len, 1e-12)[:, None]
    bend = np.arctan2(n_len, np.sum(upper * fore, -1))
    goal = targets - S
    dist = np.clip(np.linalg.norm(goal, axis=-1), np.abs(a - b) + 1e-6 * (a + b), (a + b) * (1.0 - 1e-6))
    want = np.arccos(np.clip((dist ** 2 - a ** 2 - b ** 2) / np.maximum(2.0 * a * b, _EPS), -1.0, 1.0))
    delta = np.clip(want - bend, -limit, limit)
    delta = np.where(n_len > 1e-9 * np.maximum(a * b, _EPS), delta, 0.0)
    Rb = quat.to_matrix(quat.from_rotvec(unit_normal * delta[:, None]))
    fore2 = np.einsum('tab,tb->ta', Rb, fore)
    turn = turn_towards(upper + fore2, goal, limit)
    return turn @ glob['arm'], turn @ Rb @ glob['elbow']


def keep_hand_positions(skel, rt, glob_rot, local, smpl_rest, rest_vertices, vertex_weights, cfg, contacts_cfg,
                        unit):
    """ステージ9h。local の 腕・ひじ・手首 を直した dict と HandReachResult。"""
    T = len(glob_rot)
    chest = next((name for name, j, _ in rt.specs if j == CHEST_JOINT), None)
    if not cfg.enabled or chest is None or T == 0:
        return local, HandReachResult(False, np.zeros((T, 2)), dict(enabled=False))
    J = np.asarray(smpl_rest, np.float64)
    C = rt.correction[chest]
    src = smpl_torso(J, rest_vertices, vertex_weights, C, unit)
    dst = mmd_torso(skel, chest, contacts_cfg, unit)

    glob = rt.globals_from_local(glob_rot, local)
    names = {s: [s + b for b in ('腕', 'ひじ', '手首')] for s in SIDES}
    pos = bone_positions(skel, glob, [chest] + names['左'] + names['右'], T)
    G_chest_smpl = glob_rot[:, CHEST_JOINT] @ C
    near, far = float(cfg.near_m) * unit, float(cfg.far_m) * unit
    lo, hi = np.deg2rad(np.asarray(cfg.bend_deg, np.float64))

    out = dict(local)
    weight = np.zeros((T, 2))
    for side, s in enumerate(SIDES):
        collar, shoulder, elbow, wrist = ARM_JOINTS[side]
        upper_s = glob_rot[:, shoulder] @ (J[elbow] - J[shoulder])
        fore_s = glob_rot[:, elbow] @ (J[wrist] - J[elbow])
        rel = (glob_rot[:, CHEST_JOINT] @ (J[collar] - J[CHEST_JOINT])
               + glob_rot[:, collar] @ (J[shoulder] - J[collar]) + upper_s + fore_s)
        u_smpl = np.einsum('tba,tb->ta', G_chest_smpl, rel) * unit
        u_mmd = src.map_to(dst, u_smpl)
        mapped = pos[chest] + np.einsum('tab,tb->ta', glob[chest], u_mmd)
        w_near = 1.0 - filters.smoothstep((src.distance(u_smpl) - near) / max(far - near, _EPS))
        w_bend = filters.smoothstep((_angle_between(upper_s, fore_s) - lo) / max(hi - lo, _EPS))
        w = w_near * w_bend
        arm, elb, wr = names[s]
        W = pos[wr]
        target = W + w[:, None] * (mapped - W)
        G_arm, G_elb = reach_wrists(dict(S=pos[arm], E=pos[elb], W=W), dict(arm=glob[arm], elbow=glob[elb]), target,
                                    cfg.max_deg)
        new = {arm: G_arm, elb: G_elb, wr: glob[wr]}

        def parent_global(name):
            p = rt.keyed_parent[name]
            return None if p is None else new.get(p, glob[p])

        active = w > 0.0
        for name in (arm, elb, wr) if active.any() else ():
            Gp = parent_global(name)
            L = new[name] if Gp is None else np.swapaxes(Gp, -1, -2) @ new[name]
            q = np.array(out[name], np.float64, copy=True)
            q[active] = quat.from_matrix(L[active])
            out[name] = quat.make_continuous(q)
        weight[:, side] = w
    info = dict(enabled=True, ratio=dict(width=round(dst.width / src.width, 3), height=round(dst.height / src.height, 3),
                                         depth=round(dst.depth / src.depth, 3)),
                frames=[int(v) for v in (weight > 0.5).sum(0)])
    return out, HandReachResult(True, weight, info)
