"""捩りボーン（腕捩・手捩）へのひねりの振り分け（VMD のキーを作る直前）。

腕・ひじ・手首にひねりを入れると、関節の所でメッシュがまとめてねじれてつぶれる。モデルの多くは上腕・前腕の
途中に 腕捩・手捩 があり、その子の 腕捩1〜3・手捩1〜3（回転付与）がひねりを少しずつ分けるので、ボーンの軸まわりの
回転（ひねり）を swing-twist 分解で取り出して捩りボーンへ移す（大域回転は変えない）。
  * 腕 = 曲げ · ひねり（上腕の軸まわり）→ 腕 に曲げ、腕捩 にひねり
  * 前腕のひねり = ひじ のひねり + 手首 のひねり → 手捩 に入れ、ひじ・手首 には曲げだけを残す
ひねりの角は時間方向に連続につなぎ、捩りボーンに入れるのは ±max_deg まで（残りは元のボーンへ）。
"""
from dataclasses import dataclass, field

import numpy as np

from ..core import filters, quat
from ..rig.skeleton import SIDES

# (捩りボーン, ひねりを取り出すボーン, 子側のボーン, 子側のひねりも集めるか)
TWIST_BONES = (('腕捩', '腕', 'ひじ', False), ('手捩', 'ひじ', '手首', True))
MAX_AXIS_DEG = 5.0


@dataclass
class TwistChain:
    twist: str
    parent: str
    child: str
    axis: np.ndarray
    collect_child: bool


@dataclass
class TwistResult:
    local: dict
    chains: list
    skipped: list
    angle_deg: dict = field(default_factory=dict)
    enabled: bool = True


def twist_chains(skel):
    chains, skipped = [], []
    for s in SIDES:
        for twist, parent, child, collect in TWIST_BONES:
            t, p, c = s + twist, s + parent, s + child
            if not skel.has(t):
                continue
            if not (skel.has(p) and skel.has(c) and t != p and skel.is_descendant(t, p)
                    and skel.is_descendant(c, t)):
                skipped.append(f'{t} が {p} → {c} の間にないので使いません')
                continue
            bone = skel.internal(c) - skel.internal(t)
            axis = skel.axis_internal(t)
            if axis is None or np.linalg.norm(axis) < 1e-9:
                axis = bone
            limb = skel.internal(c) - skel.internal(p)
            if max(_angle_deg(axis, bone), _angle_deg(bone, limb)) > MAX_AXIS_DEG:
                skipped.append(f'{t} の軸・位置が {p} → {c} の線と合わないので使いません')
                continue
            chains.append(TwistChain(t, p, c, axis / np.linalg.norm(axis), collect))
    return chains, skipped


def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)) / (na * nb), 0.0, 1.0))))


def twist_angle(q, axis, min_cos):
    """q (T, 4) の axis まわりのひねりの角 (T,) と、曲げの半分の角の余弦 (T,)。連続につなぐ。"""
    q = quat.normalize(q)
    s, w = q[:, :3] @ axis, q[:, 3]
    cos_half = np.hypot(s, w)
    theta = 2.0 * np.arctan2(s, w)
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    for a, b in filters.runs(cos_half >= min_cos):
        seg = np.unwrap(theta[a:b + 1])
        theta[a:b + 1] = seg - 2.0 * np.pi * np.round(np.median(seg) / (2.0 * np.pi))
    return theta, cos_half


def _about(axis, angle):
    return quat.from_rotvec(np.asarray(angle)[:, None] * axis)


def split_twist(skel, local, cfg):
    """local（ボーン名 → ローカル回転 (T, 4)）のひねりを捩りボーンへ移した TwistResult。"""
    if not cfg.enabled:
        return TwistResult(dict(local), [], [], enabled=False)
    chains, skipped = twist_chains(skel)
    out = dict(local)
    full, zero = np.deg2rad(np.asarray(cfg.fade_swing_deg, np.float64))
    cos_full, cos_zero = np.cos(0.5 * full), np.cos(0.5 * zero)
    limit = np.deg2rad(float(cfg.max_deg))

    def removable(q, axis):
        theta, cos_half = twist_angle(q, axis, cos_zero)
        return theta * filters.smoothstep((cos_half - cos_zero) / (cos_full - cos_zero))

    used = []
    for ch in chains:
        if ch.parent not in out or ch.child not in out:
            skipped.append(f'{ch.parent} か {ch.child} にキーが無いので、{ch.twist} は使いません')
            continue
        own = removable(out[ch.parent], ch.axis)
        if ch.collect_child:
            amount = np.clip(own + removable(out[ch.child], ch.axis), -limit, limit)
        else:
            own = np.clip(own, -limit, limit)
            amount = own
        # 親 · 捩り · 子 の積を変えない: 親から own を除き、捩りに amount を入れ、残り（own − amount）は子へ
        out[ch.parent] = quat.mul(out[ch.parent], quat.conj(_about(ch.axis, own)))
        out[ch.twist] = _about(ch.axis, amount)
        out[ch.child] = quat.mul(_about(ch.axis, own - amount), out[ch.child])
        used.append(ch)
    return TwistResult(out, used, skipped,
                       {ch.twist: np.rad2deg(2.0 * np.arctan2(out[ch.twist][:, :3] @ ch.axis,
                                                              out[ch.twist][:, 3])) for ch in used})
