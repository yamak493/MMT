"""ボーンのキー列（全フレーム）を作る。値は MMD 座標（VMD に書く値そのもの）。

位置 (x, y, z) → (x, y, −z)、クォータニオン (x, y, z, w) → (−x, −y, z, w)（内部座標は右手系、MMD は左手系）。
脚（足・ひざ・足首）の回転は、出力の種類ごとのセンター・足ＩＫに届くように解き直し、ひねりは捩りボーンへ移す。
"""
from dataclasses import dataclass

import numpy as np

from ..core import quat
from ..rig.skeleton import SIDES
from .legs import solve_legs
from .twist import split_twist

POSITION_TO_MMD = np.array([1.0, 1.0, -1.0])
QUAT_TO_MMD = np.array([-1.0, -1.0, 1.0, 1.0])


@dataclass
class BoneTrack:
    name: str
    positions: np.ndarray    # (T, 3) MMD 座標
    rotations: np.ndarray    # (T, 4) MMD 座標（x, y, z, w）
    movable: bool = False    # 位置のキーを持つボーン（センター・グルーブ・足ＩＫ）


def to_mmd_position(p):
    return np.asarray(p, np.float64) * POSITION_TO_MMD


def to_mmd_quat(q):
    return np.asarray(q, np.float64) * QUAT_TO_MMD


def leg_rotations(skel, center_delta, ik, local, poles, cfg):
    if poles is None or center_delta is None or ik is None or not cfg.leg_keys.enabled:
        return None
    return solve_legs(skel, center_delta, ik, local, poles, cfg.leg_keys)


def build_tracks(skel, center_delta, ik, local, cfg, poles=None):
    """VMD のキー列（全フレーム）。center_delta / ik が None なら、センター・グルーブ / 足ＩＫは打たない。"""
    legs = leg_rotations(skel, center_delta, ik, local, poles, cfg)
    if legs is not None:
        local = dict(local, **legs.local)
    local = split_twist(skel, local, cfg.twist).local
    T = len(next(iter(local.values())))
    identity = np.tile(quat.IDENTITY, (T, 1))
    zeros = np.zeros((T, 3))
    tracks = []
    if center_delta is not None:
        if skel.has('グルーブ'):
            tracks.append(BoneTrack('センター', to_mmd_position(center_delta * [1.0, 0.0, 1.0]), identity.copy(), True))
            tracks.append(BoneTrack('グルーブ', to_mmd_position(center_delta * [0.0, 1.0, 0.0]), identity.copy(), True))
        else:
            tracks.append(BoneTrack('センター', to_mmd_position(center_delta), identity.copy(), True))
    for name, q in local.items():
        tracks.append(BoneTrack(name, zeros.copy(), to_mmd_quat(quat.make_continuous(q))))
    if ik is not None:
        for side, s in enumerate(SIDES):
            tracks.append(BoneTrack(s + '足ＩＫ', to_mmd_position(ik.delta[:, side]),
                                    to_mmd_quat(quat.make_continuous(ik.rotation[:, side])), True))
    return tracks
