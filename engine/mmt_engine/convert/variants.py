"""出力の種類: フル / フル [接地優先] / フル [移動なし] / 上半身のみ。

* full        センター・グルーブ・足ＩＫ・全身の回転（変換の結果そのまま）
* locked      ジャンプ・片足上げなどで足が床から浮いた所も、足を床に着ける。回転とセンターの水平移動はフルと同じで、
              浮いていた所から離れたフレームはフルと同じ値になる（タイムラインでフルと切り替えて使える）
* no_move     体の水平移動を除く。足は着いている間その場に固定し、足跡をその間のセンターの平均位置だけずらす
* upper_body  上半身から先の回転だけ（体全体の向きを除く）
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..body.smpl import ANKLES
from ..core import filters, quat
from .center import apply_reach_clamp, stabilize_center, supporting_legs
from .contact import ContactResult, clean_flags, hysteresis
from .foot_ik import build_foot_ik, mmd_sole_heights, mmd_sole_points, sole_floor_levels
from .ground import support_height
from .tracks import build_tracks

VARIANTS = ('full', 'locked', 'no_move', 'upper_body')
LOWER_BODY_BONE = '下半身'
STRIDE_M = 0.1
STILL_M = 0.03
HOVER_M = 0.01
SAME_POS_M = 0.005
SAME_ROT_DEG = 0.5


# ================================================================ フル [接地優先]

@dataclass
class LockedMotion:
    center: object
    foot_ik: object
    contact: object
    info: dict


def jump_height(result):
    T = len(result.contact.flags)
    flight = result.ground.flight
    if flight is None or not np.any(flight):
        return np.zeros(T)
    height = support_height(result.kin, result.contact)
    return np.where(np.asarray(flight, bool), np.maximum(height, 0.0), 0.0)


def ground_feet(skel, ik, both_feet):
    """足ＩＫを上下に動かして、モデルの足裏を床に合わせる。"""
    sole = mmd_sole_heights(skel, ik)
    if both_feet:
        shift = sole
    else:
        shift = np.minimum(np.maximum(sole.min(1), 0.0)[:, None], np.maximum(sole, 0.0))
    delta = np.array(ik.delta, copy=True)
    delta[..., 1] -= shift
    return ik.with_values(delta=delta, target=np.asarray(ik.target) + (delta - ik.delta))


def foot_drops(clearance, jump, both_feet=True):
    clearance = np.asarray(clearance, np.float64)
    if both_feet:
        return np.maximum(clearance, 0.0)
    drop = np.maximum(np.maximum(clearance.min(1), 0.0), jump)
    return np.minimum(drop[:, None], np.maximum(clearance, 0.0))


def lock_flags(contact, ankles, sole, lowered, valid, cfg_contact, still, unit):
    """(T, 2) 足を床に固定するフレーム。フルの接地に、足を下ろしたフレームで足した接地を加える。"""
    full = np.asarray(contact.flags, bool)
    T = len(full)
    speeds = np.asarray(contact.speeds, np.float64)
    near = np.asarray(lowered, bool)[:, None] & (sole < float(cfg_contact.exit_height_m) * unit)
    enter = (near & (sole < float(cfg_contact.enter_height_m) * unit)
             & (speeds < float(cfg_contact.enter_speed_m_per_s) * unit).any(-1))
    if valid is not None:
        enter &= np.asarray(valid, bool)[:, None]
    stay = near & (speeds < float(cfg_contact.exit_speed_m_per_s) * unit).any(-1)
    min_len = int(cfg_contact.min_contact_frames)
    flags = full.copy()
    for foot in range(2):
        xz = np.asarray(ankles, np.float64)[:, foot][:, [0, 2]]
        f = flags[:, foot]
        for s, e in filters.runs(full[:, foot]):
            lock = np.median(xz[s:e + 1], axis=0)
            for t, step in ((e + 1, 1), (s - 1, -1)):
                while 0 <= t < T and near[t, foot] and not full[t, foot] and np.linalg.norm(xz[t] - lock) < still:
                    f[t] = True
                    t += step
        for s, e in filters.runs(hysteresis(enter[:, foot], stay[:, foot]) & ~f):
            touching = (s > 0 and f[s - 1]) or (e < T - 1 and f[e + 1])
            still_run = (np.linalg.norm(xz[s:e + 1] - np.median(xz[s:e + 1], axis=0), axis=1) < still).all()
            if e - s + 1 >= min_len and not touching and still_run:
                f[s:e + 1] = True
        flags[:, foot] = clean_flags(f, int(cfg_contact.fill_gap_frames), 1)
    return flags


def locked_motion(result):
    """フル [接地優先] の動き。"""
    cfg, k, fps = result.config, result.scale, result.fps
    kin, contact, ik_full = result.kin, result.contact, result.foot_ik
    T = len(contact.flags)
    skel = result.skeleton
    jump = filters.gaussian_time(jump_height(result), float(cfg.locked.sigma_frames))
    clearance = mmd_sole_heights(skel, ik_full)
    foot_drop = foot_drops(clearance, jump, bool(cfg.locked.both_feet))
    down = np.zeros((T, 2, 3))
    down[..., 1] = foot_drop
    lowered = maximum_filter1d(foot_drop.max(1) > float(cfg.contact.enter_height_m) * k,
                               2 * int(cfg.foot_ik.blend_frames) + 1, mode='constant')
    flags = lock_flags(contact, ik_full.raw, clearance - foot_drop, lowered, result.valid, cfg.contact,
                       float(cfg.locked.still_m) * k, k)
    locked_contact = ContactResult(flags, [filters.runs(flags[:, f]) for f in range(2)],
                                   np.asarray(contact.heights) - foot_drop[..., None], contact.speeds)
    rt = result.retargeter
    ik = build_foot_ik(ik_full.raw - down, result.ankle_rest, rt.foot_ik_quats(kin.glob_rot), locked_contact, fps, k,
                       cfg.foot_ik, swing=ik_full.swing - down, anchor=contact.flags,
                       sole_points=mmd_sole_points(skel), sole_floor=sole_floor_levels(skel, result.ankle_rest))
    ik = ground_feet(skel, ik, bool(cfg.locked.both_feet))
    body = np.zeros((T, 3))
    body[:, 1] = jump
    lower_rot = rt.global_matrix(LOWER_BODY_BONE, kin.glob_rot)
    support = supporting_legs(locked_contact.flags, mmd_sole_heights(skel, ik),
                              float(cfg.contact.exit_height_m) * k, int(cfg.foot_ik.blend_frames))
    center = stabilize_center(result.kin_raw.root_pos, kin.root_pos - body, result.pelvis_rest,
                              kin.joints[:, ANKLES] - body[:, None], ik, locked_contact, result.reach_geometry,
                              lower_rot, fps, k, cfg.center, legs=support, flight=result.ground.flight,
                              keep_jump=False)
    smoothed = np.array(center.smoothed, copy=True)
    smoothed[:, [0, 2]] = result.center.smoothed[:, [0, 2]]
    delta, corr_raw, corr, exceed_before, exceed_after = apply_reach_clamp(
        result.reach_geometry, smoothed, lower_rot, ik.delta, cfg.center, k, support)
    center = center.with_values(delta=delta, smoothed=smoothed, correction_raw=corr_raw, correction=corr,
                                exceed_before=exceed_before, exceed_after=exceed_after)
    rot_diff = np.rad2deg(quat.angle_between(ik.rotation, ik_full.rotation)).max(1)
    differ = ((np.abs(center.delta - result.center.delta).max(1) > SAME_POS_M * k)
              | (np.abs(ik.delta - ik_full.delta).max((1, 2)) > SAME_POS_M * k)
              | (rot_diff > SAME_ROT_DEG))
    info = dict(differ_frames=[list(r) for r in filters.runs(differ)],
                added_lock_frames=[int((flags[:, f] & ~contact.flags[:, f]).sum()) for f in range(2)],
                exceed_after=int(exceed_after))
    return LockedMotion(center, ik, locked_contact, info)


# ================================================================ フル [移動なし]

def footprints(segments, target, stride, still):
    """足跡の区間（接地区間を足が止まっている範囲まで広げ、近いものはまとめる）。"""
    horizontal = np.array([1.0, 0.0, 1.0])
    groups = []
    for s, e in segments:
        if groups and np.linalg.norm((target[s] - target[groups[-1][1]]) * horizontal) < stride:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    out = []
    for i, (s, e) in enumerate(groups):
        lo = out[-1][1] + 1 if out else 0
        hi = groups[i + 1][0] - 1 if i + 1 < len(groups) else len(target) - 1
        start, end = target[s], target[e]
        while s > lo and np.linalg.norm(target[s - 1] - start) < still:
            s -= 1
        while e < hi and np.linalg.norm(target[e + 1] - end) < still:
            e += 1
        out.append((s, e))
    return out


def footprint_shift(travel, segments, target, stride, still):
    """フル [移動なし] で 1 つの足ＩＫから引く水平の量 (T, 3)。"""
    travel = np.asarray(travel, np.float64)
    target = np.asarray(target, np.float64)
    T = len(travel)
    if not segments:
        return travel.copy()
    shift = np.empty_like(travel)
    horizontal = np.array([1.0, 0.0, 1.0])
    anchors = [(s, e, travel[s:e + 1].mean(0)) for s, e in footprints(segments, target, stride, still)]
    first, last = anchors[0][0], anchors[-1][1]
    if first > 0:
        anchors.insert(0, (0, 0, anchors[0][2] - (target[first] - target[0]) * horizontal))
    if last < T - 1:
        anchors.append((T - 1, T - 1, anchors[-1][2] + (target[T - 1] - target[last]) * horizontal))
    for s, e, ref in anchors:
        shift[s:e + 1] = ref
    for (_, e0, r0), (s1, _, r1) in zip(anchors[:-1], anchors[1:]):
        n = s1 - e0 - 1
        if n <= 0:
            continue
        w_time = filters.smoothstep(np.arange(1, n + 1) / (n + 1.0))
        step = (target[s1] - target[e0]) * horizontal
        length = float(np.linalg.norm(step))
        w = w_time
        if length > 0:
            progress = np.clip((target[e0 + 1:s1] - target[e0]) @ step / length ** 2, 0.0, 1.0)
            alpha = min(1.0, length / stride)
            w = alpha * progress + (1.0 - alpha) * w_time
        shift[e0 + 1:s1] = r0 + (r1 - r0) * w[:, None]
    return shift


def no_move_motion(result):
    """フル [移動なし] の (センターの差分, 足ＩＫ, 情報)。回転はフルと同じ。"""
    center, ik, contact, k = result.center, result.foot_ik, result.contact, result.scale
    T = len(center.delta)
    travel = np.zeros((T, 3))
    travel[:, [0, 2]] = center.delta[:, [0, 2]]
    shift = np.stack([footprint_shift(travel, contact.segments[foot], ik.target[:, foot], STRIDE_M * k, STILL_M * k)
                      for foot in range(2)], axis=1)
    ik = ik.with_values(target=ik.target - shift, delta=ik.delta - shift)
    center_delta = np.zeros((T, 3))
    center_delta[:, 1] = center.smoothed[:, 1]
    lower_rot = result.retargeter.global_matrix(LOWER_BODY_BONE, result.kin.glob_rot)
    center_delta, _, corr, before, after = apply_reach_clamp(result.reach_geometry, center_delta, lower_rot, ik.delta,
                                                             result.config.center, k, center.legs)
    return center_delta, ik, dict(exceed_after=int(after))


# ================================================================ 上半身のみ

def heading_matrices(rot):
    """回転 (T, 3, 3) に最も近い鉛直軸まわりの回転（前後・左右の傾きは向きを変えない）。"""
    rot = np.asarray(rot, np.float64)
    theta = np.arctan2(rot[:, 0, 2] - rot[:, 2, 0], rot[:, 0, 0] + rot[:, 2, 2])
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(rot.shape)
    out[:, 0, 0], out[:, 0, 2], out[:, 1, 1] = c, s, 1.0
    out[:, 2, 0], out[:, 2, 2] = -s, c
    return out


def upper_body_quats(result):
    """上半身のみ: ボーン名 → ローカル回転（下半身の鉛直軸まわりの向きを除く）。"""
    rt = result.retargeter
    glob = rt.global_matrices(result.kin.glob_rot)
    heading_inv = np.swapaxes(heading_matrices(glob[LOWER_BODY_BONE]), -1, -2)
    bones = [b for b in rt.bones if b != LOWER_BODY_BONE]
    out = {}
    for name in bones:
        if rt.keyed_parent[name] in bones:
            out[name] = result.local_quats[name]
        else:
            out[name] = quat.make_continuous(quat.from_matrix(heading_inv @ glob[name]))
    return out


# ================================================================ まとめ

def variant_motion(result, kind):
    """種類 kind の (センターの差分, 足ＩＫ, ローカル回転, 接地, 情報)。上半身のみはセンター・足ＩＫ・接地が None。"""
    if kind not in VARIANTS:
        raise ValueError(f'出力の種類が不正です: {kind}')
    if kind == 'full':
        return result.center.delta, result.foot_ik, result.local_quats, result.contact, {}
    if kind == 'upper_body':
        return None, None, upper_body_quats(result), None, {}
    if kind == 'locked':
        lm = locked_motion(result)
        return lm.center.delta, lm.foot_ik, result.local_quats, lm.contact, lm.info
    center_delta, ik, info = no_move_motion(result)
    return center_delta, ik, result.local_quats, result.contact, info


def motion_tracks(result, kind, center_delta, ik, local):
    """variant_motion の動きのキー列（BoneTrack のリスト）。"""
    if kind == 'full':
        return result.tracks
    poles = None if kind == 'upper_body' else result.knee_poles
    return build_tracks(result.skeleton, center_delta, ik, local, result.config, poles)


def variant_tracks(result, kind):
    """種類 kind のボーンのキー列（BoneTrack のリスト）と、その種類の接地（足ピタ）・情報。"""
    center_delta, ik, local, contact, info = variant_motion(result, kind)
    return motion_tracks(result, kind, center_delta, ik, local), contact, info
