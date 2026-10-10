"""ステージ2b: 手首の向きの補正（MediaPipe Hands の 3D の手のランドマークから）。

NLF の手首の回転は、全身を 384×384 に収めた画像から求めたもので、手は十数〜数十 px しか写っていない。
MediaPipe のランドマークモデルは手だけを 224×224 に切り出して 21 点を求めるので、手のひらの向きはこちらの
ほうが直接的に決まる。ROI の回転と視線の向きを戻して 3D の点をカメラ座標に直し、手のひらの座標系から手首の
大域回転の目標を作って、信頼できるフレームだけ NLF の手首の回転へ寄せる。
"""
from dataclasses import dataclass, field

import numpy as np

from ..core import filters, quat
from ..core.filters import rate_limit
from .motion import CAMERA_TO_YUP
from .wrist_limits import hand_axes, within_limits, wrist_angles

WRIST = (20, 21)
ELBOW = (18, 19)
HAND = (22, 23)
LM_WRIST, LM_INDEX, LM_MIDDLE, LM_PINKY = 0, 5, 9, 17


def intrinsics(image_size, fov_deg=55.0):
    """NLF が仮定しているカメラ（長辺の画角 fov_deg）。image_size = (幅, 高さ)。"""
    w, h = image_size
    f = max(w, h) / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def roi_rotation(angle):
    a = np.asarray(angle, np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1] = c, -s, s, c
    R[..., 2, 2] = 1.0
    return R


def ray_rotation(center, image_size, fov_deg=55.0):
    """光軸 (0, 0, 1) を、画像上の点 center (..., 2) を通る視線へ重ねる最小回転 (..., 3, 3)。"""
    K = intrinsics(image_size, fov_deg)
    c = np.asarray(center, np.float64)
    ray = np.stack([(c[..., 0] - K[0, 2]) / K[0, 0], (c[..., 1] - K[1, 2]) / K[1, 1], np.ones(c.shape[:-1])], axis=-1)
    flat = ray.reshape(-1, 3)
    R = np.stack([quat.to_matrix(quat.from_two_vectors([0.0, 0.0, 1.0], r)) for r in flat])
    return R.reshape(c.shape[:-1] + (3, 3))


def world_to_camera(world, roi, image_size, fov_deg=55.0):
    """MediaPipe の 3D の点 (..., 21, 3)（切り出した画像の座標系）→ カメラ座標の向き。"""
    roi = np.asarray(roi, np.float64)
    R = ray_rotation(roi[..., :2], image_size, fov_deg) @ roi_rotation(roi[..., 3])
    return np.einsum('...ab,...kb->...ka', R, np.asarray(world, np.float64))


def frame_from(forward, lateral):
    f = np.asarray(forward, np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
    a = np.asarray(lateral, np.float64)
    a = a - np.sum(a * f, axis=-1, keepdims=True) * f
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    return np.stack([f, a, np.cross(f, a)], axis=-1)


def palm_frame(landmarks):
    lm = np.asarray(landmarks, np.float64)
    return frame_from(lm[..., LM_MIDDLE, :] - lm[..., LM_WRIST, :], lm[..., LM_INDEX, :] - lm[..., LM_PINKY, :])


def rest_palm_frames(rest_joints):
    J = np.asarray(rest_joints, np.float64)
    return np.stack([frame_from(J[HAND[s]] - J[WRIST[s]], [0.0, 0.0, 1.0]) for s in range(2)])


def swing_twist(q, axis):
    q = quat.normalize(q)
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    proj = np.sum(q[..., :3] * axis, axis=-1)
    twist = 2.0 * np.arctan2(proj, q[..., 3])
    twist = (twist + np.pi) % (2.0 * np.pi) - np.pi
    tq = quat.normalize(np.concatenate([proj[..., None] * axis, q[..., 3:]], axis=-1))
    swing = quat.angle_between(quat.mul(q, quat.conj(tq)), quat.IDENTITY)
    return swing, twist


@dataclass
class WristResult:
    weight: np.ndarray
    correction_deg: np.ndarray
    info: dict = field(default_factory=dict)


def _gap_fill(q, mask):
    """mask が False のフレームの回転を、前後の True のフレームの球面線形補間で埋める。"""
    if not mask.any():
        return q
    idx = np.flatnonzero(mask)
    t = np.arange(len(mask))
    k = np.searchsorted(idx, t)
    prev = idx[np.clip(k - 1, 0, len(idx) - 1)]
    nxt = idx[np.clip(k, 0, len(idx) - 1)]
    prev = np.where(mask, t, prev)
    nxt = np.where(mask, t, nxt)
    prev = np.where(t < idx[0], nxt, prev)
    nxt = np.where(t > idx[-1], prev, nxt)
    span = np.maximum(nxt - prev, 1)
    return quat.slerp(q[prev], q[nxt], (t - prev) / span)


def _bridge_gaps(w, mask, max_len):
    w = np.array(w, np.float64, copy=True)
    for a, b in filters.runs(~mask):
        if a == 0 or b == len(mask) - 1 or b - a + 1 > max_len:
            continue
        f = (np.arange(a, b + 1) - (a - 1)) / float(b - a + 2)
        w[a:b + 1] = (1.0 - f) * w[a - 1] + f * w[b + 1]
    return w


def continuous_rotvec(r):
    """回転ベクトルの列を、同じ回転のまま時間方向に連続にする（180 度をまたいでも向きが入れ替わらない）。"""
    r = np.array(r, np.float64, copy=True)
    for t in range(1, len(r)):
        a = float(np.linalg.norm(r[t]))
        if a < 1e-9:
            continue
        alt = r[t] * (1.0 - 2.0 * np.pi / a)
        if np.linalg.norm(alt - r[t - 1]) < np.linalg.norm(r[t] - r[t - 1]):
            r[t] = alt
    return r


def observed_targets(analysis, rest_joints, cfg):
    """検出結果から、手首の大域回転の目標 (N, 2, 3, 3)・使えるか (N, 2)・重み (N, 2)。"""
    world = np.asarray(analysis['world'], np.float64)
    screen = np.asarray(analysis['screen'], np.float64)
    presence = np.asarray(analysis['presence'], np.float64)
    roi = np.asarray(analysis['roi'], np.float64)
    image_size = tuple(np.asarray(analysis['image_size']).tolist())
    finite = np.isfinite(world).all(axis=(-1, -2)) & np.isfinite(screen).all(axis=(-1, -2)) & np.isfinite(roi).all(-1)
    world, screen, roi = np.nan_to_num(world), np.nan_to_num(screen), np.nan_to_num(roi)
    ok_presence = finite & (presence >= float(cfg.min_presence))
    unrot = np.einsum('...ab,...kb->...ka', roi_rotation(roi[..., 3]), world)
    fwd3 = unrot[..., LM_MIDDLE, :] - unrot[..., LM_WRIST, :]
    fwd2 = screen[..., LM_MIDDLE, :2] - screen[..., LM_WRIST, :2]
    n3 = np.linalg.norm(fwd3[..., :2], axis=-1)
    n2 = np.linalg.norm(fwd2, axis=-1)
    measurable = (n3 > 0.3 * np.linalg.norm(fwd3, axis=-1)) & (n2 > 1.0)
    cos = np.sum(fwd3[..., :2] * fwd2, axis=-1) / np.maximum(n3 * n2, 1e-12)
    ok_2d = ~measurable | (cos >= np.cos(np.deg2rad(float(cfg.check_2d_deg))))
    cam = world_to_camera(world, roi, image_size, float(cfg.fov_deg))
    F = palm_frame(cam)
    target = np.einsum('ab,nsbc,scd->nsad', CAMERA_TO_YUP, F, np.swapaxes(rest_palm_frames(rest_joints), -1, -2))
    usable = ok_presence & ok_2d
    lo, hi = float(cfg.min_presence), float(cfg.full_presence)
    weight = np.where(usable, np.clip((presence - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.0)
    return target, usable, weight


def resample_targets(target_a, usable_a, weight_a, fps_a, fps, T):
    """検出のフレーム（fps_a）の目標・使えるか・重みを、手首のフレーム（fps、T 個）にする。"""
    N = len(target_a)
    if N == 0:
        return np.tile(np.eye(3), (T, 2, 1, 1)), np.zeros((T, 2), bool), np.zeros((T, 2))
    pos = np.arange(T) * float(fps_a) / float(fps)
    lo = np.clip(np.floor(pos + 1e-9).astype(int), 0, N - 1)
    hi = np.minimum(lo + 1, N - 1)
    frac = np.clip(pos - lo, 0.0, 1.0)
    near = np.clip(np.round(pos).astype(int), 0, N - 1)
    target, usable, weight = target_a[near], usable_a[near], weight_a[near]
    both = usable_a[lo] & usable_a[hi] & (hi > lo)[:, None]
    if both.any():
        f = np.broadcast_to(frac[:, None], both.shape)
        q = quat.slerp(quat.from_matrix(target_a[lo]), quat.from_matrix(target_a[hi]), f)
        target = np.where(both[..., None, None], quat.to_matrix(q), target)
        weight = np.where(both, (1.0 - f) * weight_a[lo] + f * weight_a[hi], weight)
        usable = usable | both
    return target, usable, weight


def correct_wrists(quats, fk, rest_joints, analysis, fps, cfg, limits=None):
    """手首のローカル回転（関節 20・21）を MediaPipe の向きへ寄せる。戻り値 (quats, WristResult)。"""
    T = len(quats)
    fps_a = float(analysis.get('fps', fps))
    target_a, usable_a, weight_a = observed_targets(analysis, rest_joints, cfg)
    target, usable, weight = resample_targets(target_a, usable_a, weight_a, fps_a, fps, T)
    G = fk(quats)
    out = np.array(quats, np.float64, copy=True)
    correction = np.zeros((T, 2))
    final_w = np.zeros((T, 2))
    J = np.asarray(rest_joints, np.float64)
    used_frames = [0, 0]
    for s in range(2):
        g_wrist, g_elbow = G[:, WRIST[s]], G[:, ELBOW[s]]
        cur = quat.make_continuous(quats[:, WRIST[s]])
        local_t = quat.from_matrix(np.swapaxes(g_elbow, -1, -2) @ target[:, s])
        diff = np.rad2deg(quat.angle_between(quat.from_matrix(g_wrist), quat.from_matrix(target[:, s])))
        swing, twist = swing_twist(local_t, J[HAND[s]] - J[WRIST[s]])
        ok_anat = (np.rad2deg(swing) <= float(cfg.max_swing_deg)) & (np.abs(np.rad2deg(twist)) <= float(cfg.max_twist_deg))
        if limits is not None:
            ax = hand_axes(J[ELBOW[s]], J[WRIST[s]], s)
            ok_anat &= within_limits(wrist_angles(quats[:, ELBOW[s]], local_t, ax), limits, float(cfg.reject_margin_deg))
        use = usable[:, s] & ok_anat & (diff <= float(cfg.max_disagree_deg))
        used_frames[s] = int(use.sum())
        if not use.any():
            continue
        w = _bridge_gaps(np.where(use, weight[:, s], 0.0), use, int(round(float(cfg.hold_sec) * fps)))
        sigma = float(cfg.blend_sec) * fps
        w = np.clip(filters.gaussian_time(w, sigma, radius=2.0 * sigma), 0.0, 1.0) * float(cfg.strength)
        local_t = _gap_fill(local_t, use)
        # NLF の回転に掛ける補正を、時間方向に連続な回転ベクトルにして重みを掛け、変わる速さを抑えてならす
        r = continuous_rotvec(quat.to_rotvec(quat.mul(quat.conj(cur), local_t)))
        d = w[:, None] * r
        if float(cfg.max_speed_deg_per_s) > 0:
            d = rate_limit(d, np.deg2rad(float(cfg.max_speed_deg_per_s)) / fps)
        active = np.linalg.norm(d, axis=-1) > 1e-9
        sm = cfg.smooth
        d = filters.one_euro(d, fps, float(sm.min_cutoff), float(sm.beta), float(sm.d_cutoff), True, vector_axis=-1)
        q = quat.mul(cur, quat.from_rotvec(d))
        q = np.where(active[:, None], q, cur)
        out[:, WRIST[s]] = quat.make_continuous(q)
        final_w[:, s] = w
        correction[:, s] = np.rad2deg(quat.angle_between(out[:, WRIST[s]], cur))
    info = dict(used_frames=used_frames, used_ratio=[round(float((final_w[:, s] > 0.5).mean()), 3) for s in range(2)])
    return out, WristResult(final_w, correction, info)
