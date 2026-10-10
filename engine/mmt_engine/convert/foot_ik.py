"""ステージ7: 足ＩＫの生成（ロック → 遊脚のならし → 境界のブレンド → 床より下を 0 に）。

足ＩＫのターゲットは足首の関節の位置。VMD に書く値は「初期位置からの差分」なので、
差分 = ターゲット − 初期姿勢で直立したときの足首の位置 とし、差分の Y = 0 が「足裏が床に着いた状態」になる。

接地区間の中は、床に着いている点を固定する。
* 足裏全体（かかと・つま先とも床に着いて止まっている）: 足首の位置と回転を部分区間の中央値・平均で固定する
* つま先だけ・かかとだけ（もう一方が上がっている、または足をその点を軸に回している）: モデルの足の形のその点を
  固定し、足の回転は推定のまま（前後の固定した回転とはなめらかにつなぐ）
"""
from dataclasses import dataclass

import numpy as np

from ..core import filters, quat
from ..rig.skeleton import SIDES

SWING, FLAT, TOE, HEEL = 0, 1, 2, 3
PHASE_POINT = {TOE: 1, HEEL: 0}   # 固定する点（sole_points の [かかと, つま先] の番号）


@dataclass
class FootIKResult:
    target: np.ndarray        # (T, 2, 3) 最終的なターゲットの位置（床 y = 0）
    delta: np.ndarray         # (T, 2, 3) VMD に書く差分（内部座標）
    rotation: np.ndarray      # (T, 2, 4) 足ＩＫの回転（内部座標）
    raw: np.ndarray           # (T, 2, 3) 処理前のターゲット
    swing: np.ndarray         # (T, 2, 3) 遊脚のならした軌道
    locks: list               # 足ごとの [(開始, 終了, ロック位置), ...]
    clamped_frames: int
    phase: np.ndarray = None  # (T, 2) SWING / FLAT / TOE / HEEL
    sole_points: np.ndarray = None

    def with_values(self, **kw):
        from dataclasses import replace
        return replace(self, **kw)


def mmd_sole_points(skel):
    """(2, 2, 3) 左右の足の、足ＩＫの初期位置から見た かかと・つま先 の床の点（内部座標）。

    かかとは足首（足ＩＫ）の真下、つま先はつま先ＩＫ（無ければつま先）の真下の床（y = 0）。
    """
    out = np.zeros((2, 2, 3))
    for side, s in enumerate(SIDES):
        ik = skel.internal(s + '足ＩＫ')
        toe = next((skel.internal(s + n) for n in ('つま先ＩＫ', 'つま先') if skel.has(s + n)),
                   ik + np.array([0.0, 0.0, 0.13 * skel.leg_length(side)]))
        out[side, 0] = [0.0, -ik[1], 0.0]
        out[side, 1] = [toe[0] - ik[0], -ik[1], toe[2] - ik[2]]
    return out


def sole_floor_levels(skel, ankle_rest):
    """(2,) ターゲット + 回転 × sole_points の点が床（y = 0）に着くときの、その点の高さ（ターゲットの空間）。"""
    ik_y = np.array([skel.internal(s + '足ＩＫ')[1] for s in SIDES])
    return np.asarray(ankle_rest, np.float64)[:, 1] - ik_y


def mmd_sole_heights(skel, ik):
    """(T, 2) MMD のモデルの足裏（かかと・つま先の低いほう）の床からの高さ（足ＩＫの位置と回転から）。"""
    points = mmd_sole_points(skel)
    rest_y = np.array([skel.internal(s + '足ＩＫ')[1] for s in SIDES])
    T = len(ik.delta)
    heights = np.empty((T, 2, 2))
    for side in range(2):
        for i in range(2):
            heights[:, side, i] = quat.rotate(ik.rotation[:, side],
                                              np.broadcast_to(points[side, i], (T, 3)))[:, 1]
    return rest_y + ik.delta[..., 1] + heights.min(-1)


def _blend_weights(n):
    i = np.arange(1, n + 1)
    return 1.0 - filters.smoothstep(i / (n + 1.0))


def _lock_pieces(anchor, s, e):
    if anchor is None or not anchor[s:e + 1].any():
        return [(s, e)]
    return [(s + a, s + b) for a, b in filters.runs(anchor[s:e + 1])]


def _hold_runs(core, keep):
    """keep の連続区間のうち、core のフレームを含むもの。"""
    out = np.zeros(len(keep), bool)
    for s, e in filters.runs(keep):
        if core[s:e + 1].any():
            out[s:e + 1] = True
    return out


def contact_phases(flags, heights, rotation, swing, sole_points, fps, unit, cfg):
    """(T,) 1 つの足の接地の状態。cfg: 設定 foot_ik.pivot。

    * 上がっている点: かかと（つま先）がその足の最も低い点より raise_m 以上高い（ヒステリシス）
    * 足を回している: 足の向き（鉛直軸まわり）が turn_deg_per_s 以上の速さで回り、合計 min_turn_deg 以上回った区間。
      軸は、水平の道のりが短いほうの点
    """
    flags = np.asarray(flags, bool)
    phase = np.where(flags, FLAT, SWING).astype(np.int8)
    if not cfg.enabled:
        return phase
    heights = np.asarray(heights, np.float64)
    rise = heights - heights.min(-1, keepdims=True)
    r_on, r_keep = (float(v) * unit for v in cfg.raise_m)
    lifted = [_hold_runs((rise[:, p] > r_on) & flags, (rise[:, p] > r_keep) & flags) for p in range(2)]
    phase[flags & lifted[0] & ~lifted[1]] = TOE
    phase[flags & ~lifted[0] & lifted[1]] = HEEL

    T = len(flags)
    pts = np.asarray(sole_points, np.float64)
    q = np.asarray(rotation, np.float64)
    along = quat.rotate(q, np.broadcast_to(pts[1] - pts[0], (T, 3)))
    yaw = np.unwrap(np.arctan2(along[:, 0], along[:, 2]))
    t = np.arange(T)
    lo, hi = np.clip(t - 2, 0, T - 1), np.clip(t + 2, 0, T - 1)
    rate = np.abs(yaw[hi] - yaw[lo]) / np.maximum(hi - lo, 1) * fps
    w_on, w_keep = (np.deg2rad(float(v)) for v in cfg.turn_deg_per_s)
    turning = _hold_runs((rate > w_on) & flags, (rate > w_keep) & flags)
    points = [np.asarray(swing, np.float64) + quat.rotate(q, np.broadcast_to(pts[p], (T, 3))) for p in range(2)]
    ratio = float(cfg.pivot_ratio)
    for s, e in filters.runs(turning & (phase == FLAT)):
        if e <= s or abs(yaw[e] - yaw[s]) < np.deg2rad(float(cfg.min_turn_deg)):
            continue
        path = [float(np.linalg.norm(np.diff(p[s:e + 1][:, [0, 2]], axis=0), axis=-1).sum()) for p in points]
        if path[1] < ratio * path[0]:
            phase[s:e + 1] = TOE
        elif path[0] < ratio * path[1]:
            phase[s:e + 1] = HEEL
    return phase


def _clean_labels(labels, min_frames):
    labels = np.array(labels, copy=True)
    for s, e, lab in filters.label_runs(labels):
        if lab in (TOE, HEEL) and e - s + 1 < min_frames:
            labels[s:e + 1] = FLAT
    return labels


def build_foot_ik(ankle_pos, ankle_rest, rot_quats, contact, fps, unit, cfg, swing=None, anchor=None,
                  sole_points=None, sole_floor=None):
    """ankle_pos: (T, 2, 3) 足首の位置 / ankle_rest: (2, 3) 直立時の足首の位置 / rot_quats: (T, 2, 4) 足ＩＫの回転。

    swing: 遊脚の軌道（与えるとならさずに使う。接地優先でフルの軌道を使うため）/ anchor: (T, 2) ロックの値を
    求めるフレーム（接地優先で、フルの接地区間の値を変えないため）/ sole_points・sole_floor: 足の形の点と床の高さ。
    """
    raw = np.asarray(ankle_pos, np.float64)
    T = len(raw)
    oe = cfg.swing_one_euro
    target = np.empty_like(raw)
    swing_in = swing
    swing = np.empty_like(raw)
    rot = np.array(rot_quats, np.float64, copy=True)
    weights = _blend_weights(int(cfg.blend_frames))
    pivot = cfg.get('pivot')
    use_pivot = (sole_points is not None and sole_floor is not None and bool(cfg.lock_rotation)
                 and pivot is not None and bool(pivot.enabled))
    phase_out = np.where(np.asarray(contact.flags, bool), FLAT, SWING).astype(np.int8)
    n_blend = int(cfg.blend_frames)
    locks = []
    for foot in range(2):
        segs = contact.segments[foot]
        flags = contact.flags[:, foot]
        if swing_in is None and float(cfg.get('swing_lowpass_hz', 0.0)) > 0:
            sw = filters.lowpass(raw[:, foot], fps, float(cfg.swing_lowpass_hz), 2)
        elif swing_in is None:
            sw = filters.one_euro(raw[:, foot] / unit, fps, oe.min_cutoff, oe.beta, oe.d_cutoff,
                                  oe.zero_phase) * unit
        else:
            sw = np.array(swing_in[:, foot], np.float64, copy=True)
        swing[:, foot] = sw
        out = sw.copy()
        q = quat.make_continuous(rot[:, foot])
        q_out = q.copy()
        phase = contact_phases(flags, np.asarray(contact.heights)[:, foot], q, sw, sole_points[foot], fps,
                               unit, pivot) if use_pivot else None
        sigma = float(pivot.smooth_sec) * fps if use_pivot else 0.0
        q_piv = quat.normalize(filters.gaussian_time(q, sigma)) if sigma > 0 else q
        foot_locks = []
        sole = np.asarray(contact.heights)[:, foot].min(-1)
        pinned = None if anchor is None else np.asarray(anchor, bool)[:, foot]

        def lock_flat(a, b):
            lock = np.median(raw[a:b + 1, foot], axis=0)
            if cfg.snap_to_floor:
                lock[1] = np.median(raw[a:b + 1, foot, 1] - sole[a:b + 1])
            return lock, (quat.average(q[a:b + 1]) if cfg.lock_rotation else None)

        def plant(a, b):
            labels = _clean_labels(phase[a:b + 1], int(pivot.min_frames)) if use_pivot \
                else np.full(b - a + 1, FLAT)
            phase_out[a:b + 1, foot] = labels
            if (labels == FLAT).all():
                lock, q_lock = lock_flat(a, b)
                out[a:b + 1] = lock
                if q_lock is not None:
                    q_out[a:b + 1] = q_lock
                foot_locks.append((a, b, lock))
                return
            parts = [(a + u, a + v, lab) for u, v, lab in filters.label_runs(labels)]
            flat_q = {i: quat.average(q[u:v + 1]) for i, (u, v, lab) in enumerate(parts) if lab == FLAT}
            pin = None
            for i, (u, v, lab) in enumerate(parts):
                if lab == FLAT:
                    q_lock = flat_q[i]
                    if pin is None:
                        lock, _ = lock_flat(u, v)
                    else:
                        lock = pin[0] - quat.rotate(q_lock, pin[1])
                    out[u:v + 1] = lock
                    q_out[u:v + 1] = q_lock
                    foot_locks.append((u, v, lock))
                    pin = None
                    continue
                o = np.asarray(sole_points[foot, PHASE_POINT[lab]], np.float64)
                if i == 0:
                    point = np.median(raw[u:v + 1, foot] + quat.rotate(q_piv[u:v + 1], o), axis=0)
                    if cfg.snap_to_floor:
                        point[1] = float(sole_floor[foot])
                else:
                    point = out[u - 1] + quat.rotate(q_out[u - 1], o)
                qs = q_piv[u:v + 1].copy()
                n = v - u + 1
                ident = np.broadcast_to(quat.IDENTITY, (n, 4))
                if i > 0:
                    k = min(n_blend, n)
                    d = quat.mul(q_out[u - 1], quat.conj(q_piv[u - 1]))
                    w = 1.0 - filters.smoothstep(np.arange(1, k + 1) / (k + 1.0))
                    qs[:k] = quat.mul(quat.slerp(ident[:k], np.broadcast_to(d, (k, 4)), w), qs[:k])
                if i + 1 < len(parts) and parts[i + 1][2] == FLAT:
                    k = min(n_blend, n)
                    d = quat.mul(flat_q[i + 1], quat.conj(q_piv[v + 1]))
                    w = filters.smoothstep(np.arange(1, k + 1) / (k + 1.0))
                    qs[n - k:] = quat.mul(quat.slerp(ident[:k], np.broadcast_to(d, (k, 4)), w), qs[n - k:])
                q_out[u:v + 1] = qs
                out[u:v + 1] = point - quat.rotate(qs, np.broadcast_to(o, (n, 3)))
                foot_locks.append((u, v, out[u].copy()))
                pin = (point, o)

        for s, e in segs:
            pieces = _lock_pieces(pinned, s, e)
            for a, b in pieces:
                plant(a, b)
            (a0, _), (_, b1) = pieces[0], pieces[-1]
            out[s:a0], out[b1 + 1:e + 1] = out[a0], out[b1]
            for (_, b), (a, _) in zip(pieces[:-1], pieces[1:]):
                w = filters.smoothstep(np.arange(1, a - b) / (a - b))
                out[b + 1:a] = out[b] + (out[a] - out[b]) * w[:, None]
                if cfg.lock_rotation:
                    q_out[b + 1:a] = quat.slerp(q_out[b], q_out[a], w)
            if cfg.lock_rotation:
                q_out[s:a0], q_out[b1 + 1:e + 1] = q_out[a0], q_out[b1]
        # 境界のブレンド: 「ロック値 − 遊脚値」を遊脚側だけで減衰させて足す
        offset = np.zeros((T, 3))
        rot_delta = []
        for s, e in segs:
            for edge, step in ((e, 1), (s, -1)):
                d = out[edge] - sw[edge]
                dq = quat.mul(q_out[edge], quat.conj(q[edge]))
                for i, w in enumerate(weights, start=1):
                    f = edge + step * i
                    if f < 0 or f >= T or flags[f]:
                        break
                    offset[f] += d * w
                    rot_delta.append((f, dq, w))
        out[~flags] += offset[~flags]
        for f, dq, w in rot_delta:
            q_out[f] = quat.mul(quat.slerp(quat.IDENTITY, dq, w), q_out[f])
        target[:, foot] = out
        rot[:, foot] = quat.make_continuous(q_out)
        locks.append(foot_locks)

    delta = target - np.asarray(ankle_rest)[None]
    clamped = 0
    if cfg.clamp_floor:
        # 残差の保険。ロックより後で行う（先にやるとロック値に歪んだ値が混ざる）
        below = delta[..., 1] < 0
        clamped = int(below.sum())
        delta[..., 1] = np.maximum(delta[..., 1], 0.0)
        target = delta + np.asarray(ankle_rest)[None]
    return FootIKResult(target, delta, rot, raw, swing, locks, clamped, phase_out,
                        None if sole_points is None else np.asarray(sole_points, np.float64))


def boundary_steps(target, contact):
    """接地区間の境界での移動量の最大値（足ごと）。"""
    out = []
    T = len(target)
    for foot in range(2):
        steps = [0.0]
        for s, e in contact.segments[foot]:
            for a, b in ((e, e + 1), (s - 1, s)):
                if 0 <= a and b < T:
                    steps.append(float(np.linalg.norm(target[b, foot] - target[a, foot])))
        out.append(max(steps))
    return out
