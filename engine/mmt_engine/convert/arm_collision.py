"""ステージ9a: 腕どうしの貫通の防止。

MMD のモデルの左右の腕を、上腕・前腕・手の 3 本のカプセルで表し、左右が重なったフレームで、自重する側の腕を
肩（腕ボーンの位置）まわりに回し、ひじを曲げ伸ばしして離す（手首の向きは変えない）。
* 1 フレームずつ順に、前のフレームの補正から始めて、補正をなるべく 0（推定の姿勢）に戻しながら重ならない
  補正を求める（相手の腕が来れば押しのけ、離れれば戻る。推定の腕が相手の腕を通り抜けても来た側に留まる）
* 奥行き優先: 画像面内の腕の位置は動画に写ったとおりで確かだが奥行きはぶれるので、腕を奥行き方向へ動かす
  補正を安く見積もり、前後へずらして離す
* 補正を時間方向にならしてから、ならして浅くなった重なりをもう一度離す
"""
from dataclasses import dataclass, replace

import numpy as np

from ..core import filters, quat
from ..rig.skeleton import SIDES
from .geometry import (angle, closest_points, constrained_step, cross, distance_to_segment, inv_sqrt,
                       rotation, rotvec, skew, toward_identity)
from .retarget import bone_positions

MODES = ('left', 'right', 'none')
SEGMENT_BONES = ('腕', 'ひじ', '手首')
PAIR_Y = np.repeat(np.arange(3), 3)
PAIR_O = np.tile(np.arange(3), 3)
ITERATIONS = 30
FINAL_ITERATIONS = 10
DAMPING_M = 0.1
MAX_STEP_DEG = 20.0
STOP_RAD = 1e-4
MIN_LEVER_M = 0.03
TOLERANCE_M = 1e-4
HAND_TIP_PERCENTILE = 95.0
MIN_VERTICES = 20
DEEP_START = 0.5
DEPTH_SAMPLES = 13
SAME_DEPTH = 0.02
RELAX_STEPS = 4
ELBOW_FLEX_DEG = (0.0, 150.0)
ELBOW_REG = 0.02
OVERLAP_TOL_M = 0.005
_EPS = 1e-9


@dataclass
class ArmModel:
    tip: np.ndarray        # (2, 3) 初期姿勢の手の先
    radius: np.ndarray     # (2, 3) 上腕・前腕・手の半径
    source: str            # mesh | config


@dataclass
class ArmCollisionResult:
    mode: str
    side: int
    radius: np.ndarray
    radius_source: str
    correction_deg: np.ndarray
    depth_before: np.ndarray
    depth_after: np.ndarray
    elbow_deg: np.ndarray = None

    def overlap_frames(self, tol):
        return int((self.depth_before > tol).sum()), int((self.depth_after > tol).sum())


def _ancestor_or_self(skel, name, candidates):
    return name if name in candidates else skel.nearest_ancestor(name, candidates)


def _hand_axis(skel, s):
    wrist = skel.internal(s + '手首')
    for target in (skel.internal(s + '中指１') if skel.has(s + '中指１') else None,
                   skel.tail_internal(s + '手首'), wrist + (wrist - skel.internal(s + 'ひじ'))):
        if target is not None and np.linalg.norm(target - wrist) > 1e-6:
            return (target - wrist) / np.linalg.norm(target - wrist)
    return np.array([1.0 if s == '左' else -1.0, 0.0, 0.0])


def arm_model(skel, cfg, unit):
    """腕のカプセルの寸法（半径は PMX のメッシュから。無ければ設定の値）× radius_scale。"""
    fallback = np.asarray(cfg.fallback_radius_m, np.float64) * unit
    radius = np.tile(fallback, (2, 1))
    tips = []
    measured = False
    mesh = skel.mesh_points is not None and len(skel.mesh_points) > 0
    if mesh:
        points = skel.mesh_points * [1.0, 1.0, -1.0]
        names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    for side, s in enumerate(SIDES):
        wrist = skel.internal(s + '手首')
        axis = _hand_axis(skel, s)
        hand_length = float(cfg.hand_length_m) * unit
        if mesh:
            roots = {s + b for b in SEGMENT_BONES}
            owner = np.array([(_ancestor_or_self(skel, n, roots) or '') if skel.has(n) else ''
                              for n in names])[inverse.reshape(-1)]
            for k, bone in enumerate(SEGMENT_BONES):
                pts = points[owner == s + bone]
                if len(pts) < MIN_VERTICES:
                    continue
                if k == 2:
                    centroid = pts.mean(0) - wrist
                    if np.linalg.norm(centroid) > 1e-6:
                        axis = centroid / np.linalg.norm(centroid)
                    hand_length = max(float(np.percentile((pts - wrist) @ axis, HAND_TIP_PERCENTILE)), 0.05 * unit)
                    end = wrist + axis * hand_length
                else:
                    end = skel.internal(s + SEGMENT_BONES[k + 1])
                d = distance_to_segment(pts, skel.internal(s + bone), end)
                radius[side, k] = float(np.percentile(d, float(cfg.radius_percentile)))
                measured = True
        tips.append(wrist + axis * hand_length)
    if cfg.radius_m != 'auto':
        radius = np.tile(np.asarray(cfg.radius_m, np.float64).reshape(3) * unit, (2, 1))
        measured = False
    radius = np.maximum(radius * float(cfg.radius_scale), 1e-6)
    return ArmModel(np.stack(tips), radius, 'mesh' if measured else 'config')


def arm_points(skel, glob, model, num_frames):
    """(T, 2, 4 [腕, ひじ, 手首, 手の先], 3) 腕のカプセルの端点。"""
    names = [s + b for s in SIDES for b in SEGMENT_BONES]
    pos = bone_positions(skel, glob, names, num_frames)
    out = np.empty((num_frames, 2, 4, 3))
    for side, s in enumerate(SIDES):
        for k, b in enumerate(SEGMENT_BONES):
            out[:, side, k] = pos[s + b]
        wrist = s + '手首'
        out[:, side, 3] = pos[wrist] + glob[wrist] @ (model.tip[side] - skel.internal(wrist))
    return out


def _pairs(Y, O):
    return closest_points(Y[..., PAIR_Y, :], Y[..., PAIR_Y + 1, :], O[..., PAIR_O, :], O[..., PAIR_O + 1, :])


def overlap_depth(Y, O, radius_y, radius_o):
    """(T,) 左右の腕のカプセルの最も深い重なり（負なら離れている）。"""
    cp, cq = _pairs(Y, O)
    dist = np.linalg.norm(cp - cq, axis=-1)
    return (radius_y[PAIR_Y] + radius_o[PAIR_O] - dist).max(-1)


def _perpendicular(Y, O):
    n = cross(Y[PAIR_Y + 1] - Y[PAIR_Y], O[PAIR_O + 1] - O[PAIR_O])
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > _EPS, n / np.maximum(norm, _EPS), [0.0, 0.0, 1.0])


@dataclass
class _Params:
    max_angle: float
    min_lever: float
    tol: float
    damping: float
    return_step: float
    axis: np.ndarray = None
    depth_weight: float = 1.0

    @property
    def prefers_depth(self):
        return self.axis is not None and self.depth_weight < 1.0


@dataclass
class _Elbow:
    b: np.ndarray     # ひじの曲げの軸（大域）
    lo: float
    hi: float


def _pose(Y, state, elbow=None):
    Q, a = state
    A = Y[0]
    cur = A + (Y - A) @ Q.T
    if elbow is not None and a != 0.0:
        E = cur[1]
        cur[2:] = E + (cur[2:] - E) @ rotation(a * (Q @ elbow.b)).T
    return cur


def _metric(Y, p):
    r = Y[1:] - Y[0]
    u = cross(r, p.axis) / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), _EPS)
    return np.eye(3) - (1.0 - p.depth_weight) * (u.T @ u) / len(u)


def _point_metric(cur, bq, p):
    A, E = cur[0], cur[1]
    W = np.eye(3) - (1.0 - p.depth_weight) * np.outer(p.axis, p.axis)
    M = np.zeros((4, 4))
    scale = 0.0
    for k in (1, 2, 3):
        r = cur[k] - A
        G = np.zeros((3, 4))
        G[:, :3] = -skew(r)
        if k >= 2:
            G[:, 3] = cross(bq, cur[k] - E)
        M += G.T @ W @ G
        scale += float(r @ r)
    return M / max(scale, _EPS) + ELBOW_REG * np.eye(4)


def _jacobian(cp, n, A, cur, bq):
    J = cross(cp - A, n)
    if bq is None:
        return J
    ve = (cp - cur[1]) * (PAIR_Y >= 1)[:, None]
    return np.concatenate([J, (cross(ve, n) @ bq)[:, None]], axis=1)


def _depth_normals(cp, d, n, pen, R, M_inv, p, A, cur, bq):
    """奥行き優先: 重なっている組の離す向きを、最近点を結ぶ向きと奥行きの軸の間から最も安いものに選び直す。"""
    a = p.axis
    over = pen > p.tol
    along = n @ a
    e = n - along[:, None] * a
    e_norm = np.linalg.norm(e, axis=-1)
    over &= e_norm > 1e-6
    if not over.any():
        return n, pen
    idx = np.flatnonzero(over)
    e_hat = e[idx] / e_norm[idx, None]
    lo = np.where(along[idx] > SAME_DEPTH, 0.0, -0.5 * np.pi)
    hi = np.where(along[idx] < -SAME_DEPTH, 0.0, 0.5 * np.pi)
    phi = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, DEPTH_SAMPLES)
    phi = np.concatenate([np.arctan2(along[idx], e_norm[idx])[:, None], phi], axis=1)
    cand = np.cos(phi)[..., None] * e_hat[:, None] + np.sin(phi)[..., None] * a
    need = R[idx, None] - np.einsum('mkc,mc->mk', cand, d[idx]) + p.tol
    j = cross(cp[idx, None] - A, cand)
    if bq is not None:
        ve = (cp[idx] - cur[1]) * (PAIR_Y[idx] >= 1)[:, None]
        j = np.concatenate([j, (cross(ve[:, None], cand) @ bq)[..., None]], axis=-1)
    ease = np.einsum('mkc,cd,mkd->mk', j, M_inv, j)
    lever = np.linalg.norm(j, axis=-1) > p.min_lever
    cost = np.where(lever, np.maximum(need, 0.0) ** 2 / np.maximum(ease, _EPS), np.inf)
    best = np.argmin(cost, axis=1)
    ok = np.isfinite(cost[np.arange(len(idx)), best])
    n, pen = n.copy(), pen.copy()
    sel = idx[ok]
    n[sel] = cand[ok, best[ok]]
    pen[sel] = need[ok, best[ok]] - p.tol
    return n, pen


def _solve_frame(Y, O, R, state, p, goal=None, ignore=None, elbow=None):
    """1 フレーム分の補正 state = (肩まわりの回転 Q, ひじの曲げ a) を初期値 state から求める。"""
    Q, a = state
    A = Y[0]
    rel = Y - A
    skip = np.zeros(len(R), bool) if ignore is None else ignore
    pulling = goal is not None
    for it in range(ITERATIONS + FINAL_ITERATIONS):
        pulling = pulling and it < ITERATIONS
        cur = A + rel @ Q.T if elbow is None else _pose(Y, (Q, a), elbow)
        bq = None if elbow is None else Q @ elbow.b
        cp, cq = _pairs(cur, O)
        d = cp - cq
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, _EPS)[:, None]
        if (dist <= _EPS).any():
            n = np.where((dist > _EPS)[:, None], n, _perpendicular(cur, O))
        pen = R - dist
        C = M = None
        if elbow is not None or p.prefers_depth:
            M = _metric(cur, p) if elbow is None else _point_metric(cur, bq, p)
            C = inv_sqrt(M)
            if p.prefers_depth:
                n, pen = _depth_normals(cp, d, n, pen, R, C @ C, p, A, cur, bq)
        J = _jacobian(cp, n, A, cur, bq)
        use = (np.linalg.norm(J, axis=-1) > p.min_lever) & ~skip
        if elbow is None:
            target = rotvec(goal[0] @ Q.T) if pulling else np.zeros(3)
        else:
            target = np.append(rotvec(goal[0] @ Q.T), goal[1] - a) if pulling else np.zeros(4)
        if not (use & (pen > p.tol)).any() and np.linalg.norm(target) < 1e-9:
            break
        x = target
        if use.any():
            Ju = J[use]
            b = pen[use] + p.tol - Ju @ target
            x = target + (constrained_step(Ju, b, p.damping) if C is None
                          else C @ constrained_step(Ju @ C, b, p.damping))
            if elbow is not None and ((a <= elbow.lo + 1e-9 and x[3] < 0.0) or (a >= elbow.hi - 1e-9 and x[3] > 0.0)):
                b = pen[use] + p.tol - Ju[:, :3] @ target[:3]
                C3 = inv_sqrt(M[:3, :3])
                x = np.append(target[:3] + C3 @ constrained_step(Ju[:, :3] @ C3, b, p.damping), 0.0)
        step = float(np.linalg.norm(x)) if elbow is None else max(float(np.linalg.norm(x[:3])), abs(float(x[3])))
        if step < STOP_RAD:
            if pulling:
                pulling = False
                continue
            break
        x = x * min(1.0, np.deg2rad(MAX_STEP_DEG) / step)
        Q = rotation(x[:3]) @ Q
        if angle(Q) > p.max_angle:
            rv = rotvec(Q)
            Q = rotation(rv * (p.max_angle / np.linalg.norm(rv)))
        if elbow is not None:
            a = float(np.clip(a + x[3], elbow.lo, elbow.hi))
    return Q, a


def _overlapping(Y, O, R, state, tol, elbow=None):
    if elbow is None:
        A = Y[0]
        cp, cq = _pairs(A + (Y - A) @ state[0].T, O)
    else:
        cp, cq = _pairs(_pose(Y, state, elbow), O)
    return R - np.linalg.norm(cp - cq, axis=-1) > tol


def _saturated(state, p):
    return angle(state[0]) >= p.max_angle - 1e-6


def _attempt(solve, Y, O, R, q, elbow=None):
    state = solve(q)
    return state, _overlapping(Y, O, R, state, q.tol, elbow), _saturated(state, q)


def _relax_depth(solve, Y, O, R, p, first, elbow=None):
    """奥行きを優先して上限に達しても離れないフレーム: depth_cost を 1 へ向けて、離れるところまで弱める。"""
    def failed(res):
        return res[2] and res[1].any()

    lo, hi = float(np.sqrt(p.depth_weight)), 1.0
    iso = _attempt(solve, Y, O, R, replace(p, depth_weight=1.0), elbow)
    if failed(iso):
        if iso[1].sum() < first[1].sum():
            return (*iso, 1.0)
        return (*first, p.depth_weight)
    best = (*iso, 1.0)
    for _ in range(RELAX_STEPS):
        mid = 0.5 * (lo + hi)
        res = _attempt(solve, Y, O, R, replace(p, depth_weight=mid * mid), elbow)
        if failed(res):
            lo = mid
        else:
            hi, best = mid, (*res, mid * mid)
    return best


def _solve_deep_start(Y, O, R, G, p, ignore, elbow=None):
    """持ち越す補正が無いのに深く重なっているフレーム: いくつかの向きから解き、最も小さい解を選ぶ。"""
    best, best_score = None, np.inf
    starts = [np.eye(3)] + [rotation(G @ (sign * np.deg2rad(MAX_STEP_DEG) * np.eye(3)[axis]))
                            for axis in range(3) for sign in (1.0, -1.0)]
    if elbow is not None:
        M = _point_metric(Y, elbow.b, p)
    else:
        M = _metric(Y, p) if p.prefers_depth else None
    for Q0 in starts:
        state = _solve_frame(Y, O, R, (Q0, 0.0), p, (np.eye(3), 0.0), ignore, elbow)
        cp, cq = _pairs(Y[0] + (Y - Y[0]) @ state[0].T if elbow is None else _pose(Y, state, elbow), O)
        pen = np.where(ignore, -np.inf, R - np.linalg.norm(cp - cq, axis=-1))
        residual = max(0.0, float(pen.max()) - p.tol)
        if M is None:
            size = angle(state[0])
        else:
            x = rotvec(state[0]) if elbow is None else np.append(rotvec(state[0]), state[1])
            size = float(np.sqrt(x @ M @ x))
        score = size + 100.0 * residual / p.min_lever
        if score < best_score:
            best, best_score = state, score
    return best


def _resolve_sequential(Y, O, R, Gp, p, elbows=None, cancel=None):
    """1 回目: フレーム順に、前のフレームの補正から始めて解く（押しのけ・戻り）。"""
    T = len(Y)
    cp, cq = _pairs(Y, O)
    raw_pen = R - np.linalg.norm(cp - cq, axis=-1)
    Q = np.tile(np.eye(3), (T, 1, 1))
    bend = np.zeros(T)
    ignored = np.zeros((T, len(R)), bool)
    weight = np.full(T, p.depth_weight)
    local = np.eye(3)
    a_prev = 0.0
    skip = np.zeros(len(R), bool)
    for t in range(T):
        if cancel is not None and t % 64 == 0:
            cancel()
        elbow = None if elbows is None else elbows[t]
        carried = angle(local) > 1e-6 or abs(a_prev) > 1e-6
        if not carried and (raw_pen[t] <= 0.0).all():
            skip[:] = False
            continue
        deep = not carried and raw_pen[t].max() > DEEP_START * R.min()
        if elbow is not None:
            a_prev = float(np.clip(a_prev, min(elbow.lo, 0.0), max(elbow.hi, 0.0)))

        def solve(q, t=t, elbow=elbow, deep=deep, local=local, a_prev=a_prev):
            if deep:
                return _solve_deep_start(Y[t], O[t], R, Gp[t], q, skip, elbow)
            goal = (Gp[t] @ toward_identity(local, q.return_step) @ Gp[t].T,
                    float(np.sign(a_prev) * max(abs(a_prev) - q.return_step, 0.0)))
            return _solve_frame(Y[t], O[t], R, (Gp[t] @ local @ Gp[t].T, a_prev), q, goal, skip, elbow)

        state, over, saturated = _attempt(solve, Y[t], O[t], R, p, elbow)
        if saturated and over.any() and p.prefers_depth:
            state, over, saturated, weight[t] = _relax_depth(solve, Y[t], O[t], R, p, (state, over, saturated), elbow)
        Q[t], bend[t] = state
        if saturated:
            skip = skip | over
        skip = skip & over
        ignored[t] = skip
        local = Gp[t].T @ Q[t] @ Gp[t]
        a_prev = bend[t]
    return Q, bend, ignored, weight


def _pose_all(Y, Q, bend, b=None):
    A = Y[:, :1]
    cur = A + np.einsum('tab,tkb->tka', Q, Y - A)
    if b is not None:
        E = cur[:, 1:2]
        Re = quat.to_matrix(quat.from_rotvec(bend[:, None] * np.einsum('tab,tb->ta', Q, b)))
        cur[:, 2:] = E + np.einsum('tab,tkb->tka', Re, cur[:, 2:] - E)
    return cur


def _resolve_residual(Y, O, R, Q, bend, ignored, p, weight, elbows=None):
    """2 回目: ならした補正で重なるフレームだけ、重ならないところまで回す。"""
    b = None if elbows is None else np.array([e.b for e in elbows])
    cp, cq = _pairs(_pose_all(Y, Q, bend, b), O)
    touching = (np.linalg.norm(cp - cq, axis=-1) < R) & ~ignored
    Q, bend = Q.copy(), bend.copy()
    for t in np.flatnonzero(touching.any(1)):
        q = p if weight[t] == p.depth_weight else replace(p, depth_weight=float(weight[t]))
        elbow = None if elbows is None else elbows[t]
        a = 0.0 if elbow is None else float(np.clip(bend[t], min(elbow.lo, 0.0), max(elbow.hi, 0.0)))
        Q[t], bend[t] = _solve_frame(Y[t], O[t], R, (Q[t], a), q, ignore=ignored[t], elbow=elbow)
    return Q, bend


def elbow_axis_local(skel, side):
    """ひじの曲げの軸（腕ボーンの初期姿勢の座標系。前腕の初期の向き × 正面）。"""
    s = SIDES[side]
    d = skel.internal(s + '手首') - skel.internal(s + 'ひじ')
    b = np.cross(d, [0.0, 0.0, 1.0])
    return b / max(float(np.linalg.norm(b)), _EPS)


def _flexion(Y, b):
    u = Y[:, 1] - Y[:, 0]
    f = Y[:, 2] - Y[:, 1]
    return np.arctan2(np.einsum('tc,tc->t', b, cross(u, f)), np.einsum('tc,tc->t', u, f))


def _elbows(Y, b, cfg):
    limit = np.deg2rad(float(cfg.elbow_max_deg))
    flex = _flexion(Y, b)
    lo = np.minimum(0.0, np.maximum(-limit, np.deg2rad(ELBOW_FLEX_DEG[0]) - flex))
    hi = np.maximum(0.0, np.minimum(limit, np.deg2rad(ELBOW_FLEX_DEG[1]) - flex))
    return [_Elbow(b[t], float(lo[t]), float(hi[t])) for t in range(len(Y))]


def resolve_arm_collisions(skel, rt, glob_rot, local, cfg, unit, fps, depth_axis=None, cancel=None):
    """ステージ9a。local の自重する側の腕・ひじの回転を直した dict と ArmCollisionResult。"""
    mode = str(cfg.mode)
    if mode not in MODES:
        raise ValueError(f'arm_collision.mode が不正です: {mode}')
    depth_cost = float(cfg.depth_cost)
    if not 0.0 < depth_cost <= 1.0:
        raise ValueError(f'arm_collision.depth_cost は 0 より大きく 1 以下で指定してください: {cfg.depth_cost}')
    axis = np.array([0.0, 0.0, 1.0]) if depth_axis is None else np.asarray(depth_axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    T = len(glob_rot)
    model = arm_model(skel, cfg, unit)
    glob = rt.globals_from_local(glob_rot, local)
    pts = arm_points(skel, glob, model, T)
    before = overlap_depth(pts[:, 0], pts[:, 1], model.radius[0], model.radius[1])
    if mode == 'none' or T == 0:
        return local, ArmCollisionResult(mode, -1, model.radius, model.source, np.zeros(T), before, before.copy(),
                                         np.zeros(T))
    y = 0 if mode == 'left' else 1
    o = 1 - y
    arm, elbow_bone = SIDES[y] + '腕', SIDES[y] + 'ひじ'
    Y, O = pts[:, y], pts[:, o]
    R = model.radius[y][PAIR_Y] + model.radius[o][PAIR_O] + float(cfg.margin_m) * unit
    p = _Params(np.deg2rad(float(cfg.max_deg)), MIN_LEVER_M * unit, TOLERANCE_M * unit, (DAMPING_M * unit) ** 2,
                np.deg2rad(float(cfg.return_deg_per_s)) / fps, axis, depth_cost ** 2)
    use_elbow = bool(cfg.elbow) and elbow_bone in local and elbow_bone in glob
    b = elbows = None
    if use_elbow:
        b = glob[arm] @ elbow_axis_local(skel, y)
        elbows = _elbows(Y, b, cfg)
    parent = rt.keyed_parent[arm]
    Gp = glob[parent] if parent is not None else np.tile(np.eye(3), (T, 1, 1))
    Gp_t = np.swapaxes(Gp, -1, -2)
    Q, bend, ignored, weight = _resolve_sequential(Y, O, R, Gp, p, elbows, cancel)
    sigma = float(cfg.smooth_sec) * fps
    rv = filters.gaussian_time(quat.to_rotvec(quat.from_matrix(Gp_t @ Q @ Gp)), sigma)
    if use_elbow:
        bend = filters.gaussian_time(bend, sigma)
    Q, bend = _resolve_residual(Y, O, R, Gp @ quat.to_matrix(quat.from_rotvec(rv)) @ Gp_t, bend, ignored, p,
                                weight, elbows)
    q_local = quat.from_matrix(Gp_t @ Q @ Gp)
    corr = np.rad2deg(np.linalg.norm(quat.to_rotvec(q_local), axis=-1))
    elbow_deg = np.rad2deg(np.abs(bend))
    if corr.max(initial=0.0) > 1e-9 or elbow_deg.max(initial=0.0) > 1e-9:
        local = dict(local)
        local[arm] = quat.make_continuous(quat.mul(q_local, local[arm]))
        if use_elbow and elbow_deg.max(initial=0.0) > 1e-9:
            # ひじの軸 Q b まわりの曲げを、ひじの親（補正後は Q·P）の座標系で表す: (Q P)ᵀ Q b = Pᵀ b
            ep = rt.keyed_parent[elbow_bone]
            P = glob[ep] if ep is not None else np.tile(np.eye(3), (T, 1, 1))
            b_parent = np.einsum('tba,tb->ta', P, b)
            q_elbow = quat.from_rotvec(bend[:, None] * b_parent)
            local[elbow_bone] = quat.make_continuous(quat.mul(q_elbow, local[elbow_bone]))
    after = overlap_depth(_pose_all(Y, Q, bend, b), O, model.radius[y], model.radius[o])
    return local, ArmCollisionResult(mode, y, model.radius, model.source, corr, before, after, elbow_deg)
