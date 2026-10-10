"""ステージ4: 床面の推定と定数のオフセット。

床の高さの補正はシーケンス全体で 1 つの定数にする（フレームごとに変わる補正は接地判定やロックの値に歪みを混ぜる）。
床の傾きは、同じフレーム内の接地点どうしを結ぶベクトル（かかと → つま先・左足 → 右足）が床と平行になるように
求める（単眼推定では体全体が視線に沿ってずれるので、全フレームの接地点に 1 枚の平面を当てはめると、ずれで
伸びた点群の向きを床の傾きと取り違える）。
"""
from dataclasses import dataclass

import numpy as np

from ..body.smpl import ANKLES
from ..core import quat


def horizontal_speed(points, fps):
    """(T, ...) の点の水平速度 [単位/s]。"""
    points = np.asarray(points, np.float64)
    if len(points) < 2:
        return np.zeros(points.shape[:-1])
    v = np.gradient(points, axis=0) * fps
    return np.linalg.norm(v[..., [0, 2]], axis=-1)


def fit_plane_ransac(points, iterations, inlier_dist, rng, max_tilt_deg=90.0):
    """RANSAC で平面を当てはめ、インライアで最小二乗し直す。戻り値 (法線, インライア, 平面内の広がり)。"""
    best = None
    n_pts = len(points)
    min_ny = np.cos(np.deg2rad(max_tilt_deg))
    for _ in range(int(iterations)):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9 or abs(n[1]) / norm < min_ny:
            continue
        n /= norm
        inl = np.abs((points - a) @ n) < inlier_dist
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        best = np.ones(n_pts, bool)
    p = points[best]
    centroid = p.mean(0)
    _, sv, vt = np.linalg.svd(p - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    spread = float(sv[1] / np.sqrt(len(p))) if len(sv) > 1 else 0.0
    return normal, best, spread


def floor_vectors(points, low, rest_points):
    """同じフレームの低速の接地点どうしを結ぶベクトルと、足裏が床に平らに着いているときのその上下成分。

    points: (T, 2 足, 2 [かかと, つま先], 3) / low: (T, 2, 2) bool / rest_points: (2, 2, 3) 初期姿勢の接地点。
    """
    rest = np.asarray(rest_points, np.float64)
    vecs, targets = [], []
    for foot in range(2):
        m = low[:, foot].all(-1)
        vecs.append(points[m, foot, 1] - points[m, foot, 0])
        targets.append(np.full(m.sum(), rest[foot, 1, 1] - rest[foot, 0, 1]))
    m = low.reshape(len(low), -1).all(-1)
    mid, rest_mid = points[m].mean(2), rest.mean(1)
    vecs.append(mid[:, 1] - mid[:, 0])
    targets.append(np.full(m.sum(), rest_mid[1, 1] - rest_mid[0, 1]))
    return np.concatenate(vecs).reshape(-1, 3), np.concatenate(targets)


def sole_normals(glob_rot, low, rest_points):
    """(N, 3) 接地中の足裏の上向き（初期姿勢の足裏の上向きを足首の大域回転で回したもの）。"""
    rest = np.asarray(rest_points, np.float64)
    out = []
    for foot in range(2):
        d = rest[foot, 1] - rest[foot, 0]
        d /= max(np.linalg.norm(d), 1e-9)
        up = np.array([0.0, 1.0, 0.0]) - d[1] * d
        up /= np.linalg.norm(up)
        out.append(glob_rot[low[:, foot].all(-1), foot] @ up)
    return np.concatenate(out).reshape(-1, 3)


def _tukey_weights(r, c=4.685, min_scale=np.sin(np.deg2rad(1.0))):
    """Tukey の重み（スケールは残差の絶対値の中央値から）。"""
    scale = max(1.4826 * float(np.median(np.abs(r))), min_scale) if len(r) else min_scale
    u = np.asarray(r) / (c * scale)
    return np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)


def fit_tilt_vectors(vecs, targets, min_spread, normals=None, normal_weight=0.0, iterations=20):
    """床と平行なはずのベクトルから床の法線を求める（外れたものの重みを下げながら最小二乗）。

    法線を n ∝ (a, 1, b) とおくと、条件 n·v = c は a, b について線形になる。接地中の足裏の向きも弱い重みで
    「n ∝ u」の条件として加え、ベクトルで決まらない方向を補う。戻り値:
    (法線, 使ったベクトル, 'full' | 'line' | 'none', 水平の広がり (2,), 使った足裏の向きの数)
    """
    vecs, targets = np.asarray(vecs, np.float64), np.asarray(targets, np.float64)
    normals = np.zeros((0, 3)) if normals is None or normal_weight <= 0 else np.asarray(normals, np.float64)
    normals = normals[normals[:, 1] > 0.5]      # 60 度以上傾いた足裏は床に着いていない
    wv, wn = np.ones(len(vecs)), np.ones(len(normals))
    g = np.zeros(2)
    mode, spread = 'none', np.zeros(2)
    U = np.eye(2)
    length = np.maximum(np.linalg.norm(vecs, axis=1), 1e-9)
    slopes = normals[:, [0, 2]] / normals[:, 1:2]
    for _ in range(int(iterations)):
        H = vecs[:, [0, 2]] * wv[:, None]
        if (wv > 0).sum() >= 3:
            w, U = np.linalg.eigh(H.T @ H / wv.sum())
            spread = np.sqrt(np.maximum(w, 0.0))
        rhs = (targets * np.sqrt(1.0 + g @ g) - vecs[:, 1]) * wv
        if (wn > 0).sum() >= 3:
            wn2 = np.repeat(normal_weight * wn, 2)
            A = np.concatenate([H, wn2[:, None] * np.tile(np.eye(2), (len(normals), 1))])
            b = np.concatenate([rhs, wn2 * slopes.reshape(-1)])
            g = np.linalg.lstsq(A, b, rcond=None)[0]
            mode = 'full'
        elif (wv > 0).sum() >= 3 and spread[0] >= min_spread:
            g = np.linalg.lstsq(H, rhs, rcond=None)[0]
            mode = 'full'
        elif (wv > 0).sum() >= 3 and spread[1] >= min_spread:
            e = U[:, 1]
            g = np.linalg.lstsq((H @ e)[:, None], rhs, rcond=None)[0][0] * e
            mode = 'line'
        else:
            mode, g = 'none', np.zeros(2)
            break
        n = np.array([g[0], 1.0, g[1]]) / np.sqrt(1.0 + g @ g)
        wv = _tukey_weights((vecs @ n - targets) / length)
        wn = _tukey_weights(np.linalg.norm(np.cross(normals, n), axis=1))
    n = np.array([g[0], 1.0, g[1]]) / np.sqrt(1.0 + g @ g)
    return n, wv > 0, mode, spread, int((wn > 0).sum())


@dataclass
class FloorResult:
    rotation: np.ndarray         # (3, 3) 床の傾きの補正（＋向きの反転）
    offset: np.ndarray           # (T, 3) 回転後に足す量（上下は定数、水平は原点合わせ）
    normal: np.ndarray           # 推定した床の法線（補正前）
    tilt_deg: float
    num_points: int
    num_inliers: int
    fallback: bool               # 候補点が足りず、傾きを推定しなかった
    tilt_applied: bool = False
    spread: float = 0.0
    tilt_mode: str = 'none'      # full | line | none

    def apply(self, kin):
        return kin.transformed(self.rotation, self.offset)

    def info(self):
        return dict(tilt_deg=round(self.tilt_deg, 2), tilt_mode=self.tilt_mode, points=self.num_points,
                    inliers=self.num_inliers, fallback=self.fallback)


def estimate_floor(kin, fps, cfg, rest_points=None, valid=None):
    """かかと・つま先の低速の点から床を推定し、床が y = 0・法線が +Y になる変換を返す。

    valid: 人物を検出できたフレーム（False のフレームは前後から補間しただけなので床の候補にしない）。
    """
    T = len(kin.contact_points)
    pts = kin.contact_points.reshape(T, -1, 3)
    observed = np.ones(T, bool) if valid is None else np.asarray(valid, bool)
    if not observed.any():
        observed = np.ones(T, bool)
    speed = horizontal_speed(pts, fps)
    low = (speed < float(cfg.low_speed_m_per_s)) & observed[:, None]
    cand = pts[low]
    rng = np.random.default_rng(int(cfg.seed))

    R = np.eye(3)
    normal = np.array([0.0, 1.0, 0.0])
    inliers = np.ones(len(cand), bool)
    fallback = len(cand) < int(cfg.min_points)
    tilt, spread, tilt_mode, num_inliers = 0.0, 0.0, 'none', len(cand)
    method = str(cfg.tilt_method)
    if not fallback and method == 'vectors':
        rest = np.zeros((2, 2, 3)) if rest_points is None else rest_points
        low4 = low.reshape(T, 2, 2)
        vecs, targets = floor_vectors(kin.contact_points, low4, rest)
        normals = sole_normals(kin.glob_rot[:, ANKLES], low4, rest)
        normal, vec_inl, mode, vec_spread, _ = fit_tilt_vectors(
            vecs, targets, cfg.min_vector_spread_m, normals, float(cfg.sole_normal_weight_m))
        tilt, spread = float(np.rad2deg(np.arccos(np.clip(normal[1], -1, 1)))), float(vec_spread[0])
        num_inliers = int(vec_inl.sum())
        if cfg.align_tilt and mode != 'none' and tilt <= float(cfg.max_tilt_deg):
            R = quat.to_matrix(quat.from_two_vectors(normal, [0.0, 1.0, 0.0]))
            tilt_mode = mode
    elif not fallback and method == 'plane':
        normal, inliers, spread = fit_plane_ransac(cand, cfg.ransac_iterations, cfg.ransac_inlier_m, rng,
                                                   cfg.max_tilt_deg)
        num_inliers = int(inliers.sum())
        tilt = float(np.rad2deg(np.arccos(np.clip(normal[1], -1, 1))))
        if cfg.align_tilt and tilt <= float(cfg.max_tilt_deg) and spread >= float(cfg.min_spread_m):
            R = quat.to_matrix(quat.from_two_vectors(normal, [0.0, 1.0, 0.0]))
            tilt_mode = 'full'
    elif not fallback:
        raise ValueError(f'floor.tilt_method が不正です: {method}')
    if cfg.flip_facing:
        R = np.diag([-1.0, 1.0, -1.0]) @ R   # Y 軸まわりに 180 度

    if fallback:
        # 低速の点が足りない: 全フレームの接地点の低いほう 10% を床とみなす
        heights = (pts[observed].reshape(-1, 3) @ R.T)[:, 1]
        heights = heights[heights <= np.percentile(heights, 10)]
    else:
        heights = (cand[inliers] @ R.T)[:, 1]
    y_off = -float(np.median(heights))

    pelvis = kin.root_pos @ R.T
    origin = str(cfg.horizontal_origin)
    if origin == 'first_frame':
        xz = pelvis[0, [0, 2]]
    elif origin == 'mean':
        xz = pelvis[:, [0, 2]].mean(0)
    elif origin == 'none':
        xz = np.zeros(2)
    else:
        raise ValueError(f'floor.horizontal_origin が不正です: {origin}')
    offset = np.zeros((T, 3))
    offset[:, 0], offset[:, 1], offset[:, 2] = -xz[0], y_off, -xz[1]
    return FloorResult(R, offset, normal, tilt, len(cand), num_inliers, fallback, tilt_mode != 'none',
                       spread, tilt_mode)
