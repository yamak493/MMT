"""当たり判定（カプセル）と小さな回転の計算の部品。"""
import numpy as np

_EPS = 1e-9


def closest_points(p0, p1, q0, q1):
    """線分 p0–p1 と q0–q1 の最近点の組 (..., 3)（Ericson『Real-Time Collision Detection』5.1.9）。"""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.maximum(np.sum(d1 * d1, -1), _EPS)
    e = np.maximum(np.sum(d2 * d2, -1), _EPS)
    b, c, f = np.sum(d1 * d2, -1), np.sum(d1 * r, -1), np.sum(d2 * r, -1)
    denom = a * e - b * b
    general = denom > 1e-12 * a * e
    s = np.where(general, np.clip((b * f - c * e) / np.where(general, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return p0 + d1 * s[..., None], q0 + d2 * t[..., None]


def distance_to_segment(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / max(float(ab @ ab), _EPS), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=-1)


def rotation(rotvec):
    """回転ベクトル (3,) → 回転行列（ロドリゲスの公式）。"""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


def rotvec(R):
    """回転行列 → 回転ベクトル（180 度未満の回転）。"""
    theta = angle(R)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(theta)
    return w * (0.5 if s < 1e-9 else 0.5 * theta / s)


def cross(a, b):
    """np.cross と同じ（小さな配列ではこちらが速い）。"""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def inv_sqrt(M):
    lam, V = np.linalg.eigh(M)
    return (V / np.sqrt(lam)) @ V.T


def constrained_step(J, b, damping):
    """J δ ≥ b をなるべく満たす小さな δ（減衰付きのアクティブセット法）。

    満たされていない条件を等式として δ = Jᵀ (J Jᵀ + damping·I)⁻¹ b を求め、乗数が負の条件を外し、満たされていない
    条件を加えることを繰り返す。減衰で、同時には満たせない条件があっても回転が暴れない。
    """
    active = [int(c) for c in np.flatnonzero(b > 0.0)]
    delta = np.zeros(J.shape[1])
    tol = 1e-9 * max(1.0, float(np.abs(b).max(initial=0.0)))
    for _ in range(2 * len(b) + 2):
        if active:
            Ja = J[active]
            lam = np.linalg.solve(Ja @ Ja.T + damping * np.eye(len(active)), b[active])
            if (lam < -tol).any():
                active.pop(int(np.argmin(lam)))
                continue
            delta = Ja.T @ lam
        short = b - J @ delta
        short[active] = -np.inf
        c = int(np.argmax(short))
        if short[c] <= tol:
            break
        active.append(c)
    return delta


def toward_identity(R, step):
    """回転 R を単位回転の向きへ最大 step [rad] 戻した回転。"""
    rv = rotvec(R)
    a = float(np.linalg.norm(rv))
    return np.eye(3) if a <= step else rotation(rv * (1.0 - step / a))


def clamp_rotation(R, limit):
    if limit <= 0.0:
        return np.eye(3)
    if angle(R) <= limit:
        return R
    rv = rotvec(R)
    norm = float(np.linalg.norm(rv))
    return R if norm <= _EPS else rotation(rv * (limit / norm))
