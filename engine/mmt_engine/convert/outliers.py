"""ステージ1b: 慣性と重力で、明らかに間違ったフレーム（外れ）を見つけて置き換える。

単眼推定には、腕や脚が数フレームだけ別の位置へ飛ぶ・体の向きが一瞬反転する・骨盤が数フレームだけ
浮く／沈む／前後へ飛ぶ、といった誤りが混ざる。平滑化の前に次の 2 つで見つけ、前後から置き換える。

* 慣性（姿勢）: 部位ごと（胴・脚・頭・左腕・右腕）に、関節の位置と向き（軸の先の点）を部位の付け根の座標系で表し、
  各フレームを前後の外れていないフレームに当てはめた 3 次式で予測する。予測との差が、その動画の推定のぶれ
  （差の中央値）の threshold 倍を超えたフレームを外れとし、外れを除いて予測し直すことを繰り返す。外れのそばの
  正しいフレーム（片側からの予測によく合う）、本当に速く動いて止まった動き（差の向きがそろわない）、区間の外から
  速さを保って出入りする本当の速い動きは外れにしない。
* 重力（重心）: 重心の加速度は重力と床から受ける力（押すだけ・摩擦の範囲・体重の数倍まで）で出せる範囲に入る。
  この範囲に入る重心の軌道のうち推定に最も近いものを ADMM で求め、それから大きく離れるフレームを外れにする
  （本当のジャンプは重力だけで動く放物線なので残る）。

置き換えは、外れた部位の関節の回転を前後から球面線形補間し、骨盤の位置は重力の条件を満たす軌道へ置き直す。
胴・脚・重心が外れたフレームは、以降の床・接地・奥行き・傾きの推定に使わない（valid = False）。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..body.smpl import forward_kinematics
from ..core import quat
from ..core.filters import frames_for_fps, median_time, runs
from .com import com_weights

GRAVITY = 9.8
MAX_SKIP = 5   # 何フレーム離れた外側まで予測に使うか

# (名前, 付け根の関節（None = 骨盤の位置・世界の向き）, 位置を使う関節, 向きを使う関節, 置き換える関節,
#  観測として使わなくするか)
PARTS = (
    ('torso', None, (3, 6, 9), (0, 9), tuple(range(24)), True),
    ('legs', 0, (4, 5, 7, 8, 10, 11), (7, 8), (1, 2, 4, 5, 7, 8, 10, 11), True),
    ('head', 9, (12, 15), (15,), (12, 15), False),
    ('left_arm', 9, (16, 18, 20, 22), (20,), (13, 16, 18, 20, 22), False),
    ('right_arm', 9, (17, 19, 21, 23), (21,), (14, 17, 19, 21, 23), False),
)
PART_LABELS = dict(torso='胴', legs='脚', head='頭', left_arm='左腕', right_arm='右腕', com='重心')


@dataclass
class OutlierResult:
    quats: np.ndarray            # (T, J, 4) 置き換えた後の回転
    root_pos: np.ndarray         # (T, 3) 置き換えた後の骨盤の位置
    valid: np.ndarray            # (T,) 観測として使うフレーム
    flags: dict                  # 部位の名前・'com' → (T,) bool 置き換えたフレーム
    scores: dict                 # 同じキー → (T,) 外れの度合い（1 を超えると外れ）
    skipped: dict = field(default_factory=dict)    # 使わなかった検出 → 理由
    long_runs: list = field(default_factory=list)  # (名前, 開始, 終了) 長すぎて置き換えなかった外れ
    enabled: bool = True

    @property
    def replaced(self):
        out = np.zeros(len(self.valid), bool)
        for f in self.flags.values():
            out |= f
        return out

    @property
    def unobserved(self):
        """(T,) 観測として使わなくしたフレーム（胴・脚・重心の外れ）。"""
        out = np.zeros(len(self.valid), bool)
        for name, *_, unobserve in PARTS:
            if unobserve and name in self.flags:
                out |= self.flags[name]
        if 'com' in self.flags:
            out |= self.flags['com']
        return out

    def info(self):
        T = len(self.valid)
        return dict(
            enabled=self.enabled,
            frames={k: int(v.sum()) for k, v in self.flags.items()},
            replaced_frames=int(self.replaced.sum()),
            unobserved_frames=int(self.unobserved.sum()),
            long_runs=[[n, int(s), int(e)] for n, s, e in self.long_runs],
            skipped=dict(self.skipped),
            ratio=round(float(self.replaced.sum()) / max(T, 1), 4))


# ================================================================ 慣性（姿勢）

def part_points(G, P, base, joints, axes_of, axis_length):
    """(T, C, 3) 部位の点: 関節の位置と、関節の x・z 軸の先の点を、付け根の座標系で表したもの。

    base が None なら骨盤からの相対位置を世界の向きのまま使う（体全体の向きの反転も外れとして見つかる）。
    """
    if base is None:
        origin, R = P[:, 0], None
    else:
        origin, R = P[:, base], G[:, base]
    pts = [P[:, list(joints)] - origin[:, None]]
    for j in axes_of:
        pts.append(np.stack([G[:, j, :, 0], G[:, j, :, 2]], axis=1) * axis_length)
    pts = np.concatenate(pts, axis=1)
    if R is not None:
        pts = np.einsum('tab,tca->tcb', R, pts)      # Rᵀ p
    return pts


def _nearest_slots(idx, T, skip):
    """各フレーム t について、t より前の good なフレームのうち近いほうから skip−1 個を飛ばした残りの末尾
    （idx の位置）と、t より後ろの同じく先頭。"""
    t = np.arange(T)
    left_end = np.searchsorted(idx, t, side='left') - (skip - 1)
    right_start = np.searchsorted(idx, t, side='right') + (skip - 1)
    return np.maximum(left_end, 0), np.minimum(right_start, len(idx))


@dataclass
class Prediction:
    value: np.ndarray        # (T, C, 3)
    ok: np.ndarray           # (T,) 予測できた
    distance: np.ndarray     # (T,) 予測に使った最も近いフレームまでの間隔
    fit_rms: np.ndarray = None   # (T, C) 3 次式の当てはまりの悪さ（両側から）


def neighbour_prediction(y, good, per_side=3, skip=1):
    """各フレームを、前後の good なフレーム（近いほうから skip−1 個ずつ飛ばして各側 per_side 個）に
    当てはめた 3 次式で予測する。good が 4 つ未満なら None。"""
    y = np.asarray(y, np.float64)
    T = len(y)
    idx = np.flatnonzero(good)
    n = len(idx)
    if n < 4:
        return None
    k, h = int(per_side), max(int(skip), 1)
    t = np.arange(T)
    left_end, right_start = _nearest_slots(idx, T, h)
    nl = np.minimum(left_end, k)
    nr = np.minimum(n - right_start, k)
    slots = np.arange(2 * k)
    col = np.where(slots[None] < nl[:, None],
                   left_end[:, None] - nl[:, None] + slots[None],
                   right_start[:, None] + slots[None] - nl[:, None])
    used = slots[None] < (nl + nr)[:, None]
    frames = idx[np.clip(col, 0, n - 1)]
    ok = (nl >= 2) & (nr >= 2)
    gap = np.where(ok, idx[np.clip(right_start, 0, n - 1)] - idx[np.clip(left_end - 1, 0, n - 1)], 0)
    dt = (frames - t[:, None]) / np.maximum(gap, 1)[:, None] * used
    V = np.stack([np.ones_like(dt), dt, dt ** 2, dt ** 3], axis=-1) * used[..., None]
    A = np.einsum('tsi,tsj->tij', V, V)
    A[~ok] = np.eye(4)
    A += np.eye(4) * 1e-9
    H = np.linalg.solve(A, np.swapaxes(V, 1, 2))
    yn = y[frames]
    beta = np.einsum('tis,ts...->ti...', H, yn)
    resid = (yn - np.einsum('tsi,ti...->ts...', V, beta)) * used[:, :, None, None]
    dof = np.maximum(used.sum(1) - 4, 1)
    fit_rms = np.sqrt(np.einsum('tscd->tc', resid ** 2) / dof[:, None])
    return Prediction(beta[:, 0], ok, (gap + 1) // 2, fit_rms)


def side_prediction(y, good, side, skip=1, per_side=3):
    """各フレームを、片側（side = −1: 前、+1: 後ろ）の good なフレームに直線を当てはめて予測する。"""
    y = np.asarray(y, np.float64)
    T = len(y)
    idx = np.flatnonzero(good)
    n = len(idx)
    t = np.arange(T)
    k = int(per_side)
    left_end, right_start = _nearest_slots(idx, T, max(int(skip), 1))
    if side < 0:
        slots = left_end[:, None] - k + np.arange(k)[None]
    else:
        slots = right_start[:, None] + np.arange(k)[None]
    used = (slots >= 0) & (slots < n)
    frames = idx[np.clip(slots, 0, max(n - 1, 0))] if n else np.zeros_like(slots)
    dt = np.where(used, (frames - t[:, None]).astype(np.float64), 0.0)
    m = used.sum(1)
    ok = m >= 2
    near = np.where(used, np.abs(dt), np.inf).min(1)
    mm = np.maximum(m, 1)
    mean = dt.sum(1) / mm
    var = np.where(used, (dt - mean[:, None]) ** 2, 0.0).sum(1)
    w = np.where(used, 1.0 / mm[:, None]
                 - (dt - mean[:, None]) * mean[:, None] / np.maximum(var, 1e-9)[:, None], 0.0)
    w = np.where(ok[:, None], w, 0.0)
    pred = np.einsum('ts,ts...->t...', w, y[frames]) if n else np.zeros_like(y)
    return Prediction(pred, ok, np.where(ok, near, 0).astype(int))


def coherent(res):
    """外れの区間の予測との差 (n, C, 3) の向きがそろっているか（段差の動きは前後で反対を向くのでそろわない）。"""
    if len(res) < 2:
        return True
    norms = np.linalg.norm(res, axis=-1)
    c = int(np.argmax(norms.sum(0)))
    return float(np.linalg.norm(res[:, c].sum(0))) >= 0.5 * float(norms[:, c].sum())


def smooth_motion(y, good, s, e, floor, max_bulge=0.8, max_jump=(1.2, 2.6), outside=2, min_bulge=3.0):
    """外れの区間 [s, e] が、前後から速さを保ったまま出入りする本当の動き（腕を振って戻す等）か。

    推定の破綻は入口・出口の 1 フレームで飛ぶが、本当の動きは区間の外から少しずつ動き始めて少しずつ戻る。
    jump（入口・出口の動き ÷ 区間の外の動き）が小さい、または bulge（入口・出口の動き ÷ 区間の中で前後を
    つないだ直線から最も離れた量）も小さければ本当の動きとする。決められないときは False（外れのまま）。
    """
    idx = np.flatnonzero(good)
    before, after = idx[idx < s], idx[idx > e]
    if len(before) <= outside or len(after) <= outside:
        return False
    a, b = before[-1], after[0]
    frac = (np.arange(s, e + 1) - a) / float(b - a)
    line = y[a][None] + frac[:, None, None] * (y[b] - y[a])[None]
    dev = np.linalg.norm(y[s:e + 1] - line, axis=-1).max(axis=0)
    c = int(np.argmax(dev))
    if dev[c] < min_bulge * floor[c]:
        return False

    def speeds(frames):
        frames = np.asarray(frames)
        return np.linalg.norm(np.diff(y[frames, c], axis=0), axis=-1) / np.diff(frames)

    j_in = np.linalg.norm(y[s, c] - y[a, c]) / (s - a)
    j_out = np.linalg.norm(y[b, c] - y[e, c]) / (b - e)
    v_in = max(speeds(before[-outside - 1:]).max(), floor[c])
    v_out = max(speeds(after[:outside + 1]).max(), floor[c])
    jump = max(j_in / v_in, j_out / v_out)
    if jump < max_jump[0]:
        return True
    if e > s:
        inner = speeds([s, s + 1])[0], speeds([e - 1, e])[0]
        jump = max(j_in / max(v_in, inner[0]), j_out / max(v_out, inner[1]))
    bulge = max(j_in * (s - a), j_out * (b - e)) / dev[c]
    return bool(bulge < max_bulge and jump < max_jump[1])


class InertiaScores:
    """前後のフレームからの予測との差 ÷ しきい値。

    * 両側から（3 次式）: 精度が高いので外れを見つけるのに使う
    * 片側から（直線）: 外れのそばの正しいフレームは、外れと反対の側からの予測によく合う（veto 以下なら外れにしない）
    しきい値は予測の種類・間隔・点ごとに、最初の予測の差の中央値の threshold 倍（min_m 以上）。両側からの
    予測は、予測に使ったフレームに 3 次式が当てはまらないときはしきい値を残差の reliability 倍まで上げる。
    """

    reliability = 5.0
    veto = 0.5
    one_sided_margin = 1.5

    def __init__(self, y, observed, threshold, min_m, max_skip, per_side=3, turn=None):
        self.y = np.asarray(y, np.float64)
        self.per_side = int(per_side)
        self.max_skip = max(int(max_skip), 1)
        T = len(self.y)
        if turn is None:
            self.skip_limit = np.full(T, self.max_skip)
        else:
            span = (np.pi / 2) / np.maximum(np.asarray(turn, np.float64), 1e-9)
            self.skip_limit = np.clip(np.floor(span).astype(int) - self.per_side + 1, 1, self.max_skip)
        mins = np.broadcast_to(np.asarray(min_m, np.float64), self.y.shape[1:2])
        self.taus = {}
        for kind in (0, -1, 1):
            for h in range(1, self.max_skip + 1):
                p = self._predict(observed, kind, h)
                use = observed & p.ok & (self.skip_limit >= h) if p is not None else None
                if use is None or use.sum() < 2 * self.per_side:
                    break
                r = np.linalg.norm(self.y - p.value, axis=-1)
                self.taus[kind, h] = np.maximum(float(threshold) * np.median(r[use], axis=0), mins)

    def _predict(self, good, kind, h):
        if kind == 0:
            return neighbour_prediction(self.y, good, self.per_side, h)
        return side_prediction(self.y, good, kind, h, self.per_side)

    def _kind(self, good, kind):
        T = len(self.y)
        hs = [h for (kd, h) in self.taus if kd == kind]
        score, avail = np.zeros(T), np.zeros(T, bool)
        if not hs:
            return score, avail
        table = np.stack([self.taus[kind, h] for h in hs])
        for h in hs:
            p = self._predict(good, kind, h)
            if p is None:
                break
            ok = p.ok & ((self.skip_limit >= h) | (h == 1))
            tau = table[np.clip(p.distance, 1, len(hs)) - 1]
            if kind == 0:
                tau = np.maximum(tau, self.reliability * p.fit_rms)
            r = (np.linalg.norm(self.y - p.value, axis=-1) / tau).max(axis=1)
            score = np.where(ok, np.maximum(score, r), score)
            avail |= ok
        return score, avail

    def full_side(self, observed, side):
        idx = np.flatnonzero(observed)
        need = self.per_side + self.max_skip - 1
        left_end, right_start = _nearest_slots(idx, len(self.y), 1)
        return (left_end >= need) if side < 0 else (len(idx) - right_start >= need)

    def explained(self, observed):
        out = np.zeros(len(self.y), bool)
        for side in (-1, 1):
            s, _ = self._kind(observed, side)
            out |= self.full_side(observed, side) & (s <= self.veto)
        return out

    def _one_side(self, good, side):
        p = self._predict(good, side, 1)
        tau = np.stack([self.taus[side, h] for h in range(1, self.max_skip + 1) if (side, h) in self.taus])
        tau = tau[np.clip(p.distance, 1, len(tau)) - 1]
        r = (np.linalg.norm(self.y - p.value, axis=-1) / tau).max(axis=1)
        return np.where(p.ok, r, 0.0)

    def grow(self, observed, flags, ends, max_len):
        """外れの区間の隣で、区間の外側からの予測に合わないフレーム。"""
        T = len(self.y)
        good = observed & ~flags
        left, right = np.zeros(T, bool), np.zeros(T, bool)
        for s, e in runs(flags):
            if e - s + 1 < max_len:
                left[max(s - 1, 0)] = True
                right[min(e + 1, T - 1)] = True
        left &= good
        right &= good
        past, future = self._one_side(good, -1), self._one_side(good, 1)
        full_past, full_future = ends
        grow = (left & full_past & (past > 1.0)) | (right & full_future & (future > 1.0))
        tau = self.taus[0, 1]
        for side, mask, full in ((1, left & ~full_past, full_future), (-1, right & ~full_future, full_past)):
            if not mask.any():
                continue
            p = self._predict(good, side, 1)
            for t in np.flatnonzero(mask & full & p.ok):
                edge = t + 1 if side > 0 else t - 1
                d_out = (np.linalg.norm(self.y[t] - self.y[edge], axis=-1) / tau).max()
                d_in = (np.linalg.norm(self.y[t] - p.value[t], axis=-1) / tau).max()
                grow[t] |= d_out < d_in and d_in > 1.0
        return grow

    def score(self, good, explained, ends):
        full_past, full_future = ends
        s2, a2 = self._kind(good, 0)
        out = np.where(a2 & full_past & full_future & ~explained, s2, 0.0)
        for side, full, other in ((1, full_future, full_past), (-1, full_past, full_future)):
            end = full & ~other
            if end.any():
                one, ok = self._kind(good, side)
                out = np.where(end & ok, one / self.one_sided_margin, out)
        return out


def _span(T, s, e):
    out = np.zeros(T, bool)
    out[s:e + 1] = True
    return out


def drop_long_runs(flags, max_len, name, long_runs):
    """max_len より長く続く外れは置き換えない（前後どちらが正しいか決められない）。"""
    flags = flags.copy()
    for s, e in runs(flags):
        if e - s + 1 > max_len:
            flags[s:e + 1] = False
            long_runs.append((name, s, e))
    return flags


def detect_part(y, observed, threshold, min_m, max_skip, max_len, per_side=3, turn=None, long_runs=None,
                max_iter=20, keep_smooth=False):
    """部位の点 y (T, C, 3) の外れ。戻り値: (外れ (T,) bool, 外れの度合い (T,))。"""
    T = len(y)
    flags = np.zeros(T, bool)
    if observed.sum() < 4 * per_side:
        return flags, np.zeros(T)
    sc = InertiaScores(y, observed, threshold, min_m, max_skip, per_side, turn)
    if not sc.taus:
        return flags, np.zeros(T)
    explained = sc.explained(observed)
    ends = (sc.full_side(observed, -1), sc.full_side(observed, 1))
    score = sc.score(observed, explained, ends)
    dropped = []
    for _ in range(max_iter):
        if flags.any():
            new = observed & (sc.score(observed & ~flags, explained, ends) > 1.0)
        else:
            new = observed & (score > 1.0)
        dropped = []
        new = drop_long_runs(new, max_len, '', dropped)
        if np.array_equal(new, flags):
            break
        flags = new
    if long_runs is not None:
        long_runs.extend((s, e) for _, s, e in dropped)
    for _ in range(max_len):
        grow = sc.grow(observed, flags, ends, max_len)
        if not grow.any():
            break
        flags |= grow
    if flags.any():
        p = neighbour_prediction(y, observed & ~flags, per_side, 1)
        if p is not None:
            res = y - p.value
            for s, e in runs(flags):
                if p.ok[s:e + 1].all() and not coherent(res[s:e + 1]):
                    flags[s:e + 1] = False
    if keep_smooth and flags.any():
        idx = np.flatnonzero(observed)
        floor = 1.5 * np.median(np.linalg.norm(np.diff(y[idx], axis=0), axis=-1), axis=0)
        # 動きの頂点（一瞬止まる所）で区間が 2 つに割れることがあるので、間が 2 フレーム以下の区間はまとめても調べる
        groups = []
        for s, e in runs(flags):
            if groups and s - groups[-1][-1][1] - 1 <= 2:
                groups[-1].append((s, e))
            else:
                groups.append([(s, e)])
        for group in groups:
            s, e = group[0][0], group[-1][1]
            if smooth_motion(y, observed & ~flags & ~_span(len(y), s, e), s, e, floor):
                flags[s:e + 1] = False
                continue
            for s, e in group if len(group) > 1 else ():
                if smooth_motion(y, observed & ~flags, s, e, floor):
                    flags[s:e + 1] = False
    return flags, np.maximum(score, sc.score(observed & ~flags, explained, ends))


def turn_rate(G, joints, span=3, window=11):
    """(T,) 関節 joints の大域回転の 1 フレームあたりの回転角 [rad] の最大（ぶれと外れに強い見積もり）。"""
    T = len(G)
    if T < 2:
        return np.zeros(T)
    t = np.arange(T)

    def rate(d):
        a, b = np.clip(t - d, 0, T - 1), np.clip(t + d, 0, T - 1)
        out = np.zeros(T)
        for j in joints:
            R = G[:, j]
            c = (np.einsum('tab,tab->t', R[b], R[a]) - 1.0) / 2.0
            out = np.maximum(out, np.arccos(np.clip(c, -1.0, 1.0)) / np.maximum(b - a, 1))
        return median_time(out, window)

    return np.minimum(rate(1), rate(span))


# ================================================================ 重力（重心）

def second_difference(T):
    """(T−2, T) 2 階差分。"""
    return sparse.diags([np.ones(T - 2), -2.0 * np.ones(T - 2), np.ones(T - 2)], [0, 1, 2],
                        shape=(T - 2, T), format='csr')


def project_gravity_cone(a, mu, max_force):
    """(N, 3) 加速度 a（Y 上向き）を、a + g·ŷ が床から受ける力の範囲（上向き ≤ max_force・水平 ≤ μ × 上向き）に
    入るよう、最も近い点へ射影する。"""
    f = np.array(a, np.float64, copy=True)
    f[:, 1] += GRAVITY
    t = f[:, 1]
    v = f[:, [0, 2]]
    r = np.linalg.norm(v, axis=1)
    inside = r <= mu * t
    polar = mu * r <= -t
    tp = (t + mu * r) / (1.0 + mu * mu)
    t_new = np.where(inside, t, np.where(polar, 0.0, tp))
    r_new = np.where(inside, r, np.where(polar, 0.0, mu * tp))
    cap = t_new > max_force
    t_new = np.where(cap, max_force, t_new)
    r_new = np.where(cap, np.minimum(r, mu * max_force), r_new)
    v_new = v * (r_new / np.maximum(r, 1e-12))[:, None]
    return np.stack([v_new[:, 0], t_new - GRAVITY, v_new[:, 1]], axis=1)


def _basis(axis):
    """奥行きの軸を 1 行目とする正規直交基底 (3, 3)。"""
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = np.cross(a, helper)
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)])


class GravityFit:
    """重力の条件を満たす重心の軌道のうち、推定の重心に（重み付きで）最も近いものを ADMM で求める。

        最小化 Σ_t w_t |Σ^{−1/2} (x_t − y_t)|²   条件 (D² x)_t · fps² ∈ 床から受ける力の範囲 − g·ŷ

    Σ は奥行きの軸の向きだけ大きい。重みを変えて何度も解くので、双対変数を持ち越す。
    """

    def __init__(self, fps, depth_axis, cfg, sigma=0.02, sigma_depth=0.1, rho=0.3, cancel=None):
        self.fps = float(fps)
        self.Q = _basis(depth_axis)
        self.sig2 = np.array([sigma_depth, sigma, sigma]) ** 2
        self.mu = float(cfg.friction)
        self.max_force = float(cfg.max_force_g) * GRAVITY
        self.iterations = int(cfg.iterations)
        self.rho = float(rho) / (sigma * sigma)
        self.z = self.u = None
        self.cancel = cancel

    def solve(self, y, w):
        T = len(y)
        D = second_difference(T)
        DtD = (D.T @ D).tocsc()
        yq = y @ self.Q.T
        lus = [splu((sparse.diags(w / s2 + 1e-9) + self.rho * DtD).tocsc()) for s2 in self.sig2]
        if self.z is None or len(self.z) != T - 2:
            self.z = project_gravity_cone(D @ y * self.fps ** 2, self.mu, self.max_force)
            self.u = np.zeros_like(self.z)
        z, u = self.z, self.u
        s = self.fps ** 2
        for it in range(self.iterations):
            if self.cancel is not None and it % 50 == 0:
                self.cancel()
            target = ((z - u) / s) @ self.Q.T
            xq = np.stack([lus[i].solve(w * yq[:, i] / self.sig2[i] + self.rho * (D.T @ target[:, i]))
                           for i in range(3)], axis=1)
            x = xq @ self.Q
            ax = D @ x * s
            z = project_gravity_cone(ax + u, self.mu, self.max_force)
            u = u + ax - z
        self.z, self.u = z, u
        return x


def com_channels(com, axis):
    """(T, 2, 3) 重心の奥行きの軸の成分と、それ以外（上下・左右）の成分。"""
    d = (com @ axis)[:, None] * axis
    return np.stack([d, com - d], axis=1)


def detect_com(com, observed, fps, depth_axis, cfg, max_len, long_runs=None, max_iter=8, huber=0.3,
               max_skip=MAX_SKIP, per_side=3, cancel=None):
    """重心 (T, 3)（重力の向きにそろえた座標 [m]）の外れ。戻り値: (外れ, 度合い, 重力の条件を満たす軌道)。

    1. 慣性で外れの候補を見つける  2. 候補を除いて重力の条件を満たす軌道を求め、差がしきい値を超えるフレームを
    外れとし、Huber の重みで解き直す  3. 2 の外れを含む候補の区間は区間全体を外れにする（本当のジャンプなど、
    2 の外れを含まない候補は外れにしない）。
    """
    T = len(com)
    flags = np.zeros(T, bool)
    if observed.sum() < 8 or T < 8:
        return flags, np.zeros(T), np.array(com, np.float64, copy=True)
    axis = np.asarray(depth_axis, np.float64) / np.linalg.norm(depth_axis)
    k = float(cfg.threshold)
    mins = (float(cfg.min_depth_m), float(cfg.min_m))
    seeds, _ = detect_part(com_channels(com, axis), observed, k, mins, max_skip, max_len, per_side,
                           long_runs=long_runs)
    fit = GravityFit(fps, axis, cfg, sigma=mins[1] / 2.0, sigma_depth=mins[0] / 2.0, cancel=cancel)
    w = (observed & ~seeds).astype(np.float64)
    flags = seeds
    tau = x = score = None
    for _ in range(max_iter):
        x = fit.solve(com, w)
        r = com - x
        rd = r @ axis
        ro = np.linalg.norm(r - rd[:, None] * axis, axis=1)
        if tau is None:
            ok = observed & ~seeds
            tau = (max(k * float(np.median(np.abs(rd[ok]))), mins[0]),
                   max(k * float(np.median(ro[ok])), mins[1]))
        score = np.maximum(np.abs(rd) / tau[0], ro / tau[1])
        new = observed & (score > 1.0)
        w_new = np.where(observed & ~new, np.minimum(1.0, huber / np.maximum(score, 1e-9)), 0.0)
        if np.array_equal(new, flags) and np.allclose(w_new, w, atol=0.02):
            break
        flags, w = new, w_new
    for s, e in runs(seeds):
        if flags[s:e + 1].any():
            flags[s:e + 1] = True
    return flags, score, x


# ================================================================ 置き換え

def interpolate_rotations(q, mask):
    """(T, J, 4) の回転の mask (T, J) のフレームを、関節ごとに前後から球面線形補間する（端は最も近い値）。"""
    q = np.array(q, np.float64, copy=True)
    T = len(q)
    for j in range(q.shape[1]):
        for s, e in runs(mask[:, j]):
            a, b = s - 1, e + 1
            if a < 0 and b >= T:
                continue
            if a < 0:
                q[s:e + 1, j] = q[b, j]
            elif b >= T:
                q[s:e + 1, j] = q[a, j]
            else:
                frac = (np.arange(s, e + 1) - a) / float(b - a)
                q[s:e + 1, j] = quat.slerp(np.broadcast_to(q[a, j], (len(frac), 4)),
                                           np.broadcast_to(q[b, j], (len(frac), 4)), frac)
    return quat.make_continuous(q)


def fill_min_accel(y, mask):
    """(T, 3) y の mask のフレームを、加速度の二乗和が最小になるように前後からつなぐ。"""
    y = np.array(y, np.float64, copy=True)
    T = len(y)
    known = ~np.asarray(mask, bool)
    if not mask.any() or known.sum() < 2 or T < 3:
        return y
    D = second_difference(T).tocsc()
    Dn, Dk = D[:, np.flatnonzero(~known)], D[:, np.flatnonzero(known)]
    A = (Dn.T @ Dn + sparse.identity(Dn.shape[1]) * 1e-12).tocsc()
    y[~known] = splu(A).solve(-(Dn.T @ (Dk @ y[known])))
    return y


def remove_outliers(quats, root_pos, valid, rest_joints, parents, fps, cfg, gravity_rotation=None,
                    depth_axis=None, cancel=None):
    """ステージ1b。quats (T, J, 4)・root_pos (T, 3) [m] はステージ1の出力（Y 上向き）。

    gravity_rotation: (3, 3) 床の傾きの補正（重力の向きを決める）/ depth_axis: 補正後の座標での奥行きの軸。
    """
    quats = quat.make_continuous(np.asarray(quats, np.float64))
    root_pos = np.asarray(root_pos, np.float64)
    observed = np.asarray(valid, bool)
    T = len(quats)
    R = np.eye(3) if gravity_rotation is None else np.asarray(gravity_rotation, np.float64)
    axis = np.array([0.0, 0.0, 1.0]) if depth_axis is None else np.asarray(depth_axis, np.float64)
    names = [p[0] for p in PARTS] + ['com']
    empty = {n: np.zeros(T, bool) for n in names}
    if not cfg.enabled or T < 8:
        return OutlierResult(quats, root_pos, observed.copy(), empty, {n: np.zeros(T) for n in names},
                             enabled=False)

    max_len = int(np.floor(float(cfg.max_run_sec) * fps))
    max_ratio = float(cfg.max_ratio)
    neighbours = frames_for_fps(cfg.pose.neighbours, fps)
    max_skip = frames_for_fps(MAX_SKIP, fps)
    turn_span, turn_window = frames_for_fps(3, fps), frames_for_fps(11, fps, odd=True)
    n_obs = max(int(observed.sum()), 1)
    flags, scores, skipped, long_runs = dict(empty), {n: np.zeros(T) for n in names}, {}, []

    def accept(name, f, dropped=()):
        long_runs.extend((name, s, e) for s, e in dropped)
        f = drop_long_runs(f, max_len, name, long_runs)
        if f.sum() > max_ratio * n_obs:
            skipped[name] = (f'{PART_LABELS[name]}の外れが {f.sum() / n_obs * 100:.0f}% と多すぎるので、'
                             '置き換えに使いませんでした')
            return np.zeros(T, bool)
        return f

    G, P = forward_kinematics(quats, root_pos, rest_joints, parents)
    joint_mask = np.zeros(quats.shape[:2], bool)
    if cfg.pose.enabled:
        for name, base, joints, axes_of, replace_joints, _ in PARTS:
            if cancel is not None:
                cancel()
            y = part_points(G, P, base, joints, axes_of, float(cfg.pose.axis_length_m))
            dropped = []
            f, scores[name] = detect_part(y, observed, float(cfg.pose.threshold), float(cfg.pose.min_m),
                                          max_skip, max_len, neighbours,
                                          turn_rate(G, axes_of, turn_span, turn_window), dropped,
                                          keep_smooth=bool(cfg.pose.keep_smooth))
            flags[name] = accept(name, f, dropped)
            joint_mask[:, list(replace_joints)] |= flags[name][:, None]
        quats = interpolate_rotations(quats, joint_mask)
        G, P = forward_kinematics(quats, root_pos, rest_joints, parents)

    w_com = com_weights(P.shape[1])
    com_cam = np.einsum('tjc,j->tc', P, w_com)
    offset = com_cam - root_pos
    com = com_cam @ R.T
    fit = None
    reposition = flags['torso'].copy()
    if cfg.gravity.enabled:
        use = observed & ~flags['torso']
        dropped = []
        f, scores['com'], fit = detect_com(com, use, fps, axis, cfg.gravity, max_len, dropped,
                                           max_skip=max_skip, per_side=frames_for_fps(3, fps), cancel=cancel)
        flags['com'] = accept('com', f, dropped)
        reposition |= flags['com']
    if reposition.any() and not reposition.all():
        good = ~reposition
        t = np.arange(T)
        target = fill_min_accel(com, reposition)
        if fit is not None and 'com' not in skipped:
            resid = com - fit
            inner = (t > t[good][0]) & (t < t[good][-1])
            bridged = fit + np.stack([np.interp(t, t[good], resid[good, i]) for i in range(3)], axis=1)
            target = np.where(inner[:, None], bridged, target)
        new_root = target @ R - offset
        root_pos = np.where(reposition[:, None], new_root, root_pos)

    result = OutlierResult(quats, root_pos, observed.copy(), flags, scores, skipped=skipped,
                           long_runs=long_runs)
    result.valid = observed & ~result.unobserved
    return result
