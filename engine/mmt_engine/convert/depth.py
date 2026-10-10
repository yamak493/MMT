"""奥行きのぶれへの対策（ステージ6の速度と、ステージ6b: 奥行きの再構成）。

単眼推定ではカメラの奥行き方向の位置が毎フレーム数 cm ぶれ、ゆっくりずれていく。骨盤の奥行きがぶれると
床に着いている足も前後に動き、接地判定が途切れて足が滑る。そこで

* 接地判定の速度: 骨盤の奥行きをならした場合と、接地点の奥行きをならした場合の小さいほうを使う
* 奥行きの再構成: 骨盤の奥行きを「接地中の足首の奥行きが一定」「推定から大きく離れない」「前後の加速度が
  小さい」の重み付き最小二乗で求め直し、体全体を奥行きの軸に沿って平行移動する
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..body.smpl import ANKLES
from ..core import filters
from .floor import horizontal_speed


@dataclass
class DepthResult:
    axis: np.ndarray         # (3,) 奥行きの軸（内部座標の単位ベクトル）
    raw: np.ndarray          # (T,) 補正前の骨盤の奥行き
    depth: np.ndarray        # (T,) 求め直した骨盤の奥行き
    enabled: bool

    @property
    def correction(self):
        return self.depth - self.raw

    def apply(self, kin):
        if not self.enabled:
            return kin
        return kin.transformed(None, self.correction[:, None] * self.axis)


def depth_axis(floor_rotation):
    """カメラの奥行き方向を、床の補正の後の内部座標で表した単位ベクトル（補正前は Z 軸が光軸）。"""
    u = np.asarray(floor_rotation, np.float64)[:, 2]
    return u / np.linalg.norm(u)


def depth_jitter(root_pos, axis):
    """骨盤の位置 (T, 3) の細かいぶれの大きさ [奥行き, 横]（白色ノイズの 2 階差分は √6 倍になることから）。"""
    lateral = np.cross([0.0, 1.0, 0.0], axis)
    lateral /= np.linalg.norm(lateral)
    root_pos = np.asarray(root_pos, np.float64)
    if len(root_pos) < 3:
        return np.zeros(2)
    acc = np.diff(root_pos, 2, axis=0)
    return np.array([np.std(acc @ axis), np.std(acc @ lateral)]) / np.sqrt(6.0)


def _lowpass(x, fps, unit, cutoff_hz):
    return filters.one_euro(x / unit, fps, float(cutoff_hz), 0.0) * unit


def detection_speeds(kin, axis, fps, unit, cfg):
    """接地判定に使う接地点の水平速度 (T, 2, 2)。骨盤の奥行き・接地点の奥行きをならした場合の小さいほう。"""
    pts = kin.contact_points
    by_root = pts
    if cfg.detect_root_cutoff_hz > 0:
        d = kin.root_pos @ axis
        shift = _lowpass(d, fps, unit, cfg.detect_root_cutoff_hz) - d
        by_root = pts + shift[:, None, None, None] * axis
    by_point = pts
    if cfg.detect_foot_cutoff_hz > 0:
        d = pts @ axis
        by_point = pts + (_lowpass(d, fps, unit, cfg.detect_foot_cutoff_hz) - d)[..., None] * axis
    return np.minimum(horizontal_speed(by_root, fps), horizontal_speed(by_point, fps))


def solve_depth(raw, rel, segments, fps, contact_sigma, prior_sigma, accel_sigma, prior_weights=None):
    """骨盤の奥行き r (T,) を重み付き最小二乗で求める（単位は m）。

    未知数は r と接地区間ごとの足首の奥行き L。残差は
      接地: (r_t + rel_t − L) / contact_sigma   事前: (r_t − raw_t) / prior_sigma
      加速度: (r_{t−1} − 2 r_t + r_{t+1}) / (accel_sigma / fps²)
    """
    raw = np.asarray(raw, np.float64)
    T = len(raw)
    rows, cols, vals, rhs = [], [], [], []

    def add(terms, b):
        start = sum(len(r) for r in rhs)
        for c, v in terms:
            rows.append(np.arange(start, start + len(b)))
            cols.append(c)
            vals.append(np.broadcast_to(v, (len(b),)))
        rhs.append(b)

    t = np.arange(T)
    wp = np.full(T, 1.0 / prior_sigma)
    if prior_weights is not None:
        pw = np.asarray(prior_weights, np.float64)
        if pw.max(initial=0.0) > 0.0:
            wp = wp * pw
    add([(t, wp)], wp * raw)
    if T >= 3:
        wa = fps ** 2 / accel_sigma
        m = t[1:-1]
        add([(m - 1, wa), (m, -2.0 * wa), (m + 1, wa)], np.zeros(T - 2))
    wc = 1.0 / contact_sigma
    num_seg = 0
    for foot in range(2):
        for s, e in segments[foot]:
            f = np.arange(s, e + 1)
            add([(f, wc), (np.full(len(f), T + num_seg), -wc)], -wc * rel[f, foot])
            num_seg += 1
    b = np.concatenate(rhs)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(b), T + num_seg))
    x = spsolve((A.T @ A).tocsc(), A.T @ b)
    return np.asarray(x[:T])


def reconstruct_depth(kin, kin_pose, contact, axis, fps, unit, cfg, valid=None):
    """ステージ6b。kin_pose: 足首 → 骨盤の相対位置を求める姿勢（ならす前の姿勢。ならした姿勢では遊脚の動きが
    接地の前後に混ざり、歩いた距離が短くなる）。単独フレームの外れはメディアン（窓 5）で除く。"""
    raw = kin.root_pos @ axis
    if not cfg.reconstruct:
        return DepthResult(axis, raw, raw.copy(), False)
    rel = filters.median_time((kin_pose.joints[:, ANKLES] - kin_pose.root_pos[:, None]) @ axis, 5)
    depth = solve_depth(raw / unit, rel / unit, contact.segments, fps, float(cfg.contact_sigma_m),
                        float(cfg.prior_sigma_m), float(cfg.accel_sigma_m_per_s2),
                        None if valid is None else np.asarray(valid, np.float64)) * unit
    return DepthResult(axis, raw, depth, True)
