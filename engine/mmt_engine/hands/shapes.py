"""手の形（指）: MediaPipe Hands の 21 点から手の形を 12 種に分け、指ボーンの回転を作る。

1. 指ごとの曲げ角から「伸び」「曲げ」の確率を出す（親指は指先が人差し指・中指の付け根から離れていることも見る）
2. 形ごとに、指のパターンに合う度合い（確率の幾何平均）を点にする（デフォルトは一定の点）
3. 見えているフレームだけで時間方向にならし、形を切り替えるたびに switch_cost を引いて、点の和が最大になる
   形の並びを選ぶ（Viterbi）。長く見えない区間はデフォルト、短く続く形は前後の形にまとめる
4. 形ごとの指の角度（presets・angles）から指ボーンのローカル回転を作る。曲げの軸はボーンの位置（手首・人指１・
   小指１から求めた手のひらの向きと各指ボーンの向き）から求めるので、モデルのローカル軸の設定に依らない

タイムラインの「手のポーズ」は、形が変わる所（キー）だけを持ち、指のキー（切り替えの始まりと終わりの 2 つ、
イーズイン・アウト）はアプリが形ごとの回転から作る。
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

from ..core import quat
from ..core.filters import label_runs, runs
from ..rig.skeleton import SIDES

SHAPES = ('default', 'thumb', 'index', 'middle', 'ring', 'pinky', 'fist', 'scissors', 'open', 'pistol', 'fox', 'ok')
LABELS = dict(default='デフォルト', thumb='親指', index='人差し指', middle='中指', ring='薬指', pinky='小指',
              fist='グー', scissors='チョキ', open='パー', pistol='ピストル', fox='きつね', ok='オッケー')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
# 形ごとの指（親指〜小指）のパターン: 1 = 伸び / 0 = 曲げ / None = どちらでもよい
# （人差し指・チョキは親指を曲げにして、親指を伸ばしたピストルと分ける）
PATTERNS = {
    'thumb': (1, 0, 0, 0, 0),
    'index': (0, 1, 0, 0, 0),
    'middle': (None, 0, 1, 0, 0),
    'ring': (None, 0, 0, 1, 0),
    'pinky': (None, 0, 0, 0, 1),
    'fist': (0, 0, 0, 0, 0),
    'scissors': (0, 1, 1, 0, 0),
    'open': (None, 1, 1, 1, 1),
    'pistol': (1, 1, None, 0, 0),   # 中指は伸びても曲げてもよい
    'fox': (0, 1, 0, 0, 1),
    'ok': (None, 0, 1, 1, 1),       # 親指と輪を作る人差し指は曲げ
}
FINGER_POINTS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


# ---- 指の曲げ → 確率 ----
def _angle_deg(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def finger_bends(landmarks):
    """指ごとの曲げ角の和 [度]（(..., 21, 3) → (..., 5)）。親指は MCP・IP の 2 関節、ほかは 3 関節。"""
    lm = np.asarray(landmarks, np.float64)
    out = []
    for f, idx in enumerate(FINGER_POINTS):
        pts = np.concatenate([lm[..., :1, :], lm[..., list(idx), :]], axis=-2)
        seg = np.diff(pts, axis=-2)
        bend = _angle_deg(seg[..., :-1, :], seg[..., 1:, :])
        out.append(bend[..., 1:].sum(-1) if f == 0 else bend.sum(-1))
    return np.stack(out, axis=-1)


def thumb_away(landmarks):
    lm = np.asarray(landmarks, np.float64)
    tip = lm[..., 4, :]
    d = np.min(np.stack([np.linalg.norm(tip - lm[..., j, :], axis=-1) for j in (5, 6, 9, 10)]), axis=0)
    width = np.linalg.norm(lm[..., 5, :] - lm[..., 17, :], axis=-1)
    return d / np.maximum(width, 1e-9)


def finger_probs(landmarks, cfg):
    bends = finger_bends(landmarks)
    s = max(float(cfg.softness_deg), 1e-6) / 4.0
    p_ext = expit((float(cfg.extended_below_deg) - bends) / s)
    p_curl = expit((bends - float(cfg.curled_above_deg)) / s)
    st = max(float(cfg.thumb_softness_deg), 1e-6) / 4.0
    sa = max(float(cfg.thumb_away_softness), 1e-6) / 4.0
    thumb = (expit((float(cfg.thumb_extended_below_deg) - bends[..., 0]) / st)
             * expit((thumb_away(landmarks) - float(cfg.thumb_away)) / sa))
    p_ext[..., 0] = thumb
    p_curl[..., 0] = 1.0 - thumb
    return p_ext, p_curl


def shape_scores(p_ext, p_curl, cfg):
    out = np.empty(p_ext.shape[:-1] + (len(SHAPES),))
    out[..., 0] = float(cfg.default_score)
    for k, shape in enumerate(SHAPES[1:], start=1):
        logs = [np.log(np.clip(p_ext[..., f] if want else p_curl[..., f], 1e-6, 1.0))
                for f, want in enumerate(PATTERNS[shape]) if want is not None]
        out[..., k] = np.exp(np.mean(logs, axis=0))
    return out


# ---- 時間方向 ----
def smooth_scores(scores, weights, sigma_frames):
    scores = np.asarray(scores, np.float64)
    w = np.asarray(weights, np.float64)
    if sigma_frames <= 0:
        out = scores.copy()
        out[w <= 0] = np.nan
        return out
    num = gaussian_filter1d(scores * w[:, None], sigma_frames, axis=0, mode='nearest')
    den = gaussian_filter1d(w, sigma_frames, mode='nearest')
    out = num / np.maximum(den, 1e-9)[:, None]
    out[den < 0.05] = np.nan
    return out


def viterbi(log_scores, switch_cost):
    ls = np.nan_to_num(np.asarray(log_scores, np.float64), nan=0.0)
    T, K = ls.shape
    if T == 0:
        return np.zeros(0, np.int64)
    back = np.zeros((T, K), np.int64)
    total = ls[0].copy()
    stay = np.arange(K)
    for t in range(1, T):
        best = int(np.argmax(total))
        switch = total[best] - switch_cost
        moved = switch > total
        back[t] = np.where(moved, best, stay)
        total = np.where(moved, switch, total) + ls[t]
    path = np.empty(T, np.int64)
    path[-1] = int(np.argmax(total))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def fill_missing(labels, missing_mask, max_gap, missing):
    labels = np.array(labels, np.int64, copy=True)
    for s, e in runs(missing_mask):
        if e - s + 1 <= max_gap:
            continue
        if missing == 'hold' and s > 0:
            labels[s:e + 1] = labels[s - 1]
        else:
            labels[s:e + 1] = 0
    return labels


def merge_short(labels, scores, min_len):
    labels = np.array(labels, np.int64, copy=True)
    if min_len <= 1:
        return labels
    while True:
        rs = label_runs(labels)
        if len(rs) < 2:
            return labels
        lengths = [e - s + 1 for s, e, _ in rs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            return labels
        s, e, _ = rs[i]
        cands = [rs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
        seg = scores[s:e + 1]

        def score(k):
            v = seg[:, k]
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else -np.inf

        labels[s:e + 1] = max(cands, key=score)


def classify(scores, weights, fps, cfg):
    smoothed = smooth_scores(scores, weights, float(cfg.smooth_sec) * fps)
    labels = viterbi(np.log(np.clip(smoothed, 1e-6, None)), float(cfg.switch_cost))
    missing = ~np.isfinite(smoothed).all(axis=1)
    labels = fill_missing(labels, missing, int(round(float(cfg.max_gap_sec) * fps)), cfg.missing_shape)
    return merge_short(labels, smoothed, int(round(float(cfg.min_hold_sec) * fps)))


def resample_labels(labels, fps_in, fps_out, num_frames):
    labels = np.asarray(labels)
    idx = np.clip(np.round(np.arange(num_frames) * fps_in / fps_out).astype(np.int64), 0, len(labels) - 1)
    return labels[idx]


def classify_analysis(analysis, cfg, fps, num_frames):
    """検出結果から、30fps の各フレームの形の番号 (T, 2) と、手が見えたフレームの割合。"""
    fps_in = float(analysis['fps'])
    lm = np.asarray(analysis['world'] if cfg.landmarks == 'world' else analysis['screen'], np.float64)
    presence = np.asarray(analysis['presence'], np.float64)
    valid = (presence >= float(cfg.min_presence)) & np.isfinite(lm).all(axis=(-1, -2))
    weights = np.where(valid, presence, 0.0)
    p_ext, p_curl = finger_probs(np.nan_to_num(lm), cfg)
    raw = shape_scores(p_ext, p_curl, cfg)
    labels = np.zeros((num_frames, 2), np.int64)
    for side in range(2):
        lab = classify(raw[:, side], weights[:, side], fps_in, cfg)
        labels[:, side] = resample_labels(lab, fps_in, fps, num_frames)
    visible = [round(float(valid[:, s].mean()), 3) if len(valid) else 0.0 for s in range(2)]
    return labels, visible


def label_events(labels):
    """形の番号の列 (T,) → 形が変わる所 [{frame, shape}]（最初のフレームを含む）。"""
    return [dict(frame=int(s), shape=SHAPES[int(k)]) for s, _, k in label_runs(labels)]


# ---- 指ボーンの回転 ----
def preset_angles(cfg):
    """形ごと・指ごとの [付け根, 第 2, 第 3, 開き] の角度 [度] (形の数, 5, 4)。"""
    table = np.zeros((len(SHAPES), len(FINGERS), 4))
    for k, shape in enumerate(SHAPES):
        states = list(cfg.presets[shape])
        for f, (finger, state) in enumerate(zip(FINGERS, states)):
            if state not in cfg.angles[finger]:
                raise ValueError(f'hands.presets.{shape} の「{state}」が不正です')
            table[k, f] = np.asarray(cfg.angles[finger][state], np.float64)
    return table


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp(v, d):
    return _unit(v - np.dot(v, d) * d)


def _bone_direction(skel, name, next_name, prev_dir):
    p = skel.internal(name)
    if next_name is not None and skel.has(next_name):
        d = skel.internal(next_name) - p
    else:
        tail = skel.tail_internal(name)
        d = tail - p if tail is not None else None
    if d is None or np.linalg.norm(d) < 1e-6:
        return prev_dir
    return _unit(d)


def build_rig(skel, side, cfg, warn=None):
    """片手の指ボーンと曲げ・開きの軸（内部座標）。手首・人指１・小指１が無ければ None。"""
    warn = warn or (lambda msg: None)
    s = SIDES[side]
    names = {f: [s + n for n in cfg.bones[f]] for f in FINGERS}
    wrist = s + '手首'
    idx1, pky1 = names['index'][0], names['pinky'][0]
    missing = [n for n in (wrist, idx1, pky1) if not skel.has(n)]
    if missing:
        warn(f'モデルに {"・".join(missing)} が無いので、{s}手の指は動かしません')
        return None
    w = skel.internal(wrist)
    a, b = skel.internal(idx1) - w, skel.internal(pky1) - w
    normal = _unit(np.cross(b, a) if side == 0 else np.cross(a, b))   # 手のひら側の法線
    to_pinky = _perp(skel.internal(pky1) - skel.internal(idx1), normal)
    across = list(cfg.thumb_across)
    rig = []
    for f, finger in enumerate(FINGERS):
        chain = names[finger]
        bones = []
        prev_dir = _unit(skel.internal(chain[0]) - w) if skel.has(chain[0]) else None
        for j, name in enumerate(chain):
            if not skel.has(name):
                continue
            nxt = chain[j + 1] if j + 1 < len(chain) else None
            d = _bone_direction(skel, name, nxt, prev_dir)
            if d is None:
                continue
            prev_dir = d
            target = normal + (float(across[j]) * to_pinky if f == 0 else 0.0)
            bones.append((name, j, _unit(np.cross(d, _perp(target, d)))))
        spread = None
        if f > 0 and bones and bones[0][1] == 0:
            d0 = _bone_direction(skel, bones[0][0], bones[1][0] if len(bones) > 1 else None, None)
            spread = _unit(np.cross(d0, _perp(-to_pinky, d0)))   # + で親指側へ
        rig.append((bones, spread))
    return rig


def finger_local_quats(rig, angles_deg):
    """angles (5, 4) [度] → {ボーン名: ローカル回転 (4,)（内部座標）}。付け根は曲げてから手のひらの面内で開く。"""
    angles = np.deg2rad(np.asarray(angles_deg, np.float64))
    out = {}
    for f, (bones, spread) in enumerate(rig):
        for name, j, axis in bones:
            q = quat.from_rotvec(angles[f, j] * axis)
            if j == 0 and spread is not None:
                q = quat.mul(quat.from_rotvec(angles[f, 3] * spread), q)
            out[name] = quat.normalize(q)
    return out


def shape_rotation_table(skel, cfg, warn=None):
    """形ごとの指ボーンのローカル回転（MMD 座標）: {'left'|'right': {形: {ボーン名: [x, y, z, w]}}}。"""
    from ..convert.tracks import to_mmd_quat
    table = preset_angles(cfg)
    out = {}
    for side, key in enumerate(('left', 'right')):
        rig = build_rig(skel, side, cfg, warn)
        if rig is None:
            continue
        out[key] = {shape: {name: to_mmd_quat(q).round(7).tolist()
                            for name, q in finger_local_quats(rig, table[k]).items()}
                    for k, shape in enumerate(SHAPES)}
    return out
