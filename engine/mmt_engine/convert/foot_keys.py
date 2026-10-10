"""タイムラインの足ピタ（単発のキー）。足ピタを使わずに変換した結果に、後から重ねる。

同じ足のキーを前から 2 つずつ組にし、組の間（固定の区間）は、最初のキーのフレームの足の水平の位置に、足裏が床に
平らに着く高さと向き（鉛直軸まわりの向きだけ残す）で足ＩＫを固定する（体が動いても足は動かさない）。組にならずに
余った最後のキーは、そのフレームだけ固定する。固定の前後 foot_ik.blend_frames フレームで元の動きになめらかに戻し、
それより外のフレームは変えない（計算もしない）。
* 続いた（重なった）固定の区間は 1 つにまとめ、最初の区間の位置に固定する
* 2 つの固定の区間の間が前後のなじませの幅（2 × blend_frames）以下なら、元の動きからの差を前の区間から次の区間へ
  移していく
* 固定した足に脚が届かなくなるフレームだけ、センターを下げる（体の高さ）
"""
import numpy as np

from ..core import filters, quat
from .center import reach_correction
from .tracks import build_tracks

# 足ピタのキーの掛け方の版（変えたら上げる。前の版で作った結果をキャッシュから使わない）
FOOT_KEYS_VERSION = 2

UP = np.array([0.0, 1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])


def normalize_keys(keys):
    """[{foot: 0|1, frame}] → フレームの順に並べ、重複と不正なものを除いたリスト。"""
    out = set()
    for item in keys or ():
        try:
            foot, frame = int(item['foot']), int(item['frame'])
        except (KeyError, TypeError, ValueError):
            continue
        if foot in (0, 1) and frame >= 0:
            out.add((foot, frame))
    return [dict(foot=f, frame=t) for f, t in sorted(out, key=lambda x: (x[1], x[0]))]


def key_frames(keys, num_frames):
    """足ごとの、キーのフレームのソート済みリスト（範囲の外は捨てる）。"""
    out = ([], [])
    for item in normalize_keys(keys):
        if item['frame'] < num_frames:
            out[item['foot']].append(item['frame'])
    return out


def flat_rotation(q):
    """足ＩＫの回転 q (4,) を、鉛直軸まわりの向きだけにしたもの（足裏が床に平ら）。"""
    fwd = quat.rotate(q, FORWARD)
    return quat.axis_angle(UP, np.arctan2(fwd[0], fwd[2]))


def hold_spans(frames):
    """1 つの足のキーのフレーム（ソート済み）→ 固定の区間 [(開始, 終了), ...]（終了を含む）。

    前から 2 つずつ組にし、余った最後のキーは 1 フレームの区間。続いた・重なった区間は 1 つにまとめる。
    """
    frames = sorted(frames)
    spans = []
    for i in range(0, len(frames), 2):
        s, e = frames[i], frames[min(i + 1, len(frames) - 1)]
        if spans and s <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [tuple(x) for x in spans]


def place_foot(delta, rotation, frames, blend):
    """1 つの足の足ＩＫ（delta (T, 3)・rotation (T, 4)）にキーを掛けた (delta, rotation, 変えたフレーム (T,), 重み (T,))。

    重みは、キーの位置への寄せ具合（キーで 1、なじませの外で 0）。
    """
    delta = np.asarray(delta, np.float64)
    rotation = quat.make_continuous(np.asarray(rotation, np.float64))
    T = len(delta)
    pos = delta.copy()
    rot = rotation.copy()
    weight = np.zeros(T)
    locks = []
    for s, e in hold_spans(frames):
        p = delta[s].copy()
        p[1] = 0.0      # 足ＩＫの差分の Y = 0 が、足裏が床に平らに着いた高さ
        q = flat_rotation(rotation[s])
        pos[s:e + 1] = p
        rot[s:e + 1] = q
        weight[s:e + 1] = 1.0
        locks.append((s, e, p, q))

    def offset(t, p, q):
        return p - delta[t], quat.mul(q, quat.conj(rotation[t]))

    def fade(frames_, dp, dq, w):
        pos[frames_] = delta[frames_] + dp * w[:, None]
        rot[frames_] = quat.mul(quat.slerp(np.broadcast_to(quat.IDENTITY, (len(w), 4)),
                                           np.broadcast_to(dq, (len(w), 4)), w), rotation[frames_])
        weight[frames_] = w

    w_fade = 1.0 - filters.smoothstep(np.arange(1, blend + 1) / (blend + 1.0))
    for i, (s, e, p, q) in enumerate(locks):
        prev = locks[i - 1] if i > 0 else None
        nxt = locks[i + 1] if i + 1 < len(locks) else None
        if prev is None or s - prev[1] - 1 > 2 * blend:
            k = min(blend, s)
            if k > 0:
                dp, dq = offset(s, p, q)
                fade(np.arange(s - 1, s - k - 1, -1), dp, dq, w_fade[:k])
        if nxt is not None and nxt[0] - e - 1 <= 2 * blend:
            m = nxt[0] - e - 1
            if m > 0:
                dp0, dq0 = offset(e, p, q)
                dp1, dq1 = offset(nxt[0], nxt[2], nxt[3])
                w = filters.smoothstep(np.arange(1, m + 1) / (m + 1.0))
                span = np.arange(e + 1, nxt[0])
                pos[span] = delta[span] + dp0 * (1.0 - w)[:, None] + dp1 * w[:, None]
                dq = quat.slerp(np.broadcast_to(dq0, (m, 4)), np.broadcast_to(dq1, (m, 4)), w)
                rot[span] = quat.mul(dq, rotation[span])
                weight[span] = 1.0
        else:
            k = min(blend, T - 1 - e)
            if k > 0:
                dp, dq = offset(e, p, q)
                fade(np.arange(e + 1, e + k + 1), dp, dq, w_fade[:k])
    changed = weight > 0
    return pos, quat.make_continuous(rot), changed, weight


def place_feet(center_delta, ik, frames_by_foot, lower_rot, geom, cfg, unit):
    """センターの差分と足ＩＫ（FootIKResult）にキーを掛けた (センターの差分, 足ＩＫ, 変えたフレーム (T,))。"""
    T = len(center_delta)
    blend = int(cfg.foot_ik.blend_frames)
    delta = np.array(ik.delta, np.float64, copy=True)
    rot = np.array(ik.rotation, np.float64, copy=True)
    weight = np.zeros((T, 2))
    for foot in range(2):
        if frames_by_foot[foot]:
            delta[:, foot], rot[:, foot], _, weight[:, foot] = place_foot(ik.delta[:, foot], ik.rotation[:, foot],
                                                                        frames_by_foot[foot], blend)
    touched = weight > 0
    if not touched.any():
        return center_delta, ik, np.zeros(T, bool)
    new_ik = ik.with_values(delta=delta, rotation=rot, target=np.asarray(ik.target) + (delta - ik.delta))

    # 体の高さ: キーを掛けた足が届かないフレームだけ、センターを下げる。下げる量は、キーの区間ごとの最大を
    # 足の寄せ具合（重み）でなじませたもの（どのフレームでも届くのに要る量以上）
    need = reach_correction(geom, center_delta, lower_rot, delta, float(cfg.center.reach_ratio),
                            float(cfg.center.reach_max_drop_m) * unit, legs=touched)
    w = weight.max(1)
    corr = np.zeros(T)
    for s, e in filters.runs(w > 0):
        peak = float(need[s:e + 1].min())
        if peak < 0:
            corr[s:e + 1] = np.minimum(need[s:e + 1], peak * w[s:e + 1])
    center = np.array(center_delta, np.float64, copy=True)
    center[:, 1] += corr
    return center, new_ik, touched.any(1)


def _slice_ik(ik, sl):
    return ik.with_values(delta=np.asarray(ik.delta)[sl], rotation=np.asarray(ik.rotation)[sl],
                          target=np.asarray(ik.target)[sl])


def splice_tracks(tracks, skel, center_delta, ik, local, cfg, poles, changed):
    """tracks（全フレームのキー列）の changed のフレームだけを、center_delta・ik から作り直したキー列。"""
    out = {t.name: t.__class__(t.name, np.array(t.positions, copy=True), np.array(t.rotations, copy=True),
                               t.movable) for t in tracks}
    for s, e in filters.runs(changed):
        sl = slice(s, e + 1)
        part = build_tracks(skel, np.asarray(center_delta)[sl], _slice_ik(ik, sl), {b: q[sl] for b, q in local.items()},
                            cfg, None if poles is None else np.asarray(poles)[sl])
        for t in part:
            base = out.get(t.name)
            if base is None:
                continue
            base.positions[sl] = t.positions
            # 前後のフレームと同じ側の符号にそろえる
            r = np.asarray(t.rotations, np.float64)
            sign = np.where(np.sum(r * base.rotations[sl], axis=-1) < 0.0, -1.0, 1.0)
            base.rotations[sl] = r * sign[:, None]
    return [out[t.name] for t in tracks]
