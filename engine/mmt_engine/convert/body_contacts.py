"""ステージ9b: 腕・手のひら・指先と、体・相手の腕の接触（当たり判定）。

MMD のモデルは服・胸・髪・スカートのぶん SMPL より体が大きく、比率も違うので、回転をコピーしただけでは
手や指先が胴・胸・スカートに入り込む。重なったフレームだけ腕を直す。
* 腕: 上腕・前腕・手（指の節まで入れるときは手のひら 3 本と指の各節）のカプセル
* 体: PMX の剛体（物理の剛体は重なりの一部だけ離す）。剛体が無ければメッシュ、それも無ければ標準の形
* 解き方: 9a と同じく、フレームの順に、前のフレームの補正から始めて推定の姿勢へ戻しながら、どの組も重ならない
  最小の補正を求める。補正する関節は肩（3 自由度）・ひじ（曲げ 1 自由度）・手首（3 自由度）で、先の関節ほど
  動かしやすくする。腕どうしの組は自重する腕の肩だけで離す。触れるのは許し、それ以上は離さない
* 部位（服・頭・胴・胸・脚）ごとに判定を外せる
"""
from dataclasses import dataclass, field

import numpy as np

from ..core import filters, quat
from ..rig.skeleton import SIDES
from .arm_collision import OVERLAP_TOL_M, arm_model
from .geometry import (angle, clamp_rotation, closest_points, constrained_step, cross, distance_to_segment,
                       rotation, rotvec)
from .retarget import bone_positions

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PARTS = ('upper', 'fore', 'palm') + FINGERS
PART_LABELS = dict(upper='上腕', fore='前腕', palm='手のひら', thumb='親指', index='人差し指', middle='中指',
                   ring='薬指', pinky='小指')
BODY_BONES = ('下半身', '上半身', '上半身2', '首', '頭', '左足', '右足', '左ひざ', '右ひざ')
BODY_PARTS = ('clothes', 'head', 'torso', 'chest', 'legs')
BODY_PART_LABELS = dict(clothes='服・装飾', head='頭部', torso='胴部', chest='胸', legs='脚')
CHEST_WORDS = ('胸', '乳', 'おっぱい', 'オッパイ', 'ｵｯﾊﾟｲ', 'bust', 'breast', 'oppai')
CLOTH_WORDS = ('スカート', 'ｽｶｰﾄ', 'skirt', 'ネクタイ', 'ﾈｸﾀｲ', 'リボン', 'ﾘﾎﾞﾝ', 'ribbon', 'フリル', 'ﾌﾘﾙ',
               'frill', '裾')
ITERATIONS = 30
FINAL_ITERATIONS = 10
MAX_STEP_DEG = 20.0
STOP_RAD = 1e-4
MIN_LEVER_M = 0.01
TOLERANCE_M = 1e-4
DAMPING = 0.05
MIN_VERTICES = 20
NEAR_M = 0.1
NEAR_ROUNDS = 4
IDENTITY_RAD = 1e-6
_EPS = 1e-9


@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    radius: float
    bone: str
    part: int = -1         # 腕: PARTS の番号 / 体: -1
    level: int = -1        # 腕: 0 = 上腕 / 1 = 前腕 / 2 = 手。体: -1
    weight: float = 1.0    # 体: 離す割合
    name: str = ''
    body_part: str = ''


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _owners(skel, candidates):
    if skel.mesh_points is None or len(skel.mesh_points) == 0:
        return None, None
    names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    own = np.array([(n if n in candidates else skel.nearest_ancestor(n, candidates)) or ''
                    if skel.has(n) else '' for n in names])
    return own[inverse.reshape(-1)], skel.mesh_points * [1.0, 1.0, -1.0]


def _finger_tip(skel, chain, base, pts):
    """指の最後のボーンの先: 〇〇先 → 表示先 → 頂点（95 パーセンタイル）→ 最後の節の 0.8 倍。"""
    last = chain[-1]
    if skel.has(last + '先'):
        return skel.internal(last + '先')
    tail = skel.tail_internal(last)
    if tail is not None and np.linalg.norm(tail - skel.internal(last)) > 1e-6:
        return tail
    prev = skel.internal(chain[-2]) if len(chain) > 1 else base
    d = skel.internal(last) - prev
    length = 0.8 * float(np.linalg.norm(d))
    if pts is not None and len(pts) >= 5:
        length = max(float(np.percentile((pts - skel.internal(last)) @ _unit(d), 95.0)), 0.3 * length)
    return skel.internal(last) + _unit(d) * length


def arm_capsules(skel, side, cfg, arm_cfg, finger_bones, unit):
    """片腕のカプセルのリスト。"""
    s = SIDES[side]
    S = skel.internal
    model = arm_model(skel, arm_cfg, unit)
    r = model.radius[side]
    caps = [Capsule(S(s + '腕'), S(s + 'ひじ'), r[0], s + '腕', 0, 0, name=s + '上腕'),
            Capsule(S(s + 'ひじ'), S(s + '手首'), r[1], s + 'ひじ', 1, 1, name=s + '前腕')]
    wrist = S(s + '手首')
    chains = {f: [s + n for n in finger_bones[f] if skel.has(s + n)] for f in FINGERS}
    idx1, pky1 = s + finger_bones['index'][0], s + finger_bones['pinky'][0]
    if not (cfg.fingers and skel.has(idx1) and skel.has(pky1)):
        caps.append(Capsule(wrist, model.tip[side], r[2], s + '手首', 2, 2, name=s + '手'))
    else:
        finger_names = {n for c in chains.values() for n in c}
        owner, points = _owners(skel, finger_names | {s + '手首'})
        palm = [(wrist, S(idx1)), (wrist, S(pky1)), (S(idx1), S(pky1))]
        radius = float(cfg.palm_radius_m) * unit
        if owner is not None:
            pts = points[owner == s + '手首']
            if len(pts) >= MIN_VERTICES:
                d = np.min([distance_to_segment(pts, a, b) for a, b in palm], axis=0)
                radius = float(np.percentile(d, float(cfg.radius_percentile)))
        for a, b in palm:
            caps.append(Capsule(a, b, radius, s + '手首', 2, 2, name=s + '手のひら'))
        for f, finger in enumerate(FINGERS):
            chain = chains[finger]
            if not chain:
                continue
            last_pts = None if owner is None else points[owner == chain[-1]]
            ends = [S(n) for n in chain] + [_finger_tip(skel, chain, wrist, last_pts)]
            for i, name in enumerate(chain):
                radius = float(cfg.finger_radius_m[f]) * unit
                if owner is not None:
                    pts = points[owner == name]
                    if len(pts) >= MIN_VERTICES // 2:
                        radius = float(np.percentile(distance_to_segment(pts, ends[i], ends[i + 1]),
                                                     float(cfg.radius_percentile)))
                caps.append(Capsule(ends[i], ends[i + 1], radius, name, 3 + f, 2, name=s + PART_LABELS[finger]))
    for c in caps:
        c.radius = max(c.radius * float(cfg.radius_scale), 1e-6)
    return caps


# ---- 体の当たり判定 ----
def _rigid_rotation(rot):
    """PMX の剛体の回転（x・y・z [rad]。Z → X → Y の順）→ 内部座標の回転行列。"""
    rx, ry, rz = (float(v) for v in rot)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    flip = np.diag([1.0, 1.0, -1.0])
    return flip @ (Ry @ Rx @ Rz) @ flip


def _box_capsules(center, axes, half, bone, weight, name):
    """箱を 1〜2 本のカプセルで包む（いちばん長い向きを線分にし、薄い向きを半径にする）。"""
    order = np.argsort(half)[::-1]
    L, M, Sh = half[order]
    long_ax, mid_ax = axes[:, order[0]], axes[:, order[1]]
    if M <= 1.3 * Sh:
        r = 0.5 * (M + Sh)
        h = max(L - r, 0.0)
        return [Capsule(center - h * long_ax, center + h * long_ax, r, bone, weight=weight, name=name)]
    h = max(L - Sh, 0.0)
    out = []
    for sign in (1.0, -1.0):
        c = center + sign * (M - Sh) * mid_ax
        out.append(Capsule(c - h * long_ax, c + h * long_ax, Sh, bone, weight=weight, name=name))
    return out


def _arm_bone(skel, name):
    return any(skel.is_descendant(name, s + '肩') or skel.is_descendant(name, s + '腕') for s in SIDES)


def _spine_bones(skel):
    out = {n for n in ('上半身', '下半身') if skel.has(n)}
    for end in ['首'] + [s + b for s in SIDES for b in ('肩', '腕', '足')]:
        if skel.has(end):
            out.update(n for n in skel.chain(end)[:-1]
                       if skel.is_descendant(n, '上半身') or skel.is_descendant(n, '下半身'))
    return out


def _has_word(name, words):
    name = (name or '').lower()
    return any(w in name for w in words)


def _chest_rigid_name(name):
    name = name or ''
    return _has_word(name, CHEST_WORDS[1:]) or ('胸' in name and any(c in name for c in '左右LR'))


def _body_part(skel, bone, spine, physics=False, name=''):
    if skel.is_descendant(bone, '首'):
        return 'head'
    if any(skel.is_descendant(bone, s + b) for s in SIDES for b in ('足', '足D')):
        return 'legs'
    n, seen = bone, set()
    while n is not None and n not in spine and n not in seen:
        if _has_word(n, CHEST_WORDS):
            return 'chest'
        seen.add(n)
        n = skel.parents.get(n)
    if bone in spine:
        if _chest_rigid_name(name):
            return 'chest'
        if not physics and not _has_word(name, CLOTH_WORDS):
            return 'torso'
    return 'clothes'


def enabled_body_parts(cfg):
    parts = cfg.get('body_parts') or {}
    return {p for p in BODY_PARTS if parts.get(p, True)}


def _rigid_capsules(skel, cfg, unit):
    caps = []
    margin = float(cfg.cloth_margin_m) * unit
    soft = float(cfg.soft_ratio)
    spine = _spine_bones(skel)
    for rb in skel.rigid_bodies or []:
        bone = rb['bone']
        if bone is None or not skel.has(bone) or _arm_bone(skel, bone):
            continue
        if not (skel.is_descendant(bone, '上半身') or skel.is_descendant(bone, '下半身')):
            continue
        weight = 1.0
        if rb['mode'] != 0:
            if skel.is_descendant(bone, '首') or any(skel.is_descendant(bone, s + '足') for s in SIDES):
                continue
            if skel.is_descendant(bone, '下半身'):
                if cfg.skirt == 'none':
                    continue
                weight = soft if cfg.skirt == 'soft' else 1.0
            else:
                weight = soft
        center = rb['position'] * [1.0, 1.0, -1.0]
        R = _rigid_rotation(rb['rotation'])
        size = np.asarray(rb['size'], np.float64)
        if rb['shape'] == 0:
            new = [Capsule(center, center, size[0] + margin, bone, weight=weight, name=rb['name'])]
        elif rb['shape'] == 2:
            h = 0.5 * size[1]
            new = [Capsule(center - h * R[:, 1], center + h * R[:, 1], size[0] + margin, bone, weight=weight,
                           name=rb['name'])]
        else:
            new = _box_capsules(center, R, size + margin, bone, weight, rb['name'])
        part = _body_part(skel, bone, spine, rb['mode'] != 0, rb['name'])
        for c in new:
            c.body_part = part
        caps += new
    return caps


def _mesh_capsules(skel, cfg, unit):
    body = {b for b in BODY_BONES if skel.has(b)}
    owner, points = _owners(skel, body)
    if owner is None:
        return []
    names, inverse = np.unique(skel.mesh_bones, return_inverse=True)
    excluded = np.array([skel.has(n) and (_arm_bone(skel, n) or any(
        skel.has(s + '足首') and skel.is_descendant(n, s + '足首') for s in SIDES)) for n in names])
    owner = np.where(excluded[inverse.reshape(-1)], '', owner)
    caps = []
    for bone in sorted(body):
        pts = points[owner == bone]
        if len(pts) < 30:
            continue
        c = pts.mean(0)
        _, vecs = np.linalg.eigh(np.cov((pts - c).T))
        half = np.percentile(np.abs((pts - c) @ vecs), 95.0, axis=0)
        new = _box_capsules(c, vecs, np.maximum(half, 1e-6), bone, 1.0, bone)
        part = 'head' if bone in ('首', '頭') else 'legs' if bone[1:] in ('足', 'ひざ') else 'torso'
        for cap in new:
            cap.body_part = part
        caps += new
    return caps


def fallback_capsules(skel, unit, cfg):
    """剛体もメッシュも無いときの標準の体の形（肩幅・腰幅に対する割合で決める）。"""
    margin = float(cfg.cloth_margin_m) * unit
    S = skel.internal
    shoulder = float(np.linalg.norm(S('左腕') - S('右腕')))
    hip = 0.5 * float(np.linalg.norm(S('左足') - S('右足')))
    chest = '上半身2' if skel.has('上半身2') else '上半身'
    lateral = np.array([0.15 * shoulder, 0.0, 0.0])
    caps = [Capsule(S('上半身') + d, S('首') + d, 0.24 * shoulder + margin, chest, name='胴', body_part='torso')
            for d in (lateral, -lateral)]
    caps.append(Capsule(S('左足'), S('右足'), 0.9 * hip + margin, '下半身', name='腰', body_part='torso'))
    head = S('頭') + [0.0, 0.25 * shoulder, 0.0]
    caps.append(Capsule(head, head, 0.33 * shoulder + margin, '頭', name='頭', body_part='head'))
    for s in SIDES:
        if skel.has(s + 'ひざ'):
            caps.append(Capsule(S(s + '足'), S(s + 'ひざ'), 0.8 * hip + margin, s + '足', name=s + '太もも',
                                body_part='legs'))
    return caps


def body_capsules(skel, cfg, unit):
    """(体のカプセルのリスト, 'rigid' | 'mesh' | 'config' | 'none')。"""
    src = str(cfg.body_source)
    if src == 'none':
        return [], 'none'
    if src in ('auto', 'rigid'):
        caps = _rigid_capsules(skel, cfg, unit)
        if caps or src == 'rigid':
            return caps, 'rigid' if caps else 'none'
    if src in ('auto', 'mesh'):
        caps = _mesh_capsules(skel, cfg, unit)
        if caps or src == 'mesh':
            return caps, 'mesh' if caps else 'none'
    return fallback_capsules(skel, unit, cfg), 'config'


# ---- FK ----
def leg_globals(skel, glob_rot, smpl_rest):
    """MMD の脚の大域回転の近似（脚のキーを打たないとき）。"""
    J = np.asarray(smpl_rest, np.float64)
    out = {}
    for side, s in enumerate(SIDES):
        for bone, child, j, jc in ((s + '足', s + 'ひざ', 1 + side, 4 + side), (s + 'ひざ', s + '足首', 4 + side, 7 + side)):
            if skel.has(bone) and skel.has(child):
                C = quat.to_matrix(quat.from_two_vectors(skel.internal(child) - skel.internal(bone), J[jc] - J[j]))
                out[bone] = glob_rot[:, j] @ C
    return out


def _effective(skel, glob, name, T):
    n, seen = name, set()
    while n is not None and n not in seen:
        if n in glob:
            return glob[n]
        seen.add(n)
        n = skel.parents.get(n)
    return np.tile(np.eye(3), (T, 1, 1))


def _attach(skel, glob, caps, T):
    """(T, n, 2, 3) カプセルの端点の位置。"""
    if not caps:
        return np.zeros((T, 0, 2, 3))
    bones = sorted({c.bone for c in caps})
    pos = bone_positions(skel, glob, bones, T)
    rot = {b: _effective(skel, glob, b, T) for b in bones}
    out = np.empty((T, len(caps), 2, 3))
    for i, c in enumerate(caps):
        base = skel.internal(c.bone)
        for k, p in enumerate((c.a, c.b)):
            out[:, i, k] = pos[c.bone] + rot[c.bone] @ (p - base)
    return out


# ---- 解く ----
@dataclass
class _Arm:
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    Gp: np.ndarray
    G_arm: np.ndarray
    G_elb: np.ndarray
    ends: np.ndarray
    level: np.ndarray
    b_local: np.ndarray

    def apply(self, state):
        As, alpha, Aw = state
        Qs = self.Gp @ As @ self.Gp.T
        E2 = self.S + Qs @ (self.E - self.S)
        b_g = Qs @ self.G_arm @ self.b_local
        Re = rotation(alpha * b_g)
        ReQs = Re @ Qs
        W2 = E2 + ReQs @ (self.W - self.E)
        G_elb2 = ReQs @ self.G_elb
        Qw = G_elb2 @ Aw @ G_elb2.T
        out = np.empty_like(self.ends)
        l0, l1, l2 = self.level == 0, self.level == 1, self.level == 2
        out[l0] = self.S + (self.ends[l0] - self.S) @ Qs.T
        out[l1] = E2 + (self.ends[l1] - self.E) @ ReQs.T
        out[l2] = W2 + (self.ends[l2] - self.W) @ (Qw @ ReQs).T
        return out, E2, W2, b_g, G_elb2


@dataclass
class _Pairs:
    kind: np.ndarray
    side: np.ndarray
    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    w: np.ndarray
    level: np.ndarray
    trim: np.ndarray


@dataclass
class _Params:
    max_angle: np.ndarray
    cost: np.ndarray
    min_lever: float
    tol: float
    return_step: float
    near: float = np.inf


def _identity_state():
    return (np.eye(3), 0.0, np.eye(3))


def _pair_geometry(arms, states, body, pairs):
    applied = [a.apply(s) for a, s in zip(arms, states)]
    P0 = np.empty((len(pairs.i), 2, 3))
    Q0 = np.empty_like(P0)
    for s in range(2):
        m = pairs.side == s
        ends = applied[s][0]
        P0[m] = ends[pairs.i[m]]
        mb = m & (pairs.kind == 0)
        ma = m & (pairs.kind == 1)
        Q0[mb] = body[pairs.j[mb]]
        Q0[ma] = applied[1 - s][0][pairs.j[ma]]
    P0[:, 0] += pairs.trim[:, None] * (P0[:, 1] - P0[:, 0])
    cp, cq = closest_points(P0[:, 0], P0[:, 1], Q0[:, 0], Q0[:, 1])
    return cp, cq, applied, P0, Q0


def _gap(arms, states, body, pairs):
    cp, cq, _, _, _ = _pair_geometry(arms, states, body, pairs)
    return pairs.R - np.linalg.norm(cp - cq, axis=-1)


def _penetration(arms, states, body, pairs):
    return _gap(arms, states, body, pairs) * pairs.w


def _subset(pairs, m):
    return _Pairs(*(getattr(pairs, f)[m] for f in ('kind', 'side', 'i', 'j', 'R', 'w', 'level', 'trim')))


def _solve_near(arms, body, pairs, states, p, goal=None, skip=None):
    """近い組だけで解き、解いた姿勢で近くなった組を加えて解き直す。"""
    skip = np.zeros(len(pairs.i), bool) if skip is None else skip
    near = _gap(arms, states, body, pairs) > -p.near
    for _ in range(NEAR_ROUNDS):
        out = _solve_frame(arms, body, _subset(pairs, near), states, p, goal, skip[near])
        grown = near | (_gap(arms, out, body, pairs) > -p.near)
        if (grown == near).all():
            return out
        near = grown
    return _solve_frame(arms, body, _subset(pairs, near), states, p, goal, skip[near])


def _solve_frame(arms, body, pairs, states, p, goal=None, skip=None):
    """1 フレーム分。両腕の補正 states = [(As, alpha, Aw)] × 2 を初期値から求める。"""
    states = [tuple(np.copy(x) if isinstance(x, np.ndarray) else x for x in st) for st in states]
    skip = np.zeros(len(pairs.i), bool) if skip is None else skip
    pulling = goal is not None
    scale = 1.0 / np.sqrt(np.repeat(p.cost, [3, 1, 3], axis=1).reshape(-1))
    scale = scale * np.tile(np.repeat(np.asarray(p.max_angle) > 0.0, [3, 1, 3]), 2)
    for it in range(ITERATIONS + FINAL_ITERATIONS):
        pulling = pulling and it < ITERATIONS
        cp, cq, applied, P0, Q0 = _pair_geometry(arms, states, body, pairs)
        d = cp - cq
        dist = np.linalg.norm(d, axis=-1)
        n = d / np.maximum(dist, _EPS)[:, None]
        flat = dist <= _EPS
        if flat.any():
            perp = cross(P0[flat, 1] - P0[flat, 0], Q0[flat, 1] - Q0[flat, 0])
            norm = np.linalg.norm(perp, axis=-1, keepdims=True)
            n[flat] = np.where(norm > _EPS, perp / np.maximum(norm, _EPS), [0.0, 0.0, 1.0])
        pen = (pairs.R - dist) * pairs.w
        J = np.zeros((len(pen), 14))
        for s in range(2):
            m = pairs.side == s
            if not m.any():
                continue
            _, E2, W2, b_g, _ = applied[s]
            c, nn = cp[m], n[m]
            o = 7 * s
            J[m, o:o + 3] = cross(c - arms[s].S, nn)
            body_rows = pairs.kind[m] == 0
            lv = pairs.level[m]
            J[m, o + 3] = (cross(c - E2, nn) @ b_g) * ((lv >= 1) & body_rows)
            J[m, o + 4:o + 7] = cross(c - W2, nn) * ((lv == 2) & body_rows)[:, None]
        use = (np.linalg.norm(J, axis=1) > p.min_lever) & ~skip
        target = np.zeros(14)
        if pulling:
            for s in range(2):
                As, alpha, Aw = states[s]
                gs, ga, gw = goal[s]
                G_elb2 = applied[s][4]
                target[7 * s:7 * s + 3] = rotvec(arms[s].Gp @ (gs @ As.T) @ arms[s].Gp.T)
                target[7 * s + 3] = ga - alpha
                target[7 * s + 4:7 * s + 7] = rotvec(G_elb2 @ (gw @ Aw.T) @ G_elb2.T)
        if not (use & (pen > p.tol)).any() and np.abs(target).max() < 1e-9:
            break
        x = target
        if use.any():
            Ju = J[use]
            u = constrained_step(Ju * scale, pen[use] + p.tol - Ju @ target, DAMPING)
            x = target + scale * u
        joint = np.array([np.linalg.norm(x[0:3]), abs(x[3]), np.linalg.norm(x[4:7]),
                          np.linalg.norm(x[7:10]), abs(x[10]), np.linalg.norm(x[11:14])])
        step = float(joint.max())
        if step < STOP_RAD:
            if pulling:
                pulling = False
                continue
            break
        x = x * min(1.0, np.deg2rad(MAX_STEP_DEG) / step)
        new = []
        for s in range(2):
            As, alpha, Aw = states[s]
            xs, xa, xw = x[7 * s:7 * s + 3], x[7 * s + 3], x[7 * s + 4:7 * s + 7]
            G_elb2 = applied[s][4]
            Gp = arms[s].Gp
            As = Gp.T @ rotation(xs) @ Gp @ As
            Aw = G_elb2.T @ rotation(xw) @ G_elb2 @ Aw
            alpha = float(np.clip(alpha + xa, -p.max_angle[1], p.max_angle[1]))
            As, Aw = clamp_rotation(As, p.max_angle[0]), clamp_rotation(Aw, p.max_angle[2])
            new.append((As, alpha, Aw))
        states = new
    return states


def _toward_identity(state, step):
    As, alpha, Aw = state

    def back(A):
        rv = rotvec(A)
        a = float(np.linalg.norm(rv))
        return np.eye(3) if a <= step else rotation(rv * (1.0 - step / a))

    return (back(As), float(np.sign(alpha) * max(abs(alpha) - step, 0.0)), back(Aw))


def _saturated(state, p):
    As, alpha, Aw = state
    lim = p.max_angle
    return ((lim[0] > 0 and angle(As) >= lim[0] - 1e-6) or (lim[1] > 0 and abs(alpha) >= lim[1] - 1e-6)
            or (lim[2] > 0 and angle(Aw) >= lim[2] - 1e-6))


def _is_identity(states):
    return all(angle(st[0]) < IDENTITY_RAD and abs(st[1]) < IDENTITY_RAD and angle(st[2]) < IDENTITY_RAD
               for st in states)


@dataclass
class ContactResult:
    enabled: bool
    body_source: str
    num_body: int
    fingers: bool
    depth_before: np.ndarray      # (T, 2 腕, 8 部位)
    depth_after: np.ndarray
    correction_deg: np.ndarray    # (T, 2 腕, 3 [肩, ひじ, 手首])
    info: dict = field(default_factory=dict)

    def overlap_frames(self, tol):
        return (int((self.depth_before > tol).any(axis=(1, 2)).sum()),
                int((self.depth_after > tol).any(axis=(1, 2)).sum()))


def _part_depth(pen, pairs, part):
    out = np.full(pen.shape[:-1] + (2, len(PARTS)), -np.inf)
    for s in range(2):
        for k in range(len(PARTS)):
            m = (pairs.side == s) & (part == k)
            if m.any():
                out[..., s, k] = pen[..., m].max(-1)
    return out


def _raw_penetration(arm_ends, body_ends, pairs):
    T, n = len(body_ends), len(pairs.i)
    P0 = np.empty((T, n, 2, 3))
    Q0 = np.empty_like(P0)
    for s in range(2):
        m = pairs.side == s
        P0[:, m] = arm_ends[s][:, pairs.i[m]]
        mb, ma = m & (pairs.kind == 0), m & (pairs.kind == 1)
        Q0[:, mb] = body_ends[:, pairs.j[mb]]
        Q0[:, ma] = arm_ends[1 - s][:, pairs.j[ma]]
    P0[:, :, 0] += pairs.trim[:, None] * (P0[:, :, 1] - P0[:, :, 0])
    cp, cq = closest_points(P0[..., 0, :], P0[..., 1, :], Q0[..., 0, :], Q0[..., 1, :])
    return pairs.R - np.linalg.norm(cp - cq, axis=-1)


def resolve_contacts(skel, rt, glob_rot, local, cfg, arm_cfg, finger_bones, yield_mode, unit, fps, smpl_rest,
                     finger_local=None, leg_glob=None, cancel=None):
    """ステージ9b。local の 腕・ひじ・手首 を直した dict と ContactResult。"""
    T = len(glob_rot)
    caps = [arm_capsules(skel, s, cfg, arm_cfg, finger_bones, unit) for s in range(2)]
    fingers = any(c.part >= 3 for c in caps[0] + caps[1])
    all_bodies, source = body_capsules(skel, cfg, unit)
    off = set(BODY_PARTS) - enabled_body_parts(cfg)
    bodies = [b for b in all_bodies if b.body_part not in off]
    parts = [np.array([c.part for c in cs]) for cs in caps]
    empty = ContactResult(False, source, len(bodies), fingers, np.full((T, 2, len(PARTS)), -np.inf),
                          np.full((T, 2, len(PARTS)), -np.inf), np.zeros((T, 2, 3)))
    empty.info = dict(enabled=False, body_source=source)
    if not cfg.enabled or T == 0:
        return local, empty

    margin = float(cfg.margin_m) * unit
    skip_len = float(cfg.upper_arm_skip)
    kind, side, ii, jj, R, w = [], [], [], [], [], []
    for s in range(2):
        for i, c in enumerate(caps[s]):
            a = c.a + (skip_len * (c.b - c.a) if c.part == 0 else 0.0)
            for j, b in enumerate(bodies):
                cp, cq = closest_points(a, c.b, b.a, b.b)
                if np.linalg.norm(cp - cq) < c.radius + b.radius + margin:
                    continue   # 初期姿勢ですでに重なっている組（脇の下の上腕と胸など）は扱わない
                kind.append(0), side.append(s), ii.append(i), jj.append(j)
                R.append(c.radius + b.radius + margin), w.append(b.weight)
    yield_side = {'left': 0, 'right': 1}.get(str(yield_mode), -1)
    if yield_side >= 0:
        o = 1 - yield_side
        for i, c in enumerate(caps[yield_side]):
            for j, d in enumerate(caps[o]):
                kind.append(1), side.append(yield_side), ii.append(i), jj.append(j)
                R.append(c.radius + d.radius + margin), w.append(1.0)
    pairs = _Pairs(np.array(kind, int), np.array(side, int), np.array(ii, int), np.array(jj, int),
                   np.array(R, float), np.array(w, float),
                   np.array([caps[s][i].level for s, i in zip(side, ii)], int),
                   np.array([skip_len if k_ == 0 and caps[s][i].part == 0 else 0.0
                             for k_, s, i in zip(kind, side, ii)], float))
    if len(pairs.i) == 0:
        return local, empty

    glob = rt.globals_from_local(glob_rot, local)
    legs = leg_globals(skel, glob_rot, smpl_rest) if leg_glob is None else leg_glob
    glob.update({k: v for k, v in legs.items() if k not in glob})
    for name in sorted((n for n in (finger_local or {}) if skel.has(n)), key=skel.depth):
        glob[name] = _effective(skel, glob, skel.parents.get(name), T) @ quat.to_matrix(finger_local[name])
    arm_ends = [_attach(skel, glob, caps[s], T) for s in range(2)]
    body_ends = _attach(skel, glob, bodies, T)
    names = [[SIDES[s] + b for b in ('腕', 'ひじ', '手首')] for s in range(2)]
    pos = bone_positions(skel, glob, names[0] + names[1], T)
    parents = [rt.keyed_parent[SIDES[s] + '腕'] for s in range(2)]
    b_locals = [_unit(np.cross(skel.internal(names[s][2]) - skel.internal(names[s][1]), [0.0, 0.0, 1.0]))
                for s in range(2)]
    levels = [np.array([c.level for c in caps[s]]) for s in range(2)]

    def arm_at(s, t):
        n = names[s]
        Gp = glob[parents[s]][t] if parents[s] is not None else np.eye(3)
        return _Arm(pos[n[0]][t], pos[n[1]][t], pos[n[2]][t], Gp, glob[n[0]][t], glob[n[1]][t],
                    arm_ends[s][t], levels[s], b_locals[s])

    k = cfg.joint_weights
    cost = np.zeros((2, 3))
    for s in range(2):
        n = names[s]
        upper = np.linalg.norm(skel.internal(n[1]) - skel.internal(n[0]))
        fore = np.linalg.norm(skel.internal(n[2]) - skel.internal(n[1]))
        hand = max(float(np.max([np.linalg.norm(c.b - skel.internal(n[2])) for c in caps[s] if c.level == 2])), 1e-3)
        cost[s] = [float(k.shoulder) * (upper + fore + hand) ** 2, float(k.elbow) * (fore + hand) ** 2,
                   float(k.wrist) * hand ** 2]
    mx = cfg.max_deg
    p = _Params(np.deg2rad([float(mx.shoulder), float(mx.elbow), float(mx.wrist)]), cost, MIN_LEVER_M * unit,
                TOLERANCE_M * unit, np.deg2rad(float(cfg.return_deg_per_s)) / fps, NEAR_M * unit)

    pair_part = np.array([parts[s][i] for s, i in zip(pairs.side, pairs.i)], int)
    raw_all = _raw_penetration(arm_ends, body_ends, pairs)
    before = _part_depth(raw_all, pairs, pair_part)
    raw_w = raw_all * pairs.w

    identity = [_identity_state(), _identity_state()]
    states = [identity] * T
    prev = identity
    skip = np.zeros(len(pairs.i), bool)
    ignored = np.zeros((T, len(pairs.i)), bool)
    for t in range(T):
        if cancel is not None and t % 32 == 0:
            cancel()
        if _is_identity(prev) and (raw_w[t] <= p.tol).all():
            skip[:] = False
            continue
        arms = [arm_at(0, t), arm_at(1, t)]
        body = body_ends[t]
        goal = [_toward_identity(st, p.return_step) for st in prev]
        cur = _solve_near(arms, body, pairs, prev, p, goal, skip)
        if _is_identity(cur):
            cur = identity
        over = _penetration(arms, cur, body, pairs) > p.tol
        for s in range(2):
            if _saturated(cur[s], p):
                skip |= over & (pairs.side == s)
        skip &= over
        ignored[t] = skip
        states[t] = cur
        prev = cur

    sigma = float(cfg.smooth_sec) * fps
    As = quat.to_rotvec(quat.from_matrix(np.array([[st[s][0] for s in range(2)] for st in states])))
    al = np.array([[st[s][1] for s in range(2)] for st in states])
    Aw = quat.to_rotvec(quat.from_matrix(np.array([[st[s][2] for s in range(2)] for st in states])))
    As, al, Aw = (filters.gaussian_time(x, sigma) for x in (As, al, Aw))
    pen_after = raw_all.copy()
    final = []
    for t in range(T):
        if cancel is not None and t % 64 == 0:
            cancel()
        st = [(rotation(As[t, s]), float(al[t, s]), rotation(Aw[t, s])) for s in range(2)]
        if _is_identity(st) and (raw_w[t] <= p.tol).all():
            final.append(identity)
            continue
        arms = [arm_at(0, t), arm_at(1, t)]
        body = body_ends[t]
        pen = _penetration(arms, st, body, pairs)
        if ((pen > p.tol) & ~ignored[t]).any():
            st = _solve_near(arms, body, pairs, st, p, skip=ignored[t])
            pen = _penetration(arms, st, body, pairs)
        pen_after[t] = pen / np.maximum(pairs.w, 1e-9)
        final.append(st)
    after = _part_depth(pen_after, pairs, pair_part)

    out = dict(local)
    corr = np.zeros((T, 2, 3))
    for s in range(2):
        n = names[s]
        qs = quat.from_matrix(np.stack([f[s][0] for f in final]))
        qa = quat.from_rotvec(np.array([f[s][1] for f in final])[:, None] * b_locals[s])
        qw = quat.from_matrix(np.stack([f[s][2] for f in final]))
        corr[:, s] = np.rad2deg(np.stack([quat.angle_between(q, quat.IDENTITY) for q in (qs, qa, qw)], axis=1))
        for name, q in zip(n, (qs, qa, qw)):
            if corr[:, s].max(initial=0.0) > 1e-9 and name in out:
                out[name] = quat.make_continuous(quat.mul(q, out[name]))
    res = ContactResult(True, source, len(bodies), fingers, before, after, corr)
    b_frames, a_frames = res.overlap_frames(OVERLAP_TOL_M * unit)
    res.info = dict(enabled=True, body_source=source, body_capsules=len(bodies), fingers=fingers,
                    overlap_frames=dict(before=b_frames, after=a_frames),
                    max_correction_deg={j: round(float(corr[..., m].max(initial=0.0)), 2)
                                        for m, j in enumerate(('shoulder', 'elbow', 'wrist'))})
    return out, res
