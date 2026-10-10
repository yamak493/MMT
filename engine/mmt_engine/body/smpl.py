"""SMPL の体モデル（numpy 版）と順運動学。

使うのはテンプレート頂点・体型ブレンドシェイプ・関節（回帰を掛けた結果）・スキニングのウェイトだけで、
ポーズ補正ブレンドシェイプは使わない（関節の位置に影響しないため）。データは NLF の TorchScript に
入っている SMPL から書き出す（BodyModel.from_torch。SMPL の公式ファイルは不要）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import quat

SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], np.int64)
SMPL_JOINT_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar',
    'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hand', 'right_hand')

# 左右の [股関節, 膝, 足首, 足先]
LEG_JOINTS = np.array([[1, 4, 7, 10], [2, 5, 8, 11]])
ANKLES = LEG_JOINTS[:, 2]
FEET = LEG_JOINTS[:, 3]

_KEYS = ('v_template', 'shapedirs', 'J_template', 'J_shapedirs', 'weights', 'parents')


def sanitize_parents(parents, num_joints):
    """親の番号を検証し、根（親なし）を -1 にそろえる。

    体モデルによっては根の親が -1 ではなく、-1 を符号なし 32 ビットにした値で入っている。親は自分より前の関節で
    なければならないので、それ以外の値は「親なし」とみなす。根が 2 つ以上になるときは、SMPL の 24 関節なら
    既知の親子関係を使う。
    """
    p = np.asarray(parents).reshape(-1).astype(np.int64)
    if len(p) != num_joints:
        raise ValueError(f'体モデルの親の数 {len(p)} が関節数 {num_joints} と一致しません')
    p = np.where((p >= 0) & (p < np.arange(num_joints)), p, -1)
    if (p[1:] < 0).any():
        if num_joints == len(SMPL_PARENTS):
            return SMPL_PARENTS.copy()
        raise ValueError('体モデルの親子関係が不正です')
    return p


class BodyModel:
    def __init__(self, v_template, shapedirs, J_template, J_shapedirs, weights, parents=SMPL_PARENTS):
        self.v_template = np.asarray(v_template, np.float64)       # (V, 3)
        self.shapedirs = np.asarray(shapedirs, np.float64)         # (V, 3, S)
        self.J_template = np.asarray(J_template, np.float64)       # (J, 3)
        self.J_shapedirs = np.asarray(J_shapedirs, np.float64)     # (J, 3, S)
        self.weights = np.asarray(weights, np.float64)             # (V, J)
        V, J = len(self.v_template), len(self.J_template)
        self.parents = sanitize_parents(parents, J)
        if self.weights.shape != (V, J):
            raise ValueError(f'体モデルのウェイトの形が不正です: {self.weights.shape}')
        if self.shapedirs.shape[:2] != (V, 3) or self.J_shapedirs.shape[:2] != (J, 3):
            raise ValueError('体モデルの体型ブレンドシェイプの形が不正です')

    @property
    def num_betas(self):
        return self.shapedirs.shape[2]

    def _betas(self, betas):
        b = np.zeros(self.num_betas)
        betas = np.asarray(betas, np.float64).reshape(-1)[:self.num_betas]
        b[:len(betas)] = betas
        return b

    def rest_joints(self, betas):
        """初期姿勢（T ポーズ）の関節位置 (J, 3)。betas が (T, S) なら (T, J, 3)。"""
        betas = np.asarray(betas, np.float64)
        if betas.ndim == 2:
            b = np.zeros((len(betas), self.num_betas))
            n = min(betas.shape[1], self.num_betas)
            b[:, :n] = betas[:, :n]
            return self.J_template + np.einsum('jcs,ts->tjc', self.J_shapedirs, b)
        return self.J_template + self.J_shapedirs @ self._betas(betas)

    def rest_vertices(self, betas):
        return self.v_template + self.shapedirs @ self._betas(betas)

    def save_npz(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in _KEYS})
        return path

    @classmethod
    def from_npz(cls, path_or_dict, prefix=''):
        if isinstance(path_or_dict, (str, Path)):
            with np.load(path_or_dict) as d:
                data = {k: d[k] for k in d.files}
        else:
            data = dict(path_or_dict)

        def get(k):
            return np.asarray(data[prefix + k])

        parents = get('parents') if (prefix + 'parents') in data else SMPL_PARENTS
        return cls(get('v_template'), get('shapedirs'), get('J_template'), get('J_shapedirs'),
                   get('weights'), parents)

    @classmethod
    def from_torch(cls, module, num_betas=10):
        """NLF の TorchScript に入っている体モデル（nn.Module）のバッファから作る。"""
        bufs = {}
        for name, t in module.named_buffers():
            t = t.detach().cpu()
            # 整数のバッファ（親子関係）は float にしない（大きな値が丸められて壊れる）
            bufs[name] = (t.float() if t.is_floating_point() else t).numpy()
        v_template = bufs['v_template']
        shapedirs = bufs['shapedirs'][:, :, :num_betas]
        if 'J_template' in bufs and 'J_shapedirs' in bufs:
            J_template = bufs['J_template']
            J_shapedirs = bufs['J_shapedirs'][:, :, :num_betas]
        else:
            reg = bufs['J_regressor'] if 'J_regressor' in bufs else bufs['J_regressor_post_lbs']
            J_template = reg @ v_template
            J_shapedirs = np.einsum('jv,vcs->jcs', reg, shapedirs)
        parents = SMPL_PARENTS
        for key in ('kintree_parents_tensor', 'kintree_parents'):
            if key in bufs:
                parents = bufs[key].astype(np.int64)
                break
        return cls(v_template, shapedirs, J_template, J_shapedirs, bufs['weights'], parents)


# ---- 順運動学 ----
def forward_kinematics(local_quats, root_pos, rest_joints, parents=SMPL_PARENTS):
    """親に対する回転 (T, J, 4) と骨盤の位置 (T, 3) から、大域回転 (T, J, 3, 3) と関節位置 (T, J, 3)。

    大域回転 G_j は「初期姿勢の骨を今の向きへ回す回転」で、P_child = P_j + G_j (J_child − J_j)。
    """
    R = quat.to_matrix(local_quats)
    T, J = R.shape[:2]
    G = np.empty_like(R)
    P = np.empty((T, J, 3))
    for j, p in enumerate(parents):
        if p < 0:
            G[:, j] = R[:, j]
            P[:, j] = root_pos
        else:
            G[:, j] = G[:, p] @ R[:, j]
            P[:, j] = P[:, p] + G[:, p] @ (rest_joints[j] - rest_joints[p])
    return G, P


def lbs_points(glob_rot, joints, rest_joints, rest_points, point_weights):
    """選んだ数頂点だけを線形ブレンドスキニングで動かす。戻り値 (T, P, 3)。"""
    rel = rest_points[:, None, :] - rest_joints[None, :, :]                 # (P, J, 3)
    moved = np.einsum('tjab,pjb->tpja', glob_rot, rel) + joints[:, None]     # (T, P, J, 3)
    return np.einsum('tpja,pj->tpa', moved, point_weights)


def select_heel_toe(rest_verts, weights, n=4, sole_band=0.025):
    """足首・足先に主にスキンされた頂点から、かかと（最も後ろ）とつま先（最も前）を選ぶ。

    SMPL の初期姿勢は +Z が正面。足裏の候補は各足の最下点から sole_band 以内の頂点。
    戻り値: (かかと (2, n), つま先 (2, n)) の頂点番号。
    """
    owner = weights.argmax(1)
    heel, toe = [], []
    for side in range(2):
        cand = np.flatnonzero(np.isin(owner, [ANKLES[side], FEET[side]]))
        if len(cand) < 2 * n:
            raise ValueError('足首・足先にスキンされた頂点が足りません（体モデルを確認してください）')
        y = rest_verts[cand, 1]
        sole = cand[y <= y.min() + sole_band]
        if len(sole) < 2 * n:
            sole = cand[np.argsort(y)[:max(2 * n, len(sole))]]
        order = np.argsort(rest_verts[sole, 2])
        heel.append(sole[order[:n]])
        toe.append(sole[order[-n:]])
    return np.array(heel), np.array(toe)


@dataclass
class RestInfo:
    """体型を固定したときの初期姿勢の情報。"""
    joints: np.ndarray          # (J, 3) 初期姿勢の関節位置
    point_idx: np.ndarray       # (2 足, 2 [かかと, つま先], n) 頂点番号
    points: np.ndarray          # (2, 2, n, 3) その頂点の初期位置
    point_weights: np.ndarray   # (2, 2, n, J)
    sole_y: float               # 初期姿勢での足裏（かかと・つま先の最下点）の高さ

    def standing(self, points):
        """初期姿勢で直立したとき（骨盤の水平位置が原点・足裏が y = 0）の座標に直す。"""
        base = np.array([self.joints[0, 0], self.sole_y, self.joints[0, 2]])
        return np.asarray(points) - base

    def leg_length(self):
        """股関節 → 膝 → 足首 の長さ（左右の平均）。"""
        j = self.joints
        lengths = [np.linalg.norm(j[knee] - j[hip]) + np.linalg.norm(j[ankle] - j[knee])
                   for hip, knee, ankle, _ in LEG_JOINTS]
        return float(np.mean(lengths))


def rest_info(body_model, betas, n_points=4, sole_band=0.025):
    joints = body_model.rest_joints(betas)
    verts = body_model.rest_vertices(betas)
    heel, toe = select_heel_toe(verts, body_model.weights, n_points, sole_band)
    idx = np.stack([heel, toe], axis=1)               # (2, 2, n)
    points = verts[idx]
    sole_y = float(points[..., 1].min())
    return RestInfo(joints, idx, points, body_model.weights[idx], sole_y)


@dataclass
class Kinematics:
    """FK の結果。床・スケールの変換は全要素に同じ変換を掛ける。"""
    glob_rot: np.ndarray        # (T, J, 3, 3)
    joints: np.ndarray          # (T, J, 3)
    contact_points: np.ndarray  # (T, 2 足, 2 [かかと, つま先], 3)

    @property
    def root_pos(self):
        return self.joints[:, 0]

    def transformed(self, rotation=None, offset=None):
        """p' = R p + offset(t)。offset は (3,) か (T, 3)。回転は大域回転にも左から掛ける。"""
        G, P, C = self.glob_rot, self.joints, self.contact_points
        if rotation is not None:
            R = np.asarray(rotation, np.float64)
            G = np.einsum('ab,tjbc->tjac', R, G)
            P = P @ R.T
            C = C @ R.T
        if offset is not None:
            off = np.asarray(offset, np.float64)
            if off.ndim == 1:
                P, C = P + off, C + off
            else:
                P, C = P + off[:, None], C + off[:, None, None]
        return Kinematics(G, P, C)

    def scaled(self, k):
        return Kinematics(self.glob_rot, self.joints * k, self.contact_points * k)


def compute_kinematics(local_quats, root_pos, rest):
    G, P = forward_kinematics(local_quats, root_pos, rest.joints)
    n = rest.points.shape[2]
    pts = lbs_points(G, P, rest.joints, rest.points.reshape(-1, 3),
                     rest.point_weights.reshape(-1, rest.point_weights.shape[-1]))
    pts = pts.reshape(len(P), 2, 2, n, 3).mean(3)
    return Kinematics(G, P, pts)
