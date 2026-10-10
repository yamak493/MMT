"""ステージ9: 回転のリターゲット（初期姿勢の違いの補正）。

SMPL は T ポーズ、MMD のモデルの多くは A ポーズなので、回転をそのままコピーせず、ボーンごとの補正回転 C で
    R_mmd_global = R_smpl_global · C,   R_mmd_local = (R_mmd_global,parent)⁻¹ · R_mmd_global
とする（C は PMX の初期姿勢を SMPL の初期姿勢に合わせる回転）。C は初期姿勢が本当に違う所だけに掛ける:
  * 腕・ひじ: 骨の向きを合わせる最小回転（T ポーズと A ポーズの違い）
  * 下半身・上半身・首・頭: どちらも直立なので上向きは鉛直とし、左右のねじれだけ合わせる
    （関節を結んだ向きを使うと、SMPL の頭の関節が首より前にあるため首が前に倒れる）
  * 肩: 補正しない（向きを合わせると肩がすくむ）/ 手首: 前腕と同じ補正（手首はまっすぐ）
  * 足ＩＫ: つま先の左右の向きだけ合わせる（SMPL の初期姿勢は約 15 度外向き）
計算はすべて内部座標（右手系・Y 上向き・+Z 正面）で行う。
"""
import numpy as np

from ..core import quat
from ..rig.skeleton import SIDES

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
DEFAULT_OPTIONS = dict(torso_up='vertical', collar='identity', wrist='forearm', foot='yaw')


def _min_rotation(a, b):
    return quat.to_matrix(quat.from_two_vectors(a, b))


def _frame_alignment(up_a, lat_a, up_b, lat_b):
    """「上向き」と「左右方向」で決まる座標系 a を座標系 b に重ねる回転。"""
    return quat.frame_from_up_lateral(up_b, lat_b) @ quat.frame_from_up_lateral(up_a, lat_a).T


class Retargeter:
    def __init__(self, skeleton, smpl_rest_joints, smpl_foot_axes=None, options=None):
        opt = dict(DEFAULT_OPTIONS, **(options or {}))
        for key, allowed in (('torso_up', ('vertical', 'joints')), ('collar', ('identity', 'direction')),
                             ('wrist', ('forearm', 'direction')), ('foot', ('yaw', 'direction'))):
            if opt[key] not in allowed:
                raise ValueError(f'retarget.{key} が不正です: {opt[key]}')
        self.skel = skeleton
        J = np.asarray(smpl_rest_joints, np.float64)
        S = skeleton.internal
        self.has_upper2 = skeleton.has('上半身2')
        self.warnings = []
        specs = []   # (MMD ボーン, 大域回転を使う SMPL の関節, 補正回転 C)

        def torso(name, joint, up_pmx, lat_pmx, up_smpl, lat_smpl):
            if opt['torso_up'] == 'vertical':
                up_pmx = up_smpl = Y_AXIS
            specs.append((name, joint, _frame_alignment(up_pmx, lat_pmx, up_smpl, lat_smpl)))

        hips_pmx = 0.5 * (S('左足') + S('右足'))
        lat_hips_pmx = S('左足') - S('右足')
        lat_arms_pmx = S('左腕') - S('右腕')
        lat_hips_smpl, lat_arms_smpl = J[1] - J[2], J[16] - J[17]
        torso('下半身', 0, S('下半身') - hips_pmx, lat_hips_pmx, J[0] - 0.5 * (J[1] + J[2]), lat_hips_smpl)
        if self.has_upper2:
            torso('上半身', 6, S('上半身2') - S('上半身'), lat_arms_pmx, J[9] - J[3], lat_arms_smpl)
            torso('上半身2', 9, S('首') - S('上半身2'), lat_arms_pmx, J[12] - J[9], lat_arms_smpl)
        else:
            # 上半身2 が無いモデルは、背骨 3 本分の回転を上半身に合成する
            torso('上半身', 9, S('首') - S('上半身'), lat_arms_pmx, J[12] - J[3], lat_arms_smpl)
        torso('首', 12, S('頭') - S('首'), X_AXIS, J[15] - J[12], X_AXIS)
        specs.append(('頭', 15, np.eye(3)))
        for side, s in enumerate(SIDES):
            collar, shoulder, elbow, wrist, hand = np.array([13, 16, 18, 20, 22]) + side
            if opt['collar'] == 'identity':
                specs.append((s + '肩', collar, np.eye(3)))
            else:
                specs.append((s + '肩', collar, _min_rotation(S(s + '腕') - S(s + '肩'), J[shoulder] - J[collar])))
            specs.append((s + '腕', shoulder, _min_rotation(S(s + 'ひじ') - S(s + '腕'), J[elbow] - J[shoulder])))
            C_elbow = _min_rotation(S(s + '手首') - S(s + 'ひじ'), J[wrist] - J[elbow])
            specs.append((s + 'ひじ', elbow, C_elbow))
            if opt['wrist'] == 'forearm':
                specs.append((s + '手首', wrist, C_elbow))
            else:
                tail = skeleton.tail_internal(s + '手首')
                if tail is None or np.linalg.norm(tail - S(s + '手首')) < 1e-6:
                    tail = S(s + '手首') + (S(s + '手首') - S(s + 'ひじ'))
                specs.append((s + '手首', wrist, _min_rotation(tail - S(s + '手首'), J[hand] - J[wrist])))

        self.specs = [sp for sp in specs if skeleton.has(sp[0])]
        self.bones = [sp[0] for sp in self.specs]
        names = set(self.bones)
        self.keyed_parent = {n: skeleton.nearest_ancestor(n, names) for n in self.bones}
        self.correction = {sp[0]: sp[2] for sp in self.specs}
        self.joint_of = {sp[0]: sp[1] for sp in self.specs}

        # 足ＩＫ: 足首の大域回転 × 補正回転
        foot_c = []
        for side, s in enumerate(SIDES):
            ankle, foot = (7, 10) if side == 0 else (8, 11)
            toe = next((S(s + n) for n in ('つま先', 'つま先ＩＫ') if skeleton.has(s + n)), None)
            base = S(s + '足首') if skeleton.has(s + '足首') else S(s + '足ＩＫ')
            if toe is None or np.linalg.norm(toe - base) < 1e-6:
                self.warnings.append(f'{s}つま先が見つからないので、足先の向きを標準の値で代用します')
                toe = base + np.array([0.0, -1.2, 1.6])
            if opt['foot'] == 'yaw':
                axis = J[foot] - J[ankle] if smpl_foot_axes is None else smpl_foot_axes[side]
                C = _frame_alignment(Y_AXIS, toe - base, Y_AXIS, axis)
            else:
                C = _min_rotation(toe - base, J[foot] - J[ankle])
            foot_c.append(C)
        self.foot_correction = np.stack(foot_c)

    def global_matrices(self, glob_rot):
        """ボーン名 → MMD のボーンの大域回転 (T, 3, 3)。"""
        return {name: glob_rot[:, j] @ C for name, j, C in self.specs}

    def global_matrix(self, name, glob_rot):
        for n, j, C in self.specs:
            if n == name:
                return glob_rot[:, j] @ C
        raise KeyError(name)

    def local_quats(self, glob_rot):
        """ボーン名 → 親（キーを打つボーンのうち最も近い祖先）に対する回転 (T, 4)（内部座標）。"""
        g = self.global_matrices(glob_rot)
        out = {}
        for name in self.bones:
            parent = self.keyed_parent[name]
            local = g[name] if parent is None else np.swapaxes(g[parent], -1, -2) @ g[name]
            out[name] = quat.make_continuous(quat.from_matrix(local))
        return out

    def globals_from_local(self, glob_rot, local):
        """キーを打つボーンの大域回転 {名前: (T, 3, 3)}（ローカル回転 local を親から順に掛けたもの）。"""
        base = self.global_matrices(glob_rot)
        out = {}
        for name in self.bones:      # 親が子より先に並んでいる
            parent = self.keyed_parent[name]
            R = quat.to_matrix(local[name])
            out[name] = R if parent is None else out[parent] @ R
        for name, G in base.items():
            out.setdefault(name, G)
        return out

    def foot_ik_quats(self, glob_rot):
        """足ＩＫの回転 (T, 2, 4)（親の 全ての親 は回らないので大域 = ローカル）。"""
        q = [quat.from_matrix(glob_rot[:, 7 + side] @ self.foot_correction[side]) for side in range(2)]
        return quat.make_continuous(np.stack(q, axis=1))


def bone_positions(skel, glob, names, num_frames):
    """ボーン名 → 位置 (T, 3)（内部座標）。glob: キーを打つボーンの大域回転。

    キーの無いボーン（捩り・肩C・腰など）は親の回転をそのまま受け継ぐ。センター・グルーブの移動は含めない。
    """
    out = {}
    for name in names:
        chain = skel.chain(name)
        p = np.tile(skel.internal(chain[0]), (num_frames, 1))
        R = glob.get(chain[0])
        for parent, child in zip(chain[:-1], chain[1:]):
            off = skel.internal(child) - skel.internal(parent)
            p = p + (off if R is None else R @ off)
            R = glob.get(child, R)
        out[name] = p
    return out
