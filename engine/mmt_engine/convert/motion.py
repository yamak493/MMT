"""ステージ1: 入力（NLF の SMPL パラメータ）の読み込みと正規化。

内部の形（Motion）:
  quats    (T, 24, 4)  親に対する関節の回転（x, y, z, w）。0 番はルートの向き
  betas    (10,)       体型（フレームごとなら中央値 1 つに固定する）
  root_pos (T, 3)      骨盤の位置 [m]（SMPL の trans ではなく、骨盤の関節そのものの位置）
座標系は Y 上向き・右手系（SMPL の初期姿勢と同じく +Z が正面、+X が体の左）。
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..body.smpl import forward_kinematics
from ..core import quat

# カメラ座標（x = 右 / y = 下 / z = 奥）→ Y 上向き: X 軸まわりに 180 度
CAMERA_TO_YUP = np.diag([1.0, -1.0, -1.0])


@dataclass
class Motion:
    quats: np.ndarray
    betas: np.ndarray
    root_pos: np.ndarray
    fps: float
    source_fps: float
    valid: np.ndarray
    fk_check_mm: float = float('nan')   # 入力の関節位置と、体モデルで求め直した関節位置の最大差
    coords: str = 'camera'

    @property
    def num_frames(self):
        return len(self.quats)

    def with_values(self, **kw):
        return replace(self, **kw)


def as_dict(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return dict(source)


def _pose_to_quats(pose):
    pose = np.asarray(pose, np.float64)
    T = pose.shape[0]
    if pose.shape[-2:] == (3, 3):
        return quat.from_matrix(pose.reshape(T, -1, 3, 3))
    if pose.shape[-1] == 4 and pose.ndim == 3:
        return quat.normalize(pose)
    return quat.from_rotvec(pose.reshape(T, -1, 3))


def resample(quats, root_pos, valid, fps, target_fps):
    """回転は球面線形補間、位置は線形補間で target_fps にそろえる。"""
    if target_fps <= 0 or abs(fps - target_fps) < 1e-6 or len(quats) < 2:
        return quats, root_pos, valid
    T = len(quats)
    t_src = np.arange(T) / fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / target_fps)
    w = t_dst * fps
    lo = np.minimum(np.floor(w).astype(int), T - 1)
    hi = np.minimum(lo + 1, T - 1)
    frac = w - lo
    q = quat.slerp(quats[lo], quats[hi], np.broadcast_to(frac[:, None], quats[lo].shape[:-1]))
    pos = np.stack([np.interp(t_dst, t_src, root_pos[:, k]) for k in range(3)], axis=1)
    return q, pos, resample_mask(valid, fps, target_fps)


def resample_mask(mask, fps, target_fps):
    """(T,) の bool 列を target_fps にそろえる（最も近いフレームの値。resample と同じフレームの並び）。"""
    mask = np.asarray(mask)
    T = len(mask)
    if target_fps <= 0 or abs(fps - target_fps) < 1e-6 or T < 2:
        return mask
    w = np.arange(0.0, (T - 1) / fps + 1e-9, 1.0 / target_fps) * fps
    return mask[np.clip(np.round(w).astype(int), 0, T - 1)]


def resampled_length(T, fps, target_fps):
    if target_fps <= 0 or abs(fps - target_fps) < 1e-6 or T < 2:
        return T
    return len(np.arange(0.0, (T - 1) / fps + 1e-9, 1.0 / target_fps))


def load_motion(source, body_model, coords='camera'):
    """モーション（npz のパスか dict）から Motion（入力の fps のまま）。

    必要なキー: pose（(T,24,3) 回転ベクトル / (T,24,4) / (T,24,3,3)）・betas・trans・fps
    任意: valid・joints3d（[mm]。体モデルの自己検証に使う）・coord_system（camera / yup）
    """
    d = as_dict(source)
    quats = _pose_to_quats(d['pose'])
    T = len(quats)
    fps = float(np.asarray(d.get('fps', 30.0)))
    trans = np.asarray(d['trans'], np.float64).reshape(T, 3)
    valid = np.asarray(d['valid'], bool) if 'valid' in d else np.ones(T, bool)
    if not valid.any():
        raise ValueError('人物を検出できたフレームがありません')

    betas = np.asarray(d['betas'], np.float64)
    per_frame = betas if betas.ndim == 2 else None
    if betas.ndim == 2:
        betas = np.median(betas[valid], axis=0)

    # 骨盤の位置 = 初期姿勢の骨盤 + trans（フレームごとの体型があればそれで求める）
    if per_frame is not None:
        pelvis = body_model.rest_joints(per_frame)[:, 0] + trans
    else:
        pelvis = body_model.rest_joints(betas)[0] + trans

    fk_check = float('nan')
    if 'joints3d' in d:
        j_in = np.asarray(d['joints3d'], np.float64).reshape(T, -1, 3)[:, :24] / 1000.0
        _, P = forward_kinematics(quats, pelvis, body_model.rest_joints(betas), body_model.parents)
        fk_check = float(np.abs(P - j_in)[valid].max() * 1000.0)

    coords = str(np.asarray(d['coord_system'])) if 'coord_system' in d else coords
    if coords == 'camera':
        # ルートの向きと位置の両方に同じ回転を掛ける
        pelvis = pelvis @ CAMERA_TO_YUP.T
        quats[:, 0] = quat.mul(quat.from_matrix(CAMERA_TO_YUP), quats[:, 0])
    elif coords != 'yup':
        raise ValueError(f'座標系が不正です: {coords}')
    return Motion(quat.make_continuous(quats), betas, pelvis, fps, fps, valid, fk_check, coords)
