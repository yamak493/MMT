"""手のランドマークの検出: MediaPipe Hands のランドマークモデルを、体の関節から作った ROI に直接使う。

MediaPipe の HandLandmarker はまず手のひら検出で手を探すが、ダンス動画の手は小さく（数十 px）ぶれているので
見つからないフレームが多い。そこで手のひら検出は使わずに、
1. NLF の手首と手先の関節から、MediaPipe と同じ形の ROI（手首 → 指の付け根が上を向く正方形）を作り、
   hand_landmarker.task に入っているランドマークモデルに直接入れる
2. 1 回目の結果から MediaPipe のトラッキングと同じ方法で ROI を作り直して、もう一度推定する（存在スコアの高いほう）
体の手首から離れた所に見つかった手（もう一方の手など）は使わない。
"""
import os
import zipfile

import numpy as np
from scipy.ndimage import map_coordinates

from ..convert.wrist import intrinsics

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/'
             'hand_landmarker.task')
INPUT_SIZE = 224
WRIST_JOINTS = (20, 21)
HAND_JOINTS = (22, 23)
_ROI_POINTS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


class HandLandmarkModel:
    """hand_landmarker.task の中のランドマークモデル（入力は 224×224 の RGB、0〜1）。"""

    def __init__(self, task_path, num_threads=None):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as e:
            raise RuntimeError('手の検出に必要な ai-edge-litert が見つかりません（実行環境を再セットアップしてください）') from e
        with zipfile.ZipFile(task_path) as z:
            name = next(n for n in z.namelist() if n.endswith('hand_landmarks_detector.tflite'))
            content = z.read(name)
        self.interpreter = Interpreter(model_content=content, num_threads=num_threads or os.cpu_count() or 1)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        # 出力: 画像上の 21 点 (63)・手の存在 (1)・右手らしさ (1)・3D の 21 点 (63)。名前の順がこの並び
        outs = sorted(self.interpreter.get_output_details(), key=lambda d: d['name'])
        self.output_index = [d['index'] for d in outs]

    def __call__(self, crop):
        it = self.interpreter
        it.set_tensor(self.input_index, np.asarray(crop, np.float32)[None])
        it.invoke()
        screen, presence, handed, world = (it.get_tensor(i)[0] for i in self.output_index)
        return (screen.reshape(21, 3).astype(np.float64), float(presence.reshape(-1)[0]),
                float(handed.reshape(-1)[0]), world.reshape(21, 3).astype(np.float64))


# ---- ROI（中心 x, y [px]・一辺 [px]・回転 [rad]）----
def roi_from_joints(wrist, hand, side, along=0.8):
    wrist, hand = np.asarray(wrist, np.float64), np.asarray(hand, np.float64)
    d = hand - wrist
    angle = np.arctan2(d[0], -d[1])
    center = wrist + along * d
    return np.array([center[0], center[1], float(side), angle])


def roi_from_landmarks(points):
    """画像上の 21 点から、MediaPipe のトラッキングと同じ ROI を作る。"""
    p = np.asarray(points, np.float64)[:, :2]
    tip = ((p[5] + p[13]) / 2 + p[9]) / 2
    d = tip - p[0]
    angle = np.arctan2(d[0], -d[1])
    c, s = np.cos(angle), np.sin(angle)
    to_roi = np.array([[c, s], [-s, c]])
    q = p[_ROI_POINTS] @ to_roi.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    w, h = hi - lo
    center = (lo + hi) / 2 + np.array([0.0, -0.1 * h])
    center = center @ to_roi
    return np.array([center[0], center[1], 2.0 * max(w, h), angle])


def _roi_grid(roi, size):
    cx, cy, side, angle = roi
    u = (np.arange(size) + 0.5) / size - 0.5
    x, y = np.meshgrid(u * side, u * side)
    c, s = np.cos(angle), np.sin(angle)
    return cx + c * x - s * y, cy + s * x + c * y


def crop(image, roi, size=INPUT_SIZE):
    """ROI を size×size に切り出す（双線形補間。はみ出した所は端の色）。float32 0〜1。"""
    X, Y = _roi_grid(roi, size)
    img = np.asarray(image)
    H, W = img.shape[:2]
    rows, cols = Y - 0.5, X - 0.5
    y0 = int(np.clip(np.floor(rows.min()) - 1, 0, H - 1))
    y1 = int(np.clip(np.ceil(rows.max()) + 1, 0, H - 1))
    x0 = int(np.clip(np.floor(cols.min()) - 1, 0, W - 1))
    x1 = int(np.clip(np.ceil(cols.max()) + 1, 0, W - 1))
    patch = img[y0:y1 + 1, x0:x1 + 1, :3].astype(np.float32)
    out = np.empty((size, size, 3), np.float32)
    for k in range(3):
        out[..., k] = map_coordinates(patch[..., k], [rows - y0, cols - x0], order=1, mode='nearest')
    return out / 255.0


def to_image(points, roi, size=INPUT_SIZE):
    cx, cy, side, angle = roi
    p = np.asarray(points, np.float64)
    k = side / size
    x, y = (p[:, 0] - size / 2) * k, (p[:, 1] - size / 2) * k
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y, p[:, 2] * k], axis=-1)


def limit_roi(roi, base, max_shift=0.5, scale=(0.5, 2.0)):
    """作り直した ROI を、最初の ROI から大きく外れないようにする。"""
    roi = np.array(roi, np.float64)
    shift = roi[:2] - base[:2]
    limit = max_shift * base[2]
    norm = float(np.linalg.norm(shift))
    if norm > limit:
        roi[:2] = base[:2] + shift * (limit / norm)
    roi[2] = np.clip(roi[2], scale[0] * base[2], scale[1] * base[2])
    return roi


def detect(model, image, roi, passes=2):
    base = np.array(roi, np.float64)
    best = None
    for _ in range(max(1, int(passes))):
        screen, presence, handed, world = model(crop(image, roi))
        pts = to_image(screen, roi)
        if best is None or presence > best['presence']:
            best = dict(screen=pts, world=world, presence=presence, handedness=handed, roi=np.array(roi))
        roi = limit_roi(roi_from_landmarks(pts), base)
    return best


def joint_rois(joints3d, image_size, roi_m=0.25, fov_deg=55.0):
    """NLF の関節（カメラ座標 [mm]、(T, 24, 3)）から、左右の手の最初の ROI (T, 2, 4) と手首の画像上の位置。"""
    j = np.asarray(joints3d, np.float64)
    K = intrinsics(image_size, fov_deg)
    z = np.maximum(j[..., 2:], 1.0)
    uv = j[..., :2] / z * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    rois = np.zeros((len(j), 2, 4))
    for side in range(2):
        wrist, hand = uv[:, WRIST_JOINTS[side]], uv[:, HAND_JOINTS[side]]
        side_px = K[0, 0] * roi_m * 1000.0 / z[:, WRIST_JOINTS[side], 0]
        for t in range(len(j)):
            rois[t, side] = roi_from_joints(wrist[t], hand[t], side_px[t])
    return rois, uv[:, list(WRIST_JOINTS)]


def detect_video(frames, rois, wrists, model, passes=2, max_wrist_offset=0.5, progress=None, wrists_raw=None,
                 cancel=None):
    """動画の各フレームの左右の手を推定する。frames: 画像 (H, W, 3) を順に返すもの。"""
    T = len(rois)
    out = dict(screen=np.full((T, 2, 21, 3), np.nan, np.float32), world=np.full((T, 2, 21, 3), np.nan, np.float32),
               presence=np.zeros((T, 2), np.float32), handedness=np.full((T, 2), np.nan, np.float32),
               roi=np.zeros((T, 2, 4), np.float32))
    n = 0
    for t, image in enumerate(frames):
        if t >= T:
            break
        if cancel is not None and t % 8 == 0:
            cancel()
        image = np.asarray(image)[..., :3]
        for side in range(2):
            r = detect(model, image, rois[t, side], passes)
            offset = np.linalg.norm(r['screen'][0, :2] - wrists[t, side])
            if wrists_raw is not None:
                offset = min(offset, np.linalg.norm(r['screen'][0, :2] - wrists_raw[t, side]))
            offset /= max(rois[t, side, 2], 1.0)
            out['screen'][t, side] = r['screen']
            out['world'][t, side] = r['world']
            out['presence'][t, side] = r['presence'] if offset <= max_wrist_offset else 0.0
            out['handedness'][t, side] = r['handedness']
            out['roi'][t, side] = r['roi']
        n += 1
        if progress is not None:
            progress(n, T)
    if n < T:
        for k in out:
            out[k] = out[k][:n]
    return out
