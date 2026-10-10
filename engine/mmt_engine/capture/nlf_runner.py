"""NLF（Neural Localizer Fields）でのモーション抽出。

NLF の TorchScript（nlf_l_multi_0.3.2.torchscript）は、人物検出 → 3D の頂点・関節の推定 → SMPL への当てはめ
（SMPLFitter）までを 1 つにまとめたもの。フレームごとに推論し、最初のフレームで選んだ人物を追跡して 1 人分を
取り出す。検出できなかったフレームは前後から補間し、体型をシーケンス全体の中央値 1 つに固定する（骨盤の位置は
推定のまま）。SMPL の体モデルも TorchScript の中に入っているので、SMPL の公式ファイルは不要。
"""
import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from ..body.smpl import BodyModel

MODEL_URL = 'https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript'
MODEL_FILENAME = 'nlf_l_multi_0.3.2.torchscript'
MODEL_MIN_BYTES = 300 * 1024 ** 2
BODY_MODEL_NAME = 'smpl'
N_JOINTS = 24
TRACK_MAX_JUMP_M = 1.5     # 前のフレームの人からこれより離れた人しかいなければ、そのフレームは検出できなかったとする
TRACK_LOST_SEC = 1.0       # これより長く見失ったら、人物を選び直す


def select_device(preference='auto'):
    import torch
    if preference == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if preference == 'cuda':
        raise RuntimeError('CUDA に対応した GPU が見つかりません')
    return 'cpu'


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return dict(name=torch.cuda.get_device_name(0), memory_mb=int(props.total_memory // (1024 ** 2)),
                capability=f'{props.major}.{props.minor}')


def ensure_linalg(log):
    """NLF の中の小さな連立方程式（cholesky_ex・cholesky_solve）が、この GPU で動くかを確かめる。

    新しい種類の GPU では、PyTorch 同梱の MAGMA にその GPU 用のカーネルが無いことがある。動かなければ cuSOLVER に、
    それも動かなければ要素ごとの演算で書いた実装に差し替える。
    """
    import torch
    if not torch.cuda.is_available():
        return

    def selftest():
        gen = torch.Generator().manual_seed(0)
        for n in (1, 4, 384):
            m = torch.randn(n, 3, 3, generator=gen)
            a = m @ m.mT + 0.1 * torch.eye(3)
            b = torch.randn(n, 3, 1, generator=gen)
            chol, _ = torch.linalg.cholesky_ex(a.cuda())
            x = torch.cholesky_solve(b.cuda(), chol).cpu()
            torch.cuda.synchronize()
            ref = torch.linalg.solve(a.double(), b.double()).float()
            if not torch.allclose(x, ref, rtol=1e-3, atol=1e-4):
                raise RuntimeError('GPU の線形代数の結果が CPU と一致しません')

    def pure_cholesky_ex(a, upper=False, check_errors=False):
        n = a.shape[-1]
        low = torch.zeros_like(a)
        info = torch.zeros(a.shape[:-2], dtype=torch.int32, device=a.device)
        for j in range(n):
            d2 = a[..., j, j] - (low[..., j, :j] ** 2).sum(-1)
            info = torch.where((info == 0) & ~(d2 > 0), torch.full_like(info, j + 1), info)
            d = torch.sqrt(torch.clamp(d2, min=torch.finfo(a.dtype).tiny))
            low[..., j, j] = d
            if j + 1 < n:
                low[..., j + 1:, j] = (a[..., j + 1:, j] - (low[..., j + 1:, :j] * low[..., j, None, :j]).sum(-1)) / d[..., None]
        return (low.mT.contiguous() if upper else low), info

    def pure_cholesky_solve(b, factor, upper):
        low = factor.mT if upper else factor
        n = low.shape[-1]
        y = []
        for i in range(n):
            s = b[..., i, :]
            for j in range(i):
                s = s - low[..., i, j, None] * y[j]
            y.append(s / low[..., i, i, None])
        x = [None] * n
        for i in reversed(range(n)):
            s = y[i]
            for j in range(i + 1, n):
                s = s - low[..., j, i, None] * x[j]
            x[i] = s / low[..., i, i, None]
        return torch.stack(x, dim=-2)

    try:
        selftest()
        return
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
    try:
        torch.backends.cuda.preferred_linalg_library('cusolver')
        selftest()
        log(f'この GPU では PyTorch 同梱の MAGMA が動かないので、線形代数を cuSOLVER で計算します（{first}）')
        return
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        global _PURE_LINALG_LIB
        _PURE_LINALG_LIB = torch.library.Library('aten', 'IMPL')
        _PURE_LINALG_LIB.impl('linalg_cholesky_ex', pure_cholesky_ex, 'CUDA')
        _PURE_LINALG_LIB.impl('_cholesky_solve_helper', pure_cholesky_solve, 'CUDA')
    selftest()
    log('この GPU では線形代数ライブラリが動かないので、NLF の中の小さな連立方程式を要素ごとの演算で計算します')


_PURE_LINALG_LIB = None
_MODEL_CACHE = {}


def _half_is_broken(model, device):
    """この GPU で半精度の推論が NaN になるか（GeForce GTX 16xx など）。人物検出のネットワークで確かめる。"""
    import torch
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 640, 640, generator=gen).to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        y = model.detector.model(x)
    y = y[0] if isinstance(y, (list, tuple)) else y
    return not bool(torch.isfinite(y).all())


def _jit_load(path, device):
    """TorchScript を読み込む。Windows の torch.jit.load はパスを ANSI のコードページで開くので、日本語などを含む
    パス（ユーザー名が日本語のときのアプリのデータのフォルダ）では開けない。Python で開いたファイルを渡す。"""
    import torch
    with open(path, 'rb') as f:
        return torch.jit.load(f, map_location=device)


def load_model(path, device, log, precision='auto'):
    """TorchScript を読み込む（同じプロセスでは使い回す）。戻り値 (モデル, 'fp16' | 'fp32')。

    precision: auto（半精度を試し、NaN になる GPU では単精度）| fp16 | fp32。CPU は常に単精度。
    """
    import torch
    import torchvision  # noqa: F401  NLF の TorchScript が使う torchvision::nms などの演算を登録する
    from .fp32_patch import make_fp32_model
    # TorchScript のグラフの最適化を止める（最初の 2 回の推論に数分かかり、入力の形が変わるたびにやり直すため。
    # 最適化しなくても推論の速さはほとんど変わらない）
    torch._C._set_graph_executor_optimize(False)
    if device == 'cpu':
        precision = 'fp32'
    key = (str(path), device, precision)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _MODEL_CACHE.clear()
    if device == 'cuda':
        torch.cuda.empty_cache()
        ensure_linalg(log)
    chosen = precision
    if precision in ('auto', 'fp16'):
        model = _jit_load(path, device).to(device).eval()
        chosen = 'fp16'
        if precision == 'auto' and _half_is_broken(model, device):
            log('この GPU では半精度の計算が正しく動かないので、単精度で推論します（少し遅く、メモリを多く使います）')
            del model
            torch.cuda.empty_cache()
            chosen = 'fp32'
    if chosen == 'fp32':
        model = _jit_load(make_fp32_model(path), device).float().to(device).eval()
    _MODEL_CACHE[key] = (model, chosen)
    return _MODEL_CACHE[key]


def body_model_from(model):
    return BodyModel.from_torch(getattr(model.body_models, BODY_MODEL_NAME))


def pick_first_person(boxes, image_w, mode):
    """boxes: (n, 5) = x, y, w, h, score。largest = 一番大きく写っている人 / center = 画面中央の人。"""
    areas = boxes[:, 2] * boxes[:, 3]
    if mode == 'center':
        big = np.flatnonzero(areas >= 0.3 * areas.max())
        cx = boxes[big, 0] + boxes[big, 2] / 2
        return int(big[np.argmin(np.abs(cx - image_w / 2))])
    return int(np.argmax(areas))


class Tracker:
    """最初のフレームで選んだ人物に最も近い検出を追跡する（別の人に乗り移らない）。"""

    def __init__(self, fps, select_mode, image_w):
        self.fps = fps
        self.mode = select_mode
        self.image_w = image_w
        self.prev_trans = None
        self.lost = 0
        self.reacquired = []

    def choose(self, frame_index, boxes, trans):
        if len(boxes) == 0:
            self.lost += 1
            return None
        if self.prev_trans is None or self.lost > TRACK_LOST_SEC * self.fps:
            if self.prev_trans is not None:
                self.reacquired.append(frame_index)
            j = pick_first_person(boxes, self.image_w, self.mode)
        else:
            d = np.linalg.norm(trans - self.prev_trans[None], axis=-1)
            j = int(np.argmin(d))
            if d[j] > TRACK_MAX_JUMP_M:
                self.lost += 1
                return None
        self.lost = 0
        self.prev_trans = trans[j]
        return j


def finite_people(arrays):
    """人ごとの推定（先頭の軸が人）が、どれもすべて有限の値か。戻り値 (人数,) の bool。"""
    n = len(arrays[0])
    ok = np.ones(n, bool)
    if n == 0:
        return ok
    for a in arrays:
        ok &= np.isfinite(np.asarray(a, np.float64).reshape(n, -1)).all(axis=1)
    return ok


def run_inference(model, frames, num_expected, fps, image_size, settings, device, progress, cancel):
    """フレームを NLF に通して、追跡した 1 人分の SMPL パラメータを集める。

    settings: batch_size・num_aug・antialias・detector_threshold・person_select
    戻り値 dict: pose (T, 72)・betas (T, 10)・trans (T, 3)・joints3d (T, 24, 3) [mm]・valid (T,)・uncertainty (T,)・
    boxes (T, 5)・reacquired（人物を選び直したフレーム）
    """
    import torch
    batch_size = max(1, int(settings.get('batch_size', 16)))
    num_aug = max(1, int(settings.get('num_aug', 3)))
    if num_aug % 2 == 0:
        num_aug += 1
    antialias = int(settings.get('antialias', 2))
    threshold = float(settings.get('detector_threshold', 0.25))
    internal_batch = max(64, num_aug * 4)
    tracker = Tracker(fps, settings.get('person_select', 'largest'), image_size[0])
    out = dict(pose=[], betas=[], trans=[], joints3d=[], valid=[], uncertainty=[], boxes=[])

    def store(frame_index, pred, k):
        # 誰も検出されなかったバッチの結果には boxes が無い
        boxes = pred['boxes'][k].detach().float().cpu().numpy() if 'boxes' in pred else np.zeros((0, 5))
        trans = pred['trans'][k].detach().float().cpu().numpy() if len(boxes) else np.zeros((0, 3))
        # 推定が NaN・無限大になった人（半精度の桁あふれなど）は、検出されなかったものとして扱う
        people = [boxes, trans]
        if len(boxes):
            people += [pred[name][k].detach().float().cpu().numpy() for name in ('pose', 'betas', 'joints3d')]
        keep = np.flatnonzero(finite_people(people))
        j = tracker.choose(frame_index, boxes[keep], trans[keep])
        if j is not None:
            j = int(keep[j])
        if j is None:
            out['pose'].append(np.zeros(72, np.float32))
            out['betas'].append(np.zeros(10, np.float32))
            out['trans'].append(np.zeros(3, np.float32))
            out['joints3d'].append(np.zeros((N_JOINTS, 3), np.float32))
            out['boxes'].append(np.zeros(5, np.float32))
            out['uncertainty'].append(np.nan)
            out['valid'].append(False)
            return
        out['pose'].append(pred['pose'][k][j].detach().float().reshape(-1).cpu().numpy())
        out['betas'].append(pred['betas'][k][j].detach().float().cpu().numpy())
        out['trans'].append(trans[j])
        out['joints3d'].append(pred['joints3d'][k][j].detach().float().cpu().numpy())
        out['boxes'].append(boxes[j])
        out['uncertainty'].append(float(pred['joint_uncertainties'][k][j].detach().float().mean().cpu()))
        out['valid'].append(True)

    def flush(imgs, idx):
        if not imgs:
            return
        try:
            images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().to(device)
            with torch.inference_mode(), torch.device(device):
                pred = model.detect_smpl_batched(
                    images, model_name=BODY_MODEL_NAME, detector_threshold=threshold,
                    internal_batch_size=internal_batch, num_aug=num_aug, antialias_factor=antialias)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or len(imgs) == 1:
                raise
            if device == 'cuda':
                torch.cuda.empty_cache()
            half = len(imgs) // 2
            flush(imgs[:half], idx[:half])
            flush(imgs[half:], idx[half:])
            return
        for k in range(len(idx)):
            store(idx[k], pred, k)

    imgs, idx = [], []
    count = 0
    for frame in frames:
        cancel()
        imgs.append(np.ascontiguousarray(frame))
        idx.append(count)
        count += 1
        if len(imgs) >= batch_size:
            flush(imgs, idx)
            imgs, idx = [], []
            progress(count, num_expected)
    flush(imgs, idx)
    progress(count, max(num_expected, count))
    if count == 0:
        raise RuntimeError('動画からフレームを読めませんでした')
    result = {k: np.stack(out[k]) for k in ('pose', 'betas', 'trans', 'joints3d', 'boxes')}
    result['valid'] = np.asarray(out['valid'], bool)
    result['uncertainty'] = np.asarray(out['uncertainty'], np.float32)
    result['reacquired'] = tracker.reacquired
    return result


def fill_gaps(arr, valid):
    """検出できなかったフレームを前後の有効なフレームから線形補間する。"""
    arr = np.asarray(arr, np.float32)
    idx = np.arange(len(arr))
    vi = idx[valid]
    flat = arr.reshape(len(arr), -1)[valid]
    w = np.interp(idx, vi, np.arange(len(vi)))
    lo = np.floor(w).astype(int)
    hi = np.ceil(w).astype(int)
    t = (w - lo)[:, None].astype(np.float32)
    return (flat[lo] * (1 - t) + flat[hi] * t).reshape(arr.shape)


def fill_rotation_gaps(rotvecs, valid):
    """回転ベクトル (T, J, 3) の欠損を、前後の回転から球面線形補間する（回転ベクトルのまま補間すると、
    全身の向きが ±π の切れ目にあるとき体が 1 回転する）。"""
    rv = np.asarray(rotvecs, np.float64)
    T, J = rv.shape[:2]
    idx = np.arange(T)
    vi = idx[valid]
    w = np.interp(idx, vi, np.arange(len(vi)))
    lo = vi[np.floor(w).astype(int)]
    hi = vi[np.ceil(w).astype(int)]
    t = (w - np.floor(w))[:, None, None]
    r_lo = Rotation.from_rotvec(rv[lo].reshape(-1, 3))
    r_hi = Rotation.from_rotvec(rv[hi].reshape(-1, 3))
    step = (r_lo.inv() * r_hi).as_rotvec().reshape(T, J, 3)
    out = r_lo * Rotation.from_rotvec((step * t).reshape(-1, 3))
    return out.as_rotvec().reshape(T, J, 3).astype(np.float32)


def finalize_motion(raw, model, device, fps, log):
    """欠損の補間・体型の固定・再ポーズをして、変換の入力（motion dict）にする。"""
    import torch
    valid = np.asarray(raw['valid'], bool)
    if not valid.any():
        raise RuntimeError('人物を 1 人も検出できませんでした。区間や検出のしきい値を変えてみてください')
    T = len(valid)
    pose = fill_rotation_gaps(raw['pose'].reshape(T, N_JOINTS, 3), valid)
    betas = fill_gaps(raw['betas'], valid)
    trans = fill_gaps(raw['trans'], valid)
    joints = fill_gaps(raw['joints3d'], valid)
    betas_const = np.median(raw['betas'][valid], axis=0).astype(np.float32)
    bm = getattr(model.body_models, BODY_MODEL_NAME)

    def repose(p, b, tr, chunk=256):
        outs = []
        for st in range(0, len(p), chunk):
            sl = slice(st, st + chunk)
            pp = torch.from_numpy(np.ascontiguousarray(p[sl])).float().to(device)
            bb = torch.from_numpy(np.ascontiguousarray(b[sl])).float().to(device)
            tt = torch.from_numpy(np.ascontiguousarray(tr[sl])).float().to(device)
            with torch.inference_mode():
                res = bm(pose_rotvecs=pp.reshape(len(pp), -1), shape_betas=bb, trans=tt)
            outs.append(res['joints'].float().cpu().numpy() * 1000.0)
        return np.concatenate(outs)

    repose_ok = True
    try:
        # 体型を 1 本に固定しても骨盤の位置が推定のままになるように trans を直す
        zero_pose = np.zeros_like(pose)
        zero_trans = np.zeros_like(trans)
        rest_t = repose(zero_pose, betas, zero_trans)
        rest_c = repose(zero_pose[:1], betas_const[None], zero_trans[:1])
        trans = (trans + (rest_t[:, 0] - rest_c[0, 0]) / 1000.0).astype(np.float32)
        joints = repose(pose, np.tile(betas_const, (T, 1)), trans)
    except Exception as e:   # 再ポーズできないときは NLF の関節をそのまま使う
        log(f'体型を固定した再ポーズができませんでした（{e}）。NLF の関節をそのまま使います')
        repose_ok = False
    return dict(pose=pose.astype(np.float32), betas=np.tile(betas_const, (T, 1)).astype(np.float32),
                betas_const=betas_const, trans=trans.astype(np.float32), joints3d=joints.astype(np.float32),
                valid=valid, joint_uncertainty=raw['uncertainty'], fps=np.float32(fps),
                coord_system=np.array('camera'), repose_ok=np.bool_(repose_ok))


def expected_frames(start, end, fps):
    return max(1, int(math.floor((end - start) * fps + 1e-6)))
