"""キャプチャ（動画 → モーション）と変換（モーション → MMD のキー列）の処理。"""
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .body.smpl import BodyModel
from .convert.center import ReachGeometry
from .convert.foot_ik import FootIKResult
from .convert.foot_keys import FOOT_KEYS_VERSION, key_frames, normalize_keys, place_feet, splice_tracks
from .convert.pipeline import convert, prepare
from .convert.tracks import build_tracks
from .convert.variants import LOWER_BODY_BONE, VARIANTS, motion_tracks, variant_motion
from .hands import shapes as hand_shapes
from .jobs import UserError
from .lipsync.keys import build_lipsync
from .output import json_text, load_npz, save_npz, segments_json, write_json, write_result
from .rig.skeleton import Skeleton
from .settings import make_config

BODY_MODEL_FILE = 'smpl_body_model.npz'


class LRUCache(OrderedDict):
    """最近使ったものから maxsize 個だけ覚える dict（エンジンを長く使っても、メモリが増え続けないようにする）。"""

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key not in self:
            return default
        self.move_to_end(key)
        return self[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


# 撮影（take）のフォルダ → PreparedMotion（変換をやり直すときに使い回す）。無ければ motion.npz から作り直す
_PREPARED_CACHE = LRUCache(2)
_BODY_CACHE = {}
# 撮影のフォルダ → TakeBase（足ピタのキーを重ねる前の変換。足ピタの編集で使い回す）。無ければ foot_base.npz から読む
_BASE_CACHE = LRUCache(8)
_DIGEST_CACHE = LRUCache(256)   # ファイルのパス → ((大きさ, 更新日時), 中身のハッシュ)
BASE_FILE = 'foot_base.npz'   # 撮影のフォルダに残す TakeBase
BASE_FORMAT = 1


def models_dir(params):
    d = params.get('modelsDir') or os.environ.get('MMT_MODELS_DIR')
    if not d:
        raise UserError('モデルの保存先が指定されていません')
    return Path(d)


def nlf_model_path(params):
    from .capture.nlf_runner import MODEL_FILENAME
    return models_dir(params) / 'nlf' / MODEL_FILENAME


def body_model_path(params):
    return models_dir(params) / 'nlf' / BODY_MODEL_FILE


def load_body_model(params):
    path = body_model_path(params)
    if not path.exists():
        raise UserError('SMPL の体モデルがありません。NLF のセットアップ（動画のキャプチャ）を先に行ってください')
    key = (str(path), path.stat().st_mtime)
    if key not in _BODY_CACHE:
        _BODY_CACHE.clear()
        _BODY_CACHE[key] = BodyModel.from_npz(path)
    return _BODY_CACHE[key]


def load_skeleton(rig_path):
    if not rig_path:
        return Skeleton.standard(), 'standard'
    text = Path(rig_path).read_text(encoding='utf-8')
    return Skeleton.from_rig(json.loads(text)), hashlib.sha1(text.encode('utf-8')).hexdigest()


def resolve_fps(value, src_fps):
    from .capture.video import auto_fps, normalize_fps
    if value in (None, '', 'auto'):
        return auto_fps(src_fps)
    if value == 'source':
        return src_fps
    fps = float(value)
    if fps <= 0:
        raise UserError('推論する fps は正の数にしてください')
    return normalize_fps(fps)


# ================================================================ キャプチャ

def capture(job, params):
    """動画の区間からモーションを抽出して撮影（take）のフォルダに保存し、変換まで行う。"""
    from .capture import nlf_runner, video
    take_dir = Path(params['takeDir'])
    take_dir.mkdir(parents=True, exist_ok=True)
    video_path = params['video']
    info = video.probe(video_path)
    start = max(0.0, float(params.get('start', 0.0)))
    end = float(params.get('end') or 0.0)
    if end <= 0 or (info['duration'] and end > info['duration']):
        end = info['duration']
    if end - start < 0.2:
        raise UserError('キャプチャする区間が短すぎます（0.2 秒以上にしてください）')
    fps = resolve_fps(params.get('fps', 'auto'), info['fps'])
    # 切り抜くときは、切り抜いた範囲を元の映像として扱う（NLF・手の検出とも同じ画像の座標になる）
    crop = video.crop_rect(info['width'], info['height'], params.get('crop'))
    src_size = (crop[2], crop[3]) if crop else (info['width'], info['height'])
    size = video.scaled_size(src_size[0], src_size[1], int(params.get('maxHeight') or 0))
    n_expected = nlf_runner.expected_frames(start, end, fps)
    hands_cfg = params.get('hands') or {}
    lip_cfg = params.get('lipsync') or {}
    use_hands = bool(hands_cfg.get('enabled', True))
    use_lipsync = bool(lip_cfg.get('enabled', True)) and info['has_audio']
    stages = [('load', 4), ('nlf', 55), ('finalize', 2), ('prepare', 5)]
    if use_hands:
        stages.append(('hands', 14))
    stages.append(('audio', 1))
    if use_lipsync:
        stages.append(('lipsync', 10))
    stages.append(('convert', 9))
    job.set_stages(stages)

    # ---- NLF の読み込み ----
    job.stage('load', 'NLF のモデルを読み込んでいます')
    model_path = nlf_model_path(params)
    if not model_path.exists():
        raise UserError('NLF の学習済みモデルがありません。セットアップをやり直してください')
    device = nlf_runner.select_device(params.get('device', 'auto'))
    if device == 'cpu':
        job.log('GPU（CUDA）が見つからないので CPU で推論します。非常に時間がかかります', 'warn')
    model, precision = nlf_runner.load_model(model_path, device, job.log, params.get('precision', 'auto'))
    bm_path = body_model_path(params)
    if not bm_path.exists():
        nlf_runner.body_model_from(model).save_npz(bm_path)
    job.check()

    # ---- NLF の推論 ----
    job.stage('nlf', f'モーションを抽出しています（{n_expected} フレーム）')
    t0 = time.time()

    def nlf_progress(done, total):
        el = time.time() - t0
        eta = el / max(done, 1) * max(total - done, 0)
        job.progress(done / max(total, 1), f'モーションを抽出しています（{done} / {total} フレーム・残り約 {_fmt_sec(eta)}）')

    settings = dict(batch_size=params.get('batchSize', 16), num_aug=params.get('numAug', 3),
                    antialias=params.get('antialias', 2), detector_threshold=params.get('detectorThreshold', 0.25),
                    person_select=params.get('personSelect', 'largest'))
    with video.FrameReader(video_path, start, end, fps, size, crop) as frames:
        raw = nlf_runner.run_inference(model, frames, n_expected, fps, size, settings, device, nlf_progress, job.check)
    if raw['reacquired']:
        sec = ', '.join(f'{start + f / fps:.1f}' for f in raw['reacquired'][:5])
        job.log(f'追跡している人を長く見失ったので、人物を選び直しました（{sec} 秒）。別の人に変わっていないか確かめてください',
                'warn')
    job.check()
    job.stage('finalize', '抽出したモーションを整えています')
    motion = nlf_runner.finalize_motion(raw, model, device, fps, job.log)
    save_npz(take_dir / 'motion.npz', **motion)
    valid_ratio = float(np.mean(motion['valid']))
    job.check()

    # ---- ステージ1〜2（手を切り出す位置にも使う） ----
    job.stage('prepare', '外れフレームの除外・ジッター制御')
    bm = BodyModel.from_npz(bm_path)
    cfg = make_config(params.get('settings') or {})
    prepared = prepare(motion, bm, cfg, cancel=job.check)
    _PREPARED_CACHE[str(take_dir)] = prepared
    job.check()

    # ---- 手のランドマーク ----
    hands_ok = False
    if use_hands:
        job.stage('hands', '手のランドマークを検出しています')
        hands_ok = _detect_hands(job, params, prepared, video_path, src_size, crop, start, end, fps, size,
                                  take_dir)

    # ---- 音声（波形・口パク） ----
    job.stage('audio', '音声を読み込んでいます')
    waveform_path = params.get('waveformPath')
    if waveform_path and not Path(waveform_path).exists() and info['has_audio']:
        peaks = video.waveform_peaks(video_path, info['duration'])
        Path(waveform_path).parent.mkdir(parents=True, exist_ok=True)
        with open(waveform_path, 'wb') as f:
            f.write(peaks.astype('<f4').tobytes())
    lipsync_ok = False
    if use_lipsync:
        job.stage('lipsync', '口パクを解析しています')
        lipsync_ok = _analyze_lipsync(job, params, lip_cfg, video_path, start, end, take_dir, device)

    capture_info = dict(engineVersion=__version__, video=str(video_path), videoInfo=info, start=start, end=end,
                        fps=fps, size=list(size), crop=list(crop) if crop else None, frames=int(len(motion['valid'])), validRatio=round(valid_ratio, 4),
                        reacquired=[round(start + f / fps, 3) for f in raw['reacquired']], hands=hands_ok,
                        lipsync=lipsync_ok, device=device, precision=precision, settings=params.get('settings') or {},
                        capturedAt=time.strftime('%Y-%m-%dT%H:%M:%S'))
    write_json(take_dir / 'capture.json', capture_info)

    # ---- 変換 ----
    job.stage('convert', 'MMD のモーションに変換しています')
    result = convert_take(job, dict(params, takeDir=str(take_dir)), progress_scale=True)
    result['capture'] = capture_info
    return result


def _fmt_sec(sec):
    sec = int(round(sec))
    if sec >= 3600:
        return f'{sec // 3600} 時間 {sec % 3600 // 60} 分'
    if sec >= 60:
        return f'{sec // 60} 分 {sec % 60} 秒'
    return f'{sec} 秒'


def _detect_hands(job, params, prepared, video_path, src_size, crop, start, end, fps, size, take_dir):
    from .capture import video
    from .hands import landmarks
    task = models_dir(params) / 'mediapipe' / 'hand_landmarker.task'
    if not task.exists():
        job.log('MediaPipe Hands のモデルが無いので、手の形・手首の向きは推定しません', 'warn')
        return False
    hcfg = params.get('hands') or {}
    try:
        model = landmarks.HandLandmarkModel(task)
    except Exception as e:
        job.log(f'手の検出を準備できませんでした（{e}）', 'warn')
        return False
    # 元の動画の解像度が高ければ、手を切り出すときだけ縮小せずに読む（手が大きく写るほど判定が安定する）
    hand_size = (src_size[0] // 2 * 2, src_size[1] // 2 * 2) if src_size[1] > size[1] else size
    roi_m = float(hcfg.get('roiM', 0.25))
    rois, wrists = landmarks.joint_rois(prepared.joints_camera_mm(True), hand_size, roi_m=roi_m)
    _, wrists_raw = landmarks.joint_rois(prepared.joints_camera_mm(False), hand_size, roi_m=roi_m)
    T = len(rois)

    def prog(done, total):
        job.progress(done / max(total, 1), f'手のランドマークを検出しています（{done} / {total} フレーム）')

    with video.FrameReader(video_path, start, end, fps, hand_size, crop) as frames:
        out = landmarks.detect_video(frames, rois, wrists, model, passes=int(hcfg.get('passes', 2)), progress=prog,
                                     wrists_raw=wrists_raw, cancel=job.check)
    if len(out['presence']) < T:
        job.log('手の検出で読めたフレームが足りませんでした', 'warn')
    save_npz(take_dir / 'hands.npz', screen=out['screen'], world=out['world'], presence=out['presence'],
             handedness=out['handedness'], roi=out['roi'], fps=np.float64(fps),
             image_size=np.array(hand_size, np.int64))
    visible = (out['presence'] >= 0.5).mean(axis=0) * 100 if len(out['presence']) else [0, 0]
    job.log(f'手が見えたフレーム: 左手 {visible[0]:.0f}% / 右手 {visible[1]:.0f}%')
    return True


def _analyze_lipsync(job, params, lip_cfg, video_path, start, end, take_dir, device):
    from .capture import video
    from .lipsync.analysis import analyze
    tmp = Path(tempfile.mkdtemp(prefix='mmt-audio-'))
    try:
        wav = video.extract_audio(lip_cfg.get('audioPath') or video_path, start, end, tmp / 'segment.wav')
        result = analyze(wav, tmp, device, use_demucs=bool(lip_cfg.get('demucs', True)),
                         demucs_model=lip_cfg.get('demucsModel', 'htdemucs'), lang=lip_cfg.get('lang', 'jpn'),
                         emit=float(lip_cfg.get('emit', 1.0)), progress=job.progress, cancel=job.check, log=job.log)
        save_npz(take_dir / 'lipsync.npz', **result)
        return True
    except UserError as e:
        job.log(str(e), 'warn')
        return False
    except Exception as e:
        if job.cancelled:
            raise
        job.log(f'口パクを解析できませんでした（{e}）', 'warn')
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================ 変換

def _file_digest(path):
    """ファイルの中身のハッシュ（大きさ・更新日時が同じ間は覚えておく）。"""
    st = path.stat()
    sig = (st.st_size, st.st_mtime_ns)
    cached = _DIGEST_CACHE.get(str(path))
    if cached is None or cached[0] != sig:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        cached = (sig, h.hexdigest())
        _DIGEST_CACHE[str(path)] = cached
    return cached[1]


def conversion_key(take_dir, rig_hash, settings, overrides):
    """変換の条件のキー。撮影のファイルは中身で見る（.mmt から開き直すと更新日時が変わるため）。"""
    h = hashlib.sha1()
    for name in ('motion.npz', 'hands.npz', 'lipsync.npz'):
        p = Path(take_dir) / name
        if p.exists():
            h.update(f'{name}:{_file_digest(p)}'.encode())
    cond = dict(rig=rig_hash, settings=settings or {}, overrides=overrides or [], engine=__version__)
    if overrides:
        cond['foot_keys_version'] = FOOT_KEYS_VERSION
    h.update(json.dumps(cond, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    return h.hexdigest()[:20]


@dataclass
class TakeBase:
    """足ピタのキーを重ねる前の変換の結果（足ピタの編集では、これにキーを重ねるだけで変換はやり直さない）。"""
    key: str
    skeleton: object
    config: object
    scale: float
    geom: object                 # ReachGeometry
    lower_rot: np.ndarray        # (T, 3, 3) 下半身の大域回転
    knee_poles: np.ndarray
    parts: dict                  # 出力の種類 → (センターの差分, 足ＩＫ, ローカル回転, キー列)
    payload: dict                # result.json の中身（key・footKeys を除く）


def convert_take(job, params, progress_scale=False):
    """撮影（take）のモーションを、モデルのリグ・設定で変換し、足ピタのキーを重ねて conv/<キー>/ に書き出す。

    足ピタのキー（footKeys）だけが変わったときは、キャッシュした基準の変換にキーを重ね直すだけ（キーの前後の
    フレームのほかは計算しない）。
    """
    take_dir = Path(params['takeDir'])
    motion_path = take_dir / 'motion.npz'
    if not motion_path.exists():
        raise UserError('撮影のモーションが見つかりません（プロジェクトのファイルが壊れている可能性があります）')
    settings = params.get('settings') or {}
    keys = normalize_keys(params.get('footKeys'))
    skel, rig_hash = load_skeleton(params.get('rigPath'))
    key = conversion_key(take_dir, rig_hash, settings, keys)
    out_dir = take_dir / 'conv' / key
    if (out_dir / 'result.json').exists() and not params.get('force'):
        return dict(resultPath=str(out_dir / 'result.json'), key=key, cached=True)
    if not progress_scale:
        job.set_stages([("convert", 1)])
        job.stage('convert', 'MMD のモーションに変換しています')
    base_key = conversion_key(take_dir, rig_hash, settings, [])
    base = _BASE_CACHE.get(str(take_dir))
    if base is None or base.key != base_key or params.get('force'):
        _BASE_CACHE.pop(str(take_dir), None)
        base = None if params.get('force') else _load_base(take_dir, base_key, skel, settings)
        if base is None:
            base = _convert_base(job, params, take_dir, skel, settings, base_key)
            _save_base(job, take_dir, base)
        _BASE_CACHE[str(take_dir)] = base
    tracks = {kind: part[3] for kind, part in base.parts.items()}
    if keys:
        job.progress(0.95, '足ピタを反映しています')
        tracks = _apply_foot_keys(base, keys)
    payload = dict(base.payload, key=key, footKeys=keys, footKeysVersion=FOOT_KEYS_VERSION)
    job.progress(0.97, '結果を書き出しています')
    path = write_result(out_dir, payload, tracks)
    _prune_conversions(take_dir, keep=int(params.get('keepConversions', 24)))
    job.progress(1.0, '変換が完了しました')
    return dict(resultPath=str(path), key=key, cached=False)


def _apply_foot_keys(base, keys):
    """基準の変換の各出力の種類のキー列に、足ピタのキーを重ねたもの（キーの前後のフレームだけ作り直す）。"""
    frames = key_frames(keys, len(base.lower_rot))
    out = {}
    for kind, (center_delta, ik, local, tracks) in base.parts.items():
        if center_delta is None or ik is None:
            out[kind] = tracks
            continue
        center_delta, ik, changed = place_feet(center_delta, ik, frames, base.lower_rot, base.geom, base.config,
                                               base.scale)
        out[kind] = splice_tracks(tracks, base.skeleton, center_delta, ik, local, base.config, base.knee_poles,
                                  changed)
    return out


def _convert_base(job, params, take_dir, skel, settings, base_key):
    """足ピタのキーを重ねる前の変換（TakeBase）。"""
    cfg = make_config(settings)
    bm = load_body_model(params)
    motion = load_npz(take_dir / 'motion.npz')
    hands = load_npz(take_dir / 'hands.npz') if (take_dir / 'hands.npz').exists() else None
    lip = load_npz(take_dir / 'lipsync.npz') if (take_dir / 'lipsync.npz').exists() else None
    prepared = _PREPARED_CACHE.get(str(take_dir))
    for old in take_dir.glob('arm_fix_*.npz'):    # 以前の版が足ピタの編集用に残したもの
        old.unlink(missing_ok=True)

    def prog(fraction, message):
        job.progress(0.8 * fraction, message)

    res = convert(motion, bm, skel, cfg, prepared=prepared, hands_analysis=hands, progress=prog, cancel=job.check)
    T = len(res.contact.flags)
    parts = {}
    extra = {}
    for i, kind in enumerate(VARIANTS):
        job.check()
        job.progress(0.8 + 0.15 * i / len(VARIANTS), '出力の種類ごとのキーを作っています')
        center_delta, ik, local, contact, info = variant_motion(res, kind)
        parts[kind] = (center_delta, ik, local, motion_tracks(res, kind, center_delta, ik, local))
        extra[kind] = dict(info=info, contacts=None if contact is None else
                           [segments_json(contact.segments[f]) for f in range(2)])

    # ---- 手のポーズ（形が変わる所）と、形ごとの指の回転 ----
    hand_payload = dict(available=False)
    warns = list(res.warnings)
    if hands is not None:
        labels, visible = hand_shapes.classify_analysis(hands, cfg.hands, res.fps, T)
        rotations = hand_shapes.shape_rotation_table(skel, cfg.hands, warns.append) if skel.source != 'standard' else {}
        hand_payload = dict(available=True, visible=visible,
                            events={side: hand_shapes.label_events(labels[:, i])
                                    for i, side in enumerate(('left', 'right'))},
                            rotations=rotations, shapes=list(hand_shapes.SHAPES), labels=hand_shapes.LABELS,
                            transitionSec=float(cfg.hands.transition_sec))
    elif skel.source != 'standard':
        hand_payload = dict(available=False, rotations=hand_shapes.shape_rotation_table(skel, cfg.hands),
                            shapes=list(hand_shapes.SHAPES), labels=hand_shapes.LABELS,
                            transitionSec=float(cfg.hands.transition_sec))

    # ---- 口パク ----
    lip_payload = dict(available=False)
    if lip is not None:
        lip_tracks, _, lip_info, lip_warns = build_lipsync(lip, cfg.lipsync, res.fps, T,
                                                           skel.morph_names if skel.source != 'standard' else None)
        warns.extend(lip_warns)
        lip_payload = dict(available=True, tracks=lip_tracks, info=lip_info)

    payload = dict(engineVersion=__version__, frames=T, fps=res.fps, modelName=skel.model_name,
                   contacts=dict(auto=[segments_json(res.auto_contact.segments[f]) for f in range(2)],
                                 effective=[segments_json(res.contact.segments[f]) for f in range(2)]),
                   jumps=res.info['jumps'], variantsInfo=extra, hands=hand_payload, lipsync=lip_payload,
                   info=res.info, warnings=warns)
    lower_rot = res.retargeter.global_matrix(LOWER_BODY_BONE, res.kin.glob_rot)
    return TakeBase(base_key, skel, cfg, float(res.scale), res.reach_geometry, lower_rot, res.knee_poles, parts,
                    json.loads(json_text(payload)))


# ---- 足ピタのキーを重ねる前の変換を、撮影のフォルダに残す ----
# .mmt にも入るので、アプリを起動し直しても・プロジェクトを開き直しても、足ピタの編集で変換をやり直さない。
# キー列は持たず、センター・足ＩＫ・ローカル回転から作り直す（同じ関数・同じ値なので、作った時と同じになる）

def _text_array(text):
    return np.frombuffer(text.encode('utf-8'), np.uint8)


def _array_text(arr):
    return np.asarray(arr, np.uint8).tobytes().decode('utf-8')


def _save_base(job, take_dir, base):
    try:
        local = base.parts['full'][2]
        names = list(local)
        arrays = dict(format=np.int64(BASE_FORMAT), key=_text_array(base.key), payload=_text_array(json_text(base.payload)),
                      scale=np.float64(base.scale), lower_rot=base.lower_rot, knee_poles=np.asarray(base.knee_poles),
                      geom_lower=base.geom.lower, geom_hips=base.geom.hips, geom_ik=base.geom.ik,
                      geom_leg_length=base.geom.leg_length,
                      local_names=_text_array('\n'.join(names)), local=np.stack([local[n] for n in names]))
        for kind, (center_delta, ik, loc, _) in base.parts.items():
            if center_delta is None:
                continue
            if loc is not local:
                raise ValueError(f'{kind} のローカル回転がフルと違います')
            arrays[f'{kind}_center'] = np.asarray(center_delta)
            arrays[f'{kind}_ik_delta'] = np.asarray(ik.delta)
            arrays[f'{kind}_ik_rotation'] = np.asarray(ik.rotation)
            arrays[f'{kind}_ik_target'] = np.asarray(ik.target)
        upper = base.parts['upper_body'][2]
        own = [n for n in upper if upper[n] is not local.get(n)]
        arrays['upper_names'] = _text_array('\n'.join(upper))
        arrays['upper_own_names'] = _text_array('\n'.join(own))
        arrays['upper_own'] = np.stack([upper[n] for n in own]) if own else np.zeros((0, len(base.lower_rot), 4))
        save_npz(take_dir / BASE_FILE, **arrays)
    except Exception as e:   # 残せなくても変換は続ける（次に足ピタを編集したときに変換し直すだけ）
        job.log(f'足ピタの編集用のデータを保存できませんでした（{e}）', 'warn')


def _load_base(take_dir, base_key, skel, settings):
    path = take_dir / BASE_FILE
    if not path.exists():
        return None
    try:
        d = load_npz(path)
        if int(d['format']) != BASE_FORMAT or _array_text(d['key']) != base_key:
            return None
        cfg = make_config(settings)
        poles = d['knee_poles']
        names = _array_text(d['local_names']).split('\n')
        local = {n: d['local'][i] for i, n in enumerate(names)}
        own_names = [n for n in _array_text(d['upper_own_names']).split('\n') if n]
        own = {n: d['upper_own'][i] for i, n in enumerate(own_names)}
        upper = {n: own[n] if n in own else local[n] for n in _array_text(d['upper_names']).split('\n') if n}
        parts = {}
        for kind in VARIANTS:
            if kind == 'upper_body':
                parts[kind] = (None, None, upper, build_tracks(skel, None, None, upper, cfg))
                continue
            center_delta = d[f'{kind}_center']
            ik = FootIKResult(d[f'{kind}_ik_target'], d[f'{kind}_ik_delta'], d[f'{kind}_ik_rotation'], None, None, [], 0)
            parts[kind] = (center_delta, ik, local, build_tracks(skel, center_delta, ik, local, cfg, poles))
        geom = ReachGeometry(d['geom_lower'], d['geom_hips'], d['geom_ik'], d['geom_leg_length'])
        return TakeBase(base_key, skel, cfg, float(d['scale']), geom, d['lower_rot'], poles, parts,
                        json.loads(_array_text(d['payload'])))
    except Exception:
        return None   # 壊れている・古い形: 変換し直す


def _prune_conversions(take_dir, keep):
    """古い変換結果を消す（足ピタの編集のたびに増えるので、新しいものから keep 個だけ残す）。"""
    conv = Path(take_dir) / 'conv'
    if not conv.exists():
        return
    dirs = sorted((d for d in conv.iterdir() if d.is_dir() and not d.name.startswith('.')),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def hand_rotations(params):
    """モデルのリグから、形ごとの指ボーンの回転（MMD 座標）。"""
    skel, _ = load_skeleton(params.get('rigPath'))
    cfg = make_config(params.get('settings') or {})
    warns = []
    table = hand_shapes.shape_rotation_table(skel, cfg.hands, warns.append) if skel.source != 'standard' else {}
    return dict(rotations=table, shapes=list(hand_shapes.SHAPES), labels=hand_shapes.LABELS, warnings=warns,
                transitionSec=float(cfg.hands.transition_sec))


def system_info():
    """Python・PyTorch・GPU の情報。"""
    import platform
    out = dict(engineVersion=__version__, python=platform.python_version())
    try:
        import torch
        out['torch'] = torch.__version__
        out['cuda'] = bool(torch.cuda.is_available())
        out['cudaVersion'] = torch.version.cuda
        if torch.cuda.is_available():
            from .capture.nlf_runner import gpu_info
            out['gpu'] = gpu_info()
    except Exception as e:
        out['torch'] = None
        out['torchError'] = str(e)
    return out


def probe_video(params):
    from .capture import video
    info = video.probe(params['path'])
    info['autoFps'] = resolve_fps('auto', info['fps'])
    return info


def expected_cost(params):
    """キャプチャのフレーム数の見積もり。"""
    fps = float(params.get('fps', 30.0))
    return dict(frames=int(math.floor((float(params['end']) - float(params['start'])) * fps)))
