"""変換の結果をアプリが読む形（JSON ＋ float32 のバイナリ）で書き出す。

conv/<キー>/result.json   メタデータ（ボーン・足ピタ・手のポーズ・口パク・情報・警告）
conv/<キー>/<種類>.bin     ボーンのキー列（全フレーム）。ボーンごとに、位置のキーを持つボーンは位置 (T×3)、
                          続けて回転 (T×4)。どれも MMD 座標の float32（リトルエンディアン）
"""
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

RESULT_FORMAT = 1


def write_tracks(path, tracks):
    """BoneTrack のリストをバイナリに書き、ボーンごとの位置（float32 の個数で数えたオフセット）を返す。"""
    layout = []
    offset = 0
    chunks = []
    for t in tracks:
        entry = dict(name=t.name, movable=bool(t.movable), offset=offset)
        if t.movable:
            chunks.append(np.asarray(t.positions, '<f4').reshape(-1))
            offset += chunks[-1].size
        chunks.append(np.asarray(t.rotations, '<f4').reshape(-1))
        offset += chunks[-1].size
        layout.append(entry)
    data = np.concatenate(chunks) if chunks else np.zeros(0, '<f4')
    with open(path, 'wb') as f:
        f.write(data.astype('<f4').tobytes())
    return layout


def segments_json(segments):
    return [[int(s), int(e)] for s, e in segments]


def write_result(out_dir, payload, variant_tracks):
    """payload（JSON にできる dict）と、種類ごとの BoneTrack のリストを out_dir に書く。

    書き終わるまで一時フォルダに書き、最後に名前を変える（途中で中止・失敗しても壊れた結果を残さない）。
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=out_dir.parent))
    try:
        variants = {}
        for kind, tracks in variant_tracks.items():
            if not tracks:
                continue
            file = f'{kind}.bin'
            T = len(tracks[0].rotations)
            variants[kind] = dict(file=file, frames=int(T), bones=write_tracks(tmp / file, tracks))
        payload = dict(payload, format=RESULT_FORMAT, variants=variants)
        (tmp / 'result.json').write_text(json_text(payload), encoding='utf-8')
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir / 'result.json'


def finite_json(o):
    """JSON にする値の numpy の値を Python の値に、NaN・無限大を None（null）にする。

    json.dumps は NaN を NaN のまま書くが、アプリ（JavaScript の JSON.parse）はそれを読めない。
    np.float64 は float の派生なので、json.dumps の default を通らずに NaN のまま書かれてしまう。
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return finite_json(o.tolist())
    if isinstance(o, np.generic):
        return finite_json(o.item())
    return o


def json_text(data):
    """data を JSON の文字列にする（numpy の値も。NaN・無限大は null）。"""
    return json.dumps(finite_json(data), ensure_ascii=False, allow_nan=False)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json_text(data), encoding='utf-8')
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
