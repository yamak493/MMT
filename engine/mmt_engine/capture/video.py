"""動画の情報と、区間のフレーム・音声の取り出し（imageio-ffmpeg に同梱の ffmpeg を使う）。"""
import math
import re
import subprocess

import numpy as np

_CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, 'STARTUPINFO') else 0


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def run_ffmpeg(args, check=True):
    p = subprocess.run([ffmpeg_exe(), '-hide_banner', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       creationflags=_CREATE_NO_WINDOW)
    text = p.stdout.decode('utf-8', errors='replace')
    if check and p.returncode != 0:
        raise RuntimeError('ffmpeg の実行に失敗しました:\n' + text[-2000:])
    return text


def probe(path):
    """動画の情報 dict(fps, duration, width, height, has_audio, rotation)。"""
    info = run_ffmpeg(['-i', str(path)], check=False)
    fps = 30.0
    m = re.search(r'Stream #.*?Video:.*?,\s*([\d.]+)\s*fps', info)
    if m:
        fps = float(m.group(1))
    else:
        m = re.search(r'Stream #.*?Video:.*?,\s*([\d.]+)\s*tbr', info)
        if m:
            fps = float(m.group(1))
    duration = 0.0
    m = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    width = height = 0
    m = re.search(r'Stream #.*?Video:.*?(\d{2,5})x(\d{2,5})', info)
    if m:
        width, height = int(m.group(1)), int(m.group(2))
    rotation = 0
    m = re.search(r'rotate\s*:\s*(-?\d+)', info) or re.search(r'displaymatrix: rotation of (-?[\d.]+) degrees', info)
    if m:
        rotation = int(round(float(m.group(1)))) % 360
    if rotation in (90, 270):
        width, height = height, width
    if width <= 0 or height <= 0:
        raise RuntimeError('動画の映像を読めませんでした（対応していない形式の可能性があります）')
    return dict(fps=normalize_fps(fps), duration=duration, width=width, height=height,
                has_audio=bool(re.search(r'Stream #.*?Audio:', info)), rotation=rotation)


def normalize_fps(fps):
    """動画の情報の fps（23.98・29.97・59.94 と丸められている）を NTSC の値に戻す。"""
    for base in (24, 30, 60, 120):
        if abs(fps - base * 1000 / 1001) < 0.01:
            return base * 1000 / 1001
    return float(fps)


def auto_fps(src_fps):
    """元の fps の整数分の 1 のうち 30 以下で最も高いもの（24・25・30 はそのまま、50 → 25、60 → 30）。"""
    return src_fps / max(1, math.ceil(src_fps / 30.0 - 1e-6))


def scaled_size(width, height, max_height):
    """高さが max_height を超えるときだけ縮小した大きさ（2 の倍数）。"""
    if max_height and height > max_height:
        k = max_height / height
        return int(width * k // 2 * 2), int(max_height // 2 * 2)
    return int(width // 2 * 2), int(height // 2 * 2)


def crop_rect(width, height, crop):
    """切り抜く範囲 dict(x, y, w, h)（回転を適用した映像の画素）を、映像の内側の整数 (x, y, w, h) にする。

    幅・高さは 2 の倍数に切り下げる（正方形は正方形のまま）。切り抜かない・全体と同じなら None。
    """
    if not crop:
        return None
    try:
        x, y, w, h = (float(crop[k]) for k in ('x', 'y', 'w', 'h'))
    except (KeyError, TypeError, ValueError):
        return None
    x = int(round(min(max(x, 0.0), width - 16)))
    y = int(round(min(max(y, 0.0), height - 16)))
    w = int(min(round(w), width - x)) // 2 * 2
    h = int(min(round(h), height - y)) // 2 * 2
    if w < 16 or h < 16:
        return None
    if x == 0 and y == 0 and w >= width // 2 * 2 and h >= height // 2 * 2:
        return None
    return x, y, w, h


class FrameReader:
    """区間 [start, end) の映像を fps で、(H, W, 3) uint8 の RGB として順に読む。

    crop = (x, y, w, h) なら、その範囲を切り抜いてから size に縮小する。
    """

    def __init__(self, path, start, end, fps, size, crop=None):
        self.width, self.height = size
        self.frame_bytes = self.width * self.height * 3
        vf = f'fps={fps},scale={self.width}:{self.height}:flags=bicubic'
        if crop:
            vf = 'crop={2}:{3}:{0}:{1},'.format(*crop) + vf
        args = [ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-ss', f'{start:.3f}', '-to', f'{end:.3f}',
                '-i', str(path), '-an', '-vf', vf,
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
        # 標準エラーはエンジンの標準エラー（アプリが読み続けて engine.log に書く）へそのまま流す。
        # PIPE にして読まないと、壊れた動画でエラーが続いたときにパイプが詰まり、ffmpeg も読み込みも止まってしまう
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None,
                                     creationflags=_CREATE_NO_WINDOW, bufsize=self.frame_bytes * 4)

    def __iter__(self):
        return self

    def __next__(self):
        buf = self.proc.stdout.read(self.frame_bytes)
        if buf is None or len(buf) < self.frame_bytes:
            raise StopIteration
        return np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_audio(path, start, end, out_wav, sample_rate=44100, channels=2):
    """区間の音声を 16bit PCM の WAV に書き出す。"""
    run_ffmpeg(['-y', '-loglevel', 'error', '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(path), '-vn',
                '-ac', str(channels), '-ar', str(sample_rate), '-c:a', 'pcm_s16le', str(out_wav)])
    return out_wav


def waveform_peaks(path, duration, rate=100):
    """動画全体の音声の波形（rate [個/秒] ごとの最大振幅 0〜1、float32）。音声が無ければ空。"""
    sr = 8000
    args = [ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-i', str(path), '-vn', '-ac', '1', '-ar', str(sr),
            '-f', 's16le', 'pipe:1']
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)
    data = np.frombuffer(p.stdout, '<i2')
    if len(data) == 0:
        return np.zeros(0, np.float32)
    hop = sr // rate
    n = int(math.ceil(len(data) / hop))
    pad = np.zeros(n * hop, np.int16)
    pad[:len(data)] = data
    peaks = np.abs(pad.reshape(n, hop).astype(np.float32)).max(axis=1) / 32768.0
    return peaks.astype(np.float32)
