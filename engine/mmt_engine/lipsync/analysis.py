"""口パクの解析: 動画の音声から Demucs でボーカルを取り出し、Allosaurus で音素とその時刻を認識する。

伴奏が混ざったままだと音素の認識が大きく崩れるので、先にボーカルを分離する。Allosaurus は言語を jpn にすると
日本語の音素だけで認識するので、母音がそのまま「あいうえお」に分かれる（英語の歌でも使える）。
"""
import math
import wave
from pathlib import Path

import numpy as np

from .keys import parse_allosaurus, vocal_level_db

LEVEL_RATE = 100.0


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768.0
    return sr, data.reshape(-1, ch)


def write_wav(path, sr, samples):
    samples = np.asarray(samples, np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def separate_vocals(mix, sr, device, model_name='htdemucs', progress=None, cancel=None):
    """Demucs でボーカルだけを取り出す。mix: (サンプル数, 2) float32。"""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    separator = get_model(model_name)
    separator.eval()
    if separator.samplerate != sr:
        raise RuntimeError(f'{model_name} は {separator.samplerate} Hz のモデルです')
    wav = torch.from_numpy(np.ascontiguousarray(mix.T))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    # 長い音声は 30 秒ずつに分けて分離する（進捗の表示と中止のため）。境目で音が途切れないように、前後 3 秒ずつ
    # 余分に分離して、真ん中だけを使う
    chunk, pad = 30 * sr, 3 * sr
    pieces = []
    total = wav.shape[1]
    vocal_index = separator.sources.index('vocals')
    with torch.no_grad():
        for st in range(0, total, chunk):
            if cancel is not None:
                cancel()
            a, b = max(0, st - pad), min(total, st + chunk + pad)
            seg = ((wav[:, a:b] - mean) / std)[None]
            src = apply_model(separator, seg, device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
            vocals = (src[vocal_index] * std + mean).cpu().numpy().T
            pieces.append(vocals[st - a:st - a + min(chunk, total - st)])
            if progress is not None:
                progress(min(st + chunk, total), total)
    del separator
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(pieces, axis=0)


def recognize_phones(vocals, sr, work_dir, lang='jpn', emit=1.0):
    """Allosaurus で音素とその時刻を認識する。戻り値 (時刻, 音素)。"""
    from allosaurus.app import read_recognizer
    from scipy.signal import resample_poly
    g = math.gcd(16000, sr)
    mono = resample_poly(np.asarray(vocals, np.float64).mean(axis=1), 16000 // g, sr // g)
    path = Path(work_dir) / 'vocals_16k.wav'
    write_wav(path, 16000, mono)
    recognizer = read_recognizer()
    if lang != 'ipa' and not recognizer.is_available(lang):
        lang = 'ipa'
    text = recognizer.recognize(str(path), lang, timestamp=True, emit=float(emit))
    try:
        path.unlink()
    except OSError:
        pass
    return parse_allosaurus(text)


def analyze(audio_wav, work_dir, device, use_demucs=True, demucs_model='htdemucs', lang='jpn', emit=1.0,
            progress=None, cancel=None, log=None):
    """口パクの解析結果 dict(phone_times, phones, level_db, level_rate, duration)。"""
    sr, mix = read_wav(audio_wav)
    duration = len(mix) / sr
    if len(mix) == 0:
        raise RuntimeError('区間に音声がありません')
    if mix.shape[1] == 1:
        mix = np.repeat(mix, 2, axis=1)
    if use_demucs:
        if log:
            log(f'Demucs（{demucs_model}）でボーカルを分離しています')
        vocals = separate_vocals(mix, sr, device, demucs_model,
                                 progress=(lambda d, t: progress(0.7 * d / t)) if progress else None, cancel=cancel)
    else:
        vocals = mix
    if cancel is not None:
        cancel()
    if progress:
        progress(0.75)
    if log:
        log('Allosaurus で音素を認識しています')
    times, phones = recognize_phones(vocals, sr, work_dir, lang, emit)
    if progress:
        progress(1.0)
    return dict(phone_times=times, phones=phones, level_db=vocal_level_db(vocals, sr, rate=LEVEL_RATE).astype(np.float32),
                level_rate=np.float64(LEVEL_RATE), duration=np.float64(duration))
