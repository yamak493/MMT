"""実行環境の確認。

python -m mmt_engine --driver-check: PyTorch を入れる前に、NVIDIA のドライバーと GPU が PyTorch の GPU 版に
    対応しているかを CUDA のドライバー API（nvcuda.dll）で確かめる。
python -m mmt_engine --selftest: 必要なライブラリを読み込めるか、GPU で計算できるかを確かめる。
どちらも結果を JSON で出力する。
"""
import importlib
import json
import sys

REQUIRED = ('numpy', 'scipy', 'imageio_ffmpeg', 'torch', 'torchvision')
OPTIONAL = ('ai_edge_litert', 'demucs', 'allosaurus')
# アプリが入れる PyTorch（2.7.1 の cu128 版。src/main/setup.ts の TORCH_BACKEND）が必要とするもの
MIN_DRIVER_CUDA = (12, 8)       # ドライバーが対応する CUDA の版
MIN_CAPABILITY = (5, 0)         # GPU の世代（Compute Capability。sm_50 = GTX 900 番台以降）
DRIVER_URL = 'https://www.nvidia.com/ja-jp/drivers/'


def _check_driver():
    """エラーの文言（問題が無ければ None）と、分かったこと（GPU の名前など）を返す。"""
    import ctypes
    info = {}
    try:
        cuda = ctypes.WinDLL('nvcuda.dll')
    except OSError:
        return ('NVIDIA の GPU のドライバーが見つかりません。このアプリには NVIDIA の GPU（GeForce GTX 900 番台以降）が必要です。'
                f'\nNVIDIA の GPU がある場合は、ドライバーを入れてください: {DRIVER_URL}'), info
    version = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
        return 'NVIDIA のドライバーの版を調べられませんでした。ドライバーを入れ直してください: ' + DRIVER_URL, info
    major, minor = version.value // 1000, version.value % 1000 // 10
    info['cudaVersion'] = f'{major}.{minor}'
    if (major, minor) < MIN_DRIVER_CUDA:
        need = '.'.join(map(str, MIN_DRIVER_CUDA))
        return (f'NVIDIA のドライバーが古いので、GPU 版の PyTorch を使えません（このドライバーが対応する CUDA: {major}.{minor}・'
                f'必要: {need} 以上）。\n最新のドライバーを入れてから、もう一度お試しください: {DRIVER_URL}'), info
    code = cuda.cuInit(0)
    count = ctypes.c_int()
    if code != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return (f'NVIDIA の GPU を使えません（CUDA の初期化に失敗しました。エラー {code}）。'
                f'\nPC を再起動するか、ドライバーを入れ直してから、もう一度お試しください: {DRIVER_URL}'), info
    # エンジンは 1 台目の GPU を使う
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, len(name), dev)
    cc_major, cc_minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(cc_major), 75, dev)    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    cuda.cuDeviceGetAttribute(ctypes.byref(cc_minor), 76, dev)    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    info['gpu'] = name.value.decode('utf-8', 'replace')
    info['capability'] = f'{cc_major.value}.{cc_minor.value}'
    if (cc_major.value, cc_minor.value) < MIN_CAPABILITY:
        return (f'この GPU（{info["gpu"]}・Compute Capability {info["capability"]}）は古いので使えません。'
                'GeForce GTX 900 番台以降の GPU が必要です。'), info
    return None, info


def driver_check():
    try:
        error, info = _check_driver()
    except Exception as e:   # 想定外の失敗も、理由が分かるように JSON で返す
        error, info = f'NVIDIA の GPU とドライバーを確認できませんでした（{type(e).__name__}: {e}）', {}
    print(json.dumps(dict(ok=error is None, error=error, **info), ensure_ascii=False))
    return 0 if error is None else 1


def main():
    report = dict(ok=True, modules={}, python=sys.version.split()[0])
    for name in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            report['modules'][name] = getattr(mod, '__version__', 'ok')
        except Exception as e:   # どのライブラリが読めないかをまとめて返す
            report['modules'][name] = None
            report.setdefault('errors', {})[name] = f'{type(e).__name__}: {e}'
            if name in REQUIRED:
                report['ok'] = False
    # GPU で実際に計算できるか（CPU 版の PyTorch が入った・この GPU 用のカーネルが無い、などをここで見つける）
    report['cuda'] = False
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'PyTorch から GPU を使えません（PyTorch {torch.__version__}）')
        x = torch.arange(1024, dtype=torch.float32, device='cuda')
        if float((x * 2).sum()) != 1023 * 1024:
            raise RuntimeError('GPU の計算結果が正しくありません')
        report['cuda'] = True
        report['gpu'] = torch.cuda.get_device_name(0)
    except Exception as e:
        report.setdefault('errors', {})['cuda'] = f'{type(e).__name__}: {e}'
        report['ok'] = False
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['ok'] else 1


def prefetch():
    """口パクに使う学習済みモデル（Demucs の htdemucs・Allosaurus）を先にダウンロードしておく（セットアップの最後に呼ぶ）。"""
    ok = True
    try:
        from demucs.pretrained import get_model
        get_model('htdemucs')
        print('demucs: ok', flush=True)
    except Exception as e:   # 失敗しても口パクのときにもう一度試すので、ここでは止めない
        print(f'demucs: {type(e).__name__}: {e}', flush=True)
        ok = False
    try:
        from allosaurus.app import read_recognizer
        read_recognizer()
        print('allosaurus: ok', flush=True)
    except Exception as e:
        print(f'allosaurus: {type(e).__name__}: {e}', flush=True)
        ok = False
    return 0 if ok else 1
