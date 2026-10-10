"""実行環境の確認（python -m mmt_engine --selftest）。必要なライブラリを読み込めるかを JSON で出力する。"""
import importlib
import json
import sys

REQUIRED = ('numpy', 'scipy', 'imageio_ffmpeg', 'torch', 'torchvision')
OPTIONAL = ('ai_edge_litert', 'demucs', 'allosaurus')


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
    try:
        import torch
        report['cuda'] = bool(torch.cuda.is_available())
        if report['cuda']:
            report['gpu'] = torch.cuda.get_device_name(0)
    except Exception:
        report['cuda'] = False
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
