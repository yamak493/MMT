"""MonoCam Motion Trace の推論・変換エンジン。

Electron のアプリから子プロセスとして起動され（python -m mmt_engine）、標準入出力の JSON Lines で
要求を受けて、動画からのモーション抽出（NLF）・手のランドマーク検出・口パクの解析と、
MMD 用のモーションへの変換を行う。
"""

__version__ = '1.0.1'
PROTOCOL_VERSION = 1
