"""アプリ（Electron）との通信: 標準入出力の JSON Lines。

要求:   {"id": 1, "method": "capture", "params": {...}}
応答:   {"type": "result", "id": 1, "result": {...}} / {"type": "error", "id": 1, "error": {"message": ..., "kind": ...}}
通知:   {"type": "progress", "job": 1, "stage": ..., "value": 0.42, "message": ...} / {"type": "log", ...}

標準出力は通信だけに使う。ライブラリが標準出力へ書く文字（C の層の出力を含む）は、ファイル記述子のレベルで
標準エラーへ回す。重い処理（キャプチャ・変換）は 1 つずつ順にワーカーのスレッドで実行し、cancel で中止できる。
"""
import io
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import PROTOCOL_VERSION, __version__
from .jobs import Cancelled, Job, UserError
from .output import finite_json

HEAVY = {'capture', 'convert'}


class Server:
    def __init__(self, out):
        self._out = out
        self._lock = threading.Lock()
        self._jobs = {}
        self._heavy = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mmt-heavy')
        self._light = ThreadPoolExecutor(max_workers=4, thread_name_prefix='mmt-light')

    def emit(self, message):
        # NaN・無限大は null にする（そのまま書くと、アプリがその行を読めず、要求が終わらなくなる）
        line = json.dumps(finite_json(message), ensure_ascii=False, default=str, allow_nan=False)
        with self._lock:
            self._out.write(line + '\n')
            self._out.flush()

    def handle(self, request):
        rid = request.get('id')
        method = request.get('method')
        params = request.get('params') or {}
        if method == 'cancel':
            job = self._jobs.get(params.get('job'))
            if job is not None:
                job.cancel()
            self.emit(dict(type='result', id=rid, result=dict(cancelled=job is not None)))
            return
        if method == 'shutdown':
            for job in list(self._jobs.values()):
                job.cancel()
            self.emit(dict(type='result', id=rid, result=dict(ok=True)))
            os._exit(0)
        job = Job(rid, self.emit)
        self._jobs[rid] = job
        pool = self._heavy if method in HEAVY else self._light
        pool.submit(self._run, rid, method, params, job)

    def _run(self, rid, method, params, job):
        try:
            if job.cancelled:
                raise Cancelled()
            result = self._dispatch(method, params, job)
            self.emit(dict(type='result', id=rid, result=result))
        except Cancelled:
            self.emit(dict(type='error', id=rid, error=dict(kind='cancelled', message='中止しました')))
        except UserError as e:
            self.emit(dict(type='error', id=rid, error=dict(kind='user', message=str(e))))
        except MemoryError:
            self.emit(dict(type='error', id=rid, error=dict(
                kind='memory', message='メモリが足りませんでした。区間を短くするか、推論の設定を軽くしてください')))
        except Exception as e:   # 予期しないエラーは詳細も返す（ログに残して報告してもらう）
            detail = traceback.format_exc()
            sys.stderr.write(detail)
            message = str(e) or type(e).__name__
            if 'out of memory' in message.lower():
                message = 'GPU のメモリが足りませんでした。バッチサイズ・TTA の枚数・解像度を下げてください'
            self.emit(dict(type='error', id=rid, error=dict(kind='internal', message=message, detail=detail)))
        finally:
            self._jobs.pop(rid, None)

    def _dispatch(self, method, params, job):
        from . import session
        if method == 'hello':
            return dict(engineVersion=__version__, protocol=PROTOCOL_VERSION, pid=os.getpid())
        if method == 'system_info':
            return session.system_info()
        if method == 'probe_video':
            return session.probe_video(params)
        if method == 'capture':
            return session.capture(job, params)
        if method == 'convert':
            return session.convert_take(job, params)
        if method == 'hand_rotations':
            return session.hand_rotations(params)
        raise UserError(f'未知の要求です: {method}')

    def serve(self, stdin):
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self.emit(dict(type='error', id=None, error=dict(kind='protocol', message='要求を読めませんでした')))
                continue
            self.handle(request)
        # アプリが終了した（標準入力が閉じた）: 実行中の処理を中止して終わる
        for job in list(self._jobs.values()):
            job.cancel()
        os._exit(0)


def main():
    # 通信用に元の標準出力を複製し、ファイル記述子 1 は標準エラーへ向け直す
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    out = io.TextIOWrapper(os.fdopen(proto_fd, 'wb', buffering=0), encoding='utf-8', newline='\n', write_through=True)
    sys.stdout = sys.stderr
    # 標準入力も複製して、ファイル記述子 0 は NUL へ向け直す。Windows では、読み込み待ちのパイプの標準入力に
    # 別のスレッドで読み込む DLL（numpy・torch など）が触ると、読み込みが終わるまで止まってしまう
    proto_in = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdin = open(os.devnull, 'r')
    stdin = io.TextIOWrapper(os.fdopen(proto_in, 'rb', buffering=0), encoding='utf-8')
    server = Server(out)
    server.emit(dict(type='ready', engineVersion=__version__, protocol=PROTOCOL_VERSION))
    server.serve(stdin)
