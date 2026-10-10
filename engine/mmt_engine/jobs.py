"""長い処理（キャプチャ・変換）の進捗の報告と中止。"""
import threading


class Cancelled(Exception):
    """ユーザーが処理を中止した。"""


class UserError(Exception):
    """ユーザーに見せるメッセージを持つエラー（設定・入力の問題）。"""


class Job:
    """進捗を段階（stage）ごとの重みで 0〜1 にまとめて報告する。"""

    def __init__(self, job_id, emit, stages=(('main', 1),)):
        self.id = job_id
        self._emit = emit
        self._cancel = threading.Event()
        self.set_stages(stages)

    def set_stages(self, stages):
        """段階と重み [(名前, 重み)]。重みの合計で割って全体の進捗にする。"""
        total = float(sum(w for _, w in stages)) or 1.0
        self._spans = {}
        acc = 0.0
        for name, w in stages:
            self._spans[name] = (acc / total, (acc + w) / total)
            acc += w
        self._stage = None
        self._last = -1.0

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def check(self):
        if self._cancel.is_set():
            raise Cancelled()

    def stage(self, name, message):
        self._stage = name
        self.progress(0.0, message, force=True)

    def progress(self, fraction, message=None, force=False):
        lo, hi = self._spans.get(self._stage, (0.0, 1.0))
        value = lo + (hi - lo) * max(0.0, min(1.0, float(fraction)))
        if not force and value - self._last < 0.002 and message is None:
            return
        self._last = value
        self._emit(dict(type='progress', job=self.id, stage=self._stage, value=round(value, 4),
                        message=message))

    def log(self, message, level='info'):
        self._emit(dict(type='log', job=self.id, level=level, message=str(message)))
