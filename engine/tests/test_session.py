import os
import time
from pathlib import Path

from mmt_engine import session
from mmt_engine.output import save_npz
from synthetic import synthetic_walk


class _Job:
    cancelled = False

    def set_stages(self, stages):
        pass

    def stage(self, *args):
        pass

    def progress(self, *args):
        pass

    def check(self):
        pass

    def log(self, *args):
        pass


def _run(take, keys):
    r = session.convert_take(_Job(), dict(takeDir=str(take), rigPath=None, settings={}, footKeys=keys))
    out = Path(r['resultPath']).parent
    return {f.name: f.read_bytes() for f in out.iterdir()}


def test_lru_cache_keeps_recent():
    cache = session.LRUCache(2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache.get('a') == 1   # a を使ったので、次に消えるのは b
    cache['c'] = 3
    assert list(cache) == ['a', 'c']
    assert cache.get('b') is None
    cache.pop('a', None)
    assert list(cache) == ['c']


def test_foot_keys_reuse_saved_base(tmp_path, monkeypatch, body_model):
    """足ピタの編集は、撮影のフォルダに残した基準の変換を使う（起動し直し・.mmt から開き直しても変換しない）。"""
    monkeypatch.setattr(session, 'load_body_model', lambda params: body_model)
    monkeypatch.setattr(session, '_BASE_CACHE', {})
    monkeypatch.setattr(session, '_DIGEST_CACHE', {})
    save_npz(tmp_path / 'motion.npz', **synthetic_walk(num_frames=120))
    keys = [dict(foot=0, frame=40), dict(foot=1, frame=80)]
    base = _run(tmp_path, [])
    keyed = _run(tmp_path, keys)
    assert (tmp_path / session.BASE_FILE).exists()

    session._BASE_CACHE.clear()
    session._DIGEST_CACHE.clear()
    later = time.time() + 100
    for f in tmp_path.glob('*.npz'):
        os.utime(f, (later, later))

    def no_convert(*args, **kwargs):
        raise AssertionError('変換し直した')

    monkeypatch.setattr(session, 'convert', no_convert)
    for d in (tmp_path / 'conv').iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    assert _run(tmp_path, keys) == keyed
    assert _run(tmp_path, []) == base
