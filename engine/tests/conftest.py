import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmt_engine.convert.pipeline import convert  # noqa: E402
from mmt_engine.rig.skeleton import Skeleton  # noqa: E402
from mmt_engine.settings import make_config  # noqa: E402
from synthetic import synthetic_body_model, synthetic_walk  # noqa: E402


@pytest.fixture(scope='session')
def body_model():
    return synthetic_body_model()


@pytest.fixture(scope='session')
def skeleton():
    return Skeleton.standard()


@pytest.fixture
def run_walk(body_model, skeleton):
    """合成の歩行を変換して (モーション, ConversionResult) を返す。"""
    def run(overrides=None, contact_overrides=None, skel=None, motion=None, **walk_kwargs):
        motion = motion if motion is not None else synthetic_walk(**walk_kwargs)
        cfg = make_config(overrides or {})
        result = convert(dict(motion), body_model, skel or skeleton, cfg, contact_overrides=contact_overrides)
        return motion, result
    return run


def track(tracks, name):
    return next(t for t in tracks if t.name == name)


def unit_rows(q):
    return np.abs(np.linalg.norm(q, axis=-1) - 1.0).max()
