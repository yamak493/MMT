"""設定・リグ・結果ファイル・指・口パク・単精度化・通信の部品のテスト。"""
import base64
import io
import json
import time
import zipfile

import numpy as np
import pytest

from mmt_engine.convert.tracks import BoneTrack, to_mmd_quat
from mmt_engine.core import filters, quat
from mmt_engine.hands import shapes
from mmt_engine.lipsync import keys as lip
from mmt_engine.output import write_result
from mmt_engine.rig.skeleton import STANDARD_BONES, Skeleton
from mmt_engine.settings import make_config, with_overrides


# ---------------------------------------------------------------- 基本

def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(50, 3))
    q = quat.from_rotvec(r)
    assert np.allclose(quat.to_rotvec(quat.from_matrix(quat.to_matrix(q))), quat.to_rotvec(q), atol=1e-9)
    smooth = quat.from_rotvec(np.linspace(0, 1, 50)[:, None] * [0.0, 2.0, 0.0])
    flipped = np.where(np.arange(50)[:, None] % 2, -smooth, smooth)
    assert (np.sum(quat.make_continuous(flipped)[1:] * quat.make_continuous(flipped)[:-1], -1) > 0).all()
    mid = quat.slerp(q[:-1], q[1:], 0.5)
    assert np.allclose(quat.angle_between(mid, q[:-1]), quat.angle_between(mid, q[1:]), atol=1e-6)


def test_one_euro_keeps_constant_velocity():
    x = np.linspace(0, 3, 120)[:, None]
    y = filters.one_euro(x, 30.0, 1.0, 0.0)
    assert np.abs(y - x).max() < 1e-6


def test_runs():
    assert filters.runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert filters.label_runs([2, 2, 3]) == [(0, 1, 2), (2, 2, 3)]


def test_settings_merge_and_unknown_key():
    cfg = make_config({'center.mode': 'B', 'contacts': {'body_parts': {'chest': False}}})
    assert cfg.center.mode == 'B' and cfg.contacts.body_parts.chest is False and cfg.contacts.body_parts.torso is True
    with pytest.raises(KeyError):
        make_config({'center.nope': 1})
    cfg2 = with_overrides(cfg, {'center.mode': 'A'})
    assert cfg2.center.mode == 'A' and cfg.center.mode == 'B'


# ---------------------------------------------------------------- リグ

def _rig_from_standard(with_mesh=True):
    names = list(STANDARD_BONES)
    bones = []
    for n in names:
        pos, parent, tail = STANDARD_BONES[n]
        bones.append(dict(name=n, parent=names.index(parent) if parent else -1, position=list(pos), flags=0,
                          tail=list(tail) if tail else [0.0, 0.0, 0.0], axis=None))
    rig = dict(format='mmt-rig', version=1, modelName='テスト', bones=bones, morphs=[dict(name='あ')], rigidBodies=[])
    if with_mesh:
        pts = np.array([[0.0, 15.0, 0.0], [1.0, 2.0, 3.0]], '<f4')
        owners = np.array([names.index('上半身2'), 99999], '<i4')
        rig['vertices'] = dict(count=2, positions=base64.b64encode(pts.tobytes()).decode(),
                               bones=base64.b64encode(owners.tobytes()).decode())
    return rig


def test_skeleton_from_rig():
    skel = Skeleton.from_rig(_rig_from_standard())
    std = Skeleton.standard()
    assert skel.source == 'model' and skel.model_name == 'テスト' and skel.morph_names == ['あ']
    assert skel.mean_leg_length() == pytest.approx(std.mean_leg_length())
    assert np.allclose(skel.internal('左ひざ'), std.internal('左ひざ'))
    assert list(skel.mesh_bones) == ['上半身2', '']
    assert skel.nearest_ancestor('左手首', {'左腕', '上半身'}) == '左腕'


def test_conversion_with_rig_matches_standard(run_walk):
    _, a = run_walk(num_frames=60)
    _, b = run_walk(num_frames=60, skel=Skeleton.from_rig(_rig_from_standard(with_mesh=False)))
    ca = next(t for t in a.tracks if t.name == 'センター').positions
    cb = next(t for t in b.tracks if t.name == 'センター').positions
    assert np.allclose(ca, cb, atol=1e-6)


# ---------------------------------------------------------------- 結果ファイル

def test_write_result_roundtrip(tmp_path):
    T = 5
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
    pos = np.arange(T * 3, dtype=float).reshape(T, 3)
    tracks = [BoneTrack('センター', pos, rot, True), BoneTrack('上半身', np.zeros((T, 3)), rot, False)]
    path = write_result(tmp_path / 'conv' / 'k', dict(frames=T, info=dict(x=np.float32(1.5))), dict(full=tracks))
    data = json.loads(path.read_text(encoding='utf-8'))
    layout = data['variants']['full']
    buf = np.fromfile(tmp_path / 'conv' / 'k' / layout['file'], '<f4')
    center = layout['bones'][0]
    assert center['movable'] and np.allclose(buf[center['offset']:center['offset'] + 15].reshape(T, 3), pos)
    upper = layout['bones'][1]
    assert not upper['movable'] and np.allclose(buf[upper['offset']:upper['offset'] + 20].reshape(T, 4), rot)
    assert data['info']['x'] == 1.5


def test_mmd_quat_conversion_is_mirror():
    # 内部座標で Y 軸まわりの回転 θ は、MMD 座標（z 反転）では −θ
    q = quat.axis_angle([0.0, 1.0, 0.0], 0.5)
    m = to_mmd_quat(q)
    assert np.allclose(m, quat.axis_angle([0.0, 1.0, 0.0], -0.5))


# ---------------------------------------------------------------- 指

def _skeleton_with_fingers():
    skel = Skeleton.standard()
    for side, sign in (('左', 1.0), ('右', -1.0)):
        wrist = skel.positions[side + '手首']
        d = np.array([0.69 * sign, -0.58, -0.1])
        d /= np.linalg.norm(d)
        lateral = np.array([0.0, 0.0, -1.0])
        for f, (finger, names) in enumerate(make_config().hands.bones.items()):
            base = wrist + d * (0.4 if finger != 'thumb' else 0.15) + lateral * (0.12 - 0.06 * f)
            parent = side + '手首'
            for i, n in enumerate(names):
                skel.positions[side + n] = base + d * 0.18 * i
                skel.parents[side + n] = parent
                parent = side + n
    return skel


def test_shape_rotation_table():
    skel = _skeleton_with_fingers()
    table = shapes.shape_rotation_table(skel, make_config().hands)
    assert set(table) == {'left', 'right'} and set(table['left']) == set(shapes.SHAPES)
    fist = np.array(table['left']['fist']['左人指２'])
    assert quat.angle_between(fist, quat.IDENTITY) > np.deg2rad(60)
    straight = np.array(table['left']['index']['左人指１'])
    assert quat.angle_between(straight, quat.IDENTITY) < np.deg2rad(1)
    # 左右対称のリグでは、左右の手の曲げは同じ大きさ
    assert quat.angle_between(np.array(table['right']['fist']['右人指２']), quat.IDENTITY) == \
        pytest.approx(quat.angle_between(fist, quat.IDENTITY), abs=1e-4)


def test_shape_scores_new_shapes():
    cfg = make_config().hands
    cases = {
        'pistol': (1, 1, 0, 0, 0), 'fox': (0, 1, 0, 0, 1), 'ok': (0, 0, 1, 1, 1),
        'index': (0, 1, 0, 0, 0), 'scissors': (0, 1, 1, 0, 0),
    }
    for shape, ext in cases.items():
        p_ext = np.where(np.array(ext) == 1, 0.95, 0.05)[None]
        scores = shapes.shape_scores(p_ext, 1.0 - p_ext, cfg)[0]
        assert shapes.SHAPES[int(np.argmax(scores))] == shape
    # 人差し指＋中指のピストル
    p_ext = np.array([[0.95, 0.95, 0.95, 0.05, 0.05]])
    assert shapes.SHAPES[int(np.argmax(shapes.shape_scores(p_ext, 1.0 - p_ext, cfg)[0]))] == 'pistol'


def test_label_events():
    events = shapes.label_events(np.array([0, 0, 6, 6, 6, 8]))
    assert events == [dict(frame=0, shape='default'), dict(frame=2, shape='fist'), dict(frame=5, shape='open')]


# ---------------------------------------------------------------- 口パク

def test_phone_shapes():
    assert lip.phone_shape('a') == 'a' and lip.phone_shape('ɯ') == 'u' and lip.phone_shape('aː') == 'a'
    assert lip.phone_shape('m') == 'n' and lip.phone_shape('k') is None and lip.phone_shape('ɔ') == 'o'


def test_build_lipsync():
    rate = 100.0
    level = np.full(300, -80.0)
    level[50:150] = -10.0              # 0.5〜1.5 秒に声
    analysis = dict(phone_times=np.array([0.6, 1.0, 2.5]), phones=np.array(['a', 'i', 'o']),
                    level_db=level, level_rate=rate, duration=3.0)
    cfg = make_config().lipsync
    tracks, weights, info, warns = lip.build_lipsync(analysis, cfg, 30.0, 90, available_morphs=['あ', 'い', 'う'])
    names = {t['name'] for t in tracks}
    assert names == {'あ', 'い', 'う'}
    assert any('お' in w for w in warns)
    a = weights[lip.SHAPES.index('a')]
    assert a[22] > 0.5 and a[70] < 1e-3    # 声の無い 2.5 秒の「お」は使わない


# ---------------------------------------------------------------- 単精度化

def test_fp32_patch(tmp_path):
    from mmt_engine.capture.fp32_patch import PATCHES, make_fp32_model
    src = tmp_path / 'm.torchscript'
    with zipfile.ZipFile(src, 'w') as z:
        files = {}
        for suffix, before, _ in PATCHES:
            files.setdefault(suffix, '')
            files[suffix] += before + '\n'
        for suffix, text in files.items():
            z.writestr(f'm/code/__torch__/{suffix}', text)
        z.writestr('m/data/0', b'\x00\x01')
    out = make_fp32_model(src)
    with zipfile.ZipFile(out) as z:
        text = ''.join(z.read(n).decode('utf-8') for n in z.namelist() if n.endswith('.py'))
        assert z.read('m/data/0') == b'\x00\x01'
    for _, before, after in PATCHES:
        assert before not in text and after in text
    bad = tmp_path / 'bad.torchscript'
    with zipfile.ZipFile(bad, 'w') as z:
        z.writestr('m/code/__torch__/x.py', 'nothing')
    with pytest.raises(RuntimeError):
        make_fp32_model(bad)


def test_jit_load_non_ascii_path(tmp_path):
    # ユーザー名が日本語のとき、アプリのデータのフォルダのパスに日本語が入る（torch.jit.load(str) では開けない）
    torch = pytest.importorskip('torch')
    from mmt_engine.capture.nlf_runner import _jit_load
    src = tmp_path / 'm.torchscript'
    torch.jit.script(torch.nn.Linear(2, 2)).save(str(src))
    dst = tmp_path / 'ユーザー 山田' / 'm.torchscript'
    dst.parent.mkdir()
    src.rename(dst)
    model = _jit_load(dst, 'cpu')
    assert model(torch.ones(1, 2)).shape == (1, 2)


def test_finite_people_skips_nan():
    from mmt_engine.capture.nlf_runner import Tracker, finite_people
    trans = np.array([[0.0, 0.0, 3.0], [np.nan, 0.0, 3.0], [1.0, 0.0, 3.0]])
    pose = np.zeros((3, 24, 3))
    pose[2, 5, 1] = np.inf
    boxes = np.array([[0, 0, 10, 20, 0.9], [0, 0, 50, 90, 0.9], [0, 0, 10, 20, 0.9]], float)
    keep = np.flatnonzero(finite_people([boxes, trans, pose]))
    assert keep.tolist() == [0]
    # 一番大きく写っている人（1 番）が NaN でも、それを選ばない
    j = Tracker(30.0, 'largest', 100).choose(0, boxes[keep], trans[keep])
    assert keep[j] == 0
    assert finite_people([np.zeros((0, 5))]).shape == (0,)


# ---------------------------------------------------------------- 通信

def test_server_protocol():
    from mmt_engine.server import Server
    out = io.StringIO()
    server = Server(out)
    server.handle(dict(id=1, method='hello'))
    server.handle(dict(id=2, method='no_such_method'))
    for _ in range(100):
        lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
        if len(lines) >= 2:
            break
        time.sleep(0.05)
    by_id = {m['id']: m for m in lines}
    assert by_id[1]['type'] == 'result' and by_id[1]['result']['protocol'] == 1
    assert by_id[2]['type'] == 'error' and by_id[2]['error']['kind'] == 'user'


def _strict_json(text):
    """JavaScript の JSON.parse と同じく、NaN・Infinity を受け付けない読み込み。"""
    def reject(name):
        raise ValueError(f'JSON にない値です: {name}')
    return json.loads(text, parse_constant=reject)


def test_non_finite_values_become_null():
    from mmt_engine.output import json_text
    from mmt_engine.server import Server
    data = dict(a=float('nan'), b=np.float64('inf'), c=np.float32('-inf'), d=[1.5, np.nan],
                e=np.array([np.nan, 2.0]), f=(np.int64(3), np.bool_(True)), g=np.float64(0.25))
    expected = dict(a=None, b=None, c=None, d=[1.5, None], e=[None, 2.0], f=[3, True], g=0.25)
    assert _strict_json(json_text(data)) == expected
    out = io.StringIO()
    Server(out).emit(dict(type='result', id=1, result=data))
    assert _strict_json(out.getvalue())['result'] == expected
