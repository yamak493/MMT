"""変換（ステージ 1〜10）の性質を合成データで確かめる。"""
import numpy as np
import pytest

from conftest import track, unit_rows
from mmt_engine.convert.variants import VARIANTS, variant_tracks
from mmt_engine.rig.skeleton import SIDES
from synthetic import add_arm_cross, add_depth_noise, add_jump, synthetic_walk, to_camera_coords


def test_outputs_are_finite_and_complete(run_walk):
    _, res = run_walk(num_frames=150, noise_deg=2.0)
    names = {t.name for t in res.tracks}
    for bone in ('センター', 'グルーブ', '下半身', '上半身', '上半身2', '首', '頭', '左腕', '右ひじ', '左足ＩＫ', '右足ＩＫ',
                 '左足', '右ひざ'):
        assert bone in names
    for t in res.tracks:
        assert t.rotations.shape == (150, 4)
        assert np.isfinite(t.positions).all() and np.isfinite(t.rotations).all()
        assert unit_rows(t.rotations) < 1e-6


def test_planted_feet_do_not_slide(run_walk):
    _, res = run_walk(num_frames=180, noise_deg=2.0)
    target = res.foot_ik.target
    for foot in range(2):
        for s, e in res.contact.segments[foot]:
            seg = target[s:e + 1, foot][:, [0, 2]]
            assert np.abs(seg - seg[0]).max() < 1e-6 * res.scale


def test_contacts_match_ground_truth(run_walk):
    motion, res = run_walk(num_frames=180, noise_deg=1.0)
    stance = motion['stance']
    agree = (res.contact.flags == stance).mean()
    assert agree > 0.85


def test_travel_distance_matches(run_walk):
    motion, res = run_walk(num_frames=180, speed=1.0)
    travelled = np.linalg.norm(res.center.delta[-1, [0, 2]] - res.center.delta[0, [0, 2]]) / res.scale
    expected = 1.0 * (180 - 1) / 30.0
    assert travelled == pytest.approx(expected, rel=0.12)


def test_camera_coords_with_depth_noise(run_walk):
    motion = add_depth_noise(to_camera_coords(synthetic_walk(num_frames=180), height=1.0, pitch_deg=10.0),
                             sigma=0.02, drift=0.05)
    _, res = run_walk(motion=motion)
    assert res.info['floor']['tilt_deg'] == pytest.approx(10.0, abs=4.0)
    # 接地中の足ＩＫは動かず、骨盤の前後の震えは入力より小さい
    z = res.center.delta[:, 2]
    raw = res.center.raw[:, 2]
    assert np.abs(np.diff(z, 2)).mean() < np.abs(np.diff(raw, 2)).mean()


def test_jump_is_kept_in_full_and_removed_in_locked(run_walk):
    motion = add_jump(synthetic_walk(num_frames=180, speed=0.0), start_sec=2.5, duration_sec=0.4, height=0.15)
    _, res = run_walk(motion=motion)
    assert res.info['jumps'], '滞空の区間が見つからない'
    s, e = res.info['jumps'][0]
    full_y = res.center.delta[s:e + 1, 1].max()
    locked, _, _ = variant_tracks(res, 'locked')
    groove = track(locked, 'グルーブ').positions[:, 1]
    assert groove[s:e + 1].max() < full_y - 0.05 * res.scale


def test_variants(run_walk):
    _, res = run_walk(num_frames=120)
    sets = {k: {t.name for t in variant_tracks(res, k)[0]} for k in VARIANTS}
    assert 'センター' not in sets['upper_body'] and '左足ＩＫ' not in sets['upper_body']
    assert '上半身' in sets['upper_body']
    no_move = variant_tracks(res, 'no_move')[0]
    assert np.abs(track(no_move, 'センター').positions[:, [0, 2]]).max() < 1e-9
    for kind in VARIANTS:
        for t in variant_tracks(res, kind)[0]:
            assert np.isfinite(t.rotations).all()


def test_contact_overrides(run_walk):
    _, base = run_walk(num_frames=150)
    s, e = base.contact.segments[0][1]
    swing = [f for f in range(150) if not base.contact.flags[f, 1]]
    a = swing[len(swing) // 2]
    overrides = [dict(foot=0, start=s, end=e, mode='off'), dict(foot=1, start=a, end=a + 3, mode='on')]
    _, res = run_walk(num_frames=150, contact_overrides=overrides)
    assert not res.contact.flags[s:e + 1, 0].any()
    assert res.contact.flags[a:a + 4, 1].all()
    # 自動判定は上書きに依らない
    assert np.array_equal(res.auto_contact.flags, base.auto_contact.flags)
    # 強制した接地の間は足ＩＫが動かない
    seg = res.foot_ik.target[a:a + 4, 1][:, [0, 2]]
    assert np.abs(seg - seg[0]).max() < 1e-6 * res.scale


def _foot_key_tracks(res, keys, kind='full'):
    from mmt_engine.convert.foot_keys import key_frames, place_feet, splice_tracks
    from mmt_engine.convert.variants import LOWER_BODY_BONE, motion_tracks, variant_motion
    center, ik, local, _, _ = variant_motion(res, kind)
    base = motion_tracks(res, kind, center, ik, local)
    lower_rot = res.retargeter.global_matrix(LOWER_BODY_BONE, res.kin.glob_rot)
    center2, ik2, changed = place_feet(center, ik, key_frames(keys, len(center)), lower_rot, res.reach_geometry,
                                       res.config, res.scale)
    return base, splice_tracks(base, res.skeleton, center2, ik2, local, res.config, res.knee_poles, changed), ik2, changed


def test_foot_key_is_local(run_walk):
    _, res = run_walk(num_frames=150)
    swing = [f for f in range(20, 130) if not res.contact.flags[f, 1]]
    a = swing[len(swing) // 2]
    blend = int(res.config.foot_ik.blend_frames)
    for kind in ('full', 'locked', 'no_move'):
        base, keyed, ik, changed = _foot_key_tracks(res, [dict(foot=1, frame=a)], kind)
        # キーのフレームでは足裏が床に平ら（差分の Y = 0、回転は鉛直軸まわりだけ）
        assert abs(ik.delta[a, 1, 1]) < 1e-9
        q = ik.rotation[a, 1]
        assert abs(q[0]) < 1e-9 and abs(q[2]) < 1e-9
        # 前後 blend フレームの外は変わらない
        assert not changed[:a - blend].any() and not changed[a + blend + 1:].any()
        outside = np.ones(len(changed), bool)
        outside[a - blend:a + blend + 1] = False
        for t0, t1 in zip(base, keyed):
            assert t0.name == t1.name
            assert np.array_equal(t0.positions[outside], t1.positions[outside])
            assert np.array_equal(t0.rotations[outside], t1.rotations[outside])
            assert np.isfinite(t1.rotations).all()
        assert not np.array_equal(track(base, '右足ＩＫ').positions, track(keyed, '右足ＩＫ').positions)


def test_foot_key_pair_holds_the_foot(run_walk):
    _, res = run_walk(num_frames=150)
    a, b = 50, 70
    blend = int(res.config.foot_ik.blend_frames)
    base, keyed, ik, changed = _foot_key_tracks(res, [dict(foot=0, frame=a), dict(foot=0, frame=b)])
    # 組の間は、体が動いても最初のキーの位置・向きのまま
    assert np.abs(res.foot_ik.delta[a:b + 1, 0] - res.foot_ik.delta[a, 0]).max() > 1e-3 * res.scale
    seg = ik.delta[a:b + 1, 0]
    assert np.abs(seg - seg[0]).max() < 1e-9 and abs(seg[0, 1]) < 1e-9
    assert np.abs(ik.rotation[a:b + 1, 0] - ik.rotation[a, 0]).max() < 1e-9
    assert np.allclose(seg[0, [0, 2]], res.foot_ik.delta[a, 0, [0, 2]])
    assert changed[a - blend:b + blend + 1].all()
    assert not changed[:a - blend].any() and not changed[b + blend + 1:].any()


def test_hold_spans():
    from mmt_engine.convert.foot_keys import hold_spans
    assert hold_spans([10, 20, 40]) == [(10, 20), (40, 40)]
    assert hold_spans([10, 20, 21, 30]) == [(10, 30)]
    assert hold_spans([]) == []


def test_consecutive_foot_keys_hold_the_foot(run_walk):
    _, res = run_walk(num_frames=150)
    swing = [f for f in range(20, 130) if not res.contact.flags[f, 0]]
    a = swing[len(swing) // 2]
    _, _, ik, _ = _foot_key_tracks(res, [dict(foot=0, frame=f) for f in range(a, a + 4)])
    seg = ik.delta[a:a + 4, 0]
    assert np.abs(seg - seg[0]).max() < 1e-9


def test_normalize_foot_keys():
    from mmt_engine.convert.foot_keys import key_frames, normalize_keys
    keys = normalize_keys([dict(foot=1, frame=5), dict(foot=0, frame=3), dict(foot=1, frame=5), dict(foot=2, frame=1),
                           dict(foot=0, frame=-1), dict(frame=4)])
    assert keys == [dict(foot=0, frame=3), dict(foot=1, frame=5)]
    assert key_frames(keys + [dict(foot=0, frame=99)], 10) == ([3], [5])


def test_arm_fix_reuse_gives_same_arms(run_walk, body_model, skeleton):
    from mmt_engine.convert.pipeline import convert
    from mmt_engine.settings import make_config
    motion = add_arm_cross(synthetic_walk(num_frames=60, speed=0.0))
    cfg = make_config()
    full = convert(dict(motion), body_model, skeleton, cfg)
    reused = convert(dict(motion), body_model, skeleton, cfg, arm_fix=full.arm_correction)
    for side in SIDES:
        for bone in ('腕', 'ひじ', '手首'):
            a = track(full.tracks, side + bone).rotations
            b = track(reused.tracks, side + bone).rotations
            assert np.abs(np.abs(np.sum(a * b, -1)) - 1).max() < 1e-6


def test_crossed_arms_are_separated(run_walk):
    motion = add_arm_cross(synthetic_walk(num_frames=60, speed=0.0))
    _, res = run_walk(motion=motion)
    before, after = res.info['arm_collision']['overlap_frames']
    assert before > 0
    assert after < before


def test_outlier_frames_are_replaced(run_walk):
    motion = synthetic_walk(num_frames=150, noise_deg=1.0)
    pose = np.array(motion['pose'], copy=True)
    pose[70:72, 18] += np.array([0.0, 1.6, 0.0])   # 左ひじが 2 フレームだけ飛ぶ
    motion = dict(motion, pose=pose)
    _, res = run_walk(motion=motion)
    assert res.info['outliers']['frames']['left_arm'] >= 2


@pytest.mark.parametrize('heading_deg', [0.0, 90.0])
def test_lean_is_corrected_in_both_directions(run_walk, heading_deg):
    """カメラの奥行き方向（heading 0）にも左右方向（heading 90）にも、傾けた体を起こす。"""
    def spine_tilt(res):
        v = res.kin.joints[:, 12] - res.kin.joints[:, 0]
        h = np.deg2rad(heading_deg)
        return np.rad2deg(np.arctan2(v[:, 0] * np.sin(h) + v[:, 2] * np.cos(h), v[:, 1])).mean()
    _, upright = run_walk(num_frames=180, speed=0.5, heading_deg=heading_deg)
    _, leaned = run_walk(num_frames=180, speed=0.5, heading_deg=heading_deg, lean_deg=10.0)
    assert spine_tilt(leaned) == pytest.approx(spine_tilt(upright), abs=1.5)
    _, off = run_walk(num_frames=180, speed=0.5, heading_deg=heading_deg, lean_deg=10.0,
                      overrides={'lean.side_enabled': False})
    assert (spine_tilt(off) - spine_tilt(upright) > 7.0) == (heading_deg == 90.0)


def test_flip_facing(run_walk):
    _, a = run_walk(num_frames=90)
    _, b = run_walk(num_frames=90, overrides={'floor.flip_facing': True})
    za = a.center.delta[-1, 2] - a.center.delta[0, 2]
    zb = b.center.delta[-1, 2] - b.center.delta[0, 2]
    assert np.sign(za) == -np.sign(zb)


def test_missing_bones_raise(body_model):
    from mmt_engine.convert.pipeline import convert
    from mmt_engine.rig.skeleton import Skeleton
    from mmt_engine.settings import make_config
    skel = Skeleton.standard()
    del skel.positions['左ひじ']
    with pytest.raises(ValueError, match='左ひじ'):
        convert(dict(synthetic_walk(num_frames=30)), body_model, skel, make_config())
