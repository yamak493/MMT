"""NLF → MMD のモーションへの変換（ステージ 1〜10 をこの順番で実行する）。

順番を入れ替えないこと（特に「床のオフセット → 接地判定 → ロック → クランプ」）。外れフレームの除外
（ステージ1b と、ステージ2 の単独フレームの外れ値の除去）は入力の fps のまま行い、MMD の 30fps への
リサンプルはその後で行う（先にリサンプルすると、1 フレームだけの破綻が前後 2 フレームに薄まって見つからない）。
ステージ2 のならしは、入力と出力の fps の高いほうで掛ける。
"""
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from ..body.smpl import ANKLES, compute_kinematics, forward_kinematics, rest_info
from ..core import quat
from ..core.filters import frames_for_fps, runs
from ..rig.skeleton import REQUIRED_BONES, SIDES
from .arm_collision import OVERLAP_TOL_M, resolve_arm_collisions
from .body_contacts import resolve_contacts
from .center import ReachGeometry, stabilize_center, supporting_legs
from .contact import apply_overrides, detect_contacts, normalize_overrides, without_frames
from .depth import depth_axis, detection_speeds, reconstruct_depth
from .floor import estimate_floor
from .foot_ik import build_foot_ik, mmd_sole_heights, mmd_sole_points, sole_floor_levels
from .ground import ground_offset
from .hand_reach import keep_hand_positions
from .jitter import remove_rotation_outliers, stabilize_pose, stabilize_root
from .lean import estimate_lean
from .legs import knee_poles
from .motion import load_motion, resample, resample_mask
from .outliers import PART_LABELS, remove_outliers
from .retarget import Retargeter
from .tracks import build_tracks, leg_rotations
from .wrist import correct_wrists
from .wrist_limits import limit_wrists


def _noop(*_a, **_k):
    pass


@dataclass
class PreparedMotion:
    """ステージ1・1b・2 の結果（入力の fps のまま）。手を切り出す位置と変換が同じものを使う。"""
    motion: object               # Motion（1b で外れを置き換えたもの）
    valid_input: np.ndarray      # (T,) 入力の valid（人物を検出できたフレーム）
    outliers: object
    despiked: np.ndarray         # (T, J, 4) 1b の後に単独フレームの外れ値だけを除いた回転
    quats: np.ndarray            # (T, J, 4) ステージ2 の後
    root: np.ndarray             # (T, 3)
    jitter_info: dict
    rest: object
    rest_joints: np.ndarray
    parents: np.ndarray
    key: str

    def joints_camera_mm(self, smoothed=True):
        """関節の位置（カメラ座標 [mm]、(T, 24, 3)）。手を切り出す位置に使う（fps は入力のまま）。"""
        from .motion import CAMERA_TO_YUP
        quats, root = (self.quats, self.root) if smoothed else (self.despiked, self.motion.root_pos)
        _, joints = forward_kinematics(quats, root, self.rest_joints, self.parents)
        return joints @ CAMERA_TO_YUP.T * 1000.0


def prepare_key(motion, bm, cfg):
    h = hashlib.sha1()
    for a in (motion.quats, motion.root_pos, motion.valid, motion.betas, [motion.fps], bm.rest_joints(motion.betas),
              bm.parents):
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    floor = {k: v for k, v in cfg.floor.to_dict().items() if k not in ('flip_facing', 'horizontal_origin')}
    sections = dict(body=cfg.body.to_dict(), floor=floor, outliers=cfg.outliers.to_dict(),
                    jitter=cfg.jitter.to_dict(), coords=motion.coords)
    h.update(json.dumps(sections, sort_keys=True, default=str).encode('utf-8'))
    return h.hexdigest()


def prepare(source, bm, cfg, cancel=None):
    """ステージ1・1b・2 を入力の fps のまま行う。"""
    cancel = cancel or _noop
    motion = load_motion(source, bm)
    return _prepare(motion, bm, cfg, cancel)


def _prepare(motion, bm, cfg, cancel):
    fps = motion.fps
    valid_input = np.asarray(motion.valid, bool)
    key = prepare_key(motion, bm, cfg)
    rest = rest_info(bm, motion.betas, int(cfg.body.heel_toe_vertices), float(cfg.body.sole_band_m))
    rest_joints = bm.rest_joints(motion.betas)
    outliers = None
    if cfg.outliers.enabled:
        # 重力の向き（床の法線）は外れを除く前の体で仮に求める（ステージ4で外れを除いた体から求め直す）
        from ..settings import with_overrides
        floor_cfg = with_overrides(cfg, {'floor.flip_facing': False}).floor
        pre_floor = estimate_floor(compute_kinematics(motion.quats, motion.root_pos, rest), fps, floor_cfg,
                                   rest.points.mean(2), valid_input)
        cancel()
        outliers = remove_outliers(motion.quats, motion.root_pos, valid_input, rest_joints, bm.parents, fps,
                                   cfg.outliers, pre_floor.rotation, depth_axis(pre_floor.rotation), cancel=cancel)
        motion = motion.with_values(quats=outliers.quats, root_pos=outliers.root_pos, valid=outliers.valid)
    cancel()
    despiked, _ = remove_rotation_outliers(quat.make_continuous(motion.quats), fps, float(cfg.jitter.outlier_deg_per_s))
    quats, jitter_info = stabilize_pose(motion.quats, fps, cfg.jitter, rest_joints)
    root = stabilize_root(motion.root_pos, frames_for_fps(cfg.jitter.root_median_window, fps, odd=True))
    return PreparedMotion(motion, valid_input, outliers, despiked, quats, root, jitter_info, rest, rest_joints,
                          np.asarray(bm.parents), key)


@dataclass
class ConversionResult:
    config: object
    fps: float
    scale: float
    skeleton: object
    retargeter: object
    kin: object
    kin_raw: object
    valid: np.ndarray            # (T,) 観測として使えるフレーム（30fps）
    valid_input: np.ndarray      # (T,) 人物を検出できたフレーム（30fps）
    floor: object
    ground: object
    contact: object
    auto_contact: object         # 上書きする前の接地判定（足ピタの自動判定）
    foot_ik: object
    center: object
    ankle_rest: np.ndarray
    pelvis_rest: np.ndarray
    reach_geometry: object
    local_quats: dict
    knee_poles: np.ndarray
    tracks: list
    legs: object = None
    arm_correction: dict = None  # 腕・ひじ・手首の補正（ボーン名 → (T, 4)）。arm_fix に渡して使い回せる
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def apply_scale(kin, k, depth_scale):
    """ステージ5: 全位置にスケール係数を掛け、奥行き方向の移動量だけを depth_scale 倍する。"""
    kin = kin.scaled(k)
    if depth_scale != 1.0:
        rz = kin.root_pos[:, 2]
        off = np.zeros((len(rz), 3))
        off[:, 2] = (depth_scale - 1.0) * (rz - rz[0])
        kin = kin.transformed(None, off)
    return kin


def solve_contacts(skel, rt, glob_rot, local, cfg, unit, fps, smpl_rest, finger_local=None, leg_glob=None,
                   cancel=None):
    """ステージ9b → 9c。9c が手首を戻した所は、手首を動かさずに 9b をもう一度解く。"""
    from ..settings import with_overrides

    def contacts(loc, c_cfg):
        return resolve_contacts(skel, rt, glob_rot, loc, c_cfg, cfg.arm_collision, cfg.hands.bones,
                                cfg.arm_collision.mode, unit, fps, smpl_rest, finger_local=finger_local,
                                leg_glob=leg_glob, cancel=cancel)

    local, result = contacts(local, cfg.contacts)
    local, limits = limit_wrists(skel, local, cfg.wrist_limits, fps)
    if result.enabled and limits.enabled and sum(limits.info['changed_frames']) > 0:
        locked = with_overrides(cfg, {'contacts.max_deg.wrist': 0.0}).contacts
        local, _ = contacts(local, locked)
        local, _ = limit_wrists(skel, local, cfg.wrist_limits, fps)
    return local, result, limits


ARM_FIX_BONES = tuple(s + b for s in SIDES for b in ('腕', 'ひじ', '手首'))


def convert(source, bm, skel, cfg, prepared=None, hands_analysis=None, contact_overrides=None, progress=None,
            cancel=None, arm_fix=None):
    """NLF のモーションを MMD のモーションに変換する（ConversionResult）。

    source: モーション（npz のパスか dict）/ bm: SMPL の BodyModel / skel: MMD の Skeleton
    prepared: prepare の結果（入力・設定が同じなら使い回す）/ hands_analysis: 手のランドマークの検出結果（手首の補正）
    contact_overrides: タイムラインで編集した足ピタ [{foot, start, end, mode}]（30fps のフレーム）
    progress(割合 0〜1, 文言) / cancel(): 中止されていれば例外を投げる関数
    """
    progress = progress or _noop
    cancel = cancel or _noop
    warns = []
    # arm_fix: 別の変換（足ピタを上書きしない基準の変換）で求めた腕の補正（ボーン名 → (T, 4) の回転）。渡すと、
    # 9h・9a・9b・9c を解かずにその補正を掛ける（接地の上書きだけを変えたときは腕の姿勢はほとんど変わらないので、再計算を速くする）
    missing = skel.missing(REQUIRED_BONES)
    if missing:
        raise ValueError('モデルに必要なボーンがありません: ' + '、'.join(missing))
    if skel.source == 'standard':
        warns.append('モデルが指定されていないので、標準的な体格のボーン寸法で変換しました')
    for s in SIDES:
        if skel.source != 'standard' and not skel.has(s + 'つま先ＩＫ'):
            warns.append(f'{s}つま先ＩＫ が無いモデルです（足先の向きが変わりません）')

    # ---- 1・1b・2 ----
    progress(0.0, '外れフレームの除外・ジッター制御')
    motion_in = load_motion(source, bm)
    if prepared is None or prepared.key != prepare_key(motion_in, bm, cfg):
        prepared = _prepare(motion_in, bm, cfg, cancel)
    cancel()
    motion = prepared.motion
    rest, rest_joints = prepared.rest, prepared.rest_joints
    src_fps = motion.fps
    target_fps = float(cfg.input.target_fps)
    fps = target_fps if target_fps > 0 else src_fps
    raw_q, raw_root, valid = resample(motion.quats, motion.root_pos, np.asarray(motion.valid, bool), src_fps, fps)
    valid_input = resample_mask(prepared.valid_input, src_fps, fps)
    if prepared.outliers is not None:
        for name, reason in prepared.outliers.skipped.items():
            warns.append(reason)
        if prepared.outliers.long_runs:
            parts = '、'.join(f'{PART_LABELS[n]} {s / src_fps:.1f}〜{(e + 1) / src_fps:.1f} 秒'
                             for n, s, e in prepared.outliers.long_runs[:3])
            warns.append(f'外れらしい動きが長く続く区間があります（{parts}）。前後どちらが正しいか決められないので'
                         '置き換えていません')
    gaps = runs(~prepared.valid_input)
    if gaps:
        longest = max(e - s + 1 for s, e in gaps) / src_fps
        if longest > float(cfg.input.max_gap_warn_sec):
            warns.append(f'人物を検出できなかった区間が {longest:.1f} 秒続いています（前後から補間しただけの動きです）')

    smooth_at_source = src_fps >= fps - 1e-6
    if smooth_at_source:
        quats, root, jitter_info = prepared.quats, prepared.root, prepared.jitter_info
        smooth_fps = src_fps
    else:
        q, r, _ = resample(prepared.despiked, motion.root_pos, valid_input, src_fps, fps)
        quats, jitter_info = stabilize_pose(quat.make_continuous(q), fps, cfg.jitter, rest_joints)
        root = stabilize_root(r, cfg.jitter.root_median_window)
        smooth_fps = fps

    # ---- 2b. 手首の向きの補正（MediaPipe） ----
    wrist_info = None
    if hands_analysis is not None and cfg.wrist.enabled:
        progress(0.08, '手首の向きの補正')

        def fk(q):
            return forward_kinematics(q, root, rest_joints, bm.parents)[0]

        quats, wrist = correct_wrists(quats, fk, rest_joints, hands_analysis, smooth_fps, cfg.wrist, cfg.wrist_limits)
        wrist_info = wrist.info
    cancel()

    if smooth_at_source:
        quats, root, _ = resample(quats, root, valid_input, src_fps, fps)
    quats = quat.make_continuous(quats)
    raw_q = quat.make_continuous(raw_q)
    T = len(quats)

    # ---- 3. FK ----
    progress(0.12, '床・接地・奥行きの推定')
    kin = compute_kinematics(quats, root, rest)
    kin_raw = compute_kinematics(raw_q, raw_root, rest)

    # ---- 4. 床 ----
    floor = estimate_floor(kin, fps, cfg.floor, rest.points.mean(2), valid)
    kin, kin_raw = floor.apply(kin), floor.apply(kin_raw)

    # ---- 5. MMD のスケール ----
    smpl_leg = rest.leg_length()
    k = skel.mean_leg_length() / smpl_leg
    kin = apply_scale(kin, k, float(cfg.scale.depth_scale))
    kin_raw = apply_scale(kin_raw, k, float(cfg.scale.depth_scale))
    ankle_rest = k * rest.standing(rest.joints[ANKLES])
    pelvis_rest = k * rest.standing(rest.joints[0])
    cancel()

    # ---- 6a・6・6c・6b（2 回繰り返す） ----
    force_on, force_off = normalize_overrides(contact_overrides, T)
    ground_valid = valid_input & ~force_on.any(1)     # 足ピタを強制した所はジャンプとみなさない
    axis = depth_axis(floor.rotation)
    ground = ground_offset(kin, fps, k, cfg.ground, ground_valid)
    judged = ground.apply(kin)
    kin_pose, lean = kin_raw, None
    auto = None
    for _ in range(max(1, int(cfg.depth.passes)) if cfg.depth.reconstruct else 1):
        cancel()
        auto = detect_contacts(judged.contact_points, fps, cfg.contact, unit=k,
                               speeds=detection_speeds(judged, axis, fps, k, cfg.depth), valid=valid)
        contact = apply_overrides(auto, force_on, force_off)
        if lean is None:
            lean = estimate_lean(judged, contact, axis, rest, fps, k, cfg.lean, ground.flight, valid)
            kin, kin_pose = lean.apply(kin), lean.apply(kin_raw)
        depth = reconstruct_depth(kin, kin_pose, contact, axis, fps, k, cfg.depth, valid)
        moved = depth.apply(kin)
        ground = ground_offset(moved, fps, k, cfg.ground, ground_valid, contact)
        judged = ground.apply(moved)
    kin = judged
    heights = np.asarray(kin.contact_points)[..., 1]
    contact = contact.__class__(contact.flags, contact.segments, heights, contact.speeds)
    auto = auto.__class__(auto.flags, auto.segments, heights, auto.speeds)
    flight = np.asarray(ground.flight, bool)
    contact = apply_overrides(without_frames(contact, flight, cfg.contact), force_on, force_off)
    auto = without_frames(auto, flight, cfg.contact)
    clamped = [name for name, c in zip(('前後', '左右'), lean.clamped) if c]
    if clamped:
        warns.append(f'{"・".join(clamped)}の傾きの補正角が上限に達しました（座る・寄りかかる動きでは傾きの補正を外してください）')

    # ---- 7. 足ＩＫ ----
    progress(0.3, '足ＩＫ・センター')
    foot_axes = rest.points[:, 1].mean(1) - rest.points[:, 0].mean(1)
    rt = Retargeter(skel, rest.joints, foot_axes, cfg.retarget)
    warns.extend(rt.warnings)
    ik = build_foot_ik(kin.joints[:, ANKLES], ankle_rest, rt.foot_ik_quats(kin.glob_rot), contact, fps, k,
                       cfg.foot_ik, sole_points=mmd_sole_points(skel), sole_floor=sole_floor_levels(skel, ankle_rest))
    cancel()

    # ---- 8. センター ----
    geom = ReachGeometry.from_skeleton(skel)
    support = supporting_legs(contact.flags, mmd_sole_heights(skel, ik), float(cfg.contact.exit_height_m) * k,
                              int(cfg.foot_ik.blend_frames), flight)
    center = stabilize_center(kin_raw.root_pos, kin.root_pos, pelvis_rest, kin.joints[:, ANKLES], ik, contact, geom,
                              rt.global_matrix('下半身', kin.glob_rot), fps, k, cfg.center, depth.enabled,
                              legs=support, flight=flight)
    if center.exceed_after:
        warns.append(f'届く高さへの補正の後も {center.exceed_after} フレームで脚が伸び切っています')
    cancel()

    # ---- 9. 回転のリターゲット・9l 脚・9h 手の位置・9a 腕どうし ----
    progress(0.4, '回転のリターゲット')
    local = rt.local_quats(kin.glob_rot)
    poles = knee_poles(kin, fps, cfg.leg_keys.pole_one_euro)
    legs = leg_rotations(skel, center.delta, ik, local, poles, cfg)
    raw_local = dict(local)
    reused = arm_fix is not None and all(b not in local or (b in arm_fix and len(arm_fix[b]) == T)
                                         for b in ARM_FIX_BONES)
    if reused:
        progress(0.6, '腕の補正（前回の結果を使います）')
        for b in ARM_FIX_BONES:
            if b in local:
                local[b] = quat.make_continuous(quat.normalize(quat.mul(arm_fix[b], local[b])))
        reach_info = arms_info = contacts_info = limits_info = dict(reused=True)
    else:
        local, reach = keep_hand_positions(skel, rt, kin.glob_rot, local, rest.joints,
                                           bm.rest_vertices(motion.betas), bm.weights, cfg.hand_reach, cfg.contacts, k)
        cancel()
        progress(0.45, '腕どうしの貫通の防止')
        local, arms = resolve_arm_collisions(skel, rt, kin.glob_rot, local, cfg.arm_collision, k, fps, axis, cancel)
        cancel()

        # ---- 9c → 9b → 9c ----
        progress(0.6, '腕・手と体の接触')
        local, _ = limit_wrists(skel, local, cfg.wrist_limits, fps)
        local, contacts, limits = solve_contacts(skel, rt, kin.glob_rot, local, cfg, k, fps, rest.joints,
                                                 leg_glob=None if legs is None else legs.glob, cancel=cancel)
        cancel()
        n_before, n_after = arms.overlap_frames(OVERLAP_TOL_M * k)
        reach_info = reach.info
        arms_info = dict(mode=arms.mode, overlap_frames=[n_before, n_after])
        contacts_info, limits_info = contacts.info, limits.info
    arm_correction = {b: quat.mul(local[b], quat.conj(raw_local[b])) for b in ARM_FIX_BONES if b in local}

    # ---- 10. キー列 ----
    progress(0.85, 'キーの作成')
    tracks = build_tracks(skel, center.delta, ik, local, cfg, poles=poles)
    info = dict(
        frames=T, fps=fps, source_fps=src_fps, scale=round(k, 5),
        floor=floor.info(), lean=lean.info(),
        jumps=[[int(s), int(e)] for s, e in runs(flight)],
        outliers=None if prepared.outliers is None else prepared.outliers.info(),
        jitter=jitter_info, wrist=wrist_info, hand_reach=reach_info, arm_collision=arms_info,
        contacts=contacts_info, wrist_limits=limits_info, legs=None if legs is None else legs.info,
        center=dict(mode=center.mode, exceed_before=center.exceed_before, exceed_after=center.exceed_after),
        foot_ik=dict(clamped_frames=ik.clamped_frames))
    progress(1.0, '変換が完了しました')
    return ConversionResult(cfg, fps, k, skel, rt, kin, kin_raw, valid, valid_input, floor, ground, contact, auto, ik,
                            center, ankle_rest, pelvis_rest, geom, local, poles, tracks, legs, arm_correction, info, warns)
