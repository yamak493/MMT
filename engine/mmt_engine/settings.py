"""変換の設定（既定値と、アプリから渡された上書きの合成）。

長さのしきい値はメートルで書き、MMD のスケールへ変換した後のステージでは内部でスケール係数を掛けて使う。
アプリの画面に出すのは一部の項目だけで、残りは既定値のまま使う（project.json に上書きだけを保存する）。
"""
import copy

DEFAULTS = {
    # ---- 1. 読み込み・正規化 ----
    'input': {
        'target_fps': 30.0,          # MMD の fps。外れフレームの除外（1b・2）は入力の fps のまま行ってからリサンプルする
        'max_gap_warn_sec': 0.5,     # 人物を検出できなかった区間がこれより長いと警告する
    },
    # ---- 1b. 慣性・重力による外れフレームの除外 ----
    'outliers': {
        'enabled': True,
        'max_run_sec': 0.5,          # これより長く続く外れは、前後どちらが正しいか決められないので置き換えない
        'max_ratio': 0.15,           # 外れがこの割合より多い検出は使わない（動画に合っていない）
        'pose': {
            'enabled': True,
            'neighbours': 3,         # 予測に使う前後それぞれのフレーム数
            'threshold': 6.0,        # 予測との差が、差の中央値のこの倍を超えたら外れ
            'min_m': 0.08,           # ただし差がこれ未満なら外れにしない
            'axis_length_m': 0.1,    # 関節の向きを、軸の先のこの長さの点の位置として測る
            'keep_smooth': True,     # 区間の外から速さを保って出入りする（本当に速い）動きは外れにしない
        },
        'gravity': {
            'enabled': True,         # 人に持ち上げられる・壁を押す・カメラが動く動画では false
            'friction': 1.0,
            'max_force_g': 6.0,
            'threshold': 6.0,
            'min_m': 0.05,
            'min_depth_m': 0.2,
            'iterations': 300,
        },
    },
    # ---- 2. 姿勢のジッター制御 ----
    'jitter': {
        'outlier_deg_per_s': 720.0,
        'one_euro': {
            'd_cutoff': 1.0,
            'zero_phase': True,
            'groups': {
                'torso': {'min_cutoff': 1.5, 'beta': 0.5},
                'head': {'min_cutoff': 1.5, 'beta': 0.5},
                'arm': {'min_cutoff': 2.5, 'beta': 1.0},
                'wrist': {'min_cutoff': 2.0, 'beta': 0.5},
                'leg': {'min_cutoff': 2.5, 'beta': 1.0},
            },
        },
        'lowpass_hz': {'pelvis': 0.0, 'spine': 4.0, 'head': 4.0},
        'lowpass_order': 2,
        'root_median_window': 5,
        'hand_position': {
            'enabled': True,
            'min_cutoff': 2.5,
            'beta': 1.0,
            'speed_m_per_s': [0.2, 0.6],
            'blend_sec': 0.07,
            'max_deg': 20.0,
        },
    },
    # ---- 2b. 手首の向きの補正（MediaPipe Hands） ----
    'wrist': {
        'enabled': True,
        'min_presence': 0.5,
        'full_presence': 0.7,
        'check_2d_deg': 45.0,
        'max_swing_deg': 100.0,
        'max_twist_deg': 160.0,
        'reject_margin_deg': 20.0,
        'max_disagree_deg': 180.0,
        'blend_sec': 0.1,
        'hold_sec': 0.5,
        'max_speed_deg_per_s': 360.0,
        'strength': 1.0,
        'fov_deg': 55.0,
        'smooth': {'min_cutoff': 1.5, 'beta': 0.5, 'd_cutoff': 1.0},
    },
    # ---- 3. FK と接地点 ----
    'body': {
        'heel_toe_vertices': 4,
        'sole_band_m': 0.025,
    },
    # ---- 4. 床面推定 ----
    'floor': {
        'low_speed_m_per_s': 0.2,
        'ransac_iterations': 300,
        'ransac_inlier_m': 0.02,
        'min_points': 30,
        'align_tilt': True,
        'max_tilt_deg': 30.0,
        'tilt_method': 'vectors',    # vectors | plane
        'min_vector_spread_m': 0.08,
        'sole_normal_weight_m': 0.05,
        'min_spread_m': 0.15,
        'horizontal_origin': 'first_frame',   # first_frame | mean | none
        'flip_facing': False,
        'seed': 0,
    },
    # ---- 5. MMD スケール ----
    'scale': {
        'depth_scale': 1.0,
    },
    # ---- 6a. 接地の拘束 ----
    'ground': {
        'enabled': True,
        'max_flight_sec': 0.6,
        'min_flight_sec': 0.15,
        'flight_height_m': 0.03,
        'flight_accel_g': [0.5, 2.0],
        'smooth_sec': 0.1,
    },
    # ---- 6. 接地判定 ----
    'contact': {
        'enter_height_m': 0.03,
        'enter_speed_m_per_s': 0.25,
        'exit_height_m': 0.05,
        'exit_speed_m_per_s': 0.4,
        'fill_gap_frames': 2,
        'min_contact_frames': 4,
    },
    # ---- 6b. 奥行きの再構成 ----
    'depth': {
        'detect_root_cutoff_hz': 0.5,
        'detect_foot_cutoff_hz': 2.0,
        'reconstruct': True,
        'passes': 2,
        'contact_sigma_m': 0.005,
        'prior_sigma_m': 0.5,
        'accel_sigma_m_per_s2': 3.0,
    },
    # ---- 6c. 前後の傾きの補正 ----
    'lean': {
        'enabled': True,
        'window_sec': 3.0,
        'max_deg': 25.0,
        'outlier_m': 0.5,
        'double_support_center_weight': 0.1,
        'support_band_m': 0.02,
        'fill_gap_sec': 0.5,
    },
    # ---- 7. 足ＩＫ ----
    'foot_ik': {
        'swing_one_euro': {'min_cutoff': 2.0, 'beta': 1.0, 'd_cutoff': 1.0, 'zero_phase': True},
        'swing_lowpass_hz': 4.0,
        'blend_frames': 5,
        'lock_rotation': True,
        'snap_to_floor': True,
        'pivot': {
            'enabled': True,
            'raise_m': [0.05, 0.01],
            'turn_deg_per_s': [90.0, 45.0],
            'min_turn_deg': 20.0,
            'pivot_ratio': 0.5,
            'min_frames': 3,
            'smooth_sec': 0.035,
        },
        'clamp_floor': True,
        'max_boundary_step_m': 0.05,
    },
    # ---- 8. センター ----
    'center': {
        'mode': 'A',                 # A: 軸別の平滑化 / B: 接地している足からの逆算
        'axis_one_euro': {
            'x': {'min_cutoff': 1.5, 'beta': 0.5},
            'y': {'min_cutoff': 1.5, 'beta': 0.5},
            'z': {'min_cutoff': 1.5, 'beta': 0.5},
            'z_raw': {'min_cutoff': 0.5, 'beta': 0.3},
        },
        'd_cutoff': 1.0,
        'zero_phase': True,
        'lowpass_hz': {'x': 4.0, 'y': 4.0, 'z': 0.0},
        'lowpass_order': 2,
        'mode_b': {'final_one_euro': {'min_cutoff': 3.0, 'beta': 0.5}},
        'reach_ratio': 0.98,
        'reach_min_window': 9,
        'reach_gaussian_sigma': 2.0,
        'reach_max_drop_m': 0.3,
    },
    # ---- 9. 回転のリターゲット ----
    'retarget': {
        'torso_up': 'vertical',      # vertical | joints
        'collar': 'identity',        # identity | direction
        'wrist': 'forearm',          # forearm | direction
        'foot': 'yaw',               # yaw | direction
    },
    # ---- 9l. 脚の回転（膝の向き） ----
    'leg_keys': {
        'enabled': True,
        'ankle': True,
        'pole_one_euro': {'min_cutoff': 3.0, 'beta': 4.0},
    },
    # ---- 9h. 胴に対する手の位置 ----
    'hand_reach': {
        'enabled': True,
        'near_m': 0.1,
        'far_m': 0.25,
        'bend_deg': [20.0, 45.0],
        'max_deg': 45.0,
    },
    # ---- 9a. 腕どうしの貫通の防止 ----
    'arm_collision': {
        'mode': 'left',              # left | right | none（どちらの腕の動きを自重するか）
        'radius_m': 'auto',
        'fallback_radius_m': [0.035, 0.03, 0.025],
        'radius_percentile': 50.0,
        'radius_scale': 1.0,
        'hand_length_m': 0.15,
        'margin_m': 0.005,
        'max_deg': 45.0,
        'return_deg_per_s': 120.0,
        'smooth_sec': 0.1,
        'depth_cost': 0.5,
        'elbow': True,
        'elbow_max_deg': 30.0,
    },
    # ---- 9b. 腕・手のひら・指先と体の接触 ----
    'contacts': {
        'enabled': True,
        'body_parts': {'clothes': True, 'head': True, 'torso': True, 'chest': True, 'legs': True},
        'body_source': 'auto',       # auto | rigid | mesh | config | none
        'cloth_margin_m': 0.01,
        'margin_m': 0.002,
        'skirt': 'soft',             # none | soft | hard
        'soft_ratio': 0.5,
        'fingers': False,
        'radius_percentile': 50.0,
        'radius_scale': 1.0,
        'finger_radius_m': [0.009, 0.008, 0.008, 0.008, 0.007],
        'palm_radius_m': 0.012,
        'upper_arm_skip': 0.35,
        'joint_weights': {'shoulder': 4.0, 'elbow': 2.0, 'wrist': 1.0},
        'max_deg': {'shoulder': 45.0, 'elbow': 30.0, 'wrist': 40.0},
        'return_deg_per_s': 120.0,
        'smooth_sec': 0.1,
    },
    # ---- 9c. 手首の可動域 ----
    'wrist_limits': {
        'enabled': True,
        'pronation_deg': 90.0,
        'supination_deg': 90.0,
        'flexion_deg': 90.0,
        'extension_deg': 70.0,
        'radial_deg': 25.0,
        'ulnar_deg': 55.0,
        'max_speed_deg_per_s': 720.0,
    },
    # ---- 10. 捩りボーン ----
    'twist': {
        'enabled': True,
        'max_deg': 160.0,
        'fade_swing_deg': [150.0, 170.0],
    },
    # ---- 出力の種類: フル [接地優先] ----
    'locked': {
        'both_feet': True,
        'sigma_frames': 1.5,
        'still_m': 0.03,
    },
    # ---- 口パク ----
    'lipsync': {
        'morphs': {'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お', 'n': 'ん'},
        'strength': 1.0,
        'min_open': 0.3,
        'open_range_db': 24.0,
        'closed_weight': 1.0,
        'reference_percentile': 95.0,
        'noise_percentile': 10.0,
        'min_reference_dbfs': -50.0,
        'voice_on_db': -30.0,
        'voice_off_db': -40.0,
        'on_above_noise_db': 12.0,
        'off_above_noise_db': 6.0,
        'max_on_db': -15.0,
        'fill_gap_sec': 0.05,
        'min_voice_sec': 0.06,
        'event_margin_sec': 0.1,
        'lead_sec': 0.05,
        'consonant_sec': 0.25,
        'max_closed_sec': 0.3,
        'fallback_vowel': 'a',
        'smooth_sec': 0.03,
        'thin_tolerance': 0.01,
    },
    # ---- 手の形（指） ----
    'hands': {
        'min_presence': 0.5,
        'landmarks': 'screen',       # screen | world
        'extended_below_deg': 65.0,
        'curled_above_deg': 110.0,
        'softness_deg': 25.0,
        'thumb_extended_below_deg': 50.0,
        'thumb_softness_deg': 20.0,
        'thumb_away': 0.6,
        'thumb_away_softness': 0.2,
        'default_score': 0.5,
        'smooth_sec': 0.1,
        'switch_cost': 3.0,
        'min_hold_sec': 0.25,
        'max_gap_sec': 0.5,
        'missing_shape': 'default',  # default | hold
        'transition_sec': 0.15,
        'bones': {
            'thumb': ['親指０', '親指１', '親指２'],
            'index': ['人指１', '人指２', '人指３'],
            'middle': ['中指１', '中指２', '中指３'],
            'ring': ['薬指１', '薬指２', '薬指３'],
            'pinky': ['小指１', '小指２', '小指３'],
        },
        'presets': {
            'default': ['relaxed', 'relaxed', 'relaxed', 'relaxed', 'relaxed'],
            'thumb': ['straight', 'curled', 'curled', 'curled', 'curled'],
            'index': ['curled', 'straight', 'curled', 'curled', 'curled'],
            'middle': ['curled', 'curled', 'straight', 'curled', 'curled'],
            'ring': ['curled', 'curled', 'curled', 'straight', 'curled'],
            'pinky': ['curled', 'curled', 'curled', 'curled', 'straight'],
            'fist': ['curled', 'curled', 'curled', 'curled', 'curled'],
            'scissors': ['curled', 'open', 'open', 'curled', 'curled'],
            'open': ['open', 'open', 'open', 'open', 'open'],
            'pistol': ['open', 'straight', 'curled', 'curled', 'curled'],
            'fox': ['curled', 'straight', 'curled', 'curled', 'straight'],
            'ok': ['pinch', 'pinch', 'open', 'open', 'open'],
        },
        # 状態ごとの指の角度 [度]: [付け根, 第 2 関節, 第 3 関節, 開き]（pinch は親指と人差し指で輪を作る形）
        'angles': {
            'thumb': {'open': [-12.0, 0.0, 0.0, 0.0], 'straight': [0.0, 0.0, 0.0, 0.0],
                      'relaxed': [8.0, 10.0, 10.0, 0.0], 'curled': [5.0, 55.0, 55.0, 0.0],
                      'pinch': [12.0, 25.0, 20.0, 0.0]},
            'index': {'open': [0.0, 0.0, 0.0, 12.0], 'straight': [0.0, 0.0, 0.0, 0.0],
                      'relaxed': [12.0, 20.0, 12.0, 0.0], 'curled': [80.0, 100.0, 65.0, 0.0],
                      'pinch': [35.0, 60.0, 40.0, 0.0]},
            'middle': {'open': [0.0, 0.0, 0.0, -6.0], 'straight': [0.0, 0.0, 0.0, 0.0],
                       'relaxed': [15.0, 22.0, 12.0, 0.0], 'curled': [85.0, 100.0, 65.0, 0.0]},
            'ring': {'open': [0.0, 0.0, 0.0, -14.0], 'straight': [0.0, 0.0, 0.0, 0.0],
                     'relaxed': [18.0, 24.0, 14.0, 0.0], 'curled': [88.0, 100.0, 65.0, 0.0]},
            'pinky': {'open': [0.0, 0.0, 0.0, -24.0], 'straight': [0.0, 0.0, 0.0, 0.0],
                      'relaxed': [22.0, 26.0, 16.0, 0.0], 'curled': [90.0, 95.0, 60.0, 0.0]},
        },
        'thumb_across': [4.0, 3.0, 3.0],
    },
}


class Config(dict):
    """cfg['jitter']['one_euro'] と cfg.jitter.one_euro の両方で読める dict。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else copy.deepcopy(v)) for k, v in self.items()}


def _wrap(value):
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    return value


def _merge(base, update, path=''):
    for key, value in update.items():
        where = f'{path}.{key}' if path else key
        if key not in base:
            raise KeyError(f'未知の設定項目です: {where}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f'設定 {where} は辞書で指定してください')
            _merge(base[key], value, where)
        else:
            base[key] = value


def make_config(*overrides):
    """既定値に上書き（入れ子の dict。ドット区切りのキー 'a.b' も可）を順に重ねた Config。"""
    cfg = copy.deepcopy(DEFAULTS)
    for item in overrides:
        if not item:
            continue
        _merge(cfg, expand_dotted(item))
    return _wrap(cfg)


def expand_dotted(item):
    """{'center.mode': 'B', 'contact': {...}} → {'center': {'mode': 'B'}, 'contact': {...}}。"""
    out = {}
    for key, value in dict(item).items():
        parts = str(key).split('.')
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
            node[parts[-1]].update(expand_dotted(value))
        elif isinstance(value, dict):
            node[parts[-1]] = expand_dotted(value)
        else:
            node[parts[-1]] = value
    return out


def with_overrides(cfg, item):
    """Config cfg に上書きを重ねた新しい Config（cfg は変えない）。"""
    base = cfg.to_dict() if isinstance(cfg, Config) else copy.deepcopy(cfg)
    _merge(base, expand_dotted(item))
    return _wrap(base)
