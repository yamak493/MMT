"""ステージ6: 接地判定（ヒステリシス付き）と、タイムラインで編集した足ピタ（接地）の上書き。"""
from dataclasses import dataclass

import numpy as np

from ..core.filters import runs
from .floor import horizontal_speed


@dataclass
class ContactResult:
    flags: np.ndarray       # (T, 2 足) bool
    segments: list          # 足ごとの [(開始, 終了), ...]（終了を含む）
    heights: np.ndarray     # (T, 2 足, 2 [かかと, つま先])
    speeds: np.ndarray      # (T, 2, 2) 水平速度


def hysteresis(enter, stay):
    state = False
    out = np.zeros(len(enter), bool)
    for t in range(len(enter)):
        state = bool(stay[t]) if state else bool(enter[t])
        out[t] = state
    return out


def clean_flags(flags, fill_gap, min_len):
    """短い非接地の隙間を埋めてから、短い接地区間を捨てる。"""
    flags = np.array(flags, bool, copy=True)
    for s, e in runs(~flags):
        if s > 0 and e < len(flags) - 1 and e - s + 1 <= fill_gap:
            flags[s:e + 1] = True
    for s, e in runs(flags):
        if e - s + 1 < min_len:
            flags[s:e + 1] = False
    return flags


def from_flags(flags, heights, speeds):
    flags = np.asarray(flags, bool)
    return ContactResult(flags, [runs(flags[:, f]) for f in range(flags.shape[1])], heights, speeds)


def without_frames(contact, mask, cfg):
    """mask のフレーム（ステージ6a の滞空）で、どちらの足も接地していないことにした ContactResult。"""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return contact
    flags = np.array(contact.flags, bool, copy=True)
    flags[mask] = False
    for foot in range(flags.shape[1]):
        flags[:, foot] = clean_flags(flags[:, foot], int(cfg.fill_gap_frames),
                                     int(cfg.min_contact_frames)) & ~mask
    return from_flags(flags, contact.heights, contact.speeds)


def detect_contacts(points, fps, cfg, unit=1.0, speeds=None, valid=None):
    """points: (T, 2 足, P 点, 3)（床 y = 0）。かかとかつま先が「低く・遅い」なら接地。

    接地の開始より終了の条件を緩くして境目のばたつきを防ぐ。speeds: 奥行きのぶれを除いた水平速度。
    valid: False のフレーム（前後から補間しただけ）では接地を始めない。
    """
    points = np.asarray(points, np.float64)
    heights = points[..., 1]
    speeds = horizontal_speed(points, fps) if speeds is None else np.asarray(speeds, np.float64)
    enter = ((heights < cfg.enter_height_m * unit) & (speeds < cfg.enter_speed_m_per_s * unit)).any(-1)
    if valid is not None:
        enter &= np.asarray(valid, bool)[:, None]
    stay = ((heights < cfg.exit_height_m * unit) & (speeds < cfg.exit_speed_m_per_s * unit)).any(-1)
    flags = np.zeros(points.shape[:2], bool)
    for foot in range(points.shape[1]):
        f = hysteresis(enter[:, foot], stay[:, foot])
        flags[:, foot] = clean_flags(f, int(cfg.fill_gap_frames), int(cfg.min_contact_frames))
    return from_flags(flags, heights, speeds)


# ---- タイムラインで編集した足ピタの上書き ----
def normalize_overrides(overrides, num_frames):
    """[{foot: 0|1, start, end, mode: 'on'|'off'}] → ((T, 2) 強制的に接地, (T, 2) 強制的に非接地)。

    後に並んでいる上書きほど優先する（同じフレームに on と off があれば後のもの）。
    """
    force_on = np.zeros((num_frames, 2), bool)
    force_off = np.zeros((num_frames, 2), bool)
    for item in overrides or ():
        foot = int(item['foot'])
        if foot not in (0, 1):
            continue
        s = max(0, int(item['start']))
        e = min(num_frames - 1, int(item['end']))
        if e < s:
            continue
        if item.get('mode') == 'off':
            force_on[s:e + 1, foot] = False
            force_off[s:e + 1, foot] = True
        else:
            force_off[s:e + 1, foot] = False
            force_on[s:e + 1, foot] = True
    return force_on, force_off


def apply_overrides(contact, force_on, force_off):
    if force_on is None or (not force_on.any() and not force_off.any()):
        return contact
    flags = (np.asarray(contact.flags, bool) | force_on) & ~force_off
    return from_flags(flags, contact.heights, contact.speeds)
