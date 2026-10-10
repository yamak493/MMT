"""ステージ6a: 接地の拘束（全身の上下のずれを除き、少なくとも片足が床に着くようにする）。

単眼推定では奥行きがぶれ、カメラが高い・見下ろしだとそれが上下のずれにもなり、両足が床から浮いた
（埋まった）状態が数秒続く。人は長く宙に浮いていられない（ジャンプの滞空は長くても 0.6 秒程度）ので、
足の高さから上下のずれを求め、体全体を上下に平行移動して打ち消す。両足が床から離れている区間は、物理的に
ジャンプとしてありうるもの（短く、骨盤が重力の放物線を描く）だけをジャンプとして残す。
"""
from dataclasses import dataclass

import numpy as np

from ..core import filters


@dataclass
class GroundResult:
    lowest: np.ndarray       # (T,) 補正前の両足の最下点の高さ
    offset: np.ndarray       # (T,) 上下のずれ。体全体からこの値を引く
    flight: np.ndarray       # (T,) bool ジャンプとして残した滞空のフレーム
    window: int
    enabled: bool
    height: np.ndarray = None

    def apply(self, kin):
        if not self.enabled:
            return kin
        shift = np.zeros((len(self.offset), 3))
        shift[:, 1] = -self.offset
        return kin.transformed(None, shift)


def lowest_foot_height(kin):
    """(T,) 両足のかかと・つま先のうち、最も低い点の高さ。"""
    h = np.asarray(kin.contact_points)[..., 1]
    return h.reshape(len(h), -1).min(axis=1)


def support_height(kin, contact=None):
    """(T,) 上下のずれを求める足の高さ。接地判定があれば接地している足（両足なら低いほう）の最も低い点、
    接地している足が無いフレームと接地判定の前は、両足の最下点。"""
    lowest = lowest_foot_height(kin)
    if contact is None:
        return lowest
    flags = np.asarray(contact.flags, bool)
    sole = np.asarray(kin.contact_points)[..., 1].min(-1)
    on = flags.any(1)
    h = lowest.copy()
    h[on] = np.where(flags, sole, np.inf).min(1)[on]
    return h


def flight_window(fps, max_flight_sec):
    """滞空が max_flight_sec 以下の区間の中央のフレームからも、窓に接地しているフレームが入る窓幅（奇数）。"""
    n = int(np.ceil(float(max_flight_sec) * fps))
    return 2 * ((n + 2) // 2) + 1


def vertical_accel(y, s, e, fps):
    """フレーム s〜e（前後 1 フレームずつ含めて）の y に 2 次式を当てはめた加速度 [単位/s²]。"""
    lo, hi = max(0, s - 1), min(len(y), e + 2)
    t = (np.arange(lo, hi) - 0.5 * (lo + hi - 1)) / fps
    return 2.0 * np.polyfit(t, y[lo:hi], 2)[0]


def is_ballistic(pelvis_y, s, e, fps, unit, cfg):
    """フレーム s〜e の滞空がジャンプとしてありうるか（長さと、骨盤の上下の加速度が重力に合うか）。"""
    n = e - s + 1
    if n > float(cfg.max_flight_sec) * fps or n < float(cfg.min_flight_sec) * fps:
        return False
    lo_g, hi_g = cfg.flight_accel_g
    a = -vertical_accel(pelvis_y, s, e, fps) / (9.8 * unit)
    return float(lo_g) <= a <= float(hi_g)


def find_flight(h, pelvis_y, s, e, fps, unit, cfg):
    """滞空の候補 s〜e のうち、足が最も高いフレームを含み、ジャンプとしてありうる最も長い部分区間。"""
    peak = s + int(np.argmax(h[s:e + 1]))
    longest = min(e - s + 1, int(np.floor(float(cfg.max_flight_sec) * fps)))
    shortest = int(np.ceil(float(cfg.min_flight_sec) * fps))
    for n in range(longest, shortest - 1, -1):
        for a in range(max(s, peak - n + 1), min(peak, e - n + 1) + 1):
            if is_ballistic(pelvis_y, a, a + n - 1, fps, unit, cfg):
                return a, a + n - 1
    return None


def ground_offset(kin, fps, unit, cfg, valid=None, contact=None):
    """ステージ6a。kin: 床 y = 0 の座標（MMD 単位）/ unit: スケール係数。

    valid: 人物を検出できたフレーム（False のフレームは滞空とみなさない）/ contact: ステージ6 の接地判定。
    """
    lowest = lowest_foot_height(kin)
    T = len(lowest)
    window = flight_window(fps, cfg.max_flight_sec)
    height = support_height(kin, contact)
    if not cfg.enabled or T == 0:
        return GroundResult(lowest, np.zeros(T), np.zeros(T, bool), window, False, height)
    h = filters.median_time(height, 5)
    envelope = filters.moving_min(h, window)
    flight = h - envelope > float(cfg.flight_height_m) * unit
    if valid is not None:
        flight &= np.asarray(valid, bool)
    pelvis_y = filters.median_time(kin.root_pos[:, 1], 3)
    for s, e in filters.runs(flight):
        flight[s:e + 1] = False
        found = find_flight(h, pelvis_y, s, e, fps, unit, cfg)
        if found is not None:
            flight[found[0]:found[1] + 1] = True
    t = np.arange(T)
    offset = np.interp(t, t[~flight], h[~flight]) if (~flight).any() else envelope
    offset = filters.gaussian_time(offset, float(cfg.smooth_sec) * fps)
    return GroundResult(lowest, offset, flight, window, True, height)


def floating_frames(lowest, fps, height, max_flight_sec):
    """両足の最下点が height より高い状態が max_flight_sec より長く続くフレームの数。"""
    limit = float(max_flight_sec) * fps
    return int(sum(e - s + 1 for s, e in filters.runs(np.asarray(lowest) > height) if e - s + 1 > limit))
