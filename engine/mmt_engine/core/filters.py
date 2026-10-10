"""時系列のフィルタと、区間（bool 列の連続部分）の操作。入力はどれも (T, ...) で時間方向は axis 0。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d

# フレーム数で決めた設定（メディアンの窓・外れの予測に使うフレーム数など）の基準の fps
REFERENCE_FPS = 30.0


def frames_for_fps(n, fps, odd=False):
    """30fps で決めたフレーム数 n を、fps で使う値にする。

    fps が 30 より高いときは同じ時間になるように増やし、低いときは n のまま（減らすと窓が小さすぎて
    メディアン・予測が成り立たなくなる）。odd なら増やした値を奇数にする（窓の中心を保つ）。
    """
    n = int(n)
    m = max(n, int(round(n * float(fps) / REFERENCE_FPS)))
    if odd and m != n and m % 2 == 0:
        m += 1
    return m


def _alpha(cutoff, fps):
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, np.float64))
    return 1.0 / (1.0 + tau * fps)


def _speed(dx, vector_axis):
    if vector_axis is None:
        return np.abs(dx)
    return np.linalg.norm(dx, axis=vector_axis, keepdims=True)


def _one_euro_pass(x, fps, min_cutoff, beta, d_cutoff, vector_axis):
    out = np.empty_like(x)
    # 先頭数フレームの傾きで一定速度に動いているとみなし、その定常状態（一定の遅れ）から始める。
    # 静止状態から始めると、動いている信号の先頭に遅れが立ち上がるまでの過渡応答が出る
    m = min(5, len(x) - 1)
    v0 = (x[m] - x[0]) / m
    lag = np.zeros_like(x[0])
    for _ in range(5):
        a = _alpha(min_cutoff + beta * _speed((v0 + lag) * fps, vector_axis), fps)
        lag = v0 * (1.0 - a) / a
    out[0] = x[0] - lag
    dx_hat = (v0 + lag) * fps
    a_d = _alpha(d_cutoff, fps)
    for t in range(1, len(x)):
        dx = (x[t] - out[t - 1]) * fps
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        a = _alpha(min_cutoff + beta * _speed(dx_hat, vector_axis), fps)
        out[t] = a * x[t] + (1.0 - a) * out[t - 1]
    return out


def _point_reflect_pad(x, pad):
    """両端を点対称に延長する（一定速度で動いている区間の端に、フィルタの立ち上がりで加速度が出ない）。"""
    return np.concatenate([2 * x[:1] - x[pad:0:-1], x, 2 * x[-1:] - x[-2:-pad - 2:-1]])


def one_euro(x, fps, min_cutoff, beta, d_cutoff=1.0, zero_phase=True, vector_axis=None):
    """One Euro フィルタ（Casiez et al. 2012）。

    遅い動きほど強く（カットオフ min_cutoff [Hz]）、速い動きほど弱く（カットオフが beta × 速さ だけ上がる）
    ならす。min_cutoff・beta は x[0] の形にブロードキャストできる配列でもよい。zero_phase なら順方向と
    逆方向の結果を平均して遅れを打ち消す（オフライン処理）。vector_axis を指定すると、その軸のノルムを
    速さに使う（クォータニオンの 4 成分でカットオフを共有するなど）。
    """
    x = np.asarray(x, np.float64)
    T = len(x)
    if T < 2 or np.all(np.asarray(min_cutoff) <= 0):
        return x.copy()
    pad = min(T - 1, int(np.ceil(2.0 * fps)))
    xp = _point_reflect_pad(x, pad)
    fwd = _one_euro_pass(xp, fps, min_cutoff, beta, d_cutoff, vector_axis)
    if zero_phase:
        bwd = _one_euro_pass(xp[::-1], fps, min_cutoff, beta, d_cutoff, vector_axis)[::-1]
        fwd = 0.5 * (fwd + bwd)
    return fwd[pad:pad + T]


def lowpass(x, fps, cutoff_hz, order=2):
    """ゼロ位相のローパス（Butterworth を順方向・逆方向に掛ける。振幅特性は 2 × order 次）。

    One Euro（1 次）は遮断周波数より上の落ち方がゆるく、細かい震えを抑えようとするとダンスの拍の
    1.5〜3Hz の動きまで削る。Butterworth は通過域が平らで落ち方が急なので、拍の動きを残して震えを落とせる。
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    T = len(x)
    if T < 2 or cutoff_hz <= 0 or cutoff_hz >= 0.5 * fps:
        return x.copy()
    sos = butter(int(order), float(cutoff_hz) / (0.5 * float(fps)), output='sos')
    pad = min(T - 1, int(np.ceil(2.0 * fps)))
    return sosfiltfilt(sos, _point_reflect_pad(x, pad), axis=0, padtype=None)[pad:pad + T]


def median_time(x, window):
    """時間方向のメディアンフィルタ（端は端の値で延長）。"""
    x = np.asarray(x, np.float64)
    if window <= 1 or len(x) < 2:
        return x.copy()
    size = (int(window),) + (1,) * (x.ndim - 1)
    return median_filter(x, size=size, mode='nearest')


def moving_min(x, window):
    x = np.asarray(x, np.float64)
    if window <= 1:
        return x.copy()
    return minimum_filter1d(x, int(window), axis=0, mode='nearest')


def gaussian_time(x, sigma, radius=None):
    """時間方向のガウシアン。radius [フレーム] を指定すると、カーネルをその幅で打ち切る。"""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x.copy()
    truncate = 4.0 if radius is None else max(float(radius), 0.5) / float(sigma)
    return gaussian_filter1d(x, float(sigma), axis=0, mode='nearest', truncate=truncate)


def rate_limit(x, step):
    """(T, ...) の列の 1 フレームの変化（最後の軸のノルム）を step 以下にする。

    前向きと後ろ向きに掛けた結果の平均（どちらも変化が step 以下なので平均も step 以下。片向きだけだと遅れる）。
    各フレームの値は前後の値を結ぶ線分の上にとるので、凸な範囲に入っている列は掛けた後も同じ範囲に入っている。
    """
    def one_way(v):
        y = np.array(v, np.float64, copy=True)
        for t in range(1, len(y)):
            d = y[t] - y[t - 1]
            n = float(np.linalg.norm(d))
            if n > step:
                y[t] = y[t - 1] + d * (step / n)
        return y

    x = np.asarray(x, np.float64)
    return 0.5 * (one_way(x) + one_way(x[::-1])[::-1])


def smoothstep(t):
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def runs(mask):
    """bool 列の True が続く区間 [(開始, 終了), ...]（終了を含む）。"""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def label_runs(labels):
    """同じ値が続く区間 [(開始, 終了, 値), ...]（終了を含む）。"""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut - 1, [len(labels) - 1]])
    return [(int(s), int(e), labels[s].item()) for s, e in zip(starts, ends)]


def mask_from_runs(segments, length):
    out = np.zeros(int(length), bool)
    for s, e in segments:
        s, e = max(0, int(s)), min(int(length) - 1, int(e))
        if e >= s:
            out[s:e + 1] = True
    return out
