"""全身の重心（体の部位の質量比から）。"""
import numpy as np

HIPS = (1, 2)
# 体の部位の質量比と重心の位置（de Leva 1996, 男性）。(近位の関節, 遠位の関節, 質量比, 近位からの割合)
# 関節の番号 -1 は左右の股関節の中点。体幹は首 → 背骨2 → 背骨1 → 股関節の中点で近似する
SEGMENTS = [
    (15, 15, 0.0694, 0.0),                                                      # 頭・首
    (12, 6, 0.1596, 0.2999), (6, 3, 0.1633, 0.4502), (3, -1, 0.1117, 0.6115),   # 体幹
    *[seg for side in range(2) for seg in (
        (16 + side, 18 + side, 0.0271, 0.5772),     # 上腕
        (18 + side, 20 + side, 0.0162, 0.4574),     # 前腕
        (20 + side, 22 + side, 0.0061, 0.7900),     # 手
        (1 + side, 4 + side, 0.1416, 0.4095),       # 大腿
        (4 + side, 7 + side, 0.0433, 0.4459),       # 下腿
        (7 + side, 10 + side, 0.0137, 0.5000))],    # 足
]


def com_weights(num_joints=24):
    """(J,) 全身の重心 = Σ 重み × 関節位置 となる関節ごとの重み（合計 1）。"""
    w = np.zeros(num_joints)

    def add(j, m):
        if j < 0:
            w[list(HIPS)] += 0.5 * m
        else:
            w[j] += m

    for a, b, m, t in SEGMENTS:
        add(a, m * (1.0 - t))
        add(b, m * t)
    return w / w.sum()


def body_com(joints):
    """(..., J, 3) の関節位置から全身の重心 (..., 3)。"""
    return np.einsum('...jc,j->...c', np.asarray(joints, np.float64), com_weights())
