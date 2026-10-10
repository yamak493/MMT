"""MMD モデルの骨格。

アプリ（レンダラー）が PMX / PMD を babylon-mmd のパーサーで読み、変換に必要な情報だけを
リグ情報（model_rig.json）として書き出す。PMD もパーサーが PMX と同じ形にそろえるので、エンジンは
形式を区別しない。モデルが無いときは標準的な体格のボーンを使う。

位置は PMX と同じ MMD 座標（左手系・Y 上向き・正面 −Z・体の左が +X）で持ち、変換の計算は
内部座標（右手系。z を反転）で行う。
"""
import base64
import json
from pathlib import Path

import numpy as np

SIDES = ('左', '右')
MMD_TO_INTERNAL = np.array([1.0, 1.0, -1.0])

# 標準的な体格（身長 20 単位前後・A ポーズ）。名前: (位置, 親, 表示先の相対位置)
_STANDARD_LEFT = {
    '左肩': ((0.3, 15.5, 0.5), '上半身2', None),
    '左腕': ((1.4, 15.3, 0.6), '左肩', None),
    '左ひじ': ((3.45, 13.58, 0.7), '左腕', None),
    '左手首': ((5.29, 12.04, 0.6), '左ひじ', (0.69, -0.58, -0.1)),
    '左足': ((0.95, 10.9, 0.2), '下半身', None),
    '左ひざ': ((0.95, 6.1, -0.1), '左足', None),
    '左足首': ((0.95, 1.25, 0.35), '左ひざ', None),
    '左つま先': ((0.95, 0.0, -1.3), '左足首', None),
    '左足ＩＫ': ((0.95, 1.25, 0.35), '全ての親', None),
    '左つま先ＩＫ': ((0.95, 0.0, -1.3), '左足ＩＫ', None),
}
STANDARD_BONES = {
    '全ての親': ((0.0, 0.0, 0.0), None, None),
    'センター': ((0.0, 8.0, 0.0), '全ての親', None),
    'グルーブ': ((0.0, 8.2, 0.0), 'センター', None),
    '下半身': ((0.0, 11.6, 0.2), 'グルーブ', None),
    '上半身': ((0.0, 11.8, 0.2), 'グルーブ', None),
    '上半身2': ((0.0, 13.2, 0.3), '上半身', None),
    '首': ((0.0, 16.0, 0.4), '上半身2', None),
    '頭': ((0.0, 16.9, 0.2), '首', (0.0, 1.6, 0.0)),
}


def _mirror(name):
    return name.replace('左', '右', 1) if name and name.startswith('左') else name


for _name, (_pos, _parent, _tail) in _STANDARD_LEFT.items():
    STANDARD_BONES[_name] = (_pos, _parent, _tail)
    STANDARD_BONES[_mirror(_name)] = ((-_pos[0], _pos[1], _pos[2]), _mirror(_parent),
                                      None if _tail is None else (-_tail[0], _tail[1], _tail[2]))

REQUIRED_BONES = (
    'センター', '下半身', '上半身', '首', '頭',
    *[s + n for s in SIDES for n in ('肩', '腕', 'ひじ', '手首', '足', 'ひざ', '足首', '足ＩＫ')])

# PMX のボーンのフラグ
FLAG_TAIL_IS_BONE = 0x0001


class Skeleton:
    def __init__(self, positions, parents, tails, model_name='', source='standard', mesh_points=None,
                 mesh_bones=None, rigid_bodies=None, axes=None, morph_names=None):
        self.positions = {k: np.asarray(v, np.float64) for k, v in positions.items()}
        self.parents = dict(parents)
        self.tails = {k: np.asarray(v, np.float64) for k, v in tails.items() if v is not None}
        self.axes = {k: np.asarray(v, np.float64) for k, v in (axes or {}).items()}
        self.model_name = model_name
        self.source = source
        # 頂点（MMD 座標）と、そのウェイトが最も大きいボーンの名前（標準の骨格では None）
        self.mesh_points = mesh_points
        self.mesh_bones = mesh_bones
        # 剛体: [dict(name, bone, shape, size, position, rotation, mode)]（無ければ None）
        self.rigid_bodies = rigid_bodies
        self.morph_names = list(morph_names or [])

    # ---- 作り方 ----
    @classmethod
    def standard(cls):
        pos = {k: v[0] for k, v in STANDARD_BONES.items()}
        par = {k: v[1] for k, v in STANDARD_BONES.items()}
        tails = {k: np.add(v[0], v[2]) for k, v in STANDARD_BONES.items() if v[2] is not None}
        return cls(pos, par, tails, '', 'standard')

    @classmethod
    def from_rig(cls, rig):
        """リグ情報（dict か model_rig.json のパス）から作る。"""
        if isinstance(rig, (str, Path)):
            rig = json.loads(Path(rig).read_text(encoding='utf-8'))
        bones = rig['bones']
        pos, par, tails, axes = {}, {}, {}, {}
        for b in bones:
            name = b['name']
            if name in pos:
                continue   # 同名のボーンは最初のものを使う（MMD と同じ）
            pos[name] = np.asarray(b['position'], np.float64)
            p = int(b.get('parent', -1))
            par[name] = bones[p]['name'] if 0 <= p < len(bones) else None
            if b.get('axis') is not None:
                axes[name] = np.asarray(b['axis'], np.float64)
            tail = b.get('tail')
            if isinstance(tail, (int, float)) and not isinstance(tail, bool):
                t = int(tail)
                if 0 <= t < len(bones):
                    tails[name] = np.asarray(bones[t]['position'], np.float64)
            elif tail is not None:
                offset = np.asarray(tail, np.float64)
                if np.linalg.norm(offset) > 1e-6:
                    tails[name] = pos[name] + offset
        points = owners = None
        verts = rig.get('vertices')
        if verts and int(verts.get('count', 0)) > 0:
            n = int(verts['count'])
            points = np.frombuffer(base64.b64decode(verts['positions']), '<f4').astype(np.float64)
            points = points.reshape(n, 3)
            idx = np.frombuffer(base64.b64decode(verts['bones']), '<i4').astype(np.int64)
            names = np.array([b['name'] for b in bones] + [''])
            owners = names[np.where((idx >= 0) & (idx < len(bones)), idx, len(bones))]
        rigids = None
        if rig.get('rigidBodies') is not None:
            rigids = []
            for rb in rig['rigidBodies']:
                b = int(rb.get('bone', -1))
                rigids.append(dict(
                    name=rb.get('name', ''), shape=int(rb['shape']),
                    size=np.asarray(rb['size'], np.float64),
                    position=np.asarray(rb['position'], np.float64),
                    rotation=np.asarray(rb['rotation'], np.float64), mode=int(rb.get('mode', 0)),
                    bone=bones[b]['name'] if 0 <= b < len(bones) else None))
        morphs = [m['name'] for m in rig.get('morphs', [])]
        return cls(pos, par, tails, rig.get('modelName', ''), 'model', points, owners, rigids, axes, morphs)

    # ---- 参照 ----
    def has(self, name):
        return name in self.positions

    def internal(self, name):
        """内部座標（右手系）での初期位置。"""
        return self.positions[name] * MMD_TO_INTERNAL

    def tail_internal(self, name):
        return self.tails[name] * MMD_TO_INTERNAL if name in self.tails else None

    def axis_internal(self, name):
        """軸制限の軸（内部座標）。無ければ None。"""
        return self.axes[name] * MMD_TO_INTERNAL if name in self.axes else None

    def missing(self, names):
        return [n for n in names if not self.has(n)]

    def leg_length(self, side):
        """足 → ひざ → 足首 の長さ。"""
        s = SIDES[side]
        p = [self.positions[s + n] for n in ('足', 'ひざ', '足首')]
        return float(np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[1]))

    def mean_leg_length(self):
        return 0.5 * (self.leg_length(0) + self.leg_length(1))

    def is_descendant(self, name, ancestor):
        """name が ancestor そのもの、またはその子孫か。"""
        seen = set()
        while name is not None and name not in seen:
            if name == ancestor:
                return True
            seen.add(name)
            name = self.parents.get(name)
        return False

    def nearest_ancestor(self, name, candidates):
        """name の祖先のうち candidates に含まれる最も近いボーン（無ければ None）。"""
        p = self.parents.get(name)
        seen = set()
        while p is not None and p not in seen:
            if p in candidates:
                return p
            seen.add(p)
            p = self.parents.get(p)
        return None

    def chain(self, name):
        """name から根元までのボーン（根元が先頭）。"""
        out, seen = [], set()
        while name is not None and self.has(name) and name not in seen:
            out.append(name)
            seen.add(name)
            name = self.parents.get(name)
        return out[::-1]

    def depth(self, name):
        d, n = 0, name
        while self.parents.get(n) is not None and d < 256:
            n = self.parents[n]
            d += 1
        return d
