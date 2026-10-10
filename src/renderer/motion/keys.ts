// キー列の共通の型と、補間曲線。

/** 補間曲線 [x1, y1, x2, y2]（0〜127。MMD の補間曲線の 2 つの制御点） */
export type Curve = readonly [number, number, number, number];

export const LINEAR: Curve = [20, 20, 107, 107];
/** 手の形の切り替えのイーズイン・アウト */
export const EASE: Curve = [40, 0, 88, 127];

export interface BoneKeys {
    name: string;
    /** 位置のキーを持つボーン（センター・グルーブ・足ＩＫ） */
    movable: boolean;
    frames: Uint32Array;
    /** (n × 3) */
    positions: Float32Array;
    /** (n × 4) x, y, z, w */
    rotations: Float32Array;
    /** キーごとの補間曲線（前のキーからこのキーまで）。null なら全部線形 */
    curves: Curve[] | null;
}

export interface MorphKeys {
    name: string;
    frames: Uint32Array;
    weights: Float32Array;
}

export interface ComposedMotion {
    bones: BoneKeys[];
    morphs: MorphKeys[];
    /** 最後のキーのフレーム */
    lastFrame: number;
}

/** 回転の符号を前のキーにそろえる（q と −q は同じ回転。補間が遠回りしないように）。 */
export function makeContinuous(rotations: Float32Array): void {
    for (let i = 4; i < rotations.length; i += 4) {
        const dot = rotations[i] * rotations[i - 4] + rotations[i + 1] * rotations[i - 3]
            + rotations[i + 2] * rotations[i - 2] + rotations[i + 3] * rotations[i - 1];
        if (dot < 0) {
            rotations[i] = -rotations[i];
            rotations[i + 1] = -rotations[i + 1];
            rotations[i + 2] = -rotations[i + 2];
            rotations[i + 3] = -rotations[i + 3];
        }
    }
}
