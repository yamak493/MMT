// PMX / PMD（babylon-mmd のパーサーで読んだ PmxObject）から、エンジンが変換に使うリグ情報（model_rig.json）を作る。
// PMD もパーサーが PMX と同じ形にそろえるので、どちらも同じに扱える。座標はファイルのまま（MMD 座標）。
import { PmxObject } from "babylon-mmd/esm/Loader/Parser/pmxObject";

const BoneFlag = PmxObject.Bone.Flag;
const WeightType = PmxObject.Vertex.BoneWeightType;

function toBase64(bytes: Uint8Array): string {
    let s = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
        s += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(s);
}

/** 頂点のウェイトが最も大きいボーン */
function dominantBone(v: PmxObject.Vertex): number {
    const bw = v.boneWeight;
    switch (v.weightType) {
        case WeightType.Bdef1:
            return bw.boneIndices as number;
        case WeightType.Bdef2:
        case WeightType.Sdef: {
            const idx = bw.boneIndices as [number, number];
            const w0 = v.weightType === WeightType.Sdef
                ? (bw.boneWeights as PmxObject.Vertex.BoneWeightSDEF).boneWeight0
                : (bw.boneWeights as number);
            return w0 >= 0.5 ? idx[0] : idx[1];
        }
        default: {
            const idx = bw.boneIndices as [number, number, number, number];
            const w = bw.boneWeights as [number, number, number, number];
            let best = 0;
            for (let i = 1; i < 4; i++) if (w[i] > w[best]) best = i;
            return idx[best];
        }
    }
}

export interface ModelRig {
    format: "mmt-rig";
    version: 1;
    modelName: string;
    bones: { name: string; parent: number; position: number[]; flags: number; tail: number | number[]; axis: number[] | null }[];
    morphs: { name: string; category: number; type: number }[];
    rigidBodies: { name: string; bone: number; shape: number; size: number[]; position: number[]; rotation: number[]; mode: number }[];
    vertices: { count: number; positions: string; bones: string };
}

export function extractRig(model: PmxObject): ModelRig {
    const n = model.vertices.length;
    const positions = new Float32Array(n * 3);
    const owners = new Int32Array(n);
    for (let i = 0; i < n; i++) {
        const v = model.vertices[i];
        positions[i * 3] = v.position[0];
        positions[i * 3 + 1] = v.position[1];
        positions[i * 3 + 2] = v.position[2];
        owners[i] = dominantBone(v);
    }
    return {
        format: "mmt-rig",
        version: 1,
        modelName: model.header.modelName,
        bones: model.bones.map((b) => ({
            name: b.name,
            parent: b.parentBoneIndex,
            position: [...b.position],
            flags: b.flag,
            tail: typeof b.tailPosition === "number" ? b.tailPosition : [...b.tailPosition],
            axis: b.flag & BoneFlag.HasAxisLimit && b.axisLimit ? [...b.axisLimit] : null
        })),
        morphs: model.morphs.map((m) => ({ name: m.name, category: m.category, type: m.type })),
        rigidBodies: model.rigidBodies.map((r) => ({
            name: r.name,
            bone: r.boneIndex,
            shape: r.shapeType,
            size: [...r.shapeSize],
            position: [...r.shapePosition],
            rotation: [...r.shapeRotation],
            mode: r.physicsMode
        })),
        vertices: {
            count: n,
            positions: toBase64(new Uint8Array(positions.buffer)),
            bones: toBase64(new Uint8Array(owners.buffer))
        }
    };
}

/** モデルが参照しているファイル（テクスチャ・スフィア・トゥーン）。モデルのフォルダからの相対パス。 */
export function referencedFiles(model: PmxObject): string[] {
    return model.textures.map((t) => t.replace(/\\/g, "/")).filter((t) => t && !/^toon(0[1-9]|10)\.bmp$/i.test(t));
}

/** 変換に必要なボーンのうち、モデルに無いもの */
export function missingBones(model: PmxObject): string[] {
    const names = new Set(model.bones.map((b) => b.name));
    const required = ["センター", "下半身", "上半身", "首", "頭",
        ...["左", "右"].flatMap((s) => ["肩", "腕", "ひじ", "手首", "足", "ひざ", "足首", "足ＩＫ"].map((n) => s + n))];
    return required.filter((n) => !names.has(n));
}
