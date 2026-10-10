// 撮影（take）の変換の結果（エンジンが書いた result.json とキー列のバイナリ）を読む。
import type { ConversionResultJson, MotionVariant, Take } from "../../shared/types";

export interface BoneSeries {
    name: string;
    movable: boolean;
    /** (frames × 3) MMD 座標の位置。位置のキーを持たないボーンは null */
    positions: Float32Array | null;
    /** (frames × 4) MMD 座標の回転（x, y, z, w） */
    rotations: Float32Array;
}

export interface VariantData {
    frames: number;
    bones: Map<string, BoneSeries>;
}

export interface TakeResult {
    take: Take;
    json: ConversionResultJson;
    variants: Map<MotionVariant, VariantData>;
}

export function resultDir(workDir: string, take: Take, key: string): string {
    return `${workDir}/${take.dir}/conv/${key}`;
}

export async function loadTakeResult(workDir: string, take: Take, key: string, resultPath?: string): Promise<TakeResult> {
    const dir = resultPath ? resultPath.replace(/[\\/]result\.json$/, "") : resultDir(workDir, take, key);
    const json = JSON.parse(await window.mmt.readTextFile(`${dir}/result.json`)) as ConversionResultJson;
    const variants = new Map<MotionVariant, VariantData>();
    for (const [kind, layout] of Object.entries(json.variants) as [MotionVariant, NonNullable<ConversionResultJson["variants"][MotionVariant]>][]) {
        const buf = await window.mmt.readFile(`${dir}/${layout.file}`);
        const data = new Float32Array(buf);
        const T = layout.frames;
        const bones = new Map<string, BoneSeries>();
        for (const b of layout.bones) {
            let o = b.offset;
            let positions: Float32Array | null = null;
            if (b.movable) {
                positions = data.subarray(o, o + 3 * T);
                o += 3 * T;
            }
            bones.set(b.name, { name: b.name, movable: b.movable, positions, rotations: data.subarray(o, o + 4 * T) });
        }
        variants.set(kind, { frames: T, bones });
    }
    return { take, json, variants };
}
