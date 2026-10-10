// 手のポーズ（形が変わる所のキー）から、指ボーンのキーを作る。
//
// 形の切り替えは、変わるフレームを中心に transition かけて行う（前後の形が続く長さの半分まで）。切り替えの始まりに
// 前の形、終わりに次の形のキーを打ち、終わりのキーの補間をイーズイン・アウトにする（エンジンの hands.py と同じ）。
import type { HandKey, HandRotationTable } from "../../shared/types";
import { EASE, LINEAR, makeContinuous, type BoneKeys, type Curve } from "./keys";

export const HAND_SHAPES = ["default", "thumb", "index", "middle", "ring", "pinky", "fist", "scissors", "open", "pistol", "fox", "ok"] as const;
export type HandShape = (typeof HAND_SHAPES)[number];

export const HAND_SHAPE_LABELS: Record<string, string> = {
    default: "デフォルト",
    thumb: "親指",
    index: "人差し指",
    middle: "中指",
    ring: "薬指",
    pinky: "小指",
    fist: "グー",
    scissors: "チョキ",
    open: "パー",
    pistol: "ピストル",
    fox: "きつね",
    ok: "オッケー"
};

export const HAND_SHAPE_COLORS: Record<string, string> = {
    default: "#9aa3ad",
    thumb: "#f0a040",
    index: "#e05a5a",
    middle: "#a070d8",
    ring: "#e080b8",
    pinky: "#b08a60",
    fist: "#4a8fe0",
    scissors: "#4cb870",
    open: "#e6c84a",
    pistol: "#3cb4c8",
    fox: "#d0703a",
    ok: "#b8d040"
};

export interface ShapeKey {
    frame: number;
    shape: string;
    ease: boolean;
}

/** 形が変わる所のキー（タイムラインのフレーム）から、指のキーを打つフレームと形の並び。 */
export function transitionKeys(handKeys: readonly HandKey[], lastFrame: number, transitionFrames: number): ShapeKey[] {
    const keys = [...handKeys].sort((a, b) => a.frame - b.frame);
    // 同じ形が続くキーはまとめる
    const runs: { s: number; e: number; shape: string }[] = [];
    for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        if (runs.length && runs[runs.length - 1].shape === k.shape) continue;
        runs.push({ s: k.frame, e: 0, shape: k.shape });
    }
    for (let i = 0; i < runs.length; i++) {
        runs[i].e = i + 1 < runs.length ? runs[i + 1].s - 1 : Math.max(lastFrame, runs[i].s);
    }
    if (!runs.length) return [];
    const n = Math.max(1, Math.round(transitionFrames));
    const before = Math.floor(n / 2);
    const after = n - before;
    const out: ShapeKey[] = [{ frame: runs[0].s, shape: runs[0].shape, ease: false }];
    for (let i = 1; i < runs.length; i++) {
        const a = runs[i - 1];
        const b = runs[i];
        const aKey = Math.max(b.s - before, Math.floor((a.s + a.e + 1) / 2));
        let bKey = Math.min(b.s + after, Math.floor((b.s + b.e + 1) / 2));
        bKey = Math.max(bKey, aKey + 1);
        if (aKey > out[out.length - 1].frame) out.push({ frame: aKey, shape: a.shape, ease: false });
        out.push({ frame: bKey, shape: b.shape, ease: true });
    }
    return out;
}

/** 片手の指ボーンのキー。table: その手の 形 → { ボーン名: 回転 }。 */
export function fingerBoneKeys(
    handKeys: readonly HandKey[],
    table: Record<string, Record<string, number[]>> | undefined,
    lastFrame: number,
    transitionFrames: number
): BoneKeys[] {
    if (!table || !handKeys.length) return [];
    const shapeKeys = transitionKeys(handKeys, lastFrame, transitionFrames);
    const bones = Object.keys(table.default ?? Object.values(table)[0] ?? {});
    const out: BoneKeys[] = [];
    for (const bone of bones) {
        const n = shapeKeys.length;
        const frames = new Uint32Array(n);
        const rotations = new Float32Array(n * 4);
        const curves: Curve[] = [];
        shapeKeys.forEach((k, i) => {
            frames[i] = k.frame;
            const q = table[k.shape]?.[bone] ?? table.default?.[bone] ?? [0, 0, 0, 1];
            rotations.set(q, i * 4);
            curves.push(k.ease ? EASE : LINEAR);
        });
        makeContinuous(rotations);
        out.push({ name: bone, movable: false, frames, positions: new Float32Array(n * 3), rotations, curves });
    }
    return out;
}

/** フレーム frame での形（そのフレーム以前の最後のキー）。 */
export function shapeAt(handKeys: readonly HandKey[], frame: number): string | null {
    let best: HandKey | null = null;
    for (const k of handKeys) {
        if (k.frame <= frame && (!best || k.frame >= best.frame)) best = k;
    }
    return best ? best.shape : null;
}

export type { HandRotationTable };
