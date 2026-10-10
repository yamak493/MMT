// タイムラインのクリップ・手のポーズ・口パクから、モデルに掛けるキー列（プレビューと VMD の出力で共通）を作る。
import type { HandKey, HandRotationTable, MotionClip, MotionVariant } from "../../shared/types";
import { fingerBoneKeys } from "./hands";
import { makeContinuous, type BoneKeys, type ComposedMotion, type MorphKeys } from "./keys";
import type { TakeResult, VariantData } from "./takeData";

export interface ComposeOptions {
    body: boolean;
    fingers: boolean;
    face: boolean;
}

export interface ComposeInput {
    clips: readonly MotionClip[];
    takes: ReadonlyMap<string, TakeResult>;
    handKeys: { left: readonly HandKey[]; right: readonly HandKey[] };
    handTable: Partial<HandRotationTable> | null;
    transitionFrames: number;
    lipsyncEnabled: Readonly<Record<string, boolean>>;
    options: ComposeOptions;
}

function variantOf(take: TakeResult, kind: MotionVariant): VariantData | null {
    return take.variants.get(kind) ?? take.variants.get("full") ?? null;
}

/** クリップの並び（重なりのない、開始フレームの順） */
export function sortedClips(clips: readonly MotionClip[]): MotionClip[] {
    return [...clips].sort((a, b) => a.start - b.start);
}

export function composeMotion(input: ComposeInput): ComposedMotion {
    const clips = sortedClips(input.clips).filter((c) => input.takes.has(c.takeId));
    const bones: BoneKeys[] = [];
    const morphs: MorphKeys[] = [];
    let lastFrame = clips.length ? clips[clips.length - 1].end : 0;

    if (input.options.body && clips.length) {
        // 使うボーン（どこかのクリップの種類にあるもの）と、位置のキーを持つか
        const movable = new Map<string, boolean>();
        for (const clip of clips) {
            const v = variantOf(input.takes.get(clip.takeId)!, clip.variant);
            v?.bones.forEach((b, name) => movable.set(name, (movable.get(name) ?? false) || b.movable));
        }
        // フレームの数を数えてから詰める
        const spans: { clip: MotionClip; take: TakeResult; v: VariantData; s: number; e: number }[] = [];
        let total = 0;
        let prevEnd = -1;
        for (const clip of clips) {
            const take = input.takes.get(clip.takeId)!;
            const v = variantOf(take, clip.variant);
            if (!v) continue;
            const s = Math.max(clip.start, take.take.startFrame, prevEnd + 1);
            const e = Math.min(clip.end, take.take.startFrame + v.frames - 1);
            if (e < s) continue;
            spans.push({ clip, take, v, s, e });
            total += e - s + 1;
            prevEnd = e;
        }
        for (const [name, isMovable] of movable) {
            const frames = new Uint32Array(total);
            const positions = new Float32Array(total * 3);
            const rotations = new Float32Array(total * 4);
            let k = 0;
            for (const { take, v, s, e } of spans) {
                const series = v.bones.get(name);
                for (let f = s; f <= e; f++, k++) {
                    const t = f - take.take.startFrame;
                    frames[k] = f;
                    if (series) {
                        if (series.positions) positions.set(series.positions.subarray(t * 3, t * 3 + 3), k * 3);
                        rotations.set(series.rotations.subarray(t * 4, t * 4 + 4), k * 4);
                    } else {
                        // その種類に無いボーン（上半身のみのクリップのセンター・脚など）は初期姿勢
                        rotations[k * 4 + 3] = 1;
                    }
                }
            }
            makeContinuous(rotations);
            bones.push({ name, movable: isMovable, frames, positions, rotations, curves: null });
        }
    }

    if (input.options.fingers && input.handTable) {
        const lastKey = Math.max(lastFrame, ...input.handKeys.left.map((k) => k.frame), ...input.handKeys.right.map((k) => k.frame));
        for (const side of ["left", "right"] as const) {
            bones.push(...fingerBoneKeys(input.handKeys[side], input.handTable[side], lastKey, input.transitionFrames));
        }
    }

    if (input.options.face) {
        const perMorph = new Map<string, Map<number, number>>();
        const covered: [number, number][] = [];
        for (const clip of clips) {
            const take = input.takes.get(clip.takeId)!;
            const lip = take.json.lipsync;
            if (!lip.available || !lip.tracks || input.lipsyncEnabled[clip.takeId] === false) continue;
            const start = Math.max(clip.start, take.take.startFrame);
            const end = Math.min(clip.end, take.take.startFrame + take.json.frames - 1);
            if (end < start) continue;
            covered.push([start, end]);
            for (const track of lip.tracks) {
                const map = perMorph.get(track.name) ?? new Map<number, number>();
                perMorph.set(track.name, map);
                const at = (f: number) => sampleLinear(track.frames, track.weights, f - take.take.startFrame);
                map.set(start, at(start));
                map.set(end, at(end));
                for (let i = 0; i < track.frames.length; i++) {
                    const f = track.frames[i] + take.take.startFrame;
                    if (f > start && f < end) map.set(f, track.weights[i]);
                }
            }
        }
        // 口パクのあるクリップの外は口を閉じる
        const inside = (f: number) => covered.some(([s, e]) => f >= s && f <= e);
        for (const [name, map] of perMorph) {
            for (const [s, e] of covered) {
                if (s - 1 >= 0 && !inside(s - 1) && !map.has(s - 1)) map.set(s - 1, 0);
                if (!inside(e + 1) && !map.has(e + 1)) map.set(e + 1, 0);
            }
            const frames = [...map.keys()].sort((a, b) => a - b);
            morphs.push({ name, frames: Uint32Array.from(frames), weights: Float32Array.from(frames.map((f) => map.get(f)!)) });
            lastFrame = Math.max(lastFrame, frames[frames.length - 1] ?? 0);
        }
    }
    for (const b of bones) {
        if (b.frames.length) lastFrame = Math.max(lastFrame, b.frames[b.frames.length - 1]);
    }
    return { bones, morphs, lastFrame };
}

/** キー（frames, weights）を線形補間した値。範囲の外は端の値。 */
export function sampleLinear(frames: readonly number[], weights: readonly number[], f: number): number {
    if (!frames.length) return 0;
    if (f <= frames[0]) return weights[0];
    if (f >= frames[frames.length - 1]) return weights[weights.length - 1];
    let lo = 0;
    let hi = frames.length - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (frames[mid] <= f) lo = mid;
        else hi = mid;
    }
    const t = (f - frames[lo]) / (frames[hi] - frames[lo]);
    return weights[lo] + (weights[hi] - weights[lo]) * t;
}

/** キーのフレームを offset だけずらす（VMD の出力で、最初のクリップを 0 フレームにするとき）。負になるキーは捨てる。 */
export function shiftMotion(motion: ComposedMotion, offset: number): ComposedMotion {
    if (offset === 0) return motion;
    const shiftKeys = <T extends { frames: Uint32Array }>(k: T, pick: (keep: number[]) => Partial<T>): T => {
        const keep: number[] = [];
        for (let i = 0; i < k.frames.length; i++) if (k.frames[i] >= offset) keep.push(i);
        return { ...k, frames: Uint32Array.from(keep.map((i) => k.frames[i] - offset)), ...pick(keep) };
    };
    const take = (arr: Float32Array, keep: number[], n: number) => {
        const out = new Float32Array(keep.length * n);
        keep.forEach((i, j) => out.set(arr.subarray(i * n, i * n + n), j * n));
        return out;
    };
    return {
        bones: motion.bones.map((b) => shiftKeys(b, (keep) => ({
            positions: take(b.positions, keep, 3), rotations: take(b.rotations, keep, 4),
            curves: b.curves ? keep.map((i) => b.curves![i]) : null
        }))),
        morphs: motion.morphs.map((m) => shiftKeys(m, (keep) => ({ weights: take(m.weights, keep, 1) }))),
        lastFrame: Math.max(0, motion.lastFrame - offset)
    };
}
