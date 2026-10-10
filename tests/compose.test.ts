import assert from "node:assert/strict";
import { test } from "node:test";
import type { ConversionResultJson, MotionClip, Take } from "../src/shared/types";
import { composeMotion, shiftMotion } from "../src/renderer/motion/compose";
import type { TakeResult } from "../src/renderer/motion/takeData";
import { thinMotion } from "../src/renderer/motion/thin";

function fakeTake(id: string, startFrame: number, frames: number, value: number): TakeResult {
    const pos = new Float32Array(frames * 3);
    const rot = new Float32Array(frames * 4);
    for (let t = 0; t < frames; t++) {
        pos[t * 3] = value + t;
        rot[t * 4 + 3] = 1;
    }
    const take = { id, dir: `takes/${id}`, start: startFrame / 30, end: (startFrame + frames) / 30, startFrame, frames } as Take;
    const json = {
        frames, lipsync: { available: true, tracks: [{ name: "あ", shape: "a", frames: [0, frames - 1], weights: [1, 1] }] }
    } as unknown as ConversionResultJson;
    return {
        take, json,
        variants: new Map([["full", {
            frames,
            bones: new Map([
                ["センター", { name: "センター", movable: true, positions: pos, rotations: rot }],
                ["上半身", { name: "上半身", movable: false, positions: null, rotations: rot }]
            ])
        }]])
    };
}

const clip = (id: string, takeId: string, start: number, end: number): MotionClip => ({ id, takeId, start, end, variant: "full" });

test("クリップの範囲だけ撮影のキーを使い、つなぐ", () => {
    const takes = new Map([["a", fakeTake("a", 0, 100, 0)], ["b", fakeTake("b", 50, 100, 1000)]]);
    const m = composeMotion({
        clips: [clip("c1", "a", 0, 59), clip("c2", "b", 60, 149)], takes, handKeys: { left: [], right: [] },
        handTable: null, transitionFrames: 5, lipsyncEnabled: {}, options: { body: true, fingers: true, face: true }
    });
    const center = m.bones.find((b) => b.name === "センター")!;
    assert.equal(center.frames.length, 150);
    assert.equal(center.positions[59 * 3], 59);
    assert.equal(center.positions[60 * 3], 1000 + 10);   // 撮影 b の 10 フレーム目
    assert.equal(m.lastFrame, 150);
    const morph = m.morphs[0];
    assert.equal(morph.weights[0], 1);
    assert.equal(morph.frames[morph.frames.length - 1], 150);   // クリップの後で口を閉じる
    assert.equal(morph.weights[morph.weights.length - 1], 0);
});

test("口パクを外した撮影はモーフのキーを作らない・0 フレームから始める・間引く", () => {
    const takes = new Map([["a", fakeTake("a", 30, 60, 0)]]);
    const m = composeMotion({
        clips: [clip("c1", "a", 30, 89)], takes, handKeys: { left: [], right: [] }, handTable: null,
        transitionFrames: 5, lipsyncEnabled: { a: false }, options: { body: true, fingers: false, face: true }
    });
    assert.equal(m.morphs.length, 0);
    const shifted = shiftMotion(m, 30);
    assert.equal(shifted.bones[0].frames[0], 0);
    const thin = thinMotion(shifted);
    const center = thin.bones.find((b) => b.name === "センター")!;
    assert.deepEqual([...center.frames], [0, 59]);   // 一定の速さで動くので両端だけ
    assert.equal(thin.bones.find((b) => b.name === "上半身")!.frames.length, 2);
});
