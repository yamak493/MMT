import assert from "node:assert/strict";
import { test } from "node:test";
import { EASE, LINEAR, type ComposedMotion } from "../src/renderer/motion/keys";
import { encodeSjis, interpolationBytes, sjisByteLength, writeVmd } from "../src/renderer/motion/vmd";

// nlf2vmd（デモ版）の LINEAR_INTERPOLATION・EASE_INTERPOLATION と同じバイト列
const LINEAR_REF = [20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0];
const EASE_REF = [40, 40, 0, 0, 0, 0, 0, 0, 88, 88, 88, 88, 127, 127, 127, 127, 40, 40, 40, 0, 0, 0, 0, 88, 88, 88, 88, 127, 127, 127, 127, 0, 40, 40, 0, 0, 0, 0, 88, 88, 88, 88, 127, 127, 127, 127, 0, 0, 40, 0, 0, 0, 0, 88, 88, 88, 88, 127, 127, 127, 127, 0, 0, 0];

test("補間のバイト列が MMD Tools と同じ並び", () => {
    assert.deepEqual([...interpolationBytes(LINEAR)], LINEAR_REF);
    assert.deepEqual([...interpolationBytes(EASE)], EASE_REF);
});

test("Shift_JIS の切り詰めは文字の途中で切らない", () => {
    assert.equal(sjisByteLength("センター"), 8);
    assert.equal(encodeSjis("左足ＩＫ", 15).length, 8);
    assert.equal(encodeSjis("あいうえおかきくけこ", 15).length, 14);
    assert.equal(encodeSjis("abc", 15).length, 3);
});

test("VMD を書いて読み戻す", () => {
    const motion: ComposedMotion = {
        bones: [
            {
                name: "センター", movable: true, frames: Uint32Array.from([0, 5]), positions: Float32Array.from([1, 2, 3, 4, 5, 6]),
                rotations: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1]), curves: null
            },
            {
                name: "左人指１", movable: false, frames: Uint32Array.from([3]), positions: new Float32Array(3),
                rotations: Float32Array.from([0.5, 0.5, 0.5, 0.5]), curves: [EASE]
            }
        ],
        morphs: [{ name: "あ", frames: Uint32Array.from([1, 2]), weights: Float32Array.from([0.25, 1]) }],
        lastFrame: 5
    };
    const { buffer, boneKeys, morphKeys } = writeVmd(motion, "テストモデル");
    assert.equal(boneKeys, 3);
    assert.equal(morphKeys, 2);
    const v = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    assert.equal(new TextDecoder().decode(bytes.subarray(0, 25)), "Vocaloid Motion Data 0002");
    assert.equal(v.getUint32(50, true), 3);
    // フレームの順に並ぶ: センター 0、左人指１ 3、センター 5
    const frameAt = (i: number) => v.getUint32(54 + i * 111 + 15, true);
    assert.deepEqual([frameAt(0), frameAt(1), frameAt(2)], [0, 3, 5]);
    assert.equal(v.getFloat32(54 + 2 * 111 + 19, true), 4);
    assert.deepEqual([...bytes.subarray(54 + 111 + 47, 54 + 111 + 47 + 64)], EASE_REF);
    const morphStart = 54 + 3 * 111;
    assert.equal(v.getUint32(morphStart, true), 2);
    assert.equal(v.getFloat32(morphStart + 4 + 23 + 19, true), 1);
    assert.equal(buffer.byteLength, morphStart + 4 + 2 * 23 + 12);
});
