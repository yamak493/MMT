import assert from "node:assert/strict";
import { test } from "node:test";
import type { MotionClip } from "../src/shared/types";
import { transitionKeys } from "../src/renderer/motion/hands";
import { History } from "../src/renderer/state/history";
import {
    addFootKey, addTake, carve, emptyTimeline, footHoldSpans, moveFootKey, moveHandKey, normalizeTimeline, removeFootKey, setHandKey, splitClip, trimClip
} from "../src/renderer/state/timelineOps";

const clip = (id: string, start: number, end: number): MotionClip => ({ id, takeId: "t", start, end, variant: "full" });

test("区間を空けるとクリップを切り詰め・分ける", () => {
    const out = carve([clip("a", 0, 99)], 40, 59);
    assert.deepEqual(out.map((c) => [c.start, c.end]), [[0, 39], [60, 99]]);
    assert.equal(carve([clip("a", 10, 20)], 0, 30).length, 0);
});

test("撮影を入れると手のポーズは区間の後ろで元の形に戻る", () => {
    let s = emptyTimeline();
    s = setHandKey(s, "left", 0, "open");
    s = addTake(s, "t1", 100, 199, "full", { left: [{ frame: 0, shape: "fist" }, { frame: 50, shape: "default" }], right: [] });
    assert.deepEqual(s.handKeys.left, [
        { frame: 0, shape: "open" }, { frame: 100, shape: "fist" }, { frame: 150, shape: "default" }, { frame: 200, shape: "open" }
    ]);
    assert.equal(s.clips.length, 1);
    assert.equal(s.lipsyncEnabled.t1, true);
});

test("分割・範囲の変更", () => {
    let s = { ...emptyTimeline(), clips: [clip("a", 0, 99)] };
    s = splitClip(s, "a", 50);
    assert.deepEqual(s.clips.map((c) => [c.start, c.end]), [[0, 49], [50, 99]]);
    s = trimClip(s, s.clips[1].id, 60, 90);
    assert.deepEqual(s.clips.map((c) => [c.start, c.end]), [[0, 49], [60, 90]]);
});

test("足ピタのキーの追加・移動・削除", () => {
    let s = addFootKey(emptyTimeline(), "t", 0, 20);
    s = addFootKey(s, "t", 1, 10);
    s = addFootKey(s, "t", 0, 20);
    assert.deepEqual(s.footKeys.t, [{ foot: 1, frame: 10 }, { foot: 0, frame: 20 }]);
    s = moveFootKey(s, "t", 0, 20, 10);
    assert.deepEqual(s.footKeys.t, [{ foot: 0, frame: 10 }, { foot: 1, frame: 10 }]);
    s = removeFootKey(s, "t", 1, 10);
    assert.deepEqual(s.footKeys.t, [{ foot: 0, frame: 10 }]);
});

test("足ピタのキーは 2 つずつ組にして固定の区間にする", () => {
    assert.deepEqual(footHoldSpans([40, 10, 20]), [[10, 20], [40, 40]]);
    assert.deepEqual(footHoldSpans([10, 20, 21, 30]), [[10, 30]]);
    assert.deepEqual(footHoldSpans([]), []);
});

test("以前の版の足ピタの区間は両端のキーに直す", () => {
    const old = { ...emptyTimeline(), footKeys: undefined as never, footOverrides: {
        t: [{ foot: 0 as const, start: 10, end: 14, mode: "on" as const }, { foot: 1 as const, start: 30, end: 30, mode: "on" as const },
            { foot: 0 as const, start: 40, end: 50, mode: "off" as const }]
    } };
    const s = normalizeTimeline(old);
    assert.deepEqual(s.footKeys.t, [{ foot: 0, frame: 10 }, { foot: 0, frame: 14 }, { foot: 1, frame: 30 }]);
    assert.equal("footOverrides" in s, false);
});

test("手のポーズの移動は移動先のキーを置き換える", () => {
    let s = setHandKey(setHandKey(emptyTimeline(), "right", 10, "fist"), "right", 20, "open");
    s = moveHandKey(s, "right", 10, 20);
    assert.deepEqual(s.handKeys.right, [{ frame: 20, shape: "fist" }]);
});

test("元に戻す・やり直すは何回でもできる", () => {
    const h = new History(emptyTimeline());
    for (let i = 0; i < 50; i++) h.push(setHandKey(h.current, "left", i, "fist"), `追加 ${i}`);
    assert.equal(h.current.handKeys.left.length, 50);
    while (h.canUndo) h.undo();
    assert.equal(h.current.handKeys.left.length, 0);
    while (h.canRedo) h.redo();
    assert.equal(h.current.handKeys.left.length, 50);
    assert.equal(h.push(h.current, "同じ"), false);
    h.undo();
    h.push(setHandKey(h.current, "right", 1, "open"), "別");
    assert.equal(h.canRedo, false);
});

test("手の形の切り替えのキー", () => {
    const keys = transitionKeys([{ frame: 0, shape: "default" }, { frame: 30, shape: "fist" }], 100, 5);
    assert.deepEqual(keys, [
        { frame: 0, shape: "default", ease: false }, { frame: 28, shape: "default", ease: false }, { frame: 33, shape: "fist", ease: true }
    ]);
});
