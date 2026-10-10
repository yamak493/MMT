// 変換・キャプチャの設定の既定値と、エンジンの設定（settings.py のキー）への変換。
import type { CaptureOptions, ConversionSettings } from "../../shared/types";

export function defaultConversion(): ConversionSettings {
    return {
        centerMode: "A",
        depthScale: 1,
        flipFacing: false,
        armCollision: "left",
        bodyParts: { clothes: true, head: true, torso: true, chest: true, legs: true },
        handReach: true,
        clothMarginCm: 1,
        wristFromHands: false,
        legKeys: true,
        groundEnabled: true,
        leanEnabled: true,
        lockedBothFeet: true,
        lipsyncMorphs: { a: "あ", i: "い", u: "う", e: "え", o: "お", n: "ん" },
        lipsyncStrength: 1,
        handSwitchCost: 3
    };
}

export function defaultCapture(start = 0, end = 0): CaptureOptions {
    return {
        start,
        end,
        fps: "auto",
        maxHeight: 720,
        // NLF の人物の切り出しは正方形なので、映像も人を中心に正方形へ切り抜くのを既定にする
        crop: null,
        cropMode: "square",
        personSelect: "largest",
        batchSize: 8,
        // video2vmd のノートブックの既定（NUM_AUG=3・ANTIALIAS=2）と同じ「標準」。推定のぶれを発生源で減らす
        numAug: 3,
        antialias: 2,
        detectorThreshold: 0.25,
        device: "auto",
        hands: true,
        lipsync: true,
        demucs: true,
        lipsyncLang: "jpn",
        conversion: defaultConversion()
    };
}

/** 画面の設定 → エンジンの設定の上書き（ドット区切りのキー） */
export function engineSettings(c: ConversionSettings): Record<string, unknown> {
    return {
        "center.mode": c.centerMode,
        "scale.depth_scale": c.depthScale,
        "floor.flip_facing": c.flipFacing,
        "arm_collision.mode": c.armCollision,
        "contacts.body_parts": { ...c.bodyParts },
        "contacts.cloth_margin_m": c.clothMarginCm / 100,
        "hand_reach.enabled": c.handReach,
        "wrist.enabled": c.wristFromHands,
        "leg_keys.enabled": c.legKeys,
        "ground.enabled": c.groundEnabled,
        "lean.enabled": c.leanEnabled,
        "locked.both_feet": c.lockedBothFeet,
        "lipsync.morphs": { ...c.lipsyncMorphs },
        "lipsync.strength": c.lipsyncStrength,
        "hands.switch_cost": c.handSwitchCost
    };
}

/** 精度のプリセット（TTA の枚数と、切り出しの超サンプリング） */
export const QUALITY_PRESETS = {
    fast: { label: "速度優先", numAug: 1, antialias: 1 },
    standard: { label: "標準", numAug: 3, antialias: 2 },
    high: { label: "高精度（時間がかかります）", numAug: 5, antialias: 2 }
} as const;

export type QualityPreset = keyof typeof QUALITY_PRESETS;

export function presetOf(o: CaptureOptions): QualityPreset | "custom" {
    for (const [k, v] of Object.entries(QUALITY_PRESETS)) {
        if (v.numAug === o.numAug && v.antialias === o.antialias) return k as QualityPreset;
    }
    return "custom";
}

/** 1 フレームあたりの推論時間の目安 [秒]（GPU の世代でかなり変わるので大まかな値） */
export function estimateSeconds(frames: number, o: CaptureOptions, gpu: boolean): number {
    const perFrame = (gpu ? 0.12 : 6) * o.numAug * (o.antialias > 1 ? 1.3 : 1) * Math.max(0.5, o.maxHeight / 720);
    let total = frames * perFrame;
    if (o.hands) total += frames * 0.03;
    if (o.lipsync) total += (frames / 30) * (gpu ? 0.4 : 2);
    total += frames * 0.08;   // 変換
    return total;
}
