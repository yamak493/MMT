// VMD（Vocaloid Motion Data 0002）の書き出し。
//
// ヘッダ 30 バイト＋モデル名 20 バイト（Shift_JIS）
// ボーンのキー: 個数 uint32 ＋ 1 キー 111 バイト（名前 15・フレーム uint32・位置 float32×3・回転 float32×4・補間 64）
// モーフのキー: 個数 uint32 ＋ 1 キー 23 バイト（名前 15・フレーム uint32・値 float32）
// カメラ・照明・セルフ影のキー: どれも 0 個
// 補間 64 バイトの並びは MMD Tools（blender_mmd_tools）の実装と同じ: 16 バイトの行
//   [X_x1, Y_x1, Z_x1, R_x1, X_y1, Y_y1, Z_y1, R_y1, X_x2, Y_x2, Z_x2, R_x2, X_y2, Y_y2, Z_y2, R_y2]
// を 1 行ごとに 1 つずつ左へずらしたもの（はみ出た末尾は 0）。1 行目の 3・4 バイト目は 0。
import Encoding from "encoding-japanese";
import { LINEAR, type ComposedMotion, type Curve } from "./keys";

const SIGNATURE = "Vocaloid Motion Data 0002";

/** 文字列を Shift_JIS（cp932）のバイト列にする。maxBytes を超えるときは文字の途中で切らないように切り詰める。 */
export function encodeSjis(text: string, maxBytes: number): Uint8Array {
    const codes = Encoding.convert(Encoding.stringToCode(text), { to: "SJIS", from: "UNICODE" }) as number[];
    if (codes.length <= maxBytes) return Uint8Array.from(codes);
    let n = 0;
    let i = 0;
    while (i < codes.length) {
        const lead = codes[i];
        const width = (lead >= 0x81 && lead <= 0x9f) || (lead >= 0xe0 && lead <= 0xfc) ? 2 : 1;
        if (n + width > maxBytes) break;
        n += width;
        i += width;
    }
    return Uint8Array.from(codes.slice(0, n));
}

export function sjisByteLength(text: string): number {
    return (Encoding.convert(Encoding.stringToCode(text), { to: "SJIS", from: "UNICODE" }) as number[]).length;
}

/** 補間 64 バイト（位置 X・Y・Z と回転を同じ曲線にする） */
export function interpolationBytes(curve: Curve = LINEAR): Uint8Array {
    const [x1, y1, x2, y2] = curve;
    const base = [x1, x1, x1, x1, y1, y1, y1, y1, x2, x2, x2, x2, y2, y2, y2, y2];
    const out = new Uint8Array(64);
    for (let row = 0; row < 4; row++) {
        for (let i = 0; i < 16; i++) {
            const src = i + row;
            out[row * 16 + i] = src < 16 ? base[src] : 0;
        }
    }
    out[2] = 0;
    out[3] = 0;
    return out;
}

export interface VmdWriteResult {
    buffer: ArrayBuffer;
    boneKeys: number;
    morphKeys: number;
    truncatedNames: string[];
}

export function writeVmd(motion: ComposedMotion, modelName: string): VmdWriteResult {
    const boneCount = motion.bones.reduce((s, b) => s + b.frames.length, 0);
    const morphCount = motion.morphs.reduce((s, m) => s + m.frames.length, 0);
    const size = 30 + 20 + 4 + boneCount * 111 + 4 + morphCount * 23 + 4 * 3;
    const buffer = new ArrayBuffer(size);
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const truncated: string[] = [];
    let o = 0;
    bytes.set(new TextEncoder().encode(SIGNATURE), 0);
    o = 30;
    bytes.set(encodeSjis(modelName, 20), o);
    o += 20;

    // ボーンのキー（フレームの順に並べる。MMD はどちらでも読めるが、他のツールとの互換のため）
    view.setUint32(o, boneCount, true);
    o += 4;
    const linear = interpolationBytes(LINEAR);
    const curveCache = new Map<Curve, Uint8Array>([[LINEAR, linear]]);
    const order: [number, number][] = [];
    motion.bones.forEach((b, bi) => b.frames.forEach((_f, ki) => order.push([bi, ki])));
    order.sort((a, b) => motion.bones[a[0]].frames[a[1]] - motion.bones[b[0]].frames[b[1]]);
    const names = motion.bones.map((b) => {
        if (sjisByteLength(b.name) > 15) truncated.push(b.name);
        return encodeSjis(b.name, 15);
    });
    for (const [bi, ki] of order) {
        const b = motion.bones[bi];
        bytes.set(names[bi], o);
        view.setUint32(o + 15, b.frames[ki], true);
        for (let k = 0; k < 3; k++) view.setFloat32(o + 19 + k * 4, b.movable ? b.positions[ki * 3 + k] : 0, true);
        for (let k = 0; k < 4; k++) view.setFloat32(o + 31 + k * 4, b.rotations[ki * 4 + k], true);
        const curve = b.curves ? b.curves[ki] : LINEAR;
        let interp = curveCache.get(curve);
        if (!interp) {
            interp = interpolationBytes(curve);
            curveCache.set(curve, interp);
        }
        bytes.set(interp, o + 47);
        o += 111;
    }

    view.setUint32(o, morphCount, true);
    o += 4;
    const morphOrder: [number, number][] = [];
    motion.morphs.forEach((m, mi) => m.frames.forEach((_f, ki) => morphOrder.push([mi, ki])));
    morphOrder.sort((a, b) => motion.morphs[a[0]].frames[a[1]] - motion.morphs[b[0]].frames[b[1]]);
    const morphNames = motion.morphs.map((m) => {
        if (sjisByteLength(m.name) > 15) truncated.push(m.name);
        return encodeSjis(m.name, 15);
    });
    for (const [mi, ki] of morphOrder) {
        const m = motion.morphs[mi];
        bytes.set(morphNames[mi], o);
        view.setUint32(o + 15, m.frames[ki], true);
        view.setFloat32(o + 19, m.weights[ki], true);
        o += 23;
    }
    // カメラ・照明・セルフ影
    view.setUint32(o, 0, true);
    view.setUint32(o + 4, 0, true);
    view.setUint32(o + 8, 0, true);
    return { buffer, boneKeys: boneCount, morphKeys: morphCount, truncatedNames: truncated };
}
