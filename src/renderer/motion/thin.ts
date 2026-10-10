// キーの間引き（VMD の出力で選んだとき）。線形補間（回転は球面線形補間）したときの誤差が許容内のキーを省く。
import type { BoneKeys, ComposedMotion, MorphKeys } from "./keys";

function slerpInto(out: Float32Array, a: Float32Array, ai: number, b: Float32Array, bi: number, t: number): void {
    let bx = b[bi], by = b[bi + 1], bz = b[bi + 2], bw = b[bi + 3];
    let d = a[ai] * bx + a[ai + 1] * by + a[ai + 2] * bz + a[ai + 3] * bw;
    if (d < 0) {
        bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d;
    }
    let w0 = 1 - t;
    let w1 = t;
    if (d < 0.9995) {
        const th = Math.acos(Math.min(1, d));
        const s = Math.sin(th);
        w0 = Math.sin((1 - t) * th) / s;
        w1 = Math.sin(t * th) / s;
    }
    out[0] = w0 * a[ai] + w1 * bx;
    out[1] = w0 * a[ai + 1] + w1 * by;
    out[2] = w0 * a[ai + 2] + w1 * bz;
    out[3] = w0 * a[ai + 3] + w1 * bw;
    const n = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
    out[0] /= n; out[1] /= n; out[2] /= n; out[3] /= n;
}

/** 残すキーの印。err(a, b) は a〜b の内側のキーの誤差（許容値で割った値）の最大とその位置。 */
function rdp(n: number, forced: (i: number) => boolean, err: (a: number, b: number) => [number, number]): Uint8Array {
    const keep = new Uint8Array(n);
    if (n <= 2) {
        keep.fill(1);
        return keep;
    }
    keep[0] = keep[n - 1] = 1;
    for (let i = 1; i < n - 1; i++) if (forced(i)) keep[i] = 1;
    const anchors: number[] = [];
    for (let i = 0; i < n; i++) if (keep[i]) anchors.push(i);
    const stack: [number, number][] = [];
    for (let i = 0; i + 1 < anchors.length; i++) stack.push([anchors[i], anchors[i + 1]]);
    while (stack.length) {
        const [a, b] = stack.pop()!;
        if (b - a < 2) continue;
        const [e, m] = err(a, b);
        if (e > 1) {
            keep[m] = 1;
            stack.push([a, m], [m, b]);
        }
    }
    return keep;
}

function thinBone(b: BoneKeys, posTol: number, rotTolRad: number): BoneKeys {
    const n = b.frames.length;
    const q = new Float32Array(4);
    // フレームが連続していない所（クリップの間）は両側のキーを残す
    const forced = (i: number) => b.frames[i] - b.frames[i - 1] > 1 || b.frames[i + 1] - b.frames[i] > 1 || (b.curves !== null);
    const keep = rdp(n, forced, (a, c) => {
        let worst = 0;
        let at = a + 1;
        for (let i = a + 1; i < c; i++) {
            const t = (b.frames[i] - b.frames[a]) / (b.frames[c] - b.frames[a]);
            let e = 0;
            if (b.movable) {
                for (let k = 0; k < 3; k++) {
                    const p = b.positions[a * 3 + k] + (b.positions[c * 3 + k] - b.positions[a * 3 + k]) * t;
                    e = Math.max(e, Math.abs(p - b.positions[i * 3 + k]) / posTol);
                }
            }
            slerpInto(q, b.rotations, a * 4, b.rotations, c * 4, t);
            const d = Math.abs(q[0] * b.rotations[i * 4] + q[1] * b.rotations[i * 4 + 1] + q[2] * b.rotations[i * 4 + 2] + q[3] * b.rotations[i * 4 + 3]);
            e = Math.max(e, (2 * Math.acos(Math.min(1, d))) / rotTolRad);
            if (e > worst) {
                worst = e;
                at = i;
            }
        }
        return [worst, at];
    });
    return pick(b, keep);
}

function pick(b: BoneKeys, keep: Uint8Array): BoneKeys {
    const idx: number[] = [];
    keep.forEach((v, i) => v && idx.push(i));
    const frames = Uint32Array.from(idx.map((i) => b.frames[i]));
    const positions = new Float32Array(idx.length * 3);
    const rotations = new Float32Array(idx.length * 4);
    idx.forEach((i, j) => {
        positions.set(b.positions.subarray(i * 3, i * 3 + 3), j * 3);
        rotations.set(b.rotations.subarray(i * 4, i * 4 + 4), j * 4);
    });
    return { ...b, frames, positions, rotations, curves: b.curves ? idx.map((i) => b.curves![i]) : null };
}

function thinMorph(m: MorphKeys, tol: number): MorphKeys {
    const keep = rdp(m.frames.length, (i) => m.frames[i] - m.frames[i - 1] > 1 || m.frames[i + 1] - m.frames[i] > 1, (a, c) => {
        let worst = 0;
        let at = a + 1;
        for (let i = a + 1; i < c; i++) {
            const t = (m.frames[i] - m.frames[a]) / (m.frames[c] - m.frames[a]);
            const e = Math.abs(m.weights[a] + (m.weights[c] - m.weights[a]) * t - m.weights[i]) / tol;
            if (e > worst) {
                worst = e;
                at = i;
            }
        }
        return [worst, at];
    });
    const idx: number[] = [];
    keep.forEach((v, i) => v && idx.push(i));
    return { ...m, frames: Uint32Array.from(idx.map((i) => m.frames[i])), weights: Float32Array.from(idx.map((i) => m.weights[i])) };
}

/** posTol: 位置の許容誤差 [MMD 単位]、rotTolDeg: 回転の許容誤差 [度] */
export function thinMotion(motion: ComposedMotion, posTol = 0.03, rotTolDeg = 0.5, morphTol = 0.01): ComposedMotion {
    const rot = (rotTolDeg * Math.PI) / 180;
    return {
        bones: motion.bones.map((b) => (b.curves ? b : thinBone(b, posTol, rot))),
        morphs: motion.morphs.map((m) => thinMorph(m, morphTol)),
        lastFrame: motion.lastFrame
    };
}
