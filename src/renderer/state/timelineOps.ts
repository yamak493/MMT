// タイムラインの編集の操作（状態 → 新しい状態。画面に依らないので単体で試せる）。
import type { ConversionSettings, FootKey, HandEvent, HandKey, MotionClip, MotionVariant, TimelineState } from "../../shared/types";

export function emptyTimeline(): TimelineState {
    return { clips: [], footKeys: {}, handKeys: { left: [], right: [] }, lipsyncEnabled: {}, conversions: {} };
}

let counter = 0;
export function newId(prefix: string): string {
    counter = (counter + 1) % 1e6;
    return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`;
}

function byStart(a: MotionClip, b: MotionClip): number {
    return a.start - b.start;
}

/** clips から [s, e] の範囲を空ける（重なるクリップを切り詰め・分ける・消す）。 */
export function carve(clips: MotionClip[], s: number, e: number): MotionClip[] {
    const out: MotionClip[] = [];
    for (const c of clips) {
        if (c.end < s || c.start > e) {
            out.push(c);
            continue;
        }
        if (c.start < s) out.push({ ...c, end: s - 1 });
        if (c.end > e) out.push({ ...c, id: c.start < s ? newId("clip") : c.id, start: e + 1 });
    }
    return out.sort(byStart);
}

/** 撮影の結果をタイムラインに入れる: 区間のクリップを作り、手のポーズをその区間の自動判定で置き換える。 */
export function addTake(
    state: TimelineState,
    takeId: string,
    start: number,
    end: number,
    variant: MotionVariant,
    handEvents: { left: HandEvent[]; right: HandEvent[] } | null
): TimelineState {
    const clips = carve(state.clips, start, end);
    clips.push({ id: newId("clip"), takeId, start, end, variant });
    clips.sort(byStart);
    const handKeys = { left: [...state.handKeys.left], right: [...state.handKeys.right] };
    if (handEvents) {
        for (const side of ["left", "right"] as const) {
            const old = handKeys[side];
            const kept = old.filter((k) => k.frame < start || k.frame > end);
            const added = handEvents[side].map((ev) => ({ frame: ev.frame + start, shape: ev.shape }))
                .filter((k) => k.frame >= start && k.frame <= end);
            // 区間の後ろは、区間の前から続いていた形に戻す（後ろのクリップの手の形を変えない）
            const shapeAfter = [...old].reverse().find((k) => k.frame <= end + 1)?.shape;
            if (shapeAfter && added.length && !kept.some((k) => k.frame === end + 1)) {
                kept.push({ frame: end + 1, shape: shapeAfter });
            }
            handKeys[side] = dedupeHandKeys([...kept, ...added]);
        }
    }
    return {
        ...state,
        clips,
        handKeys,
        lipsyncEnabled: { ...state.lipsyncEnabled, [takeId]: state.lipsyncEnabled[takeId] ?? true }
    };
}

export function dedupeHandKeys(keys: HandKey[]): HandKey[] {
    const map = new Map<number, string>();
    for (const k of keys) map.set(Math.max(0, Math.round(k.frame)), k.shape);
    return [...map.entries()].sort((a, b) => a[0] - b[0]).map(([frame, shape]) => ({ frame, shape }));
}

export function removeClip(state: TimelineState, clipId: string): TimelineState {
    return { ...state, clips: state.clips.filter((c) => c.id !== clipId) };
}

export function splitClip(state: TimelineState, clipId: string, frame: number): TimelineState {
    const clip = state.clips.find((c) => c.id === clipId);
    if (!clip || frame <= clip.start || frame > clip.end) return state;
    const left = { ...clip, end: frame - 1 };
    const right = { ...clip, id: newId("clip"), start: frame };
    return { ...state, clips: state.clips.flatMap((c) => (c.id === clipId ? [left, right] : [c])).sort(byStart) };
}

export function setClipVariant(state: TimelineState, clipId: string, variant: MotionVariant): TimelineState {
    return { ...state, clips: state.clips.map((c) => (c.id === clipId ? { ...c, variant } : c)) };
}

/** クリップの端を動かせる範囲（撮影の範囲と、隣のクリップの間） */
export function clipBounds(state: TimelineState, clip: MotionClip, takeStart: number, takeEnd: number): [number, number] {
    let lo = takeStart;
    let hi = takeEnd;
    for (const c of state.clips) {
        if (c.id === clip.id) continue;
        if (c.end < clip.start) lo = Math.max(lo, c.end + 1);
        if (c.start > clip.end) hi = Math.min(hi, c.start - 1);
    }
    return [lo, hi];
}

export function trimClip(state: TimelineState, clipId: string, start: number, end: number): TimelineState {
    if (end < start) return state;
    return { ...state, clips: state.clips.map((c) => (c.id === clipId ? { ...c, start, end } : c)).sort(byStart) };
}

// ---------------------------------------------------------------- 足ピタ（単発のキー）

function sortFootKeys(keys: FootKey[]): FootKey[] {
    return keys.sort((a, b) => a.frame - b.frame || a.foot - b.foot);
}

/**
 * 1 つの足のキーのフレーム → 足を固定する区間 [開始, 終了]（エンジンの foot_keys.hold_spans と同じ規則）。
 * 前から 2 つずつ組にし、余った最後のキーは 1 フレームの区間。続いた・重なった区間は 1 つにまとめる。
 */
export function footHoldSpans(frames: readonly number[]): [number, number][] {
    const sorted = [...frames].sort((a, b) => a - b);
    const spans: [number, number][] = [];
    for (let i = 0; i < sorted.length; i += 2) {
        const s = sorted[i];
        const e = sorted[Math.min(i + 1, sorted.length - 1)];
        const last = spans[spans.length - 1];
        if (last && s <= last[1] + 1) last[1] = Math.max(last[1], e);
        else spans.push([s, e]);
    }
    return spans;
}

/** 撮影 takeId の foot の frame（撮影の中のフレーム）に足ピタのキーを置く（同じ所にあれば何もしない） */
export function addFootKey(state: TimelineState, takeId: string, foot: 0 | 1, frame: number): TimelineState {
    const current = state.footKeys[takeId] ?? [];
    if (frame < 0 || current.some((k) => k.foot === foot && k.frame === frame)) return state;
    return setFootKeys(state, takeId, sortFootKeys([...current, { foot, frame }]));
}

export function removeFootKey(state: TimelineState, takeId: string, foot: 0 | 1, frame: number): TimelineState {
    const current = state.footKeys[takeId] ?? [];
    return setFootKeys(state, takeId, current.filter((k) => k.foot !== foot || k.frame !== frame));
}

/** キーを from から to へ動かす（to にキーがあれば、それと 1 つにまとめる） */
export function moveFootKey(state: TimelineState, takeId: string, foot: 0 | 1, from: number, to: number): TimelineState {
    if (from === to || to < 0) return state;
    const current = state.footKeys[takeId] ?? [];
    if (!current.some((k) => k.foot === foot && k.frame === from)) return state;
    const rest = current.filter((k) => k.foot !== foot || (k.frame !== from && k.frame !== to));
    return setFootKeys(state, takeId, sortFootKeys([...rest, { foot, frame: to }]));
}

export function setFootKeys(state: TimelineState, takeId: string, keys: FootKey[]): TimelineState {
    return { ...state, footKeys: { ...state.footKeys, [takeId]: keys } };
}

/**
 * 読み込んだ状態を今の形にそろえる。以前の版の足ピタ（接地の区間の上書き）は、接地にした区間の両端をキーにする
 * （外した区間は、自動の判定のまま残るので捨てる）。
 */
export function normalizeTimeline(state: TimelineState): TimelineState {
    if (state.footKeys && !state.footOverrides) return state;
    const footKeys: Record<string, FootKey[]> = { ...(state.footKeys ?? {}) };
    for (const [takeId, overrides] of Object.entries(state.footOverrides ?? {})) {
        const keys = [...(footKeys[takeId] ?? [])];
        for (const o of overrides) {
            if (o.mode !== "on") continue;
            for (const frame of new Set([o.start, o.end])) {
                if (!keys.some((k) => k.foot === o.foot && k.frame === frame)) keys.push({ foot: o.foot, frame });
            }
        }
        footKeys[takeId] = sortFootKeys(keys);
    }
    const { footOverrides: _legacy, ...rest } = state;
    return { ...rest, footKeys };
}

// ---------------------------------------------------------------- 手のポーズ

export function setHandKey(state: TimelineState, side: "left" | "right", frame: number, shape: string): TimelineState {
    const keys = dedupeHandKeys([...state.handKeys[side].filter((k) => k.frame !== frame), { frame, shape }]);
    return { ...state, handKeys: { ...state.handKeys, [side]: keys } };
}

export function moveHandKey(state: TimelineState, side: "left" | "right", from: number, to: number): TimelineState {
    const key = state.handKeys[side].find((k) => k.frame === from);
    if (!key || from === to) return state;
    const keys = dedupeHandKeys([...state.handKeys[side].filter((k) => k.frame !== from && k.frame !== to), { frame: to, shape: key.shape }]);
    return { ...state, handKeys: { ...state.handKeys, [side]: keys } };
}

export function removeHandKey(state: TimelineState, side: "left" | "right", frame: number): TimelineState {
    return { ...state, handKeys: { ...state.handKeys, [side]: state.handKeys[side].filter((k) => k.frame !== frame) } };
}

export function setConversion(state: TimelineState, takeId: string, settings: ConversionSettings): TimelineState {
    return { ...state, conversions: { ...(state.conversions ?? {}), [takeId]: settings } };
}

/** 撮影を使うクリップが無くなったら、その撮影の足ピタのキーも消す */
export function pruneTakes(state: TimelineState, takeIds: Set<string>): TimelineState {
    const footKeys: Record<string, FootKey[]> = {};
    for (const [k, v] of Object.entries(state.footKeys)) if (takeIds.has(k)) footKeys[k] = v;
    const conversions: Record<string, ConversionSettings> = {};
    for (const [k, v] of Object.entries(state.conversions ?? {})) if (takeIds.has(k)) conversions[k] = v;
    return { ...state, footKeys, conversions };
}
