// タイムライン（キャンバスで描く）。楽曲波形・モーション（クリップ）・手のポーズ（左右で 2 行）・足ピタ（接地。左右で 2 行）の行。
// 行の高さは、パネルの高さに合わせて伸ばす。
//
// 編集の操作はどれも timelineOps の関数で新しい状態を作り、host.commit で履歴に積む（元に戻す・やり直すの対象）。
import type { FootKey, MotionClip, MotionVariant, Take, TimelineState } from "../../shared/types";
import { MMD_FPS, VARIANT_LABELS } from "../../shared/types";
import { HAND_SHAPE_COLORS, HAND_SHAPE_LABELS, HAND_SHAPES } from "../motion/hands";
import type { TakeResult } from "../motion/takeData";
import {
    addFootKey, clipBounds, footHoldSpans, moveFootKey, moveHandKey, removeClip, removeFootKey, removeHandKey, setClipVariant, setFootKeys,
    setHandKey, splitClip, trimClip
} from "../state/timelineOps";
import { $, formatTime, h } from "../ui/dom";
import { openMenu, type MenuEntry } from "../ui/menu";

export interface TakeView {
    take: Take;
    result: TakeResult | null;
    busy: boolean;
}

export interface TimelineHost {
    readonly timeline: TimelineState;
    readonly durationFrames: number;
    readonly currentFrame: number;
    readonly playing: boolean;
    takeView(takeId: string): TakeView | undefined;
    waveform(): { peaks: Float32Array; rate: number } | null;
    seek(frame: number): void;
    commit(next: TimelineState, label: string): void;
    reconvertTake(takeId: string): void;
    captureAgain(take: Take): void;
}

type Row = "ruler" | "wave" | "motion" | "foot" | "hand";

interface RowLayout {
    row: Row;
    label: string;
    sub?: string;
    /** 足ピタの行の足（0: 左, 1: 右） */
    foot?: 0 | 1;
    /** 手のポーズの行の手 */
    side?: "left" | "right";
    y: number;
    h: number;
}

interface RowSpec {
    row: Row;
    label: string;
    sub?: string;
    foot?: 0 | 1;
    side?: "left" | "right";
    /** 最小の高さ（パネルが高ければ、伸ばす） */
    h: number;
    /** 伸ばすときの重み（0 なら伸ばさない） */
    grow: number;
}

interface FootKeyView {
    takeId: string;
    foot: 0 | 1;
    /** タイムラインのフレーム */
    frame: number;
    /** 撮影の中のフレーム */
    takeFrame: number;
    clip: MotionClip;
}

const ROWS: RowSpec[] = [
    { row: "ruler", label: "", h: 22, grow: 0 },
    { row: "wave", label: "楽曲波形", h: 44, grow: 1.4 },
    { row: "motion", label: "モーション", h: 38, grow: 1 },
    { row: "hand", label: "手のポーズ", sub: "左手", side: "left", h: 26, grow: 1 },
    { row: "hand", label: "手のポーズ", sub: "右手", side: "right", h: 26, grow: 1 },
    { row: "foot", label: "足ピタ（試験的）", sub: "左足", foot: 0, h: 26, grow: 1 },
    { row: "foot", label: "足ピタ（試験的）", sub: "右足", foot: 1, h: 26, grow: 1 }
];
/** 行を伸ばすのは、最小の高さのこの倍まで */
const MAX_GROW = 2.2;

const COLORS = {
    bg: "#1b1d21",
    rowAlt: "#1f2226",
    grid: "#2a2e34",
    gridFaint: "#24272c",
    gridStrong: "#363b43",
    ruler: "#2b2f36",
    rulerText: "#9aa3ad",
    playhead: "#f5d76e",
    wave: "#4f7fb8",
    waveCovered: "#6aa0e0",
    clip: { full: "#3e6fa8", locked: "#3e8f6a", no_move: "#8a6a3a", upper_body: "#7a5aa0" } as Record<MotionVariant, string>,
    clipText: "#eef3f8",
    diamond: "#e8edf3",
    foot: ["#d08a4a", "#4ab0c0"],
    footAuto: "#5a6068",
    jump: "rgba(240, 200, 80, 0.25)",
    select: "#f5d76e"
};

/** 足ピタのキーの前後で元の動きになじませるフレーム数（エンジンの設定 foot_ik.blend_frames） */
const FOOT_BLEND_FRAMES = 5;

const MIN_PPF = 0.05;
const MAX_PPF = 24;

export class Timeline {
    private canvas = $<HTMLCanvasElement>("timeline-canvas");
    private wrap = $("timeline-canvas-wrap");
    private labels = $("timeline-labels");
    private scrollEl = $("timeline-scroll");
    private scrollInner = $("timeline-scroll-inner");
    private ctx = this.canvas.getContext("2d")!;
    private layout: RowLayout[] = [];
    private layoutHeight = -1;
    private ppf = 2;           // 1 フレームあたりのピクセル
    private scrollFrame = 0;   // 左端のフレーム
    private width = 0;
    private height = 0;
    private dirty = true;
    private selection: { kind: "clip"; id: string } | { kind: "hand"; side: "left" | "right"; frame: number }
        | { kind: "foot"; takeId: string; foot: 0 | 1; frame: number } | null = null;
    private drag: null | {
        kind: "seek" | "trim" | "foot-move" | "hand-move";
        startX: number;
        moved: boolean;
        data: Record<string, unknown>;
    } = null;
    /** 足ピタのフレームは撮影の中のフレーム */
    private preview: { kind: "foot"; takeId: string; foot: 0 | 1; from: number; to: number }
        | { kind: "clip"; id: string; start: number; end: number }
        | { kind: "hand"; side: "left" | "right"; from: number; to: number } | null = null;
    private tooltip: HTMLDivElement | null = null;
    onZoomChange: ((ppf: number) => void) | null = null;

    constructor(private readonly host: TimelineHost) {
        this.buildLayout(0);
        new ResizeObserver(() => this.resize()).observe(this.wrap);
        this.scrollEl.addEventListener("scroll", () => {
            this.scrollFrame = this.scrollEl.scrollLeft / this.ppf;
            this.invalidate();
        });
        this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
        window.addEventListener("mousemove", (e) => this.onMouseMove(e));
        window.addEventListener("mouseup", (e) => this.onMouseUp(e));
        this.canvas.addEventListener("contextmenu", (e) => this.onContextMenu(e));
        this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
        this.canvas.addEventListener("mouseleave", () => this.hideTooltip());
        const loop = () => {
            if (this.dirty || this.host.playing) this.draw();
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    // ---------------------------------------------------------------- 配置・座標

    /** 行の配置を決める。height はキャンバスの高さで、余った分を行に配る。 */
    private buildLayout(height: number): void {
        const base = ROWS.reduce((sum, r) => sum + r.h, 0);
        const weight = ROWS.reduce((sum, r) => sum + r.h * r.grow, 0);
        const scale = Math.min(MAX_GROW - 1, Math.max(0, height - base) / weight);
        let y = 0;
        this.layout = ROWS.map((spec) => {
            const hh = Math.floor(spec.h * (1 + spec.grow * scale));
            const r: RowLayout = { row: spec.row, label: spec.label, sub: spec.sub, foot: spec.foot, side: spec.side, y, h: hh };
            y += hh;
            return r;
        });
        this.layoutHeight = height;
        this.labels.innerHTML = "";
        this.layout.forEach((r, i) => {
            const style = { top: `${r.y}px`, height: `${r.h}px` };
            if (r.row === "ruler") {
                this.labels.append(h("div", { class: "timeline-label", style }, h("span", { class: "sub" }, "フレーム")));
                return;
            }
            // 左右の 2 行は、行の名前を上の行にだけ書く
            const first = this.layout[i - 1]?.row !== r.row;
            const pairedAbove = this.layout[i + 1]?.row === r.row;
            const color = r.foot !== undefined ? COLORS.foot[r.foot] : null;
            this.labels.append(h("div", { class: `timeline-label${pairedAbove ? " pair-top" : ""}`, style },
                h("span", null, first ? r.label : ""),
                r.sub ? h("span", { class: "sub", style: color ? { color } : {} }, r.sub) : null));
        });
    }

    private rowAt(y: number): RowLayout | null {
        return this.layout.find((r) => y >= r.y && y < r.y + r.h) ?? null;
    }

    private rowOf(row: Row): RowLayout {
        return this.layout.find((r) => r.row === row)!;
    }

    /** 足ピタ・手のポーズの行の範囲（左右の 2 行を合わせたもの） */
    private rowSpan(row: "foot" | "hand"): { y: number; h: number } {
        const rows = this.layout.filter((r) => r.row === row);
        const y = rows[0].y;
        const last = rows[rows.length - 1];
        return { y, h: last.y + last.h - y };
    }

    private x(frame: number): number {
        return (frame - this.scrollFrame) * this.ppf;
    }

    private frameAt(x: number): number {
        return Math.max(0, Math.round(x / this.ppf + this.scrollFrame));
    }

    private resize(): void {
        const r = this.wrap.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.width = r.width;
        this.height = r.height;
        this.canvas.width = Math.max(1, Math.round(r.width * dpr));
        this.canvas.height = Math.max(1, Math.round(r.height * dpr));
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (Math.round(r.height) !== this.layoutHeight) this.buildLayout(Math.round(r.height));
        this.updateScroll();
        this.invalidate();
    }

    private updateScroll(): void {
        const total = (this.host.durationFrames + 60) * this.ppf;
        this.scrollInner.style.width = `${Math.max(total, this.width)}px`;
        const maxFrame = Math.max(0, (total - this.width) / this.ppf);
        if (this.scrollFrame > maxFrame) this.scrollFrame = maxFrame;
        this.scrollEl.scrollLeft = this.scrollFrame * this.ppf;
    }

    invalidate(): void {
        this.dirty = true;
    }

    /** 長さ・状態が変わったとき */
    refresh(): void {
        this.updateScroll();
        this.invalidate();
    }

    get pixelsPerFrame(): number {
        return this.ppf;
    }

    setZoom(ppf: number, anchorX = this.width / 2): void {
        const anchorFrame = anchorX / this.ppf + this.scrollFrame;
        this.ppf = Math.max(MIN_PPF, Math.min(MAX_PPF, ppf));
        this.scrollFrame = Math.max(0, anchorFrame - anchorX / this.ppf);
        this.updateScroll();
        this.invalidate();
        this.onZoomChange?.(this.ppf);
    }

    zoomToFit(): void {
        const frames = Math.max(this.host.durationFrames, 30);
        this.ppf = Math.max(MIN_PPF, Math.min(MAX_PPF, (this.width - 20) / frames));
        this.scrollFrame = 0;
        this.updateScroll();
        this.invalidate();
        this.onZoomChange?.(this.ppf);
    }

    /** 再生位置が見えるようにスクロールする */
    follow(frame: number): void {
        const x = this.x(frame);
        if (x < 0 || x > this.width - 30) {
            this.scrollFrame = Math.max(0, frame - (this.width * 0.15) / this.ppf);
            this.updateScroll();
        }
        this.invalidate();
    }

    // ---------------------------------------------------------------- 撮影・接地の参照

    private clipAt(frame: number): MotionClip | null {
        return this.host.timeline.clips.find((c) => frame >= c.start && frame <= c.end) ?? null;
    }

    /** 自動で判定した接地区間（タイムラインのフレーム。表示だけ）。クリップの範囲で切る。 */
    private autoContactSegments(): { foot: 0 | 1; start: number; end: number }[] {
        const out: { foot: 0 | 1; start: number; end: number }[] = [];
        for (const clip of this.host.timeline.clips) {
            const view = this.host.takeView(clip.takeId);
            if (!view?.result) continue;
            const base = view.take.startFrame;
            const auto = view.result.json.contacts.auto;
            for (const foot of [0, 1] as const) {
                for (const [s, e] of auto[foot] ?? []) {
                    const a = Math.max(s + base, clip.start);
                    const b = Math.min(e + base, clip.end);
                    if (b >= a) out.push({ foot, start: a, end: b });
                }
            }
        }
        return out;
    }

    /** 表示する足ピタのキー（クリップの中にあるもの）。frame はタイムライン、takeFrame は撮影の中のフレーム */
    /** 撮影の足ピタのキー。withPreview なら、ドラッグで動かしている途中のキーを動かした先で返す */
    private takeFootKeys(takeId: string, withPreview: boolean): FootKey[] {
        const keys = this.host.timeline.footKeys[takeId] ?? [];
        const p = this.preview;
        if (!withPreview || p?.kind !== "foot" || p.takeId !== takeId || p.from === p.to) return keys;
        return keys.filter((k) => k.foot !== p.foot || (k.frame !== p.from && k.frame !== p.to))
            .concat({ foot: p.foot, frame: p.to });
    }

    private footKeyViews(withPreview = false): FootKeyView[] {
        const out: FootKeyView[] = [];
        for (const clip of this.host.timeline.clips) {
            const view = this.host.takeView(clip.takeId);
            if (!view) continue;
            const base = view.take.startFrame;
            for (const k of this.takeFootKeys(clip.takeId, withPreview)) {
                const frame = k.frame + base;
                if (frame >= clip.start && frame <= clip.end) out.push({ takeId: clip.takeId, foot: k.foot, frame, takeFrame: k.frame, clip });
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- 描画

    private draw(): void {
        this.dirty = false;
        const ctx = this.ctx;
        const W = this.width;
        const H = this.height;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, W, H);
        // 行の背景は種類ごとに交互に。左右の 2 行の間は薄い線
        let group = 0;
        this.layout.forEach((r, i) => {
            if (r.row === "ruler") return;
            if (this.layout[i - 1]?.row !== r.row) group++;
            ctx.fillStyle = group % 2 ? COLORS.bg : COLORS.rowAlt;
            ctx.fillRect(0, r.y, W, r.h);
            ctx.fillStyle = this.layout[i + 1]?.row === r.row ? COLORS.gridFaint : COLORS.grid;
            ctx.fillRect(0, r.y + r.h - 1, W, 1);
        });
        const duration = this.host.durationFrames;
        const endX = this.x(duration);
        if (endX < W) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
            ctx.fillRect(Math.max(0, endX), this.rowOf("wave").y, W - endX, H);
        }
        this.drawRuler();
        this.drawWave();
        this.drawClips();
        this.drawFoot();
        this.drawHands();
        // 再生位置
        const px = Math.round(this.x(this.host.currentFrame)) + 0.5;
        if (px >= 0 && px <= W) {
            ctx.strokeStyle = COLORS.playhead;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, H);
            ctx.stroke();
            ctx.fillStyle = COLORS.playhead;
            ctx.beginPath();
            ctx.moveTo(px - 5, 0);
            ctx.lineTo(px + 5, 0);
            ctx.lineTo(px, 7);
            ctx.fill();
        }
    }

    private tickStep(): [number, number] {
        // 目盛りの間隔（フレーム）: 主・副
        const candidates = [1, 2, 5, 10, 15, 30, 60, 150, 300, 600, 1800, 3600, 9000, 18000];
        const major = candidates.find((c) => c * this.ppf >= 80) ?? 18000;
        const minor = candidates.slice().reverse().find((c) => c < major && major % c === 0 && c * this.ppf >= 8) ?? major;
        return [major, minor];
    }

    private drawRuler(): void {
        const ctx = this.ctx;
        const r = this.rowOf("ruler");
        ctx.fillStyle = COLORS.ruler;
        ctx.fillRect(0, r.y, this.width, r.h);
        const [major, minor] = this.tickStep();
        const first = Math.floor(this.scrollFrame / minor) * minor;
        const last = this.scrollFrame + this.width / this.ppf;
        ctx.font = "11px Consolas, monospace";
        ctx.textBaseline = "top";
        for (let f = first; f <= last; f += minor) {
            const x = Math.round(this.x(f)) + 0.5;
            const isMajor = f % major === 0;
            ctx.strokeStyle = isMajor ? "#6e7781" : "#4a515c";
            ctx.beginPath();
            ctx.moveTo(x, r.y + (isMajor ? 8 : 15));
            ctx.lineTo(x, r.y + r.h);
            ctx.stroke();
            if (isMajor) {
                ctx.fillStyle = COLORS.rulerText;
                const label = major >= 30 ? `${f}  ${formatTime(f / MMD_FPS).replace(/\.\d+$/, "")}` : String(f);
                ctx.fillText(label, x + 3, r.y + 2);
                ctx.fillStyle = COLORS.gridStrong;
                ctx.fillRect(x - 0.5, r.y + r.h, 1, this.height);
            }
        }
        ctx.fillStyle = COLORS.gridStrong;
        ctx.fillRect(0, r.y + r.h - 1, this.width, 1);
    }

    private drawWave(): void {
        const wf = this.host.waveform();
        const r = this.rowOf("wave");
        const ctx = this.ctx;
        if (!wf || !wf.peaks.length) {
            ctx.fillStyle = "#4a515c";
            ctx.font = "11px sans-serif";
            ctx.textBaseline = "middle";
            ctx.fillText("キャプチャすると、動画の音声の波形が表示されます", 8, r.y + r.h / 2);
            return;
        }
        const mid = r.y + r.h / 2;
        const amp = r.h / 2 - 3;
        const perFrame = wf.rate / MMD_FPS;
        for (let px = 0; px < this.width; px++) {
            const f0 = (px / this.ppf + this.scrollFrame) * perFrame;
            const f1 = ((px + 1) / this.ppf + this.scrollFrame) * perFrame;
            const a = Math.max(0, Math.floor(f0));
            const b = Math.min(wf.peaks.length, Math.max(a + 1, Math.ceil(f1)));
            if (a >= wf.peaks.length) break;
            let m = 0;
            for (let i = a; i < b; i++) m = Math.max(m, wf.peaks[i]);
            const hh = Math.max(0.5, Math.sqrt(m) * amp);
            const frame = px / this.ppf + this.scrollFrame;
            ctx.fillStyle = this.clipAt(Math.floor(frame)) ? COLORS.waveCovered : COLORS.wave;
            ctx.fillRect(px, mid - hh, 1, hh * 2);
        }
    }

    private diamond(x: number, y: number, size: number, fill: string, stroke?: string): void {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(x, y - size);
        ctx.lineTo(x + size, y);
        ctx.lineTo(x, y + size);
        ctx.lineTo(x - size, y);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        if (stroke) {
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    private drawClips(): void {
        const ctx = this.ctx;
        const r = this.rowOf("motion");
        const y = r.y + 6;
        const hh = r.h - 12;
        for (const c0 of this.host.timeline.clips) {
            let c = c0;
            if (this.preview?.kind === "clip" && this.preview.id === c.id) c = { ...c, start: this.preview.start, end: this.preview.end };
            const x0 = this.x(c.start);
            const x1 = this.x(c.end + 1);
            if (x1 < -10 || x0 > this.width + 10) continue;
            const view = this.host.takeView(c.takeId);
            const selected = this.selection?.kind === "clip" && this.selection.id === c.id;
            ctx.fillStyle = COLORS.clip[c.variant];
            ctx.globalAlpha = view?.result ? 0.9 : 0.45;
            ctx.fillRect(x0, y, Math.max(1, x1 - x0), hh);
            ctx.globalAlpha = 1;
            // 区間の中の波線（~~~）
            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.beginPath();
            const midY = y + hh / 2;
            for (let x = Math.max(x0 + 6, 0); x < Math.min(x1 - 6, this.width); x += 2) {
                const yy = midY + Math.sin(x / 4) * 2;
                if (x === Math.max(x0 + 6, 0)) ctx.moveTo(x, yy);
                else ctx.lineTo(x, yy);
            }
            ctx.stroke();
            if (selected) {
                ctx.strokeStyle = COLORS.select;
                ctx.lineWidth = 2;
                ctx.strokeRect(x0 + 1, y + 1, Math.max(1, x1 - x0 - 2), hh - 2);
                ctx.lineWidth = 1;
            }
            this.diamond(x0, midY, 6, COLORS.diamond, "#00000080");
            this.diamond(x1, midY, 6, COLORS.diamond, "#00000080");
            const label = `${VARIANT_LABELS[c.variant]}${view?.busy ? "（再計算中…）" : ""}`;
            ctx.font = "11px sans-serif";
            ctx.textBaseline = "middle";
            const tw = ctx.measureText(label).width;
            if (x1 - x0 > tw + 24) {
                ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
                ctx.fillRect(Math.max(x0, 0) + 10, midY - 8, tw + 8, 16);
                ctx.fillStyle = COLORS.clipText;
                ctx.fillText(label, Math.max(x0, 0) + 14, midY);
            }
        }
    }

    private footLane(foot: 0 | 1): RowLayout {
        return this.layout.find((r) => r.row === "foot" && r.foot === foot)!;
    }

    private handLane(side: "left" | "right"): RowLayout {
        return this.layout.find((r) => r.row === "hand" && r.side === side)!;
    }

    private drawFoot(): void {
        const ctx = this.ctx;
        // ジャンプの区間
        for (const clip of this.host.timeline.clips) {
            const view = this.host.takeView(clip.takeId);
            if (!view?.result) continue;
            const r = this.rowSpan("foot");
            for (const [s, e] of view.result.json.jumps ?? []) {
                const a = Math.max(s + view.take.startFrame, clip.start);
                const b = Math.min(e + view.take.startFrame, clip.end);
                if (b < a) continue;
                ctx.fillStyle = COLORS.jump;
                ctx.fillRect(this.x(a), r.y, this.x(b + 1) - this.x(a), r.h);
            }
        }
        // 自動の足ピタ（表示だけ）
        for (const seg of this.autoContactSegments()) {
            const lane = this.footLane(seg.foot);
            const x0 = this.x(seg.start);
            const x1 = this.x(seg.end + 1);
            if (x1 < -10 || x0 > this.width + 10) continue;
            ctx.fillStyle = COLORS.footAuto;
            ctx.fillRect(x0, lane.y + lane.h / 2 - 1.5, Math.max(1, x1 - x0), 3);
        }
        // 手動の足ピタ: 足を固定する区間（キー 2 つの組の間）を太い線、前後のなじませる範囲を薄く、キーをひし形で
        for (const span of this.footHoldViews()) {
            const lane = this.footLane(span.foot);
            const mid = lane.y + lane.h / 2;
            const b0 = this.x(Math.max(span.clip.start, span.start - FOOT_BLEND_FRAMES));
            const b1 = this.x(Math.min(span.clip.end, span.end + FOOT_BLEND_FRAMES) + 1);
            if (b1 < -10 || b0 > this.width + 10) continue;
            ctx.fillStyle = COLORS.foot[span.foot];
            ctx.globalAlpha = 0.22;
            ctx.fillRect(b0, lane.y + 4, b1 - b0, lane.h - 8);
            ctx.globalAlpha = 1;
            const x0 = this.x(span.start);
            const x1 = this.x(span.end);
            if (x1 > x0) ctx.fillRect(x0, mid - 2, x1 - x0, 4);
        }
        for (const k of this.footKeyViews(true)) {
            const x = this.x(k.frame);
            if (x < -10 || x > this.width + 10) continue;
            const mid = this.footLane(k.foot).y + this.footLane(k.foot).h / 2;
            const selected = this.selection?.kind === "foot" && this.selection.takeId === k.takeId
                && this.selection.foot === k.foot && this.selection.frame === k.takeFrame;
            this.diamond(x, mid, 5, selected ? COLORS.select : COLORS.foot[k.foot], "#000000b0");
        }
    }

    /** 足を固定する区間（タイムラインのフレーム。クリップの範囲で切る） */
    private footHoldViews(): { takeId: string; foot: 0 | 1; start: number; end: number; clip: MotionClip }[] {
        const out: { takeId: string; foot: 0 | 1; start: number; end: number; clip: MotionClip }[] = [];
        for (const clip of this.host.timeline.clips) {
            const view = this.host.takeView(clip.takeId);
            if (!view) continue;
            const base = view.take.startFrame;
            for (const foot of [0, 1] as const) {
                // 組は撮影のキー全体で作る（クリップの外のキーとも組になる）
                const frames = this.takeFootKeys(clip.takeId, true).filter((k) => k.foot === foot).map((k) => k.frame);
                for (const [s, e] of footHoldSpans(frames)) {
                    const a = Math.max(s + base, clip.start);
                    const b = Math.min(e + base, clip.end);
                    if (b >= a) out.push({ takeId: clip.takeId, foot, start: a, end: b, clip });
                }
            }
        }
        return out;
    }

    private drawHands(): void {
        const ctx = this.ctx;
        const state = this.host.timeline;
        for (const side of ["left", "right"] as const) {
            const lane = this.handLane(side);
            let keys = state.handKeys[side];
            if (this.preview?.kind === "hand" && this.preview.side === side) {
                const p = this.preview;
                keys = keys.filter((k) => k.frame !== p.from && k.frame !== p.to)
                    .concat({ frame: p.to, shape: state.handKeys[side].find((k) => k.frame === p.from)?.shape ?? "default" })
                    .sort((a, b) => a.frame - b.frame);
            }
            const mid = lane.y + lane.h / 2;
            for (let i = 0; i < keys.length; i++) {
                const k = keys[i];
                // 最後のキーの帯は、そのキーのあるクリップの終わりまで（クリップの外なら短く）
                const clipEnd = (this.clipAt(k.frame)?.end ?? k.frame + 14) + 1;
                const next = keys[i + 1]?.frame ?? clipEnd;
                const x0 = this.x(k.frame);
                const x1 = this.x(next);
                if (x1 < 0 || x0 > this.width) continue;
                ctx.fillStyle = HAND_SHAPE_COLORS[k.shape] ?? "#888";
                ctx.globalAlpha = 0.28;
                ctx.fillRect(x0, lane.y + 3, Math.max(0, x1 - x0), lane.h - 6);
                ctx.globalAlpha = 1;
                const label = HAND_SHAPE_LABELS[k.shape] ?? k.shape;
                ctx.font = "10px sans-serif";
                ctx.textBaseline = "middle";
                if (x1 - x0 > ctx.measureText(label).width + 16) {
                    ctx.fillStyle = "#d8dde3";
                    ctx.fillText(label, Math.max(x0, 0) + 8, mid + 0.5);
                }
            }
            for (const k of keys) {
                const x = this.x(k.frame);
                if (x < -8 || x > this.width + 8) continue;
                const selected = this.selection?.kind === "hand" && this.selection.side === side && this.selection.frame === k.frame;
                this.diamond(x, mid, 5, selected ? COLORS.select : HAND_SHAPE_COLORS[k.shape] ?? "#888", "#000000b0");
            }
        }
    }

    // ---------------------------------------------------------------- 当たり判定

    private hitClipEdge(x: number): { clip: MotionClip; edge: "start" | "end" } | null {
        for (const c of this.host.timeline.clips) {
            if (Math.abs(this.x(c.start) - x) <= 6) return { clip: c, edge: "start" };
            if (Math.abs(this.x(c.end + 1) - x) <= 6) return { clip: c, edge: "end" };
        }
        return null;
    }

    private hitFoot(x: number, foot: 0 | 1): FootKeyView | null {
        let best: FootKeyView | null = null;
        let bestD = 7;
        for (const k of this.footKeyViews()) {
            if (k.foot !== foot) continue;
            const d = Math.abs(this.x(k.frame) - x);
            if (d <= bestD) {
                best = k;
                bestD = d;
            }
        }
        return best;
    }

    private hitHand(x: number, side: "left" | "right"): number | null {
        let best: number | null = null;
        let bestD = 7;
        for (const k of this.host.timeline.handKeys[side]) {
            const d = Math.abs(this.x(k.frame) - x);
            if (d <= bestD) {
                best = k.frame;
                bestD = d;
            }
        }
        return best;
    }

    private local(e: MouseEvent): { x: number; y: number } {
        const r = this.canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    // ---------------------------------------------------------------- マウス

    private onMouseDown(e: MouseEvent): void {
        if (e.button !== 0) return;
        const { x, y } = this.local(e);
        const row = this.rowAt(y);
        const frame = this.frameAt(x);
        if (!row) return;
        e.preventDefault();
        if (row.row === "motion") {
            const edge = this.hitClipEdge(x);
            if (edge) {
                const view = this.host.takeView(edge.clip.takeId);
                if (view) {
                    const takeEnd = view.take.startFrame + view.take.frames - 1;
                    const [lo, hi] = clipBounds(this.host.timeline, edge.clip, view.take.startFrame, takeEnd);
                    this.selection = { kind: "clip", id: edge.clip.id };
                    this.drag = { kind: "trim", startX: x, moved: false, data: { clip: edge.clip, edge: edge.edge, lo, hi } };
                    this.invalidate();
                    return;
                }
            }
            const clip = this.clipAt(frame);
            this.selection = clip ? { kind: "clip", id: clip.id } : null;
            this.invalidate();
        } else if (row.row === "foot") {
            const foot = row.foot!;
            const hit = this.hitFoot(x, foot);
            if (hit) {
                this.selection = { kind: "foot", takeId: hit.takeId, foot, frame: hit.takeFrame };
                this.drag = { kind: "foot-move", startX: x, moved: false, data: { key: hit } };
                this.invalidate();
                return;
            }
            // 何もない所は再生位置を動かす（足ピタの追加は右クリックのメニューから）
            this.selection = null;
        } else if (row.row === "hand") {
            const side = row.side!;
            const hit = this.hitHand(x, side);
            if (hit !== null) {
                this.selection = { kind: "hand", side, frame: hit };
                this.drag = { kind: "hand-move", startX: x, moved: false, data: { side, from: hit } };
                this.invalidate();
                return;
            }
            this.selection = null;
        }
        // それ以外: 再生位置を動かす
        this.drag = { kind: "seek", startX: x, moved: false, data: {} };
        this.host.seek(frame);
        this.invalidate();
    }

    private onMouseMove(e: MouseEvent): void {
        const { x, y } = this.local(e);
        if (!this.drag) {
            this.updateCursor(x, y, e);
            return;
        }
        const d = this.drag;
        if (Math.abs(x - d.startX) > 2) d.moved = true;
        const frame = this.frameAt(x);
        switch (d.kind) {
            case "seek":
                this.host.seek(Math.min(frame, this.host.durationFrames));
                break;
            case "trim": {
                const clip = d.data.clip as MotionClip;
                const lo = d.data.lo as number;
                const hi = d.data.hi as number;
                let start = clip.start;
                let end = clip.end;
                if (d.data.edge === "start") start = Math.max(lo, Math.min(frame, clip.end));
                else end = Math.min(hi, Math.max(frame - 1, clip.start));
                this.preview = { kind: "clip", id: clip.id, start, end };
                this.showTooltip(e, `${start} 〜 ${end}（${end - start + 1} フレーム）`);
                break;
            }
            case "foot-move": {
                // キーのあるクリップの中で動かす
                const key = d.data.key as FootKeyView;
                const to = Math.max(key.clip.start, Math.min(key.clip.end, frame));
                this.preview = { kind: "foot", takeId: key.takeId, foot: key.foot, from: key.takeFrame, to: key.takeFrame + to - key.frame };
                this.showTooltip(e, `足ピタ ${to} フレーム`);
                break;
            }
            case "hand-move": {
                const to = Math.max(0, frame);
                this.preview = { kind: "hand", side: d.data.side as "left" | "right", from: d.data.from as number, to };
                this.showTooltip(e, `${to} フレーム`);
                break;
            }
        }
        this.invalidate();
    }

    private onMouseUp(e: MouseEvent): void {
        const d = this.drag;
        if (!d) return;
        this.drag = null;
        this.hideTooltip();
        const preview = this.preview;
        this.preview = null;
        const state = this.host.timeline;
        if (d.kind === "trim" && preview?.kind === "clip" && d.moved) {
            const clip = d.data.clip as MotionClip;
            if (preview.start !== clip.start || preview.end !== clip.end) {
                this.host.commit(trimClip(state, clip.id, preview.start, preview.end), "クリップの範囲の変更");
            }
        } else if (d.kind === "foot-move" && preview?.kind === "foot" && d.moved && preview.from !== preview.to) {
            this.selection = { kind: "foot", takeId: preview.takeId, foot: preview.foot, frame: preview.to };
            this.host.commit(moveFootKey(state, preview.takeId, preview.foot, preview.from, preview.to), "足ピタの移動");
        } else if (d.kind === "foot-move" && !d.moved) {
            this.host.seek((d.data.key as FootKeyView).frame);
        } else if (d.kind === "hand-move" && preview?.kind === "hand" && d.moved && preview.from !== preview.to) {
            this.selection = { kind: "hand", side: preview.side, frame: preview.to };
            this.host.commit(moveHandKey(state, preview.side, preview.from, preview.to), "手のポーズの移動");
        } else if (d.kind === "hand-move" && !d.moved) {
            this.host.seek(d.data.from as number);
        }
        this.invalidate();
        void e;
    }

    private updateCursor(x: number, y: number, e: MouseEvent): void {
        const row = this.rowAt(y);
        let cursor = "default";
        let tip: string | null = null;
        if (row?.row === "ruler") cursor = "pointer";
        else if (row?.row === "motion") {
            if (this.hitClipEdge(x)) cursor = "ew-resize";
            const clip = this.clipAt(this.frameAt(x));
            if (clip) tip = `${VARIANT_LABELS[clip.variant]}  ${clip.start} 〜 ${clip.end}（右クリックでメニュー）`;
        } else if (row?.row === "foot") {
            const foot = row.foot!;
            const hit = this.hitFoot(x, foot);
            const frame = this.frameAt(x);
            if (hit) {
                cursor = "grab";
                const span = this.footHoldViews().find((h) => h.takeId === hit.takeId && h.foot === foot && hit.frame >= h.start && hit.frame <= h.end);
                const fixed = span && span.end > span.start ? `${span.start} 〜 ${span.end} フレームで固定` : `${hit.frame} フレームで固定（次のキーと組にすると、その間を固定）`;
                tip = `${foot === 0 ? "左" : "右"}足ピタ: ${fixed}。前後 ${FOOT_BLEND_FRAMES} フレームでなじませます`;
            } else if (this.clipAt(frame)) {
                const auto = this.autoContactSegments().some((s) => s.foot === foot && frame >= s.start && frame <= s.end);
                tip = `${auto ? "自動の足ピタ（編集できません）。" : ""}右クリックで足ピタのキーを追加`;
            }
        } else if (row?.row === "hand") {
            const side = row.side!;
            const hit = this.hitHand(x, side);
            if (hit !== null) {
                cursor = "grab";
                const k = this.host.timeline.handKeys[side].find((kk) => kk.frame === hit);
                tip = `${side === "left" ? "左" : "右"}手: ${HAND_SHAPE_LABELS[k?.shape ?? ""] ?? k?.shape}（${hit} フレーム）`;
            } else tip = "右クリックで手のポーズを追加";
        }
        this.canvas.style.cursor = cursor;
        if (tip) this.showTooltip(e, tip);
        else this.hideTooltip();
    }

    private showTooltip(e: MouseEvent, text: string): void {
        if (!this.tooltip) {
            this.tooltip = h("div", { class: "timeline-tooltip" });
            document.body.append(this.tooltip);
        }
        this.tooltip.textContent = text;
        this.tooltip.style.left = `${e.clientX + 14}px`;
        this.tooltip.style.top = `${e.clientY + 16}px`;
    }

    private hideTooltip(): void {
        this.tooltip?.remove();
        this.tooltip = null;
    }

    private onWheel(e: WheelEvent): void {
        e.preventDefault();
        const { x } = this.local(e);
        if (e.ctrlKey || e.metaKey) {
            this.setZoom(this.ppf * Math.exp(-e.deltaY * 0.0015), x);
        } else {
            const delta = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / this.ppf;
            this.scrollFrame = Math.max(0, this.scrollFrame + delta);
            this.updateScroll();
            this.invalidate();
        }
    }

    // ---------------------------------------------------------------- 右クリックのメニュー

    private onContextMenu(e: MouseEvent): void {
        e.preventDefault();
        this.hideTooltip();
        const { x, y } = this.local(e);
        const row = this.rowAt(y);
        const frame = this.frameAt(x);
        if (!row) return;
        if (row.row === "motion" || row.row === "wave") {
            const clip = this.clipAt(frame);
            if (clip) {
                this.selection = { kind: "clip", id: clip.id };
                this.invalidate();
                this.openClipMenu(e, clip, frame);
            }
        } else if (row.row === "foot") {
            const foot = row.foot!;
            this.openFootMenu(e, foot, frame, this.hitFoot(x, foot));
        } else if (row.row === "hand") {
            const side = row.side!;
            this.openHandMenu(e, side, frame, this.hitHand(x, side));
        }
    }

    private openClipMenu(e: MouseEvent, clip: MotionClip, frame: number): void {
        const state = this.host.timeline;
        const view = this.host.takeView(clip.takeId);
        const variants = (Object.keys(VARIANT_LABELS) as MotionVariant[]).map((v) => ({
            label: VARIANT_LABELS[v],
            checked: clip.variant === v,
            enabled: !!view?.result?.variants.has(v),
            action: () => this.host.commit(setClipVariant(state, clip.id, v), `クリップの種類を「${VARIANT_LABELS[v]}」に変更`)
        }));
        const entries: MenuEntry[] = [
            { heading: `モーション（${clip.start} 〜 ${clip.end}）` },
            { label: "出力の種類", submenu: variants },
            "separator",
            {
                label: "再生位置で分割", shortcut: "S",
                enabled: frame > clip.start && frame <= clip.end && this.host.currentFrame > clip.start && this.host.currentFrame <= clip.end,
                action: () => this.host.commit(splitClip(state, clip.id, this.host.currentFrame), "クリップの分割")
            },
            { label: "このクリップを削除", shortcut: "Delete", action: () => this.host.commit(removeClip(state, clip.id), "クリップの削除") },
            "separator",
            { label: "この撮影の足ピタのキーをすべて削除", enabled: !!state.footKeys[clip.takeId]?.length,
                action: () => this.host.commit(setFootKeys(state, clip.takeId, []), "足ピタのキーをすべて削除") },
            "separator",
            { label: "変換の設定を変えて再計算…", enabled: !!view, action: () => this.host.reconvertTake(clip.takeId) },
            { label: "この区間をキャプチャし直す…", enabled: !!view, action: () => view && this.host.captureAgain(view.take) }
        ];
        openMenu(entries, e.clientX, e.clientY);
    }

    private openFootMenu(e: MouseEvent, foot: 0 | 1, frame: number, key: FootKeyView | null): void {
        const state = this.host.timeline;
        const side = foot === 0 ? "左" : "右";
        const entries: MenuEntry[] = [{ heading: `${side}足の足ピタ（${key?.frame ?? frame} フレーム）` }];
        if (key) {
            this.selection = { kind: "foot", takeId: key.takeId, foot, frame: key.takeFrame };
            this.invalidate();
            entries.push({ label: "このキーを削除", shortcut: "Delete", action: () =>
                this.host.commit(removeFootKey(state, key.takeId, foot, key.takeFrame), "足ピタの削除") });
        } else {
            const clip = this.clipAt(frame);
            if (!clip || !this.host.takeView(clip.takeId)) return;
            entries.push({ label: "ここに足ピタのキーを追加", action: () => this.addFootKeyAt(foot, frame) });
        }
        openMenu(entries, e.clientX, e.clientY);
    }

    /** タイムラインの frame に足ピタのキーを置く（クリップの中だけ） */
    private addFootKeyAt(foot: 0 | 1, frame: number): void {
        const clip = this.clipAt(frame);
        const view = clip ? this.host.takeView(clip.takeId) : undefined;
        if (!clip || !view) return;
        const takeFrame = frame - view.take.startFrame;
        if (takeFrame < 0 || takeFrame >= view.take.frames) return;
        this.selection = { kind: "foot", takeId: clip.takeId, foot, frame: takeFrame };
        this.host.commit(addFootKey(this.host.timeline, clip.takeId, foot, takeFrame), "足ピタの追加");
    }

    private openHandMenu(e: MouseEvent, side: "left" | "right", frame: number, keyFrame: number | null): void {
        const state = this.host.timeline;
        const sideLabel = side === "left" ? "左手" : "右手";
        const at = keyFrame ?? frame;
        const current = keyFrame !== null ? state.handKeys[side].find((k) => k.frame === keyFrame)?.shape : null;
        const shapes: MenuEntry[] = HAND_SHAPES.map((s) => ({
            label: HAND_SHAPE_LABELS[s],
            checked: current === s,
            action: () => this.host.commit(setHandKey(state, side, at, s), keyFrame !== null ? "手のポーズの変更" : "手のポーズの追加")
        }));
        const entries: MenuEntry[] = [{ heading: `${sideLabel}のポーズ（${at} フレーム）` }];
        if (keyFrame !== null) {
            this.selection = { kind: "hand", side, frame: keyFrame };
            this.invalidate();
            entries.push({ label: "形を変える", submenu: shapes },
                { label: "このキーを削除", shortcut: "Delete", action: () => this.host.commit(removeHandKey(state, side, keyFrame), "手のポーズの削除") });
        } else {
            entries.push({ label: "ここに手のポーズを追加", submenu: shapes });
        }
        openMenu(entries, e.clientX, e.clientY);
    }

    // ---------------------------------------------------------------- キーボードから

    /** 選んでいる物を消す。消したら true */
    deleteSelection(): boolean {
        const s = this.selection;
        const state = this.host.timeline;
        if (!s) return false;
        if (s.kind === "clip") {
            this.host.commit(removeClip(state, s.id), "クリップの削除");
        } else if (s.kind === "hand") {
            this.host.commit(removeHandKey(state, s.side, s.frame), "手のポーズの削除");
        } else if (s.kind === "foot") {
            this.host.commit(removeFootKey(state, s.takeId, s.foot, s.frame), "足ピタの削除");
        }
        this.selection = null;
        this.invalidate();
        return true;
    }

    /** 選んでいるクリップを再生位置で分ける */
    splitAtPlayhead(): boolean {
        const frame = this.host.currentFrame;
        const clip = this.selection?.kind === "clip"
            ? this.host.timeline.clips.find((c) => c.id === (this.selection as { id: string }).id)
            : this.clipAt(frame);
        if (!clip || frame <= clip.start || frame > clip.end) return false;
        this.host.commit(splitClip(this.host.timeline, clip.id, frame), "クリップの分割");
        return true;
    }

    clearSelection(): void {
        this.selection = null;
        this.invalidate();
    }
}
