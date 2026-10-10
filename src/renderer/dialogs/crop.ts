// キャプチャのポップアップのプレビューで、推論する前に動画を切り抜く範囲を選ぶ（枠のドラッグで移動、四隅で大きさ）。
import type { CropMode, CropRect, VideoInfo } from "../../shared/types";
import { h } from "../ui/dom";

/** 切り抜く範囲の最小の辺 [px] */
const MIN_SIDE = 64;
/** プレビューの枠の縦横比（.crop-stage と同じ） */
const STAGE_ASPECT = 16 / 9;

export interface CropEditor {
    /** 動画と切り抜きの枠を重ねたプレビュー */
    stage: HTMLElement;
    /** 切り抜きの方法の選択と、大きさの表示 */
    controls: HTMLElement;
    read(): { mode: CropMode; crop: CropRect | null };
    /** 大きさの表示を更新する（推論する解像度を変えたとき） */
    refresh(): void;
    dispose(): void;
}

type Drag =
    | { kind: "move"; px: number; py: number; rect: CropRect }
    | { kind: "corner"; ax: number; ay: number; sx: number; sy: number };

/** 高さが maxHeight を超えるときだけ縮小した大きさ（エンジンの scaled_size と同じ） */
function scaledSize(w: number, h: number, maxHeight: number): [number, number] {
    if (maxHeight && h > maxHeight) return [Math.floor((w * maxHeight) / h / 2) * 2, Math.floor(maxHeight / 2) * 2];
    return [Math.floor(w / 2) * 2, Math.floor(h / 2) * 2];
}

export function cropEditor(video: HTMLVideoElement, info: VideoInfo, mode: CropMode, initial: CropRect | null,
    maxHeight: () => number, onChange: () => void): CropEditor {
    const W = Math.max(1, info.width);
    const H = Math.max(1, info.height);
    let rect: CropRect | null = initial && initial.w > 0 && initial.h > 0 ? clampRect({ ...initial }) : null;

    const handles = (["nw", "ne", "sw", "se"] as const).map((c) => h("div", { class: `crop-handle ${c}`, dataset: { corner: c } }));
    const area = h("div", { class: "crop-area", title: "ドラッグで移動、四隅で大きさを変えます" }, ...handles);
    // 枠の外側を暗くする上・下・左・右の板
    const shades = [0, 1, 2, 3].map(() => h("div", { class: "crop-shade" }));
    const frame = h("div", { class: "crop-frame" }, video, ...shades, area);
    const stage = h("div", { class: "crop-stage" }, frame);
    // 動画の縦横比のまま、プレビューの枠に収める
    const va = W / H;
    const fw = va >= STAGE_ASPECT ? 100 : (va / STAGE_ASPECT) * 100;
    const fh = va >= STAGE_ASPECT ? (STAGE_ASPECT / va) * 100 : 100;
    Object.assign(frame.style, { width: `${fw}%`, height: `${fh}%`, left: `${(100 - fw) / 2}%`, top: `${(100 - fh) / 2}%` });

    const modeSel = h("select", null,
        ...([["none", "切り抜かない"], ["square", "正方形（推奨）"], ["free", "自由な縦横比"]] as [CropMode, string][])
            .map(([v, label]) => h("option", { value: v, selected: v === mode }, label)));
    const maxBtn = h("button", { class: "button small", title: "今の縦横比で、中央に一番大きく取ります" }, "中央に最大");
    const sizeText = h("span", { class: "dim" });
    const controls = h("div", { class: "form-row" }, "切り抜き", modeSel, maxBtn, sizeText);

    function clampRect(r: CropRect): CropRect {
        r.w = Math.max(Math.min(MIN_SIDE, W), Math.min(r.w, W));
        r.h = Math.max(Math.min(MIN_SIDE, H), Math.min(r.h, H));
        r.x = Math.max(0, Math.min(r.x, W - r.w));
        r.y = Math.max(0, Math.min(r.y, H - r.h));
        return r;
    }

    /** 中央に一番大きな範囲（正方形なら短い辺の正方形） */
    function largest(m: CropMode): CropRect {
        if (m === "square") {
            const s = Math.min(W, H);
            return { x: (W - s) / 2, y: (H - s) / 2, w: s, h: s };
        }
        return { x: 0, y: 0, w: W, h: H };
    }

    function setMode(m: CropMode) {
        if (m === "square") {
            if (rect) {
                // 今の範囲の中心のまま、短い辺の正方形にする
                const s = Math.min(rect.w, rect.h);
                rect = clampRect({ x: rect.x + (rect.w - s) / 2, y: rect.y + (rect.h - s) / 2, w: s, h: s });
            } else rect = largest("square");
        } else if (m === "free" && !rect) {
            rect = largest("square");
        }
        mode = m;
        render();
    }

    function render() {
        const on = mode !== "none" && !!rect;
        frame.classList.toggle("cropping", on);
        maxBtn.disabled = mode === "none";
        if (on && rect) {
            const l = (rect.x / W) * 100;
            const t = (rect.y / H) * 100;
            const r = ((rect.x + rect.w) / W) * 100;
            const b = ((rect.y + rect.h) / H) * 100;
            const place = (el: HTMLElement, x0: number, y0: number, x1: number, y1: number) => Object.assign(el.style, {
                left: `${x0}%`, top: `${y0}%`, width: `${x1 - x0}%`, height: `${y1 - y0}%`
            });
            place(area, l, t, r, b);
            place(shades[0], 0, 0, 100, t);
            place(shades[1], 0, b, 100, 100);
            place(shades[2], 0, t, l, b);
            place(shades[3], r, t, 100, b);
        }
        const r = read().crop;
        const [sw, sh] = r ? [r.w, r.h] : [W, H];
        const [iw, ih] = scaledSize(sw, sh, maxHeight());
        sizeText.textContent = r
            ? `${r.w} × ${r.h}（元 ${W} × ${H}）→ 推論 ${iw} × ${ih}`
            : `元 ${W} × ${H} → 推論 ${iw} × ${ih}`;
        onChange();
    }

    function read(): { mode: CropMode; crop: CropRect | null } {
        if (mode === "none" || !rect) return { mode, crop: null };
        const x = Math.round(rect.x);
        const y = Math.round(rect.y);
        let w = Math.min(Math.round(rect.w), W - x);
        let hh = Math.min(Math.round(rect.h), H - y);
        if (mode === "square") w = hh = Math.min(w, hh);
        if (x <= 0 && y <= 0 && w >= W && hh >= H) return { mode, crop: null };   // 全体と同じ
        return { mode, crop: { x, y, w, h: hh } };
    }

    // ---- ドラッグ ----
    let drag: Drag | null = null;
    const toSrc = (e: MouseEvent): [number, number] => {
        const r = frame.getBoundingClientRect();
        return [
            Math.max(0, Math.min(W, ((e.clientX - r.left) / Math.max(r.width, 1)) * W)),
            Math.max(0, Math.min(H, ((e.clientY - r.top) / Math.max(r.height, 1)) * H))
        ];
    };
    /** 角 (ax, ay) を固定して、反対の角を (px, py) へ動かす */
    function resizeFrom(d: Extract<Drag, { kind: "corner" }>, px: number, py: number) {
        const dx = px - d.ax;
        const dy = py - d.ay;
        const sx = dx === 0 ? d.sx : Math.sign(dx);
        const sy = dy === 0 ? d.sy : Math.sign(dy);
        const roomX = sx > 0 ? W - d.ax : d.ax;
        const roomY = sy > 0 ? H - d.ay : d.ay;
        let w: number;
        let hh: number;
        if (mode === "square") {
            w = hh = Math.min(Math.max(MIN_SIDE, Math.abs(dx), Math.abs(dy)), roomX, roomY);
        } else {
            w = Math.min(Math.max(MIN_SIDE, Math.abs(dx)), roomX);
            hh = Math.min(Math.max(MIN_SIDE, Math.abs(dy)), roomY);
        }
        rect = clampRect({ x: sx > 0 ? d.ax : d.ax - w, y: sy > 0 ? d.ay : d.ay - hh, w, h: hh });
    }

    frame.addEventListener("mousedown", (e) => {
        if (mode === "none" || e.button !== 0) return;
        e.preventDefault();
        const [px, py] = toSrc(e);
        const corner = (e.target as HTMLElement).dataset?.corner;
        if (corner && rect) {
            // 掴んだ角の反対の角を固定する
            const west = corner.includes("w");
            const north = corner.includes("n");
            drag = { kind: "corner", ax: west ? rect.x + rect.w : rect.x, ay: north ? rect.y + rect.h : rect.y, sx: west ? -1 : 1, sy: north ? -1 : 1 };
        } else if (rect && e.target === area) {
            drag = { kind: "move", px, py, rect: { ...rect } };
        } else {
            // 枠の外から、新しく範囲を描く
            drag = { kind: "corner", ax: px, ay: py, sx: 1, sy: 1 };
            resizeFrom(drag, px, py);
            render();
        }
    });
    const onMove = (e: MouseEvent) => {
        if (!drag) return;
        const [px, py] = toSrc(e);
        if (drag.kind === "move") {
            rect = clampRect({ ...drag.rect, x: drag.rect.x + px - drag.px, y: drag.rect.y + py - drag.py });
        } else resizeFrom(drag, px, py);
        render();
    };
    const onUp = () => { drag = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    modeSel.addEventListener("change", () => setMode(modeSel.value as CropMode));
    maxBtn.addEventListener("click", () => {
        if (mode === "none") return;
        if (mode === "square" || !rect) rect = largest(mode);
        else {
            // 自由な縦横比: 今の縦横比のまま、中央に一番大きく
            const k = Math.min(W / rect.w, H / rect.h);
            const w = rect.w * k;
            const hh = rect.h * k;
            rect = { x: (W - w) / 2, y: (H - hh) / 2, w, h: hh };
        }
        render();
    });

    // 最初の表示（onChange は呼び出し側の準備ができてから refresh で呼ぶ）
    if (mode !== "none" && !rect) rect = largest(mode);
    if (mode === "square" && rect && rect.w !== rect.h) {
        const s = Math.min(rect.w, rect.h);
        rect = clampRect({ x: rect.x + (rect.w - s) / 2, y: rect.y + (rect.h - s) / 2, w: s, h: s });
    }

    return {
        stage,
        controls,
        read,
        refresh: render,
        dispose: () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        }
    };
}
