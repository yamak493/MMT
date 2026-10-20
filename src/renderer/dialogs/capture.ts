// キャプチャのポップアップ: 動画の区間と、モーション抽出・変換の細かな設定を選ぶ。「適用」で決定。
import type { CaptureOptions, ConversionSettings, GpuInfo, VideoInfo } from "../../shared/types";
import { MMD_FPS } from "../../shared/types";
import { defaultConversion, estimateSeconds, presetOf, QUALITY_PRESETS, type QualityPreset } from "../state/conversion";
import { formatTime, h, parseTime } from "../ui/dom";
import { Modal } from "../ui/modal";
import { cropEditor } from "./crop";

export interface CaptureDialogInput {
    videoPath: string;
    info: VideoInfo;
    initial: CaptureOptions;
    gpu: GpuInfo | null;
    hasModel: boolean;
    /** 既にキャプチャした区間（重なると上書きになることを知らせる） */
    capturedRanges: [number, number][];
    title?: string;
}

function numberInput(value: number, attrs: Record<string, unknown> = {}): HTMLInputElement {
    return h("input", { type: "number", value: String(value), ...attrs });
}

function select<T extends string>(options: [T, string][], value: T): HTMLSelectElement {
    const el = h("select", null, ...options.map(([v, label]) => h("option", { value: v, selected: v === value }, label)));
    return el;
}

function check(label: string, checked: boolean, title?: string): [HTMLLabelElement, HTMLInputElement] {
    const box = h("input", { type: "checkbox", checked });
    return [h("label", { class: "check", title }, box, label), box];
}

/** 変換の設定のフォーム（キャプチャのポップアップと、再計算のダイアログで共通） */
export function conversionForm(initial: ConversionSettings): { el: HTMLElement; read: () => ConversionSettings } {
    const c = structuredClone(initial);
    const centerMode = select([["A", "A: 軸ごとにならす（標準）"], ["B", "B: 接地している足から求める"]], c.centerMode);
    const depthScale = numberInput(c.depthScale, { step: "0.1", min: "0", max: "2", style: "width: 80px" });
    const armCollision = select([["left", "左腕の動きを自重する"], ["right", "右腕の動きを自重する"], ["none", "何もしない"]], c.armCollision);
    const [flipL, flip] = check("体の向きを反転する（MMD で後ろを向くとき）", c.flipFacing);
    const [reachL, reach] = check("胴に当てた手の位置を保つ", c.handReach);
    const [wristL, wrist] = check("手首の向きを手の検出で補正する", c.wristFromHands);
    const [legL, leg] = check("膝の向きを合わせる（脚の回転のキー）", c.legKeys);
    const [groundL, ground] = check("浮いた体を床に戻す（座る・寝転ぶ動きでは外す）", c.groundEnabled);
    const [leanL, lean] = check("体の前後・左右の傾きを補正する（寄りかかる動きでは外す）", c.leanEnabled);
    const [bothL, both] = check("接地優先で両足を床に着ける", c.lockedBothFeet);
    const parts: [keyof ConversionSettings["bodyParts"], string][] = [
        ["torso", "胴部"], ["chest", "胸"], ["head", "頭部"], ["legs", "脚"], ["clothes", "服・装飾"]
    ];
    const partBoxes = parts.map(([k, label]) => check(label, c.bodyParts[k]));
    const margin = numberInput(c.clothMarginCm, { step: "0.5", min: "0", max: "10", style: "width: 70px" });
    const morphKeys = ["a", "i", "u", "e", "o", "n"] as const;
    const morphInputs = morphKeys.map((k) => h("input", { type: "text", value: c.lipsyncMorphs[k], style: "width: 54px" }));
    const strength = numberInput(c.lipsyncStrength, { step: "0.1", min: "0", max: "1", style: "width: 70px" });
    const switchCost = numberInput(c.handSwitchCost, { step: "0.5", min: "0", max: "20", style: "width: 70px" });

    const el = h("div", null,
        h("div", { class: "form-grid" },
            h("label", null, "センターの求め方"), centerMode,
            h("label", null, "前後の移動の倍率"), h("div", { class: "form-row" }, depthScale, h("span", { class: "form-help" }, "カメラの奥行き方向の移動を縮める（1 = そのまま）")),
            h("label", null, "腕どうしの貫通"), armCollision,
            h("label", null, "腕・指先の当たり判定"), h("div", { class: "form-row" }, ...partBoxes.map(([l]) => l)),
            h("label", null, "服の厚み [cm]"), margin,
            h("div", { class: "full form-row" }, reachL, wristL, legL),
            h("div", { class: "full form-row" }, groundL, leanL, bothL, flipL),
            h("label", null, "口のモーフ名"), h("div", { class: "form-row" }, ...morphKeys.flatMap((_k, i) => [h("span", { class: "form-help" }, "あいうえおん"[i]), morphInputs[i]])),
            h("label", null, "口の開き"), strength,
            h("label", null, "手の形の切り替えにくさ"), h("div", { class: "form-row" }, switchCost, h("span", { class: "form-help" }, "大きいほど手の形がちらつかない"))));
    const read = (): ConversionSettings => ({
        centerMode: centerMode.value as "A" | "B",
        depthScale: clampNum(depthScale.value, 0, 2, 1),
        flipFacing: flip.checked,
        armCollision: armCollision.value as ConversionSettings["armCollision"],
        bodyParts: Object.fromEntries(parts.map(([k], i) => [k, partBoxes[i][1].checked])) as ConversionSettings["bodyParts"],
        handReach: reach.checked,
        clothMarginCm: clampNum(margin.value, 0, 10, 1),
        wristFromHands: wrist.checked,
        legKeys: leg.checked,
        groundEnabled: ground.checked,
        leanEnabled: lean.checked,
        lockedBothFeet: both.checked,
        lipsyncMorphs: Object.fromEntries(morphKeys.map((k, i) => [k, morphInputs[i].value.trim()])) as ConversionSettings["lipsyncMorphs"],
        lipsyncStrength: clampNum(strength.value, 0, 1, 1),
        handSwitchCost: clampNum(switchCost.value, 0, 20, 3)
    });
    return { el, read };
}

function clampNum(text: string, lo: number, hi: number, fallback: number): number {
    const v = Number(text);
    return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;
}

/** 戻り値: 決めた設定（キャンセルなら null） */
export function captureDialog(input: CaptureDialogInput): Promise<CaptureOptions | null> {
    return new Promise((resolve) => {
        const info = input.info;
        const o = structuredClone(input.initial);
        const duration = info.duration;
        o.start = Math.max(0, Math.min(o.start, duration));
        o.end = o.end > o.start ? Math.min(o.end, duration) : duration;
        let result: CaptureOptions | null = null;

        // ---- 区間のプレビュー ----
        const video = h("video", { src: window.mmt.mediaUrl(input.videoPath), preload: "auto", muted: true });
        video.muted = true;
        const sel = h("div", { class: "range-sel" });
        const cursor = h("div", { class: "range-cursor" });
        const hStart = h("div", { class: "range-handle", title: "始点" });
        const hEnd = h("div", { class: "range-handle", title: "終点" });
        const captured = input.capturedRanges.map(([s, e]) => h("div", {
            style: `position:absolute;top:0;height:5px;background:#4cb870;left:${(s / duration) * 100}%;width:${((e - s) / duration) * 100}%`,
            title: "キャプチャ済みの区間"
        }));
        const bar = h("div", { class: "range-bar", title: "クリックで位置を確認、つまみで始点・終点を変えます" }, ...captured, sel, cursor, hStart, hEnd);
        const startIn = h("input", { type: "text", class: "time-input", value: formatTime(o.start) });
        const endIn = h("input", { type: "text", class: "time-input", value: formatTime(o.end) });
        const lengthText = h("span", { class: "dim" });
        const setStartBtn = h("button", { class: "button small" }, "現在位置を始点に");
        const setEndBtn = h("button", { class: "button small" }, "現在位置を終点に");
        const playBtn = h("button", { class: "button small" }, "▶ 区間を再生");
        const allBtn = h("button", { class: "button small" }, "全体");

        // ---- 設定 ----
        const fpsSel = select<string>([["auto", `自動（${info.autoFps.toFixed(2).replace(/\.00$/, "")} fps）`], ["source", `元の動画のまま（${info.fps.toFixed(2).replace(/\.00$/, "")} fps）`],
            ["30", "30 fps"], ["25", "25 fps"], ["24", "24 fps"], ["15", "15 fps"]], String(o.fps));
        const heightSel = select<string>([["480", "480p（速い）"], ["720", "720p（標準）"], ["1080", "1080p（小さく写った人向け）"], ["0", "元の解像度"]], String(o.maxHeight));
        const quality = select<string>([...Object.entries(QUALITY_PRESETS).map(([k, v]) => [k, v.label] as [string, string]), ["custom", "カスタム"]], presetOf(o));
        const personSel = select([["largest", "一番大きく写っている人"], ["center", "画面の中央の人"]], o.personSelect);
        const [handsL, handsBox] = check("手の形・手首の向きを推定する", o.hands);
        const [lipL, lipBox] = check(info.has_audio ? "音声から口パクを作る" : "音声から口パクを作る（この動画には音声がありません）", o.lipsync && info.has_audio);
        lipBox.disabled = !info.has_audio;
        const [demucsL, demucsBox] = check("ボーカルを分離してから認識する（伴奏のある曲）", o.demucs);
        const langSel = select([["jpn", "日本語の音素で認識（英語の歌でも可）"], ["eng", "英語"], ["ipa", "全言語"]], o.lipsyncLang);
        const batch = numberInput(o.batchSize, { min: "1", max: "64", step: "1", style: "width: 70px" });
        const numAug = select([["1", "1"], ["3", "3"], ["5", "5"], ["7", "7"]], String(o.numAug));
        const antialias = select([["1", "1"], ["2", "2"], ["4", "4"]], String(o.antialias));
        const thresh = numberInput(o.detectorThreshold, { min: "0.05", max: "0.9", step: "0.05", style: "width: 70px" });
        const conv = conversionForm(o.conversion ?? defaultConversion());
        const estimate = h("div", { class: "estimate" });
        // 切り抜き（新しいキャプチャの既定は正方形。切り抜きの無かった頃の撮影は切り抜かない）
        const crop = cropEditor(video, info, o.cropMode ?? (o.crop ? "free" : "none"), o.crop ?? null,
            () => Number(heightSel.value), () => update());

        quality.addEventListener("change", () => {
            const p = QUALITY_PRESETS[quality.value as QualityPreset];
            if (p) {
                numAug.value = String(p.numAug);
                antialias.value = String(p.antialias);
            }
            update();
        });
        for (const el of [numAug, antialias]) el.addEventListener("change", () => {
            const n = Number(numAug.value);
            const a = Number(antialias.value);
            const match = Object.entries(QUALITY_PRESETS).find(([, v]) => v.numAug === n && v.antialias === a);
            quality.value = match ? match[0] : "custom";
            update();
        });

        const body = h("div", null,
            !input.hasModel ? h("div", { class: "notice info" }, "モデルを読み込んでいないので、標準的な体格のボーン寸法で変換します。あとでモデルを読み込むと、そのモデルに合わせて変換し直します。") : null,
            h("div", { class: "capture-layout" },
                h("div", { class: "capture-preview" },
                    crop.stage, crop.controls, bar,
                    h("div", { class: "form-row" }, "始点", startIn, "終点", endIn, lengthText),
                    h("div", { class: "form-row" }, setStartBtn, setEndBtn, playBtn, allBtn),
                    estimate),
                h("div", null,
                    h("fieldset", { class: "form-section" }, h("legend", null, "モーション抽出"),
                        h("div", { class: "form-grid", style: "grid-template-columns: 110px 1fr" },
                            h("label", null, "精度"), quality,
                            h("label", null, "推論する fps"), fpsSel,
                            h("label", null, "解像度"), heightSel,
                            h("label", null, "対象の人物"), personSel)),
                    h("fieldset", { class: "form-section" }, h("legend", null, "手・口パク"),
                        h("div", { class: "form-row" }, handsL),
                        h("div", { class: "form-row" }, lipL),
                        h("div", { class: "form-row", style: "padding-left: 20px" }, demucsL),
                        h("div", { class: "form-row", style: "padding-left: 20px" }, langSel)))),
            h("details", { class: "form-section" }, h("summary", null, "変換の設定（あとで変えて再計算もできます）"), conv.el),
            h("details", { class: "form-section" }, h("summary", null, "推論の詳細設定"),
                h("div", { class: "form-grid" },
                    h("label", null, "バッチサイズ"), h("div", { class: "form-row" }, batch, h("span", { class: "form-help" }, "GPU のメモリが余っていれば大きく（足りなければ自動で分割します）")),
                    h("label", null, "TTA の枚数"), h("div", { class: "form-row" }, numAug, h("span", { class: "form-help" }, "複数の切り出しで推定して統合（震えが減るが時間は比例して増える）")),
                    h("label", null, "超サンプリング"), antialias,
                    h("label", null, "人物検出のしきい値"), h("div", { class: "form-row" }, thresh, h("span", { class: "form-help" }, "検出が途切れるときは 0.15 程度に下げる")))));

        const modal = new Modal({
            title: input.title ?? "動画を読み込む",
            size: "xwide",
            body,
            footerNote: "適用すると、初回だけ AI のモデルのダウンロードと利用規約への同意を求めます",
            buttons: [
                { label: "キャンセル" },
                {
                    label: "適用", kind: "primary", isDefault: true,
                    onClick: () => {
                        result = read();
                        if (result.end - result.start < 0.2) {
                            result = null;
                            startIn.focus();
                            return false;
                        }
                    }
                }
            ],
            onClose: () => {
                video.pause();
                video.removeAttribute("src");
                video.load();
                crop.dispose();
                resolve(result);
            }
        }).open();

        function read(): CaptureOptions {
            const fps = fpsSel.value === "auto" || fpsSel.value === "source" ? fpsSel.value : Number(fpsSel.value);
            const c = crop.read();
            return {
                start: o.start,
                end: o.end,
                fps: fps as CaptureOptions["fps"],
                maxHeight: Number(heightSel.value),
                crop: c.crop,
                cropMode: c.mode,
                personSelect: personSel.value as CaptureOptions["personSelect"],
                batchSize: Math.round(clampNum(batch.value, 1, 64, 8)),
                numAug: Number(numAug.value),
                antialias: Number(antialias.value),
                detectorThreshold: clampNum(thresh.value, 0.05, 0.9, 0.25),
                hands: handsBox.checked,
                lipsync: lipBox.checked && info.has_audio,
                demucs: demucsBox.checked,
                lipsyncLang: langSel.value as CaptureOptions["lipsyncLang"],
                conversion: conv.read()
            };
        }

        function update() {
            const pct = (t: number) => `${(t / Math.max(duration, 1e-6)) * 100}%`;
            sel.style.left = pct(o.start);
            sel.style.width = `${((o.end - o.start) / Math.max(duration, 1e-6)) * 100}%`;
            hStart.style.left = pct(o.start);
            hEnd.style.left = pct(o.end);
            startIn.value = formatTime(o.start);
            endIn.value = formatTime(o.end);
            const len = o.end - o.start;
            const fps = fpsSel.value === "auto" ? info.autoFps : fpsSel.value === "source" ? info.fps : Number(fpsSel.value);
            const frames = Math.floor(len * fps);
            lengthText.textContent = `長さ ${len.toFixed(2)} 秒（${frames} フレーム・MMD では ${Math.round(len * MMD_FPS)} フレーム）`;
            const sec = estimateSeconds(frames, read());
            const overlap = input.capturedRanges.some(([s, e]) => s < o.end && e > o.start);
            const lines = [`処理時間の目安: 約 ${sec >= 120 ? `${Math.round(sec / 60)} 分` : `${Math.max(5, Math.round(sec))} 秒`}${input.gpu ? `（GPU: ${input.gpu.name}）` : ""}`];
            if (!input.gpu) lines.push("NVIDIA の GPU を確認できませんでした。キャプチャには NVIDIA の GPU（GeForce GTX 900 番台以降）が必要です。");
            if (len > 120) lines.push("区間が長いので時間がかかります。まずは 10〜30 秒程度で試すのがおすすめです。");
            if (overlap) lines.push("キャプチャ済みの区間と重なっています。重なった所のタイムラインは新しいキャプチャで置き換えます。");
            if (Math.abs(fps - info.fps) > 0.01 && Math.abs(info.fps / fps - Math.round(info.fps / fps)) > 0.01) {
                lines.push("元の動画の fps の整数分の 1 ではないので、フレームの複製・間引きで動きがカクつくことがあります（「自動」がおすすめです）。");
            }
            estimate.textContent = lines.join("\n");
            estimate.style.whiteSpace = "pre-line";
            estimate.classList.toggle("warn", lines.length > 1);
        }

        function setRange(s: number, e: number) {
            o.start = Math.max(0, Math.min(s, duration));
            o.end = Math.max(o.start, Math.min(e, duration));
            update();
        }

        const timeAt = (clientX: number) => {
            const r = bar.getBoundingClientRect();
            return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * duration;
        };
        let dragging: "start" | "end" | "seek" | null = null;
        hStart.addEventListener("mousedown", (e) => { dragging = "start"; e.stopPropagation(); e.preventDefault(); });
        hEnd.addEventListener("mousedown", (e) => { dragging = "end"; e.stopPropagation(); e.preventDefault(); });
        bar.addEventListener("mousedown", (e) => {
            dragging = "seek";
            video.currentTime = timeAt(e.clientX);
            e.preventDefault();
        });
        const onMove = (e: MouseEvent) => {
            if (!dragging) return;
            const t = timeAt(e.clientX);
            if (dragging === "start") setRange(Math.min(t, o.end - 0.2), o.end);
            else if (dragging === "end") setRange(o.start, Math.max(t, o.start + 0.2));
            video.currentTime = dragging === "seek" ? t : dragging === "start" ? o.start : o.end;
        };
        const onUp = () => { dragging = null; };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        void modal.closedPromise.then(() => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        });
        video.addEventListener("timeupdate", () => {
            cursor.style.left = `${(video.currentTime / Math.max(duration, 1e-6)) * 100}%`;
            if (playingRange && video.currentTime >= o.end) {
                video.pause();
                playingRange = false;
            }
        });
        let playingRange = false;
        playBtn.addEventListener("click", () => {
            video.currentTime = o.start;
            playingRange = true;
            void video.play();
        });
        setStartBtn.addEventListener("click", () => setRange(Math.min(video.currentTime, o.end - 0.2), o.end));
        setEndBtn.addEventListener("click", () => setRange(o.start, Math.max(video.currentTime, o.start + 0.2)));
        allBtn.addEventListener("click", () => setRange(0, duration));
        startIn.addEventListener("change", () => {
            const t = parseTime(startIn.value);
            if (Number.isFinite(t)) setRange(Math.min(t, o.end - 0.2), o.end);
            else update();
        });
        endIn.addEventListener("change", () => {
            const t = parseTime(endIn.value);
            if (Number.isFinite(t)) setRange(o.start, Math.max(t, o.start + 0.2));
            else update();
        });
        for (const el of [fpsSel, handsBox, lipBox, batch]) el.addEventListener("change", update);
        heightSel.addEventListener("change", () => crop.refresh());
        video.addEventListener("loadedmetadata", () => { video.currentTime = o.start; });
        crop.refresh();
    });
}
