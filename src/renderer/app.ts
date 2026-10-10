// アプリの本体: プロジェクトの状態（モデル・動画・撮影・タイムライン）を持ち、画面の各部品をつなぐ。
import type {
    CaptureOptions, ConversionResultJson, GpuInfo, HandRotationTable, ModelRef, OpenedProject, ProjectDocument,
    Take, TimelineState, VideoInfo
} from "../shared/types";
import { APP_REPOSITORY, FOOT_KEYS_VERSION, MMD_FPS } from "../shared/types";
import { captureDialog, conversionForm } from "./dialogs/capture";
import { exportDialog, type ExportOptions } from "./dialogs/export";
import { ensureReady } from "./dialogs/setup";
import { showAboutDialog, showMotionTermsDialog } from "./dialogs/terms";
import { EngineClient, EngineRequestError } from "./engineClient";
import { composeMotion, shiftMotion, sortedClips } from "./motion/compose";
import { loadTakeResult, type TakeResult } from "./motion/takeData";
import { thinMotion } from "./motion/thin";
import { writeVmd } from "./motion/vmd";
import { defaultCapture, defaultConversion, engineSettings } from "./state/conversion";
import { History } from "./state/history";
import { addTake, emptyTimeline, normalizeTimeline, pruneTakes, setConversion } from "./state/timelineOps";
import { Timeline, type TakeView, type TimelineHost } from "./timeline/timeline";
import { $, basename, formatTime, h } from "./ui/dom";
import { MenuBar, type MenuEntry } from "./ui/menu";
import { alertDialog, anyModalOpen, choiceDialog, confirmDialog, Modal } from "./ui/modal";
import { ProgressDialog } from "./ui/progressDialog";
import { toast } from "./ui/toast";
import { VideoPlayer } from "./video/videoPlayer";
import { extractRig, missingBones, referencedFiles } from "./viewer/rig";
import { Viewer } from "./viewer/viewer";

const VIDEO_FILTERS = [{ name: "動画（mp4 / mov / webm）", extensions: ["mp4", "mov", "webm", "m4v"] }];
const MODEL_FILTERS = [{ name: "MMD モデル（PMX / PMD）", extensions: ["pmx", "pmd"] }];
const PROJECT_FILTERS = [{ name: "MonoCam Motion Trace プロジェクト", extensions: ["mmt"] }];
const VMD_FILTERS = [{ name: "VMD モーション", extensions: ["vmd"] }];
const WAVEFORM_RATE = 100;

interface TakeRuntime {
    result: TakeResult | null;
    busy: boolean;
    /** 読み込んだ結果を作った条件（足ピタのキー・設定・モデル） */
    loadedSignature: string | null;
    error: string | null;
}

function errorMessage(e: unknown): string {
    if (e instanceof EngineRequestError) return e.error.message;
    const msg = (e as Error)?.message ?? String(e);
    return msg.replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}

export class App implements TimelineHost {
    private engine = new EngineClient();
    private viewer: Viewer;
    private timelineView: Timeline;
    private history = new History<TimelineState>(emptyTimeline());
    private player: VideoPlayer | null = null;
    private video = $<HTMLVideoElement>("video");

    private workDir = "";
    private filePath: string | null = null;
    private doc: ProjectDocument = this.emptyDocument();
    private takes = new Map<string, TakeRuntime>();
    private waveformData: { peaks: Float32Array; rate: number } | null = null;
    private handTable: Partial<HandRotationTable> | null = null;
    /** 今のモデルのリグから作った、形ごとの指の回転（撮影の変換結果のものより優先する） */
    private modelHandTable: { rigPath: string; rotations: Partial<HandRotationTable>; transitionSec: number } | null = null;
    private transitionFrames = 5;
    private savedHistoryRef: TimelineState | null = null;
    private docDirty = false;
    private version = "";
    private gpu: GpuInfo | null = null;
    private reconcileTimer = 0;
    private reconciling = false;
    /** プロジェクトを開いている途中（撮影の結果を読み終えるまで、変換し直さない） */
    private openingProject = false;
    private composeTimer = 0;
    private captureRunning = false;
    private lastFollow = 0;

    constructor() {
        this.viewer = new Viewer($<HTMLCanvasElement>("viewer-canvas"));
        this.timelineView = new Timeline(this);
        this.savedHistoryRef = this.history.current;
        this.history.onChange = () => this.onTimelineChanged();
        this.viewer.onTick = (frame, playing) => this.onTick(frame, playing);
        this.buildMenus();
        this.bindControls();
        this.bindKeys();
        this.bindSplitters();
        window.mmt.onCloseRequested(() => void this.requestClose());
        window.mmt.onOpenFile((path) => void this.openDroppedFile(path));
    }

    async init(): Promise<void> {
        this.version = await window.mmt.appVersion();
        $("status-version").textContent = `v${this.version}`;
        await this.viewer.initPhysics();
        if (!this.viewer.physicsAvailable) {
            ($<HTMLInputElement>("chk-physics")).checked = false;
            ($<HTMLInputElement>("chk-physics")).disabled = true;
        }
        const { workDir } = await window.mmt.newProject();
        this.workDir = workDir;
        this.refreshAll();
        this.timelineView.zoomToFit();
        void window.mmt.setupStatus().then((s) => { this.gpu = s.gpu ?? null; });
        // 作業フォルダを作り終えてから、起動の引数（.mmt のダブルクリック）のプロジェクトを開く。
        // 先に開くと、開いたプロジェクトの作業フォルダを上の空の作業フォルダで上書きしてしまう
        const file = await window.mmt.takeOpenFile();
        if (file) await this.openProject(file);
    }

    private emptyDocument(): ProjectDocument {
        return {
            format: "mmt-project", version: 1, appVersion: this.version, savedAt: "",
            model: null, video: null, waveform: null, takes: [], timeline: emptyTimeline(), conversion: defaultConversion(),
            view: { playbackRate: 1, zoom: 2, physics: true }
        };
    }

    // ================================================================ TimelineHost

    get timeline(): TimelineState {
        return this.history.current;
    }

    get durationFrames(): number {
        const videoFrames = this.doc.video ? Math.ceil(this.doc.video.info.duration * MMD_FPS) : 0;
        const clipEnd = this.timeline.clips.reduce((m, c) => Math.max(m, c.end + 1), 0);
        return Math.max(videoFrames, clipEnd, 1);
    }

    get currentFrame(): number {
        return this.viewer.currentFrame;
    }

    get playing(): boolean {
        return this.viewer.playing;
    }

    takeView(takeId: string): TakeView | undefined {
        const take = this.doc.takes.find((t) => t.id === takeId);
        if (!take) return undefined;
        const rt = this.takes.get(takeId);
        return { take, result: rt?.result ?? null, busy: rt?.busy ?? false };
    }

    waveform(): { peaks: Float32Array; rate: number } | null {
        return this.waveformData;
    }

    seek(frame: number): void {
        void this.viewer.seek(Math.max(0, Math.min(frame, this.durationFrames - 1)));
        this.updateTimeDisplay(frame);
        this.timelineView.invalidate();
    }

    commit(next: TimelineState, label: string): void {
        if (this.history.push(next, label)) this.setStatus(label);
    }

    reconvertTake(takeId: string): void {
        void this.openReconvertDialog(takeId);
    }

    captureAgain(take: Take): void {
        void this.startCapture({ ...take.options, conversion: this.conversionOf(take), start: take.start, end: take.end }, "この区間をキャプチャし直す");
    }

    // ================================================================ 画面の部品

    private buildMenus(): void {
        const hasProjectContent = () => !!(this.doc.model || this.doc.video || this.doc.takes.length);
        new MenuBar($("menubar"), [
            {
                label: "ファイル",
                items: (): MenuEntry[] => [
                    { label: "新しく作成", shortcut: "Ctrl+N", action: () => void this.newProject() },
                    { label: "開く…", shortcut: "Ctrl+O", action: () => void this.openProject() },
                    "separator",
                    { label: "名前を付けて保存…", shortcut: "Ctrl+Shift+S", enabled: hasProjectContent(), action: () => void this.saveProject(true) },
                    { label: "上書き保存", shortcut: "Ctrl+S", enabled: hasProjectContent(), action: () => void this.saveProject(false) },
                    "separator",
                    { label: "モデルを読み込む（PMX / PMD）…", action: () => void this.loadModelDialog() },
                    { label: "動画を読み込む（mp4 / mov / webm）…", action: () => void this.loadVideoDialog() },
                    { label: "キャプチャ…", enabled: !!this.doc.video, action: () => void this.openCaptureForCurrent() },
                    "separator",
                    { label: "モーションを出力（VMD）…", shortcut: "Ctrl+E", enabled: this.timeline.clips.length > 0 || this.hasHandKeys(), action: () => void this.exportVmd() },
                    "separator",
                    { label: "エンジンのログを開く", action: () => void this.openEngineLog() },
                    { label: "実行環境を再セットアップ…", enabled: !this.captureRunning, action: () => void this.reinstallRuntime() }
                ]
            },
            {
                label: "ソースコード",
                items: (): MenuEntry[] => [
                    { label: "リポジトリを開く（GitHub）", action: () => void window.mmt.openExternal(APP_REPOSITORY) },
                    { label: "不具合を報告（Issue を作成）", action: () => void this.reportIssue() }
                ]
            },
            {
                label: "利用規約",
                items: (): MenuEntry[] => [
                    { label: "本ソフトウェアについて", action: () => showAboutDialog(this.version) },
                    { label: "本ソフトウェアで作成したモーションについて《重要》", action: () => showMotionTermsDialog() }
                ]
            }
        ]);
    }

    private bindControls(): void {
        $("btn-export").addEventListener("click", () => void this.exportVmd());
        $("btn-load-video").addEventListener("click", () => void this.loadVideoDialog());
        $("btn-load-model").addEventListener("click", () => void this.loadModelDialog());
        $("btn-capture").addEventListener("click", () => void this.openCaptureForCurrent());
        $("btn-play").addEventListener("click", () => void this.togglePlay());
        $("btn-to-start").addEventListener("click", () => this.seek(0));
        $("btn-to-end").addEventListener("click", () => this.seek(this.durationFrames - 1));
        $("btn-step-back").addEventListener("click", () => this.step(-1));
        $("btn-step-forward").addEventListener("click", () => this.step(1));
        $("btn-undo").addEventListener("click", () => this.undo());
        $("btn-redo").addEventListener("click", () => this.redo());
        $("btn-zoom-fit").addEventListener("click", () => this.timelineView.zoomToFit());
        $("btn-camera-reset").addEventListener("click", () => this.viewer.focusModel());
        const frameInput = $<HTMLInputElement>("frame-input");
        frameInput.addEventListener("change", () => {
            const f = Math.round(Number(frameInput.value));
            if (Number.isFinite(f)) this.seek(f);
        });
        const speed = $<HTMLSelectElement>("speed-select");
        speed.addEventListener("change", () => {
            const v = Number(speed.value);
            this.viewer.timeScale = v;
            if (this.player) this.player.playbackRate = v;
            this.doc.view.playbackRate = v;
        });
        this.bindVolume();
        const zoom = $<HTMLInputElement>("zoom-range");
        const toPpf = (v: number) => 0.05 * Math.pow(480, v / 100);
        const fromPpf = (p: number) => (Math.log(p / 0.05) / Math.log(480)) * 100;
        zoom.addEventListener("input", () => this.timelineView.setZoom(toPpf(Number(zoom.value))));
        this.timelineView.onZoomChange = (ppf) => {
            zoom.value = String(Math.round(fromPpf(ppf)));
        };
        $<HTMLInputElement>("chk-physics").addEventListener("change", (e) => {
            const on = (e.target as HTMLInputElement).checked;
            this.viewer.setPhysics(on);
            this.doc.view.physics = on;
        });
        $<HTMLInputElement>("chk-grid").addEventListener("change", (e) => {
            this.viewer.gridVisible = (e.target as HTMLInputElement).checked;
        });
        this.video.addEventListener("error", () => {
            if (this.doc.video) toast("動画を再生できませんでした。対応していない形式（コーデック）の可能性があります。", "error");
        });
        // ドラッグ＆ドロップ
        document.addEventListener("dragover", (e) => e.preventDefault());
        document.addEventListener("drop", (e) => {
            e.preventDefault();
            const file = e.dataTransfer?.files[0];
            const path = file ? window.mmt.pathForFile(file) : undefined;
            if (path) void this.openDroppedFile(path);
        });
    }

    private bindKeys(): void {
        window.addEventListener("keydown", (e) => {
            if (anyModalOpen()) return;
            const target = e.target as HTMLElement;
            const typing = target instanceof HTMLInputElement && target.type !== "checkbox" && target.type !== "range"
                || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
            const ctrl = e.ctrlKey || e.metaKey;
            const key = e.key.toLowerCase();
            if (ctrl && key === "z" && !e.shiftKey) { e.preventDefault(); this.undo(); return; }
            if (ctrl && (key === "y" || (key === "z" && e.shiftKey))) { e.preventDefault(); this.redo(); return; }
            if (ctrl && key === "s") { e.preventDefault(); void this.saveProject(e.shiftKey); return; }
            if (ctrl && key === "o") { e.preventDefault(); void this.openProject(); return; }
            if (ctrl && key === "n") { e.preventDefault(); void this.newProject(); return; }
            if (ctrl && key === "e") { e.preventDefault(); void this.exportVmd(); return; }
            if (typing) return;
            switch (e.key) {
                case " ": e.preventDefault(); void this.togglePlay(); break;
                case "ArrowLeft": e.preventDefault(); this.step(e.shiftKey ? -10 : -1); break;
                case "ArrowRight": e.preventDefault(); this.step(e.shiftKey ? 10 : 1); break;
                case "Home": e.preventDefault(); this.seek(0); break;
                case "End": e.preventDefault(); this.seek(this.durationFrames - 1); break;
                case "Delete": case "Backspace": if (this.timelineView.deleteSelection()) e.preventDefault(); break;
                case "s": case "S": if (!ctrl && this.timelineView.splitAtPlayhead()) e.preventDefault(); break;
                case "m": case "M": if (!ctrl) { e.preventDefault(); this.toggleMute(); } break;
                case "Escape": this.timelineView.clearSelection(); break;
            }
        });
    }

    /** 再生する音声（動画の音）の音量。プロジェクトではなくアプリの設定として覚えておく */
    private bindVolume(): void {
        const range = $<HTMLInputElement>("volume-range");
        const mute = $("btn-mute");
        try {
            const saved = JSON.parse(localStorage.getItem("mmt.volume") ?? "null") as { volume?: number; muted?: boolean } | null;
            if (typeof saved?.volume === "number") this.video.volume = Math.min(1, Math.max(0, saved.volume));
            this.video.muted = saved?.muted === true;
        } catch { /* 覚えていなければ初期値（100%） */ }
        const update = () => {
            const percent = Math.round(this.video.volume * 100);
            range.value = String(percent);
            range.title = `音量 ${percent}%`;
            const silent = this.video.muted || this.video.volume === 0;
            mute.textContent = silent ? "🔇" : percent < 50 ? "🔉" : "🔊";
            mute.title = this.video.muted ? "消音を解除（M）" : "消音（M）";
            try {
                localStorage.setItem("mmt.volume", JSON.stringify({ volume: this.video.volume, muted: this.video.muted }));
            } catch { /* 覚えられなくても再生には関係ない */ }
        };
        range.addEventListener("input", () => {
            this.video.volume = Number(range.value) / 100;
            this.video.muted = false;
            update();
        });
        // つまみを動かした後も、← → でフレームを送れるようにする
        range.addEventListener("change", () => range.blur());
        mute.addEventListener("click", () => this.toggleMute());
        this.video.addEventListener("volumechange", update);
        update();
    }

    private toggleMute(): void {
        // 音量 0 のまま消音を解除したときは、聞こえる音量に戻す
        if (!this.video.muted && this.video.volume === 0) this.video.volume = 0.5;
        else this.video.muted = !this.video.muted;
    }

    private bindSplitters(): void {
        const splitV = $("splitter-v");
        const videoPane = $("video-pane");
        const splitH = $("splitter-h");
        const timelinePane = $("timeline-pane");
        const drag = (el: HTMLElement, onMove: (e: MouseEvent) => void) => {
            el.addEventListener("mousedown", (e) => {
                e.preventDefault();
                el.classList.add("dragging");
                const move = (ev: MouseEvent) => onMove(ev);
                const up = () => {
                    el.classList.remove("dragging");
                    window.removeEventListener("mousemove", move);
                    window.removeEventListener("mouseup", up);
                };
                window.addEventListener("mousemove", move);
                window.addEventListener("mouseup", up);
            });
        };
        drag(splitV, (e) => {
            const w = Math.max(200, Math.min(window.innerWidth - 260, e.clientX));
            videoPane.style.flex = `0 0 ${w}px`;
        });
        drag(splitH, (e) => {
            const hh = Math.max(200, Math.min(window.innerHeight - 220, window.innerHeight - e.clientY - 24));
            timelinePane.style.flex = `0 0 ${hh}px`;
        });
    }

    // ================================================================ 状態の表示

    private onTick(frame: number, playing: boolean): void {
        if (playing) {
            const now = performance.now();
            if (now - this.lastFollow > 120) {
                this.lastFollow = now;
                this.updateTimeDisplay(frame);
                this.timelineView.follow(frame);
            }
        }
        const btn = $("btn-play");
        const label = playing ? "⏸" : "▶";
        if (btn.textContent !== label) btn.textContent = label;
    }

    private updateTimeDisplay(frame = this.currentFrame): void {
        const f = Math.round(frame);
        const input = $<HTMLInputElement>("frame-input");
        if (document.activeElement !== input) input.value = String(f);
        $("time-text").textContent = formatTime(f / MMD_FPS);
        $("duration-text").textContent = ` / ${formatTime(this.durationFrames / MMD_FPS)}`;
        const overlay = $("video-overlay");
        const clip = this.timeline.clips.find((c) => f >= c.start && f <= c.end);
        overlay.classList.toggle("hidden", !this.doc.video);
        overlay.textContent = `${f} f${clip ? "  ● キャプチャ済み" : ""}`;
    }

    private setStatus(text: string): void {
        $("status-text").textContent = text;
    }

    private get dirty(): boolean {
        return this.docDirty || this.savedHistoryRef !== this.history.current;
    }

    private updateTitle(): void {
        const name = this.filePath ? basename(this.filePath) : "無題";
        const title = `${name}${this.dirty ? " *" : ""}`;
        $("title-text").textContent = title;
        window.mmt.setTitle(`${title} - MonoCam Motion Trace`);
    }

    private refreshAll(): void {
        const hasVideo = !!this.doc.video;
        const hasModel = !!this.doc.model;
        $("video-empty").classList.toggle("hidden", hasVideo);
        $("model-empty").classList.toggle("hidden", hasModel);
        $("video-empty-text").textContent = hasModel ? "動画を読み込むと、ここに表示されます" : "モデルを読み込んでから、動画を読み込んでください";
        ($<HTMLButtonElement>("btn-capture")).disabled = !hasVideo || this.captureRunning;
        $("video-info").textContent = hasVideo
            ? `${this.doc.video!.name}（${this.doc.video!.info.width}×${this.doc.video!.info.height}・${formatTime(this.doc.video!.info.duration)}）`
            : "";
        $("model-info").textContent = this.doc.model ? this.doc.model.displayName : "";
        const captured = this.doc.takes.length;
        $("video-capture-hint").textContent = hasVideo
            ? captured ? `キャプチャ済み ${captured} 区間。別の区間を追加でキャプチャできます` : "区間を選んでモーションを抽出します"
            : "";
        ($<HTMLButtonElement>("btn-export")).disabled = !(this.timeline.clips.length || this.hasHandKeys());
        ($<HTMLButtonElement>("btn-undo")).disabled = !this.history.canUndo;
        ($<HTMLButtonElement>("btn-redo")).disabled = !this.history.canRedo;
        $("btn-undo").title = this.history.undoLabel ? `元に戻す: ${this.history.undoLabel}（Ctrl+Z）` : "元に戻す（Ctrl+Z）";
        $("btn-redo").title = this.history.redoLabel ? `やり直す: ${this.history.redoLabel}（Ctrl+Y）` : "やり直す（Ctrl+Y）";
        this.updateTitle();
        this.updateTimeDisplay();
        this.updateTimelineStatus();
        this.viewer.setDuration(this.durationFrames);
        this.timelineView.refresh();
    }

    private updateTimelineStatus(): void {
        const el = $("timeline-status");
        const busy = [...this.takes.values()].filter((t) => t.busy).length;
        el.textContent = !busy && !this.timeline.clips.length && this.doc.video ? "「キャプチャ…」でモーションを抽出します" : "";
        // 変換中はステータスバーの右に出す
        const chip = $("status-engine");
        chip.classList.toggle("hidden", !busy);
        chip.replaceChildren(...(busy ? [h("div", { class: "spinner" }), `足ピタ・設定の変更を反映しています（${busy}）…`] : []));
    }

    private hasHandKeys(): boolean {
        return this.timeline.handKeys.left.length + this.timeline.handKeys.right.length > 0;
    }

    // ================================================================ タイムラインの変化

    private onTimelineChanged(): void {
        this.refreshAll();
        this.scheduleCompose();
        this.scheduleReconcile();
    }

    private undo(): void {
        const e = this.history.undo();
        if (e) this.setStatus(`元に戻しました: ${e.label}`);
    }

    private redo(): void {
        const e = this.history.redo();
        if (e) this.setStatus(`やり直しました: ${e.label}`);
    }

    private scheduleCompose(): void {
        clearTimeout(this.composeTimer);
        this.composeTimer = window.setTimeout(() => this.compose(), 30);
    }

    private loadedTakes(): Map<string, TakeResult> {
        const map = new Map<string, TakeResult>();
        for (const [id, rt] of this.takes) if (rt.result) map.set(id, rt.result);
        return map;
    }

    private compose(): void {
        const motion = composeMotion({
            clips: this.timeline.clips,
            takes: this.loadedTakes(),
            handKeys: this.timeline.handKeys,
            handTable: this.handTable,
            transitionFrames: this.transitionFrames,
            lipsyncEnabled: this.timeline.lipsyncEnabled,
            options: { body: true, fingers: true, face: true }
        });
        this.viewer.setMotion(motion);
        this.timelineView.invalidate();
    }

    private updateHandTable(): void {
        const m = this.modelHandTable;
        if (m && m.rigPath === this.doc.model?.rigPath && (m.rotations.left || m.rotations.right)) {
            this.handTable = m.rotations;
            this.transitionFrames = Math.max(1, Math.round(m.transitionSec * MMD_FPS));
            return;
        }
        // エンジンから取れないときは、撮影の変換結果の表（変換した時点の形の一覧）を使う
        for (const rt of this.takes.values()) {
            const hands = rt.result?.json.hands;
            if (hands?.rotations && (hands.rotations.left || hands.rotations.right)) {
                this.handTable = hands.rotations;
                this.transitionFrames = Math.max(1, Math.round((hands.transitionSec ?? 0.15) * MMD_FPS));
                return;
            }
        }
        this.handTable = null;
    }

    /**
     * 今のモデルの形ごとの指の回転をエンジンに作ってもらう。撮影の変換結果の表は変換した時点の形の一覧なので、
     * あとから増えた形が無いことがある（無い形はデフォルトの形になる）。
     */
    private async loadHandRotations(): Promise<void> {
        const rigPath = this.doc.model?.rigPath;
        if (!rigPath) return;
        try {
            const res = await this.engine.call<{ rotations: Partial<HandRotationTable>; transitionSec: number }>("hand_rotations", { rigPath });
            if (this.doc.model?.rigPath !== rigPath) return;
            this.modelHandTable = { rigPath, rotations: res.rotations, transitionSec: res.transitionSec ?? 0.15 };
            this.updateHandTable();
            this.scheduleCompose();
        } catch (e) {
            console.warn("指の回転を作れませんでした", e);
        }
    }

    // ================================================================ 撮影の変換（足ピタ・設定・モデルの変更の反映）

    /** 撮影の今の変換の設定（タイムラインで変えたものがあればそれ） */
    private conversionOf(take: Take) {
        return this.timeline.conversions?.[take.id] ?? take.options.conversion;
    }

    private takeSignature(take: Take): string {
        return JSON.stringify({
            footKeys: this.timeline.footKeys[take.id] ?? [],
            settings: engineSettings(this.conversionOf(take)),
            rig: this.doc.model ? `${this.doc.model.path}|${this.doc.model.rigPath}` : null
        });
    }

    private scheduleReconcile(delay = 250): void {
        clearTimeout(this.reconcileTimer);
        this.reconcileTimer = window.setTimeout(() => void this.reconcile(), delay);
    }

    /** 今の状態と違う条件で変換した撮影を、順に変換し直す。 */
    private async reconcile(): Promise<void> {
        if (this.reconciling || this.captureRunning || this.openingProject) return;
        this.reconciling = true;
        try {
            for (;;) {
                const used = new Set(this.timeline.clips.map((c) => c.takeId));
                const take = this.doc.takes.find((t) => used.has(t.id) && this.takes.get(t.id)?.loadedSignature !== this.takeSignature(t)
                    && !this.takes.get(t.id)?.error);
                if (!take) break;
                await this.convertTake(take);
            }
        } finally {
            this.reconciling = false;
            this.updateTimelineStatus();
        }
    }

    private async convertTake(take: Take, force = false): Promise<boolean> {
        const rt = this.takes.get(take.id) ?? { result: null, busy: false, loadedSignature: null, error: null };
        this.takes.set(take.id, rt);
        const signature = this.takeSignature(take);
        rt.busy = true;
        this.updateTimelineStatus();
        this.timelineView.invalidate();
        try {
            const req = await this.engine.start<{ resultPath: string; key: string }>("convert", {
                takeDir: `${this.workDir}/${take.dir}`,
                rigPath: this.doc.model?.rigPath ?? null,
                settings: engineSettings(this.conversionOf(take)),
                footKeys: this.timeline.footKeys[take.id] ?? [],
                force
            });
            const res = await req.result;
            const result = await loadTakeResult(this.workDir, take, res.key, res.resultPath);
            rt.result = result;
            rt.loadedSignature = signature;
            rt.error = null;
            take.resultKey = res.key;
            this.updateHandTable();
            this.scheduleCompose();
            return true;
        } catch (e) {
            rt.error = errorMessage(e);
            rt.loadedSignature = signature;
            toast(`モーションを変換できませんでした: ${rt.error}`, "error");
            return false;
        } finally {
            rt.busy = false;
            this.updateTimelineStatus();
            this.timelineView.invalidate();
        }
    }

    private async openReconvertDialog(takeId: string): Promise<void> {
        const take = this.doc.takes.find((t) => t.id === takeId);
        if (!take) return;
        const form = conversionForm(this.conversionOf(take));
        let ok = false;
        const modal = new Modal({
            title: "変換の設定を変えて再計算",
            size: "wide",
            body: h("div", null,
                h("p", { class: "form-help" }, `区間 ${formatTime(take.start)} 〜 ${formatTime(take.end)} の撮影を、この設定でもう一度 MMD のモーションに変換します（動画の解析はやり直しません。元に戻すこともできます）。`),
                form.el),
            buttons: [{ label: "キャンセル" }, { label: "再計算", kind: "primary", isDefault: true, onClick: () => { ok = true; } }]
        }).open();
        await modal.closedPromise;
        if (!ok) return;
        const settings = form.read();
        this.doc.conversion = settings;
        const rt = this.takes.get(take.id);
        if (rt) rt.error = null;
        this.commit(setConversion(this.timeline, take.id, settings), "変換の設定の変更");
    }

    // ================================================================ モデル

    private async loadModelDialog(): Promise<void> {
        const path = await window.mmt.showOpenDialog("モデルを読み込む", MODEL_FILTERS);
        if (path) await this.loadModel(path);
    }

    private async loadModel(path: string, existing?: ModelRef): Promise<boolean> {
        const loading = $("viewer-loading");
        loading.classList.remove("hidden");
        this.setStatus(`モデルを読み込んでいます: ${basename(path)}`);
        try {
            const loaded = await this.viewer.loadModel(path, window.mmt.mediaUrl(path));
            this.viewer.setPhysics(($<HTMLInputElement>("chk-physics")).checked);
            const missing = missingBones(loaded.object);
            if (missing.length) {
                toast(`変換に必要なボーンがモデルにありません: ${missing.join("、")}\n標準的な MMD のボーン構造のモデルを使ってください。`, "warn", 10000);
            }
            if (existing) {
                this.doc.model = existing;
            } else {
                const rigPath = await window.mmt.writeRig(this.workDir, JSON.stringify(extractRig(loaded.object)));
                this.doc.model = await window.mmt.importModel(this.workDir, path, referencedFiles(loaded.object), rigPath, loaded.name);
                this.docDirty = true;
            }
            this.setStatus(`モデルを読み込みました: ${loaded.name}`);
            this.handTable = null;
            void this.loadHandRotations();
            for (const rt of this.takes.values()) rt.error = null;
            this.refreshAll();
            this.scheduleCompose();
            if (!existing) this.scheduleReconcile(0);
            return true;
        } catch (e) {
            await alertDialog("モデルを読み込めませんでした", `${basename(path)}\n${errorMessage(e)}`, "error");
            return false;
        } finally {
            loading.classList.add("hidden");
        }
    }

    // ================================================================ 動画

    private async loadVideoDialog(): Promise<void> {
        if (!this.doc.model) {
            const choice = await choiceDialog("モデルを先に読み込んでください",
                "モーションはモデルの体格（脚の長さ・腕の太さ・剛体など）に合わせて作ります。先にモデル（PMX / PMD）を読み込むのがおすすめです。",
                ["モデルを読み込む", "モデルなしで続ける", "キャンセル"], 0);
            if (choice === 0) {
                await this.loadModelDialog();
                if (!this.doc.model) return;
            } else if (choice !== 1) return;
        }
        const path = await window.mmt.showOpenDialog("動画を読み込む", VIDEO_FILTERS);
        if (!path) return;
        if (this.doc.video && this.doc.takes.length) {
            const ok = await confirmDialog("動画を置き換えますか？",
                "今の動画からキャプチャしたモーションはタイムラインに残りますが、新しい動画とは時刻が合わなくなります。", "置き換える");
            if (!ok) return;
        }
        if (await this.loadVideo(path)) await this.openCaptureForCurrent();
    }

    private async probeVideo(path: string): Promise<VideoInfo> {
        const status = await window.mmt.setupStatus();
        if (status.runtimeReady) {
            try {
                return await this.engine.call<VideoInfo>("probe_video", { path });
            } catch {
                /* 下の方法で */
            }
        }
        // 実行環境がまだ無い: 動画の要素から読む（fps は分からないので 30 とみなし、キャプチャの前に調べ直す）
        return new Promise((resolve, reject) => {
            const v = document.createElement("video");
            v.preload = "metadata";
            v.onloadedmetadata = () => {
                resolve({ fps: 30, duration: v.duration, width: v.videoWidth, height: v.videoHeight, has_audio: true, rotation: 0, autoFps: 30 });
                v.removeAttribute("src");
                v.load();
            };
            v.onerror = () => reject(new Error("動画を読めませんでした（対応していない形式の可能性があります）"));
            v.src = window.mmt.mediaUrl(path);
        });
    }

    private async loadVideo(path: string, existingInfo?: VideoInfo): Promise<boolean> {
        try {
            this.setStatus(`動画を読み込んでいます: ${basename(path)}`);
            const info = existingInfo ?? await this.probeVideo(path);
            if (!(info.duration > 0)) throw new Error("動画の長さを読めませんでした");
            this.player?.dispose();
            this.video.src = window.mmt.mediaUrl(path);
            this.player = new VideoPlayer(this.video);
            this.player.playbackRate = Number(($<HTMLSelectElement>("speed-select")).value);
            await this.viewer.setPlayer(this.player);
            if (!existingInfo) {
                this.doc.video = { path, name: basename(path), info };
                this.waveformData = null;
                this.doc.waveform = null;
                this.docDirty = true;
            }
            this.refreshAll();
            this.timelineView.zoomToFit();
            this.setStatus(`動画を読み込みました: ${basename(path)}`);
            return true;
        } catch (e) {
            await alertDialog("動画を読み込めませんでした", `${basename(path)}\n${errorMessage(e)}`, "error");
            return false;
        }
    }

    private async loadWaveform(): Promise<void> {
        const wf = this.doc.waveform;
        if (!wf || !(await window.mmt.fileExists(wf.path))) {
            this.waveformData = null;
            return;
        }
        const buf = await window.mmt.readFile(wf.path);
        this.waveformData = { peaks: new Float32Array(buf), rate: wf.rate };
        this.timelineView.invalidate();
    }

    // ================================================================ キャプチャ

    private async openCaptureForCurrent(): Promise<void> {
        if (!this.doc.video) {
            await this.loadVideoDialog();
            return;
        }
        const f = this.currentFrame / MMD_FPS;
        const dur = this.doc.video.info.duration;
        const start = this.timeline.clips.length ? Math.min(f, dur) : 0;
        const end = this.timeline.clips.length ? Math.min(dur, start + 30) : Math.min(dur, 30);
        const initial: CaptureOptions = { ...defaultCapture(start, end), conversion: structuredClone(this.doc.conversion) };
        const last = this.doc.takes[this.doc.takes.length - 1];
        if (last) Object.assign(initial, { ...last.options, start, end });
        await this.startCapture(initial);
    }

    private async startCapture(initial: CaptureOptions, title?: string): Promise<void> {
        if (!this.doc.video || this.captureRunning) return;
        const options = await captureDialog({
            videoPath: this.doc.video.path, info: this.doc.video.info, initial, gpu: this.gpu,
            hasModel: !!this.doc.model, capturedRanges: this.doc.takes.map((t) => [t.start, t.end]), title
        });
        if (!options) return;
        if (!(await ensureReady())) {
            this.setStatus("キャプチャを中止しました");
            return;
        }
        const status = await window.mmt.setupStatus();
        this.gpu = status.gpu ?? null;
        // 実行環境ができてから、動画の情報（fps・音声の有無）を調べ直す
        try {
            const info = await this.engine.call<VideoInfo>("probe_video", { path: this.doc.video.path });
            this.doc.video.info = info;
            if (!info.has_audio) options.lipsync = false;
        } catch (e) {
            await alertDialog("動画を読めませんでした", errorMessage(e), "error");
            return;
        }
        await this.runCapture(options);
    }

    private async runCapture(options: CaptureOptions): Promise<void> {
        if (!this.doc.video) return;
        this.captureRunning = true;
        this.viewer.pause();
        this.refreshAll();
        const dir = await window.mmt.newTakeDir(this.workDir);
        let request: { cancel: () => void } | null = null;
        let cancelled = false;
        const dialog = new ProgressDialog("モーションのキャプチャ", {
            showLog: true,
            onCancel: () => { cancelled = true; request?.cancel(); },
            note: `区間 ${formatTime(options.start)} 〜 ${formatTime(options.end)}（${(options.end - options.start).toFixed(1)} 秒）`
        });
        dialog.message("準備しています…");
        // 波形は動画ごとに別のファイルにする（エンジンは波形のファイルが無いときだけ作るので、同じ名前のままだと
        // 動画を置き換えた後も前の動画の波形が使われてしまう）。今の動画の波形があれば、それを使い回す
        const waveformPath = this.doc.waveform?.path ?? `${this.workDir}/audio/waveform-${dir.id}.f32`;
        try {
            const req = await this.engine.start<{ resultPath: string; key: string; capture: { frames: number; hands: boolean; lipsync: boolean; validRatio: number; precision: string } }>(
                "capture", {
                    takeDir: dir.abs,
                    video: this.doc.video.path,
                    start: options.start,
                    end: options.end,
                    fps: options.fps,
                    maxHeight: options.maxHeight,
                    crop: options.crop ?? null,
                    personSelect: options.personSelect,
                    batchSize: options.batchSize,
                    numAug: options.numAug,
                    antialias: options.antialias,
                    detectorThreshold: options.detectorThreshold,
                    device: options.device,
                    hands: { enabled: options.hands },
                    lipsync: { enabled: options.lipsync, demucs: options.demucs, lang: options.lipsyncLang },
                    rigPath: this.doc.model?.rigPath ?? null,
                    waveformPath,
                    settings: engineSettings(options.conversion)
                },
                (p) => {
                    if (p.message) dialog.message(p.message);
                    dialog.progress(p.value);
                    dialog.detail(p.stage ? `段階: ${p.stage}` : "");
                },
                (l) => dialog.log(l.message, l.level)
            );
            request = req;
            if (cancelled) req.cancel();
            const res = await req.result;
            const json = JSON.parse(await window.mmt.readTextFile(res.resultPath)) as ConversionResultJson;
            const startFrame = Math.round(options.start * MMD_FPS);
            const take: Take = {
                id: dir.id, dir: dir.dir, start: options.start, end: options.end, startFrame, frames: json.frames,
                capturedAt: new Date().toISOString(), options, hasHands: !!res.capture.hands, hasLipsync: !!res.capture.lipsync,
                resultKey: res.key
            };
            const result = await loadTakeResult(this.workDir, take, res.key, res.resultPath);
            this.doc.takes.push(take);
            this.doc.conversion = structuredClone(options.conversion);
            if (await window.mmt.fileExists(waveformPath)) {
                this.doc.waveform = { path: waveformPath, rate: WAVEFORM_RATE };
                await this.loadWaveform();
            }
            this.takes.set(take.id, { result, busy: false, loadedSignature: null, error: null });
            this.docDirty = true;
            this.updateHandTable();
            const next = addTake(this.timeline, take.id, startFrame, startFrame + json.frames - 1, "full", json.hands.events ?? null);
            // 変換の条件の記録（足ピタのキーは無し）
            this.takes.get(take.id)!.loadedSignature = JSON.stringify({
                footKeys: [], settings: engineSettings(options.conversion),
                rig: this.doc.model ? `${this.doc.model.path}|${this.doc.model.rigPath}` : null
            });
            this.commit(next, "キャプチャ");
            this.seek(startFrame);
            const notes = [`${json.frames} フレームのモーションを作成しました`];
            if (res.capture.validRatio < 0.9) notes.push(`人物を検出できたフレームは ${Math.round(res.capture.validRatio * 100)}% でした`);
            if (res.capture.precision === "fp32") notes.push("この GPU では単精度で推論しました");
            toast(notes.join("\n"), "ok", 6000);
            for (const w of json.warnings ?? []) toast(w, "warn", 9000);
        } catch (e) {
            if (e instanceof EngineRequestError && e.cancelled) {
                this.setStatus("キャプチャを中止しました");
            } else {
                const detail = e instanceof EngineRequestError && e.error.detail ? `\n\n詳細はエンジンのログを確認してください（ファイル → エンジンのログを開く）。` : "";
                await alertDialog("キャプチャに失敗しました", errorMessage(e) + detail, "error");
            }
        } finally {
            dialog.close();
            this.captureRunning = false;
            this.refreshAll();
            this.scheduleReconcile();
        }
    }

    // ================================================================ 再生

    private async togglePlay(): Promise<void> {
        if (this.viewer.playing) this.viewer.pause();
        else {
            if (this.currentFrame >= this.durationFrames - 1) this.seek(0);
            try {
                await this.viewer.play();
            } catch (e) {
                toast(`再生できませんでした: ${errorMessage(e)}`, "error");
            }
        }
    }

    private step(delta: number): void {
        this.viewer.pause();
        this.seek(Math.round(this.currentFrame) + delta);
    }

    // ================================================================ プロジェクト

    private async confirmDiscard(action: string): Promise<boolean> {
        if (!this.dirty) return true;
        const choice = await choiceDialog("変更を保存しますか？", `プロジェクトの変更が保存されていません。${action}前に保存しますか？`,
            ["保存する", "保存しない", "キャンセル"], 0);
        if (choice === 0) return this.saveProject(false);
        return choice === 1;
    }

    private async newProject(): Promise<void> {
        if (this.captureRunning) return;
        if (!(await this.confirmDiscard("新しく作成する"))) return;
        await this.resetTo(null);
        this.setStatus("新しいプロジェクトを作成しました");
    }

    private async resetTo(opened: OpenedProject | null): Promise<void> {
        const oldDir = this.workDir;
        this.viewer.pause();
        this.viewer.unloadModel();
        await this.viewer.setPlayer(null);
        this.player?.dispose();
        this.player = null;
        this.video.removeAttribute("src");
        this.video.load();
        this.takes.clear();
        this.handTable = null;
        this.modelHandTable = null;
        this.waveformData = null;
        if (opened) {
            this.workDir = opened.workDir;
            this.filePath = opened.filePath;
            this.doc = opened.document;
            this.doc.timeline = normalizeTimeline(this.doc.timeline);
        } else {
            this.workDir = (await window.mmt.newProject()).workDir;
            this.filePath = null;
            this.doc = this.emptyDocument();
        }
        if (oldDir && oldDir !== this.workDir) void window.mmt.discardProject(oldDir);
        this.history.reset(this.doc.timeline);
        this.savedHistoryRef = this.history.current;
        this.docDirty = false;
        this.refreshAll();
    }

    private async openProject(path?: string): Promise<void> {
        if (this.captureRunning) return;
        if (!(await this.confirmDiscard("ほかのプロジェクトを開く"))) return;
        const file = path ?? await window.mmt.showOpenDialog("プロジェクトを開く", PROJECT_FILTERS);
        if (!file) return;
        const dialog = new ProgressDialog("プロジェクトを開く");
        const off = window.mmt.onProjectProgress((p) => {
            dialog.message(p.message);
            dialog.progress(p.value);
        });
        this.openingProject = true;
        try {
            const opened = await window.mmt.openProject(file);
            await this.resetTo(opened);
            const doc = this.doc;
            ($<HTMLSelectElement>("speed-select")).value = String(doc.view.playbackRate ?? 1);
            this.viewer.timeScale = doc.view.playbackRate ?? 1;
            ($<HTMLInputElement>("chk-physics")).checked = doc.view.physics !== false && this.viewer.physicsAvailable;
            dialog.message("モデルを読み込んでいます");
            dialog.progress(null);
            if (doc.model) await this.loadModel(doc.model.path, doc.model);
            if (doc.video) await this.loadVideo(doc.video.path, doc.video.info);
            await this.loadWaveform();
            dialog.message("モーションを読み込んでいます");
            for (const take of doc.takes) {
                const rt: TakeRuntime = { result: null, busy: false, loadedSignature: null, error: null };
                this.takes.set(take.id, rt);
                if (!take.resultKey) continue;
                try {
                    rt.result = await loadTakeResult(this.workDir, take, take.resultKey);
                    // 以前の版の足ピタ（区間の上書き）で作った結果は、キーで作り直す
                    const legacy = (rt.result.json as { overrides?: unknown[] }).overrides;
                    // キーの掛け方が前の版の結果も作り直す
                    const json = rt.result.json;
                    const madeWith = json.footKeys?.length && json.footKeysVersion !== FOOT_KEYS_VERSION ? null
                        : json.footKeys ?? (legacy?.length ? null : []);
                    if (JSON.stringify(madeWith) === JSON.stringify(this.timeline.footKeys[take.id] ?? [])) {
                        rt.loadedSignature = this.takeSignature(take);
                    }
                } catch {
                    rt.result = null;   // 結果が無い（古い版で保存した等）: 変換し直す
                }
            }
            this.openingProject = false;
            this.updateHandTable();
            this.refreshAll();
            this.scheduleCompose();
            this.scheduleReconcile(0);
            this.timelineView.zoomToFit();
            this.setStatus(`開きました: ${basename(file)}`);
        } catch (e) {
            await alertDialog("プロジェクトを開けませんでした", errorMessage(e), "error");
        } finally {
            this.openingProject = false;
            off();
            dialog.close();
        }
    }

    private async saveProject(saveAs: boolean): Promise<boolean> {
        if (this.captureRunning) {
            toast("キャプチャ中は保存できません", "warn");
            return false;
        }
        let path = this.filePath;
        if (saveAs || !path) {
            const name = this.filePath ? basename(this.filePath) : `${this.doc.video ? this.doc.video.name.replace(/\.[^.]+$/, "") : "無題"}.mmt`;
            path = await window.mmt.showSaveDialog("名前を付けて保存", name, PROJECT_FILTERS);
            if (!path) return false;
            if (!/\.mmt$/i.test(path)) path += ".mmt";
        }
        // タイムラインで使っている撮影だけを保存する（キャプチャし直して使わなくなった撮影は入れない）
        const used = new Set(this.timeline.clips.map((c) => c.takeId));
        this.doc.appVersion = this.version;
        this.doc.view.zoom = this.timelineView.pixelsPerFrame;
        this.doc.timeline = pruneTakes(this.timeline, used);
        const toSave: ProjectDocument = { ...this.doc, takes: this.doc.takes.filter((t) => used.has(t.id)) };
        const dialog = new ProgressDialog("プロジェクトの保存");
        const off = window.mmt.onProjectProgress((p) => {
            dialog.message(p.message);
            dialog.progress(p.value);
        });
        try {
            await window.mmt.saveProject(this.workDir, path, toSave);
            this.filePath = path;
            this.savedHistoryRef = this.history.current;
            this.docDirty = false;
            this.updateTitle();
            this.setStatus(`保存しました: ${basename(path)}`);
            return true;
        } catch (e) {
            await alertDialog("保存できませんでした", errorMessage(e), "error");
            return false;
        } finally {
            off();
            dialog.close();
        }
    }

    private async openDroppedFile(path: string): Promise<void> {
        const ext = path.split(".").pop()?.toLowerCase() ?? "";
        if (ext === "mmt") await this.openProject(path);
        else if (ext === "pmx" || ext === "pmd") await this.loadModel(path);
        else if (["mp4", "mov", "webm", "m4v"].includes(ext)) {
            if (await this.loadVideo(path)) await this.openCaptureForCurrent();
        } else toast("読み込めない種類のファイルです", "warn");
    }

    private async requestClose(): Promise<void> {
        if (this.captureRunning) {
            const ok = await confirmDialog("キャプチャ中です", "キャプチャを中止して終了しますか？", "終了する");
            if (!ok) return;
        }
        if (!(await this.confirmDiscard("終了する"))) return;
        // 作業フォルダの動画を開いたままだと、Windows ではフォルダを消せないので、先に手放す
        this.viewer.pause();
        this.video.removeAttribute("src");
        this.video.load();
        // 消せなくても終了する（残ったものは、ウィンドウを閉じた後と次の起動時に消す）
        await window.mmt.discardProject(this.workDir).catch(() => undefined);
        window.mmt.confirmClose();
    }

    // ================================================================ VMD 出力

    private async exportVmd(): Promise<void> {
        if (!this.timeline.clips.length && !this.hasHandKeys()) {
            toast("出力するモーションがありません。動画をキャプチャしてください。", "warn");
            return;
        }
        if ([...this.takes.values()].some((t) => t.busy)) {
            toast("足ピタ・設定の変更を反映しています。終わってから出力してください。", "warn");
            return;
        }
        const clips = sortedClips(this.timeline.clips);
        const takes = this.loadedTakes();
        const hasFace = clips.some((c) => takes.get(c.takeId)?.json.lipsync.available && this.timeline.lipsyncEnabled[c.takeId] !== false);
        const firstFrame = clips.length ? clips[0].start : 0;
        const lastFrame = clips.length ? clips[clips.length - 1].end : this.durationFrames - 1;
        const modelName = this.doc.model?.displayName ?? "MonoCam Motion Trace";
        const opts = await exportDialog({ hasFingers: !!this.handTable && this.hasHandKeys(), hasFace, firstFrame, lastFrame, modelName });
        if (!opts) return;
        const base = this.doc.video ? this.doc.video.name.replace(/\.[^.]+$/, "") : "motion";
        const path = await window.mmt.showSaveDialog("VMD 出力", `${base}.vmd`, VMD_FILTERS);
        if (!path) return;
        try {
            const { buffer, boneKeys, morphKeys, truncated } = this.buildVmd(opts, firstFrame, modelName);
            await window.mmt.writeFile(/\.vmd$/i.test(path) ? path : `${path}.vmd`, buffer);
            toast(`VMD を書き出しました: ${basename(path)}\nボーンのキー ${boneKeys}・モーフのキー ${morphKeys}`, "ok", 6000);
            if (truncated.length) toast(`名前が 15 バイトを超えるので切り詰めました: ${truncated.join("、")}`, "warn", 9000);
        } catch (e) {
            await alertDialog("VMD を書き出せませんでした", errorMessage(e), "error");
        }
    }

    private buildVmd(opts: ExportOptions, firstFrame: number, modelName: string) {
        let motion = composeMotion({
            clips: this.timeline.clips,
            takes: this.loadedTakes(),
            handKeys: this.timeline.handKeys,
            handTable: this.handTable,
            transitionFrames: this.transitionFrames,
            lipsyncEnabled: this.timeline.lipsyncEnabled,
            options: { body: opts.body, fingers: opts.fingers, face: opts.face }
        });
        if (opts.startAtZero) motion = shiftMotion(motion, firstFrame);
        if (opts.thin) motion = thinMotion(motion);
        const r = writeVmd(motion, modelName);
        return { buffer: r.buffer, boneKeys: r.boneKeys, morphKeys: r.morphKeys, truncated: r.truncatedNames };
    }

    // ================================================================ その他

    private async openEngineLog(): Promise<void> {
        const path = await window.mmt.engineLogPath();
        if (!(await window.mmt.fileExists(path))) {
            toast("まだログがありません", "info");
            return;
        }
        const text = await window.mmt.readTextFile(path);
        const area = h("textarea", { readonly: true, rows: 22, style: "width: 100%" }, text.slice(-60000));
        new Modal({
            title: "エンジンのログ", size: "xwide",
            body: h("div", null, h("p", { class: "form-help" }, path), area),
            buttons: [{ label: "閉じる", kind: "primary", isDefault: true }]
        }).open();
        setTimeout(() => (area.scrollTop = area.scrollHeight), 0);
    }

    /** 実行環境（Python・PyTorch など）を消して入れ直し、学習済みモデルを照合する（壊れて動かなくなったとき用） */
    private async reinstallRuntime(): Promise<void> {
        if (this.captureRunning) return;
        if ([...this.takes.values()].some((t) => t.busy)) {
            toast("足ピタ・設定の変更を反映しています。終わってから再セットアップしてください。", "warn");
            return;
        }
        const ok = await confirmDialog("実行環境を再セットアップしますか？",
            "Python・PyTorch などの実行環境を削除してインストールし直し、学習済みモデルが壊れていないかを確かめます（壊れていればダウンロードし直します）。\n"
            + "キャプチャ・変換が失敗し続けるときにお試しください。約 3〜4 GB をダウンロードするので、時間がかかります。", "再セットアップ");
        if (!ok) return;
        if (await ensureReady(true)) {
            this.gpu = (await window.mmt.setupStatus()).gpu ?? null;
            toast("実行環境の再セットアップが完了しました", "ok", 6000);
        }
    }

    private async reportIssue(): Promise<void> {
        const status = await window.mmt.setupStatus();
        const gpu = status.gpu ? `${status.gpu.name}（ドライバー ${status.gpu.driver}・CUDA ${status.gpu.cudaVersion ?? "不明"}）` : "なし";
        const body = [
            "## 不具合の内容", "", "（何をしたら、何が起きたか）", "",
            "## 再現する手順", "", "1. ", "",
            "## 環境",
            `- MonoCam Motion Trace: ${this.version}`,
            `- OS: ${navigator.userAgent.match(/Windows NT [\d.]+/)?.[0] ?? navigator.platform}`,
            `- GPU: ${gpu}`,
            `- PyTorch: ${status.runtimeVariant ?? "未セットアップ"}`, "",
            "（エラーが出た場合は、ファイル → エンジンのログを開く の内容の最後の部分を貼り付けてください）"
        ].join("\n");
        const url = `${APP_REPOSITORY}/issues/new?title=${encodeURIComponent("[不具合] ")}&body=${encodeURIComponent(body)}`;
        await window.mmt.openExternal(url);
    }
}
