// メインプロセスとレンダラーで共有する型。

/** MMD のフレームレート（タイムラインの 1 フレーム） */
export const MMD_FPS = 30;

export const APP_REPOSITORY = "https://github.com/yamak493/MMT";

// ---------------------------------------------------------------- 動画・モデル

export interface VideoInfo {
    fps: number;
    duration: number;
    width: number;
    height: number;
    has_audio: boolean;
    rotation: number;
    autoFps: number;
}

export type MotionVariant = "full" | "locked" | "no_move" | "upper_body";

export const VARIANT_LABELS: Record<MotionVariant, string> = {
    full: "フル",
    locked: "フル [接地優先]",
    no_move: "フル [移動なし]",
    upper_body: "上半身のみ"
};

// ---------------------------------------------------------------- キャプチャ・変換の設定

/** 変換の設定のうち、画面で変えられるもの（エンジンの settings.py の上書きになる） */
export interface ConversionSettings {
    centerMode: "A" | "B";
    depthScale: number;
    flipFacing: boolean;
    armCollision: "left" | "right" | "none";
    bodyParts: { clothes: boolean; head: boolean; torso: boolean; chest: boolean; legs: boolean };
    handReach: boolean;
    clothMarginCm: number;
    wristFromHands: boolean;
    legKeys: boolean;
    groundEnabled: boolean;
    leanEnabled: boolean;
    lockedBothFeet: boolean;
    lipsyncMorphs: { a: string; i: string; u: string; e: string; o: string; n: string };
    lipsyncStrength: number;
    handSwitchCost: number;
}

/** 動画を切り抜く範囲（回転を適用した映像の画素） */
export interface CropRect {
    x: number;
    y: number;
    w: number;
    h: number;
}

export type CropMode = "none" | "square" | "free";

export interface CaptureOptions {
    start: number;
    end: number;
    fps: "auto" | "source" | number;
    maxHeight: number;
    /** 推論する前に切り抜く範囲（無い・null なら映像の全体。古いプロジェクトには無い） */
    crop?: CropRect | null;
    cropMode?: CropMode;
    personSelect: "largest" | "center";
    batchSize: number;
    numAug: number;
    antialias: number;
    detectorThreshold: number;
    hands: boolean;
    lipsync: boolean;
    demucs: boolean;
    lipsyncLang: "jpn" | "eng" | "ipa";
    conversion: ConversionSettings;
}

/** 足ピタのキーの掛け方の版（エンジンの foot_keys.FOOT_KEYS_VERSION と同じ値） */
export const FOOT_KEYS_VERSION = 2;

/** 足ピタのキー。同じ足のキーを 2 つずつ組にし、その間は足裏を床に着けて固定する（前後のフレームでなじませる） */
export interface FootKey {
    foot: 0 | 1;
    frame: number;   // 撮影（take）の中のフレーム
}

/** 以前の版の足ピタ（接地の区間の上書き）。プロジェクトを開くときに FootKey に直す */
export interface ContactOverride {
    foot: 0 | 1;
    start: number;   // 撮影（take）の中のフレーム
    end: number;
    mode: "on" | "off";
}

// ---------------------------------------------------------------- 変換の結果（エンジンの result.json）

export interface BoneLayout {
    name: string;
    movable: boolean;
    offset: number;
}

export interface VariantLayout {
    file: string;
    frames: number;
    bones: BoneLayout[];
}

export interface HandEvent {
    frame: number;
    shape: string;
}

export type HandRotationTable = Record<"left" | "right", Record<string, Record<string, number[]>>>;

export interface LipTrack {
    name: string;
    shape: string;
    frames: number[];
    weights: number[];
}

export interface ConversionResultJson {
    key: string;
    engineVersion: string;
    frames: number;
    fps: number;
    modelName: string;
    contacts: { auto: [number, number][][]; effective: [number, number][][] };
    jumps: [number, number][];
    variants: Partial<Record<MotionVariant, VariantLayout>>;
    variantsInfo: Record<string, unknown>;
    hands: {
        available: boolean;
        visible?: number[];
        events?: { left: HandEvent[]; right: HandEvent[] };
        rotations?: Partial<HandRotationTable>;
        shapes?: string[];
        labels?: Record<string, string>;
        transitionSec?: number;
    };
    lipsync: { available: boolean; tracks?: LipTrack[]; info?: Record<string, unknown> };
    info: Record<string, unknown>;
    warnings: string[];
    footKeys?: FootKey[];
    /** 足ピタのキーの掛け方の版（エンジンの FOOT_KEYS_VERSION） */
    footKeysVersion?: number;
}

// ---------------------------------------------------------------- エンジン・セットアップ

export interface EngineProgress {
    job: number;
    stage?: string;
    value: number;
    message?: string | null;
}

export interface EngineLog {
    job?: number;
    level: "info" | "warn" | "error";
    message: string;
}

export interface EngineError {
    kind: "cancelled" | "user" | "memory" | "internal" | "protocol" | "engine";
    message: string;
    detail?: string;
}

export interface SetupStatus {
    /** 実行環境（Python・PyTorch など）が使える状態か */
    runtimeReady: boolean;
    /** NLF の学習済みモデルがあるか */
    modelsReady: boolean;
    /** 利用規約に同意したか（同意した版） */
    termsAccepted: boolean;
    runtimeVariant?: string;
    gpu?: GpuInfo | null;
}

export interface GpuInfo {
    name: string;
    driver: string;
    cudaVersion: string | null;
    memoryMb: number | null;
}

export interface SetupProgress {
    phase: "runtime" | "packages" | "models" | "verify" | "done";
    message: string;
    value: number | null;   // 0〜1。null は割合が分からない
    detail?: string;
}

// ---------------------------------------------------------------- プロジェクト

export interface ProjectFileRef {
    /** 今の場所（作業フォルダの中か、外部のファイル） */
    path: string;
    name: string;
}

export interface ModelRef extends ProjectFileRef {
    /** モデルのフォルダの中で、プロジェクトに入れるファイル（PMX と、参照しているテクスチャ） */
    files: string[];
    rigPath: string;
    displayName: string;
}

export interface VideoRef extends ProjectFileRef {
    info: VideoInfo;
}

export interface Take {
    id: string;
    /** 作業フォルダからの相対パス（takes/<id>） */
    dir: string;
    start: number;
    end: number;
    startFrame: number;
    frames: number;
    capturedAt: string;
    options: CaptureOptions;
    hasHands: boolean;
    hasLipsync: boolean;
    /** 今の足ピタの上書きで変換した結果のキー（takes/<id>/conv/<key>/result.json） */
    resultKey: string | null;
}

export interface MotionClip {
    id: string;
    takeId: string;
    start: number;   // タイムラインのフレーム（含む）
    end: number;     // 含む
    variant: MotionVariant;
}

export interface HandKey {
    frame: number;
    shape: string;
}

/** 履歴（元に戻す・やり直す）の対象になるタイムラインの状態 */
export interface TimelineState {
    clips: MotionClip[];
    /** 撮影ごとの足ピタのキー */
    footKeys: Record<string, FootKey[]>;
    /** 以前の版の足ピタ（読み込みのときだけ） */
    footOverrides?: Record<string, ContactOverride[]>;
    handKeys: { left: HandKey[]; right: HandKey[] };
    lipsyncEnabled: Record<string, boolean>;
    /** 撮影ごとの変換の設定（「変換の設定を変えて再計算」で変えたもの。無ければキャプチャのときの設定） */
    conversions?: Record<string, ConversionSettings>;
}

export interface ProjectDocument {
    format: "mmt-project";
    version: 1;
    appVersion: string;
    savedAt: string;
    model: ModelRef | null;
    video: VideoRef | null;
    waveform: { path: string; rate: number } | null;
    takes: Take[];
    timeline: TimelineState;
    conversion: ConversionSettings;
    view: { playbackRate: number; zoom: number; physics: boolean };
}

export interface OpenedProject {
    workDir: string;
    filePath: string | null;
    document: ProjectDocument;
}

// ---------------------------------------------------------------- メイン ↔ レンダラーの API

export interface DialogFilter {
    name: string;
    extensions: string[];
}

export interface MmtApi {
    platform: string;
    appVersion: () => Promise<string>;

    // ウィンドウ・外部
    openExternal: (url: string) => Promise<void>;
    closeWindow: () => void;
    minimizeWindow: () => void;
    toggleMaximize: () => void;
    setTitle: (title: string) => void;
    onCloseRequested: (cb: () => void) => () => void;
    onOpenFile: (cb: (path: string) => void) => () => void;
    /** 準備ができたことを知らせ、それまでに開くよう頼まれたプロジェクト（起動の引数など）を受け取る（無ければ null） */
    takeOpenFile: () => Promise<string | null>;
    confirmClose: () => void;

    // ファイル
    showOpenDialog: (title: string, filters: DialogFilter[]) => Promise<string | null>;
    showSaveDialog: (title: string, defaultName: string, filters: DialogFilter[]) => Promise<string | null>;
    readFile: (path: string) => Promise<ArrayBuffer>;
    readTextFile: (path: string) => Promise<string>;
    writeFile: (path: string, data: ArrayBuffer | string) => Promise<void>;
    fileExists: (path: string) => Promise<boolean>;
    mediaUrl: (path: string) => string;
    /** ドロップしたファイルのパス */
    pathForFile: (file: File) => string;

    // プロジェクト
    newProject: () => Promise<{ workDir: string }>;
    openProject: (filePath: string) => Promise<OpenedProject>;
    saveProject: (workDir: string, filePath: string, doc: ProjectDocument) => Promise<ProjectDocument>;
    importModel: (workDir: string, modelPath: string, files: string[], rigPath: string, displayName: string) => Promise<ModelRef>;
    writeRig: (workDir: string, rigJson: string) => Promise<string>;
    newTakeDir: (workDir: string) => Promise<{ id: string; dir: string; abs: string }>;
    discardProject: (workDir: string) => Promise<void>;
    onProjectProgress: (cb: (p: { message: string; value: number | null }) => void) => () => void;

    // セットアップ・エンジン
    setupStatus: () => Promise<SetupStatus>;
    acceptTerms: () => Promise<void>;
    /** 足りないものを入れる。force なら実行環境を消して入れ直す */
    runSetup: (force?: boolean) => Promise<SetupStatus>;
    cancelSetup: () => Promise<void>;
    onSetupProgress: (cb: (p: SetupProgress) => void) => () => void;
    onSetupLog: (cb: (line: string) => void) => () => void;
    engineRequest: (method: string, params: Record<string, unknown>) => Promise<number>;
    engineCancel: (job: number) => Promise<void>;
    onEngineEvent: (cb: (e: EngineEvent) => void) => () => void;
    engineLogPath: () => Promise<string>;
    modelsDir: () => Promise<string>;
}

export type EngineEvent =
    | { type: "progress"; data: EngineProgress }
    | { type: "log"; data: EngineLog }
    | { type: "result"; id: number; result: unknown }
    | { type: "error"; id: number; error: EngineError }
    | { type: "status"; running: boolean; message?: string };
