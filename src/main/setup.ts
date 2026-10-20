// 実行環境のセットアップ: uv（Python のパッケージ管理）・Python 3.11・PyTorch・エンジンの依存ライブラリと、
// 学習済みモデル（NLF・MediaPipe Hands）をダウンロードして入れる。利用規約に同意してから行う。
import { spawn, type ChildProcess } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statfsSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import type { SetupProgress, SetupStatus } from "../shared/types";
import { download, formatBytes } from "./downloads";
import { detectGpu } from "./gpu";
import { devPython, engineDir, handModelPath, modelsDir, nlfModelPath, runtimeDir, userDataDir, uvExe, venvDir, venvPython } from "./paths";
import { loadSettings, RUNTIME_VERSION, saveSettings, TERMS_VERSION } from "./settings";
import { extractZip } from "./zip";

const UV_VERSION = "0.12.12";
const UV_URL = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-pc-windows-msvc.zip`;
/** 配布元の uv-x86_64-pc-windows-msvc.zip.sha256 の値。UV_VERSION を変えたら合わせる */
const UV_SHA256 = "3d54912924c36e862c14f427d04f2ed70a99e8001d1c30caa101f6d5711626d5";
const PYTHON_VERSION = "3.11.9";
/** Visual C++ ランタイムの DLL（msvcp140.dll など。Microsoft が配布を認めているもの）を収めた wheel */
const MSVC_RUNTIME_PACKAGE = "msvc-runtime==14.44.35112";
/** 版を変えたら engine/requirements.in の torch 等も合わせ、requirements.txt を作り直す */
const TORCH_PACKAGES =["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"];
/**
 * 入れる PyTorch の種類（CUDA 12.8 版だけ。CPU 版は使わない）。2.7.1 の cu128 版は GTX 900 番台〜RTX 50 番台
 * （sm_50〜sm_120）に対応する。変えたら engine/mmt_engine/selftest.py の MIN_DRIVER_CUDA も合わせる
 */
const TORCH_BACKEND = "cu128";
export const NLF_URL = "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript";
const NLF_MIN_BYTES = 300 * 1024 ** 2;
const NLF_SHA256 = "52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad";
export const HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const HAND_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1";
/** 実行環境のインストールに必要なディスクの空き（README の「約 16 GB」） */
const RUNTIME_FREE_BYTES = 16 * 1024 ** 3;
/** 学習済みモデル（NLF・単精度版・口パクのモデル）に必要なディスクの空き */
const MODELS_FREE_BYTES = 2 * 1024 ** 3;

function readyMarker(): string {
    return join(runtimeDir(), "ready.json");
}

function lipsyncMarker(): string {
    return join(modelsDir(), "lipsync.ready");
}

/** NLF のモデルからエンジンが作る単精度版（消しても、次のキャプチャで作り直す） */
function fp32ModelPath(): string {
    return join(dirname(nlfModelPath()), "nlf_l_multi_0.3.2.fp32.torchscript");
}

/** dir のあるドライブに bytes 以上の空きがあるか確かめる（調べられないときは確かめない） */
function ensureFreeSpace(dir: string, bytes: number, log: (line: string) => void): void {
    let free: number;
    try {
        mkdirSync(dir, { recursive: true });
        const st = statfsSync(dir);
        free = st.bavail * st.bsize;
    } catch (e) {
        log(`ディスクの空き容量を調べられませんでした（${(e as Error).message}）`);
        return;
    }
    if (free < bytes) {
        throw new Error(`ディスクの空き容量が足りません（必要: 約 ${formatBytes(bytes)}・空き: ${formatBytes(free)}）。\n`
            + `${dir} のあるドライブの空きを増やしてから、もう一度お試しください。`);
    }
}

export class SetupCancelled extends Error {
    constructor() {
        super("セットアップを中止しました");
    }
}

export async function setupStatus(): Promise<SetupStatus> {
    const settings = loadSettings();
    let runtimeReady = false;
    try {
        const marker = JSON.parse(readFileSync(readyMarker(), "utf-8"));
        // 以前の版が入れた CPU 版などの実行環境は、入れ直す
        runtimeReady = marker.version === RUNTIME_VERSION && marker.variant === TORCH_BACKEND && existsSync(venvPython());
    } catch {
        runtimeReady = false;
    }
    if (devPython()) runtimeReady = existsSync(venvPython());
    const modelsReady = existsSync(nlfModelPath()) && statSync(nlfModelPath()).size >= NLF_MIN_BYTES && existsSync(handModelPath());
    return {
        runtimeReady,
        modelsReady,
        termsAccepted: settings.termsAcceptedVersion >= TERMS_VERSION,
        runtimeVariant: settings.runtimeVariant ?? undefined
    };
}

export function acceptTerms(): void {
    saveSettings({ termsAcceptedVersion: TERMS_VERSION, termsAcceptedAt: new Date().toISOString() });
}

export interface SetupHooks {
    /** 実行環境（venv）を消して作り直す直前に呼ぶ（それを使っているエンジンを止めるため） */
    beforeRuntimeInstall?: () => Promise<void>;
}

export class Setup {
    private abort = new AbortController();
    private child: ChildProcess | null = null;
    private cancelled = false;

    constructor(
        private readonly report: (p: SetupProgress) => void,
        private readonly log: (line: string) => void,
        private readonly hooks: SetupHooks = {}
    ) {}

    cancel(): void {
        this.cancelled = true;
        this.abort.abort();
        if (this.child && this.child.pid) {
            // uv が起動した子プロセスもまとめて止める
            spawn("taskkill", ["/pid", String(this.child.pid), "/T", "/F"], { windowsHide: true });
        }
    }

    private check(): void {
        if (this.cancelled) throw new SetupCancelled();
    }

    /**
     * 足りないものを入れる。force なら、実行環境を消して入れ直し、単精度版の NLF・口パクのモデルも
     * 作り直す（学習済みモデルは SHA-256 を照合し、壊れていればダウンロードし直す）。
     */
    async run(force = false): Promise<void> {
        if (loadSettings().termsAcceptedVersion < TERMS_VERSION) {
            throw new Error("利用規約に同意していません");
        }
        if (force) await this.removeForReinstall();
        const status = await setupStatus();
        if (!status.runtimeReady) await this.installRuntime();
        this.check();
        await this.downloadModels();
        this.report({ phase: "done", message: "セットアップが完了しました", value: 1 });
    }

    /** 再セットアップの前に、実行環境と、単精度版の NLF・口パクのモデルを消す */
    private async removeForReinstall(): Promise<void> {
        if (devPython()) throw new Error("開発用の Python（MMT_PYTHON）を使っている間は、再セットアップできません");
        await this.hooks.beforeRuntimeInstall?.();
        this.check();
        this.report({ phase: "runtime", message: "今の実行環境を削除しています", value: null });
        try {
            for (const path of [runtimeDir(), fp32ModelPath(), lipsyncMarker(), join(modelsDir(), "torch")]) {
                rmSync(path, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
            }
        } catch (e) {
            throw new Error("実行環境を削除できませんでした（使用中のファイルがあります）。"
                + `アプリを起動し直してから、もう一度お試しください。\n${(e as Error).message}`);
        }
        this.log("実行環境を削除しました");
    }

    private async installRuntime(): Promise<void> {
        await this.hooks.beforeRuntimeInstall?.();
        this.check();
        mkdirSync(runtimeDir(), { recursive: true });
        rmSync(readyMarker(), { force: true });
        // ---- uv ----
        if (!existsSync(uvExe())) {
            const zip = join(runtimeDir(), "uv.zip");
            await download(UV_URL, zip, (p) => this.report({
                phase: "runtime", message: "パッケージ管理ツール（uv）をダウンロードしています",
                value: p.total ? p.received / p.total : null, detail: formatBytes(p.received)
            }), this.abort.signal, { sha256: UV_SHA256 });
            this.check();
            await extractZip(zip, join(runtimeDir(), "uv"), undefined, (name) => (name.endsWith("uv.exe") ? "uv.exe" : null));
            rmSync(zip, { force: true });
        }
        this.check();
        // ---- Python と仮想環境 ----
        if (existsSync(venvDir())) rmSync(venvDir(), { recursive: true, force: true });
        ensureFreeSpace(userDataDir(), RUNTIME_FREE_BYTES, this.log);
        this.report({ phase: "runtime", message: `Python ${PYTHON_VERSION} を準備しています`, value: null });
        // PC にすでにある Python（Microsoft Store 版・conda など）は使わず、uv が入れた Python に揃える
        await this.uv(["venv", "--python", PYTHON_VERSION, "--python-preference", "only-managed", "--seed", venvDir()]);
        this.check();
        // ---- NVIDIA のドライバーと GPU ----
        // PyTorch（約 2.5 GB）をダウンロードする前に、GPU 版が動くかを CUDA のドライバー API で確かめる
        this.report({ phase: "runtime", message: "NVIDIA の GPU とドライバーを確認しています", value: null });
        const driver = await this.engineJson<{ ok: boolean; error?: string; cudaVersion?: string; gpu?: string; capability?: string }>(["--driver-check"]);
        if (!driver.ok) throw new Error(driver.error ?? "NVIDIA の GPU とドライバーを確認できませんでした");
        this.log(`GPU: ${driver.gpu}（Compute Capability ${driver.capability}・ドライバーの CUDA ${driver.cudaVersion}）`);
        this.check();
        // ---- Visual C++ ランタイム ----
        // PyTorch は msvcp140.dll などを使うが、Visual C++ 再頒布可能パッケージが無い PC もあるので、Python の横に置く
        this.report({ phase: "runtime", message: "Visual C++ ランタイムを準備しています", value: null });
        await this.uv(["pip", "install", "--python", venvPython(), MSVC_RUNTIME_PACKAGE]);
        this.installMsvcRuntime();
        this.check();
        // ---- PyTorch（GPU 版）----
        const gpu = await detectGpu();
        this.report({
            phase: "packages",
            message: `PyTorch（GPU 版・${TORCH_BACKEND}）をインストールしています（約 2.5 GB。時間がかかります）`,
            value: null,
            detail: `GPU: ${driver.gpu}${gpu ? `（ドライバー ${gpu.driver}）` : ""}`
        });
        await this.uv(["pip", "install", "--python", venvPython(), ...TORCH_PACKAGES,
            "--index-url", `https://download.pytorch.org/whl/${TORCH_BACKEND}`]);
        this.check();
        // ---- エンジンの依存ライブラリ ----
        this.report({ phase: "packages", message: "エンジンの依存ライブラリをインストールしています", value: null });
        await this.uv(["pip", "install", "--python", venvPython(), "-r", join(engineDir(), "requirements.txt")]);
        this.check();
        // ---- 確認 ----
        this.report({ phase: "verify", message: "実行環境を確認しています", value: null });
        const report = await this.engineJson<{ ok: boolean; errors?: Record<string, string> }>(["--selftest"]);
        if (!report.ok) {
            const errors = Object.entries(report.errors ?? {}).map(([k, v]) => `${k}: ${v}`).join("\n");
            throw new Error(`実行環境の確認に失敗しました。\n${errors}`);
        }
        writeFileSync(readyMarker(), JSON.stringify({ version: RUNTIME_VERSION, variant: TORCH_BACKEND, at: new Date().toISOString(), report }, null, 2));
        saveSettings({ runtimeVersion: RUNTIME_VERSION, runtimeVariant: TORCH_BACKEND });
        // ダウンロードしたパッケージのキャッシュ（約 6 GB）は、インストールした後は使わないので消す
        this.report({ phase: "verify", message: "ダウンロードしたパッケージのキャッシュを削除しています", value: null });
        try {
            await this.uv(["cache", "clean"]);
        } catch (e) {
            if (e instanceof SetupCancelled) throw e;
            this.log(`uv のキャッシュを削除できませんでした（${(e as Error).message}）`);
        }
    }

    /**
     * msvc-runtime が仮想環境のフォルダに置いた DLL を、Python 本体（python.exe）のフォルダへコピーする。
     * 仮想環境の python.exe は本体の python.exe を起動するだけなので、DLL は本体のフォルダに無いと読み込まれない。
     */
    private installMsvcRuntime(): void {
        const cfg = readFileSync(join(venvDir(), "pyvenv.cfg"), "utf-8");
        const home = /^home\s*=\s*(.+)$/m.exec(cfg)?.[1].trim();
        if (!home) throw new Error("仮想環境の Python の場所が分かりません（pyvenv.cfg に home がありません）");
        // アプリが入れた Python 以外のフォルダには書き込まない
        if (!resolve(home).toLowerCase().startsWith(resolve(runtimeDir()).toLowerCase() + sep)) {
            this.log(`Python がアプリの実行環境の外にあるので、Visual C++ ランタイムをコピーしません: ${home}`);
            return;
        }
        const dlls = readdirSync(venvDir()).filter((name) => name.toLowerCase().endsWith(".dll"));
        if (!dlls.some((name) => name.toLowerCase() === "msvcp140.dll")) {
            throw new Error("Visual C++ ランタイム（msvcp140.dll）が見つかりません");
        }
        for (const name of dlls) {
            const dest = join(home, name);
            if (!existsSync(dest)) copyFileSync(join(venvDir(), name), dest);
        }
        this.log(`Visual C++ ランタイムを置きました: ${home}（${dlls.join(", ")}）`);
    }

    private async downloadModels(): Promise<void> {
        const nlf = nlfModelPath();
        if (!existsSync(nlf)) ensureFreeSpace(modelsDir(), MODELS_FREE_BYTES, this.log);
        // すでにあるモデルも中身を照合する（壊れていればダウンロードし直す）
        this.report({ phase: "models", message: "学習済みモデルを確認しています", value: null });
        const nlfBefore = existsSync(nlf) ? statSync(nlf).mtimeMs : null;
        await download(NLF_URL, nlf, (p) => this.report({
            phase: "models", message: "NLF の学習済みモデルをダウンロードしています",
            value: p.total ? p.received / p.total : null,
            detail: `${formatBytes(p.received)}${p.total ? ` / ${formatBytes(p.total)}` : ""}`
        }), this.abort.signal, { minBytes: NLF_MIN_BYTES, sha256: NLF_SHA256 });
        // ダウンロードし直したら、前のモデルから作った単精度版も作り直す
        if (nlfBefore !== null && statSync(nlf).mtimeMs !== nlfBefore) {
            rmSync(fp32ModelPath(), { force: true });
        }
        this.check();
        await download(HAND_URL, handModelPath(), (p) => this.report({
            phase: "models", message: "MediaPipe Hands のモデルをダウンロードしています",
            value: p.total ? p.received / p.total : null, detail: formatBytes(p.received)
        }), this.abort.signal, { minBytes: 1024 * 1024, sha256: HAND_SHA256 });
        this.check();
        // 口パクのモデル（Demucs・Allosaurus）。失敗しても、キャプチャのときにもう一度ダウンロードを試す
        const marker = lipsyncMarker();
        if (!existsSync(marker)) {
            this.report({ phase: "models", message: "口パクのモデル（Demucs・Allosaurus）をダウンロードしています", value: null });
            const code = await this.runEngine(["--prefetch"]);
            if (code === 0) writeFileSync(marker, new Date().toISOString());
            else this.log("口パクのモデルをダウンロードできませんでした（キャプチャのときにもう一度試します）");
        }
    }

    private engineEnv(): NodeJS.ProcessEnv {
        return {
            ...process.env, PYTHONPATH: engineDir(), PYTHONUTF8: "1", PYTHONUNBUFFERED: "1",
            MMT_MODELS_DIR: modelsDir(), TORCH_HOME: join(modelsDir(), "torch"), VIRTUAL_ENV: ""
        };
    }

    private runEngine(args: string[]): Promise<number> {
        return new Promise((resolve) => {
            const child = spawn(venvPython(), ["-m", "mmt_engine", ...args], { cwd: engineDir(), windowsHide: true, env: this.engineEnv() });
            this.child = child;
            const onData = (b: Buffer) => b.toString("utf-8").split(/\r?\n/).filter((l) => l.trim()).forEach((l) => this.log(l));
            child.stdout.on("data", onData);
            child.stderr.on("data", onData);
            child.on("error", () => resolve(-1));
            child.on("close", (code) => {
                this.child = null;
                resolve(this.cancelled ? -1 : code ?? -1);
            });
        });
    }

    private uv(args: string[]): Promise<void> {
        return new Promise((resolve, reject) => {
            const env = {
                ...process.env,
                UV_PYTHON_INSTALL_DIR: join(runtimeDir(), "python"),
                UV_CACHE_DIR: join(runtimeDir(), "uv-cache"),
                UV_NO_PROGRESS: "1",
                UV_LINK_MODE: "copy",
                // TLS の証明書を OS（Windows の証明書ストア）から読む。ウイルス対策ソフトの HTTPS の検査や社内のプロキシが
                // 独自のルート証明書で通信を中継している環境でも、ダウンロードに失敗しないようにする。
                // uv 0.12 で UV_NATIVE_TLS から改名された（旧名は非推奨で、将来の版で削除される）
                UV_SYSTEM_CERTS: "1",
                VIRTUAL_ENV: ""
            };
            this.log(`> uv ${args.join(" ")}`);
            const child = spawn(uvExe(), args, { env, windowsHide: true });
            this.child = child;
            let tail = "";
            const onData = (buf: Buffer) => {
                const text = buf.toString("utf-8");
                tail = (tail + text).slice(-4000);
                for (const line of text.split(/\r?\n/)) {
                    if (line.trim()) this.log(line);
                }
            };
            child.stdout.on("data", onData);
            child.stderr.on("data", onData);
            child.on("error", reject);
            child.on("close", (code) => {
                this.child = null;
                if (this.cancelled) reject(new SetupCancelled());
                else if (code === 0) resolve();
                else reject(new Error(`インストールに失敗しました（終了コード ${code}）。\n${tail.trim().split(/\r?\n/).slice(-8).join("\n")}`));
            });
        });
    }

    /** エンジンを起動し、標準出力の最後の行の JSON を返す（--selftest・--driver-check） */
    private engineJson<T>(args: string[]): Promise<T> {
        return new Promise((resolve, reject) => {
            const child = spawn(venvPython(), ["-m", "mmt_engine", ...args], {
                cwd: engineDir(), windowsHide: true, env: this.engineEnv()
            });
            this.child = child;
            let out = "";
            child.stdout.on("data", (b: Buffer) => (out += b.toString("utf-8")));
            child.stderr.on("data", (b: Buffer) => this.log(b.toString("utf-8")));
            child.on("error", reject);
            child.on("close", () => {
                this.child = null;
                try {
                    resolve(JSON.parse(out.trim().split(/\r?\n/).pop() ?? "{}"));
                } catch {
                    reject(new Error(`実行環境を確認できませんでした:\n${out.slice(-2000)}`));
                }
            });
        });
    }
}

export { modelsDir };
