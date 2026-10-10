// アプリが使うフォルダ。実行環境・学習済みモデルは大きいので、ユーザーごとのアプリのデータのフォルダに置く。
import { app } from "electron";
import { existsSync } from "node:fs";
import { join } from "node:path";

export function userDataDir(): string {
    return app.getPath("userData");
}

/** Python の実行環境（uv・Python・仮想環境）の置き場所 */
export function runtimeDir(): string {
    return join(userDataDir(), "runtime");
}

export function venvDir(): string {
    return join(runtimeDir(), "venv");
}

/** 開発用: MMT_PYTHON に既存の Python（エンジンの依存ライブラリを入れたもの）を指定すると、それを使う */
export function devPython(): string | null {
    return process.env.MMT_PYTHON || null;
}

export function venvPython(): string {
    return devPython() ?? join(venvDir(), "Scripts", "python.exe");
}

export function uvExe(): string {
    return join(runtimeDir(), "uv", "uv.exe");
}

/** 学習済みモデル（NLF・MediaPipe・Demucs）の置き場所 */
export function modelsDir(): string {
    return process.env.MMT_MODELS_DIR || join(userDataDir(), "models");
}

export function nlfModelPath(): string {
    return join(modelsDir(), "nlf", "nlf_l_multi_0.3.2.torchscript");
}

export function handModelPath(): string {
    return join(modelsDir(), "mediapipe", "hand_landmarker.task");
}

/** 開いているプロジェクトの作業フォルダ */
export function projectsDir(): string {
    return join(userDataDir(), "projects");
}

export function logsDir(): string {
    return join(userDataDir(), "logs");
}

/** エンジン（Python のパッケージ mmt_engine）の場所。開発中はリポジトリの engine/、配布版は resources/engine */
export function engineDir(): string {
    const packaged = join(process.resourcesPath ?? "", "engine");
    if (app.isPackaged && existsSync(packaged)) return packaged;
    return join(app.getAppPath(), "engine");
}

export function rendererDir(): string {
    return join(app.getAppPath(), "dist", "renderer");
}
