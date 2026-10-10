// アプリの設定（利用規約への同意・実行環境の状態など）。userData/settings.json に保存する。
import { readFileSync, renameSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { userDataDir } from "./paths";

/** 利用規約の版。内容を変えたら上げる（もう一度同意してもらう） */
export const TERMS_VERSION = 1;
/** 実行環境の版。必要なライブラリを変えたら上げる（セットアップをやり直す） */
export const RUNTIME_VERSION = 1;

export interface AppSettings {
    termsAcceptedVersion: number;
    termsAcceptedAt: string | null;
    runtimeVersion: number;
    runtimeVariant: string | null;
    lastProjectDir: string | null;
    lastVideoDir: string | null;
    lastModelDir: string | null;
}

const DEFAULTS: AppSettings = {
    termsAcceptedVersion: 0,
    termsAcceptedAt: null,
    runtimeVersion: 0,
    runtimeVariant: null,
    lastProjectDir: null,
    lastVideoDir: null,
    lastModelDir: null
};

let cache: AppSettings | null = null;

function path(): string {
    return join(userDataDir(), "settings.json");
}

export function loadSettings(): AppSettings {
    if (cache) return cache;
    try {
        cache = { ...DEFAULTS, ...JSON.parse(readFileSync(path(), "utf-8")) };
    } catch {
        cache = { ...DEFAULTS };
    }
    return cache!;
}

export function saveSettings(update: Partial<AppSettings>): AppSettings {
    const next = { ...loadSettings(), ...update };
    cache = next;
    mkdirSync(dirname(path()), { recursive: true });
    const tmp = path() + ".tmp";
    writeFileSync(tmp, JSON.stringify(next, null, 2), "utf-8");
    renameSync(tmp, path());
    return next;
}
