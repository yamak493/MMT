// メインプロセス: ウィンドウ・独自スキーム・プロジェクトのファイル・セットアップ・エンジンを扱い、画面と IPC でつなぐ。
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";
import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { DialogFilter, EngineEvent, ProjectDocument, SetupProgress } from "../shared/types";
import { EngineProcess } from "./engine";
import { modelsDir, userDataDir } from "./paths";
import * as project from "./project";
import { handleSchemes, mediaUrl, registerSchemes } from "./protocol";
import { loadSettings, saveSettings } from "./settings";
import { acceptTerms, Setup, setupStatus } from "./setup";
import { detectGpu } from "./gpu";

const isDev = process.argv.includes("--dev");
// --smoke=<png>: 起動して画面を撮り、コンソールのエラーを標準出力へ書いて終了する（ビルドの確認用）
const smokeArg = process.argv.find((a) => a.startsWith("--smoke="));
let win: BrowserWindow | null = null;
let closeConfirmed = false;
let setup: Setup | null = null;
/** 画面の準備ができる前に開くよう頼まれたプロジェクト（起動の引数・2 つ目の起動）。画面が init の後に取りに来る */
let pendingOpenFile: string | null = projectArg(process.argv);
/** 画面が準備を終えて、開くファイルを受け取れる状態か */
let rendererReady = false;
const engine = new EngineProcess((e: EngineEvent) => win?.webContents.send("engine:event", e));

// 開発・動作確認用: MMT_USER_DATA でアプリのデータの場所（実行環境・モデル・設定）を変える
if (process.env.MMT_USER_DATA) app.setPath("userData", process.env.MMT_USER_DATA);
registerSchemes();
// 予期しない例外でメインプロセスが落ちないようにし、ログに残す
process.on("uncaughtException", (err) => logMainError("uncaughtException", err));
process.on("unhandledRejection", (err) => logMainError("unhandledRejection", err));

function logMainError(kind: string, err: unknown): void {
    try {
        const dir = join(userDataDir(), "logs");
        mkdirSync(dir, { recursive: true });
        appendFileSync(join(dir, "main.log"), `${new Date().toISOString()} ${kind}: ${(err as Error)?.stack ?? String(err)}\n`);
    } catch {
        /* ログを書けなくても続ける */
    }
}
app.setAppUserModelId("com.yamak493.monocam-motion-tracer");
// 2 つ目の起動は、1 つ目のウィンドウにファイルを渡して（second-instance）すぐ終わる。
// app.quit() の後も whenReady は呼ばれるので、下の起動の処理（前回の作業フォルダの掃除など）は 1 つ目だけで行う。
// 2 つ目が掃除をすると、1 つ目の保存していないキャプチャを消してしまう
const isPrimaryInstance = app.requestSingleInstanceLock();
if (!isPrimaryInstance) {
    app.quit();
}

const HEADER_COLOR = "#2b2f36";

function createWindow(): void {
    win = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 960,
        minHeight: 640,
        title: "MonoCam Motion Trace",
        backgroundColor: "#1e2126",
        show: false,
        titleBarStyle: "hidden",
        titleBarOverlay: { color: HEADER_COLOR, symbolColor: "#e6e8eb", height: 34 },
        icon: join(app.getAppPath(), "build", "icon.png"),
        webPreferences: {
            preload: join(__dirname, "preload.js"),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true,
            spellcheck: false,
            devTools: true
        }
    });
    Menu.setApplicationMenu(null);
    win.once("ready-to-show", () => win?.show());
    // 読み込み直したときは、画面がもう一度準備を終えるまでファイルを送らない
    win.webContents.on("did-start-loading", () => {
        rendererReady = false;
    });
    if (smokeArg) runSmoke(win, smokeArg.slice("--smoke=".length));
    win.on("close", (e) => {
        if (closeConfirmed) return;
        e.preventDefault();
        win?.webContents.send("app:close-requested");
    });
    win.on("closed", () => {
        win = null;
    });
    // 画面の外へのナビゲーション・新しいウィンドウは既定のブラウザで開く
    win.webContents.setWindowOpenHandler(({ url }) => {
        if (/^https?:\/\//.test(url)) void shell.openExternal(url);
        return { action: "deny" };
    });
    win.webContents.on("will-navigate", (e, url) => {
        if (!url.startsWith("app://")) {
            e.preventDefault();
            if (/^https?:\/\//.test(url)) void shell.openExternal(url);
        }
    });
    win.webContents.on("before-input-event", (_e, input) => {
        if (input.type === "keyDown" && input.key === "F12" && (isDev || input.control)) win?.webContents.toggleDevTools();
    });
    void win.loadURL("app://mmt/index.html");
}

function runSmoke(w: BrowserWindow, out: string): void {
    const errors: string[] = [];
    w.webContents.on("console-message", (e) => {
        if (e.level === "error" || e.level === "warning") errors.push(`[${e.level}] ${e.message}`);
    });
    w.webContents.on("did-finish-load", () => {
        setTimeout(async () => {
            const script = process.env.MMT_SMOKE_SCRIPT;
            if (script) {
                try {
                    const r = await w.webContents.executeJavaScript(script, true);
                    console.log("SMOKE_RESULT", JSON.stringify(r));
                } catch (e) {
                    errors.push(`[script] ${(e as Error).message}`);
                }
            }
            const img = await w.webContents.capturePage();
            await writeFile(out, img.toPNG());
            console.log("SMOKE_ERRORS", JSON.stringify(errors));
            closeConfirmed = true;
            app.exit(0);
        }, Number(process.env.MMT_SMOKE_WAIT ?? 4000));
    });
}

/** 起動の引数（.mmt のダブルクリック）で渡されたプロジェクトのパス */
function projectArg(argv: string[]): string | null {
    return argv.slice(1).find((a) => /\.mmt$/i.test(a) && !a.startsWith("--")) ?? null;
}

function send(channel: string, payload: unknown): void {
    win?.webContents.send(channel, payload);
}

/** プロジェクトを開く。画面の準備がまだなら、準備を終えた画面が app:take-open-file で取りに来るまで預かる */
function requestOpenFile(file: string): void {
    if (rendererReady && win) send("app:open-file", file);
    else pendingOpenFile = file;
}

function registerIpc(): void {
    ipcMain.handle("app:version", () => app.getVersion());
    // 画面が準備（新しいプロジェクトの作業フォルダを作るなど）を終えてから呼ぶ。以後のファイルは app:open-file で送る
    ipcMain.handle("app:take-open-file", () => {
        rendererReady = true;
        const file = pendingOpenFile;
        pendingOpenFile = null;
        return file;
    });
    ipcMain.handle("shell:open-external", (_e, url: string) => {
        if (/^https:\/\//.test(url)) return shell.openExternal(url);
    });
    ipcMain.on("win:close", () => win?.close());
    ipcMain.on("win:minimize", () => win?.minimize());
    ipcMain.on("win:toggle-maximize", () => (win?.isMaximized() ? win.unmaximize() : win?.maximize()));
    ipcMain.on("win:set-title", (_e, title: string) => win?.setTitle(title));
    ipcMain.on("app:confirm-close", () => {
        closeConfirmed = true;
        setup?.cancel();
        engine.stop();
        win?.close();
    });

    // ---- ファイル ----
    ipcMain.handle("dialog:open", async (_e, title: string, filters: DialogFilter[]) => {
        const key = filters[0]?.extensions.includes("mmt") ? "lastProjectDir"
            : filters[0]?.extensions.some((x) => ["pmx", "pmd"].includes(x)) ? "lastModelDir" : "lastVideoDir";
        const settings = loadSettings();
        const r = await dialog.showOpenDialog(win!, {
            title, filters, properties: ["openFile"], defaultPath: settings[key] ?? undefined
        });
        if (r.canceled || !r.filePaths[0]) return null;
        saveSettings({ [key]: dirname(r.filePaths[0]) });
        return r.filePaths[0];
    });
    ipcMain.handle("dialog:save", async (_e, title: string, defaultName: string, filters: DialogFilter[]) => {
        const settings = loadSettings();
        const base = settings.lastProjectDir ?? app.getPath("documents");
        const r = await dialog.showSaveDialog(win!, { title, filters, defaultPath: join(base, defaultName) });
        if (r.canceled || !r.filePath) return null;
        saveSettings({ lastProjectDir: dirname(r.filePath) });
        return r.filePath;
    });
    ipcMain.handle("fs:read", async (_e, path: string) => {
        const buf = await readFile(path);
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    });
    ipcMain.handle("fs:read-text", (_e, path: string) => readFile(path, "utf-8"));
    ipcMain.handle("fs:write", async (_e, path: string, data: ArrayBuffer | string) => {
        await mkdir(dirname(path), { recursive: true });
        await writeFile(path, typeof data === "string" ? data : Buffer.from(data));
    });
    ipcMain.handle("fs:exists", (_e, path: string) => existsSync(path));
    ipcMain.on("media:url", (e, path: string) => {
        e.returnValue = mediaUrl(path);
    });

    // ---- プロジェクト ----
    const progress = (message: string, value: number | null) => send("project:progress", { message, value });
    ipcMain.handle("project:new", () => {
        const workDir = project.createWorkDir();
        return { workDir, filePath: null, document: null };
    });
    ipcMain.handle("project:open", (_e, filePath: string) => project.openProject(filePath, progress));
    ipcMain.handle("project:save", (_e, workDir: string, filePath: string, doc: ProjectDocument) =>
        project.saveProject(workDir, filePath, doc, progress));
    ipcMain.handle("project:import-model", (_e, _workDir: string, modelPath: string, files: string[], rigPath: string, name: string) =>
        project.modelRef(modelPath, files, rigPath, name));
    ipcMain.handle("project:write-rig", (_e, workDir: string, rigJson: string) => project.writeRig(workDir, rigJson));
    ipcMain.handle("project:new-take", (_e, workDir: string) => project.newTakeDir(workDir));
    ipcMain.handle("project:discard", (_e, workDir: string) => project.discardWorkDir(workDir));

    // ---- セットアップ ----
    ipcMain.handle("setup:status", async () => ({ ...(await setupStatus()), gpu: await detectGpu() }));
    ipcMain.handle("setup:accept-terms", () => acceptTerms());
    // force: 実行環境を消して入れ直す（「実行環境を再セットアップ」）
    ipcMain.handle("setup:run", async (_e, force?: boolean) => {
        if (setup) throw new Error("セットアップはすでに実行中です");
        setup = new Setup((p: SetupProgress) => send("setup:progress", p), (line) => send("setup:log", line), {
            // 実行環境（venv）を作り直す前に、それを使っているエンジンを止める（使用中の python.exe は消せない）。
            // 作り直している間は、エンジンを起動させない
            beforeRuntimeInstall: async () => {
                engine.block("実行環境を準備しています。準備が終わってから、もう一度お試しください");
                await engine.shutdown();
            }
        });
        try {
            await setup.run(force === true);
        } finally {
            engine.unblock();
            setup = null;
        }
        return { ...(await setupStatus()), gpu: await detectGpu() };
    });
    ipcMain.handle("setup:cancel", () => setup?.cancel());

    // ---- エンジン ----
    ipcMain.handle("engine:request", (_e, method: string, params: Record<string, unknown>) =>
        engine.request(method, { modelsDir: modelsDir(), ...params }));
    ipcMain.handle("engine:cancel", (_e, job: number) => engine.cancel(job));
    ipcMain.handle("engine:log-path", () => engine.logPath());
    ipcMain.handle("app:models-dir", () => modelsDir());
    ipcMain.handle("app:user-data", () => userDataDir());
}

if (isPrimaryInstance) {
    app.whenReady().then(() => {
        handleSchemes();
        project.cleanupWorkDirs();
        registerIpc();
        createWindow();
    });

    app.on("second-instance", (_e, argv) => {
        const file = projectArg(argv);
        if (file) requestOpenFile(file);
        if (win) {
            if (win.isMinimized()) win.restore();
            win.focus();
        }
    });

    app.on("window-all-closed", () => {
        engine.stop();
        project.discardLeftoverWorkDirs();
        app.quit();
    });
}
