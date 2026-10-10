// プリロード: 画面（レンダラー）に window.mmt として、メインプロセスの機能を限られた形で渡す。
import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";
import type { DialogFilter, EngineEvent, MmtApi, ProjectDocument, SetupProgress } from "../shared/types";

function listen<T>(channel: string, cb: (payload: T) => void): () => void {
    const handler = (_e: IpcRendererEvent, payload: T) => cb(payload);
    ipcRenderer.on(channel, handler);
    return () => ipcRenderer.removeListener(channel, handler);
}

const api: MmtApi = {
    platform: process.platform,
    appVersion: () => ipcRenderer.invoke("app:version"),

    openExternal: (url) => ipcRenderer.invoke("shell:open-external", url),
    closeWindow: () => ipcRenderer.send("win:close"),
    minimizeWindow: () => ipcRenderer.send("win:minimize"),
    toggleMaximize: () => ipcRenderer.send("win:toggle-maximize"),
    setTitle: (title) => ipcRenderer.send("win:set-title", title),
    onCloseRequested: (cb) => listen("app:close-requested", cb),
    onOpenFile: (cb) => listen("app:open-file", cb),
    takeOpenFile: () => ipcRenderer.invoke("app:take-open-file"),
    confirmClose: () => ipcRenderer.send("app:confirm-close"),

    showOpenDialog: (title: string, filters: DialogFilter[]) => ipcRenderer.invoke("dialog:open", title, filters),
    showSaveDialog: (title: string, defaultName: string, filters: DialogFilter[]) =>
        ipcRenderer.invoke("dialog:save", title, defaultName, filters),
    readFile: (path) => ipcRenderer.invoke("fs:read", path),
    readTextFile: (path) => ipcRenderer.invoke("fs:read-text", path),
    writeFile: (path, data) => ipcRenderer.invoke("fs:write", path, data),
    fileExists: (path) => ipcRenderer.invoke("fs:exists", path),
    mediaUrl: (path) => ipcRenderer.sendSync("media:url", path),
    pathForFile: (file) => webUtils.getPathForFile(file),

    newProject: () => ipcRenderer.invoke("project:new"),
    openProject: (filePath) => ipcRenderer.invoke("project:open", filePath),
    saveProject: (workDir: string, filePath: string, doc: ProjectDocument) => ipcRenderer.invoke("project:save", workDir, filePath, doc),
    importModel: (workDir, modelPath, files, rigPath, displayName) =>
        ipcRenderer.invoke("project:import-model", workDir, modelPath, files, rigPath, displayName),
    writeRig: (workDir, rigJson) => ipcRenderer.invoke("project:write-rig", workDir, rigJson),
    newTakeDir: (workDir) => ipcRenderer.invoke("project:new-take", workDir),
    discardProject: (workDir) => ipcRenderer.invoke("project:discard", workDir),
    onProjectProgress: (cb) => listen("project:progress", cb),

    setupStatus: () => ipcRenderer.invoke("setup:status"),
    acceptTerms: () => ipcRenderer.invoke("setup:accept-terms"),
    runSetup: (force) => ipcRenderer.invoke("setup:run", force === true),
    cancelSetup: () => ipcRenderer.invoke("setup:cancel"),
    onSetupProgress: (cb: (p: SetupProgress) => void) => listen("setup:progress", cb),
    onSetupLog: (cb: (line: string) => void) => listen("setup:log", cb),
    engineRequest: (method, params) => ipcRenderer.invoke("engine:request", method, params),
    engineCancel: (job) => ipcRenderer.invoke("engine:cancel", job),
    onEngineEvent: (cb: (e: EngineEvent) => void) => listen("engine:event", cb),
    engineLogPath: () => ipcRenderer.invoke("engine:log-path"),
    modelsDir: () => ipcRenderer.invoke("app:models-dir")
};

contextBridge.exposeInMainWorld("mmt", api);
