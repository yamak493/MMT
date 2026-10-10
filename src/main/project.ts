// プロジェクト（.mmt）の作業フォルダと、保存・読み込み。
//
// 開いているプロジェクトは userData/projects/<id>/ の作業フォルダで扱う（撮影の結果などはここに書かれる）。
// .mmt は ZIP で、project.json（タイムラインの状態など）と、動画・モデル（PMX とテクスチャ）・撮影の結果をまとめて入れる。
// 動画とモデルは、読み込んだ時点では元の場所のまま参照し、保存するときに ZIP へ入れる。
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, relative, resolve } from "node:path";
import type { ModelRef, OpenedProject, ProjectDocument } from "../shared/types";
import { projectsDir } from "./paths";
import { extractZip, writeZip, type ZipEntry } from "./zip";

const DOC_NAME = "project.json";

export class ProjectError extends Error {}

export function createWorkDir(): string {
    const dir = join(projectsDir(), randomUUID());
    mkdirSync(join(dir, "takes"), { recursive: true });
    return dir;
}

/** 前回の終了時に残った作業フォルダを消す（保存していない内容は .mmt には無いので残す意味がない） */
export function cleanupWorkDirs(keep: Set<string> = new Set()): void {
    const root = projectsDir();
    if (!existsSync(root)) return;
    for (const name of readdirSync(root)) {
        const dir = join(root, name);
        if (keep.has(dir)) continue;
        try {
            rmSync(dir, { recursive: true, force: true });
        } catch {
            /* 使われている（もう 1 つのウィンドウなど）ときは残す */
        }
    }
}

/** 消せなかった作業フォルダ（画面が動画などを開いたままだった）。ウィンドウを閉じた後にもう一度消す */
const leftoverWorkDirs = new Set<string>();

/** 作業フォルダを消す。消せなくても例外は投げない（終了を止めない。残ったものは終了時・次の起動時に消す） */
export function discardWorkDir(workDir: string): void {
    if (!resolve(workDir).startsWith(resolve(projectsDir()))) return;
    try {
        rmSync(workDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
        leftoverWorkDirs.delete(workDir);
    } catch {
        leftoverWorkDirs.add(workDir);
    }
}

/** 消せなかった作業フォルダを、もう一度消す（ウィンドウを閉じて、ファイルを開いている物が無くなってから） */
export function discardLeftoverWorkDirs(): void {
    for (const dir of [...leftoverWorkDirs]) discardWorkDir(dir);
}

function isInside(path: string, dir: string): boolean {
    const rel = relative(dir, path);
    return !!rel && !rel.startsWith("..") && !/^[a-zA-Z]:/.test(rel);
}

/** モデルのフォルダの中の、プロジェクトに入れるファイル（存在するものだけ） */
export function modelRef(modelPath: string, files: string[], rigPath: string, displayName: string): ModelRef {
    const dir = dirname(modelPath);
    const rels = new Set<string>([basename(modelPath)]);
    for (const f of files) {
        if (!f) continue;
        const abs = resolve(dir, f.replace(/\\/g, "/"));
        if (isInside(abs, dir) && existsSync(abs) && statSync(abs).isFile()) {
            rels.add(relative(dir, abs).replace(/\\/g, "/"));
        }
    }
    return { path: modelPath, name: basename(modelPath), files: [...rels], rigPath, displayName };
}

function zipName(rel: string): string {
    return rel.replace(/\\/g, "/").replace(/^\/+/, "");
}

/** .mmt に保存する。作業フォルダ（撮影の結果）と、元の場所の動画・モデルをまとめる。 */
export async function saveProject(
    workDir: string,
    filePath: string,
    doc: ProjectDocument,
    onProgress: (message: string, value: number | null) => void
): Promise<ProjectDocument> {
    const entries: ZipEntry[] = [];
    const out: ProjectDocument = JSON.parse(JSON.stringify(doc));
    out.savedAt = new Date().toISOString();

    if (doc.video) {
        if (!existsSync(doc.video.path)) throw new ProjectError(`動画が見つかりません: ${doc.video.path}`);
        const name = `video/${basename(doc.video.path)}`;
        entries.push({ name, file: doc.video.path, compress: false });
        out.video!.path = name;
    }
    if (doc.model) {
        const dir = dirname(doc.model.path);
        for (const rel of doc.model.files) {
            const abs = join(dir, rel);
            if (existsSync(abs)) {
                const ext = extname(rel).toLowerCase();
                entries.push({ name: zipName(`model/${rel}`), file: abs, compress: ![".png", ".jpg", ".jpeg"].includes(ext) });
            }
        }
        out.model!.path = zipName(`model/${basename(doc.model.path)}`);
        if (existsSync(doc.model.rigPath)) {
            entries.push({ name: "model_rig.json", file: doc.model.rigPath });
            out.model!.rigPath = "model_rig.json";
        }
    }
    if (doc.waveform && existsSync(doc.waveform.path)) {
        entries.push({ name: "audio/waveform.f32", file: doc.waveform.path });
        out.waveform = { ...doc.waveform, path: "audio/waveform.f32" };
    } else {
        out.waveform = null;
    }
    for (const take of doc.takes) {
        const dir = join(workDir, take.dir);
        if (!existsSync(dir)) continue;
        for (const name of readdirSync(dir)) {
            const abs = join(dir, name);
            if (statSync(abs).isFile() && !name.endsWith(".part") && !name.includes(".tmp")) {
                entries.push({ name: zipName(`${take.dir}/${name}`), file: abs });
            }
        }
        // 変換の結果は今使っているものだけを入れる（足ピタの編集のたびに増えたものは捨てる）
        if (take.resultKey) {
            const conv = join(dir, "conv", take.resultKey);
            if (existsSync(conv)) {
                for (const name of readdirSync(conv)) {
                    entries.push({ name: zipName(`${take.dir}/conv/${take.resultKey}/${name}`), file: join(conv, name) });
                }
            }
        }
    }
    entries.unshift({ name: DOC_NAME, data: Buffer.from(JSON.stringify(out, null, 2), "utf-8") });

    const tmp = filePath + ".saving";
    onProgress("プロジェクトを保存しています", 0);
    try {
        await writeZip(tmp, entries, (done, total) => onProgress("プロジェクトを保存しています", total ? done / total : null));
        renameSync(tmp, filePath);
    } catch (e) {
        rmSync(tmp, { force: true });
        throw e;
    }
    onProgress("保存しました", 1);
    return out;
}

/** .mmt を開く（新しい作業フォルダに展開し、パスを作業フォルダの中の絶対パスに直す）。 */
export async function openProject(filePath: string, onProgress: (message: string, value: number | null) => void): Promise<OpenedProject> {
    const workDir = createWorkDir();
    try {
        onProgress("プロジェクトを開いています", 0);
        await extractZip(filePath, workDir, (done, total) => onProgress("プロジェクトを開いています", done / total));
        const docPath = join(workDir, DOC_NAME);
        if (!existsSync(docPath)) throw new ProjectError("MonoCam Motion Trace のプロジェクトではありません");
        const doc = JSON.parse(readFileSync(docPath, "utf-8")) as ProjectDocument;
        if (doc.format !== "mmt-project") throw new ProjectError("MonoCam Motion Trace のプロジェクトではありません");
        if (doc.version > 1) throw new ProjectError("新しい版の MonoCam Motion Trace で保存されたプロジェクトです。アプリを更新してください");
        if (doc.video) doc.video.path = join(workDir, doc.video.path);
        if (doc.model) {
            doc.model.path = join(workDir, doc.model.path);
            doc.model.rigPath = join(workDir, doc.model.rigPath);
        }
        if (doc.waveform) doc.waveform.path = join(workDir, doc.waveform.path);
        onProgress("開きました", 1);
        return { workDir, filePath, document: doc };
    } catch (e) {
        discardWorkDir(workDir);
        if (e instanceof ProjectError) throw e;
        throw new ProjectError(`プロジェクトを開けませんでした（${(e as Error).message}）`);
    }
}

export function writeRig(workDir: string, rigJson: string): string {
    const path = join(workDir, "model_rig.json");
    const tmp = path + ".tmp";
    writeFileSync(tmp, rigJson, "utf-8");
    renameSync(tmp, path);
    return path;
}

export function newTakeDir(workDir: string): { id: string; dir: string; abs: string } {
    const id = randomUUID().slice(0, 8);
    const dir = `takes/${id}`;
    const abs = join(workDir, dir);
    mkdirSync(abs, { recursive: true });
    return { id, dir, abs };
}
