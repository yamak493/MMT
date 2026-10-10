// ZIP の読み書き（.mmt プロジェクトと、ダウンロードした uv の展開）。大きな動画も扱うのでストリームで処理する。
import { createWriteStream, mkdirSync, statSync } from "node:fs";
import { dirname, join, normalize, sep } from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import yauzl from "yauzl";
import yazl from "yazl";

export interface ZipEntry {
    /** ZIP の中のパス（/ 区切り） */
    name: string;
    /** 書き込むファイル（どちらか一方） */
    file?: string;
    data?: Buffer;
    /** 圧縮するか（動画など、すでに圧縮されているものは圧縮しない） */
    compress?: boolean;
}

export async function writeZip(dest: string, entries: ZipEntry[], onProgress?: (done: number, total: number) => void): Promise<void> {
    mkdirSync(dirname(dest), { recursive: true });
    const zip = new yazl.ZipFile();
    const total = entries.reduce((s, e) => s + (e.file ? statSync(e.file).size : e.data!.length), 0);
    for (const e of entries) {
        const compress = e.compress ?? true;
        if (e.file) {
            const size = statSync(e.file).size;
            zip.addFile(e.file, e.name, { compress, forceZip64Format: size > 0xfffffff0 });
        } else {
            zip.addBuffer(e.data!, e.name, { compress });
        }
    }
    zip.end();
    // 進捗は書き出したバイト数で数える（圧縮するファイルがあるので、合計は目安）
    let done = 0;
    const counter = new Transform({
        transform(chunk: Buffer, _enc, cb) {
            done += chunk.length;
            onProgress?.(Math.min(done, total), total);
            cb(null, chunk);
        }
    });
    const failed = new Promise<never>((_, reject) => zip.on("error", reject));
    await Promise.race([pipeline(zip.outputStream, counter, createWriteStream(dest)), failed]);
}

function openZip(path: string): Promise<yauzl.ZipFile> {
    return new Promise((resolve, reject) =>
        yauzl.open(path, { lazyEntries: true, autoClose: true, decodeStrings: true }, (err, zf) => (err ? reject(err) : resolve(zf!)))
    );
}

/** ZIP を destDir に展開する。filter で展開するエントリを選べる（戻り値を書き込み先のパスにする。null なら飛ばす）。 */
export async function extractZip(
    path: string,
    destDir: string,
    onProgress?: (done: number, total: number) => void,
    filter?: (name: string) => string | null
): Promise<string[]> {
    const zf = await openZip(path);
    const root = normalize(destDir + sep);
    const written: string[] = [];
    let entries = 0;
    return new Promise((resolve, reject) => {
        const count = Math.max(1, zf.entryCount);
        zf.on("error", reject);
        zf.on("end", () => resolve(written));
        zf.on("entry", (entry: yauzl.Entry) => {
            const next = () => {
                entries += 1;
                onProgress?.(entries, count);
                zf.readEntry();
            };
            if (/\/$/.test(entry.fileName)) {
                next();
                return;
            }
            const rel = filter ? filter(entry.fileName) : entry.fileName;
            if (rel === null) {
                next();
                return;
            }
            const target = normalize(join(destDir, rel));
            if (!target.startsWith(root)) {
                reject(new Error(`ZIP の中に不正なパスがあります: ${entry.fileName}`));
                zf.close();
                return;
            }
            mkdirSync(dirname(target), { recursive: true });
            zf.openReadStream(entry, (err, stream) => {
                if (err || !stream) {
                    reject(err ?? new Error("ZIP を読めませんでした"));
                    return;
                }
                let got = 0;
                const size = Math.max(1, entry.uncompressedSize);
                stream.on("data", (chunk: Buffer) => {
                    got += chunk.length;
                    onProgress?.(entries + got / size, count);
                });
                pipeline(stream, createWriteStream(target))
                    .then(() => {
                        written.push(target);
                        next();
                    })
                    .catch(reject);
            });
        });
        zf.readEntry();
    });
}

/** ZIP の中の 1 つのファイルを読む（無ければ null）。 */
export async function readZipEntry(path: string, name: string): Promise<Buffer | null> {
    const zf = await openZip(path);
    return new Promise((resolve, reject) => {
        let found = false;
        zf.on("error", reject);
        zf.on("end", () => {
            if (!found) resolve(null);
        });
        zf.on("entry", (entry: yauzl.Entry) => {
            if (entry.fileName !== name) {
                zf.readEntry();
                return;
            }
            found = true;
            zf.openReadStream(entry, (err, stream) => {
                if (err || !stream) {
                    reject(err ?? new Error("ZIP を読めませんでした"));
                    return;
                }
                const chunks: Buffer[] = [];
                stream.on("data", (c: Buffer) => chunks.push(c));
                stream.on("end", () => {
                    zf.close();
                    resolve(Buffer.concat(chunks));
                });
                stream.on("error", reject);
            });
        });
        zf.readEntry();
    });
}
