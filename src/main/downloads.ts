// ファイルのダウンロード（進捗・中止・途中からの再開・SHA-256 の照合）。Electron の net.fetch を使う（OS のプロキシの設定に従う）。
import { net } from "electron";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, statSync, unlinkSync } from "node:fs";
import { dirname } from "node:path";

export interface DownloadProgress {
    received: number;
    total: number | null;
}

export interface DownloadOptions {
    /** これより小さいファイルは失敗とする [バイト] */
    minBytes?: number;
    /** 中身の SHA-256（16 進の小文字）。指定すると、すでにあるファイルも照合し、違えばダウンロードし直す */
    sha256?: string;
}

export class DownloadError extends Error {}

export async function download(
    url: string,
    dest: string,
    onProgress: (p: DownloadProgress) => void,
    signal: AbortSignal,
    { minBytes = 0, sha256 }: DownloadOptions = {}
): Promise<void> {
    if (existsSync(dest) && statSync(dest).size >= minBytes && statSync(dest).size > 0) {
        if (!sha256 || (await sha256File(dest)) === sha256) return;
        unlinkSync(dest);   // 壊れている・別のファイル: ダウンロードし直す
    }
    mkdirSync(dirname(dest), { recursive: true });
    const part = dest + ".part";
    // 途中から再開したファイルが壊れていたときは、1 回だけ最初からダウンロードし直す
    for (let pass = 0; ; pass++) {
        let offset = existsSync(part) ? statSync(part).size : 0;
        for (let attempt = 0; attempt < 4; attempt++) {
            try {
                await downloadOnce(url, part, offset, onProgress, signal);
                break;
            } catch (e) {
                if (signal.aborted) throw e;
                if (attempt === 3) throw new DownloadError(`ダウンロードに失敗しました（${(e as Error).message}）: ${url}`);
                await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
                offset = existsSync(part) ? statSync(part).size : 0;
            }
        }
        const size = statSync(part).size;
        if (size < minBytes) {
            unlinkSync(part);
            throw new DownloadError(`ダウンロードしたファイルが小さすぎます（${size} バイト）: ${url}`);
        }
        if (!sha256 || (await sha256File(part)) === sha256) break;
        unlinkSync(part);
        if (pass > 0) throw new DownloadError(`ダウンロードしたファイルが壊れています（SHA-256 が一致しません）: ${url}`);
    }
    renameSync(part, dest);
}

export async function sha256File(path: string): Promise<string> {
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
    return hash.digest("hex");
}

async function downloadOnce(
    url: string,
    part: string,
    offset: number,
    onProgress: (p: DownloadProgress) => void,
    signal: AbortSignal
): Promise<void> {
    const headers: Record<string, string> = { "User-Agent": "MonoCam-Motion-Tracer" };
    if (offset > 0) headers.Range = `bytes=${offset}-`;
    const res = await net.fetch(url, { headers, signal, redirect: "follow" });
    if (res.status === 416) return;   // すでに最後まで受け取っている
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const resumed = res.status === 206;
    if (!resumed) offset = 0;
    const length = Number(res.headers.get("content-length") ?? 0);
    const total = length > 0 ? offset + length : null;
    const out = createWriteStream(part, { flags: resumed ? "a" : "w" });
    let received = offset;
    let last = 0;
    try {
        const reader = res.body!.getReader();
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!out.write(value)) await new Promise<void>((r) => out.once("drain", () => r()));
            received += value.length;
            const now = Date.now();
            if (now - last > 200) {
                last = now;
                onProgress({ received, total });
            }
        }
        onProgress({ received, total });
    } finally {
        await new Promise<void>((resolve, reject) => out.end((err?: Error | null) => (err ? reject(err) : resolve())));
    }
    if (total !== null && received < total) throw new Error("接続が途中で切れました");
}

export function formatBytes(n: number): string {
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
    if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${n} B`;
}
