// 独自のスキーム。
//   app://mmt/<パス>        アプリの画面（dist/renderer）
//   media://local/<パス>    ユーザーのファイル（動画・モデル・テクスチャ）。Range の要求に応える（動画のシーク）
// file:// のままだと、画面から fetch・XHR で読めない・WebAssembly を読み込めないので、どちらも独自のスキームで配る。
import { protocol } from "electron";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { Readable } from "node:stream";
import { rendererDir } from "./paths";

const MIME: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tga": "image/x-tga",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".dds": "application/octet-stream",
    ".spa": "image/bmp",
    ".sph": "image/bmp",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2"
};

export function registerSchemes(): void {
    protocol.registerSchemesAsPrivileged([
        { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } },
        { scheme: "media", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } }
    ]);
}

/** ローカルのパス → media:// の URL（パスの区切りを保つので、モデルのテクスチャの相対パスがそのまま解決できる） */
export function mediaUrl(path: string): string {
    const p = path.replace(/\\/g, "/");
    if (p.startsWith("//")) {
        return "media://unc/" + p.slice(2).split("/").map(encodeURIComponent).join("/");
    }
    return "media://local/" + p.split("/").map((s, i) => (i === 0 && /^[A-Za-z]:$/.test(s) ? s : encodeURIComponent(s))).join("/");
}

function mediaPath(url: URL): string {
    const parts = url.pathname.split("/").filter((s, i) => i > 0 || s !== "").map(decodeURIComponent);
    const joined = parts.join("/");
    if (url.host === "unc") return "\\\\" + joined.replace(/\//g, "\\");
    return normalize(joined);
}

function fileResponse(path: string, request: Request): Response {
    let size: number;
    try {
        const st = statSync(path);
        if (!st.isFile()) return new Response("not found", { status: 404 });
        size = st.size;
    } catch {
        return new Response("not found", { status: 404 });
    }
    const type = MIME[extname(path).toLowerCase()] ?? "application/octet-stream";
    const range = request.headers.get("range");
    const headers: Record<string, string> = {
        "Content-Type": type,
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*"
    };
    const m = range ? /bytes=(\d*)-(\d*)/.exec(range) : null;
    if (m && (m[1] !== "" || m[2] !== "")) {
        let start: number;
        let end: number;
        if (m[1] === "") {
            const suffix = parseInt(m[2], 10);
            start = Math.max(0, size - suffix);
            end = size - 1;
        } else {
            start = parseInt(m[1], 10);
            end = m[2] === "" ? size - 1 : Math.min(parseInt(m[2], 10), size - 1);
        }
        if (start >= size || start > end) {
            return new Response(null, { status: 416, headers: { ...headers, "Content-Range": `bytes */${size}` } });
        }
        const stream = Readable.toWeb(createReadStream(path, { start, end })) as ReadableStream;
        return new Response(stream, {
            status: 206,
            headers: { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": String(end - start + 1) }
        });
    }
    const stream = Readable.toWeb(createReadStream(path)) as ReadableStream;
    return new Response(stream, { status: 200, headers: { ...headers, "Content-Length": String(size) } });
}

export function handleSchemes(): void {
    const root = rendererDir();
    protocol.handle("app", (request) => {
        const url = new URL(request.url);
        const rel = decodeURIComponent(url.pathname).replace(/^\/+/, "") || "index.html";
        const path = normalize(join(root, rel));
        if (!path.startsWith(normalize(root + sep)) && path !== normalize(root)) {
            return new Response("forbidden", { status: 403 });
        }
        return fileResponse(path, request);
    });
    protocol.handle("media", (request) => fileResponse(mediaPath(new URL(request.url)), request));
}
