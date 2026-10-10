// アプリのビルド: メインプロセス・プリロード・レンダラーを esbuild でまとめ、静的ファイルを dist/ にコピーする。
//   node scripts/build.mjs          本番用（圧縮あり）
//   node scripts/build.mjs --dev    開発用（ソースマップあり・圧縮なし）
//   node scripts/build.mjs --watch  変更を監視してビルドし直す
import { build, context } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeNotices } from "./notices.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dev = process.argv.includes("--dev") || process.argv.includes("--watch");
const watch = process.argv.includes("--watch");
const dist = join(root, "dist");

rmSync(dist, { recursive: true, force: true });
mkdirSync(join(dist, "renderer"), { recursive: true });

const common = {
    bundle: true,
    sourcemap: dev ? "inline" : false,
    minify: !dev,
    legalComments: "linked",
    logLevel: "info",
    define: { "process.env.NODE_ENV": JSON.stringify(dev ? "development" : "production") }
};

const targets = [
    {
        ...common,
        entryPoints: [join(root, "src/main/main.ts")],
        outfile: join(dist, "main/main.js"),
        platform: "node",
        format: "cjs",
        target: "node22",
        external: ["electron"]
    },
    {
        ...common,
        entryPoints: [join(root, "src/main/preload.ts")],
        outfile: join(dist, "main/preload.js"),
        platform: "node",
        format: "cjs",
        target: "node22",
        external: ["electron"]
    },
    {
        ...common,
        entryPoints: [join(root, "src/renderer/main.ts")],
        outfile: join(dist, "renderer/renderer.js"),
        platform: "browser",
        format: "esm",
        target: "chrome130",
        loader: { ".wgsl": "text", ".fx": "text" }
    }
];

function copyStatic() {
    cpSync(join(root, "src/renderer/index.html"), join(dist, "renderer/index.html"));
    cpSync(join(root, "src/renderer/styles"), join(dist, "renderer/styles"), { recursive: true });
    cpSync(join(root, "src/renderer/assets"), join(dist, "renderer/assets"), { recursive: true });
    // babylon-mmd の物理演算（Bullet）の WebAssembly。バンドルの中から new URL("index_bg.wasm", import.meta.url) で読む
    cpSync(join(root, "node_modules/babylon-mmd/esm/Runtime/Optimized/wasm/spr/index_bg.wasm"),
        join(dist, "renderer/index_bg.wasm"));
    // 同梱するライブラリのライセンス文（インストーラーでは resources/ に置く）
    writeNotices(root, join(dist, "THIRD_PARTY_NOTICES.txt"));
}

copyStatic();
if (watch) {
    for (const t of targets) {
        const ctx = await context(t);
        await ctx.watch();
    }
    console.log("変更を監視しています…");
} else {
    await Promise.all(targets.map((t) => build(t)));
}
