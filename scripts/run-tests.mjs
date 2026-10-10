// 画面側（TypeScript）の単体テスト: tests/*.test.ts を esbuild でまとめて node:test で実行する。
import { build } from "esbuild";
import { readdirSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, ".test-build");
rmSync(out, { recursive: true, force: true });
const entries = readdirSync(join(root, "tests")).filter((f) => f.endsWith(".test.ts")).map((f) => join(root, "tests", f));
await build({
    entryPoints: entries, outdir: out, bundle: true, platform: "node", format: "esm", target: "node20",
    outExtension: { ".js": ".mjs" }, logLevel: "error"
});
const files = readdirSync(out).filter((f) => f.endsWith(".mjs")).map((f) => join(out, f));
const r = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" });
rmSync(out, { recursive: true, force: true });
process.exit(r.status ?? 1);
