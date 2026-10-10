// 配布物に同梱するライブラリ（esbuild で main.js・renderer.js にまとめた npm のパッケージと、コピーしている
// WebAssembly）のライセンス文を、THIRD_PARTY_NOTICES.txt にまとめる。MIT・Apache-2.0 は、配布物にライセンス文と
// 著作権表示を含めることを求めている。
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LICENSE_FILE = /^(licen[cs]e|notice|copying)(\.(md|txt))?$/i;

/** package-lock.json の本番の依存のほかに、ファイルをコピーして同梱しているもの */
const EXTRA = [
    { title: "babylon-mmd の物理演算（index_bg.wasm）", dir: "node_modules/babylon-mmd/esm/Runtime/Optimized/wasm/spr" }
];

const HEADER = `MonoCam Motion Trace に同梱しているソフトウェアのライセンス
================================================================

本ソフトウェア自体のライセンスは LICENSE（GNU Affero General Public License v3.0）にあります。
Electron・Chromium のライセンスは、インストール先の LICENSE.electron.txt・LICENSES.chromium.html にあります。
学習済みモデル（NLF・MediaPipe Hands・Demucs・Allosaurus）と PyTorch などの実行環境は同梱しておらず、
初回のキャプチャのときに各配布元からダウンロードします（それぞれのライセンスに従います）。
`;

function section(title, dir) {
    const files = readdirSync(dir).filter((f) => LICENSE_FILE.test(f)).sort();
    if (!files.length) throw new Error(`ライセンスのファイルが見つかりません: ${dir}`);
    const rule = "-".repeat(78);
    return [rule, title, rule, ...files.map((f) => readFileSync(join(dir, f), "utf-8").trim())].join("\n\n");
}

/** root の package-lock.json と node_modules から、out にライセンス文をまとめて書く */
export function writeNotices(root, out) {
    const lock = JSON.parse(readFileSync(join(root, "package-lock.json"), "utf-8"));
    const sections = [];
    for (const [path, info] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
        if (!path.startsWith("node_modules/") || info.dev || info.devOptional) continue;
        const name = path.slice(path.lastIndexOf("node_modules/") + "node_modules/".length);
        sections.push(section(`${name} ${info.version}（${info.license ?? "ライセンスの記載なし"}）`, join(root, path)));
    }
    for (const e of EXTRA) sections.push(section(e.title, join(root, e.dir)));
    writeFileSync(out, [HEADER, ...sections].join("\n\n") + "\n", "utf-8");
}
