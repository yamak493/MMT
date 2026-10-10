// 利用規約（本ソフトウェアについて・作成したモーションについて）と、AI の重みのダウンロード前の同意。
import { APP_REPOSITORY } from "../../shared/types";
import { h } from "../ui/dom";
import { Modal } from "../ui/modal";

function link(url: string, text = url): HTMLAnchorElement {
    return h("a", { href: "#", onclick: (e: Event) => { e.preventDefault(); void window.mmt.openExternal(url); } }, text);
}

/** 使用しているソフトウェア・学習済みモデル（名前, 用途, ライセンス, URL） */
const LIBRARIES: [string, string, string, string][] = [
    ["Electron", "アプリの基盤", "MIT", "https://www.electronjs.org/"],
    ["Babylon.js", "3D の表示", "Apache-2.0", "https://www.babylonjs.com/"],
    ["babylon-mmd", "MMD のモデル・モーションの表示と物理演算", "MIT", "https://github.com/noname0310/babylon-mmd"],
    ["encoding-japanese / yazl / yauzl", "文字コード・ZIP（プロジェクトの保存）", "MIT", "https://github.com/polygonplanet/encoding.js"],
    ["NLF（Neural Localizer Fields）", "動画からの 3D の姿勢と体型の推定（学習済みモデルは初回にダウンロード）", "コード: MIT / 学習済みモデル: 非商用の研究用途", "https://github.com/isarandi/nlf"],
    ["SMPL（NLF に同梱）", "人体のモデル", "Max Planck の独自ライセンス（非商用に限る）", "https://smpl.is.tue.mpg.de/"],
    ["YOLOv8（NLF に同梱）", "人物の検出", "AGPL-3.0", "https://github.com/ultralytics/ultralytics"],
    ["MediaPipe Hands", "手の 21 点の推定", "Apache-2.0", "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker"],
    ["LiteRT（ai-edge-litert）", "MediaPipe のモデルの実行", "Apache-2.0", "https://ai.google.dev/edge/litert"],
    ["Demucs", "楽曲からボーカルの分離（口パク）", "MIT", "https://github.com/adefossez/demucs"],
    ["Allosaurus", "音素の認識（口パク）", "GPL-3.0", "https://github.com/xinjli/allosaurus"],
    ["PyTorch / torchvision", "推論の実行", "BSD-3-Clause", "https://pytorch.org/"],
    ["NumPy / SciPy", "数値計算（モーションの変換）", "BSD-3-Clause", "https://numpy.org/"],
    ["FFmpeg（imageio-ffmpeg）", "動画・音声の読み込み", "GPL-3.0（同梱のバイナリ）", "https://ffmpeg.org/"],
    ["uv", "Python の実行環境の準備", "MIT / Apache-2.0", "https://github.com/astral-sh/uv"],
    ["Microsoft Visual C++ ランタイム（msvc-runtime）", "PyTorch などの実行", "Visual Studio 2022 の再頒布可能コード", "https://learn.microsoft.com/visualstudio/releases/2022/redistribution"]
];

export function showAboutDialog(version: string): void {
    const table = h("table", { class: "lib-table" },
        h("tr", null, h("th", null, "名前"), h("th", null, "用途"), h("th", null, "ライセンス")),
        ...LIBRARIES.map(([name, use, lic, url]) => h("tr", null, h("td", null, link(url, name)), h("td", null, use), h("td", null, lic))));
    new Modal({
        title: "本ソフトウェアについて",
        size: "wide",
        body: h("div", null,
            h("h3", null, `MonoCam Motion Trace ${version}`),
            h("p", null, "単眼カメラで撮影した動画から人体のモーションを推定し、MMD（MikuMikuDance）用のモーション（VMD）を作成するソフトウェアです。"),
            h("p", null, "本ソフトウェアのソースコードは GNU Affero General Public License v3.0 で公開しています: ", link(APP_REPOSITORY)),
            h("h3", null, "使用しているライブラリ・学習済みモデル"),
            table,
            h("h3", null, "論文"),
            h("p", { class: "form-help" }, "István Sárándi and Gerard Pons-Moll, \"Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation\", NeurIPS 2024."),
            h("p", { class: "form-help" }, "学習済みモデル（NLF・MediaPipe・Demucs・Allosaurus）と PyTorch は本ソフトウェアに同梱しておらず、初回のキャプチャのときに各配布元からダウンロードします。")),
        buttons: [{ label: "閉じる", kind: "primary", isDefault: true }]
    }).open();
}

/** 作品の投稿の可否を NLF の著者に質問したときの issue */
const NLF_PERMISSION_ISSUE = "https://github.com/isarandi/nlf/issues/51";

/** 作品を公開するときに記載するクレジット */
const CREDIT_TEXT = [
    "Neural Localizer Fields / István Sárándi, Gerard Pons-Moll",
    "MonoCam Motion Trace / yamak493"
].join("\n");

/** コピーのボタンが付いたコードブロック */
function copyableBlock(text: string): HTMLElement {
    const button = h("button", { class: "code-copy", type: "button", title: "クリップボードにコピー" }, "コピー");
    let timer = 0;
    button.addEventListener("click", async () => {
        let ok = true;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            ok = false;
        }
        button.textContent = ok ? "コピーしました" : "コピーできませんでした";
        button.classList.toggle("done", ok);
        clearTimeout(timer);
        timer = window.setTimeout(() => {
            button.textContent = "コピー";
            button.classList.remove("done");
        }, 1800);
    });
    return h("div", { class: "code-block" }, h("pre", null, h("code", null, text)), button);
}

/** 作成したモーションの利用条件（NLF・SMPL の規約による） */
export function motionTermsBody(): HTMLElement {
    return h("div", null,
        h("div", { class: "notice" }, h("strong", null, "本ソフトウェアで作成したモーションは、商用・ポルノ・軍事・監視の目的には利用できません。")),
        h("p", null, "モーションの推定に使う NLF の学習済みモデルは「非商用の研究用途」、それに含まれる SMPL の人体モデルは「非商用の科学研究・教育・芸術作品」に限って公開されています。そのため、本ソフトウェアで作成したモーション（VMD）にも次の条件が付きます。"),
        h("h3", null, "利用できない用途"),
        h("ul", null,
            h("li", null, h("strong", null, "営利目的での使用"), "：販売・有料配布・有料サービス、収益化した動画や配信（広告収益・投げ銭・メンバー限定公開など）、クリエイター支援サービス（有料支援者向けの公開を含む）、企業案件での使用"),
            h("li", null, h("strong", null, "ポルノ目的での使用"), "：性的なコンテンツ（R-18 の作品を含む）、過度な暴力表現（グロテスク）"),
            h("li", null, h("strong", null, "軍事・監視の目的での使用")),
            h("li", null, h("strong", null, "虚偽・名誉毀損・誤解を招くコンテンツ"), "：実在の人物がしていない動きをしたように見せる、特定の人物を中傷する など")),
        h("h3", null, "守っていただきたいこと"),
        h("ul", null,
            h("li", null, "本ソフトウェアを使用した作品は、利用条件の範囲内で投稿・公開できます。（",
                link(NLF_PERMISSION_ISSUE, "isarandi/nlf#51"), "）"),
            h("li", null, "作品を投稿・公開するときは、次のクレジットを記載してください", copyableBlock(CREDIT_TEXT)),
            h("li", null, "入力する動画（踊り手の肖像・振付）・楽曲・MMD モデルの権利および利用規約は、利用者ご自身で確認してください。"),
            h("li", null, "プロジェクトファイル（.mmt）には SMPL のパラメータと元の動画が入っています。他人に配布しないでください")),
        h("p", { class: "form-help" }, "これは法的な助言ではありません。各ライセンスの原文が優先します: ",
            link("https://github.com/isarandi/nlf", "NLF"), "・", link("https://smpl.is.tue.mpg.de/modellicense.html", "SMPL"), "・",
            link("https://smpl-x.is.tue.mpg.de/modellicense.html", "SMPL-X")));
}

export function showMotionTermsDialog(): void {
    new Modal({
        title: "本ソフトウェアで作成したモーションについて《重要》",
        size: "wide",
        body: motionTermsBody(),
        buttons: [{ label: "閉じる", kind: "primary", isDefault: true }]
    }).open();
}

/** AI の重みのダウンロードの前の同意。すべてにチェックを入れると true。 */
export function consentDialog(): Promise<boolean> {
    return new Promise((resolve) => {
        let agreed = false;
        const items: [string, string][] = [
            ["ポルノコンテンツには使用しません", "性的なコンテンツ（R-18 を含む）やグロテスクな表現には使用しません。"],
            ["営利目的では使用しません", "商用利用・収益化した動画や配信・クリエイター支援サービス（有料支援者向けの公開を含む）では使用しません。"],
            ["SMPL / SMPL-X のライセンスに同意します", "非商用の研究・教育・芸術作品に限ること、軍事・監視・虚偽・名誉毀損の目的に使わないこと、SMPL のデータを再配布しないことを含みます。"],
            ["NLF のライセンスに同意します", "NLF の学習済みモデルは非商用の研究用途で公開されています（趣味の非商用の創作は含まれます）。"]
        ];
        const boxes: HTMLInputElement[] = [];
        const list = h("div", { class: "terms-list" }, ...items.map(([title, text]) => {
            const box = h("input", { type: "checkbox" });
            boxes.push(box);
            box.addEventListener("change", update);
            return h("label", { class: "terms-item" }, box, h("div", { class: "terms-text" }, h("strong", null, title), h("span", null, text)));
        }));
        const modal = new Modal({
            title: "利用規約への同意",
            size: "wide",
            body: h("div", null,
                h("p", null, "モーションの抽出には AI の学習済みモデル（NLF・MediaPipe Hands など）を使います。初回は実行環境（PyTorch など、約 3〜4 GB）と学習済みモデル（約 0.5 GB）をダウンロードします。ディスクの空き容量が約 16 GB 必要です。"),
                h("p", null, "ダウンロードを始める前に、次の利用条件をすべて確認してチェックを入れてください。"),
                list,
                h("details", null, h("summary", { class: "dim", style: "cursor: pointer" }, "作成したモーションの利用条件の詳細"), motionTermsBody())),
            buttons: [
                { label: "キャンセル" },
                { label: "同意してダウンロードを開始", kind: "primary", disabled: true, onClick: () => { agreed = true; } }
            ],
            onClose: () => resolve(agreed)
        }).open();
        function update() {
            modal.setButtonEnabled(1, boxes.every((b) => b.checked));
        }
    });
}
