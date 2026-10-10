// VMD 出力のダイアログ: 出力する内容（体・指・口パク）、範囲、キーの間引きを選ぶ。
// 体の動きの種類はタイムラインのクリップごとの種類のとおりに出力する。
import { h } from "../ui/dom";
import { Modal } from "../ui/modal";

export interface ExportOptions {
    body: boolean;
    fingers: boolean;
    face: boolean;
    startAtZero: boolean;
    thin: boolean;
}

export interface ExportDialogInput {
    hasFingers: boolean;
    hasFace: boolean;
    firstFrame: number;
    lastFrame: number;
    modelName: string;
    initial?: Partial<ExportOptions>;
}

export function exportDialog(input: ExportDialogInput): Promise<ExportOptions | null> {
    return new Promise((resolve) => {
        let result: ExportOptions | null = null;
        const box = (checked: boolean, disabled = false) => h("input", { type: "checkbox", checked: checked && !disabled, disabled });
        const body = box(input.initial?.body ?? true);
        const fingers = box(input.initial?.fingers ?? true, !input.hasFingers);
        const face = box(input.initial?.face ?? true, !input.hasFace);
        const startAtZero = box(input.initial?.startAtZero ?? true);
        const thin = box(input.initial?.thin ?? false);
        const summary = h("div", { class: "estimate" });
        const update = () => {
            const parts = [body.checked ? "体の動き" : null, fingers.checked ? "指（手のポーズ）" : null, face.checked ? "口パク" : null].filter(Boolean);
            summary.textContent = parts.length
                ? `出力: ${parts.join("・")} / フレーム ${startAtZero.checked ? 0 : input.firstFrame} 〜 ${input.lastFrame - (startAtZero.checked ? input.firstFrame : 0)} / モデル名「${input.modelName}」`
                : "出力する内容を 1 つ以上選んでください";
            modal.setButtonEnabled(1, parts.length > 0);
        };
        const modal = new Modal({
            title: "VMD 出力",
            body: h("div", null,
                h("p", null, "タイムラインで編集したモーションを VMD ファイルに書き出します。"),
                h("fieldset", { class: "form-section" }, h("legend", null, "出力する内容"),
                    h("div", { class: "form-row" },
                        h("label", { class: "check" }, body, "体の動き"),
                        h("label", { class: "check", title: input.hasFingers ? "" : "手のポーズのキーがありません" }, fingers, "指（手のポーズ）"),
                        h("label", { class: "check", title: input.hasFace ? "" : "口パクがありません" }, face, "表情（口パク）"))),
                h("div", { class: "form-grid", style: "grid-template-columns: 110px 1fr" },
                    h("label", null, "フレーム"), h("label", { class: "check" }, startAtZero, "最初のクリップを 0 フレームから始める"),
                    h("label", null, "キー"), h("label", { class: "check" }, thin, "変化の少ないキーを間引く（試験的）")),
                h("p", { class: "form-help", style: "margin-top: 10px" },
                    "体の動きは、タイムラインのクリップごとの種類（右クリックの「出力の種類」）のとおりに出力します。"),
                summary,
                h("p", { class: "form-help", style: "margin-top: 8px" }, "作成したモーションは、商用・ポルノ・軍事・監視の目的には利用できません（利用規約を参照）。")),
            buttons: [
                { label: "キャンセル" },
                {
                    label: "書き出す…", kind: "primary", isDefault: true,
                    onClick: () => {
                        result = {
                            body: body.checked, fingers: fingers.checked, face: face.checked,
                            startAtZero: startAtZero.checked, thin: thin.checked
                        };
                    }
                }
            ],
            onClose: () => resolve(result)
        }).open();
        for (const el of [body, fingers, face, startAtZero]) el.addEventListener("change", update);
        update();
    });
}
