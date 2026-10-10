// 実行環境（Python・PyTorch）と学習済みモデルの準備。利用規約に同意してからダウンロードする。
import type { SetupStatus } from "../../shared/types";
import { alertDialog } from "../ui/modal";
import { ProgressDialog } from "../ui/progressDialog";
import { consentDialog } from "./terms";

const PHASE_LABELS: Record<string, string> = {
    runtime: "実行環境",
    packages: "ライブラリ",
    models: "学習済みモデル",
    verify: "確認",
    done: "完了"
};

/**
 * キャプチャの前に呼ぶ。準備ができていれば true（同意しなかった・失敗したら false）。
 * force なら、準備ができていても実行環境を消して入れ直す（「実行環境を再セットアップ」）。
 */
export async function ensureReady(force = false): Promise<boolean> {
    let status: SetupStatus = await window.mmt.setupStatus();
    if (!force && status.runtimeReady && status.modelsReady && status.termsAccepted) return true;
    if (!status.termsAccepted) {
        if (!(await consentDialog())) return false;
        await window.mmt.acceptTerms();
    }
    const dialog = new ProgressDialog(force ? "実行環境の再セットアップ" : "AI のモデルの準備", {
        showLog: true,
        onCancel: () => void window.mmt.cancelSetup(),
        note: force ? "中止した場合は、次のキャプチャのときに続きから準備します。"
            : "初回だけ行います。中止しても、次回はダウンロードの続きから再開します。"
    });
    dialog.message("準備を始めています…");
    const offProgress = window.mmt.onSetupProgress((p) => {
        dialog.message(`［${PHASE_LABELS[p.phase] ?? p.phase}］${p.message}`);
        dialog.progress(p.value);
        dialog.detail(p.detail ?? "");
    });
    const offLog = window.mmt.onSetupLog((line) => dialog.log(line, /error|エラー/i.test(line) ? "error" : /warn/i.test(line) ? "warn" : "info"));
    try {
        status = await window.mmt.runSetup(force);
        return status.runtimeReady && status.modelsReady;
    } catch (e) {
        const msg = (e as Error).message.replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
        if (!/中止/.test(msg)) {
            await alertDialog("準備に失敗しました",
                `${msg}\n\nインターネットの接続と、ディスクの空き容量（約 16 GB）を確認して、もう一度お試しください。`, "error");
        }
        return false;
    } finally {
        offProgress();
        offLog();
        dialog.close();
    }
}
