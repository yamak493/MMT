// 画面のエントリ。
import { App } from "./app";
import { exportDialog } from "./dialogs/export";
import { consentDialog, showAboutDialog, showMotionTermsDialog } from "./dialogs/terms";
import { alertDialog } from "./ui/modal";

window.addEventListener("error", (e) => console.error(e.error ?? e.message));
window.addEventListener("unhandledrejection", (e) => console.error(e.reason));

const app = new App();
// 開発・動作確認用（コンソールから状態を見る）
Object.assign(window, { __mmtApp: app, __mmtDialogs: { consentDialog, exportDialog, showAboutDialog, showMotionTermsDialog } });
app.init().catch((e) => {
    void alertDialog("起動できませんでした", String((e as Error)?.message ?? e), "error");
});
