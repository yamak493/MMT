// 画面の右下に短く出すお知らせ。
import { $, h } from "./dom";

export function toast(message: string, kind: "info" | "ok" | "warn" | "error" = "info", ms = 4000): void {
    const el = h("div", { class: `toast ${kind}` }, message);
    $("toast-root").append(el);
    const remove = () => el.remove();
    const timer = setTimeout(remove, kind === "error" ? Math.max(ms, 8000) : ms);
    el.addEventListener("click", () => {
        clearTimeout(timer);
        remove();
    });
}
