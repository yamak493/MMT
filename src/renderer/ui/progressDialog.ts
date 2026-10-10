// 長い処理の進捗ダイアログ（全体の進捗・今の段階・ログ・中止）。
import { h } from "./dom";
import { Modal } from "./modal";

export class ProgressDialog {
    private modal: Modal;
    private messageEl = h("div", { class: "progress-message" });
    private bar = h("div", { class: "progress-bar indeterminate" }, h("div", { style: "width: 0%" }));
    private detailEl = h("div", { class: "progress-detail" });
    private percentEl = h("span", { class: "dim" });
    private logEl: HTMLDivElement | null = null;
    private started = performance.now();
    private elapsedEl = h("span", { class: "dim" });
    private timer: number;

    constructor(title: string, opts: { onCancel?: () => void; cancelLabel?: string; showLog?: boolean; note?: string } = {}) {
        const body = h("div", { class: "progress-wrap" },
            this.messageEl, this.bar,
            h("div", { class: "form-row", style: "justify-content: space-between" }, this.detailEl,
                h("span", null, this.percentEl, "　", this.elapsedEl)),
            opts.note ? h("div", { class: "form-help" }, opts.note) : null);
        if (opts.showLog) {
            this.logEl = h("div", { class: "progress-log" });
            body.append(h("details", null, h("summary", { class: "dim", style: "cursor: pointer" }, "詳しいログ"), this.logEl));
        }
        this.modal = new Modal({
            title, body, size: "wide", closable: false,
            buttons: opts.onCancel ? [{
                label: opts.cancelLabel ?? "中止", kind: "normal",
                onClick: () => {
                    opts.onCancel!();
                    this.message("中止しています…");
                    this.modal.setButtonEnabled(0, false);
                    return false;
                }
            }] : []
        }).open();
        this.timer = window.setInterval(() => this.tick(), 1000);
        this.tick();
    }

    private tick(): void {
        const sec = Math.round((performance.now() - this.started) / 1000);
        this.elapsedEl.textContent = `経過 ${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
    }

    message(text: string): void {
        this.messageEl.textContent = text;
    }

    detail(text: string): void {
        this.detailEl.textContent = text;
    }

    /** value: 0〜1、null は割合が分からない */
    progress(value: number | null): void {
        const inner = this.bar.firstElementChild as HTMLDivElement;
        if (value === null || !Number.isFinite(value)) {
            this.bar.classList.add("indeterminate");
            this.percentEl.textContent = "";
        } else {
            this.bar.classList.remove("indeterminate");
            inner.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 1000) / 10}%`;
            this.percentEl.textContent = `${Math.floor(value * 100)}%`;
        }
    }

    log(line: string, level: "info" | "warn" | "error" = "info"): void {
        if (!this.logEl) return;
        const atBottom = this.logEl.scrollTop + this.logEl.clientHeight >= this.logEl.scrollHeight - 4;
        this.logEl.append(h("div", { class: level }, line));
        while (this.logEl.childElementCount > 800) this.logEl.firstElementChild?.remove();
        if (atBottom) this.logEl.scrollTop = this.logEl.scrollHeight;
    }

    close(): void {
        clearInterval(this.timer);
        this.modal.close();
    }
}
