// ダイアログ（モーダル）。Esc で閉じる（closable のとき）、Enter で既定のボタン。
import { $, h } from "./dom";

export interface ModalButton {
    label: string;
    kind?: "primary" | "danger" | "normal";
    /** false を返すと閉じない */
    onClick?: () => boolean | void | Promise<boolean | void>;
    isDefault?: boolean;
    disabled?: boolean;
}

export interface ModalOptions {
    title: string;
    body: Node | string;
    buttons?: ModalButton[];
    size?: "normal" | "wide" | "xwide";
    closable?: boolean;
    onClose?: () => void;
    footerNote?: Node | string;
}

const stack: Modal[] = [];

export class Modal {
    readonly root: HTMLDivElement;
    readonly dialog: HTMLDivElement;
    readonly bodyEl: HTMLDivElement;
    readonly footerEl: HTMLDivElement;
    readonly buttons: HTMLButtonElement[] = [];
    private closed = false;
    private resolveClosed!: () => void;
    readonly closedPromise = new Promise<void>((r) => (this.resolveClosed = r));

    constructor(private readonly opts: ModalOptions) {
        const closable = opts.closable ?? true;
        this.bodyEl = h("div", { class: "modal-body" }, opts.body);
        this.footerEl = h("div", { class: "modal-footer" });
        if (opts.footerNote) this.footerEl.append(h("span", { class: "footer-note" }, opts.footerNote));
        for (const b of opts.buttons ?? []) {
            const btn = h("button", { class: `button ${b.kind ?? "normal"}`, disabled: b.disabled }, b.label);
            btn.addEventListener("click", async () => {
                if (btn.disabled) return;
                const r = await b.onClick?.();
                if (r !== false) this.close();
            });
            if (b.isDefault) btn.dataset.default = "1";
            this.buttons.push(btn);
            this.footerEl.append(btn);
        }
        const header = h("div", { class: "modal-header" }, h("span", { class: "modal-title" }, opts.title),
            closable ? h("button", { class: "modal-close", title: "閉じる", onclick: () => this.close() }, "✕") : null);
        this.dialog = h("div", { class: `modal ${opts.size ?? ""}`, role: "dialog" }, header, this.bodyEl,
            (opts.buttons?.length || opts.footerNote) ? this.footerEl : null);
        this.root = h("div", { class: "modal-backdrop" }, this.dialog);
        this.root.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && closable) {
                e.stopPropagation();
                this.close();
            } else if (e.key === "Enter" && !(e.target instanceof HTMLTextAreaElement) && !(e.target instanceof HTMLButtonElement)) {
                const def = this.buttons.find((b) => b.dataset.default === "1" && !b.disabled);
                if (def) {
                    e.preventDefault();
                    def.click();
                }
            }
            e.stopPropagation();
        });
    }

    open(): this {
        $("modal-root").append(this.root);
        stack.push(this);
        const focusable = this.bodyEl.querySelector<HTMLElement>("input:not([disabled]), select, textarea")
            ?? this.footerEl.querySelector<HTMLElement>("button.primary:not([disabled])")
            ?? this.footerEl.querySelector<HTMLElement>("button");
        setTimeout(() => focusable?.focus(), 0);
        return this;
    }

    close(): void {
        if (this.closed) return;
        this.closed = true;
        this.root.remove();
        const i = stack.indexOf(this);
        if (i >= 0) stack.splice(i, 1);
        this.opts.onClose?.();
        this.resolveClosed();
    }

    get isOpen(): boolean {
        return !this.closed;
    }

    setButtonEnabled(index: number, enabled: boolean): void {
        if (this.buttons[index]) this.buttons[index].disabled = !enabled;
    }
}

export function anyModalOpen(): boolean {
    return stack.length > 0;
}

export function alertDialog(title: string, message: string | Node, kind: "info" | "warn" | "error" = "info"): Promise<void> {
    const body = typeof message === "string" ? h("div", { style: "white-space: pre-wrap" }, message) : message;
    const content = kind === "info" ? body : h("div", null, h("div", { class: `notice ${kind === "error" ? "danger" : ""}` }, body));
    const m = new Modal({ title, body: content, buttons: [{ label: "OK", kind: "primary", isDefault: true }] }).open();
    return m.closedPromise;
}

/** 戻り値: 押したボタンの番号（閉じたら -1） */
export function choiceDialog(title: string, message: string | Node, labels: string[], primary = 0): Promise<number> {
    return new Promise((resolve) => {
        let chosen = -1;
        const body = typeof message === "string" ? h("div", { style: "white-space: pre-wrap" }, message) : message;
        new Modal({
            title, body,
            buttons: labels.map((label, i) => ({
                label, kind: i === primary ? "primary" : "normal", isDefault: i === primary, onClick: () => { chosen = i; }
            })),
            onClose: () => resolve(chosen)
        }).open();
    });
}

export async function confirmDialog(title: string, message: string | Node, ok = "OK", cancel = "キャンセル"): Promise<boolean> {
    return (await choiceDialog(title, message, [ok, cancel], 0)) === 0;
}
