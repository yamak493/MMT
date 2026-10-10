// メニュー（ヘッダーのメニューバーと、右クリックのメニュー）。
import { $, h } from "./dom";

export interface MenuItem {
    label: string;
    shortcut?: string;
    enabled?: boolean | (() => boolean);
    checked?: boolean;
    action?: () => void;
    submenu?: MenuEntry[] | (() => MenuEntry[]);
}

export type MenuEntry = MenuItem | "separator" | { heading: string };

let current: { el: HTMLElement; close: () => void } | null = null;

function isEnabled(item: MenuItem): boolean {
    return typeof item.enabled === "function" ? item.enabled() : item.enabled ?? true;
}

export function closeMenus(): void {
    current?.close();
    current = null;
}

/** 画面の (x, y) にメニューを開く。 */
export function openMenu(entries: MenuEntry[], x: number, y: number, onClose?: () => void, parent?: HTMLElement): HTMLElement {
    if (!parent) closeMenus();
    const layer = $("menu-layer");
    const el = h("div", { class: "menu-popup", role: "menu" });
    let sub: HTMLElement | null = null;
    const closeSub = () => {
        sub?.remove();
        sub = null;
    };
    for (const entry of entries) {
        if (entry === "separator") {
            el.append(h("div", { class: "menu-sep" }));
            continue;
        }
        if ("heading" in entry) {
            el.append(h("div", { class: "menu-heading" }, entry.heading));
            continue;
        }
        const item = entry;
        const btn = h("button", { class: "menu-entry", disabled: !isEnabled(item), role: "menuitem" },
            item.checked ? h("span", { class: "check-mark" }, "✓") : null,
            h("span", null, item.label),
            item.shortcut ? h("span", { class: "shortcut" }, item.shortcut) : null,
            item.submenu ? h("span", { class: "submenu-mark" }, "▸") : null);
        btn.addEventListener("mouseenter", () => {
            el.querySelectorAll(".menu-entry.active").forEach((e) => e.classList.remove("active"));
            closeSub();
            if (item.submenu && isEnabled(item)) {
                btn.classList.add("active");
                const r = btn.getBoundingClientRect();
                const items = typeof item.submenu === "function" ? item.submenu() : item.submenu;
                sub = openMenu(items, r.right - 2, r.top - 4, undefined, el);
            }
        });
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (item.submenu || !isEnabled(item)) return;
            closeMenus();
            item.action?.();
        });
        el.append(btn);
    }
    layer.append(el);
    // 画面からはみ出さない位置へ
    const r = el.getBoundingClientRect();
    el.style.left = `${Math.max(4, Math.min(x, window.innerWidth - r.width - 4))}px`;
    el.style.top = `${Math.max(4, Math.min(y, window.innerHeight - r.height - 4))}px`;
    if (!parent) {
        const onDown = (e: MouseEvent) => {
            if (!(e.target instanceof Node) || !layer.contains(e.target)) close();
        };
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") close();
        };
        const close = () => {
            layer.innerHTML = "";
            window.removeEventListener("mousedown", onDown, true);
            window.removeEventListener("keydown", onKey, true);
            window.removeEventListener("blur", close);
            onClose?.();
            if (current?.el === el) current = null;
        };
        setTimeout(() => {
            window.addEventListener("mousedown", onDown, true);
            window.addEventListener("keydown", onKey, true);
            window.addEventListener("blur", close);
        }, 0);
        current = { el, close };
    }
    return el;
}

/** ヘッダーのメニューバー。項目にマウスを載せると、開いているメニューを切り替える。 */
export class MenuBar {
    private open: HTMLElement | null = null;

    constructor(root: HTMLElement, menus: { label: string; title?: string; items: () => MenuEntry[] }[]) {
        for (const m of menus) {
            const btn = h("button", { class: "menubar-item", title: m.title }, m.label);
            btn.addEventListener("mousedown", (e) => {
                e.preventDefault();
                if (this.open === btn) {
                    closeMenus();
                    return;
                }
                this.show(btn, m.items());
            });
            btn.addEventListener("mouseenter", () => {
                if (this.open && this.open !== btn) this.show(btn, m.items());
            });
            root.append(btn);
        }
    }

    private show(btn: HTMLElement, items: MenuEntry[]): void {
        closeMenus();
        const r = btn.getBoundingClientRect();
        btn.classList.add("open");
        this.open = btn;
        openMenu(items, r.left, r.bottom, () => {
            btn.classList.remove("open");
            if (this.open === btn) this.open = null;
        });
    }
}
