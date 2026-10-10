// DOM を組み立てる小さな関数。

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, unknown> & {
    class?: string;
    style?: Partial<CSSStyleDeclaration> | string;
    dataset?: Record<string, string>;
};

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs | null = null, ...children: (Child | Child[])[]): HTMLElementTagNameMap[K] {
    const el = document.createElement(tag);
    if (attrs) {
        for (const [key, value] of Object.entries(attrs)) {
            if (value === undefined || value === null || value === false) continue;
            if (key === "class") el.className = String(value);
            else if (key === "style") {
                if (typeof value === "string") el.setAttribute("style", value);
                else Object.assign(el.style, value);
            } else if (key === "dataset") Object.assign(el.dataset, value);
            else if (key.startsWith("on") && typeof value === "function") {
                el.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
            } else if (key in el && typeof value !== "string") {
                (el as unknown as Record<string, unknown>)[key] = value;
            } else if (value === true) el.setAttribute(key, "");
            else el.setAttribute(key, String(value));
        }
    }
    for (const c of children.flat()) {
        if (c === null || c === undefined || c === false) continue;
        el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
}

export function $<T extends HTMLElement = HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`要素がありません: ${id}`);
    return el as T;
}

export function clear(el: Element): void {
    while (el.firstChild) el.removeChild(el.firstChild);
}

/** 秒 → 「mm:ss.ff」（1 時間を超えると h:mm:ss.ff） */
export function formatTime(sec: number): string {
    if (!Number.isFinite(sec) || sec < 0) sec = 0;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    const ss = s.toFixed(2).padStart(5, "0");
    return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${String(m).padStart(2, "0")}:${ss}`;
}

/** 「mm:ss.ff」「ss.ff」「h:mm:ss」 → 秒（読めなければ NaN） */
export function parseTime(text: string): number {
    const parts = text.trim().split(":").map((p) => p.trim());
    if (!parts.length || parts.some((p) => p === "" || isNaN(Number(p)))) return NaN;
    return parts.reduce((acc, p) => acc * 60 + Number(p), 0);
}

export function escapeHtml(s: string): string {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

export function basename(path: string): string {
    return path.split(/[\\/]/).pop() ?? path;
}

export function dirname(path: string): string {
    const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return i >= 0 ? path.slice(0, i) : "";
}
