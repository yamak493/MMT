// 元に戻す・やり直す（タイムラインの状態をまるごと記録する。回数の上限なし）。
//
// 状態は小さな JSON（クリップ・足ピタの上書き・手のポーズのキー）なので、毎回の操作で全体を写しても軽い。

export interface HistoryEntry<T> {
    state: T;
    label: string;
}

export class History<T> {
    private past: HistoryEntry<T>[] = [];
    private future: HistoryEntry<T>[] = [];
    onChange: (() => void) | null = null;

    constructor(private present: T, private readonly clone: (s: T) => T = (s) => structuredClone(s)) {}

    get current(): T {
        return this.present;
    }

    get canUndo(): boolean {
        return this.past.length > 0;
    }

    get canRedo(): boolean {
        return this.future.length > 0;
    }

    get undoLabel(): string | null {
        return this.past.length ? this.past[this.past.length - 1].label : null;
    }

    get redoLabel(): string | null {
        return this.future.length ? this.future[this.future.length - 1].label : null;
    }

    /** 新しい状態にする（操作の名前を付けて履歴に積む）。同じ状態なら何もしない。 */
    push(next: T, label: string): boolean {
        if (JSON.stringify(next) === JSON.stringify(this.present)) return false;
        this.past.push({ state: this.present, label });
        this.present = this.clone(next);
        this.future = [];
        this.onChange?.();
        return true;
    }

    undo(): HistoryEntry<T> | null {
        const entry = this.past.pop();
        if (!entry) return null;
        this.future.push({ state: this.present, label: entry.label });
        this.present = entry.state;
        this.onChange?.();
        return entry;
    }

    redo(): HistoryEntry<T> | null {
        const entry = this.future.pop();
        if (!entry) return null;
        this.past.push({ state: this.present, label: entry.label });
        this.present = entry.state;
        this.onChange?.();
        return entry;
    }

    /** 履歴を消して、この状態から始める（プロジェクトを開いたときなど） */
    reset(state: T): void {
        this.present = this.clone(state);
        this.past = [];
        this.future = [];
        this.onChange?.();
    }

    /** 今の状態の写し（変更して push するための） */
    draft(): T {
        return this.clone(this.present);
    }
}
