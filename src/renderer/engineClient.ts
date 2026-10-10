// エンジンへの要求を Promise にする（進捗・ログは要求ごとのコールバックへ）。
import type { EngineError, EngineEvent, EngineLog, EngineProgress } from "../shared/types";

export class EngineRequestError extends Error {
    constructor(readonly error: EngineError) {
        super(error.message);
    }
    get cancelled(): boolean {
        return this.error.kind === "cancelled";
    }
}

interface Pending {
    resolve: (v: unknown) => void;
    reject: (e: Error) => void;
    onProgress?: (p: EngineProgress) => void;
    onLog?: (l: EngineLog) => void;
}

export interface RunningRequest<T> {
    id: number;
    result: Promise<T>;
    cancel: () => void;
}

export class EngineClient {
    private pending = new Map<number, Pending>();
    /** 要求の番号が分かる前に届いた通知 */
    private early = new Map<number, EngineEvent[]>();
    onStatus: ((running: boolean) => void) | null = null;

    constructor() {
        window.mmt.onEngineEvent((e) => this.dispatch(e));
    }

    private dispatch(e: EngineEvent): void {
        if (e.type === "status") {
            this.onStatus?.(e.running);
            return;
        }
        const id = e.type === "progress" ? e.data.job : e.type === "log" ? e.data.job : e.id;
        if (id === undefined) return;
        const p = this.pending.get(id);
        if (!p) {
            const list = this.early.get(id) ?? [];
            list.push(e);
            this.early.set(id, list);
            return;
        }
        this.apply(id, p, e);
    }

    private apply(id: number, p: Pending, e: EngineEvent): void {
        switch (e.type) {
            case "progress":
                p.onProgress?.(e.data);
                break;
            case "log":
                p.onLog?.(e.data);
                break;
            case "result":
                this.pending.delete(id);
                p.resolve(e.result);
                break;
            case "error":
                this.pending.delete(id);
                p.reject(new EngineRequestError(e.error));
                break;
        }
    }

    async start<T>(method: string, params: Record<string, unknown>,
                   onProgress?: (p: EngineProgress) => void, onLog?: (l: EngineLog) => void): Promise<RunningRequest<T>> {
        const id = await window.mmt.engineRequest(method, params);
        const result = new Promise<T>((resolve, reject) => {
            const p: Pending = { resolve: resolve as (v: unknown) => void, reject, onProgress, onLog };
            this.pending.set(id, p);
            for (const e of this.early.get(id) ?? []) this.apply(id, p, e);
            this.early.delete(id);
        });
        return { id, result, cancel: () => void window.mmt.engineCancel(id) };
    }

    async call<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
        const r = await this.start<T>(method, params);
        return r.result;
    }
}
