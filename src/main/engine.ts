// エンジン（Python の mmt_engine）の子プロセスの管理。標準入出力の JSON Lines で要求を送り、結果・進捗を画面へ流す。
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { appendFileSync, mkdirSync, statSync, renameSync } from "node:fs";
import { join } from "node:path";
import type { EngineEvent } from "../shared/types";
import { engineDir, logsDir, modelsDir, venvPython } from "./paths";

/** 止めるときに、標準入力を閉じてから強制終了するまでの猶予 [ms] */
const STOP_GRACE_MS = 3000;
/** shutdown で終了を待つ上限 [ms]（強制終了しても終わらないときに、待ち続けないように） */
const SHUTDOWN_TIMEOUT_MS = 10000;

export class EngineProcess {
    private proc: ChildProcessWithoutNullStreams | null = null;
    private nextId = 1;
    /** 今のプロセスに送って、結果をまだ受け取っていない要求（プロセスごとに作る） */
    private pending = new Set<number>();
    private stderrTail: string[] = [];
    private readyPromise: Promise<void> | null = null;
    /** 起動させない理由（実行環境を作り直している間など）。null なら起動できる */
    private blockedReason: string | null = null;

    constructor(private readonly emit: (e: EngineEvent) => void) {}

    logPath(): string {
        return join(logsDir(), "engine.log");
    }

    private log(text: string): void {
        try {
            mkdirSync(logsDir(), { recursive: true });
            const path = this.logPath();
            try {
                if (statSync(path).size > 5 * 1024 * 1024) renameSync(path, path + ".old");
            } catch {
                /* まだログが無い */
            }
            appendFileSync(path, text);
        } catch {
            /* ログを書けなくても処理は続ける */
        }
    }

    private start(): Promise<void> {
        if (this.blockedReason) return Promise.reject(new Error(this.blockedReason));
        if (this.readyPromise) return this.readyPromise;
        const ready = new Promise<void>((resolve, reject) => {
            const env = {
                ...process.env,
                PYTHONPATH: engineDir(),
                PYTHONUTF8: "1",
                PYTHONUNBUFFERED: "1",
                MMT_MODELS_DIR: modelsDir(),
                TORCH_HOME: join(modelsDir(), "torch"),
                VIRTUAL_ENV: ""
            };
            this.log(`\n==== ${new Date().toISOString()} エンジンを起動します\n`);
            let proc: ChildProcessWithoutNullStreams;
            try {
                proc = spawn(venvPython(), ["-m", "mmt_engine"], { cwd: engineDir(), env, windowsHide: true });
            } catch (e) {
                reject(e);
                return;
            }
            // 受信の途中の行・送った要求は、プロセスごとに持つ（止めている古いプロセスと、新しいプロセスが混ざらない）
            const pending = new Set<number>();
            this.proc = proc;
            this.pending = pending;
            let ready = false;
            let buffer = "";
            /** このプロセスが今のプロセスなら、手放す（次の要求で起動し直す） */
            const release = () => {
                if (this.proc !== proc) return;
                this.proc = null;
                this.readyPromise = null;
            };
            // 終了しかけたプロセスへの書き込みの失敗（EPIPE）で、メインプロセスの例外にしない
            proc.stdin.on("error", (err) => this.log(`[stdin] ${err.message}\n`));
            proc.stdout.setEncoding("utf-8");
            proc.stdout.on("data", (chunk: string) => {
                buffer += chunk;
                let idx: number;
                while ((idx = buffer.indexOf("\n")) >= 0) {
                    const line = buffer.slice(0, idx).trim();
                    buffer = buffer.slice(idx + 1);
                    if (!line) continue;
                    let msg: Record<string, unknown>;
                    try {
                        msg = JSON.parse(line);
                    } catch {
                        this.log(`[stdout] ${line}\n`);
                        continue;
                    }
                    if (msg.type === "ready") {
                        ready = true;
                        this.emit({ type: "status", running: true });
                        resolve();
                    } else {
                        this.handleMessage(msg, pending);
                    }
                }
            });
            proc.stderr.setEncoding("utf-8");
            proc.stderr.on("data", (chunk: string) => {
                this.log(chunk);
                this.stderrTail.push(...chunk.split(/\r?\n/).filter((l) => l.trim()));
                if (this.stderrTail.length > 40) this.stderrTail.splice(0, this.stderrTail.length - 40);
            });
            proc.on("error", (err) => {
                this.log(`起動できませんでした: ${err.message}\n`);
                if (!ready) {
                    // 起動に失敗したプロセスは exit が来ないことがあるので、ここで手放す
                    release();
                    reject(new Error(`エンジンを起動できませんでした（${err.message}）`));
                }
            });
            proc.on("exit", (code) => {
                this.log(`==== エンジンが終了しました（${code}）\n`);
                release();
                const detail = this.stderrTail.slice(-12).join("\n");
                for (const id of pending) {
                    this.emit({
                        type: "error", id,
                        error: { kind: "engine", message: `エンジンが予期せず終了しました（終了コード ${code}）`, detail }
                    });
                }
                pending.clear();
                // 止めている間に新しいプロセスが起動していれば、そちらは動いている
                if (!this.proc) this.emit({ type: "status", running: false });
                if (!ready) reject(new Error(`エンジンを起動できませんでした（終了コード ${code}）\n${detail}`));
            });
        });
        this.readyPromise = ready;
        // 起動に失敗したら、次の要求でもう一度起動を試す（実行環境を入れた後など）
        ready.catch(() => {
            if (this.readyPromise === ready) this.readyPromise = null;
        });
        return ready;
    }

    private handleMessage(msg: Record<string, unknown>, pending: Set<number>): void {
        switch (msg.type) {
            case "progress":
                this.emit({ type: "progress", data: msg as never });
                break;
            case "log":
                this.log(`[log] ${String(msg.message)}\n`);
                this.emit({ type: "log", data: msg as never });
                break;
            case "result":
                pending.delete(msg.id as number);
                this.emit({ type: "result", id: msg.id as number, result: msg.result });
                break;
            case "error": {
                const err = msg.error as { kind: string; message: string; detail?: string };
                if (err.detail) this.log(`[error] ${err.detail}\n`);
                if (typeof msg.id === "number") {
                    pending.delete(msg.id);
                    this.emit({ type: "error", id: msg.id, error: err as never });
                }
                break;
            }
        }
    }

    /** 要求を送り、その番号を返す（結果は emit で届く）。 */
    async request(method: string, params: Record<string, unknown>): Promise<number> {
        const id = this.nextId++;
        const fail = (message: string) =>
            setTimeout(() => this.emit({ type: "error", id, error: { kind: "engine", message } }), 0);
        try {
            await this.start();
        } catch (e) {
            fail((e as Error).message);
            return id;
        }
        const proc = this.proc;
        if (!proc) {
            fail("エンジンが終了しました。もう一度お試しください");
            return id;
        }
        this.pending.add(id);
        proc.stdin.write(JSON.stringify({ id, method, params }) + "\n");
        return id;
    }

    cancel(job: number): void {
        if (!this.proc) return;
        const id = this.nextId++;
        this.proc.stdin.write(JSON.stringify({ id, method: "cancel", params: { job } }) + "\n");
    }

    /** 止める（標準入力を閉じるとエンジンは終わる。終わらなければ強制終了する）。終了を待たない */
    stop(): void {
        const proc = this.proc;
        if (!proc) return;
        // すぐに手放す。この後の要求は新しいプロセスで受ける（終了しかけたプロセスへは送らない）。
        // 古いプロセスに送った要求は、そのプロセスが終わったときにエラーとして返す
        this.proc = null;
        this.readyPromise = null;
        try {
            proc.stdin.end();
        } catch {
            /* すでに閉じている */
        }
        setTimeout(() => {
            if (proc.exitCode === null && proc.signalCode === null) proc.kill();
        }, STOP_GRACE_MS);
    }

    /** 止めて、プロセスが終わるまで待つ（実行環境を作り直す前に。Windows では使用中の python.exe・DLL を消せない） */
    async shutdown(): Promise<void> {
        const proc = this.proc;
        if (!proc) return;
        const exited = new Promise<void>((resolve) => {
            if (proc.exitCode !== null || proc.signalCode !== null) resolve();
            else proc.once("exit", () => resolve());
        });
        this.stop();
        await Promise.race([exited, new Promise<void>((r) => setTimeout(r, SHUTDOWN_TIMEOUT_MS))]);
    }

    /** エンジンを起動させないようにする。その間の要求は reason のエラーになる */
    block(reason: string): void {
        this.blockedReason = reason;
    }

    unblock(): void {
        this.blockedReason = null;
    }
}
