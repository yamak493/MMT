// NVIDIA の GPU の名前・ドライバー・CUDA の版を nvidia-smi で調べる（画面に出す情報だけに使う）。
// 入れる PyTorch は GPU に関わらず決まっていて、ドライバーが対応しているかはセットアップでドライバー API に聞いて確かめる。
import { execFile } from "node:child_process";
import type { GpuInfo } from "../shared/types";

function run(cmd: string, args: string[]): Promise<string> {
    return new Promise((resolve) => {
        execFile(cmd, args, { windowsHide: true, timeout: 15000 }, (err, stdout) => resolve(err ? "" : String(stdout)));
    });
}

export async function detectGpu(): Promise<GpuInfo | null> {
    // driver_version は 610 系のドライバーで非推奨になった（CUDA 14 で削除予定。無い項目を問い合わせると nvidia-smi は
    // 何も出さずに失敗する）。新しい kmd_version を先に試し、それが無い古いドライバーでは driver_version を使う
    let line: string | undefined;
    for (const field of ["kmd_version", "driver_version"]) {
        const query = await run("nvidia-smi", [`--query-gpu=name,${field},memory.total`, "--format=csv,noheader,nounits"]);
        line = query.split(/\r?\n/).find((l) => l.trim());
        if (line) break;
    }
    if (!line) return null;
    const [name, driver, mem] = line.split(",").map((s) => s.trim());
    const full = await run("nvidia-smi", []);
    // 見出しの表記はドライバーの版で変わる（「CUDA Version: 12.8」、610 系以降は「CUDA UMD Version: 13.3」など）
    const m = /CUDA(?:\s+\w+)?\s+Version:\s*([\d.]+)/.exec(full);
    return { name, driver, cudaVersion: m ? m[1] : null, memoryMb: mem ? Number(mem) : null };
}
