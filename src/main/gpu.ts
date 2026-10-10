// NVIDIA の GPU と、ドライバーが対応する CUDA の版を nvidia-smi で調べる（PyTorch の種類を選ぶのに使う）。
import { execFile } from "node:child_process";
import type { GpuInfo } from "../shared/types";

function run(cmd: string, args: string[]): Promise<string> {
    return new Promise((resolve) => {
        execFile(cmd, args, { windowsHide: true, timeout: 15000 }, (err, stdout) => resolve(err ? "" : String(stdout)));
    });
}

export async function detectGpu(): Promise<GpuInfo | null> {
    const query = await run("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]);
    const line = query.split(/\r?\n/).find((l) => l.trim());
    if (!line) return null;
    const [name, driver, mem] = line.split(",").map((s) => s.trim());
    const full = await run("nvidia-smi", []);
    const m = /CUDA Version:\s*([\d.]+)/.exec(full);
    return { name, driver, cudaVersion: m ? m[1] : null, memoryMb: mem ? Number(mem) : null };
}

/** ドライバーが対応する CUDA の版から、入れる PyTorch の種類（cu128 / cu126 / cu118 / cpu）を選ぶ。 */
export function torchVariant(gpu: GpuInfo | null): "cu128" | "cu126" | "cu118" | "cpu" {
    if (!gpu || !gpu.cudaVersion) return "cpu";
    const [major, minor] = gpu.cudaVersion.split(".").map(Number);
    const v = major * 100 + (minor || 0);
    if (v >= 1208) return "cu128";
    if (v >= 1206) return "cu126";
    if (v >= 1108) return "cu118";
    return "cpu";
}
