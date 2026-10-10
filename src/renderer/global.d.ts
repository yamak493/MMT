import type { MmtApi } from "../shared/types";

declare global {
    interface Window {
        mmt: MmtApi;
    }
}

export {};
