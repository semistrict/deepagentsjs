/**
 * The durable runtime on Node: the kernel loaded from its WebAssembly file,
 * and session files on the local file system.
 */
import { readFileSync } from "node:fs";
import { load } from "./load.js";

load(readFileSync(new URL("../wasm/durable_bg.wasm", import.meta.url)));

export * from "./index.js";
export { NodeFiles } from "./files/node.js";
