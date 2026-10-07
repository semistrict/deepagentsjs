/**
 * The durable runtime on Node: the kernel loaded from its WebAssembly file,
 * and session files on the local file system.
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { load } from "./load.js";

load(
  readFileSync(
    createRequire(import.meta.url).resolve("durable-wasm/durable_bg.wasm"),
  ),
);

export * from "./index.js";
export { NodeFiles } from "./files/node.js";
