/**
 * The durable runtime in a Cloudflare Worker: the kernel imported as a
 * WebAssembly module, and session files in a Durable Object's storage.
 */
import wasm from "durable-wasm/durable_bg.wasm";
import { load } from "./load.js";

load(wasm);

export * from "./index.js";
export { DurableObjectFiles, type SqlStorage } from "./files/durable-object.js";
