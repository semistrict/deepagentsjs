/**
 * Instantiating the kernel's WebAssembly. Each host entry point loads it the
 * way its platform delivers modules: Node reads the file, a Worker imports it.
 */
import { initSync } from "../wasm/durable.js";

let loaded = false;

/** Instantiate the kernel once per process or isolate; later calls do nothing. */
export function load(module: WebAssembly.Module | BufferSource): void {
  if (loaded) return;
  initSync({ module });
  loaded = true;
}

/** Fail unless a host entry point loaded the kernel. */
export function assertLoaded(): void {
  if (!loaded) {
    throw new Error(
      'The durable kernel is not loaded: import "deepagents-durable/node" or "deepagents-durable/cloudflare" first.',
    );
  }
}
