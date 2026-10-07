import { defineConfig } from "tsdown";

// Mark npm packages as external, the kernel's (`durable-wasm`) among them:
// each host entry loads its WebAssembly its own way.
const external = (id: string) =>
  !id.startsWith(".") && !id.startsWith("/") && !/^[A-Za-z]:[\\/]/.test(id);

export default defineConfig({
  entry: [
    "./src/index.ts",
    "./src/node.ts",
    "./src/cloudflare.ts",
    "./src/agent/index.ts",
  ],
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: true,
  outDir: "dist",
  outExtensions: () => ({ js: ".js" }),
  external,
});
