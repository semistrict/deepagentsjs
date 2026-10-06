import { defineConfig } from "tsdown";

// Mark npm packages and the kernel's WebAssembly as external: the wasm/
// directory ships beside dist/, where each host entry loads it its own way.
const external = (id: string) =>
  id.endsWith(".wasm") ||
  (!id.startsWith(".") && !id.startsWith("/") && !/^[A-Za-z]:[\\/]/.test(id));

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
