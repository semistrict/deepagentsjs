/** Workers and their bundlers import WebAssembly files as compiled modules. */
declare module "*.wasm" {
  const module: WebAssembly.Module;
  export default module;
}
