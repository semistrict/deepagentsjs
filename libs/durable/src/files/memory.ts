/**
 * A session's SQLite files in memory, surviving the session that wrote them.
 */
import type { SessionFiles } from "../../wasm/durable.js";

/**
 * Files kept in this object: closing a session and opening another on the
 * same path and files sees everything it committed, as a restart would.
 */
export class MemoryFiles implements SessionFiles {
  readonly #files = new Map<string, Uint8Array>();

  #file(name: string): Uint8Array {
    const file = this.#files.get(name);
    if (file === undefined) throw new Error(`${name} does not exist`);
    return file;
  }

  exists(name: string): boolean {
    return this.#files.has(name);
  }

  create(name: string): void {
    this.#files.set(name, new Uint8Array(0));
  }

  delete(name: string): void {
    this.#files.delete(name);
  }

  read(name: string, into: Uint8Array, offset: number): number {
    const available = this.#file(name).subarray(offset, offset + into.length);
    into.set(available);
    return available.length;
  }

  write(name: string, data: Uint8Array, offset: number): void {
    let file = this.#file(name);
    if (offset + data.length > file.length) {
      const grown = new Uint8Array(offset + data.length);
      grown.set(file);
      file = grown;
      this.#files.set(name, file);
    }
    file.set(data, offset);
  }

  truncate(name: string, size: number): void {
    const file = this.#file(name);
    const resized = new Uint8Array(size);
    resized.set(file.subarray(0, size));
    this.#files.set(name, resized);
  }

  sync(): void {}

  size(name: string): number {
    return this.#file(name).length;
  }
}
