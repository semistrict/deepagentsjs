/**
 * A session's SQLite files on the local file system.
 */
import {
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  ftruncateSync,
  mkdirSync,
  openSync,
  readSync,
  rmSync,
  writeSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import type { SessionFiles } from "../../wasm/durable.js";

/**
 * Files on disk, each held open from first use until the session closes.
 *
 * One process should own a session file at a time: a second process writing
 * the same file corrupts it. Within one process the kernel refuses a second
 * open of the same path.
 */
export class NodeFiles implements SessionFiles {
  readonly #open = new Map<string, number>();

  /** Files on one machine are the same file wherever their path resolves to. */
  identity(path: string): string {
    return resolve(path);
  }

  #fd(name: string): number {
    let fd = this.#open.get(name);
    if (fd === undefined) {
      fd = openSync(name, "r+");
      this.#open.set(name, fd);
    }
    return fd;
  }

  exists(name: string): boolean {
    return this.#open.has(name) || existsSync(name);
  }

  create(name: string): void {
    mkdirSync(dirname(name), { recursive: true });
    this.#open.set(name, openSync(name, "w+"));
  }

  delete(name: string): void {
    const fd = this.#open.get(name);
    if (fd !== undefined) {
      this.#open.delete(name);
      closeSync(fd);
    }
    rmSync(name, { force: true });
  }

  read(name: string, into: Uint8Array, offset: number): number {
    const fd = this.#fd(name);
    let total = 0;
    while (total < into.length) {
      const read = readSync(
        fd,
        into,
        total,
        into.length - total,
        offset + total,
      );
      if (read === 0) break;
      total += read;
    }
    return total;
  }

  write(name: string, data: Uint8Array, offset: number): void {
    const fd = this.#fd(name);
    let total = 0;
    while (total < data.length) {
      total += writeSync(fd, data, total, data.length - total, offset + total);
    }
  }

  truncate(name: string, size: number): void {
    ftruncateSync(this.#fd(name), size);
  }

  sync(name: string): void {
    fsyncSync(this.#fd(name));
  }

  size(name: string): number {
    return fstatSync(this.#fd(name)).size;
  }

  close(): void {
    for (const fd of this.#open.values()) closeSync(fd);
    this.#open.clear();
  }
}
