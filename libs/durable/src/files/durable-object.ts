/**
 * A session's SQLite files in a Cloudflare Durable Object's storage.
 */
import type { SessionFiles } from "../../wasm/durable.js";

/** The part of a Durable Object's `ctx.storage.sql` the files use. */
export interface SqlStorage {
  exec(
    query: string,
    ...bindings: unknown[]
  ): { toArray(): Record<string, unknown>[] };
}

/** Bytes per stored block. SQLite pages and WAL frames span one or two. */
const BLOCK = 32 * 1024;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS durable_files (
  name TEXT PRIMARY KEY,
  size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS durable_file_blocks (
  name TEXT NOT NULL,
  block INTEGER NOT NULL,
  data BLOB NOT NULL,
  PRIMARY KEY (name, block)
);
`;

/**
 * Files stored as fixed-size blocks in the Durable Object's SQLite storage,
 * through its synchronous SQL API.
 *
 * Writes land in the object's storage as they happen, and the platform makes
 * them durable before the object's next output leaves, so `sync` has nothing
 * to do. A Durable Object is single-threaded and the only writer of its
 * storage, which makes it the session file's one owner.
 */
export class DurableObjectFiles implements SessionFiles {
  constructor(readonly sql: SqlStorage) {
    for (const statement of SCHEMA.split(";").map((part) => part.trim())) {
      if (statement) sql.exec(statement);
    }
  }

  #size(name: string): number | undefined {
    const [row] = this.sql
      .exec("SELECT size FROM durable_files WHERE name = ?", name)
      .toArray();
    return row === undefined ? undefined : Number(row.size);
  }

  #block(name: string, block: number): Uint8Array | undefined {
    const [row] = this.sql
      .exec(
        "SELECT data FROM durable_file_blocks WHERE name = ? AND block = ?",
        name,
        block,
      )
      .toArray();
    return row === undefined
      ? undefined
      : new Uint8Array(row.data as ArrayBuffer);
  }

  exists(name: string): boolean {
    return this.#size(name) !== undefined;
  }

  create(name: string): void {
    this.sql.exec(
      "INSERT INTO durable_files (name, size) VALUES (?, 0) ON CONFLICT (name) DO UPDATE SET size = 0",
      name,
    );
    this.sql.exec("DELETE FROM durable_file_blocks WHERE name = ?", name);
  }

  delete(name: string): void {
    this.sql.exec("DELETE FROM durable_files WHERE name = ?", name);
    this.sql.exec("DELETE FROM durable_file_blocks WHERE name = ?", name);
  }

  read(name: string, into: Uint8Array, offset: number): number {
    const size = this.size(name);
    const end = Math.min(size, offset + into.length);
    for (let position = offset; position < end;) {
      const block = Math.floor(position / BLOCK);
      const within = position - block * BLOCK;
      const length = Math.min(BLOCK - within, end - position);
      const data = this.#block(name, block);
      const target = into.subarray(
        position - offset,
        position - offset + length,
      );
      // A block never written reads as zeros, as a sparse file does.
      if (data === undefined) target.fill(0);
      else {
        const available = data.subarray(within, within + length);
        target.set(available);
        target.fill(0, available.length);
      }
      position += length;
    }
    return Math.max(0, end - offset);
  }

  write(name: string, data: Uint8Array, offset: number): void {
    for (let position = offset; position < offset + data.length;) {
      const block = Math.floor(position / BLOCK);
      const within = position - block * BLOCK;
      const length = Math.min(BLOCK - within, offset + data.length - position);
      const existing = this.#block(name, block);
      const stored = new Uint8Array(
        Math.max(existing?.length ?? 0, within + length),
      );
      if (existing !== undefined) stored.set(existing);
      stored.set(
        data.subarray(position - offset, position - offset + length),
        within,
      );
      this.sql.exec(
        "INSERT INTO durable_file_blocks (name, block, data) VALUES (?, ?, ?) ON CONFLICT (name, block) DO UPDATE SET data = excluded.data",
        name,
        block,
        stored,
      );
      position += length;
    }
    const size = this.size(name);
    if (offset + data.length > size) {
      this.sql.exec(
        "UPDATE durable_files SET size = ? WHERE name = ?",
        offset + data.length,
        name,
      );
    }
  }

  truncate(name: string, size: number): void {
    const last = Math.ceil(size / BLOCK);
    this.sql.exec(
      "DELETE FROM durable_file_blocks WHERE name = ? AND block >= ?",
      name,
      last,
    );
    const within = size - (last - 1) * BLOCK;
    if (last > 0 && within < BLOCK) {
      const tail = this.#block(name, last - 1);
      if (tail !== undefined && tail.length > within) {
        this.sql.exec(
          "UPDATE durable_file_blocks SET data = ? WHERE name = ? AND block = ?",
          tail.slice(0, within),
          name,
          last - 1,
        );
      }
    }
    this.sql.exec(
      "UPDATE durable_files SET size = ? WHERE name = ?",
      size,
      name,
    );
  }

  sync(): void {}

  size(name: string): number {
    const size = this.#size(name);
    if (size === undefined) throw new Error(`${name} does not exist`);
    return size;
  }
}
