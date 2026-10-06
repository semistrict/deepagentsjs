import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { builtinModules } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { Miniflare } from "miniflare";
import { afterAll, beforeAll, describe, expect, test } from "vitest";

const here = fileURLToPath(new URL(".", import.meta.url));

interface Fixture {
  /** The Worker's entry module in src/testing/. */
  entry: string;
  binding: string;
  className: string;
}

/**
 * A test Worker in workerd, as Cloudflare runs it: bundled, with the
 * kernel's WebAssembly as a separate module, its Durable Objects on SQLite
 * storage persisted to a directory that outlives a restart.
 */
class TestWorker {
  readonly persist = mkdtempSync(join(tmpdir(), "durable-do-"));
  #script: string | undefined;
  #miniflare: Miniflare | undefined;

  constructor(readonly fixture: Fixture) {}

  async start(): Promise<void> {
    if (this.#script === undefined) {
      const bundled = await build({
        entryPoints: [join(here, "testing", this.fixture.entry)],
        bundle: true,
        format: "esm",
        platform: "neutral",
        conditions: ["workerd", "worker", "browser"],
        mainFields: ["module", "main"],
        write: false,
        // Node built-ins come from workerd's `nodejs_compat`, as Wrangler leaves them.
        external: ["*.wasm", "node:*", ...builtinModules],
        // The bundle sits in src/, beside the wasm/ directory's relative import.
        outdir: here,
        // CommonJS dependencies `require` Node built-ins, which ES modules reach through `createRequire`.
        banner: {
          js: 'import { createRequire } from "node:module"; const require = createRequire("/");',
        },
      });
      this.#script = bundled.outputFiles[0].text;
    }
    this.#miniflare = new Miniflare({
      modules: [
        { type: "ESModule", path: "src/worker.js", contents: this.#script },
        {
          type: "CompiledWasm",
          path: "wasm/durable_bg.wasm",
          contents: readFileSync(join(here, "..", "wasm", "durable_bg.wasm")),
        },
      ],
      compatibilityDate: "2026-08-01",
      compatibilityFlags: ["nodejs_compat"],
      durableObjects: {
        [this.fixture.binding]: {
          className: this.fixture.className,
          useSQLite: true,
        },
      },
      durableObjectsPersist: this.persist,
    });
  }

  /** Stop the runtime, keeping the objects' storage, and start it again. */
  async restart(): Promise<void> {
    await this.#miniflare?.dispose();
    await this.start();
  }

  async call(action: string, object = "default"): Promise<any> {
    const response = await this.#miniflare!.dispatchFetch(
      `http://worker/${action}?object=${object}`,
    );
    if (!response.ok)
      throw new Error(`${action}: ${response.status} ${await response.text()}`);
    return response.json();
  }

  async dispose(): Promise<void> {
    await this.#miniflare?.dispose();
    rmSync(this.persist, { recursive: true, force: true });
  }
}

describe("the kernel on a Cloudflare Durable Object", () => {
  const worker = new TestWorker({
    entry: "durable-object-worker.ts",
    binding: "KERNEL",
    className: "KernelObject",
  });
  beforeAll(() => worker.start());
  afterAll(() => worker.dispose());

  test("commits land in the object's storage and survive reopening the session", async () => {
    await worker.call("write", "reopen");
    expect(await worker.call("reopen", "reopen")).toEqual({
      entries: ["kept"],
    });
  });

  test("the session survives the runtime restarting", async () => {
    await worker.call("write", "restart");
    await worker.call("write", "restart");
    await worker.restart();
    expect(await worker.call("read", "restart")).toEqual({
      entries: ["kept", "kept"],
      big: 100_000,
    });
  });

  test("the scheduler runs a task's phases on the object's event loop", async () => {
    expect(await worker.call("count", "count")).toEqual({
      state: {
        status: "terminal",
        outcome: { status: "completed", result: { counted: 3 } },
      },
      entries: [1, 2, 3],
    });
  });

  test("objects keep separate sessions", async () => {
    await worker.call("write", "first");
    expect(await worker.call("read", "second")).toEqual({
      entries: [],
      big: null,
    });
  });
});

describe("a Deep Agent on a Cloudflare Durable Object", () => {
  const worker = new TestWorker({
    entry: "agent-object-worker.ts",
    binding: "AGENT",
    className: "AgentObject",
  });
  beforeAll(() => worker.start());
  afterAll(() => worker.dispose());

  test("asks for approval, survives a restart while waiting, and finishes", async () => {
    const first = await worker.call("start");
    expect(first.asked).toEqual(["write_file"]);
    await worker.restart();
    const second = await worker.call("approve");
    expect([second.asked, second.files]).toEqual([
      ["write_file"],
      ["/notes.md"],
    ]);
    const done = await worker.call("approve");
    expect(done).toEqual({
      messages: [
        ["human", "take notes"],
        ["ai", ""],
        ["tool", "Successfully wrote to '/notes.md'"],
        ["ai", ""],
        ["tool", "Successfully wrote to '/more.md'"],
        ["ai", "Both written."],
      ],
      files: ["/notes.md", "/more.md"],
      asked: [],
    });
  });
});
