import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import {
  Kernel,
  MemoryFiles,
  NodeFiles,
  isDurableError,
  type Task,
  type TaskInvocation,
} from "./node.js";

const root = { kind: "session" } as const;

/** Resolves with the next value `notify` is called with. */
function signal<T = void>(): { notify: (value: T) => void; next: Promise<T> } {
  let notify!: (value: T) => void;
  const next = new Promise<T>((resolve) => (notify = resolve));
  return { notify, next };
}

/** Resolves once the invocation's signal aborts, then throws as a cancelled handler does. */
function untilAborted(invocation: TaskInvocation): Promise<never> {
  return new Promise((_, reject) => {
    const stop = () => reject(invocation.signal.reason);
    if (invocation.signal.aborted) stop();
    else invocation.signal.addEventListener("abort", stop, { once: true });
  });
}

const aborted = {
  abort: async (invocation: TaskInvocation) => {
    await invocation.step((step) => step.aborted("stopped"));
  },
};

async function start(
  kernel: Kernel,
  kind: string,
  checkpoint: object,
): Promise<number> {
  return kernel.commit((tx) => {
    const conversation = tx.createRoot();
    return tx.createTask(conversation, kind, null, checkpoint);
  });
}

const kernels: Kernel[] = [];
async function open(...args: Parameters<typeof Kernel.open>): Promise<Kernel> {
  const kernel = await Kernel.open(...args);
  kernels.push(kernel);
  return kernel;
}

afterEach(async () => {
  for (const kernel of kernels.splice(0)) await kernel.close();
});

describe("transactions", () => {
  test("documents store deltas and read back as of any commit", async () => {
    const kernel = await open();
    const conversation = await kernel.commit((tx) => tx.createRoot());
    const scope = {
      kind: "conversation",
      conversationId: conversation,
    } as const;
    const seqs: number[] = [];
    for (const partial of ["Par", "Paris", "Paris is"]) {
      const frame = await kernel.commit((tx) => {
        tx.putDoc("app.live", scope, { partial }, { history: "rewindable" });
        return kernel.session.seq() + 1;
      });
      seqs.push(frame);
    }
    expect(
      seqs.map(
        (at) => kernel.session.doc("app.live", scope, undefined, at)?.value,
      ),
    ).toEqual([
      { partial: "Par" },
      { partial: "Paris" },
      { partial: "Paris is" },
    ]);
  });

  test("a commit returns its frame, with document deltas", async () => {
    const kernel = await open();
    const tx = await kernel.session.transaction();
    tx.putDoc("app.note", root, { text: "Par" });
    tx.commit();
    tx.free();
    const update = await kernel.session.transaction();
    update.putDoc("app.note", root, { text: "Paris" });
    const frame = update.commit();
    update.free();
    expect(
      frame.docs.map(({ type, ...change }) => [
        type,
        "ops" in change ? change.ops : undefined,
      ]),
    ).toEqual([["updated", [["a", ["text"], "is"]]]]);
  });

  test("reads inside a transaction see its own document writes, and nothing after rollback", async () => {
    const kernel = await open();
    const tx = await kernel.session.transaction();
    tx.putDoc("app.note", root, { text: "draft" });
    expect(tx.doc("app.note", root)).toEqual({ text: "draft" });
    tx.rollback();
    tx.free();
    expect(kernel.session.doc("app.note", root)).toBeNull();
  });

  test("a second transaction waits for the first to commit", async () => {
    const kernel = await open();
    const first = await kernel.session.transaction();
    let second = false;
    const waiting = kernel.session.transaction().then((tx) => {
      second = true;
      return tx;
    });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(second).toBe(false);
    first.createRoot();
    first.commit();
    const tx = await waiting;
    expect(tx.conversation(1)).toEqual({ id: 1 });
    tx.rollback();
  });

  test("a table read after a table write fails", async () => {
    const kernel = await open();
    const error = await kernel
      .commit((tx) => {
        tx.createRoot();
        return tx.conversation(1);
      })
      .catch((caught: unknown) => caught);
    expect(isDurableError(error, "ReadAfterWriteError")).toBe(true);
  });

  test("a transaction body that returns a promise writes nothing", async () => {
    const kernel = await open();
    await expect(kernel.commit(async (tx) => tx.createRoot())).rejects.toThrow(
      "a transaction's body returned a promise",
    );
    expect(kernel.session.conversation(1)).toBeNull();
  });

  test("subscribers see every commit's frame in order", async () => {
    const kernel = await open();
    const frames = kernel.session.subscribe();
    await kernel.commit((tx) => tx.createRoot());
    await kernel.commit((tx) => tx.appendEntry(1, "note", { data: "hi" }));
    const first = await frames.next();
    const second = await frames.next();
    expect([first?.conversations, second?.entries]).toEqual([
      [{ id: 1 }],
      [{ id: 2, conversationId: 1, kind: "note", data: "hi" }],
    ]);
  });
});

describe("files", () => {
  let dir: string;
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("a session file on disk is SQLite in WAL mode and reopens with its data", async () => {
    dir = mkdtempSync(join(tmpdir(), "durable-node-"));
    const path = join(dir, "threads", "session.sqlite");
    const kernel = await Kernel.open({ path, files: new NodeFiles() });
    await kernel.commit((tx) => {
      tx.createRoot();
      tx.appendEntry(1, "note", { data: "kept" });
    });
    await kernel.close();

    const header = readFileSync(path).subarray(0, 20);
    expect([header.subarray(0, 15).toString(), header[18], header[19]]).toEqual(
      ["SQLite format 3", 2, 2],
    );
    const reopened = await open({ path, files: new NodeFiles() });
    expect(reopened.session.entries(1).map(({ entry }) => entry.data)).toEqual([
      "kept",
    ]);
  });

  test("a path has one open session at a time", async () => {
    dir = mkdtempSync(join(tmpdir(), "durable-node-"));
    const path = join(dir, "session.sqlite");
    const owner = await Kernel.open({ path, files: new NodeFiles() });
    const second = await Kernel.open({ path, files: new NodeFiles() }).catch(
      (caught: unknown) => caught,
    );
    expect(isDurableError(second, "SessionLockedError")).toBe(true);
    await owner.close();
    await (await Kernel.open({ path, files: new NodeFiles() })).close();
  });

  test("a path needs files to keep it", async () => {
    dir = mkdtempSync(join(tmpdir(), "durable-node-"));
    await expect(
      Kernel.open({ path: join(dir, "session.sqlite") }),
    ).rejects.toThrow("a session with a path needs `files` to keep it");
  });
});

describe("scheduler", () => {
  test("a reopened session resumes a task at its last checkpoint", async () => {
    const files = new MemoryFiles();
    const reached = signal();
    const kernel = await Kernel.open({ path: "session.sqlite", files });
    kernel.register({
      kind: "work",
      ...aborted,
      phases: {
        start: async (invocation) => {
          await invocation.step((step) =>
            step.advance({ phase: "effect", saved: "before the crash" }),
          );
          reached.notify();
          await untilAborted(invocation);
        },
      },
    });
    const task = await start(kernel, "work", { phase: "start" });
    await reached.next;
    await kernel.close();

    const reopened = await open({ path: "session.sqlite", files });
    expect(reopened.session.task(task)?.state).toEqual({
      status: "pending",
      checkpoint: { phase: "effect", saved: "before the crash" },
    });
    reopened.register({
      kind: "work",
      ...aborted,
      phases: {
        effect: async (invocation) => {
          const saved = invocation.checkpoint.saved;
          await invocation.step((step) => step.finish(saved));
        },
      },
    });
    expect((await reopened.session.waitTask(task)).state).toEqual({
      status: "terminal",
      outcome: { status: "completed", result: "before the crash" },
    });
  });

  test("aborting a task stops its running work and runs abort handlers bottom-up", async () => {
    const kernel = await open();
    const order: string[] = [];
    const childStarted = signal();
    kernel.register({
      kind: "child",
      phases: {
        block: async (invocation) => {
          await invocation.step((step) =>
            step.advance({ phase: "block", started: true }),
          );
          childStarted.notify();
          await untilAborted(invocation);
        },
      },
      abort: async (invocation) => {
        order.push("child");
        await aborted.abort(invocation);
      },
    });
    kernel.register({
      kind: "parent",
      phases: {
        spawn: async (invocation) => {
          await invocation.step((step) => {
            const child = step.tx.createTask(
              invocation.conversation,
              "child",
              null,
              { phase: "block" },
              { owner: invocation.id },
            );
            step.wait([child], { phase: "never" });
          });
        },
      },
      abort: async (invocation) => {
        order.push("parent");
        await aborted.abort(invocation);
      },
    });
    const parent = await start(kernel, "parent", { phase: "spawn" });
    await childStarted.next;
    await kernel.commit((tx) => tx.abortTask(parent));

    const settled = await kernel.session.waitTask(parent);
    expect(order).toEqual(["child", "parent"]);
    expect(settled.state).toEqual({
      status: "terminal",
      outcome: { status: "aborted", reason: "stopped" },
    });
  });

  test("a failing phase faults its task inside a commit that runs the fault hook", async () => {
    const kernel = await open();
    kernel.register({
      kind: "faulty",
      ...aborted,
      phases: {
        start: async () => {
          throw new RangeError("cannot continue");
        },
      },
      onFault: (tx, task: Task, message) => {
        tx.appendEntry(task.conversationId, "fault", { data: message });
      },
    });
    const task = await start(kernel, "faulty", { phase: "start" });
    const settled = await kernel.session.waitTask(task);
    expect(settled.state).toEqual({
      status: "terminal",
      outcome: {
        status: "faulted",
        error: { message: "RangeError: cannot continue" },
      },
    });
    expect(
      kernel.session
        .entries(settled.conversationId)
        .map(({ entry }) => entry.data),
    ).toEqual(["RangeError: cannot continue"]);
  });

  test("a phase that returns without committing progress faults", async () => {
    const kernel = await open();
    kernel.register({
      kind: "idle",
      ...aborted,
      phases: { start: async () => {} },
    });
    const task = await start(kernel, "idle", { phase: "start" });
    expect((await kernel.session.waitTask(task)).state).toEqual({
      status: "terminal",
      outcome: {
        status: "faulted",
        error: {
          message: `task ${task} returned from a phase without committing progress`,
        },
      },
    });
  });
});
