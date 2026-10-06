/**
 * A Worker whose Durable Object runs the kernel over its own storage, for
 * the Cloudflare tests. Each request runs one scenario step and answers JSON.
 */
import {
  DurableObjectFiles,
  Kernel,
  type SqlStorage,
  type TaskInvocation,
} from "../cloudflare.js";

interface State {
  storage: { sql: SqlStorage };
}

interface Namespace {
  idFromName(name: string): unknown;
  get(id: unknown): { fetch(request: Request): Promise<Response> };
}

export class KernelObject {
  #kernel: Promise<Kernel> | undefined;

  constructor(readonly state: State) {}

  #open(): Promise<Kernel> {
    this.#kernel ??= Kernel.open({
      path: "session.sqlite",
      files: new DurableObjectFiles(this.state.storage.sql),
    }).then((kernel) => {
      kernel.register({
        kind: "count",
        phases: {
          start: async (invocation: TaskInvocation<{ to: number }>) => {
            for (let n = 1; n <= invocation.input.to; n += 1) {
              await invocation.step((step) => {
                step.tx.appendEntry(invocation.conversation, "count", {
                  data: n,
                });
                if (n < invocation.input.to)
                  step.advance({ phase: "start", n });
                else step.finish({ counted: n });
              });
            }
          },
        },
        abort: async (invocation) => {
          await invocation.step((step) => step.aborted());
        },
      });
      return kernel;
    });
    return this.#kernel;
  }

  async fetch(request: Request): Promise<Response> {
    const action = new URL(request.url).pathname.slice(1);
    const kernel = await this.#open();
    switch (action) {
      case "write": {
        const entry = await kernel.commit((tx) => {
          const root = tx.conversation(1) === null ? tx.createRoot() : 1;
          tx.putDoc(
            "app.big",
            { kind: "session" },
            { text: "x".repeat(100_000) },
          );
          return tx.appendEntry(root, "note", { data: "kept" });
        });
        return Response.json({ entry });
      }
      case "read":
        return Response.json({
          entries:
            kernel.session.conversation(1) === null
              ? []
              : kernel.session.entries(1).map(({ entry }) => entry.data),
          big:
            kernel.session.doc("app.big", { kind: "session" })?.value.text
              .length ?? null,
        });
      case "reopen": {
        await kernel.close();
        this.#kernel = undefined;
        const reopened = await this.#open();
        return Response.json({
          entries: reopened.session.entries(1).map(({ entry }) => entry.data),
        });
      }
      case "count": {
        const task = await kernel.commit((tx) => {
          const conversation = tx.createConversation();
          return tx.createTask(
            conversation,
            "count",
            { to: 3 },
            { phase: "start" },
          );
        });
        const settled = await kernel.session.waitTask(task);
        return Response.json({
          state: settled.state,
          entries: kernel.session
            .entries(settled.conversationId)
            .map(({ entry }) => entry.data),
        });
      }
      default:
        return new Response(`unknown action ${action}`, { status: 404 });
    }
  }
}

export default {
  fetch(request: Request, env: { KERNEL: Namespace }): Promise<Response> {
    const name = new URL(request.url).searchParams.get("object") ?? "default";
    return env.KERNEL.get(env.KERNEL.idFromName(name)).fetch(request);
  },
};
