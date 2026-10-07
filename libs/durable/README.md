# deepagents-durable

Experimental. Deep Agents running on a durable kernel instead of LangGraph's
graph execution and checkpointer, on Node and in Cloudflare Durable Objects.

The kernel is the Rust crate `durable-core` from
[deepagents](https://github.com/semistrict/deepagents/tree/semistrict/sdk/durable-runtime/libs/durable),
whose `crates/wasm` compiles it to WebAssembly and binds it with
`wasm-bindgen`. That build ships as the `durable-wasm` package, a release
asset of deepagents this package depends on by URL. It stores
conversations, immutable entries, durable tasks, submissions, and JSON
documents in a SQLite file in pi-durable's format, and schedules tasks on the
host's event loop. The agent loop runs as durable tasks on it: a crash repeats
at most the step in flight, and a run stopped for human input resumes by
running only the step that asked.

In the kernel SQLite runs inline, so reads and commits are synchronous calls;
only waiting for the mutation line, for a task or a submission, and for frames
return promises. SQLite's files are kept by the host through `SessionFiles`.

## Entry points

- `deepagents-durable/node`: the kernel, loaded from its file, and `NodeFiles`
  for sessions on disk.
- `deepagents-durable/cloudflare`: the kernel, imported as a WebAssembly
  module, and `DurableObjectFiles` for sessions in a Durable Object's storage.
- `deepagents-durable/agent`: `createAgent` and `createDeepAgent`, whose loops
  run on the kernel. Middleware, tools, and models written for `langchain`'s
  `createAgent` run unchanged.

## Use

```ts
import { Kernel, NodeFiles } from "deepagents-durable/node";
import { createDeepAgent } from "deepagents-durable/agent";

const kernel = await Kernel.open({ path: "threads.sqlite", files: new NodeFiles() });
const agent = createDeepAgent({ model: "anthropic:claude-sonnet-4-6", checkpointer: kernel });
await agent.invoke({ messages: [{ role: "user", content: "hi" }] }, { configurable: { thread_id: "t1" } });
```

In a Durable Object, open the kernel over the object's storage. The agent
needs the `nodejs_compat` compatibility flag for `AsyncLocalStorage`:

```ts
import { DurableObjectFiles, Kernel } from "deepagents-durable/cloudflare";

const kernel = await Kernel.open({ path: "agent.sqlite", files: new DurableObjectFiles(ctx.storage.sql) });
```

`deepagents`' own `createDeepAgent` takes an experimental `agentFactory`;
`agentFactory(kernel)` from `deepagents-durable/agent` runs any agent it
assembles, and its declarative subagents, on the kernel.

## Kernel changes

The kernel is built in deepagents: `make wasm` in `libs/durable` writes the
package and its tarball to `crates/wasm/pkg`. To try a local build, point the
`durable-wasm` dependency at that tarball (`file:`). To ship one, bump
`crates/wasm`'s version and push a `durable-wasm-v<version>` tag; CI publishes
the tarball as that release's asset, and the dependency here moves to its URL.
