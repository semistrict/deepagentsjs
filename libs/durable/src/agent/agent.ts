/**
 * The public agent: `createAgent` on a durable kernel instead of a graph.
 *
 * `DurableAgent` keeps the calling conventions of a `createAgent` agent:
 * `invoke` returns the final state, `stream` yields the same chunk shapes for
 * the `values`, `updates`, `messages`, and `custom` modes, `getState` reads a
 * thread back, and `updateState` changes it between runs. Leaving a stream
 * early aborts its run. A thread is a conversation in the kernel's session.
 *
 * An agent invoked from inside a tool call runs as a subagent: its own
 * conversation, owned by the tool call, streamed under the tool's namespace.
 * A subagent awaiting approval stops its tool call, and so its parent's run;
 * the parent's resume input reaches it when the tool call runs again.
 */
import { type RunnableConfig, mergeConfigs } from "@langchain/core/runnables";
import { AsyncLocalStorageProviderSingleton } from "@langchain/core/singletons";
import {
  Command,
  END,
  GraphRecursionError,
  type Interrupt,
  isCommand,
} from "@langchain/langgraph";
import { Kernel } from "../kernel.js";
import { END as END_EVENT } from "./events.js";
import {
  AgentRuntime,
  AwaitingInputError,
  type RunContext,
  type ToolScope,
  loaded,
} from "./runtime.js";
import { TOOL_KEY, isStepKey } from "./scope.js";
import { dump } from "./serde.js";
import { AgentSpec, type AgentParams } from "./spec.js";
import { create, find, scope as threadScope, type Update } from "./thread.js";

/** Conversation document family mapping a tool call ID to the subagent conversation it runs. */
const CHILDREN = "lc.children";
const RUNTIME = Symbol("deepagents-durable.agents");
const TAP = "durable_token_tap";

/** A run ended without an answer. */
export class RunFailedError extends Error {
  override name = "RunFailedError";
}

/** The agent runtime serving a kernel, created on first use. */
export function runtimeFor(kernel: Kernel): AgentRuntime {
  let runtime = kernel.services.get(RUNTIME) as AgentRuntime | undefined;
  if (runtime === undefined) {
    runtime = new AgentRuntime(kernel);
    kernel.services.set(RUNTIME, runtime);
  }
  return runtime;
}

let fallback: Promise<Kernel> | undefined;

/** The in-memory kernel agents without a checkpointer share. */
function defaultKernel(): Promise<Kernel> {
  fallback ??= Kernel.open();
  return fallback;
}

/** The tool call this code runs under, from the given config or the ambient one. */
function toolScope(config: RunnableConfig | undefined): ToolScope | undefined {
  const given = config?.configurable?.[TOOL_KEY] as ToolScope | undefined;
  if (given !== undefined) return given;
  const ambient = AsyncLocalStorageProviderSingleton.getRunnableConfig() as
    | RunnableConfig
    | undefined;
  return ambient?.configurable?.[TOOL_KEY] as ToolScope | undefined;
}

/** Callbacks without a parent step's token tap: a subagent streams its own tokens. */
function withoutTaps(
  callbacks: RunnableConfig["callbacks"],
): RunnableConfig["callbacks"] {
  if (Array.isArray(callbacks))
    return callbacks.filter(
      (handler) => (handler as { name?: string }).name !== TAP,
    );
  if (callbacks !== undefined && "copy" in callbacks) {
    const copy = callbacks.copy();
    for (const handler of [...copy.handlers])
      if (handler.name === TAP) copy.removeHandler(handler);
    return copy;
  }
  return callbacks;
}

/** A config without the keys the runtime sets per step. */
function storable(config: RunnableConfig): RunnableConfig {
  const configurable = Object.fromEntries(
    Object.entries(config.configurable ?? {}).filter(
      ([key]) => !isStepKey(key),
    ),
  );
  const metadata = Object.fromEntries(
    Object.entries(config.metadata ?? {}).filter(
      ([key]) => !key.startsWith("langgraph_") && key !== "checkpoint_ns",
    ),
  );
  const {
    context: _context,
    store: _store,
    writer: _writer,
    signal: _signal,
    ...rest
  } = config as RunnableConfig & Record<string, unknown>;
  return {
    ...rest,
    configurable,
    metadata,
    callbacks: withoutTaps(config.callbacks),
  };
}

function payload(input: unknown): {
  input?: unknown;
  command?: { update: unknown; resume: unknown };
} {
  if (isCommand(input)) {
    return {
      command: {
        update: input.update === undefined ? null : dump(input.update),
        resume: dump(input.resume),
      },
    };
  }
  return { input: dump(input) };
}

/** Where one invocation runs. */
interface Target {
  runtime: AgentRuntime;
  /** The agent's registry key: a subagent's is scoped under its parent's. */
  key: string;
  conversation: number;
  threadId: string | null;
  events: [number, string[]];
  scope: ToolScope | undefined;
}

export type StreamMode = "values" | "updates" | "messages" | "custom";

export interface StreamOptions extends RunnableConfig {
  streamMode?: StreamMode | StreamMode[];
  subgraphs?: boolean;
  context?: unknown;
}

export interface InvokeConfig extends RunnableConfig {
  context?: unknown;
}

/** A thread's state as `getState` reads it, shaped as LangGraph's state snapshot. */
export interface StateSnapshot {
  values: Record<string, unknown>;
  next: string[];
  config: RunnableConfig;
  metadata: Record<string, unknown> | undefined;
  createdAt: string | undefined;
  parentConfig: RunnableConfig | undefined;
  tasks: { id: string; name: string; interrupts: Interrupt[] }[];
}

/** A LangChain agent whose loop runs as durable tasks. */
export class DurableAgent {
  readonly name: string | undefined;
  readonly key: string;
  readonly config: RunnableConfig;

  constructor(
    readonly spec: AgentSpec,
    readonly checkpointer?: Kernel,
    config: RunnableConfig = {},
  ) {
    this.name = spec.name;
    this.key = spec.name ?? "agent";
    this.config = mergeConfigs(spec.config, config);
  }

  /** A copy with `config` merged into the bound config. */
  withConfig(config: RunnableConfig): DurableAgent {
    const merged = mergeConfigs(this.config, config);
    if (config.recursionLimit !== undefined)
      merged.recursionLimit = config.recursionLimit;
    return new DurableAgent(this.spec, this.checkpointer, merged);
  }

  async #target(
    config: RunnableConfig,
    scope: ToolScope | undefined,
  ): Promise<Target> {
    if (scope !== undefined) {
      const key = `${scope.agent}/${this.key}`;
      scope.runtime.register(key, this.spec);
      const conversation = await this.#child(scope);
      return {
        runtime: scope.runtime,
        key,
        conversation,
        threadId: null,
        events: [scope.thread, scope.ns],
        scope,
      };
    }
    const raw = config.configurable?.thread_id;
    const threadId = raw === undefined || raw === null ? null : String(raw);
    const runtime = runtimeFor(this.checkpointer ?? (await defaultKernel()));
    runtime.register(this.key, this.spec);
    const conversation = await this.#thread(runtime, threadId);
    return {
      runtime,
      key: this.key,
      conversation,
      threadId,
      events: [conversation, []],
      scope: undefined,
    };
  }

  async #thread(
    runtime: AgentRuntime,
    threadId: string | null,
  ): Promise<number> {
    const known = threadId === null ? undefined : runtime.threads.get(threadId);
    if (known !== undefined) return known;
    const conversation = await runtime.kernel.commit(
      (tx) =>
        (threadId === null ? null : find(tx, threadId)) ?? create(tx, threadId),
    );
    if (threadId !== null) runtime.threads.set(threadId, conversation);
    return conversation;
  }

  /** The subagent conversation of a tool call, created on its first run. */
  #child(scope: ToolScope): Promise<number> {
    return scope.runtime.kernel.commit((tx) => {
      const found = tx.doc(
        CHILDREN,
        threadScope(scope.conversation),
        scope.callId,
      ) as { conversationId: number } | null;
      if (found !== null) return found.conversationId;
      const child = tx.createConversation({
        owner: { conversationId: scope.conversation, taskId: scope.task },
      });
      tx.putDoc(
        CHILDREN,
        threadScope(scope.conversation),
        { conversationId: child },
        { key: scope.callId },
      );
      return child;
    });
  }

  /** The run's config: a subagent's starts from its tool call's ambient config. */
  #merged(
    config: RunnableConfig | undefined,
    scope: ToolScope | undefined,
  ): RunnableConfig {
    const ambient =
      scope === undefined
        ? {}
        : storable(
            (AsyncLocalStorageProviderSingleton.getRunnableConfig() as RunnableConfig) ??
              {},
          );
    return mergeConfigs(
      mergeConfigs(ambient, this.config),
      storable(config ?? {}),
    );
  }

  async #submit(
    target: Target,
    input: unknown,
    config: RunnableConfig,
    context: unknown,
  ) {
    const { scope } = target;
    let given = input;
    if (
      scope?.resume !== undefined &&
      scope.resume !== null &&
      target.runtime.state(target.conversation, this.spec).pending !== null
    ) {
      given = new Command({ resume: scope.resume });
    }
    const run: RunContext = {
      config,
      context: context ?? scope?.context.context,
    };
    return target.runtime.submit(
      target.key,
      target.conversation,
      payload(given),
      run,
      target.threadId,
      target.events,
    );
  }

  async #settled(
    target: Target,
    submission: number,
  ): Promise<Record<string, unknown>> {
    const record =
      await target.runtime.kernel.session.waitSubmission(submission);
    if (record.status === "done")
      return record.result as Record<string, unknown>;
    const detail = String(
      record.detail ?? record.reason ?? "the run ended without an answer",
    );
    if (detail.startsWith("GraphRecursionError: ")) {
      throw new GraphRecursionError(
        detail.slice("GraphRecursionError: ".length),
        { lc_error_code: "GRAPH_RECURSION_LIMIT" },
      );
    }
    throw new RunFailedError(detail);
  }

  #interrupts(target: Target): Interrupt[] {
    const pending = target.runtime.state(
      target.conversation,
      this.spec,
    ).pending;
    return pending === null ? [] : loaded(pending.interrupts);
  }

  /** Run until the agent answers or asks for input; returns the final state. */
  async invoke(
    input: unknown,
    config?: InvokeConfig,
  ): Promise<Record<string, any>> {
    const scope = toolScope(config);
    const merged = this.#merged(config, scope);
    const target = await this.#target(merged, scope);
    const { submission } = await this.#submit(
      target,
      input,
      merged,
      config?.context,
    );
    const result = await this.#settled(target, submission);
    const state = target.runtime.state(target.conversation, this.spec);
    const output = state.output();
    if (result.status === "interrupted") {
      if (scope !== undefined)
        throw new AwaitingInputError(state.pending!.interrupts);
      output.__interrupt__ = this.#interrupts(target);
    }
    return output;
  }

  /** Stream a run's events in LangGraph's chunk shapes; `streamMode` defaults to `"values"`. */
  async stream(
    input: unknown,
    options: StreamOptions = {},
  ): Promise<AsyncGenerator<any>> {
    const {
      streamMode = "values",
      subgraphs = false,
      context,
      ...config
    } = options;
    const modes = Array.isArray(streamMode) ? streamMode : [streamMode];
    const single = !Array.isArray(streamMode);
    const scope = toolScope(config);
    const merged = this.#merged(config, scope);
    const target = await this.#target(merged, scope);
    const subscription = target.runtime.bus.subscribe(target.events[0]);
    let run: number | undefined;
    let submission: number;
    try {
      ({ submission, run } = await this.#submit(
        target,
        input,
        merged,
        context,
      ));
    } catch (error) {
      subscription.close();
      throw error;
    }
    const base = target.events[1];
    const settled = () => this.#settled(target, submission);
    return (async function* chunks() {
      try {
        for (;;) {
          const { value: event } = await subscription.next();
          if (event.mode === END_EVENT && event.run === run) {
            run = undefined;
            break;
          }
          if (!modes.includes(event.mode as StreamMode)) continue;
          if (base.some((part, index) => event.ns[index] !== part)) continue;
          const ns = event.ns.slice(base.length);
          if (ns.length > 0 && !subgraphs) continue;
          if (subgraphs)
            yield single ? [ns, event.data] : [ns, event.mode, event.data];
          else yield single ? event.data : [event.mode, event.data];
        }
        await settled();
      } finally {
        subscription.close();
        // The caller stopped listening before the run ended: stop the run too.
        if (run !== undefined) await target.runtime.abort(run);
      }
    })();
  }

  /** A thread's current state; empty for a thread that does not exist yet. */
  async getState(config: RunnableConfig): Promise<StateSnapshot> {
    const merged = mergeConfigs(this.config, config);
    const target = await this.#target(merged, undefined);
    const state = target.runtime.state(target.conversation, this.spec);
    const seq = target.runtime.kernel.session.seq();
    const pending = state.pending;
    return {
      values: state.values(),
      next: pending === null ? [] : [pending.node],
      config: {
        configurable: {
          thread_id: target.threadId,
          checkpoint_ns: "",
          checkpoint_id: String(seq),
        },
      },
      metadata: { source: "loop", step: seq },
      createdAt: undefined,
      parentConfig: undefined,
      tasks:
        pending === null
          ? []
          : [
              {
                id: String(target.conversation),
                name: pending.node,
                interrupts: loaded(pending.interrupts),
              },
            ],
    };
  }

  /**
   * Change a thread between runs, merging `values` as a step's updates would.
   * `asNode` set to `END` also settles the thread: what a stopped run awaits
   * is dropped. Other node names are accepted and ignored, since there are no
   * nodes. A run in progress is waited for first.
   */
  async updateState(
    config: RunnableConfig,
    values: Record<string, unknown> | null,
    asNode?: string,
  ): Promise<RunnableConfig> {
    const target = await this.#target(
      mergeConfigs(this.config, config),
      undefined,
    );
    const updates = Object.entries(values ?? {}) as Update[];
    await target.runtime.update(
      target.conversation,
      this.spec,
      updates,
      asNode === END,
    );
    const seq = target.runtime.kernel.session.seq();
    return {
      configurable: {
        thread_id: target.threadId,
        checkpoint_ns: "",
        checkpoint_id: String(seq),
      },
    };
  }
}

export interface CreateAgentParams extends AgentParams {
  /** Where the threads live. Without one, a subagent uses its parent's kernel, and others a shared in-memory one. */
  checkpointer?: Kernel;
}

/** Build an agent like `langchain`'s `createAgent`, running on a durable kernel. */
export function createAgent(params: CreateAgentParams): DurableAgent {
  const { checkpointer, ...rest } = params;
  return new DurableAgent(new AgentSpec(rest), checkpointer);
}
