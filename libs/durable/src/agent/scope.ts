/**
 * What code running inside one step of a run can reach.
 *
 * Hooks, model calls, and tools run inside a {@link StepScope}: the thread's
 * state, the updates sent mid-step (a backend writing a file before its tool
 * returns), and the answers to the step's earlier `interrupt()` calls.
 *
 * Middleware written for `createAgent` reach these through LangGraph helpers
 * that read the ambient config: `interrupt()`, `getCurrentTaskInput()`,
 * `getConfig()`, and `StateBackend`'s file reads and writes. So the scope is
 * also exposed as that config, under the few keys those helpers read.
 */
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { BaseMessage, BaseMessageChunk } from "@langchain/core/messages";
import type { ChatGenerationChunk } from "@langchain/core/outputs";
import { type RunnableConfig, mergeConfigs } from "@langchain/core/runnables";
import { AsyncLocalStorageProviderSingleton } from "@langchain/core/singletons";
import "./context.js";
import type { Update } from "./thread.js";
import type { ThreadState } from "./thread.js";

/** Config keys the LangGraph helpers read. */
const SCRATCHPAD = "__pregel_scratchpad";
const SEND = "__pregel_send";
const READ = "__pregel_read";
const CHECKPOINTER = "__pregel_checkpointer";
/** The write `interrupt()` makes to record answers; the runtime holds them already. */
const RESUME = "__resume__";
/** Chat model runs tagged this way are not streamed, as in LangGraph. */
const NOSTREAM = "langsmith:nostream";

/** Where this runtime keeps its own scope in the ambient config. */
export const STEP_KEY = "__durable_step";
export const TOOL_KEY = "__durable_tool";

/** Keys this runtime and LangGraph's helpers set per step; never carried into another run. */
export function isStepKey(key: string): boolean {
  return (
    key.startsWith("__pregel_") ||
    key.startsWith("__durable_") ||
    key === "checkpoint_ns" ||
    key === "checkpoint_id"
  );
}

export type TokenSink = (
  chunk: BaseMessageChunk,
  metadata: Record<string, unknown>,
) => void;

/**
 * Forwards streamed chat model chunks with their run's metadata, as
 * LangGraph's messages stream does. Preferring streaming makes chat models
 * stream. Tokens are not durable: the committed message is.
 */
class TokenTap extends BaseCallbackHandler {
  name = "durable_token_tap";
  lc_prefer_streaming = true;
  readonly #runs = new Map<string, Record<string, unknown>>();

  constructor(readonly sink: TokenSink) {
    super();
  }

  override handleChatModelStart(
    _llm: unknown,
    _messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    _extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
  ): void {
    if (!tags?.includes(NOSTREAM))
      this.#runs.set(runId, { ...metadata, tags: tags ?? [] });
  }

  override handleLLMNewToken(
    _token: string,
    _index: unknown,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    fields?: { chunk?: ChatGenerationChunk | unknown },
  ): void {
    const metadata = this.#runs.get(runId);
    const chunk = fields?.chunk as ChatGenerationChunk | undefined;
    if (metadata !== undefined && chunk !== undefined && "message" in chunk) {
      this.sink(chunk.message as BaseMessageChunk, metadata);
    }
  }

  override handleLLMEnd(_output: unknown, runId: string): void {
    this.#runs.delete(runId);
  }

  override handleLLMError(_error: unknown, runId: string): void {
    this.#runs.delete(runId);
  }
}

export interface StepOptions {
  state: ThreadState;
  /** The step's node name, as stream metadata and interrupt IDs know it. */
  node: string;
  /** The loop step number, as `langgraph_step`. */
  step: number;
  /** Namespace of this step: its parents' plus `node:id`. Interrupt IDs hash it. */
  ns: string;
  threadId: string | null;
  agentName: string | undefined;
  context: unknown;
  store: unknown;
  writer: (chunk: unknown) => void;
  signal: AbortSignal | undefined;
  tokens: TokenSink;
  /** Answers to the step's `interrupt()` calls, from earlier runs of it. */
  answers?: unknown[];
  /** Extra entries for the ambient configurable, such as a tool call's scope. */
  configurable?: Record<string, unknown>;
}

/** One step's view of its thread, and the updates it sends before returning. */
export class StepScope {
  readonly sent: Update[] = [];

  constructor(readonly options: StepOptions) {}

  /** Queue updates the step applies with its result. */
  send(updates: Update[]): void {
    this.sent.push(...updates);
  }

  /** A field as this step sees it: the thread's value plus the step's own sent updates. */
  read(name: string): unknown {
    const { state } = this.options;
    const current = name === "messages" ? state.messages : state.fields[name];
    const own = this.sent
      .filter(([sent]) => sent === name)
      .map(([, value]) => value);
    return own.length === 0 ? current : state.schema.merge(name, current, own);
  }

  /** The runtime LangChain middleware and tools receive. */
  runtime(context: unknown = this.options.context): Record<string, unknown> {
    const { store, writer, signal, threadId } = this.options;
    return Object.freeze({
      context,
      store,
      writer,
      signal,
      configurable: threadId === null ? {} : { thread_id: threadId },
    });
  }

  /** The ambient config for this step: the run's config plus where the step is. */
  config(base: RunnableConfig): RunnableConfig {
    const {
      state,
      node,
      step,
      ns,
      threadId,
      agentName,
      context,
      store,
      writer,
      signal,
      tokens,
      answers,
    } = this.options;
    const scratchpad = {
      callCounter: 0,
      interruptCounter: -1,
      subgraphCounter: 0,
      resume: [...(answers ?? [])],
      nullResume: undefined,
      consumeNullResume: () => undefined,
      currentTaskInput: state.values(),
    };
    const configurable: Record<string, unknown> = {
      ...this.options.configurable,
      [STEP_KEY]: this,
      [SCRATCHPAD]: scratchpad,
      [SEND]: (writes: Update[]) =>
        this.send(writes.filter(([name]) => name !== RESUME)),
      [READ]: (names: string | string[]) =>
        Array.isArray(names)
          ? Object.fromEntries(names.map((name) => [name, this.read(name)]))
          : this.read(names),
      // `interrupt()` refuses to run without one; answers come from the scratchpad instead.
      [CHECKPOINTER]: true,
      checkpoint_ns: ns,
    };
    const metadata: Record<string, unknown> = {
      langgraph_step: step,
      langgraph_node: node,
      langgraph_checkpoint_ns: ns,
      checkpoint_ns: ns,
    };
    if (threadId !== null)
      configurable.thread_id = metadata.thread_id = threadId;
    if (agentName !== undefined) metadata.lc_agent_name = agentName;
    const merged = mergeConfigs(base, {
      configurable,
      metadata,
      callbacks: [new TokenTap(tokens)],
    } as RunnableConfig);
    return Object.assign(merged, { context, store, writer, signal });
  }

  /** Run `call(config)` with this scope's config as the ambient one. */
  run<T>(
    base: RunnableConfig,
    call: (config: RunnableConfig) => Promise<T>,
  ): Promise<T> {
    const config = this.config(base);
    return AsyncLocalStorageProviderSingleton.runWithConfig(config, () =>
      call(config),
    );
  }
}

/** The step scope of the code running now, if it runs inside one. */
export function currentStep(): StepScope | undefined {
  const config = AsyncLocalStorageProviderSingleton.getRunnableConfig() as
    | RunnableConfig
    | undefined;
  return config?.configurable?.[STEP_KEY] as StepScope | undefined;
}
