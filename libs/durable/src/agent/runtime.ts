/**
 * The agent loop as durable tasks.
 *
 * A run is one `lc.run` task per submitted input, carrying that input. The
 * loop is three sequences of steps:
 *
 * - `agent`: the `beforeAgent` hooks, once per run;
 * - `turn`: the `beforeModel` hooks, the model call through every
 *   `wrapModelCall` middleware, and the `afterModel` hooks, in reverse;
 * - `finish`: the `afterAgent` hooks, in reverse.
 *
 * The run's phases drive them, each ending in one atomic commit of its state
 * changes and the next phase. `start` places the input and runs `agent` and
 * the first `turn`; when the model calls tools, the same commit creates one
 * `lc.tool` child task per call and the run waits for them. `collect`
 * applies the tool results in call order and runs the next `turn`; `finish`
 * runs its sequence. Where the loop goes after each step follows
 * `createAgent`'s routing, including hooks' `jumpTo`.
 *
 * The input and tool results are durable before the phase that uses them,
 * so a crash repeats at most the phase in flight. A step that calls
 * `interrupt()` stops the run instead, keeping what the steps before it did;
 * the next input carries the answer, and the run continues by running that
 * step again (see `scope.ts`). A tool call that asks, or whose subagent asks,
 * stops its round the same way.
 */
import {
  AIMessage,
  type BaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { type RunnableConfig, mergeConfigs } from "@langchain/core/runnables";
import {
  Command,
  GraphBubbleUp,
  GraphRecursionError,
  type Interrupt,
  isCommand,
  isGraphInterrupt,
} from "@langchain/langgraph";
import type { AgentMiddleware } from "langchain";
import type { Step, Tx } from "../../wasm/durable.js";
import type { Kernel } from "../kernel.js";
import type { TaskInvocation } from "../tasks.js";
import { END as END_EVENT, EventBus, type Event } from "./events.js";
import { StepScope, TOOL_KEY } from "./scope.js";
import { dump, load } from "./serde.js";
import {
  type AgentSpec,
  contextFor,
  commandUpdates,
  type JumpTo,
  MODEL,
  TOOLS,
} from "./spec.js";
import {
  Change,
  loadThread,
  PENDING,
  scope as threadScope,
  type ThreadState,
  type Update,
} from "./thread.js";

export const RUN = "lc.run";
export const TOOL = "lc.tool";
/** Threads whose state stays in memory between runs. */
const WARM_THREADS = 64;
/** Not a phase: the run settles in the same commit. */
const DONE = "done";

type Sequence = "agent" | "turn" | "finish";
/** Where the loop goes next: a sequence from its start, the model call, the tool round, or the end. */
type Next = "turn" | "model" | "tools" | "finish" | typeof DONE;

/** A subagent stopped for human input; the tool call that runs it stops with it. */
export class AwaitingInputError extends GraphBubbleUp {
  override name = "AwaitingInputError";

  constructor(readonly interrupts: StoredInterrupt[]) {
    super("a subagent is awaiting input");
  }

  static isInstance(error: unknown): error is AwaitingInputError {
    return (
      typeof error === "object" &&
      error !== null &&
      (error as { name?: unknown }).name === "AwaitingInputError" &&
      Array.isArray((error as { interrupts?: unknown }).interrupts)
    );
  }
}

/** A thread already has a run in progress. */
export class ThreadBusyError extends Error {
  override name = "ThreadBusyError";
}

/** What a run uses that cannot be stored: the caller's config objects and context. */
export interface RunContext {
  config: RunnableConfig;
  context?: unknown;
}

/** Set while a tool call runs, so an agent the tool invokes runs as its subagent. */
export interface ToolScope {
  runtime: AgentRuntime;
  task: number;
  conversation: number;
  callId: string;
  thread: number;
  ns: string[];
  resume: unknown;
  context: RunContext;
  /** The registry key of the agent whose tool this is. */
  agent: string;
}

export interface StoredInterrupt {
  id: string | undefined;
  value: unknown;
}

interface RunInput {
  agent: string;
  threadId: string | null;
  thread: number;
  ns: string[];
  submission: number;
  payload: { input?: unknown; command?: { update: unknown; resume: unknown } };
}

interface ToolInput extends Omit<RunInput, "payload"> {
  step: number;
  call: ToolCall;
  answers?: unknown[];
  resume?: unknown;
}

/** A run's state while this process drives it. */
interface Live {
  state: ThreadState;
  key: string;
  /** IDs of messages already streamed in messages mode. */
  seen: Set<string>;
}

/** One phase's work: its change, and the node updates to stream once it commits. */
class Work {
  readonly change = new Change();
  readonly nodes: [string, Update[]][] = [];
  steps = 0;

  record(state: ThreadState, node: string, updates: Update[]): void {
    this.change.extend(state.apply(updates));
    this.nodes.push([node, updates]);
  }
}

/** A step asked for input: the questions, and where the run continues once answered. */
class Stopped {
  constructor(
    readonly interrupts: StoredInterrupt[],
    readonly sequence: Sequence,
    readonly at: number,
    readonly node: string,
    readonly answers: unknown[],
  ) {}
}

type Stage =
  | {
      hook: "beforeAgent" | "beforeModel" | "afterModel" | "afterAgent";
      middleware: AgentMiddleware;
    }
  | { hook: "model" };

function stages(spec: AgentSpec, sequence: Sequence): Stage[] {
  if (sequence === "agent")
    return spec.hooks.beforeAgent.map((middleware) => ({
      hook: "beforeAgent",
      middleware,
    }));
  if (sequence === "turn") {
    return [
      ...spec.hooks.beforeModel.map((middleware) => ({
        hook: "beforeModel" as const,
        middleware,
      })),
      { hook: "model" },
      ...[...spec.hooks.afterModel]
        .reverse()
        .map((middleware) => ({ hook: "afterModel" as const, middleware })),
    ];
  }
  return [...spec.hooks.afterAgent]
    .reverse()
    .map((middleware) => ({ hook: "afterAgent", middleware }));
}

function nodeName(stage: Stage): string {
  if (stage.hook === "model") return MODEL;
  const suffix = {
    beforeAgent: "before_agent",
    beforeModel: "before_model",
    afterModel: "after_model",
    afterAgent: "after_agent",
  }[stage.hook];
  return `${stage.middleware.name}.${suffix}`;
}

/** Updates as the dict stream consumers see; repeated list fields concatenate. */
function merged(updates: Update[]): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [name, value] of updates) {
    const existing = result[name];
    result[name] =
      Array.isArray(existing) && Array.isArray(value)
        ? [...existing, ...value]
        : value;
  }
  return result;
}

function lastAi(messages: BaseMessage[]): AIMessage | undefined {
  return messages
    .filter((message): message is AIMessage => AIMessage.isInstance(message))
    .at(-1);
}

/** Tool calls of the last AI message that have no result yet, other than structured output. */
function pendingCalls(spec: AgentSpec, messages: BaseMessage[]): ToolCall[] {
  const ai = lastAi(messages);
  if (ai === undefined) return [];
  const answered = new Set(
    messages
      .filter(ToolMessage.isInstance)
      .map((message) => message.tool_call_id),
  );
  return (ai.tool_calls ?? []).filter(
    (call) => !answered.has(call.id!) && !spec.isOutputTool(call.name),
  );
}

function hasCalls(message: BaseMessage | undefined): message is AIMessage {
  return AIMessage.isInstance(message) && (message.tool_calls?.length ?? 0) > 0;
}

/** The answer in a resume payload for one interrupt: keyed by its ID, or the whole payload. */
function resumeValue(resume: unknown, id: string | undefined): unknown {
  if (
    id !== undefined &&
    resume !== null &&
    typeof resume === "object" &&
    id in resume
  ) {
    return (resume as Record<string, unknown>)[id];
  }
  return resume;
}

function stored(interrupt: Interrupt): StoredInterrupt {
  return { id: interrupt.id, value: dump(interrupt.value) };
}

export function loaded(interrupts: StoredInterrupt[]): Interrupt[] {
  return interrupts.map((item) => ({ id: item.id, value: load(item.value) }));
}

/** A hook's result as state updates, and where it moves the loop. */
function hookResult(
  result: unknown,
  middleware: AgentMiddleware,
  jumps: JumpTo[],
): { updates: Update[]; jump?: JumpTo } {
  if (result === undefined || result === null) return { updates: [] };
  let values = result as Record<string, unknown>;
  let jump = values.jumpTo as JumpTo | undefined;
  if (typeof jump === "string" && !jumps.includes(jump)) {
    const suggestion =
      jumps.length > 0
        ? `must be one of: ${jumps.join(", ")}.`
        : `no canJumpTo defined in middleware ${middleware.name}`;
    throw new Error(`Invalid jump target: ${jump}, ${suggestion}.`);
  }
  if ("type" in values) {
    if (values.type !== "terminate")
      throw new Error(`Invalid control action: ${JSON.stringify(values)}`);
    if (values.error) throw values.error;
    jump = values.jumpTo as JumpTo | undefined;
    values = (values.result ?? {}) as Record<string, unknown>;
  }
  const updates = Object.entries(values).filter(
    ([name, value]) => name !== "jumpTo" && value !== undefined,
  ) as Update[];
  return { updates, jump: jump ?? undefined };
}

/**
 * Runs agents' loops as durable tasks on one kernel.
 *
 * A kernel's session has one owner, this process, so the runtime keeps
 * what it knows in memory: warm thread states, which threads are busy, and
 * which conversation each thread ID maps to.
 */
export class AgentRuntime {
  readonly bus = new EventBus();
  readonly #agents = new Map<string, AgentSpec>();
  readonly #waiting = new Map<string, ((spec: AgentSpec) => void)[]>();
  readonly #contexts = new Map<number, RunContext>();
  readonly #live = new Map<number, Live>();
  /** Each warm thread's state, most recently used last. */
  readonly #warm = new Map<string, ThreadState>();
  /** The run in progress on each busy conversation. */
  readonly #busy = new Map<number, number>();
  #recovered = false;
  /** The conversation of each thread ID seen so far; the mapping never changes. */
  readonly threads = new Map<string, number>();

  constructor(readonly kernel: Kernel) {
    kernel.register({
      kind: RUN,
      phases: {
        start: (invocation) => this.#start(invocation),
        turn: (invocation) => this.#turn(invocation),
        collect: (invocation) => this.#collect(invocation),
        finish: (invocation) => this.#finish(invocation),
      },
      abort: (invocation) => this.#abortRun(invocation),
      onFault: (tx, task, message) =>
        this.#runFaulted(
          tx,
          task.id,
          task.conversationId,
          task.input as RunInput,
          message,
        ),
    });
    kernel.register({
      kind: TOOL,
      phases: { call: (invocation) => this.#toolCall(invocation) },
      abort: (invocation) => invocation.step((step) => step.aborted()),
    });
  }

  /** Make an agent runnable under `key`; runs waiting for it proceed. */
  register(key: string, spec: AgentSpec): void {
    this.#agents.set(key, spec);
    for (const resolve of this.#waiting.get(key) ?? []) resolve(spec);
    this.#waiting.delete(key);
  }

  /** The agent registered under `key`, waiting until one is. */
  agent(key: string): Promise<AgentSpec> {
    const spec = this.#agents.get(key);
    if (spec !== undefined) return Promise.resolve(spec);
    return new Promise((resolve) => {
      const waiting = this.#waiting.get(key) ?? [];
      waiting.push(resolve);
      this.#waiting.set(key, waiting);
    });
  }

  /** Learn, once, which conversations have runs left over from an earlier process. */
  #recoverBusy(): void {
    if (this.#recovered) return;
    this.#recovered = true;
    for (const task of this.kernel.session.tasks({ kind: RUN, live: true })) {
      if (!this.#busy.has(task.conversationId))
        this.#busy.set(task.conversationId, task.id);
    }
  }

  /** The run in progress on a conversation, if any. */
  active(conversation: number): number | undefined {
    this.#recoverBusy();
    const run = this.#busy.get(conversation);
    if (run === undefined) return undefined;
    // The run may have just settled and not yet cleaned up after itself.
    const task = this.kernel.session.task(run);
    return task !== null && task.state.status !== "terminal" ? run : undefined;
  }

  /**
   * Admit a run on an idle conversation; returns its submission and run task.
   *
   * @throws ThreadBusyError when the conversation already has a run in progress.
   */
  async submit(
    key: string,
    conversation: number,
    payload: RunInput["payload"],
    context: RunContext,
    threadId: string | null,
    events: [number, string[]] = [conversation, []],
  ): Promise<{ submission: number; run: number }> {
    const busy = this.active(conversation);
    if (busy !== undefined)
      throw new ThreadBusyError(
        `thread ${threadId ?? conversation} is busy with run ${busy}`,
      );
    const [thread, ns] = events;
    const admitted = await this.kernel.commit((tx) => {
      const submission = tx.createSubmission(conversation, { type: "input" });
      const input: RunInput = {
        agent: key,
        threadId,
        thread,
        ns,
        payload,
        submission,
      };
      const run = tx.createTask(conversation, RUN, input, {
        phase: "start",
        step: 0,
      });
      return { submission, run };
    });
    this.#busy.set(conversation, admitted.run);
    this.#contexts.set(admitted.run, context);
    return admitted;
  }

  /** Abort a run and its tool calls, and wait until it has settled. */
  async abort(run: number): Promise<void> {
    const task = this.kernel.session.task(run);
    if (task !== null && task.state.status !== "terminal") {
      await this.kernel.commit((tx) => tx.abortTask(run));
      await this.kernel.session.waitTask(run);
    }
  }

  /**
   * Change a thread between runs, as if a step had made `updates`. `settle`
   * also drops what a stopped run awaits, so the thread starts fresh at its
   * next input. A run in progress is waited for first.
   */
  async update(
    conversation: number,
    spec: AgentSpec,
    updates: Update[],
    settle: boolean,
  ): Promise<void> {
    const busy = this.active(conversation);
    if (busy !== undefined) await this.kernel.session.waitTask(busy);
    const state = this.state(conversation, spec);
    const change = state.apply(updates);
    const retire = settle && state.pending !== null;
    if (retire) state.pending = null;
    try {
      await this.kernel.commit((tx) => {
        state.persist(tx, conversation, change);
        if (retire) tx.retireDoc(PENDING, threadScope(conversation));
      });
    } catch (error) {
      this.#warm.delete(this.#key(conversation, spec));
      throw error;
    }
  }

  #key(conversation: number, spec: AgentSpec): string {
    return `${conversation}:${spec.schemaKey}`;
  }

  /** A thread's current state, kept warm between runs since only this process writes it. */
  state(conversation: number, spec: AgentSpec): ThreadState {
    const key = this.#key(conversation, spec);
    let state = this.#warm.get(key);
    if (state === undefined) {
      state = loadThread(this.kernel.session, conversation, spec.schema);
      while (this.#warm.size >= WARM_THREADS)
        this.#warm.delete(this.#warm.keys().next().value!);
    } else {
      this.#warm.delete(key);
    }
    this.#warm.set(key, state);
    return state;
  }

  // Shared plumbing.

  #context(run: number): RunContext {
    return this.#contexts.get(run) ?? { config: {} };
  }

  #config(run: number, spec: AgentSpec): RunnableConfig {
    return mergeConfigs(spec.config, this.#context(run).config);
  }

  #liveState(invocation: TaskInvocation, spec: AgentSpec): Live {
    const run = invocation.task.owner ?? invocation.id;
    let live = this.#live.get(run);
    if (live === undefined) {
      live = {
        state: this.state(invocation.conversation, spec),
        key: this.#key(invocation.conversation, spec),
        seen: new Set(),
      };
      this.#live.set(run, live);
    }
    return live;
  }

  /** Drop a run's state, and the thread's warm copy, which it may have changed without committing. */
  #forget(run: number): void {
    const live = this.#live.get(run);
    this.#live.delete(run);
    if (live !== undefined) this.#warm.delete(live.key);
  }

  #scope(
    invocation: TaskInvocation<RunInput | ToolInput>,
    spec: AgentSpec,
    live: Live,
    node: string,
    id: string,
    step: number,
    answers: unknown[] = [],
    configurable?: Record<string, unknown>,
  ): StepScope {
    const run = invocation.task.owner ?? invocation.id;
    const input = invocation.input;
    const config = this.#context(run).config;
    const signals = [invocation.signal, config.signal].filter(
      (signal): signal is AbortSignal => signal !== undefined,
    );
    return new StepScope({
      state: live.state,
      node,
      step,
      ns: [...input.ns, `${node}:${id}`].join("|"),
      threadId: input.threadId,
      agentName: spec.name,
      context: this.#context(run).context,
      store: spec.params.store,
      writer: (chunk) => this.#publish(invocation, "custom", chunk),
      signal: signals.length === 1 ? signals[0] : AbortSignal.any(signals),
      answers,
      configurable,
      tokens: (chunk, metadata) => {
        if (chunk.id) live.seen.add(chunk.id);
        this.#publish(invocation, "messages", [chunk, metadata]);
      },
    });
  }

  #publish(
    invocation: TaskInvocation<RunInput | ToolInput>,
    mode: string,
    data: unknown,
  ): void {
    const input = invocation.input;
    const event: Event = {
      thread: input.thread,
      ns: input.ns,
      mode,
      data,
      run: invocation.id,
    };
    this.bus.publish(event);
  }

  /** Stream a committed phase: its node updates, its new messages, and the values. */
  #published(
    invocation: TaskInvocation<RunInput>,
    live: Live,
    work: Work,
    metadata: Record<string, unknown>,
  ): void {
    for (const [node, updates] of work.nodes) {
      // Each tool call streamed its own update when it finished.
      if (node !== TOOLS)
        this.#publish(invocation, "updates", { [node]: merged(updates) });
    }
    for (const [operation, message] of work.change.messages) {
      if (
        (operation === "add" || operation === "replace") &&
        !live.seen.has(message.id!)
      ) {
        live.seen.add(message.id!);
        this.#publish(invocation, "messages", [message, metadata]);
      }
    }
    this.#publish(invocation, "values", live.state.output());
  }

  #ended(invocation: TaskInvocation<RunInput>): void {
    this.#busy.delete(invocation.conversation);
    this.#live.delete(invocation.id);
    this.#contexts.delete(invocation.id);
    this.#publish(invocation, END_EVENT, null);
  }

  #settled(
    invocation: TaskInvocation<RunInput>,
    settlement: Record<string, unknown>,
  ) {
    return {
      id: invocation.input.submission,
      conversationId: invocation.conversation,
      type: "input",
      ...settlement,
    } as never;
  }

  #step(invocation: TaskInvocation<RunInput>): number {
    return (invocation.checkpoint.step as number | undefined) ?? 0;
  }

  /** Create a task for each pending tool call and wait for them, in the step's commit. */
  #spawn(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    step: Step,
    number: number,
    done: Record<string, unknown>,
    answers: Record<string, Partial<ToolInput>> = {},
  ): void {
    const calls = pendingCalls(spec, live.state.messages);
    const { agent, threadId, thread, ns, submission } = invocation.input;
    const tasks = calls
      .filter((call) => !(call.id! in done))
      .map((call) => {
        const input: ToolInput = {
          agent,
          threadId,
          thread,
          ns,
          submission,
          step: number + 1,
          call,
          ...answers[call.id!],
        };
        return step.tx.createTask(
          invocation.conversation,
          TOOL,
          input,
          { phase: "call" },
          { owner: invocation.id },
        );
      });
    const following = {
      phase: "collect",
      step: number,
      calls: calls.map((call) => call.id),
      tasks,
      done,
    };
    if (tasks.length > 0) step.wait(tasks, following);
    else step.advance(following);
  }

  /** Store a phase's change and move the run on: to tools, a phase, or the end. */
  async #commit(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    next: Next,
    at: number,
    retirePending = false,
  ): Promise<void> {
    if (retirePending) live.state.pending = null;
    const number = this.#step(invocation) + work.steps;
    // Read before the commit, which may end the task and its checkpoint with it.
    const metadata = this.#metadata(invocation, spec, live, number);
    try {
      await invocation.step((step) => {
        live.state.persist(
          step.tx,
          invocation.conversation,
          work.change,
          invocation.id,
        );
        if (retirePending)
          step.tx.retireDoc(PENDING, threadScope(invocation.conversation));
        if (next === DONE) {
          step.tx.putSubmission(
            this.#settled(invocation, {
              status: "done",
              result: { status: "success" },
            }),
          );
          step.finish({ status: "success" });
        } else if (next === "tools") {
          this.#spawn(invocation, spec, live, step, number, {});
        } else if (next === "model") {
          step.advance({
            phase: "turn",
            step: number,
            at: this.#modelAt(spec),
          });
        } else {
          step.advance({ phase: next, step: number, at });
        }
      });
    } catch (error) {
      this.#forget(invocation.id);
      throw error;
    }
    this.#published(invocation, live, work, metadata);
    if (next === DONE) this.#ended(invocation);
  }

  /** Stop the run awaiting input, saving what it waits for and where it continues. */
  async #stop(
    invocation: TaskInvocation<RunInput>,
    live: Live,
    work: Work,
    pending: Record<string, unknown>,
  ): Promise<void> {
    live.state.pending = pending;
    try {
      await invocation.step((step) => {
        live.state.persist(
          step.tx,
          invocation.conversation,
          work.change,
          invocation.id,
        );
        step.tx.putDoc(PENDING, threadScope(invocation.conversation), pending);
        step.tx.putSubmission(
          this.#settled(invocation, {
            status: "done",
            result: { status: "interrupted" },
          }),
        );
        step.finish({ status: "interrupted" });
      });
    } catch (error) {
      this.#forget(invocation.id);
      throw error;
    }
    this.#published(invocation, live, work, {});
    const interrupts = loaded(pending.interrupts as StoredInterrupt[]);
    this.#publish(invocation, "updates", { __interrupt__: interrupts });
    this.#publish(invocation, "values", { __interrupt__: interrupts });
    this.#ended(invocation);
  }

  #metadata(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    number: number,
  ): Record<string, unknown> {
    const scope = this.#scope(
      invocation,
      spec,
      live,
      MODEL,
      String(invocation.id),
      number,
    );
    return {
      ...(scope.config(this.#config(invocation.id, spec)).metadata ?? {}),
    };
  }

  #modelAt(spec: AgentSpec): number {
    return spec.hooks.beforeModel.length;
  }

  // Steps and sequences.

  async #hookStep(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    stage: Exclude<Stage, { hook: "model" }>,
    number: number,
    answers: unknown[],
  ): Promise<JumpTo | undefined> {
    const node = nodeName(stage);
    const scope = this.#scope(
      invocation,
      spec,
      live,
      node,
      `${number}`,
      number,
      answers,
    );
    const { run, jumps } = spec.hook(stage.middleware, stage.hook);
    const result = await scope.run(
      this.#config(invocation.id, spec),
      async () =>
        run(
          live.state.values(),
          scope.runtime(contextFor(stage.middleware, scope.options.context)),
        ),
    );
    const { updates, jump } = hookResult(result, stage.middleware, jumps);
    work.record(live.state, node, [...scope.sent, ...updates]);
    return jump;
  }

  /** The model step; returns whether its output asks for another model call. */
  async #model(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    number: number,
    answers: unknown[],
  ): Promise<boolean> {
    const last = live.state.messages.at(-1);
    // A direct-return tool just answered: there is nothing to ask the model, as in `createAgent`.
    if (
      ToolMessage.isInstance(last) &&
      last.name !== undefined &&
      spec.returnDirect.has(last.name)
    ) {
      work.record(live.state, MODEL, []);
      return false;
    }
    const scope = this.#scope(
      invocation,
      spec,
      live,
      MODEL,
      `${number}`,
      number,
      answers,
    );
    const outcome = await scope.run(
      this.#config(invocation.id, spec),
      (config) => spec.callModel(live.state.values(), scope.runtime(), config),
    );
    work.record(live.state, MODEL, [...scope.sent, ...outcome.updates]);
    return outcome.retry;
  }

  /** Run a sequence's steps from `at`; returns where the loop goes next, or why it stopped. */
  async #sequence(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    sequence: Sequence,
    at: number,
    answers: unknown[],
  ): Promise<Next | Stopped> {
    const all = stages(spec, sequence);
    let retry = false;
    for (let index = at; index < all.length; index += 1) {
      const stage = all[index];
      const number = this.#step(invocation) + work.steps + 1;
      this.#checkLimit(invocation, number);
      const given = index === at ? answers : [];
      let jump: JumpTo | undefined;
      try {
        if (stage.hook === "model")
          retry = await this.#model(
            invocation,
            spec,
            live,
            work,
            number,
            given,
          );
        else
          jump = await this.#hookStep(
            invocation,
            spec,
            live,
            work,
            stage,
            number,
            given,
          );
      } catch (error) {
        if (!isGraphInterrupt(error)) throw error;
        return new Stopped(
          error.interrupts.map(stored),
          sequence,
          index,
          nodeName(stage),
          given,
        );
      }
      work.steps += 1;
      if (jump !== undefined)
        return this.#jumped(spec, live.state, stage, index, all, jump);
    }
    return this.#following(spec, live.state, sequence, retry);
  }

  /** Where a hook's jump moves the loop, as `createAgent`'s routers send it. */
  #jumped(
    spec: AgentSpec,
    state: ThreadState,
    stage: Stage,
    index: number,
    all: Stage[],
    jump: JumpTo,
  ): Next {
    const lastToRun = index === all.length - 1;
    const exit: Next = spec.hooks.afterAgent.length > 0 ? "finish" : DONE;
    switch (stage.hook) {
      case "beforeAgent":
        if (jump === "end") return exit;
        if (jump === "tools") return spec.hasTools ? "tools" : exit;
        return "model";
      case "beforeModel":
        if (jump === "end") return DONE;
        if (jump === "tools") return spec.hasTools ? "tools" : DONE;
        return "model";
      case "afterModel": {
        if (!lastToRun) {
          if (jump === "end") return DONE;
          if (jump === "tools") return spec.hasTools ? "tools" : DONE;
          return "model";
        }
        // The last `afterModel` hook to run: a final answer ends the loop whatever it asks.
        const last = state.messages.at(-1);
        if (AIMessage.isInstance(last) && !hasCalls(last)) return exit;
        if (jump === "end") return exit;
        if (jump === "tools") return spec.hasTools ? "tools" : exit;
        return "model";
      }
      case "afterAgent":
        if (jump === "end") return DONE;
        if (jump === "tools") return spec.hasTools ? "tools" : DONE;
        return "model";
      default:
        throw new Error(`a ${stage.hook} step cannot jump`);
    }
  }

  /** Where the loop goes after a sequence runs through, as `createAgent` routes it. */
  #following(
    spec: AgentSpec,
    state: ThreadState,
    sequence: Sequence,
    retry: boolean,
  ): Next {
    const exit: Next = spec.hooks.afterAgent.length > 0 ? "finish" : DONE;
    if (sequence === "agent") return "turn";
    if (sequence === "finish") return DONE;
    if (retry) return "model";
    const messages = state.messages;
    const last = messages.at(-1);
    if (spec.hooks.afterModel.length === 0) {
      // The model router: a final answer or structured output ends the loop; tool calls run.
      if (!hasCalls(last)) return exit;
      if (last.tool_calls!.every((call) => spec.isOutputTool(call.name)))
        return exit;
      return spec.hasTools ? "tools" : exit;
    }
    // The router after the last `afterModel` hook.
    if (AIMessage.isInstance(last) && !hasCalls(last)) return exit;
    if (pendingCalls(spec, messages).length > 0)
      return spec.hasTools ? "tools" : exit;
    const ai = lastAi(messages);
    const structured =
      ai?.tool_calls?.some((call) => spec.isOutputTool(call.name)) ?? false;
    if (
      !structured &&
      spec.params.responseFormat !== undefined &&
      ai !== undefined
    )
      return "model";
    if (!hasCalls(last)) return exit;
    if (last.tool_calls!.every((call) => spec.isOutputTool(call.name)))
      return exit;
    return spec.hasTools ? "tools" : exit;
  }

  /** Where the loop goes after a tool round: back to the model, unless a direct-return tool answered last. */
  #afterTools(spec: AgentSpec, state: ThreadState): Next {
    const last = state.messages.at(-1);
    if (
      ToolMessage.isInstance(last) &&
      last.name !== undefined &&
      spec.returnDirect.has(last.name)
    ) {
      if (spec.params.responseFormat !== undefined) return "turn";
      return spec.hooks.afterAgent.length > 0 ? "finish" : DONE;
    }
    return "turn";
  }

  /**
   * Run sequences from `sequence` until the run must commit, then commit or stop.
   * The `agent` sequence flows straight into the first turn; any other
   * destination is the next phase.
   */
  async #drive(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    sequence: Sequence,
    at = 0,
    answers: unknown[] = [],
    retirePending = false,
  ): Promise<void> {
    let given = answers;
    for (;;) {
      const next = await this.#sequence(
        invocation,
        spec,
        live,
        work,
        sequence,
        at,
        given,
      );
      if (typeof next !== "string") {
        await this.#stop(invocation, live, work, {
          kind: "interrupt",
          sequence: next.sequence,
          at: next.at,
          node: next.node,
          step: this.#step(invocation) + work.steps,
          answers: next.answers.map(dump),
          interrupts: next.interrupts,
        });
        return;
      }
      // Within a phase the loop keeps going through turns until it needs tools or ends.
      if (next === "turn" || next === "model") {
        if (sequence === "agent") {
          sequence = "turn";
          at = next === "model" ? this.#modelAt(spec) : 0;
          given = [];
          continue;
        }
      }
      const target =
        next === "finish" && spec.hooks.afterAgent.length === 0 ? DONE : next;
      await this.#commit(
        invocation,
        spec,
        live,
        work,
        target,
        0,
        retirePending,
      );
      return;
    }
  }

  #checkLimit(invocation: TaskInvocation<RunInput>, number: number): void {
    const run = invocation.task.owner ?? invocation.id;
    const limit = this.#context(run).config.recursionLimit ?? 25;
    if (number > limit) {
      throw new GraphRecursionError(
        `Recursion limit of ${limit} reached without hitting a stop condition. You can increase the limit by setting the "recursionLimit" config key.`,
        { lc_error_code: "GRAPH_RECURSION_LIMIT" },
      );
    }
  }

  // Run phases.

  async #start(invocation: TaskInvocation<RunInput>): Promise<void> {
    const spec = await this.agent(invocation.input.agent);
    const live = this.#liveState(invocation, spec);
    const { payload } = invocation.input;
    const command = payload.command;
    const work = new Work();
    const defaults = Object.entries(spec.schema.defaults()).filter(
      ([name]) => !(name in live.state.fields),
    ) as Update[];
    if (defaults.length > 0) work.record(live.state, "__start__", defaults);
    const input = (load(payload.input) ?? {}) as Record<string, unknown>;
    work.record(
      live.state,
      "__start__",
      Object.entries(spec.schema.inputs(input)) as Update[],
    );
    if (command?.update !== undefined && command.update !== null) {
      work.record(
        live.state,
        "__start__",
        Object.entries(load(command.update) as object) as Update[],
      );
    }
    work.nodes.length = 0;
    // Input is not streamed back as output, but the values with it are, as the run's first state.
    for (const [operation, message] of work.change.messages) {
      if (operation === "add" || operation === "replace")
        live.seen.add(message.id!);
    }
    this.#publish(invocation, "values", live.state.output());
    const pending = live.state.pending;
    const resume = command === undefined ? undefined : load(command.resume);
    if (pending !== null && resume !== undefined && resume !== null) {
      await this.#resume(invocation, spec, live, work, pending, resume);
      return;
    }
    await this.#drive(
      invocation,
      spec,
      live,
      work,
      "agent",
      0,
      [],
      pending !== null,
    );
  }

  /** Continue a stopped thread with the input it was waiting for. */
  async #resume(
    invocation: TaskInvocation<RunInput>,
    spec: AgentSpec,
    live: Live,
    work: Work,
    pending: Record<string, any>,
    resume: unknown,
  ): Promise<void> {
    if (pending.kind === "interrupt") {
      const answer = resumeValue(resume, pending.interrupts[0].id);
      const answers = [...(pending.answers as unknown[]).map(load), answer];
      await this.#drive(
        invocation,
        spec,
        live,
        work,
        pending.sequence,
        pending.at,
        answers,
        true,
      );
      return;
    }
    const answers: Record<string, Partial<ToolInput>> = {};
    for (const [call, waiting] of Object.entries(
      pending.waiting as Record<string, any>,
    )) {
      answers[call] = waiting.subagent
        ? { resume: dump(resume) }
        : {
            answers: [
              ...waiting.answers,
              dump(resumeValue(resume, waiting.interrupts[0])),
            ],
          };
    }
    live.state.pending = null;
    try {
      await invocation.step((step) => {
        live.state.persist(
          step.tx,
          invocation.conversation,
          work.change,
          invocation.id,
        );
        step.tx.retireDoc(PENDING, threadScope(invocation.conversation));
        this.#spawn(
          invocation,
          spec,
          live,
          step,
          pending.step,
          pending.done,
          answers,
        );
      });
    } catch (error) {
      this.#forget(invocation.id);
      throw error;
    }
  }

  async #turn(invocation: TaskInvocation<RunInput>): Promise<void> {
    const spec = await this.agent(invocation.input.agent);
    const at = (invocation.checkpoint.at as number | undefined) ?? 0;
    await this.#drive(
      invocation,
      spec,
      this.#liveState(invocation, spec),
      new Work(),
      "turn",
      at,
    );
  }

  async #finish(invocation: TaskInvocation<RunInput>): Promise<void> {
    const spec = await this.agent(invocation.input.agent);
    await this.#drive(
      invocation,
      spec,
      this.#liveState(invocation, spec),
      new Work(),
      "finish",
    );
  }

  // Tool rounds.

  async #collect(invocation: TaskInvocation<RunInput>): Promise<void> {
    const spec = await this.agent(invocation.input.agent);
    const checkpoint = invocation.checkpoint as unknown as {
      step: number;
      calls: string[];
      tasks: number[];
      done: Record<string, unknown>;
    };
    const done: Record<string, unknown> = { ...checkpoint.done };
    const waiting: Record<string, unknown> = {};
    const interrupts: StoredInterrupt[] = [];
    for (const id of checkpoint.tasks) {
      const record = invocation.session.task(id);
      const outcome =
        record?.state.status === "terminal" ? record.state.outcome : undefined;
      if (outcome?.status !== "completed") {
        throw new Error(
          outcome !== undefined && "error" in outcome
            ? outcome.error.message
            : `tool task ${id} ended ${outcome?.status ?? "missing"}`,
        );
      }
      const result = outcome.result;
      if ("interrupts" in result) {
        interrupts.push(...result.interrupts);
        waiting[result.call] = {
          subagent: result.subagent ?? false,
          answers: result.answers ?? [],
          interrupts: result.interrupts.map((item: StoredInterrupt) => item.id),
        };
      } else {
        done[result.call] = result.updates;
      }
    }
    const live = this.#liveState(invocation, spec);
    if (interrupts.length > 0) {
      await this.#stop(invocation, live, new Work(), {
        kind: "tools",
        node: TOOLS,
        step: checkpoint.step,
        done,
        waiting,
        interrupts,
      });
      return;
    }
    const work = new Work();
    work.steps = 1;
    const updates = checkpoint.calls.flatMap((call) =>
      ((done[call] ?? []) as [string, unknown][]).map(
        ([name, value]) => [name, load(value)] as Update,
      ),
    );
    work.record(live.state, TOOLS, updates);
    const next = this.#afterTools(spec, live.state);
    if (next === "turn") {
      await this.#drive(invocation, spec, live, work, "turn");
    } else {
      await this.#commit(invocation, spec, live, work, next, 0);
    }
  }

  async #toolCall(invocation: TaskInvocation<ToolInput>): Promise<void> {
    const spec = await this.agent(invocation.input.agent);
    const { call } = invocation.input;
    const run = invocation.task.owner!;
    const live = this.#liveState(invocation, spec);
    const answers = (invocation.input.answers ?? []).map(load);
    const tool: ToolScope = {
      runtime: this,
      task: invocation.id,
      conversation: invocation.conversation,
      callId: call.id!,
      thread: invocation.input.thread,
      ns: [...invocation.input.ns, `${TOOLS}:${call.id}`],
      resume: load(invocation.input.resume),
      context: this.#context(run),
      agent: invocation.input.agent,
    };
    const scope = this.#scope(
      invocation,
      spec,
      live,
      TOOLS,
      call.id!,
      invocation.input.step,
      answers,
      { [TOOL_KEY]: tool },
    );
    let output: ToolMessage | Command;
    try {
      output = await scope.run(this.#config(run, spec), (config) =>
        spec.callTool(call, live.state.values(), scope.runtime(), config),
      );
    } catch (error) {
      if (AwaitingInputError.isInstance(error)) {
        await invocation.step((step) =>
          step.finish({
            call: call.id,
            interrupts: error.interrupts,
            subagent: true,
          }),
        );
        return;
      }
      if (isGraphInterrupt(error)) {
        const asked = error.interrupts.map(stored);
        await invocation.step((step) =>
          step.finish({
            call: call.id,
            interrupts: asked,
            answers: invocation.input.answers ?? [],
          }),
        );
        return;
      }
      throw error;
    }
    const updates: Update[] = [
      ...scope.sent,
      ...(isCommand(output)
        ? commandUpdates(output)
        : [["messages", [output]] as Update]),
    ];
    await invocation.step((step) =>
      step.finish({
        call: call.id,
        updates: updates.map(([name, value]) => [name, dump(value)]),
      }),
    );
    this.#publish(invocation, "updates", { [TOOLS]: merged(updates) });
  }

  // Aborts and faults.

  async #abortRun(invocation: TaskInvocation<RunInput>): Promise<void> {
    this.#forget(invocation.id);
    await invocation.step((step) => {
      step.tx.putSubmission(
        this.#settled(invocation, { status: "unanswered", reason: "aborted" }),
      );
      step.aborted();
    });
    this.#ended(invocation);
  }

  #runFaulted(
    tx: Tx,
    run: number,
    conversation: number,
    input: RunInput,
    message: string,
  ): void {
    tx.putSubmission({
      id: input.submission,
      conversationId: conversation,
      type: "input",
      status: "unanswered",
      reason: "faulted",
      detail: message,
    });
    this.#forget(run);
    this.#contexts.delete(run);
    this.#busy.delete(conversation);
    this.bus.publish({
      thread: input.thread,
      ns: input.ns,
      mode: END_EVENT,
      data: null,
      run,
    });
  }
}
