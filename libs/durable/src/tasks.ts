/**
 * Durable tasks, scheduled by the Rust kernel and implemented in TypeScript.
 *
 * A task's checkpoint names its next phase. The kernel's scheduler reserves a
 * pending task, calls the matching phase handler, and keeps going while each
 * phase commits progress through {@link TaskInvocation.step}. After a crash
 * the task is pending again at its last checkpoint and that phase simply runs
 * again. Aborts flow down the ownership tree: the scheduler aborts a marked
 * run invocation's `signal`, waits for it to settle, and later runs the abort
 * handler, bottom-up.
 */
import type {
  Invocation,
  Session,
  Step,
  Task,
  TaskHandler,
  Tx,
} from "durable-wasm";

/** A checkpoint: which phase runs next, and whatever that phase needs. */
export interface Checkpoint {
  phase: string;
  [field: string]: unknown;
}

/** One reserved run or abort of a task, and the only way its code commits. */
export class TaskInvocation<Input = any> {
  #task: Task<Input> | undefined;

  constructor(readonly core: Invocation) {}

  /** The task record as of this invocation's last commit. */
  get task(): Task<Input> {
    this.#task ??= this.core.task;
    return this.#task;
  }

  get mode(): "run" | "abort" {
    return this.core.mode;
  }

  get session(): Session {
    return this.core.session;
  }

  /** Aborts when the scheduler wants this invocation to stop. */
  get signal(): AbortSignal {
    return this.core.signal;
  }

  get id(): number {
    return this.task.id;
  }

  get conversation(): number {
    return this.task.conversationId;
  }

  get input(): Input {
    return this.task.input;
  }

  /** Where the task resumes. */
  get checkpoint(): Checkpoint {
    const { state } = this.task;
    if (!("checkpoint" in state)) {
      throw new Error(
        `task ${this.id} is ${state.status} and has no checkpoint`,
      );
    }
    return state.checkpoint;
  }

  /**
   * Commit `write`'s changes and the step's new task state atomically.
   *
   * Waits only for the mutation line; `write` itself is synchronous, since
   * every read and write inside a transaction is. When it throws, nothing is
   * written.
   *
   * @throws InvocationEndedError at commit, when the task is no longer
   *   running under this invocation or a run invocation's task was marked
   *   for abort.
   */
  async step<T>(write: (step: Step) => T): Promise<T> {
    const step = await this.core.step();
    try {
      const value = synchronous(write(step));
      step.commit();
      return value;
    } catch (error) {
      step.rollback();
      throw error;
    } finally {
      this.#task = undefined;
      step.free();
    }
  }
}

/** A transaction's body must not await: everything inside one is synchronous. */
export function synchronous<T>(value: T): T {
  if (
    value !== null &&
    typeof value === "object" &&
    typeof (value as { then?: unknown }).then === "function"
  ) {
    throw new TypeError(
      "a transaction's body returned a promise: reads and writes inside a transaction are synchronous, so it must not await",
    );
  }
  return value;
}

type Phase<Input> = (invocation: TaskInvocation<Input>) => Promise<void>;

/**
 * A kind of durable task: one handler per checkpoint phase, plus an abort handler.
 *
 * `checkpoint.phase` selects the handler. A handler must commit progress
 * through {@link TaskInvocation.step} before returning, or the task faults.
 * `onFault` runs inside the commit that faults a task of this kind, so it can
 * settle whatever the task was responsible for.
 */
export interface TaskDefinition<Input = any> {
  kind: string;
  phases: Record<string, Phase<Input>>;
  abort: Phase<Input>;
  onFault?: (tx: Tx, task: Task<Input>, message: string) => void;
}

/** The object the kernel's scheduler calls for tasks of `definition.kind`. */
export function handler<Input>(definition: TaskDefinition<Input>): TaskHandler {
  const run = async (
    core: Invocation,
    phase: (invocation: TaskInvocation<Input>) => Phase<Input>,
  ) => {
    try {
      const invocation = new TaskInvocation<Input>(core);
      await phase(invocation)(invocation);
    } finally {
      core.free();
    }
  };
  const handler: TaskHandler = {
    run: (core) =>
      run(core, (invocation) => {
        const phase = invocation.checkpoint.phase;
        const found = Object.hasOwn(definition.phases, phase)
          ? definition.phases[phase]
          : undefined;
        if (found === undefined) {
          throw new Error(
            `task kind ${definition.kind} has no phase ${JSON.stringify(phase)}`,
          );
        }
        return found;
      }),
    abort: (core) => run(core, () => definition.abort),
  };
  const { onFault } = definition;
  if (onFault !== undefined) {
    handler.fault = (tx, task, message) => onFault(tx, task, message);
  }
  return handler;
}
