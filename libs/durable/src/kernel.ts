/**
 * The durable session and the task scheduler that runs on it.
 */
import {
  Scheduler,
  Session,
  type OpenOptions,
  type Tx,
} from "../wasm/durable.js";
import { assertLoaded } from "./load.js";
import { handler, synchronous, type TaskDefinition } from "./tasks.js";

/**
 * One open durable session plus the scheduler that runs its tasks.
 *
 * Everything durable goes through {@link Kernel.commit}. Work the process
 * starts is a task: the scheduler reserves it, runs its phases, and resumes
 * it from its last checkpoint after a restart.
 */
export class Kernel {
  readonly scheduler: Scheduler;
  /** What runs on this kernel, keyed by who put it there, such as the agent runtime. */
  readonly services = new Map<symbol, unknown>();

  private constructor(readonly session: Session) {
    this.scheduler = new Scheduler(session);
  }

  /** Open a session over a SQLite file kept by `files`, or in memory without a path. */
  static async open(options?: OpenOptions): Promise<Kernel> {
    assertLoaded();
    return new Kernel(await Session.open(options));
  }

  /**
   * Run `change` in one transaction and commit it atomically; returns what it returned.
   *
   * Waits only for the mutation line. `change` itself is synchronous: every
   * read and write inside a transaction is. When it throws, nothing is
   * written.
   */
  async commit<T>(change: (tx: Tx) => T): Promise<T> {
    const tx = await this.session.transaction();
    try {
      const value = synchronous(change(tx));
      tx.commit();
      return value;
    } catch (error) {
      tx.rollback();
      throw error;
    } finally {
      tx.free();
    }
  }

  /** Run tasks of `definition.kind` here; a later registration replaces an earlier one. */
  register(definition: TaskDefinition): void {
    this.scheduler.register(definition.kind, handler(definition));
  }

  /**
   * Stop every invocation without writing outcomes, then close storage.
   * Interrupted work stays pending and resumes when the session is reopened.
   */
  async close(): Promise<void> {
    await this.scheduler.stop();
    await this.session.close();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}
