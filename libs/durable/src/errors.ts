/**
 * Kernel errors are plain `Error`s told apart by `name`.
 */

export type DurableErrorName =
  /** A write broke a record contract; nothing was committed. */
  | "InvalidWriteError"
  /** A table was read after a table write in the same transaction. */
  | "ReadAfterWriteError"
  /** The transaction was already committed or rolled back. */
  | "TransactionFinishedError"
  /** The session is closed. */
  | "SessionClosedError"
  /** Another session holds the file open. */
  | "SessionLockedError"
  /** Stored state contradicts itself; the session must not continue. */
  | "CorruptStorageError"
  /** The invocation may no longer commit: its task finished, moved, or was marked for abort. */
  | "InvocationEndedError"
  /** A frame subscriber fell behind and must resynchronize. */
  | "FramesLaggedError";

/** Whether `error` is the kernel error named `name`. */
export function isDurableError(
  error: unknown,
  name: DurableErrorName,
): error is Error {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { name?: unknown }).name === name
  );
}
