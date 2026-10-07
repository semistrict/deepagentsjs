/**
 * Agent state fields: how updates to each one merge, whether it is stored,
 * and whether invocations see it.
 *
 * Middleware declare their state as a LangGraph `StateSchema` or a Zod
 * object, as `createAgent` reads them. A `ReducedValue`, or a Zod field with
 * a reducer registered, merges updates with its reducer; any other field
 * takes the last value written. `UntrackedValue` fields live only in memory.
 * Fields named with a leading underscore are private: kept in the thread,
 * but neither accepted as input nor returned. The first schema to declare a
 * field wins. `messages` is not merged here: the thread stores it as entries.
 */
import {
  getInteropZodObjectShape,
  isInteropZodObject,
  isZodSchemaV4,
} from "@langchain/core/utils/types";
import {
  ReducedValue,
  StateSchema,
  UntrackedValue,
} from "@langchain/langgraph";
import { schemaMetaRegistry } from "@langchain/langgraph/zod";

export const MESSAGES = "messages";

/** A write that replaces a reduced field's value instead of reducing into it. */
const OVERWRITE = "__overwrite__";

export interface Field {
  /** Combines the current value with one write; absent keeps the last write. */
  reducer?: (current: any, next: any) => any;
  /** The value before any write, from the field's schema default. */
  initial?: () => unknown;
  /** Whether the value is persisted; untracked fields live only in memory. */
  stored: boolean;
  /** Whether invocations accept and return it; private fields are neither. */
  visible: boolean;
}

interface StandardSchema {
  "~standard": {
    validate(
      value: unknown,
    ): { value?: unknown; issues?: unknown } | Promise<unknown>;
  };
}

/** A schema's default, read the way any Standard Schema validates `undefined`. */
function defaultOf(schema: unknown): (() => unknown) | undefined {
  if (
    schema === null ||
    typeof schema !== "object" ||
    !("~standard" in schema)
  ) {
    return undefined;
  }
  const standard = (schema as StandardSchema)["~standard"];
  let probe: unknown;
  try {
    probe = standard.validate(undefined);
  } catch {
    return undefined;
  }
  // A schema that validates asynchronously has no default to read synchronously.
  if (typeof (probe as { then?: unknown })?.then === "function")
    return undefined;
  const result = probe as { value?: unknown; issues?: unknown };
  if (result.issues !== undefined || result.value === undefined)
    return undefined;
  return () => {
    const fresh = standard.validate(undefined) as { value?: unknown };
    return fresh.value;
  };
}

function stateSchemaField(field: unknown, visible: boolean): Field {
  if (ReducedValue.isInstance(field)) {
    return {
      reducer: field.reducer as Field["reducer"],
      initial: defaultOf(field.valueSchema),
      stored: true,
      visible,
    };
  }
  if (UntrackedValue.isInstance(field)) {
    return { stored: false, visible };
  }
  return { initial: defaultOf(field), stored: true, visible };
}

function zodField(field: unknown, visible: boolean): Field {
  const meta = isZodSchemaV4(field as never)
    ? schemaMetaRegistry.get(field as never)
    : undefined;
  const reducer = meta?.reducer?.fn as Field["reducer"] | undefined;
  const initial =
    (meta?.default as (() => unknown) | undefined) ?? defaultOf(field);
  return { reducer, initial, stored: true, visible };
}

/** The fields of an agent's state, merged from its schemas; the first to declare a field wins. */
export class Schema {
  readonly fields = new Map<string, Field>();

  constructor(schemas: Iterable<unknown>) {
    // `jumpTo` steers the loop within one step and is never kept.
    this.fields.set("jumpTo", { stored: false, visible: false });
    for (const schema of schemas) this.#add(schema);
  }

  #add(schema: unknown): void {
    if (StateSchema.isInstance(schema)) {
      for (const [name, field] of Object.entries(schema.fields)) {
        if (name === MESSAGES || this.fields.has(name)) continue;
        this.fields.set(name, stateSchemaField(field, !name.startsWith("_")));
      }
    } else if (isInteropZodObject(schema)) {
      for (const [name, field] of Object.entries(
        getInteropZodObjectShape(schema),
      )) {
        if (name === MESSAGES || this.fields.has(name)) continue;
        this.fields.set(name, zodField(field, !name.startsWith("_")));
      }
    }
  }

  /** A field's value after one step's writes, in order. */
  merge(name: string, current: unknown, writes: unknown[]): unknown {
    const reducer = this.fields.get(name)?.reducer;
    let value = current;
    for (const write of writes) {
      if (write !== null && typeof write === "object" && OVERWRITE in write) {
        value = (write as Record<string, unknown>)[OVERWRITE];
      } else if (reducer === undefined) {
        value = write;
      } else {
        value = value === undefined ? write : reducer(value, write);
      }
    }
    return value;
  }

  stored(name: string): boolean {
    return this.fields.get(name)?.stored ?? true;
  }

  /** The fields that start with a value, and what it is. */
  defaults(): Record<string, unknown> {
    const values: Record<string, unknown> = {};
    for (const [name, field] of this.fields) {
      if (field.initial !== undefined) values[name] = field.initial();
    }
    return values;
  }

  /** The part of an invocation's input the schema accepts. */
  inputs(values: Record<string, unknown>): Record<string, unknown> {
    return Object.fromEntries(
      Object.entries(values).filter(
        ([name]) =>
          name === MESSAGES || (this.fields.get(name)?.visible ?? false),
      ),
    );
  }

  /** The part of the state an invocation returns. */
  outputs(values: Record<string, unknown>): Record<string, unknown> {
    return Object.fromEntries(
      Object.entries(values).filter(
        ([name]) =>
          name === MESSAGES || (this.fields.get(name)?.visible ?? true),
      ),
    );
  }
}
