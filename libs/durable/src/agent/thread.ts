/**
 * An agent thread: a conversation whose transcript holds the messages.
 *
 * - Each message is an immutable `lc.message` entry, appended once.
 * - Replacing or removing a message appends an `lc.edit` entry naming the
 *   entry it edits, as pi-durable's context edits do; the original stays.
 * - Removing every message appends an `lc.reset` entry that starts a new context.
 * - Every other stored field is one rewindable conversation document.
 *
 * A step therefore writes only what changed, and a thread grows linearly with
 * its messages.
 */
import {
  type BaseMessage,
  type BaseMessageLike,
  type StoredMessage,
  RemoveMessage,
  coerceMessageLikeToMessage,
} from "@langchain/core/messages";
import { REMOVE_ALL_MESSAGES } from "@langchain/langgraph";
import type { Scope, Session, StoredEntry, Tx } from "durable-wasm";
import { dump, dumpMessage, load, loadMessage } from "./serde.js";
import { MESSAGES, type Schema } from "./schema.js";

export const THREAD_IDS = "lc.thread-id";
export const FIELD = "lc.field";
/** Conversation document: what a stopped run awaits, and where it continues. */
export const PENDING = "lc.pending";
const MESSAGE = "lc.message";
const EDIT = "lc.edit";
const RESET = "lc.reset";

const SESSION: Scope = { kind: "session" };

/** The document scope of a conversation. */
export function scope(conversation: number): Scope {
  return { kind: "conversation", conversationId: conversation };
}

/** The conversation of a thread ID, as the transaction sees it. */
export function find(tx: Tx, threadId: string): number | null {
  const found = tx.doc(THREAD_IDS, SESSION, threadId) as {
    conversationId: number;
  } | null;
  return found === null ? null : found.conversationId;
}

/** Create a thread's conversation, registering its thread ID when it has one. */
export function create(tx: Tx, threadId: string | null): number {
  const conversation = tx.createConversation();
  if (threadId !== null)
    tx.putDoc(
      THREAD_IDS,
      SESSION,
      { conversationId: conversation },
      { key: threadId },
    );
  return conversation;
}

export type Update = [name: string, value: unknown];

export type MessageOp =
  | ["add" | "replace", BaseMessage]
  | ["remove", string]
  | ["reset", null];

/** What one step changed, to store and to stream. */
export class Change {
  /** The step's updates as applied, messages coerced and with IDs. */
  readonly updates: Update[] = [];
  readonly messages: MessageOp[] = [];
  readonly fields = new Set<string>();

  get empty(): boolean {
    return this.messages.length === 0 && this.fields.size === 0;
  }

  extend(other: Change): void {
    this.updates.push(...other.updates);
    this.messages.push(...other.messages);
    for (const name of other.fields) this.fields.add(name);
  }
}

/** A messages update as messages with IDs, as LangGraph's messages reducer coerces them. */
function coerce(value: unknown): BaseMessage[] {
  const items = (Array.isArray(value) ? value : [value]) as BaseMessageLike[];
  return items.map((item) => {
    const message = coerceMessageLikeToMessage(item);
    if (message.id === undefined || message.id === null || message.id === "") {
      message.id = crypto.randomUUID();
      message.lc_kwargs.id = message.id;
    }
    return message;
  });
}

function messageId(message: BaseMessage): string {
  if (!message.id)
    throw new Error(
      `a thread message has no ID: ${JSON.stringify(dumpMessage(message))}`,
    );
  return message.id;
}

/** A thread's messages and fields, as this process last committed or loaded them. */
export class ThreadState {
  #messages: (BaseMessage | undefined)[] = [];
  #position = new Map<string, number>();
  /** The entry that stores each message, for edits. */
  #entry = new Map<string, number>();
  #view: BaseMessage[] | undefined = [];
  fields: Record<string, unknown> = {};
  /** What a stopped run of this thread awaits, if one did. */
  pending: Record<string, any> | null = null;

  constructor(readonly schema: Schema) {}

  /** The current messages, in order. Do not mutate the array. */
  get messages(): BaseMessage[] {
    this.#view ??= this.#messages.filter(
      (message): message is BaseMessage => message !== undefined,
    );
    return this.#view;
  }

  /** The whole state, as hooks and tools see it. */
  values(): Record<string, unknown> {
    return { [MESSAGES]: this.messages, ...this.fields };
  }

  /** The state an invocation returns. */
  output(): Record<string, unknown> {
    return this.schema.outputs(this.values());
  }

  #add(message: BaseMessage): "add" | "replace" {
    this.#view = undefined;
    const id = messageId(message);
    const position = this.#position.get(id);
    if (position === undefined) {
      this.#position.set(id, this.#messages.length);
      this.#messages.push(message);
      return "add";
    }
    this.#messages[position] = message;
    return "replace";
  }

  #remove(id: string): void {
    const position = this.#position.get(id);
    if (position === undefined) {
      throw new Error(
        `Attempting to delete a message with an ID that doesn't exist ('${id}')`,
      );
    }
    this.#view = undefined;
    this.#position.delete(id);
    this.#messages[position] = undefined;
  }

  #reset(): void {
    this.#messages = [];
    this.#position = new Map();
    this.#entry = new Map();
    this.#view = [];
  }

  #applyMessages(messages: BaseMessage[], change: Change): void {
    // As LangGraph's reducer: the last remove-all discards everything before it.
    const resetAt = messages.findLastIndex(
      (message) =>
        RemoveMessage.isInstance(message) && message.id === REMOVE_ALL_MESSAGES,
    );
    if (resetAt >= 0) {
      this.#reset();
      change.messages.push(["reset", null]);
    }
    for (const message of messages.slice(resetAt + 1)) {
      if (RemoveMessage.isInstance(message)) {
        this.#remove(messageId(message));
        change.messages.push(["remove", messageId(message)]);
      } else {
        const position = this.#position.get(messageId(message));
        // Writing back the very message a step read changes nothing.
        if (position !== undefined && this.#messages[position] === message)
          continue;
        change.messages.push([this.#add(message), message]);
      }
    }
  }

  /** Apply one step's updates in order. */
  apply(updates: Update[]): Change {
    const change = new Change();
    const writes = new Map<string, unknown[]>();
    for (const [name, value] of updates) {
      if (name === MESSAGES) {
        const messages = coerce(value);
        this.#applyMessages(messages, change);
        change.updates.push([name, messages]);
      } else {
        if (!writes.has(name)) writes.set(name, []);
        writes.get(name)!.push(value);
        change.updates.push([name, value]);
      }
    }
    for (const [name, values] of writes) {
      this.fields[name] = this.schema.merge(name, this.fields[name], values);
      change.fields.add(name);
    }
    return change;
  }

  /** Store a change applied to this state. */
  persist(tx: Tx, conversation: number, change: Change, byTask?: number): void {
    for (const [operation, payload] of change.messages) {
      if (operation === "reset") {
        tx.appendEntry(conversation, RESET, null, { head: "self", byTask });
      } else if (operation === "remove") {
        const target = this.#entry.get(payload);
        this.#entry.delete(payload);
        if (target !== undefined) {
          tx.appendEntry(
            conversation,
            EDIT,
            { edits: [{ target, action: "omit" }] },
            { byTask },
          );
        }
      } else if (
        operation === "replace" &&
        this.#entry.has(messageId(payload))
      ) {
        const edit = {
          target: this.#entry.get(messageId(payload)),
          action: "replace",
          messages: [dumpMessage(payload)],
        };
        tx.appendEntry(conversation, EDIT, { edits: [edit] }, { byTask });
      } else {
        const entry = tx.appendEntry(
          conversation,
          MESSAGE,
          { data: dumpMessage(payload) },
          { byTask },
        );
        this.#entry.set(messageId(payload), entry);
      }
    }
    for (const name of [...change.fields].sort()) {
      if (!this.schema.stored(name)) continue;
      tx.putDoc(
        FIELD,
        scope(conversation),
        { value: dump(this.fields[name]) },
        {
          key: name,
          history: "rewindable",
          fork: "asOf",
        },
      );
    }
  }

  /** Rebuild the messages from a conversation's active context entries. */
  fold(entries: StoredEntry[]): void {
    this.#reset();
    const byEntry = new Map<number, string>();
    for (const { entry } of entries) {
      if (entry.kind === MESSAGE) {
        const message = loadMessage(entry.data as StoredMessage);
        this.#add(message);
        this.#entry.set(messageId(message), entry.id);
        byEntry.set(entry.id, messageId(message));
      } else if (entry.kind === EDIT) {
        for (const edit of entry.edits as {
          target: number;
          action: string;
          messages?: StoredMessage[];
        }[]) {
          const id = byEntry.get(edit.target);
          if (id === undefined) continue;
          if (edit.action === "omit") this.#remove(id);
          else this.#add(loadMessage(edit.messages![0]));
        }
      }
    }
  }
}

/** A thread's state now, or as of commit `at`. */
export function loadThread(
  session: Session,
  conversation: number,
  schema: Schema,
  at?: number,
): ThreadState {
  const state = new ThreadState(schema);
  state.fold(session.context(conversation, at));
  if (at === undefined) {
    state.pending = session.doc(PENDING, scope(conversation))?.value ?? null;
  }
  for (const record of session.docs(scope(conversation), FIELD)) {
    const name = record.key!;
    const stored = session.doc(FIELD, scope(conversation), name, at);
    if (stored !== null)
      state.fields[name] = load((stored.value as { value: unknown }).value);
  }
  return state;
}
