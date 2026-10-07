/**
 * JSON encoding of messages and state values for durable records.
 *
 * Messages use LangChain's stored-message form. Other values are plain JSON,
 * with tagged objects for messages nested inside values and for bytes, so
 * they come back as the same types.
 */
import {
  type BaseMessage,
  type StoredMessage,
  isBaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";

const MESSAGE = "__lc_message__";
const BYTES = "__bytes__";

/** A message as LangChain's `{type, data}` stored form. */
export function dumpMessage(message: BaseMessage): StoredMessage {
  return mapChatMessagesToStoredMessages([message])[0];
}

export function loadMessage(stored: StoredMessage): BaseMessage {
  return mapStoredMessagesToChatMessages([stored])[0];
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function unbase64(text: string): Uint8Array {
  return Uint8Array.from(atob(text), (char) => char.charCodeAt(0));
}

/** A built-in object's type, read the same in every realm. */
function tag(value: object): string {
  return Object.prototype.toString.call(value).slice(8, -1);
}

/** A state value as JSON. */
export function dump(value: unknown): unknown {
  if (value === null || value === undefined) return null;
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return value;
  if (isBaseMessage(value)) return { [MESSAGE]: dumpMessage(value) };
  if (tag(value) === "Uint8Array")
    return { [BYTES]: base64(value as Uint8Array) };
  if (Array.isArray(value)) return value.map(dump);
  if (tag(value) === "Date") return (value as Date).toISOString();
  if (typeof value === "object") {
    const encoded: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      if (item !== undefined) encoded[key] = dump(item);
    }
    return encoded;
  }
  throw new TypeError(`cannot store a value of type ${typeof value}`);
}

/** A value encoded by {@link dump}. */
export function load(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(load);
  if (value === null || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  if (MESSAGE in record) return loadMessage(record[MESSAGE] as StoredMessage);
  if (BYTES in record) return unbase64(record[BYTES] as string);
  const decoded: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(record)) decoded[key] = load(item);
  return decoded;
}
