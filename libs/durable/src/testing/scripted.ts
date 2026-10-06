/**
 * A scripted chat model for tests.
 */
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";

/**
 * Answers with the next scripted message, repeating the last one when the
 * script runs out. Streams a text answer word by word, so token streaming
 * can be observed.
 */
export class ScriptedModel extends BaseChatModel {
  readonly calls: BaseMessage[][] = [];

  /**
   * With `byHistory`, the answer is chosen by how many AI messages the
   * conversation already has, so a fresh instance picks up where an earlier
   * one left off, as after a restart.
   */
  constructor(
    readonly script: AIMessage[],
    readonly options: { byHistory?: boolean } = {},
  ) {
    super({});
  }

  _llmType(): string {
    return "scripted";
  }

  #next(messages: BaseMessage[]): AIMessage {
    this.calls.push([...messages]);
    const turn = this.options.byHistory
      ? messages.filter((message) => message.type === "ai").length + 1
      : this.calls.length;
    const message = this.script[Math.min(turn, this.script.length) - 1];
    return new AIMessage({
      content: message.content,
      tool_calls: message.tool_calls,
      id: crypto.randomUUID(),
    });
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    const message = this.#next(messages);
    return { generations: [{ text: message.text, message }] };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const message = this.#next(messages);
    const words =
      typeof message.content === "string" && message.content !== ""
        ? message.content.split(" ")
        : [""];
    for (const [index, word] of words.entries()) {
      const last = index === words.length - 1;
      const text = last ? word : `${word} `;
      const chunk = new ChatGenerationChunk({
        text,
        message: new AIMessageChunk({
          content: text,
          id: message.id,
          tool_call_chunks: last
            ? (message.tool_calls ?? []).map((call, position) => ({
                name: call.name,
                args: JSON.stringify(call.args),
                id: call.id,
                index: position,
                type: "tool_call_chunk" as const,
              }))
            : [],
        }),
      });
      await runManager?.handleLLMNewToken(
        text,
        { prompt: 0, completion: 0 },
        undefined,
        undefined,
        undefined,
        { chunk },
      );
      yield chunk;
    }
  }

  override bindTools(): this {
    return this;
  }
}

/** A tool call. */
export function call(
  name: string,
  id: string,
  args: Record<string, unknown>,
): ToolCall {
  return { name, args, id, type: "tool_call" };
}

/** An AI message with tool calls, or a final answer. */
export function ai(...calls: ToolCall[]): AIMessage;
export function ai(text: string): AIMessage;
export function ai(...parts: (ToolCall | string)[]): AIMessage {
  const [first] = parts;
  if (typeof first === "string") return new AIMessage({ content: first });
  return new AIMessage({ content: "", tool_calls: parts as ToolCall[] });
}
