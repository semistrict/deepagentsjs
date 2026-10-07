import {
  AIMessage,
  HumanMessage,
  RemoveMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  ReducedValue,
  REMOVE_ALL_MESSAGES,
  StateSchema,
} from "@langchain/langgraph";
import { describe, expect, test } from "vitest";
import { z } from "zod";
import { Kernel } from "../node.js";
import { Schema } from "./schema.js";
import { create, loadThread, ThreadState, type Update } from "./thread.js";

const Counted = new StateSchema({
  counts: new ReducedValue(
    z.record(z.string(), z.number()).default(() => ({})),
    {
      reducer: (
        current: Record<string, number>,
        update: Record<string, number>,
      ) =>
        Object.fromEntries(
          [...new Set([...Object.keys(current), ...Object.keys(update)])].map(
            (key) => [key, (current[key] ?? 0) + (update[key] ?? 0)],
          ),
        ),
    },
  ),
  _private: z.string().optional(),
});

async function roundtrip(steps: Update[][]) {
  const schema = new Schema([Counted]);
  const state = new ThreadState(schema);
  const kernel = await Kernel.open();
  try {
    const conversation = await kernel.commit((tx) => create(tx, "t"));
    for (const step of steps) {
      const change = state.apply(step);
      await kernel.commit((tx) => state.persist(tx, conversation, change));
    }
    const loaded = loadThread(kernel.session, conversation, schema);
    const kinds = kernel.session
      .entries(conversation)
      .map(({ entry }) => entry.kind);
    return { state, loaded, kinds };
  } finally {
    await kernel.close();
  }
}

const contents = (messages: { id?: string; content: unknown }[]) =>
  messages.map((message) => [message.id, message.content]);

describe("threads", () => {
  test("replaced and removed messages are edits that reload identically", async () => {
    const { state, loaded, kinds } = await roundtrip([
      [
        [
          "messages",
          [
            new HumanMessage({ content: "hi", id: "m1" }),
            new AIMessage({ content: "hello", id: "m2" }),
          ],
        ],
        ["counts", { a: 1 }],
      ],
      [
        ["messages", [new AIMessage({ content: "hello, edited", id: "m2" })]],
        ["counts", { a: 2, b: 1 }],
      ],
      [
        [
          "messages",
          [
            new HumanMessage({ content: "more", id: "m3" }),
            new RemoveMessage({ id: "m1" }),
          ],
        ],
      ],
    ]);
    expect(contents(loaded.messages)).toEqual(contents(state.messages));
    expect(contents(loaded.messages)).toEqual([
      ["m2", "hello, edited"],
      ["m3", "more"],
    ]);
    expect(loaded.fields).toEqual({ counts: { a: 3, b: 1 } });
    expect(state.fields).toEqual({ counts: { a: 3, b: 1 } });
    expect(kinds).toEqual([
      "lc.message",
      "lc.message",
      "lc.edit",
      "lc.message",
      "lc.edit",
    ]);
  });

  test("removing every message starts a new context", async () => {
    const { loaded, kinds } = await roundtrip([
      [["messages", [new HumanMessage({ content: "old", id: "m1" })]]],
      [
        [
          "messages",
          [
            new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
            new HumanMessage({ content: "new", id: "m2" }),
          ],
        ],
      ],
    ]);
    expect(contents(loaded.messages)).toEqual([["m2", "new"]]);
    expect(kinds).toEqual(["lc.message", "lc.reset", "lc.message"]);
  });

  test("tool calls and tool results reload as the same message types", async () => {
    const call = {
      name: "search",
      args: { q: "x" },
      id: "c1",
      type: "tool_call" as const,
    };
    const { loaded } = await roundtrip([
      [
        [
          "messages",
          [
            new AIMessage({ content: "", tool_calls: [call], id: "a1" }),
            new ToolMessage({
              content: "found",
              tool_call_id: "c1",
              name: "search",
              id: "t1",
            }),
          ],
        ],
      ],
    ]);
    const [ai, tool] = loaded.messages;
    expect([
      AIMessage.isInstance(ai) && ai.tool_calls,
      ToolMessage.isInstance(tool) && tool.tool_call_id,
    ]).toEqual([[call], "c1"]);
  });

  test("message-likes are coerced and given IDs", () => {
    const state = new ThreadState(new Schema([]));
    state.apply([["messages", [{ role: "user", content: "hi" }]]]);
    expect([state.messages[0].type, typeof state.messages[0].id]).toEqual([
      "human",
      "string",
    ]);
  });

  test("removing a message that does not exist fails as LangGraph's reducer does", () => {
    const state = new ThreadState(new Schema([]));
    expect(() =>
      state.apply([["messages", [new RemoveMessage({ id: "nope" })]]]),
    ).toThrow(
      "Attempting to delete a message with an ID that doesn't exist ('nope')",
    );
  });

  test("private and untracked fields are kept from invocations", () => {
    const schema = new Schema([Counted]);
    expect(
      schema.outputs({
        messages: [],
        counts: {},
        _private: "x",
        jumpTo: "end",
        other: 1,
      }),
    ).toEqual({
      messages: [],
      counts: {},
      other: 1,
    });
    expect(schema.defaults()).toEqual({ counts: {} });
  });
});
