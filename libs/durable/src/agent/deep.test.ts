/**
 * The full Deep Agents middleware stack on the durable runtime, against `deepagents`' own `createDeepAgent`.
 */
import {
  type AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import * as deepagents from "deepagents";
import { todoListMiddleware } from "langchain";
import { afterEach, expect, test } from "vitest";
import { Kernel } from "../node.js";
import { ScriptedModel, ai, call } from "../testing/scripted.js";
import { createDeepAgent } from "./deep.js";

function shape(messages: BaseMessage[]): [string, unknown, string[]][] {
  return messages.map((message) => [
    message.type,
    message.content,
    ((message as AIMessage).tool_calls ?? []).map((c) => c.name),
  ]);
}

const kernels: Kernel[] = [];
afterEach(async () => {
  for (const opened of kernels.splice(0)) await opened.close();
});

async function both(
  script: () => AIMessage[],
  params: Record<string, unknown> = {},
) {
  const kernel = await Kernel.open();
  kernels.push(kernel);
  const durable = createDeepAgent({
    model: new ScriptedModel(script()),
    checkpointer: kernel,
    ...params,
  });
  const ours = await durable.invoke({
    messages: [new HumanMessage("take notes")],
  });
  const reference = deepagents.createDeepAgent({
    model: new ScriptedModel(script()),
    ...params,
  });
  const theirs: Record<string, any> = await reference.invoke({
    messages: [new HumanMessage("take notes")],
  });
  return { ours, theirs };
}

test("todos and state backend files match createDeepAgent", async () => {
  const script = () => [
    ai(
      call("write_todos", "t1", {
        todos: [{ content: "write notes", status: "in_progress" }],
      }),
    ),
    ai(
      call("write_file", "w1", {
        file_path: "/notes.md",
        content: "hello\nworld\n",
      }),
    ),
    ai(call("read_file", "r1", { file_path: "/notes.md" })),
    ai(
      call("edit_file", "e1", {
        file_path: "/notes.md",
        old_string: "world",
        new_string: "durable world",
      }),
    ),
    ai("Notes written."),
  ];
  const { ours, theirs } = await both(script, {
    middleware: [todoListMiddleware()],
  });
  expect(shape(ours.messages)).toEqual(shape(theirs.messages));
  expect(ours.todos).toEqual(theirs.todos);
  expect(ours.todos).toEqual([
    { content: "write notes", status: "in_progress" },
  ]);
  const contents = (
    files: Record<string, { content: unknown; mimeType: unknown }>,
  ) =>
    Object.fromEntries(
      Object.entries(files).map(([path, file]) => [
        path,
        [file.content, file.mimeType],
      ]),
    );
  expect(contents(ours.files)).toEqual(contents(theirs.files));
  expect(contents(ours.files)).toEqual({
    "/notes.md": ["hello\ndurable world\n", "text/markdown"],
  });
});

test("the general-purpose subagent matches createDeepAgent", async () => {
  const script = () => [
    ai(
      call("task", "s1", {
        description: "Write /sub.md saying hi",
        subagent_type: "general-purpose",
      }),
    ),
    ai(call("write_file", "w1", { file_path: "/sub.md", content: "hi\n" })),
    ai("Wrote it."),
    ai("The subagent wrote /sub.md."),
  ];
  const { ours, theirs } = await both(script);
  expect(shape(ours.messages)).toEqual(shape(theirs.messages));
  expect(Object.keys(ours.files)).toEqual(Object.keys(theirs.files));
  expect(Object.keys(ours.files)).toEqual(["/sub.md"]);
});
