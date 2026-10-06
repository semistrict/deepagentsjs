/**
 * The agent loop on the durable kernel, checked against `langchain`'s `createAgent`.
 */
import {
  type AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { Command, interrupt, MemorySaver } from "@langchain/langgraph";
import {
  createAgent as createLangChainAgent,
  createMiddleware,
  humanInTheLoopMiddleware,
} from "langchain";
import { afterEach, describe, expect, test } from "vitest";
import { z } from "zod";
import { Kernel, MemoryFiles } from "../node.js";
import { ScriptedModel, ai, call } from "../testing/scripted.js";
import { createAgent } from "./agent.js";

const getWeather = tool(
  async ({ city }: { city: string }) => `It is sunny in ${city}.`,
  {
    name: "get_weather",
    description: "Get the weather for a city.",
    schema: z.object({ city: z.string() }),
  },
);

/** Messages without IDs, for comparing runtimes. */
function shape(messages: BaseMessage[]): [string, unknown, string[]][] {
  return messages.map((message) => [
    message.type,
    message.content,
    ((message as AIMessage).tool_calls ?? []).map((c) => c.name),
  ]);
}

const weatherScript = () => [
  ai(
    call("get_weather", "c1", { city: "Paris" }),
    call("get_weather", "c2", { city: "Rome" }),
  ),
  ai("Sunny in both."),
];

const kernels: Kernel[] = [];
async function kernel(
  ...args: Parameters<typeof Kernel.open>
): Promise<Kernel> {
  const opened = await Kernel.open(...args);
  kernels.push(opened);
  return opened;
}

afterEach(async () => {
  for (const opened of kernels.splice(0)) await opened.close();
});

async function collect<T>(
  chunks: Promise<AsyncIterable<T>> | AsyncIterable<T>,
): Promise<T[]> {
  const items: T[] = [];
  for await (const chunk of await chunks) items.push(chunk);
  return items;
}

describe("the loop matches createAgent", () => {
  test("for a parallel tool round", async () => {
    const ours = await createAgent({
      model: new ScriptedModel(weatherScript()),
      tools: [getWeather],
      checkpointer: await kernel(),
    }).invoke({
      messages: [new HumanMessage("weather?")],
    });
    const theirs = await createLangChainAgent({
      model: new ScriptedModel(weatherScript()),
      tools: [getWeather],
    }).invoke({
      messages: [new HumanMessage("weather?")],
    });
    expect(shape(ours.messages)).toEqual(shape(theirs.messages));
    expect(ours.messages[2].content).toBe("It is sunny in Paris.");
  });

  test("in the updates it streams", async () => {
    const durable = createAgent({
      model: new ScriptedModel(weatherScript()),
      tools: [getWeather],
      checkpointer: await kernel(),
    });
    const ours = await collect(
      durable.stream(
        { messages: [new HumanMessage("weather?")] },
        { streamMode: "updates" },
      ),
    );
    const reference = createLangChainAgent({
      model: new ScriptedModel(weatherScript()),
      tools: [getWeather],
    });
    const theirs = await collect(
      reference.stream(
        { messages: [new HumanMessage("weather?")] },
        { streamMode: "updates" },
      ),
    );
    expect(ours.map((chunk) => Object.keys(chunk))).toEqual(
      theirs.map((chunk) => Object.keys(chunk)),
    );
    expect(ours.map((chunk) => Object.keys(chunk))).toEqual([
      ["model_request"],
      ["tools"],
      ["tools"],
      ["model_request"],
    ]);
  });

  test("in the tokens it streams", async () => {
    const script = [ai("The answer is 42.")];
    const durable = createAgent({
      model: new ScriptedModel(script),
      checkpointer: await kernel(),
      name: "oracle",
    });
    const ours = await collect(
      durable.stream(
        { messages: [new HumanMessage("?")] },
        { streamMode: "messages" },
      ),
    );
    const reference = createLangChainAgent({
      model: new ScriptedModel(script),
      tools: [],
      name: "oracle",
    });
    const theirs = await collect(
      reference.stream(
        { messages: [new HumanMessage("?")] },
        { streamMode: "messages" },
      ),
    );
    const tokens = (chunks: [BaseMessage, Record<string, unknown>][]) =>
      chunks.map(([message]) => [message.type, message.text]);
    expect(tokens(ours)).toEqual(tokens(theirs));
    expect(tokens(ours)).toEqual([
      ["ai", "The "],
      ["ai", "answer "],
      ["ai", "is "],
      ["ai", "42."],
    ]);
    const keys = (chunks: [BaseMessage, Record<string, unknown>][]) =>
      chunks.map(([, metadata]) => [
        metadata.langgraph_node,
        metadata.lc_agent_name,
      ]);
    expect(keys(ours)).toEqual(keys(theirs));
  });

  test.each([
    ["updates", false],
    [["updates"], false],
    [["updates", "values"], false],
    ["updates", true],
    [["updates", "values"], true],
  ] as const)(
    "in its chunk shapes for stream mode %j with subgraphs %s",
    async (streamMode, subgraphs) => {
      const shapes = (chunks: unknown[]) =>
        chunks.map((chunk) =>
          Array.isArray(chunk)
            ? chunk.map((part) => (Array.isArray(part) ? "array" : typeof part))
            : typeof chunk,
        );
      const durable = createAgent({
        model: new ScriptedModel([ai("hi")]),
        checkpointer: await kernel(),
      });
      const ours = await collect(
        durable.stream(
          { messages: [new HumanMessage("?")] },
          { streamMode: streamMode as never, subgraphs },
        ),
      );
      const reference = createLangChainAgent({
        model: new ScriptedModel([ai("hi")]),
        tools: [],
      });
      const theirs = await collect(
        reference.stream(
          { messages: [new HumanMessage("?")] },
          { streamMode: streamMode as never, subgraphs },
        ),
      );
      expect(shapes(ours)).toEqual(shapes(theirs));
    },
  );
});

describe("durability", () => {
  test("a thread continues after reopening its session", async () => {
    const files = new MemoryFiles();
    const config = { configurable: { thread_id: "t1" } };
    const first = await Kernel.open({ path: "threads.sqlite", files });
    await createAgent({
      model: new ScriptedModel([ai("Hello Ada.")]),
      checkpointer: first,
    }).invoke({ messages: [new HumanMessage("I am Ada.")] }, config);
    await first.close();

    const model = new ScriptedModel([ai("You are Ada.")]);
    const agent = createAgent({
      model,
      checkpointer: await kernel({ path: "threads.sqlite", files }),
    });
    const result = await agent.invoke(
      { messages: [new HumanMessage("Who am I?")] },
      config,
    );
    expect(
      result.messages.map((message: BaseMessage) => message.content),
    ).toEqual(["I am Ada.", "Hello Ada.", "Who am I?", "You are Ada."]);
    expect(model.calls[0].map((message) => message.content)).toEqual([
      "I am Ada.",
      "Hello Ada.",
      "Who am I?",
    ]);
  });

  test("a run resumes after a crash mid-tool, without calling the model again for committed steps", async () => {
    const files = new MemoryFiles();
    const config = { configurable: { thread_id: "t" } };
    let started!: () => void;
    const hung = new Promise<void>((resolve) => (started = resolve));
    const hanging = tool(
      async (_input, runtime) => {
        started();
        return new Promise<string>((_, reject) =>
          runtime.signal?.addEventListener("abort", () =>
            reject(runtime.signal?.reason),
          ),
        );
      },
      {
        name: "slow",
        description: "A tool that hangs the first time.",
        schema: z.object({ query: z.string() }),
      },
    );
    const script = [ai(call("slow", "c1", { query: "x" })), ai("Finished.")];
    const first = await Kernel.open({ path: "crash.sqlite", files });
    const run = createAgent({
      model: new ScriptedModel(script),
      tools: [hanging],
      checkpointer: first,
    })
      .invoke({ messages: [new HumanMessage("go")] }, config)
      .catch((error: unknown) => error);
    await hung;
    await first.close();
    expect(await run).toBeInstanceOf(Error);

    const reopened = await kernel({ path: "crash.sqlite", files });
    const slow = tool(
      async ({ query }: { query: string }) => `done: ${query}`,
      {
        name: "slow",
        description: "A tool that hangs the first time.",
        schema: z.object({ query: z.string() }),
      },
    );
    const model = new ScriptedModel(script.slice(1));
    const agent = createAgent({ model, tools: [slow], checkpointer: reopened });
    // Reading the thread registers the agent; the interrupted run then resumes on its own.
    await agent.getState(config);
    for (const task of reopened.session.tasks({ kind: "lc.run", live: true }))
      await reopened.session.waitTask(task.id);
    const state = await agent.getState(config);
    expect(shape(state.values.messages as BaseMessage[])).toEqual([
      ["human", "go", []],
      ["ai", "", ["slow"]],
      ["tool", "done: x", []],
      ["ai", "Finished.", []],
    ]);
    expect(model.calls).toHaveLength(1);
  });
});

const turns = () =>
  createMiddleware({
    name: "Turns",
    beforeModel: (state) => ({
      messages: [
        new HumanMessage(
          `turn ${state.messages.filter((message) => message.type === "ai").length}`,
        ),
      ],
    }),
    afterModel: (state) => ({
      messages: [
        new HumanMessage(`saw ${state.messages.at(-1)!.text || "tool calls"}`),
      ],
    }),
  });

const endEarly = () =>
  createMiddleware({
    name: "EndEarly",
    afterModel: {
      canJumpTo: ["end"],
      hook: () => ({
        messages: [new HumanMessage("stopping")],
        jumpTo: "end" as const,
      }),
    },
  });

const keepGoing = () =>
  createMiddleware({
    name: "KeepGoing",
    afterAgent: {
      canJumpTo: ["model"],
      hook: (state) =>
        state.messages.some((message) => message.content === "keep going")
          ? undefined
          : {
              messages: [new HumanMessage("keep going")],
              jumpTo: "model" as const,
            },
    },
  });

describe("middleware", () => {
  test("hooks run where createAgent runs them", async () => {
    const script = [
      ai(call("get_weather", "c1", { city: "Oslo" })),
      ai("Cold."),
    ];
    const ours = await createAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [turns()],
      checkpointer: await kernel(),
    }).invoke({
      messages: [new HumanMessage("go")],
    });
    const theirs = await createLangChainAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [turns()],
    }).invoke({
      messages: [new HumanMessage("go")],
    });
    expect(shape(ours.messages)).toEqual(shape(theirs.messages));
    expect(
      ours.messages
        .filter((m: BaseMessage) => m.type === "human")
        .map((m: BaseMessage) => m.content),
    ).toEqual(["go", "turn 0", "saw tool calls", "turn 1", "saw Cold."]);
  });

  test.each([
    ["afterModel jumping to the end", endEarly],
    ["afterAgent jumping back to the model", keepGoing],
  ])("%s moves the loop as in createAgent", async (_, middleware) => {
    const script = [
      ai(call("get_weather", "c1", { city: "Oslo" })),
      ai("Cold."),
      ai("Still cold."),
    ];
    const ours = await createAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [middleware()],
      checkpointer: await kernel(),
    }).invoke({ messages: [new HumanMessage("go")] });
    const theirs = await createLangChainAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [middleware()],
    }).invoke({
      messages: [new HumanMessage("go")],
    });
    expect(shape(ours.messages)).toEqual(shape(theirs.messages));
  });
});

const ask = tool(
  async ({ question }: { question: string }) =>
    `user said: ${interrupt({ question })}`,
  {
    name: "ask",
    description: "Ask the user a question.",
    schema: z.object({ question: z.string() }),
  },
);

const confirm = () =>
  createMiddleware({
    name: "Confirm",
    beforeModel: () => {
      const first = interrupt("first?");
      const second = interrupt("second?");
      return { messages: [new HumanMessage(`${first} then ${second}`)] };
    },
  });

/** Run to the end, answering each interrupt in turn; returns the questions asked and the final state. */
async function converse(
  agent: { invoke: (input: unknown, config: object) => Promise<any> },
  answers: string[],
  config: object,
) {
  let result = await agent.invoke(
    { messages: [new HumanMessage("go")] },
    config,
  );
  const asked: unknown[] = [];
  for (const answer of answers) {
    const [pending] = result.__interrupt__;
    asked.push(pending.value);
    result = await agent.invoke(new Command({ resume: answer }), config);
  }
  return { asked, result };
}

describe("interrupts", () => {
  test.each([
    ["a tool", [ask], [], ["blue"]],
    ["a hook", [], [confirm], ["yes", "no"]],
  ] as const)(
    "asked by %s, resume as in createAgent",
    async (_, tools, middleware, answers) => {
      const script =
        tools.length > 0
          ? [
              ai(call("ask", "c1", { question: "Favorite color?" })),
              ai("Noted."),
            ]
          : [ai("Noted.")];
      const config = { configurable: { thread_id: "asking" } };
      const model = new ScriptedModel(script);
      const durable = createAgent({
        model,
        tools: [...tools],
        middleware: middleware.map((m) => m()),
        checkpointer: await kernel(),
      });
      const ours = await converse(durable, [...answers], config);
      const reference = createLangChainAgent({
        model: new ScriptedModel(script),
        tools: [...tools],
        middleware: middleware.map((m) => m()),
        checkpointer: new MemorySaver(),
      });
      const theirs = await converse(reference, [...answers], config);
      expect(ours.asked).toEqual(theirs.asked);
      expect(shape(ours.result.messages)).toEqual(
        shape(theirs.result.messages),
      );
      expect(model.calls).toHaveLength(script.length);
    },
  );

  test("human in the loop interrupts and resumes", async () => {
    const script = [
      ai(call("get_weather", "c1", { city: "Paris" })),
      ai("Done."),
    ];
    const config = { configurable: { thread_id: "hitl" } };
    const agent = createAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [
        humanInTheLoopMiddleware({ interruptOn: { get_weather: true } }),
      ],
      checkpointer: await kernel(),
    });
    const first = await agent.invoke(
      { messages: [new HumanMessage("weather?")] },
      config,
    );
    expect(first.__interrupt__[0].value.actionRequests[0].name).toBe(
      "get_weather",
    );
    const result = await agent.invoke(
      new Command({ resume: { decisions: [{ type: "approve" }] } }),
      config,
    );
    expect(shape(result.messages).slice(-2)).toEqual([
      ["tool", "It is sunny in Paris.", []],
      ["ai", "Done.", []],
    ]);
    expect(result.__interrupt__).toBeUndefined();
  });

  test("a human in the loop rejection resumes as in createAgent", async () => {
    const script = [
      ai(call("get_weather", "c1", { city: "Paris" })),
      ai("Done."),
    ];
    const config = { configurable: { thread_id: "same" } };
    const decisions = () =>
      new Command({
        resume: { decisions: [{ type: "reject", message: "not now" }] },
      });
    const durable = createAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [
        humanInTheLoopMiddleware({ interruptOn: { get_weather: true } }),
      ],
      checkpointer: await kernel(),
    });
    await durable.invoke({ messages: [new HumanMessage("weather?")] }, config);
    const ours = await durable.invoke(decisions(), config);
    const reference = createLangChainAgent({
      model: new ScriptedModel(script),
      tools: [getWeather],
      middleware: [
        humanInTheLoopMiddleware({ interruptOn: { get_weather: true } }),
      ],
      checkpointer: new MemorySaver(),
    });
    await reference.invoke(
      { messages: [new HumanMessage("weather?")] },
      config,
    );
    const theirs = await reference.invoke(decisions(), config);
    expect(shape(ours.messages)).toEqual(shape(theirs.messages));
  });
});

describe("runs", () => {
  test("leaving a stream early aborts its run", async () => {
    let started!: () => void;
    const waiting = new Promise<void>((resolve) => (started = resolve));
    const waitForever = tool(
      async (_input, runtime) => {
        started();
        return new Promise<string>((_, reject) =>
          runtime.signal?.addEventListener("abort", () =>
            reject(runtime.signal?.reason),
          ),
        );
      },
      {
        name: "wait_forever",
        description: "Never returns.",
        schema: z.object({ reason: z.string() }),
      },
    );
    const config = { configurable: { thread_id: "left" } };
    const opened = await kernel();
    const agent = createAgent({
      model: new ScriptedModel([
        ai(call("wait_forever", "c1", { reason: "x" })),
        ai("Back."),
      ]),
      tools: [waitForever],
      checkpointer: opened,
    });
    const stream = await agent.stream(
      { messages: [new HumanMessage("go")] },
      config,
    );
    const first = stream.next();
    await waiting;
    await first;
    await stream.return(undefined);
    expect(opened.session.tasks({ kind: "lc.run", live: true })).toEqual([]);
    await agent.updateState(config, {
      messages: [new HumanMessage("interrupted")],
    });
    const result = await agent.invoke(
      { messages: [new HumanMessage("again")] },
      config,
    );
    expect(shape(result.messages)).toEqual([
      ["human", "go", []],
      ["ai", "", ["wait_forever"]],
      ["human", "interrupted", []],
      ["human", "again", []],
      ["ai", "Back.", []],
    ]);
  });

  test("an agent called from a tool runs as a subagent", async () => {
    const researcher = createAgent({
      model: new ScriptedModel([ai("Rome is in Italy.")]),
      name: "researcher",
    });
    const research = tool(
      async ({ question }: { question: string }) => {
        const result = await researcher.invoke({
          messages: [new HumanMessage(question)],
        });
        return result.messages.at(-1).content;
      },
      {
        name: "research",
        description: "Ask the researcher.",
        schema: z.object({ question: z.string() }),
      },
    );
    const script = [
      ai(call("research", "c1", { question: "Where is Rome?" })),
      ai("It is in Italy."),
    ];
    const opened = await kernel();
    const agent = createAgent({
      model: new ScriptedModel(script),
      tools: [research],
      checkpointer: opened,
    });
    const chunks = await collect(
      agent.stream(
        { messages: [new HumanMessage("Rome?")] },
        { streamMode: "updates", subgraphs: true },
      ),
    );
    const namespaces = new Set(chunks.map(([ns]) => JSON.stringify(ns)));
    expect(namespaces.has("[]")).toBe(true);
    expect([...namespaces].some((ns) => /^\["tools:c1"\]$/.test(ns))).toBe(
      true,
    );
    expect(chunks.at(-1)[1].model_request.messages[0].content).toBe(
      "It is in Italy.",
    );
    const conversations = opened.session.conversations();
    expect(conversations).toHaveLength(2);
    expect(conversations[1].owner?.conversationId).toBe(conversations[0].id);
  });

  test("a subagent's approval stops and resumes its parent", async () => {
    const deleteFile = tool(
      async ({ path }: { path: string }) => `deleted ${path}`,
      {
        name: "delete_file",
        description: "Delete a file.",
        schema: z.object({ path: z.string() }),
      },
    );
    const cleaner = createAgent({
      model: new ScriptedModel([
        ai(call("delete_file", "d1", { path: "/work/x" })),
        ai("Cleaned."),
      ]),
      tools: [deleteFile],
      middleware: [
        humanInTheLoopMiddleware({ interruptOn: { delete_file: true } }),
      ],
      name: "cleaner",
    });
    const clean = tool(
      async ({ path }: { path: string }) => {
        const result = await cleaner.invoke({
          messages: [new HumanMessage(`remove ${path}`)],
        });
        return result.messages.at(-1).content;
      },
      {
        name: "clean",
        description: "Ask the cleaner to remove a path.",
        schema: z.object({ path: z.string() }),
      },
    );
    const config = { configurable: { thread_id: "parent" } };
    const agent = createAgent({
      model: new ScriptedModel([
        ai(call("clean", "c1", { path: "/work/x" })),
        ai("All clean."),
      ]),
      tools: [clean],
      checkpointer: await kernel(),
    });
    const first = await agent.invoke(
      { messages: [new HumanMessage("clean up")] },
      config,
    );
    const [pending] = first.__interrupt__;
    expect(pending.value.actionRequests[0].args).toEqual({ path: "/work/x" });
    const result = await agent.invoke(
      new Command({
        resume: { [pending.id]: { decisions: [{ type: "approve" }] } },
      }),
      config,
    );
    expect(shape(result.messages).slice(-2)).toEqual([
      ["tool", "Cleaned.", []],
      ["ai", "All clean.", []],
    ]);
  });
});
