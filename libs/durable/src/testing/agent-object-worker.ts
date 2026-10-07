/**
 * A Worker whose Durable Object runs a durable Deep Agent over its own
 * storage, for the Cloudflare tests. Each request runs one step of a
 * conversation and answers the thread's messages as JSON.
 */
import { HumanMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";
import { createDeepAgent } from "../agent/index.js";
import { DurableObjectFiles, Kernel, type SqlStorage } from "../cloudflare.js";
import { ScriptedModel, ai, call } from "./scripted.js";

interface State {
  storage: { sql: SqlStorage };
}

interface Namespace {
  idFromName(name: string): unknown;
  get(id: unknown): { fetch(request: Request): Promise<Response> };
}

const script = () => [
  ai(call("write_file", "w1", { file_path: "/notes.md", content: "hello\n" })),
  ai(call("write_file", "w2", { file_path: "/more.md", content: "more\n" })),
  ai("Both written."),
];

export class AgentObject {
  #kernel: Promise<Kernel> | undefined;

  constructor(readonly state: State) {}

  #agent(kernel: Kernel) {
    return createDeepAgent({
      model: new ScriptedModel(script(), { byHistory: true }),
      checkpointer: kernel,
      interruptOn: { write_file: true },
    });
  }

  async fetch(request: Request): Promise<Response> {
    const action = new URL(request.url).pathname.slice(1);
    this.#kernel ??= Kernel.open({
      path: "agent.sqlite",
      files: new DurableObjectFiles(this.state.storage.sql),
    });
    const agent = this.#agent(await this.#kernel);
    const config = { configurable: { thread_id: "notes" } };
    const result =
      action === "start"
        ? await agent.invoke(
            { messages: [new HumanMessage("take notes")] },
            config,
          )
        : await agent.invoke(
            new Command({ resume: { decisions: [{ type: "approve" }] } }),
            config,
          );
    return Response.json({
      messages: result.messages.map(
        (message: { type: string; content: unknown }) => [
          message.type,
          message.content,
        ],
      ),
      files: Object.keys(result.files ?? {}),
      asked: (result.__interrupt__ ?? []).map(
        (pending: { value: { actionRequests: { name: string }[] } }) =>
          pending.value.actionRequests[0].name,
      ),
    });
  }
}

export default {
  fetch(request: Request, env: { AGENT: Namespace }): Promise<Response> {
    return env.AGENT.get(env.AGENT.idFromName("agent")).fetch(request);
  },
};
