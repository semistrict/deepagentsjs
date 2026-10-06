/**
 * An agent definition: what `createAgent` assembles, as data instead of a graph.
 *
 * The model call, structured output, and the composition of `wrapModelCall`
 * and `wrapToolCall` middleware follow `langchain`'s `createAgent`: the same
 * request objects, the same validation, the same errors, so middleware
 * written for it run unchanged.
 */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  type BaseMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { Runnable, type RunnableConfig } from "@langchain/core/runnables";
import { ToolInputParsingException } from "@langchain/core/tools";
import {
  getInteropZodObjectShape,
  interopParse,
  isInteropZodObject,
} from "@langchain/core/utils/types";
import { Command, isCommand, isGraphBubbleUp } from "@langchain/langgraph";
import {
  type AgentMiddleware,
  MiddlewareError,
  MultipleStructuredOutputsError,
  ProviderStrategy,
  StructuredOutputParsingError,
  ToolInvocationError,
  ToolStrategy,
} from "langchain";
import { initChatModel } from "langchain/chat_models/universal";
import { Schema } from "./schema.js";
import type { Update } from "./thread.js";

/** The model node's name, as `createAgent` streams it. */
export const MODEL = "model_request";
export const TOOLS = "tools";

type AnyTool = { name: string; [key: string]: any };
type Hook = "beforeAgent" | "beforeModel" | "afterModel" | "afterAgent";
export type JumpTo = "model" | "tools" | "end";

/** What `createAgent` takes, and what this runtime honors of it. */
export interface AgentParams {
  model: string | BaseChatModel;
  tools?: readonly AnyTool[];
  systemPrompt?: string | SystemMessage;
  middleware?: readonly AgentMiddleware[];
  responseFormat?: unknown;
  stateSchema?: unknown;
  contextSchema?: unknown;
  store?: unknown;
  name?: string;
  description?: string;
}

/** A client tool runs in this process; a server tool is a provider's, described by a plain object. */
function isClientTool(tool: AnyTool): boolean {
  return Runnable.isRunnable(tool);
}

function hookFunction(hook: unknown): (state: any, runtime: any) => unknown {
  return typeof hook === "function"
    ? (hook as never)
    : (hook as { hook: never }).hook;
}

function hookJumps(hook: unknown): JumpTo[] {
  return typeof hook === "function"
    ? []
    : ((hook as { canJumpTo?: JumpTo[] }).canJumpTo ?? []);
}

/** The context a middleware sees: the run's, through its own context schema when it has one. */
export function contextFor(
  middleware: AgentMiddleware,
  context: unknown,
): unknown {
  const schema = middleware.contextSchema;
  if (schema === undefined || !isInteropZodObject(schema)) return context;
  const given = (context ?? {}) as Record<string, unknown>;
  const relevant = Object.fromEntries(
    Object.keys(getInteropZodObjectShape(schema))
      .filter((key) => key in given)
      .map((key) => [key, given[key]]),
  );
  return interopParse(schema, relevant);
}

type ResponseFormat =
  | { type: "tool"; tools: Record<string, ToolStrategy> }
  | { type: "native"; strategy: ProviderStrategy };

/** A tool strategy, recognized by shape so another copy of `langchain` passes too. */
function isToolStrategy(value: unknown): value is ToolStrategy {
  return (
    typeof value === "object" &&
    value !== null &&
    "tool" in value &&
    "schema" in value &&
    typeof (value as ToolStrategy).parse === "function"
  );
}

/** A provider strategy, recognized by shape so another copy of `langchain` passes too. */
function isProviderStrategy(value: unknown): value is ProviderStrategy {
  return (
    typeof value === "object" &&
    value !== null &&
    !("tool" in value) &&
    "schema" in value &&
    "strict" in value &&
    typeof (value as ProviderStrategy).parse === "function"
  );
}

/** A response format as the strategies `createAgent` would use for it. */
function strategies(
  format: unknown,
  model: unknown,
): (ToolStrategy | ProviderStrategy)[] {
  if (format === undefined || format === null) return [];
  if (typeof format === "object" && "__responseFormatUndefined" in format)
    return [];
  if (Array.isArray(format)) {
    if (
      format.every((item) => isToolStrategy(item) || isProviderStrategy(item))
    )
      return format;
    return format.map((item) => ToolStrategy.fromSchema(item));
  }
  if (isToolStrategy(format) || isProviderStrategy(format)) return [format];
  const profile = (
    model as { profile?: { structuredOutput?: boolean } } | undefined
  )?.profile;
  return profile?.structuredOutput === true
    ? [ProviderStrategy.fromSchema(format as never)]
    : [ToolStrategy.fromSchema(format as never)];
}

function responseFormat(
  format: unknown,
  model: unknown,
): ResponseFormat | undefined {
  const resolved = strategies(format, model);
  if (resolved.length === 0) return undefined;
  if (resolved.every(isProviderStrategy)) {
    return { type: "native", strategy: resolved[0] };
  }
  const tools: Record<string, ToolStrategy> = {};
  for (const item of resolved)
    if (isToolStrategy(item)) tools[item.name] = item;
  return { type: "tool", tools };
}

export interface ModelRequest {
  model: any;
  messages: BaseMessage[];
  systemPrompt: string;
  systemMessage: SystemMessage;
  tools: AnyTool[];
  toolChoice?: unknown;
  responseFormat?: unknown;
  state: Record<string, unknown>;
  runtime: Record<string, unknown>;
  modelSettings?: Record<string, unknown>;
}

type ModelResult =
  | AIMessage
  | Command
  | { structuredResponse: unknown; messages: BaseMessage[] };

function isModelResult(value: unknown): value is ModelResult {
  return (
    AIMessage.isInstance(value) ||
    isCommand(value) ||
    (typeof value === "object" &&
      value !== null &&
      "structuredResponse" in value &&
      "messages" in value)
  );
}

/** What one model call produced: the step's updates, and whether it asks for another model call. */
export interface ModelOutcome {
  updates: Update[];
  retry: boolean;
}

let specs = 0;

/** Everything a run needs to execute one agent's loop. */
export class AgentSpec {
  readonly name: string | undefined;
  readonly systemMessage: SystemMessage;
  readonly middleware: readonly AgentMiddleware[];
  readonly tools: AnyTool[];
  readonly clientTools: Map<string, AnyTool>;
  readonly returnDirect: Set<string>;
  readonly schema: Schema;
  readonly hooks: Record<Hook, AgentMiddleware[]>;
  readonly wrapsModel: AgentMiddleware[];
  readonly wrapsTools: AgentMiddleware[];
  /** Tools can run: client tools, or middleware that executes tools added at runtime. */
  readonly hasTools: boolean;
  readonly config: RunnableConfig;
  /** Tells apart the warm thread states of agents whose fields differ. */
  readonly schemaKey = `schema-${++specs}`;
  #model: Promise<any> | undefined;
  #format: { format: ResponseFormat | undefined } | undefined;

  constructor(readonly params: AgentParams) {
    this.name = params.name;
    this.systemMessage =
      params.systemPrompt === undefined
        ? new SystemMessage("")
        : typeof params.systemPrompt === "string"
          ? new SystemMessage(params.systemPrompt)
          : params.systemPrompt;
    this.middleware = params.middleware ?? [];
    const names = new Set<string>();
    for (const middleware of this.middleware) {
      if (names.has(middleware.name))
        throw new Error(
          `Middleware ${middleware.name} is defined multiple times`,
        );
      names.add(middleware.name);
    }
    this.tools = [
      ...(params.tools ?? []),
      ...this.middleware.flatMap(
        (middleware) => (middleware.tools ?? []) as AnyTool[],
      ),
    ];
    this.clientTools = new Map(
      this.tools.filter(isClientTool).map((tool) => [tool.name, tool]),
    );
    this.returnDirect = new Set(
      [...this.clientTools.values()]
        .filter((tool) => tool.returnDirect)
        .map((tool) => tool.name),
    );
    this.schema = new Schema([
      params.stateSchema,
      ...this.middleware.map((middleware) => middleware.stateSchema),
    ]);
    const having = (key: keyof AgentMiddleware) =>
      this.middleware.filter((middleware) => middleware[key] !== undefined);
    this.hooks = {
      beforeAgent: having("beforeAgent"),
      beforeModel: having("beforeModel"),
      afterModel: having("afterModel"),
      afterAgent: having("afterAgent"),
    };
    this.wrapsModel = having("wrapModelCall");
    this.wrapsTools = having("wrapToolCall");
    this.hasTools = this.clientTools.size > 0 || this.wrapsTools.length > 0;
    this.config = {
      metadata: {
        ls_integration: "langchain_create_agent",
        ...(params.name ? { lc_agent_name: params.name } : {}),
      },
      configurable: { ls_agent_type: "root" },
    };
  }

  model(): Promise<any> {
    this.#model ??=
      typeof this.params.model === "string"
        ? initChatModel(this.params.model)
        : Promise.resolve(this.params.model);
    return this.#model;
  }

  /**
   * The strategies a request's response format uses. The agent's own format
   * resolves once, so its output tools keep the names they were bound with.
   */
  #formatFor(requested: unknown, model: unknown): ResponseFormat | undefined {
    if (requested !== this.params.responseFormat)
      return responseFormat(requested, model);
    this.#format ??= { format: responseFormat(requested, model) };
    return this.#format.format;
  }

  /** Whether a tool call asks for structured output rather than a client tool, as `createAgent` routes it. */
  isOutputTool(name: string): boolean {
    const format = this.#format?.format;
    return (
      (format?.type === "tool" && name in format.tools) ||
      name.startsWith("extract-")
    );
  }

  hook(
    middleware: AgentMiddleware,
    hook: Hook,
  ): { run: (state: any, runtime: any) => unknown; jumps: JumpTo[] } {
    const declared = middleware[hook];
    return { run: hookFunction(declared), jumps: hookJumps(declared) };
  }

  /**
   * Call the model through every `wrapModelCall` middleware, first outermost,
   * as `createAgent`'s model node does; returns the step's updates.
   */
  async callModel(
    state: Record<string, unknown>,
    runtime: Record<string, unknown>,
    config: RunnableConfig,
  ): Promise<ModelOutcome> {
    const model = await this.model();
    const context = runtime.context;
    let current = this.systemMessage;
    let lastAi: AIMessage | null = null;
    const collected: Command[] = [];

    const base = async (request: ModelRequest): Promise<ModelResult> => {
      const format = this.#formatFor(request.responseFormat, request.model);
      const bound = await this.#bind(request.model, request, format);
      const messages = [
        ...(current.text === "" ? [] : [current]),
        ...request.messages,
      ];
      const response = (await bound.invoke(messages, config)) as AIMessage;
      lastAi = response;
      if (format?.type === "native") {
        const structuredResponse = format.strategy.parse(response);
        if (structuredResponse)
          return { structuredResponse, messages: [response] };
        if (!response.tool_calls?.length) {
          const title =
            typeof format.strategy.schema?.title === "string"
              ? format.strategy.schema.title
              : "providerStrategy";
          throw new StructuredOutputParsingError(title, [
            "Model output did not satisfy the provided response schema.",
          ]);
        }
        return response;
      }
      if (format === undefined || !response.tool_calls) return response;
      const calls = response.tool_calls.filter(
        (call) => call.name in format.tools,
      );
      if (calls.length === 0) return response;
      if (calls.length > 1) {
        return this.#outputError(
          new MultipleStructuredOutputsError(calls.map((call) => call.name)),
          response,
          calls[0],
          format,
        );
      }
      const strategy = format.tools[calls[0].name];
      try {
        const structuredResponse = strategy.parse(calls[0].args);
        return {
          structuredResponse,
          messages: [
            response,
            new ToolMessage({
              tool_call_id: calls[0].id ?? "",
              content: JSON.stringify(structuredResponse),
              name: calls[0].name,
            }),
            new AIMessage(
              strategy.options?.toolMessageContent ??
                `Returning structured response: ${JSON.stringify(structuredResponse)}`,
            ),
          ],
        };
      } catch (error) {
        return this.#outputError(error as Error, response, calls[0], format);
      }
    };

    let handler = base;
    for (const middleware of [...this.wrapsModel].reverse()) {
      const inner = handler;
      handler = async (request) => {
        const baseline = current;
        const own = {
          ...request,
          state,
          runtime: Object.freeze({
            ...runtime,
            context: contextFor(middleware, context),
          }),
        } as ModelRequest;
        const validated = async (next: ModelRequest): Promise<ModelResult> => {
          current = baseline;
          this.#checkTools(middleware, next.tools ?? []);
          let normalized = next;
          const promptChanged = next.systemPrompt !== current.text;
          const messageChanged = next.systemMessage !== current;
          if (promptChanged && messageChanged) {
            throw new Error(
              "Cannot change both systemPrompt and systemMessage in the same request.",
            );
          }
          if (promptChanged) {
            current = new SystemMessage({
              content: [{ type: "text", text: next.systemPrompt }],
            });
            normalized = {
              ...next,
              systemPrompt: current.text,
              systemMessage: current,
            };
          }
          if (messageChanged) {
            current = new SystemMessage({ ...next.systemMessage });
            normalized = {
              ...next,
              systemPrompt: current.text,
              systemMessage: current,
            };
          }
          const result = await inner(normalized);
          // Middleware always see a message from their handler; a command it carried is kept.
          if (isCommand(result) && lastAi !== null) {
            if (!collected.includes(result)) collected.push(result);
            return lastAi;
          }
          return result;
        };
        try {
          const result = await middleware.wrapModelCall!(
            own as never,
            validated as never,
          );
          if (!isModelResult(result)) {
            throw new Error(
              `Invalid response from "wrapModelCall" in middleware "${middleware.name}": expected AIMessage or Command, got ${typeof result}`,
            );
          }
          if (AIMessage.isInstance(result)) lastAi = result;
          else if (isCommand(result)) collected.push(result);
          return result;
        } catch (error) {
          throw MiddlewareError.wrap(error, middleware.name);
        }
      };
    }

    current = this.systemMessage;
    const response = await handler({
      model,
      responseFormat: this.params.responseFormat,
      systemPrompt: current.text,
      systemMessage: current,
      messages: state.messages as BaseMessage[],
      tools: this.tools,
      state,
      runtime,
    });

    if (
      typeof response === "object" &&
      response !== null &&
      "structuredResponse" in response &&
      "messages" in response
    ) {
      return {
        updates: [
          ["messages", response.messages],
          ["structuredResponse", response.structuredResponse],
        ],
        retry: false,
      };
    }
    const updates: Update[] = [];
    const ai: AIMessage | null = AIMessage.isInstance(response)
      ? response
      : lastAi;
    if (ai !== null) {
      ai.name = this.name;
      ai.lc_kwargs.name = this.name;
      updates.push(["messages", [ai]]);
    }
    const commands =
      isCommand(response) && !collected.includes(response)
        ? [response, ...collected]
        : collected;
    let retry = false;
    for (const command of commands) {
      updates.push(...commandUpdates(command));
      const goto = Array.isArray(command.goto) ? command.goto : [command.goto];
      retry ||= goto.includes(MODEL);
    }
    return { updates, retry };
  }

  /** Tools a `wrapModelCall` added need a `wrapToolCall` to run them; replacing a registered tool is refused. */
  #checkTools(middleware: AgentMiddleware, tools: AnyTool[]): void {
    const added = tools.filter(
      (tool) => isClientTool(tool) && !this.clientTools.has(tool.name),
    );
    if (added.length > 0 && this.wrapsTools.length === 0) {
      throw new Error(
        `You have added a new tool in "wrapModelCall" hook of middleware "${middleware.name}": ${added
          .map((tool) => tool.name)
          .join(
            ", ",
          )}. This is not supported unless a middleware provides a "wrapToolCall" handler to execute it.`,
      );
    }
    const replaced = tools.filter(
      (tool) =>
        isClientTool(tool) &&
        this.clientTools.has(tool.name) &&
        this.clientTools.get(tool.name) !== tool,
    );
    if (replaced.length > 0) {
      throw new Error(
        `You have modified a tool in "wrapModelCall" hook of middleware "${middleware.name}": ${replaced
          .map((tool) => tool.name)
          .join(", ")}. This is not supported.`,
      );
    }
  }

  async #bind(
    model: any,
    request: ModelRequest,
    format: ResponseFormat | undefined,
  ): Promise<Runnable> {
    const outputTools =
      format?.type === "tool"
        ? Object.values(format.tools).map((strategy) => strategy.tool)
        : [];
    const tools = [...(request.tools ?? this.tools), ...outputTools];
    const toolChoice =
      request.toolChoice || (outputTools.length > 0 ? "any" : undefined);
    const options: Record<string, unknown> = {};
    if (format?.type === "native") {
      const strict =
        request.modelSettings?.strict ?? format.strategy.strict ?? true;
      const schema = format.strategy.schema;
      Object.assign(options, {
        response_format: {
          type: "json_schema",
          json_schema: {
            name: schema?.name ?? "extract",
            description: schema?.description,
            schema,
            strict,
          },
        },
        outputConfig: { format: { type: "json_schema", schema } },
        responseSchema: schema,
        ls_structured_output_format: {
          kwargs: { method: "json_schema" },
          schema,
        },
        strict: request.modelSettings?.strict,
      });
    }
    const settings = {
      ...options,
      ...request.modelSettings,
      tool_choice: toolChoice,
    };
    let target = model;
    if (
      typeof target.bindTools !== "function" &&
      typeof target._getModelInstance === "function"
    ) {
      target = await target._getModelInstance();
    }
    if (typeof target.bindTools !== "function") {
      throw new Error(
        `model ${target?.constructor?.name ?? typeof target} cannot bind tools`,
      );
    }
    return target.bindTools(tools, settings);
  }

  /** A structured output the model got wrong: tell it why and call it again, as `createAgent` does. */
  async #outputError(
    error: Error,
    response: AIMessage,
    call: ToolCall,
    format: Extract<ResponseFormat, { type: "tool" }>,
  ): Promise<Command> {
    const handle = Object.values(format.tools)[0]?.options?.handleError;
    if (!call.id)
      throw new Error(
        "Tool call ID is required to handle tool output errors. Please provide a tool call ID.",
      );
    if (handle === false) throw error;
    let content = error.message;
    if (typeof handle === "string") content = handle;
    else if (typeof handle === "function") {
      content = await (handle as (error: Error) => string | Promise<string>)(
        error,
      );
      if (typeof content !== "string")
        throw new Error("Error handler must return a string.");
    }
    return new Command({
      update: {
        messages: [
          response,
          new ToolMessage({ content, tool_call_id: call.id }),
        ],
      },
      goto: MODEL,
    });
  }

  /**
   * Run one tool call through every `wrapToolCall` middleware, first
   * outermost, handling errors as `createAgent`'s tool node does.
   */
  async callTool(
    call: ToolCall,
    state: Record<string, unknown>,
    runtime: Record<string, unknown>,
    config: RunnableConfig,
  ): Promise<ToolMessage | Command> {
    const invalid = (name: string) =>
      new ToolMessage({
        content: `Error: ${name} is not a valid tool, try one of [${[...this.clientTools.keys()].join(", ")}].`,
        tool_call_id: call.id!,
        name,
        status: "error",
      });
    const base = async (request: {
      toolCall: ToolCall;
      tool?: AnyTool;
    }): Promise<ToolMessage | Command> => {
      const { toolCall } = request;
      const tool = request.tool ?? this.clientTools.get(toolCall.name);
      if (tool === undefined) return invalid(toolCall.name);
      let output: unknown;
      try {
        output = await tool.invoke(
          { ...toolCall, type: "tool_call" },
          {
            ...config,
            config,
            toolCallId: toolCall.id!,
            state,
            signal: config.signal,
          },
        );
      } catch (error) {
        // `@langchain/core` is a peer dependency, so its error class is the tools' own, as in `createAgent`.
        // oxlint-disable-next-line no-instanceof/no-instanceof
        if (error instanceof ToolInputParsingException)
          throw new ToolInvocationError(error, toolCall);
        throw error;
      }
      if (ToolMessage.isInstance(output) || isCommand(output)) return output;
      return new ToolMessage({
        name: tool.name,
        content: typeof output === "string" ? output : JSON.stringify(output),
        tool_call_id: toolCall.id!,
      });
    };
    const request = {
      toolCall: call,
      tool: this.clientTools.get(call.name),
      state,
      runtime,
    };

    if (this.wrapsTools.length > 0) {
      let handler = base as (request: any) => Promise<ToolMessage | Command>;
      for (const middleware of [...this.wrapsTools].reverse()) {
        const inner = handler;
        handler = async (outer) => {
          const original = outer.state;
          const next = (passed: any) =>
            inner({ ...passed, state: { ...original, ...passed.state } });
          try {
            const result = await middleware.wrapToolCall!(
              {
                ...outer,
                runtime: Object.freeze({
                  ...runtime,
                  context: contextFor(middleware, runtime.context),
                }),
              },
              next,
            );
            if (!ToolMessage.isInstance(result) && !isCommand(result)) {
              throw new Error(
                `Invalid response from "wrapToolCall" in middleware "${middleware.name}": expected ToolMessage or Command, got ${typeof result}`,
              );
            }
            return result;
          } catch (error) {
            throw MiddlewareError.wrap(error, middleware.name);
          }
        };
      }
      // Errors from middleware reach the run, as `createAgent`'s default error handling lets them.
      return handler(request);
    }

    if (!this.clientTools.has(call.name)) return invalid(call.name);
    try {
      return await base(request);
    } catch (error) {
      if (isGraphBubbleUp(error) || config.signal?.aborted) throw error;
      // Thrown just above, by this module's own copy of the class.
      // oxlint-disable-next-line no-instanceof/no-instanceof
      if (error instanceof ToolInvocationError) {
        return new ToolMessage({
          content: error.message,
          tool_call_id: call.id!,
          name: call.name,
        });
      }
      return new ToolMessage({
        content: `${error}\n Please fix your mistakes.`,
        tool_call_id: call.id!,
        name: call.name,
      });
    }
  }
}

/** A command's state update as updates. Only updates are followed: there is no graph to send to. */
export function commandUpdates(command: Command): Update[] {
  if (command.graph !== undefined) {
    throw new Error(
      `Command(graph) is a graph instruction the durable runtime does not follow: ${JSON.stringify(command)}`,
    );
  }
  const update = command.update;
  if (update === undefined || update === null) return [];
  return Array.isArray(update)
    ? (update as Update[])
    : (Object.entries(update) as Update[]);
}
