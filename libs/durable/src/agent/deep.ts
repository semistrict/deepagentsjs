/**
 * Deep Agents on the durable runtime: `createDeepAgent`, and a factory for
 * apps that assemble agents themselves.
 */
import * as deepagents from "deepagents";
import type { AgentFactory, CreateDeepAgentParams } from "deepagents";
import type { Kernel } from "../kernel.js";
import {
  createAgent,
  type CreateAgentParams,
  type DurableAgent,
} from "./agent.js";

/**
 * A `createAgent` replacement for `agentFactory` options, building agents
 * whose threads live in `checkpointer`. The LangGraph checkpointer callers
 * pass along is ignored; `store`, when given, replaces theirs.
 */
export function agentFactory(
  checkpointer?: Kernel,
  options: { store?: unknown } = {},
): AgentFactory {
  return (params) => {
    const { checkpointer: _ignored, ...rest } = params as typeof params & {
      checkpointer?: unknown;
    };
    const store = options.store ?? rest.store;
    return createAgent({
      ...(rest as unknown as CreateAgentParams),
      store,
      checkpointer,
    }) as never;
  };
}

/** `createDeepAgent`'s parameters, for any of its type parameterizations. */
// oxlint-disable-next-line typescript/no-explicit-any
type DeepAgentParams = CreateDeepAgentParams<
  any,
  any,
  any,
  any,
  any,
  any,
  any,
  any
>;

/**
 * Build a Deep Agent whose loop, and every declarative subagent's, runs on a
 * durable kernel. Takes `deepagents`' `createDeepAgent` parameters;
 * `checkpointer` is the kernel the threads live in.
 */
export function createDeepAgent(
  params: Omit<DeepAgentParams, "checkpointer" | "agentFactory"> & {
    checkpointer?: Kernel;
  } = {},
): DurableAgent {
  const { checkpointer, ...rest } = params;
  return deepagents.createDeepAgent({
    ...rest,
    agentFactory: agentFactory(checkpointer),
  }) as unknown as DurableAgent;
}
