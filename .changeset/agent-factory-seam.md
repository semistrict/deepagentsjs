---
"deepagents": minor
---

Add an experimental `agentFactory` option to `createDeepAgent` and `createSubAgentMiddleware`. It builds the agent and its declarative subagents from `createAgent`'s parameters, so another runtime can run the same middleware stack; it defaults to `createAgent`.
