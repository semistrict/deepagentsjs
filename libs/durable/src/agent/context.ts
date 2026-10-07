/**
 * Hooks, model calls, and tools find their step through LangChain's async
 * context (see `scope.ts`). LangGraph installs it on Node; its web build,
 * which Workers load, leaves that to the app. Node and Workers with the
 * `nodejs_compat` flag both provide `AsyncLocalStorage`, so install it here,
 * unless something already has.
 */
import { AsyncLocalStorage } from "node:async_hooks";
import { AsyncLocalStorageProviderSingleton } from "@langchain/core/singletons";

AsyncLocalStorageProviderSingleton.initializeGlobalInstance(
  new AsyncLocalStorage(),
);
