/**
 * Stream events from runs to `stream` callers.
 *
 * A run publishes each event right after the commit it describes, so a
 * caller never sees anything that is not durable (tokens aside, which are
 * published as they arrive).
 */

/** Mode of the event a run publishes once it has ended. */
export const END = "__end__";

export interface Event {
  /** The conversation of the top-level thread the event belongs to. */
  thread: number;
  /** Namespace below that thread: `[]` for the thread itself. */
  ns: string[];
  mode: string;
  data: unknown;
  /** The run task that published it. */
  run: number;
}

/** The events of one thread, in publication order. */
export class Subscription implements AsyncIterator<Event> {
  readonly #queue: Event[] = [];
  #waiting: ((event: IteratorResult<Event>) => void) | undefined;

  constructor(
    readonly thread: number,
    readonly close: () => void,
  ) {}

  deliver(event: Event): void {
    const waiting = this.#waiting;
    if (waiting === undefined) this.#queue.push(event);
    else {
      this.#waiting = undefined;
      waiting({ value: event, done: false });
    }
  }

  next(): Promise<IteratorResult<Event>> {
    const queued = this.#queue.shift();
    if (queued !== undefined)
      return Promise.resolve({ value: queued, done: false });
    return new Promise((resolve) => (this.#waiting = resolve));
  }
}

/** Fans events out to the subscriptions of their thread. */
export class EventBus {
  #subscriptions: Subscription[] = [];

  /** Receive events of `thread` from now on. */
  subscribe(thread: number): Subscription {
    const subscription: Subscription = new Subscription(thread, () => {
      this.#subscriptions = this.#subscriptions.filter(
        (existing) => existing !== subscription,
      );
    });
    this.#subscriptions.push(subscription);
    return subscription;
  }

  publish(event: Event): void {
    for (const subscription of this.#subscriptions) {
      if (subscription.thread === event.thread) subscription.deliver(event);
    }
  }
}
