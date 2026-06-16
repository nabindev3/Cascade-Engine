/**
 * A minimal three-state circuit breaker.
 *
 * Protects the gateway from hammering an unhealthy Python core: after
 * `failureThreshold` consecutive failures the circuit OPENs and calls are
 * short-circuited (fail fast) for `cooldownMs`. After the cooldown the breaker
 * goes HALF_OPEN and allows a single probe; success closes it, failure re-opens
 * it for another cooldown window.
 *
 *   CLOSED ──(failures ≥ threshold)──▶ OPEN
 *     ▲                                  │ (cooldown elapsed)
 *     │ (probe succeeds)                 ▼
 *   HALF_OPEN ◀──────────────────── (allow one probe)
 *     │ (probe fails) ─────────────▶ OPEN
 */

export type CircuitState = "closed" | "open" | "half_open";

export interface CircuitBreakerOptions {
  failureThreshold: number;
  cooldownMs: number;
  /** Injectable clock for deterministic tests. Defaults to Date.now. */
  now?: () => number;
}

export class CircuitBreaker {
  private readonly failureThreshold: number;
  private readonly cooldownMs: number;
  private readonly now: () => number;

  private failures = 0;
  private state: CircuitState = "closed";
  private openedAt = 0;

  constructor(opts: CircuitBreakerOptions) {
    this.failureThreshold = Math.max(1, opts.failureThreshold);
    this.cooldownMs = Math.max(0, opts.cooldownMs);
    this.now = opts.now ?? Date.now;
  }

  /**
   * Whether a request may proceed. Transitions OPEN → HALF_OPEN when the
   * cooldown has elapsed, so calling this has side effects by design.
   */
  allowRequest(): boolean {
    if (this.state === "open") {
      if (this.now() - this.openedAt >= this.cooldownMs) {
        this.state = "half_open";
        return true; // single probe
      }
      return false;
    }
    // closed or half_open both allow the call through.
    return true;
  }

  recordSuccess(): void {
    this.failures = 0;
    this.state = "closed";
  }

  recordFailure(): void {
    this.failures += 1;
    // A failed probe in half-open immediately re-opens the circuit.
    if (this.state === "half_open" || this.failures >= this.failureThreshold) {
      this.state = "open";
      this.openedAt = this.now();
    }
  }

  getState(): CircuitState {
    return this.state;
  }

  /** Snapshot for /health and observability. */
  snapshot(): { state: CircuitState; failures: number; openedAt: number } {
    return { state: this.state, failures: this.failures, openedAt: this.openedAt };
  }
}
