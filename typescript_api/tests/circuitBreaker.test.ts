import { describe, it, expect } from "vitest";
import { CircuitBreaker } from "../src/circuitBreaker.ts";

describe("CircuitBreaker", () => {
  it("stays closed below the failure threshold, opens at it", () => {
    const cb = new CircuitBreaker({ failureThreshold: 3, cooldownMs: 1000, now: () => 0 });
    expect(cb.allowRequest()).toBe(true);
    cb.recordFailure();
    cb.recordFailure();
    expect(cb.getState()).toBe("closed");
    cb.recordFailure(); // third failure trips it
    expect(cb.getState()).toBe("open");
    expect(cb.allowRequest()).toBe(false);
  });

  it("half-opens after the cooldown and closes on a successful probe", () => {
    let t = 0;
    const cb = new CircuitBreaker({ failureThreshold: 1, cooldownMs: 1000, now: () => t });
    cb.recordFailure();
    expect(cb.getState()).toBe("open");
    expect(cb.allowRequest()).toBe(false);

    t = 1000; // cooldown elapsed
    expect(cb.allowRequest()).toBe(true);
    expect(cb.getState()).toBe("half_open");

    cb.recordSuccess();
    expect(cb.getState()).toBe("closed");
  });

  it("re-opens immediately if the half-open probe fails", () => {
    let t = 0;
    const cb = new CircuitBreaker({ failureThreshold: 1, cooldownMs: 1000, now: () => t });
    cb.recordFailure();
    t = 1000;
    cb.allowRequest(); // → half_open
    cb.recordFailure(); // probe failed
    expect(cb.getState()).toBe("open");
    expect(cb.allowRequest()).toBe(false); // back into cooldown
  });

  it("a success resets the consecutive-failure count", () => {
    const cb = new CircuitBreaker({ failureThreshold: 3, cooldownMs: 1000, now: () => 0 });
    cb.recordFailure();
    cb.recordFailure();
    cb.recordSuccess(); // resets
    cb.recordFailure();
    cb.recordFailure();
    expect(cb.getState()).toBe("closed");
  });
});
