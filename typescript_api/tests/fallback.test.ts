/**
 * Integration tests for the circuit breaker + direct cloud fallback (Step 8).
 *
 * Env is configured BEFORE the server module loads: a low failure threshold so
 * the breaker trips quickly, and a fallback key so degraded-mode is active.
 * `fetch` is stubbed to distinguish core (`/infer`) from the cloud fallback
 * (`/chat/completions`).
 */
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import type { Express } from "express";
import type { CircuitBreaker } from "../src/circuitBreaker.ts";

process.env.NODE_ENV = "test";
process.env.API_KEYS = "test-key";
process.env.RATE_LIMIT_MAX = "100000";
process.env.CORE_CIRCUIT_FAILURE_THRESHOLD = "2";
process.env.CORE_CIRCUIT_COOLDOWN_MS = "10000";
process.env.FALLBACK_OPENAI_API_KEY = "fallback-key";
process.env.FALLBACK_MODEL = "gpt-4o-mini";

const AUTH = "Bearer test-key";

let app: Express;
let coreBreaker: CircuitBreaker;

beforeAll(async () => {
  const mod = await import("../src/server.ts");
  app = mod.default;
  coreBreaker = mod.coreBreaker;
});

beforeEach(() => {
  coreBreaker.recordSuccess(); // reset breaker state between tests
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function cloudCompletion(content: string): Response {
  return new Response(
    JSON.stringify({ choices: [{ message: { content } }] }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
}

describe("Direct cloud fallback", () => {
  it("serves a degraded answer from the cloud when the core is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.includes("/chat/completions")) return cloudCompletion("fallback answer");
        throw new Error("core down");
      })
    );

    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "hi" });

    expect(res.status).toBe(200);
    expect(res.body.degraded).toBe(true);
    expect(res.body.content).toBe("fallback answer");
    expect(res.body.engine_used).toMatch(/gateway-fallback/);
  });

  it("opens the circuit after repeated failures and stops calling the core", async () => {
    let coreCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.includes("/chat/completions")) return cloudCompletion("fb");
        coreCalls += 1;
        throw new Error("core down");
      })
    );

    // Threshold is 2 → two failing requests open the breaker.
    await request(app).post("/v1/infer").set("Authorization", AUTH).send({ prompt: "a" });
    await request(app).post("/v1/infer").set("Authorization", AUTH).send({ prompt: "b" });
    expect(coreBreaker.getState()).toBe("open");

    const callsAfterOpen = coreCalls;
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "c" });

    expect(coreCalls).toBe(callsAfterOpen); // short-circuited: no new core call
    expect(res.status).toBe(200); // still answered, via fallback
    expect(res.body.degraded).toBe(true);
  });

  it("returns 503 when the circuit is open and the fallback also fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => {
        throw new Error("everything down");
      })
    );

    await request(app).post("/v1/infer").set("Authorization", AUTH).send({ prompt: "a" });
    await request(app).post("/v1/infer").set("Authorization", AUTH).send({ prompt: "b" });
    expect(coreBreaker.getState()).toBe("open");

    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "c" });

    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/circuit open/i);
  });

  it("does NOT fall back on a 4xx from the core (client error propagates)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.includes("/chat/completions")) return cloudCompletion("should not be used");
        return new Response("bad request", { status: 400 });
      })
    );

    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "hi" });

    expect(res.status).toBe(400);
    expect(res.body.detail).toBe("bad request");
    expect(coreBreaker.getState()).toBe("closed"); // 4xx must not trip the breaker
  });
});
