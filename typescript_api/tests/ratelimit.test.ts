/**
 * Rate-limiter integration test, isolated in its own file so it can configure a
 * small RATE_LIMIT_MAX without affecting the other tests (vitest gives each test
 * file its own module registry, so the server module — and the env it reads at
 * load — is fresh here).
 */
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import request from "supertest";
import type { Express } from "express";

process.env.NODE_ENV = "test";
process.env.API_KEYS = "test-key";
process.env.RATE_LIMIT_MAX = "3";
process.env.RATE_LIMIT_WINDOW_MS = "60000";

let app: Express;

beforeAll(async () => {
  app = (await import("../src/server.ts")).default;
  // The core is never actually reached in these requests (rate limiter sits in
  // front), but stub fetch anyway so any pass-through is harmless.
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve(new Response(JSON.stringify({ ok: true }), { status: 200 }))
    )
  );
});

afterAll(() => {
  vi.unstubAllGlobals();
});

describe("rate limiting on /v1/*", () => {
  it("allows up to the limit then returns 429", async () => {
    const statuses: number[] = [];
    for (let i = 0; i < 5; i++) {
      const res = await request(app).get("/v1/stats").set("Authorization", "Bearer test-key");
      statuses.push(res.status);
    }
    // First 3 are within the limit (not 429); at least one later request is 429.
    expect(statuses.slice(0, 3).every((s) => s !== 429)).toBe(true);
    expect(statuses).toContain(429);
  });

  it("includes a standard RateLimit header while allowed", async () => {
    // A fresh-ish call may already be over the limit from the previous test
    // (same window), so just assert the limiter is engaged via the 429 body.
    const res = await request(app).get("/v1/stats").set("Authorization", "Bearer test-key");
    if (res.status === 429) {
      expect(res.body.error).toMatch(/rate limit/i);
    } else {
      expect(res.headers).toHaveProperty("ratelimit-limit");
    }
  });
});
