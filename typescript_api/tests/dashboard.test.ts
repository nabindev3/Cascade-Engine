/**
 * The live dashboard (Step 6) is served by the gateway as a static HTML shell
 * that polls /health and /v1/stats from the browser.
 */
import { beforeAll, describe, expect, it } from "vitest";
import request from "supertest";
import type { Express } from "express";

process.env.NODE_ENV = "test";
process.env.API_KEYS = "test-key";
process.env.RATE_LIMIT_MAX = "100000";

let app: Express;

beforeAll(async () => {
  app = (await import("../src/server.ts")).default;
});

describe("GET /dashboard", () => {
  it("serves the dashboard HTML without auth", async () => {
    const res = await request(app).get("/dashboard");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/html/);
    expect(res.text).toMatch(/Cascade Engine/);
    expect(res.text).toMatch(/\/v1\/stats/); // wired to the stats endpoint
  });

  it("sets a CSP that permits the React/Babel CDN", async () => {
    const res = await request(app).get("/dashboard");
    expect(res.headers["content-security-policy"]).toMatch(/unpkg\.com/);
  });
});
