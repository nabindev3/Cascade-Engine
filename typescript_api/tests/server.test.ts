/**
 * Integration tests for the Cascade API Gateway.
 *
 * These drive the Express app via supertest with the Python core's `fetch`
 * calls mocked, so they validate the gateway's own responsibilities — auth,
 * input validation, proxying, status/error propagation, and batch limits —
 * without a running core service. The rate limiter is effectively disabled
 * here (RATE_LIMIT_MAX is set very high) and tested in isolation in
 * `ratelimit.test.ts`.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import request from "supertest";
import type { Express } from "express";

// Must be set BEFORE importing the server module (env is read at module load).
process.env.NODE_ENV = "test";
process.env.API_KEYS = "test-key,second-key";
process.env.RATE_LIMIT_MAX = "100000"; // don't let the limiter interfere here

const AUTH = "Bearer test-key";

let app: Express;

beforeAll(async () => {
  app = (await import("../src/server.ts")).default;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Helper: stub global fetch with a sequence-agnostic responder. */
function stubFetch(impl: (url: string, init?: any) => Promise<Response> | Response) {
  vi.stubGlobal("fetch", vi.fn(impl as any));
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("GET /health", () => {
  it("reports core health when the core is reachable", async () => {
    stubFetch(() => jsonResponse({ status: "ok", engines: 3 }));
    const res = await request(app).get("/health");
    expect(res.status).toBe(200);
    expect(res.body.gateway).toBe("ok");
    expect(res.body.core).toEqual({ status: "ok", engines: 3 });
  });

  it("returns 503 with core 'unreachable' when the core is down", async () => {
    stubFetch(() => {
      throw new Error("ECONNREFUSED");
    });
    const res = await request(app).get("/health");
    expect(res.status).toBe(503);
    expect(res.body.core).toBe("unreachable");
  });
});

describe("POST /v1/infer — auth", () => {
  it("rejects a request with no Authorization header (401)", async () => {
    const res = await request(app).post("/v1/infer").send({ prompt: "hi" });
    expect(res.status).toBe(401);
  });

  it("rejects a non-Bearer Authorization header (401)", async () => {
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", "Basic abc")
      .send({ prompt: "hi" });
    expect(res.status).toBe(401);
  });

  it("rejects an unknown API key (403)", async () => {
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", "Bearer wrong-key")
      .send({ prompt: "hi" });
    expect(res.status).toBe(403);
  });

  it("accepts any configured key", async () => {
    stubFetch(() => jsonResponse(sampleCoreResponse()));
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", "Bearer second-key")
      .send({ prompt: "hi" });
    expect(res.status).toBe(200);
  });
});

describe("POST /v1/infer — validation & proxying", () => {
  it("returns 400 when prompt is missing", async () => {
    const res = await request(app).post("/v1/infer").set("Authorization", AUTH).send({});
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/prompt/);
  });

  it("returns 400 when prompt is not a string", async () => {
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: 42 });
    expect(res.status).toBe(400);
  });

  it("proxies to the core and merges gateway_latency_ms into the result", async () => {
    let capturedBody: any = null;
    stubFetch((url, init) => {
      capturedBody = JSON.parse(init.body);
      return jsonResponse(sampleCoreResponse());
    });
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "Summarize this", task_type: "generation", max_cost: 0.01 });

    expect(res.status).toBe(200);
    expect(res.body.engine_used).toBe("local-tier1");
    expect(res.body).toHaveProperty("gateway_latency_ms");
    // The gateway forwards routing fields and injects tracking metadata.
    expect(capturedBody.prompt).toBe("Summarize this");
    expect(capturedBody.task_type).toBe("generation");
    expect(capturedBody.metadata).toHaveProperty("gateway_request_id");
  });

  it("propagates a core error status and detail", async () => {
    stubFetch(() => new Response("boom", { status: 500 }));
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "hi" });
    expect(res.status).toBe(500);
    expect(res.body.error).toMatch(/core service/i);
    expect(res.body.detail).toBe("boom");
  });

  it("returns 502 when the core is unreachable", async () => {
    stubFetch(() => {
      throw new Error("network down");
    });
    const res = await request(app)
      .post("/v1/infer")
      .set("Authorization", AUTH)
      .send({ prompt: "hi" });
    expect(res.status).toBe(502);
    expect(res.body.error).toMatch(/core inference service/i);
  });
});

describe("POST /v1/infer/batch", () => {
  it("requires a non-empty prompts array (400)", async () => {
    const res = await request(app)
      .post("/v1/infer/batch")
      .set("Authorization", AUTH)
      .send({ prompts: [] });
    expect(res.status).toBe(400);
  });

  it("rejects more than 20 prompts (400)", async () => {
    const res = await request(app)
      .post("/v1/infer/batch")
      .set("Authorization", AUTH)
      .send({ prompts: new Array(21).fill("x") });
    expect(res.status).toBe(400);
  });

  it("fans out and returns one result per prompt", async () => {
    stubFetch(() => jsonResponse(sampleCoreResponse()));
    const res = await request(app)
      .post("/v1/infer/batch")
      .set("Authorization", AUTH)
      .send({ prompts: ["a", "b", "c"] });
    expect(res.status).toBe(200);
    expect(res.body.results).toHaveLength(3);
  });

  it("requires auth (401)", async () => {
    const res = await request(app).post("/v1/infer/batch").send({ prompts: ["a"] });
    expect(res.status).toBe(401);
  });
});

describe("GET /v1/stats", () => {
  it("requires auth (401)", async () => {
    const res = await request(app).get("/v1/stats");
    expect(res.status).toBe(401);
  });

  it("proxies core stats when authed", async () => {
    stubFetch(() => jsonResponse({ total_requests: 7 }));
    const res = await request(app).get("/v1/stats").set("Authorization", AUTH);
    expect(res.status).toBe(200);
    expect(res.body.total_requests).toBe(7);
  });
});

function sampleCoreResponse() {
  return {
    request_id: "core-1",
    content: "hello",
    engine_used: "local-tier1",
    tier: 1,
    confidence: 0.9,
    latency_ms: 12,
    cost_usd: 0.0001,
    success: true,
    routing_path: ["local-tier1"],
    escalation_reasons: [],
  };
}
