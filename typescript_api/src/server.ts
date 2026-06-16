/**
 * Cascade API Gateway
 *
 * Public-facing TypeScript API that:
 * - Authenticates requests (API key)
 * - Rate limits per client
 * - Proxies to the Python core service
 * - Adds request tracking and structured logging
 *
 * Runs on port 3000, expects Python core on port 8000.
 */

import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { v4 as uuidv4 } from "uuid";
import pino from "pino";
import { CircuitBreaker } from "./circuitBreaker.js";

// The pino-pretty transport runs in a worker thread; skip it under tests so the
// test runner can tear down cleanly (and to keep test output quiet).
const logger =
  process.env.NODE_ENV === "test"
    ? pino({ level: "silent" })
    : pino({ transport: { target: "pino-pretty", options: { colorize: true } } });

const app = express();
const PORT = parseInt(process.env.GATEWAY_PORT || "3000");
const CORE_URL = process.env.CORE_SERVICE_URL || "http://localhost:8000";
const API_KEYS = new Set((process.env.API_KEYS || "dev-key-123").split(","));

// ─── Circuit breaker + direct cloud fallback ────────────────────────────────
//
// When the Python core is unhealthy (network error, 5xx, or timeout) the
// breaker opens and the gateway stops hammering it. While the core is down,
// requests are served by a direct call to a cloud LLM (degraded mode) if a
// fallback key is configured; otherwise the gateway fails fast.
const CORE_CIRCUIT_FAILURE_THRESHOLD = parseInt(
  process.env.CORE_CIRCUIT_FAILURE_THRESHOLD || "5"
);
const CORE_CIRCUIT_COOLDOWN_MS = parseInt(
  process.env.CORE_CIRCUIT_COOLDOWN_MS || "30000"
);
const FALLBACK_API_KEY = process.env.FALLBACK_OPENAI_API_KEY || "";
const FALLBACK_BASE_URL =
  process.env.FALLBACK_OPENAI_BASE_URL || "https://api.openai.com/v1";
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || "gpt-4o-mini";

const coreBreaker = new CircuitBreaker({
  failureThreshold: CORE_CIRCUIT_FAILURE_THRESHOLD,
  cooldownMs: CORE_CIRCUIT_COOLDOWN_MS,
});

// ─── Middleware ───────────────────────────────────────────────────────────────

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// Rate limiting: 100 requests per minute per IP (overridable via env so the
// integration tests can exercise the limiter without firing hundreds of calls).
const RATE_LIMIT_WINDOW_MS = parseInt(process.env.RATE_LIMIT_WINDOW_MS || "60000");
const RATE_LIMIT_MAX = parseInt(process.env.RATE_LIMIT_MAX || "100");
const limiter = rateLimit({
  windowMs: RATE_LIMIT_WINDOW_MS,
  max: RATE_LIMIT_MAX,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Rate limit exceeded", retry_after_ms: RATE_LIMIT_WINDOW_MS },
});
app.use("/v1/", limiter);

// Auth middleware
function authenticate(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
): void {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({ error: "Missing or invalid Authorization header" });
    return;
  }
  const key = authHeader.slice(7);
  if (!API_KEYS.has(key)) {
    res.status(403).json({ error: "Invalid API key" });
    return;
  }
  next();
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface InferRequest {
  prompt: string;
  task_type?: string;
  max_tokens?: number;
  temperature?: number;
  min_tier?: number;
  max_cost?: number;
  metadata?: Record<string, unknown>;
}

interface InferResponse {
  request_id: string;
  content: string;
  engine_used: string;
  tier: number;
  confidence: number;
  latency_ms: number;
  cost_usd: number;
  success: boolean;
  failure_mode?: string;
  routing_path: string[];
  escalation_reasons: string[];
  /** Set when the response came from the gateway's direct cloud fallback. */
  degraded?: boolean;
}

/** Error thrown when the breaker is open and the call is short-circuited. */
class CircuitOpenError extends Error {
  constructor() {
    super("core circuit open");
  }
}

/** Error carrying a propagatable HTTP status from the core. */
class CoreHttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
    /** true ⇒ core is unhealthy (5xx), should trip breaker + try fallback. */
    public readonly unhealthy: boolean
  ) {
    super(`core HTTP ${status}`);
  }
}

// ─── Core call + fallback ───────────────────────────────────────────────────

/** Call the Python core, updating the circuit breaker. Throws on failure. */
async function callCoreInfer(payload: object): Promise<InferResponse> {
  if (!coreBreaker.allowRequest()) {
    throw new CircuitOpenError();
  }

  let resp: Response;
  try {
    resp = await fetch(`${CORE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    coreBreaker.recordFailure();
    throw err;
  }

  if (resp.status >= 500) {
    coreBreaker.recordFailure();
    throw new CoreHttpError(resp.status, await resp.text(), true);
  }
  if (!resp.ok) {
    // 4xx is a client problem, not core ill-health — don't trip the breaker.
    throw new CoreHttpError(resp.status, await resp.text(), false);
  }

  coreBreaker.recordSuccess();
  return (await resp.json()) as InferResponse;
}

/**
 * Degraded-mode path: call a cloud LLM directly when the core is unavailable.
 * Returns null when no fallback is configured or the fallback call fails.
 */
async function directCloudFallback(
  body: InferRequest,
  requestId: string
): Promise<InferResponse | null> {
  if (!FALLBACK_API_KEY) return null;
  try {
    const resp = await fetch(`${FALLBACK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${FALLBACK_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: FALLBACK_MODEL,
        messages: [{ role: "user", content: body.prompt }],
        max_tokens: body.max_tokens ?? 512,
        temperature: body.temperature ?? 0.0,
      }),
    });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const content = data?.choices?.[0]?.message?.content ?? "";
    return {
      request_id: requestId,
      content,
      engine_used: `gateway-fallback:${FALLBACK_MODEL}`,
      tier: 0,
      confidence: 0,
      latency_ms: 0,
      cost_usd: 0,
      success: true,
      routing_path: ["gateway-fallback"],
      escalation_reasons: ["core unavailable — direct cloud fallback"],
      degraded: true,
    };
  } catch {
    return null;
  }
}

type DispatchOutcome =
  | { ok: true; result: InferResponse }
  | { ok: false; status: number; payload: Record<string, unknown> };

/**
 * Run one inference: try the core, then (on core ill-health) the cloud
 * fallback, preserving legacy status propagation when no fallback exists.
 * Shared by the single and batch endpoints.
 */
async function dispatchInference(
  body: InferRequest,
  requestId: string,
  clientIp?: string
): Promise<DispatchOutcome> {
  const payload = {
    prompt: body.prompt,
    task_type: body.task_type || "general",
    max_tokens: body.max_tokens || 512,
    temperature: body.temperature || 0.0,
    min_tier: body.min_tier,
    max_cost: body.max_cost,
    metadata: {
      ...body.metadata,
      gateway_request_id: requestId,
      client_ip: clientIp,
    },
  };

  try {
    return { ok: true, result: await callCoreInfer(payload) };
  } catch (err) {
    // 4xx from the core: propagate verbatim, never fall back.
    if (err instanceof CoreHttpError && !err.unhealthy) {
      return {
        ok: false,
        status: err.status,
        payload: { error: "Core service error", detail: err.body, request_id: requestId },
      };
    }

    // Core is unhealthy (5xx / network / circuit open): try degraded fallback.
    const fb = await directCloudFallback(body, requestId);
    if (fb) {
      logger.warn({ msg: "Served via cloud fallback", request_id: requestId });
      return { ok: true, result: fb };
    }

    // No fallback available — surface the most accurate error.
    if (err instanceof CircuitOpenError) {
      return {
        ok: false,
        status: 503,
        payload: {
          error: "Core service unavailable (circuit open)",
          request_id: requestId,
        },
      };
    }
    if (err instanceof CoreHttpError) {
      return {
        ok: false,
        status: err.status,
        payload: { error: "Core service error", detail: err.body, request_id: requestId },
      };
    }
    return {
      ok: false,
      status: 502,
      payload: {
        error: "Failed to reach core inference service",
        request_id: requestId,
      },
    };
  }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health check (no auth required)
app.get("/health", async (_req, res) => {
  try {
    const coreResp = await fetch(`${CORE_URL}/health`);
    const coreHealth = await coreResp.json();
    res.json({
      gateway: "ok",
      core: coreHealth,
      circuit: coreBreaker.snapshot(),
      fallback_configured: Boolean(FALLBACK_API_KEY),
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    res.status(503).json({
      gateway: "ok",
      core: "unreachable",
      circuit: coreBreaker.snapshot(),
      fallback_configured: Boolean(FALLBACK_API_KEY),
      timestamp: new Date().toISOString(),
    });
  }
});

// Main inference endpoint
app.post("/v1/infer", authenticate, async (req, res) => {
  const requestId = uuidv4();
  const startTime = Date.now();

  const body: InferRequest = req.body;

  if (!body.prompt || typeof body.prompt !== "string") {
    res.status(400).json({ error: "Missing required field: prompt" });
    return;
  }

  logger.info({
    msg: "Inference request",
    request_id: requestId,
    task_type: body.task_type || "general",
    prompt_length: body.prompt.length,
  });

  const outcome = await dispatchInference(body, requestId, req.ip);
  const gatewayLatency = Date.now() - startTime;

  if (outcome.ok) {
    logger.info({
      msg: "Inference complete",
      request_id: requestId,
      engine: outcome.result.engine_used,
      tier: outcome.result.tier,
      latency_ms: gatewayLatency,
      degraded: outcome.result.degraded === true,
      success: outcome.result.success,
    });
    res.json({ ...outcome.result, gateway_latency_ms: gatewayLatency });
    return;
  }

  logger.error({ msg: "Request failed", request_id: requestId, status: outcome.status });
  res.status(outcome.status).json({ ...outcome.payload, gateway_latency_ms: gatewayLatency });
});

// Stats endpoint
app.get("/v1/stats", authenticate, async (_req, res) => {
  try {
    const coreResp = await fetch(`${CORE_URL}/stats`);
    const stats = await coreResp.json();
    res.json(stats);
  } catch (err) {
    res.status(502).json({ error: "Cannot reach core service" });
  }
});

// Batch inference (fire multiple prompts)
app.post("/v1/infer/batch", authenticate, async (req, res) => {
  const { prompts, ...shared } = req.body as { prompts: string[] } & Omit<InferRequest, "prompt">;

  if (!Array.isArray(prompts) || prompts.length === 0) {
    res.status(400).json({ error: "prompts must be a non-empty array" });
    return;
  }

  if (prompts.length > 20) {
    res.status(400).json({ error: "Max 20 prompts per batch" });
    return;
  }

  const results = await Promise.allSettled(
    prompts.map((prompt) =>
      dispatchInference({ ...shared, prompt } as InferRequest, uuidv4(), req.ip)
    )
  );

  res.json({
    results: results.map((r) => {
      if (r.status !== "fulfilled") return { error: String(r.reason) };
      return r.value.ok ? r.value.result : r.value.payload;
    }),
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────

// Don't bind a port when imported by the test runner (supertest drives the
// app object directly); only listen when started as a real process.
if (process.env.NODE_ENV !== "test") {
  app.listen(PORT, () => {
    logger.info(`🚀 Cascade API Gateway running on port ${PORT}`);
    logger.info(`   Core service: ${CORE_URL}`);
    logger.info(`   Rate limit: ${RATE_LIMIT_MAX} req / ${RATE_LIMIT_WINDOW_MS}ms`);
  });
}

export default app;
export { coreBreaker };
