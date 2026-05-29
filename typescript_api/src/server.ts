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
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    res.status(503).json({
      gateway: "ok",
      core: "unreachable",
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

  try {
    const coreResp = await fetch(`${CORE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: body.prompt,
        task_type: body.task_type || "general",
        max_tokens: body.max_tokens || 512,
        temperature: body.temperature || 0.0,
        min_tier: body.min_tier,
        max_cost: body.max_cost,
        metadata: {
          ...body.metadata,
          gateway_request_id: requestId,
          client_ip: req.ip,
        },
      }),
    });

    if (!coreResp.ok) {
      const errText = await coreResp.text();
      logger.error({ msg: "Core service error", status: coreResp.status, body: errText });
      res.status(coreResp.status).json({
        error: "Core service error",
        detail: errText,
        request_id: requestId,
      });
      return;
    }

    const result: InferResponse = await coreResp.json() as InferResponse;
    const gatewayLatency = Date.now() - startTime;

    logger.info({
      msg: "Inference complete",
      request_id: requestId,
      engine: result.engine_used,
      tier: result.tier,
      confidence: result.confidence,
      latency_ms: gatewayLatency,
      cost_usd: result.cost_usd,
      success: result.success,
    });

    res.json({
      ...result,
      gateway_latency_ms: gatewayLatency,
    });
  } catch (err) {
    const gatewayLatency = Date.now() - startTime;
    logger.error({ msg: "Request failed", request_id: requestId, error: String(err) });
    res.status(502).json({
      error: "Failed to reach core inference service",
      request_id: requestId,
      gateway_latency_ms: gatewayLatency,
    });
  }
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
    prompts.map(async (prompt) => {
      const resp = await fetch(`${CORE_URL}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, ...shared }),
      });
      return resp.json();
    })
  );

  res.json({
    results: results.map((r) =>
      r.status === "fulfilled" ? r.value : { error: String(r.reason) }
    ),
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
