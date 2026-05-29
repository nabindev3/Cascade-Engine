"""
FastAPI service — exposes the cascade engine over HTTP.

This is the internal service that the TypeScript API gateway calls.
Runs on port 8000 by default.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engines.base import InferenceRequest
from .engines.local_engine import OllamaEngine
from .engines.cloud_engine import create_mid_tier_engine, create_premium_engine
from .router.cascade_router import CascadeRouter, RouterConfig
from .monitor.event_logger import EventLogger
from .monitor.sqlite_logger import instrument_inference
from .config.loader import load_config


# ─── Request / Response Models ─────────────────────────────────────────────────


class InferRequest(BaseModel):
    prompt: str
    task_type: str = "general"
    image_url: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    min_tier: int | None = None
    max_cost: float | None = None
    metadata: dict = Field(default_factory=dict)
    domain_context: dict = Field(default_factory=dict)


class InferResponse(BaseModel):
    request_id: str
    content: str
    engine_used: str
    tier: int
    confidence: float
    latency_ms: float
    cost_usd: float
    success: bool
    failure_mode: str | None = None
    was_escalated: bool = False
    routing_path: list[str] = []
    escalation_reasons: list[str] = []


# ─── App Lifecycle ─────────────────────────────────────────────────────────────


router_instance: CascadeRouter | None = None
logger_instance: EventLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance, logger_instance

    config = load_config()

    # Build engines from config
    engines = []

    if config.get("engines", {}).get("local", {}).get("enabled", True):
        engines.append(OllamaEngine(config=config.get("engines", {}).get("local", {})))

    if config.get("engines", {}).get("mid", {}).get("enabled", True):
        mid_cfg = config.get("engines", {}).get("mid", {})
        if mid_cfg.get("api_key"):
            engines.append(create_mid_tier_engine(mid_cfg))

    if config.get("engines", {}).get("premium", {}).get("enabled", True):
        premium_cfg = config.get("engines", {}).get("premium", {})
        if premium_cfg.get("api_key"):
            engines.append(create_premium_engine(premium_cfg))

    # Build router
    router_cfg = RouterConfig(**config.get("router", {}))
    router_instance = CascadeRouter(engines=engines, config=router_cfg)

    # Build logger
    logger_instance = EventLogger(
        output_dir=config.get("logging", {}).get("output_dir", "./data/logs")
    )

    yield

    # Shutdown
    if logger_instance:
        logger_instance.flush()


app = FastAPI(
    title="Cascade Inference Engine",
    version="0.1.0",
    description="Adaptive multi-tier LLM inference with cost-aware routing",
    lifespan=lifespan,
)


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/infer", response_model=InferResponse)
@instrument_inference
async def infer(req: InferRequest):
    """Run inference through the adaptive cascade."""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")

    request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=req.prompt,
        task_type=req.task_type,
        image_url=req.image_url,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        metadata=req.metadata,
        domain_context=req.domain_context,
        min_tier=req.min_tier,
        max_cost=req.max_cost,
    )

    response, decision = await router_instance.route(request)

    # Log the event
    if logger_instance:
        event = logger_instance.create_event(request, response, decision)
        logger_instance.log(event)

    return InferResponse(
        request_id=request.request_id,
        content=response.content,
        engine_used=response.engine_id,
        tier=response.tier,
        confidence=response.confidence,
        latency_ms=decision.total_latency_ms,
        cost_usd=decision.total_cost_usd,
        success=response.success,
        failure_mode=response.failure_mode.value if not response.success else None,
        was_escalated=response.was_escalated,
        routing_path=decision.engines_tried,
        escalation_reasons=decision.escalation_reasons,
    )


@app.get("/health")
async def health():
    """Health check for all engines."""
    if not router_instance:
        return {"status": "not_initialized"}
    statuses = await router_instance.health_check_all()
    return {"status": "ok", "engines": statuses}


@app.get("/stats")
async def stats():
    """Engine reliability stats and today's event summary."""
    result = {}
    if router_instance:
        result["engines"] = router_instance.get_engine_stats()
    if logger_instance:
        result["today"] = logger_instance.get_stats_summary()
    return result


@app.get("/stats/events")
async def event_stats():
    """Detailed event statistics for research analysis."""
    if not logger_instance:
        return {"error": "Logger not initialized"}
    return logger_instance.get_stats_summary()
