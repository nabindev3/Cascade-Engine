"""
Base engine interface for the multi-tier inference system.

Each engine represents one "tier" in the cascade — from cheap/fast local models
to expensive/capable cloud APIs. The router decides which engine to call based on
input complexity, cost budget, and reliability history.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
import time
from pydantic import BaseModel, Field


class EngineStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class FailureMode(Enum):
    """Categorized failure modes for Paper 1 analysis."""
    NONE = "none"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    PARSE_ERROR = "parse_error"          # Malformed output (bad JSON, etc.)
    SEMANTIC_FAILURE = "semantic_failure"  # Model didn't understand the task
    COLD_START = "cold_start"            # Engine not ready
    INFRASTRUCTURE = "infrastructure"     # 500, network error, etc.


class InferenceRequest(BaseModel):
    """A single inference request flowing through the cascade."""
    request_id: str
    prompt: str
    task_type: str = "general"           # e.g., "classification", "generation", "extraction"
    image_url: Optional[str] = None      # Multi-modal support (Tier 2/3 vision)
    max_tokens: int = 512
    temperature: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain_context: dict[str, Any] = Field(default_factory=dict) # Replaces specific industry terminology

    # Routing hints (optional — the router can override these)
    min_tier: Optional[int] = None       # Skip tiers below this
    max_cost: Optional[float] = None     # Budget ceiling for this request (cost SLO)
    latency_slo_ms: Optional[float] = None  # Latency SLO — skip tiers likely to breach it


class InferenceResponse(BaseModel):
    """Response from an engine, enriched with observability data."""
    request_id: str
    engine_id: str
    tier: int

    # Output
    content: str
    raw_output: Any = None

    # Confidence (engine's self-reported or calibrated)
    confidence: float = 0.0              # 0.0 to 1.0

    # Observability
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    token_count_input: int = 0
    token_count_output: int = 0

    # Failure tracking
    success: bool = True
    failure_mode: FailureMode = FailureMode.NONE
    error_message: Optional[str] = None
    was_escalated: bool = False

    # Timestamps
    timestamp: float = Field(default_factory=time.time)


class BaseEngine(ABC):
    """
    Abstract base for all inference engines.

    Each engine must declare its tier, cost model, and implement the `predict` method.
    The router uses `estimated_cost` and `health` to make routing decisions.
    """

    def __init__(self, engine_id: str, tier: int, config: dict[str, Any]) -> None:
        self.engine_id: str = engine_id
        self.tier: int = tier
        self.config: dict[str, Any] = config
        self._status: EngineStatus = EngineStatus.HEALTHY
        self._consecutive_failures: int = 0
        self._total_calls: int = 0
        self._total_failures: int = 0

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def reliability(self) -> float:
        """Empirical reliability = success_rate over all calls."""
        if self._total_calls == 0:
            return 1.0  # Optimistic prior
        return 1.0 - (self._total_failures / self._total_calls)

    @abstractmethod
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference. Must handle its own errors and return InferenceResponse."""
        ...

    @abstractmethod
    def estimated_cost(self, request: InferenceRequest) -> float:
        """Estimate cost in USD for this request (before execution)."""
        ...

    def estimated_latency_ms(self, percentile: float = 0.5) -> float:
        """Estimate response latency at a given percentile, for SLA gating.

        Reads ``latency_p50_ms`` / ``latency_p99_ms`` from config when present,
        otherwise falls back to a tier-based heuristic (cloud tiers are slower).
        Interpolates linearly between p50 and p99 over ``percentile ∈ [0.5, 0.99]``
        so a risk-averse router can budget against a tail latency rather than the
        median. Engines with real calibration data should override this.
        """
        p50: float = self.config.get("latency_p50_ms") or {
            1: 200.0, 2: 600.0, 3: 1200.0
        }.get(self.tier, 800.0)
        p99: float = self.config.get("latency_p99_ms") or (p50 * 5.0)
        if percentile <= 0.5:
            return float(p50)
        frac: float = min(1.0, (percentile - 0.5) / 0.49)
        return float(p50 + (p99 - p50) * frac)

    @abstractmethod
    async def health_check(self) -> EngineStatus:
        """Probe engine readiness. Updates internal status."""
        ...

    def record_success(self) -> None:
        """Called by router after successful inference."""
        self._total_calls += 1
        self._consecutive_failures = 0
        if self._status == EngineStatus.DEGRADED:
            self._status = EngineStatus.HEALTHY

    def record_failure(self, mode: FailureMode) -> None:
        """Called by router after failed inference."""
        self._total_calls += 1
        self._total_failures += 1
        self._consecutive_failures += 1

        # Circuit breaker: mark unavailable after 3 consecutive failures
        if self._consecutive_failures >= 3:
            self._status = EngineStatus.UNAVAILABLE
        elif self._consecutive_failures >= 1:
            self._status = EngineStatus.DEGRADED

    def reset_circuit(self) -> None:
        """Manually reset circuit breaker (e.g., after cooldown period)."""
        self._status = EngineStatus.HEALTHY
        self._consecutive_failures = 0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.engine_id} tier={self.tier} status={self._status.value}>"

