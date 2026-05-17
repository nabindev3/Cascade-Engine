"""
Base engine interface for the multi-tier inference system.

Each engine represents one "tier" in the cascade — from cheap/fast local models
to expensive/capable cloud APIs. The router decides which engine to call based on
input complexity, cost budget, and reliability history.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


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


@dataclass
class InferenceRequest:
    """A single inference request flowing through the cascade."""
    request_id: str
    prompt: str
    task_type: str = "general"           # e.g., "classification", "generation", "extraction"
    image_url: Optional[str] = None      # Multi-modal support (Tier 2/3 vision)
    max_tokens: int = 512
    temperature: float = 0.0
    metadata: dict = field(default_factory=dict)
    domain_context: dict = field(default_factory=dict) # Replaces specific industry terminology

    # Routing hints (optional — the router can override these)
    min_tier: Optional[int] = None       # Skip tiers below this
    max_cost: Optional[float] = None     # Budget ceiling for this request


@dataclass
class InferenceResponse:
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
    timestamp: float = field(default_factory=time.time)


class BaseEngine(ABC):
    """
    Abstract base for all inference engines.

    Each engine must declare its tier, cost model, and implement the `infer` method.
    The router uses `estimated_cost` and `health` to make routing decisions.
    """

    def __init__(self, engine_id: str, tier: int, config: dict):
        self.engine_id = engine_id
        self.tier = tier
        self.config = config
        self._status = EngineStatus.HEALTHY
        self._consecutive_failures = 0
        self._total_calls = 0
        self._total_failures = 0

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
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference. Must handle its own errors and return InferenceResponse."""
        ...

    @abstractmethod
    def estimated_cost(self, request: InferenceRequest) -> float:
        """Estimate cost in USD for this request (before execution)."""
        ...

    @abstractmethod
    async def health_check(self) -> EngineStatus:
        """Probe engine readiness. Updates internal status."""
        ...

    def record_success(self):
        """Called by router after successful inference."""
        self._total_calls += 1
        self._consecutive_failures = 0
        if self._status == EngineStatus.DEGRADED:
            self._status = EngineStatus.HEALTHY

    def record_failure(self, mode: FailureMode):
        """Called by router after failed inference."""
        self._total_calls += 1
        self._total_failures += 1
        self._consecutive_failures += 1

        # Circuit breaker: mark unavailable after 3 consecutive failures
        if self._consecutive_failures >= 3:
            self._status = EngineStatus.UNAVAILABLE
        elif self._consecutive_failures >= 1:
            self._status = EngineStatus.DEGRADED

    def reset_circuit(self):
        """Manually reset circuit breaker (e.g., after cooldown period)."""
        self._status = EngineStatus.HEALTHY
        self._consecutive_failures = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.engine_id} tier={self.tier} status={self._status.value}>"
