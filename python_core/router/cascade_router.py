"""
Adaptive Cascade Router — the core algorithmic contribution.

Implements a confidence-gated cascading policy:
1. Start at Tier 1 (cheapest).
2. If confidence < threshold OR engine fails → escalate to next tier.
3. Respect cost budget constraints.
4. Track reliability per-engine with exponential moving average.

Research extensions (Paper 2):
- Replace static thresholds with a learned routing policy (MAB / MDP).
- Model stochastic reliability (engines whose failure rate varies with time-of-day, load).
- Optimize for Pareto frontier: cost vs. accuracy vs. latency.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from ..engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)


@dataclass
class RoutingDecision:
    """Records the router's decision path for a single request — key for Paper 1 data."""
    request_id: str
    tiers_attempted: List[int] = field(default_factory=list)
    engines_tried: List[str] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)
    final_engine: Optional[str] = None
    final_tier: Optional[int] = None
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    success: bool = False


@dataclass
class RouterConfig:
    """
    Configuration for the cascade router.

    confidence_thresholds: dict mapping tier -> minimum confidence to accept response.
        e.g., {1: 0.7, 2: 0.8} means Tier 1 needs 0.7 confidence, Tier 2 needs 0.8.
        Tier 3 (max) always accepts (no escalation possible).

    max_cost_per_request: float — hard budget ceiling per request in USD.
    reliability_ema_alpha: float — smoothing factor for reliability EMA (0 < α ≤ 1).
    min_reliability_to_attempt: float — skip engine if reliability below this.
    enable_parallel_fallback: bool — if True, fire next tier in parallel on low confidence.
    """
    confidence_thresholds: dict = field(default_factory=lambda: {1: 0.65, 2: 0.80})
    max_cost_per_request: float = 0.05  # $0.05 default ceiling
    reliability_ema_alpha: float = 0.1
    min_reliability_to_attempt: float = 0.3
    enable_parallel_fallback: bool = False
    timeout_per_tier_ms: float = 30000.0


class CascadeRouter:
    """
    The adaptive cascade router.

    Given a set of engines sorted by tier, routes each request through the cascade
    until a sufficiently confident response is obtained or all tiers are exhausted.
    """

    def __init__(self, engines: List[BaseEngine], config: RouterConfig = None):
        self.config = config or RouterConfig()
        # Sort engines by tier (ascending = cheapest first)
        self.engines = sorted(engines, key=lambda e: e.tier)
        # Reliability EMA per engine
        self._reliability_ema: dict[str, float] = {
            e.engine_id: 1.0 for e in self.engines
        }

    async def route(self, request: InferenceRequest) -> tuple[InferenceResponse, RoutingDecision]:
        """
        Route a request through the cascade.

        Returns:
            (best_response, routing_decision) — even on total failure,
            returns the last response attempted.
        """
        decision = RoutingDecision(request_id=request.request_id)
        start = time.perf_counter()

        best_response: Optional[InferenceResponse] = None
        cumulative_cost = 0.0
        is_escalated = False

        for engine in self.engines:
            # --- Gate 1: Minimum tier constraint ---
            if request.min_tier and engine.tier < request.min_tier:
                continue

            # --- Gate 2: Engine health ---
            if engine.status == EngineStatus.UNAVAILABLE:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: unavailable (circuit open)"
                )
                continue

            # --- Gate 3: Reliability threshold ---
            ema = self._reliability_ema.get(engine.engine_id, 1.0)
            if ema < self.config.min_reliability_to_attempt:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: reliability too low ({ema:.2f})"
                )
                continue

            # --- Gate 4: Cost budget ---
            estimated = engine.estimated_cost(request)
            if request.max_cost and (cumulative_cost + estimated) > request.max_cost:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: would exceed request budget"
                )
                continue
            if (cumulative_cost + estimated) > self.config.max_cost_per_request:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: would exceed global budget"
                )
                continue

            # --- Attempt inference ---
            decision.tiers_attempted.append(engine.tier)
            decision.engines_tried.append(engine.engine_id)

            response = await engine.infer(request)
            cumulative_cost += response.cost_usd

            # Update reliability EMA
            self._update_reliability(engine.engine_id, response.success)
            response.was_escalated = is_escalated

            # --- Evaluate response ---
            if not response.success:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: failed ({response.failure_mode.value})"
                )
                best_response = response
                is_escalated = True
                continue

            # Check confidence against tier threshold
            threshold = self.config.confidence_thresholds.get(engine.tier, 0.0)
            
            # Critic Model Mock (Confidence Scoring)
            if response.confidence < threshold and response.confidence > 0.1:
                # Mocking a critic review that sometimes boosts or penalizes confidence
                response.confidence = response.confidence * 0.9

            if response.confidence >= threshold or engine.tier == max(e.tier for e in self.engines):
                # Accept this response
                decision.final_engine = engine.engine_id
                decision.final_tier = engine.tier
                decision.success = True
                decision.total_latency_ms = (time.perf_counter() - start) * 1000
                decision.total_cost_usd = cumulative_cost
                return response, decision
            else:
                # Confidence too low — escalate
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: confidence {response.confidence:.2f} < threshold {threshold:.2f}"
                )
                best_response = response
                is_escalated = True
                continue

        # All tiers exhausted
        decision.total_latency_ms = (time.perf_counter() - start) * 1000
        decision.total_cost_usd = cumulative_cost

        if best_response is None:
            # No engine was even attempted
            best_response = InferenceResponse(
                request_id=request.request_id,
                engine_id="none",
                tier=0,
                content="",
                success=False,
                failure_mode=FailureMode.INFRASTRUCTURE,
                error_message="No available engine in cascade",
            )

        return best_response, decision

    def _update_reliability(self, engine_id: str, success: bool):
        """Exponential moving average of success rate."""
        alpha = self.config.reliability_ema_alpha
        current = self._reliability_ema.get(engine_id, 1.0)
        observation = 1.0 if success else 0.0
        self._reliability_ema[engine_id] = alpha * observation + (1 - alpha) * current

    def get_engine_stats(self) -> dict:
        """Return current engine status and reliability for monitoring."""
        return {
            engine.engine_id: {
                "tier": engine.tier,
                "status": engine.status.value,
                "reliability_ema": round(self._reliability_ema.get(engine.engine_id, 0), 4),
                "empirical_reliability": round(engine.reliability, 4),
                "total_calls": engine._total_calls,
                "consecutive_failures": engine._consecutive_failures,
            }
            for engine in self.engines
        }

    async def health_check_all(self) -> dict:
        """Probe all engines and return their status."""
        results = await asyncio.gather(
            *[engine.health_check() for engine in self.engines],
            return_exceptions=True,
        )
        return {
            engine.engine_id: (
                result.value if isinstance(result, EngineStatus) else "error"
            )
            for engine, result in zip(self.engines, results)
        }
