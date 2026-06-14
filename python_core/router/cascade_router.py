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
from typing import Any, Optional, Tuple

from pydantic import BaseModel, Field

from ..engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)


class RoutingDecision(BaseModel):
    """Records the router's decision path for a single request — key for Paper 1 data."""
    request_id: str
    tiers_attempted: list[int] = Field(default_factory=list)
    engines_tried: list[str] = Field(default_factory=list)
    escalation_reasons: list[str] = Field(default_factory=list)
    final_engine: Optional[str] = None
    final_tier: Optional[int] = None
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    success: bool = False


class RouterConfig(BaseModel):
    """
    Configuration for the cascade router.

    confidence_thresholds: dict mapping tier -> minimum confidence to accept response.
        e.g., {1: 0.7, 2: 0.8} means Tier 1 needs 0.7 confidence, Tier 2 needs 0.8.
        Tier 3 (max) always accepts (no escalation possible).

    max_cost_per_request: float — hard budget ceiling per request in USD.
    reliability_ema_alpha: float — smoothing factor for reliability EMA (0 < α ≤ 1).
    min_reliability_to_attempt: float — skip engine if reliability below this.
    enable_parallel_fallback: bool — if True, fire next tier in parallel on low confidence.
    enable_local_fallback: bool — if True, when every tier at/above the request's
        min_tier fails, retry the cheaper tiers that min_tier originally skipped.
        This is the "downgrade to local" safety net: a degraded answer from a local
        model beats a hard failure when the cloud tiers are down/rate-limited.
    """
    confidence_thresholds: dict[int, float] = Field(default_factory=lambda: {1: 0.65, 2: 0.80})
    max_cost_per_request: float = 0.05  # $0.05 default ceiling
    reliability_ema_alpha: float = 0.1
    min_reliability_to_attempt: float = 0.3
    enable_parallel_fallback: bool = False
    enable_local_fallback: bool = True
    timeout_per_tier_ms: float = 30000.0


class CascadeRouter:
    """
    The adaptive cascade router.

    Given a set of engines sorted by tier, routes each request through the cascade
    until a sufficiently confident response is obtained or all tiers are exhausted.
    """

    def __init__(self, engines: list[BaseEngine], config: Optional[RouterConfig] = None) -> None:
        self.config: RouterConfig = config or RouterConfig()
        # Sort engines by tier (ascending = cheapest first)
        self.engines: list[BaseEngine] = sorted(engines, key=lambda e: e.tier)
        # Reliability EMA per engine
        self._reliability_ema: dict[str, float] = {
            e.engine_id: 1.0 for e in self.engines
        }

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        """
        Route a request through the cascade.

        Returns:
            (best_response, routing_decision) — even on total failure,
            returns the last response attempted.
        """
        decision: RoutingDecision = RoutingDecision(request_id=request.request_id)
        start: float = time.perf_counter()

        best_response: Optional[InferenceResponse] = None
        cumulative_cost: float = 0.0
        is_escalated: bool = False
        # Engines skipped *only* by the min_tier gate — candidates for the
        # local-fallback safety net if every higher tier fails.
        deferred_low_tier: list[BaseEngine] = []

        for engine in self.engines:
            # --- Gate 1: Minimum tier constraint ---
            if request.min_tier and engine.tier < request.min_tier:
                deferred_low_tier.append(engine)
                continue

            # --- Gates 2-4: health, reliability, cost budget ---
            if not self._engine_is_eligible(engine, request, cumulative_cost, decision):
                continue

            # --- Attempt inference ---
            decision.tiers_attempted.append(engine.tier)
            decision.engines_tried.append(engine.engine_id)

            response: InferenceResponse = await engine.predict(request)
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
            threshold: float = self.config.confidence_thresholds.get(engine.tier, 0.0)
            
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

        # --- Local-fallback safety net ---
        # Every tier at/above min_tier failed (or none was eligible). Downgrade
        # to the cheaper tiers that min_tier skipped rather than hard-failing.
        if (
            not decision.success
            and self.config.enable_local_fallback
            and deferred_low_tier
        ):
            for engine in deferred_low_tier:  # already sorted cheapest-first
                if not self._engine_is_eligible(engine, request, cumulative_cost, decision):
                    continue

                decision.tiers_attempted.append(engine.tier)
                decision.engines_tried.append(engine.engine_id)
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: local fallback after higher tiers failed"
                )

                response = await engine.predict(request)
                cumulative_cost += response.cost_usd
                self._update_reliability(engine.engine_id, response.success)
                response.was_escalated = True

                if response.success:
                    # Accept any success here — this is a degraded-mode last resort,
                    # so confidence thresholds are intentionally not applied.
                    decision.final_engine = engine.engine_id
                    decision.final_tier = engine.tier
                    decision.success = True
                    decision.total_latency_ms = (time.perf_counter() - start) * 1000
                    decision.total_cost_usd = cumulative_cost
                    return response, decision

                decision.escalation_reasons.append(
                    f"{engine.engine_id}: fallback failed ({response.failure_mode.value})"
                )
                best_response = response

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

    def _engine_is_eligible(
        self,
        engine: BaseEngine,
        request: InferenceRequest,
        cumulative_cost: float,
        decision: RoutingDecision,
    ) -> bool:
        """Health, reliability, and cost-budget gates shared by the main cascade
        and the local-fallback path. Appends a reason to ``decision`` when an
        engine is gated out, and returns False; returns True if the engine may run.
        """
        # --- Engine health ---
        if engine.status == EngineStatus.UNAVAILABLE:
            decision.escalation_reasons.append(
                f"{engine.engine_id}: unavailable (circuit open)"
            )
            return False

        # --- Reliability threshold ---
        ema: float = self._reliability_ema.get(engine.engine_id, 1.0)
        if ema < self.config.min_reliability_to_attempt:
            decision.escalation_reasons.append(
                f"{engine.engine_id}: reliability too low ({ema:.2f})"
            )
            return False

        # --- Cost budget ---
        estimated: float = engine.estimated_cost(request)
        if request.max_cost and (cumulative_cost + estimated) > request.max_cost:
            decision.escalation_reasons.append(
                f"{engine.engine_id}: would exceed request budget"
            )
            return False
        if (cumulative_cost + estimated) > self.config.max_cost_per_request:
            decision.escalation_reasons.append(
                f"{engine.engine_id}: would exceed global budget"
            )
            return False

        return True

    def _update_reliability(self, engine_id: str, success: bool) -> None:
        """Exponential moving average of success rate."""
        alpha: float = self.config.reliability_ema_alpha
        current: float = self._reliability_ema.get(engine_id, 1.0)
        observation: float = 1.0 if success else 0.0
        self._reliability_ema[engine_id] = alpha * observation + (1 - alpha) * current

    def get_engine_stats(self) -> dict[str, dict[str, Any]]:
        """Return structured engine performance statistics."""
        return {
            e.engine_id: {
                "tier": e.tier,
                "status": e.status.value,
                "reliability": e.reliability,
                "reliability_ema": self._reliability_ema.get(e.engine_id, 1.0),
            }
            for e in self.engines
        }

    async def health_check_all(self) -> dict[str, str]:
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
