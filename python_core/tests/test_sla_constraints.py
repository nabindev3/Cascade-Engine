"""Tests for Step 5 — risk-sensitive SLA-constrained routing.

The router treats `latency_slo_ms` as a CMDP-style constraint: it skips tiers
whose risk-adjusted estimated latency would breach the remaining budget, and
relaxes to best-effort (flagging `sla_violated`) when no tier can satisfy it.
"""

from python_core.engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)
from python_core.router.cascade_router import CascadeRouter, RouterConfig


class LatencyEngine(BaseEngine):
    """Engine with controllable estimated (p50/p99) and actual latency."""

    def __init__(
        self,
        engine_id: str,
        tier: int,
        *,
        p50: float,
        p99: float,
        actual_ms: float = 10.0,
        confidence: float = 0.95,
        succeed: bool = True,
        cost: float = 0.0001,
    ) -> None:
        super().__init__(engine_id=engine_id, tier=tier, config={})
        self._p50 = p50
        self._p99 = p99
        self._actual = actual_ms
        self._confidence = confidence
        self._succeed = succeed
        self._cost = cost
        self.calls = 0

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        return InferenceResponse(
            request_id=request.request_id, engine_id=self.engine_id, tier=self.tier,
            content="ok" if self._succeed else "",
            confidence=self._confidence if self._succeed else 0.0,
            cost_usd=self._cost, latency_ms=self._actual, success=self._succeed,
            failure_mode=FailureMode.NONE if self._succeed else FailureMode.TIMEOUT,
        )

    def estimated_cost(self, request: InferenceRequest) -> float:
        return self._cost

    def estimated_latency_ms(self, percentile: float = 0.5) -> float:
        if percentile <= 0.5:
            return self._p50
        frac = min(1.0, (percentile - 0.5) / 0.49)
        return self._p50 + (self._p99 - self._p50) * frac

    async def health_check(self) -> EngineStatus:
        return EngineStatus.HEALTHY


async def test_sla_feasible_routes_within_budget():
    fast = LatencyEngine("fast-1", tier=1, p50=100, p99=200, confidence=0.9)
    router = CascadeRouter([fast], RouterConfig(confidence_thresholds={1: 0.65}))

    req = InferenceRequest(request_id="s1", prompt="q", latency_slo_ms=500)
    resp, decision = await router.route(req)

    assert resp.success
    assert decision.sla_latency_slo_ms == 500
    assert decision.sla_violated is False


async def test_sla_infeasible_serves_best_effort_and_flags():
    slow1 = LatencyEngine("slow-1", tier=1, p50=2000, p99=4000, confidence=0.9)
    slow2 = LatencyEngine("slow-2", tier=2, p50=3000, p99=6000, confidence=0.9)
    router = CascadeRouter([slow1, slow2], RouterConfig())

    req = InferenceRequest(request_id="s2", prompt="q", latency_slo_ms=500)
    resp, decision = await router.route(req)

    assert resp.success, "infeasible SLO must still answer best-effort"
    assert decision.sla_violated is True
    assert any("infeasible" in r.lower() for r in decision.escalation_reasons)
    assert slow1.calls == 1, "the constraint should be relaxed, letting tier 1 run"


async def test_latency_gate_skips_slow_next_tier():
    # tier-1 low confidence forces escalation; the cumulative latency budget
    # (300ms used + 300ms est) busts the 500ms SLO, so tier 2 is gated out.
    t1 = LatencyEngine("t1", tier=1, p50=300, p99=300, actual_ms=300, confidence=0.2)
    t2 = LatencyEngine("t2", tier=2, p50=300, p99=300, actual_ms=300, confidence=0.95)
    router = CascadeRouter(
        [t1, t2],
        RouterConfig(confidence_thresholds={1: 0.65, 2: 0.80}, sla_risk_aversion=0.0),
    )

    req = InferenceRequest(request_id="s3", prompt="q", latency_slo_ms=500)
    _, decision = await router.route(req)

    assert "t2" not in decision.engines_tried
    assert t2.calls == 0
    assert any("breach SLO" in r for r in decision.escalation_reasons)


async def test_risk_aversion_uses_tail_latency():
    # p50=400 fits the 500ms SLO; p99=900 busts it.
    low = LatencyEngine("e", tier=1, p50=400, p99=900, confidence=0.9)
    r_low = CascadeRouter([low], RouterConfig(sla_risk_aversion=0.0))
    _, d_low = await r_low.route(
        InferenceRequest(request_id="s4a", prompt="q", latency_slo_ms=500)
    )
    assert d_low.final_engine == "e"  # median fits → admitted
    assert d_low.sla_violated is False

    high = LatencyEngine("e2", tier=1, p50=400, p99=900, confidence=0.9)
    r_high = CascadeRouter([high], RouterConfig(sla_risk_aversion=1.0))
    _, d_high = await r_high.route(
        InferenceRequest(request_id="s4b", prompt="q", latency_slo_ms=500)
    )
    # Tail (~p99) busts the SLO → infeasible → best-effort, flagged.
    assert d_high.sla_violated is True


async def test_no_slo_leaves_sla_fields_untouched():
    eng = LatencyEngine("e", tier=1, p50=9999, p99=99999, confidence=0.9)
    router = CascadeRouter([eng], RouterConfig())

    _, decision = await router.route(InferenceRequest(request_id="s6", prompt="q"))

    assert decision.sla_latency_slo_ms is None
    assert decision.sla_violated is False
