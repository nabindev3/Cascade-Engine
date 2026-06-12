"""Calibrated simulation engine — for FRAMEWORK VALIDATION ONLY.

⚠️  READ THIS BEFORE USING IN A PAPER  ⚠️

This engine does NOT call any model. It exists so the pipeline (router →
benchmark → Pareto → figures) can be exercised end-to-end and so figures are
*reproducible* without API spend.
"""

import asyncio
import random
import time
from typing import Any, Optional, Tuple, List

from .base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)

SIM_PREFIX: str = "SIM:"


class CalibratedSimulatedEngine(BaseEngine):
    """A tier whose cost is real, latency is calibrated, quality is simulated."""

    def __init__(
        self,
        engine_id: str,
        tier: int,
        cost_per_input_token: float,
        cost_per_output_token: float,
        latency_p50_ms: float,
        latency_p99_ms: float,
        competence: float,
        failure_rate: float = 0.01,
        avg_output_tokens: int = 256,
    ) -> None:
        # Force the SIM: prefix so provenance can never be lost downstream.
        sim_id: str = engine_id if engine_id.startswith(SIM_PREFIX) else f"{SIM_PREFIX}{engine_id}"
        super().__init__(engine_id=sim_id, tier=tier, config={})
        self.cost_per_input_token: float = cost_per_input_token
        self.cost_per_output_token: float = cost_per_output_token
        self.latency_p50_ms: float = latency_p50_ms
        self.latency_p99_ms: float = latency_p99_ms
        self.competence: float = max(0.0, min(1.0, competence))
        self.failure_rate: float = failure_rate
        self.avg_output_tokens: int = avg_output_tokens

    # ----- cost: REAL (uses published per-token prices) -----
    def estimated_cost(self, request: InferenceRequest) -> float:
        n_in: int = max(1, len(request.prompt.split()))
        n_out: int = self.avg_output_tokens
        return n_in * self.cost_per_input_token + n_out * self.cost_per_output_token

    # ----- latency: calibrated log-normal -----
    def _sample_latency_ms(self) -> float:
        # Solve a log-normal so that median == p50 and ~p99 quantile == p99.
        import math
        mu: float = math.log(max(self.latency_p50_ms, 1.0))
        # z_{0.99} ≈ 2.326
        sigma: float = max(
            (math.log(max(self.latency_p99_ms, self.latency_p50_ms + 1.0)) - mu) / 2.326,
            1e-3,
        )
        return float(random.lognormvariate(mu, sigma))

    # ----- quality: SIMULATED via competence -----
    def _synthetic_response(self, request: InferenceRequest) -> str:
        """Produce a response whose informativeness scales with competence.

        The reward model scores coherence/completeness, so a higher-competence
        tier yields a higher reward — a *simulated* quality ladder. This is the
        explicitly non-real component.
        """
        prompt: str = request.prompt.strip()
        # Competence controls how many structured, on-topic sentences we emit.
        n_sentences: int = 1 + int(round(self.competence * 6))
        head: str = f"Regarding: {prompt[:80]}"
        body: str = " ".join(
            f"Point {i+1}: a relevant, well-formed consideration addressing the request."
            for i in range(n_sentences)
        )
        # Low-competence tiers also inject hedging/incompleteness.
        if self.competence < 0.5 and random.random() < (0.5 - self.competence):
            body += " (response may be incomplete)"
        return f"{head}. {body}"

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        latency_ms: float = self._sample_latency_ms()
        # Don't actually sleep the wall clock by seconds in a benchmark; emulate.
        await asyncio.sleep(0)

        if random.random() < self.failure_rate:
            self.record_failure(FailureMode.INFRASTRUCTURE)
            return InferenceResponse(
                request_id=request.request_id, engine_id=self.engine_id,
                tier=self.tier, content="", confidence=0.0,
                latency_ms=latency_ms, cost_usd=0.0, success=False,
                failure_mode=FailureMode.INFRASTRUCTURE,
                error_message="simulated infrastructure failure",
            )

        content: str = self._synthetic_response(request)
        n_in: int = max(1, len(request.prompt.split()))
        n_out: int = max(1, len(content.split()))
        cost: float = n_in * self.cost_per_input_token + n_out * self.cost_per_output_token
        # Self-reported confidence is competence + noise — the router may use
        # it, but quality_score in the benchmark comes from the reward model.
        confidence: float = max(0.0, min(1.0, self.competence + random.uniform(-0.1, 0.1)))
        self.record_success()
        return InferenceResponse(
            request_id=request.request_id, engine_id=self.engine_id,
            tier=self.tier, content=content, confidence=confidence,
            latency_ms=latency_ms, cost_usd=cost, success=True,
            failure_mode=FailureMode.NONE,
        )

    async def health_check(self) -> EngineStatus:
        return EngineStatus.HEALTHY


def build_calibrated_sim_engines() -> Tuple[List[CalibratedSimulatedEngine], str]:
    """Three tiers calibrated to public 2025 list prices and reported latencies.

    Prices are real (per-token, USD). Latencies are plausible public figures.
    Competence is the SIMULATED knob and is the only non-real quantity.

    Returns (engines, label) where label is explicitly a simulation marker so
    run_experiment records it in the manifest.
    """
    engines: List[CalibratedSimulatedEngine] = [
        # Tier 1: small local model (≈ Llama-3.2-3B on commodity GPU).
        CalibratedSimulatedEngine(
            "local-3b", tier=1,
            cost_per_input_token=1e-7, cost_per_output_token=1e-7,
            latency_p50_ms=120, latency_p99_ms=600,
            competence=0.55, failure_rate=0.02,
        ),
        # Tier 2: GPT-4o-mini class.
        CalibratedSimulatedEngine(
            "mid-4o-mini", tier=2,
            cost_per_input_token=1.5e-7, cost_per_output_token=6.0e-7,
            latency_p50_ms=400, latency_p99_ms=2500,
            competence=0.78, failure_rate=0.01,
        ),
        # Tier 3: GPT-4o class.
        CalibratedSimulatedEngine(
            "premium-4o", tier=3,
            cost_per_input_token=2.5e-6, cost_per_output_token=1.0e-5,
            latency_p50_ms=900, latency_p99_ms=6000,
            competence=0.93, failure_rate=0.005,
        ),
    ]
    return engines, "CALIBRATED SIMULATION (cost real, latency calibrated, quality simulated) — NOT real model calls"

