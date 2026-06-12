"""Shared fixtures for router tests."""

# transformers eagerly imports TensorFlow if it is installed, which collides
# with protobuf on macOS ("file defined twice") and breaks every heavy test as
# well as the FrugalGPT/RouteLLM baseline construction. We only ever use the
# torch backend, so disable the TF/Flax probes before transformers is imported.
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pytest

from python_core.engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)


class FakeEngine(BaseEngine):
    """Deterministic engine for tests. Quality and cost scale with tier."""

    def __init__(self, engine_id: str, tier: int, confidence: float = 0.9, cost_per_call: float = 0.001) -> None:
        super().__init__(engine_id=engine_id, tier=tier, config={})
        self._confidence: float = confidence
        self._cost_per_call: float = cost_per_call
        self.calls: list[str] = []

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        self.calls.append(request.prompt)
        return InferenceResponse(
            request_id=request.request_id,
            engine_id=self.engine_id,
            tier=self.tier,
            content=f"[{self.engine_id}] response to: {request.prompt[:40]}",
            confidence=self._confidence,
            cost_usd=self._cost_per_call,
            latency_ms=10.0,
            success=True,
            failure_mode=FailureMode.NONE,
        )

    def estimated_cost(self, request: InferenceRequest) -> float:
        return self._cost_per_call

    async def health_check(self) -> EngineStatus:
        return EngineStatus.HEALTHY


@pytest.fixture
def fake_engines() -> list[BaseEngine]:
    return [
        FakeEngine("local-tier1", tier=1, confidence=0.6, cost_per_call=0.0001),
        FakeEngine("mid-tier2", tier=2, confidence=0.8, cost_per_call=0.001),
        FakeEngine("premium-tier3", tier=3, confidence=0.95, cost_per_call=0.01),
    ]


@pytest.fixture
def simple_request() -> InferenceRequest:
    return InferenceRequest(request_id="test-001", prompt="What is the capital of France?")

