"""Tests for Phase-1 robustness: exponential backoff (cloud engine) and the
downgrade-to-local fallback (cascade router).

These exercise the failure paths that the framework must survive in production:
transient rate-limits / server errors on the cloud tiers, and a total cloud
outage that should still produce an answer from a local tier.
"""

import httpx
import pytest

from python_core.engines import cloud_engine
from python_core.engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)
from python_core.engines.cloud_engine import OpenAIEngine
from python_core.router.cascade_router import CascadeRouter, RouterConfig


# ─── Helpers ────────────────────────────────────────────────────────────────


def _patch_transport(monkeypatch, handler) -> None:
    """Route OpenAIEngine's internally-constructed AsyncClient through a
    MockTransport so no real network call is made."""
    real_async_client = httpx.AsyncClient

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(cloud_engine.httpx, "AsyncClient", factory)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )


def _engine(monkeypatch, **overrides) -> OpenAIEngine:
    # Tiny backoff so retries don't slow the suite; jitter still exercised.
    cfg = {"api_key": "test", "model": "gpt-test", "backoff_base_s": 0.001, **overrides}
    eng = OpenAIEngine(engine_id="cloud-test", tier=2, config=cfg)
    return eng


REQ = InferenceRequest(request_id="r1", prompt="hi there")


# ─── Backoff: cloud engine ──────────────────────────────────────────────────


async def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return _ok_response()

    _patch_transport(monkeypatch, handler)
    eng = _engine(monkeypatch, max_retries=2)

    resp = await eng.predict(REQ)

    assert calls["n"] == 2, "should have retried once after the 429"
    assert resp.success
    assert resp.content == "hello"


async def test_retries_on_transient_5xx(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, text="unavailable")
        return _ok_response()

    _patch_transport(monkeypatch, handler)
    eng = _engine(monkeypatch, max_retries=3)

    resp = await eng.predict(REQ)

    assert calls["n"] == 3
    assert resp.success


async def test_gives_up_after_max_retries_with_rate_limit_mode(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "nope"})

    _patch_transport(monkeypatch, handler)
    eng = _engine(monkeypatch, max_retries=2)

    resp = await eng.predict(REQ)

    assert not resp.success
    assert resp.failure_mode == FailureMode.RATE_LIMIT


async def test_auth_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    _patch_transport(monkeypatch, handler)
    eng = _engine(monkeypatch, max_retries=5)

    resp = await eng.predict(REQ)

    assert calls["n"] == 1, "auth errors must fail fast, no retries"
    assert resp.failure_mode == FailureMode.AUTH_ERROR


async def test_backoff_honors_retry_after_header(monkeypatch):
    captured = {}

    async def fake_sleep(self, attempt, retry_after):
        captured["retry_after"] = retry_after

    monkeypatch.setattr(OpenAIEngine, "_backoff_sleep", fake_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return _ok_response()

    _patch_transport(monkeypatch, handler)
    eng = _engine(monkeypatch, max_retries=2)

    resp = await eng.predict(REQ)

    assert resp.success
    assert captured["retry_after"] == "7"


# ─── Fallback: cascade router downgrade-to-local ────────────────────────────


class ScriptedEngine(BaseEngine):
    """Engine that always succeeds or always fails, on demand."""

    def __init__(self, engine_id: str, tier: int, *, succeed: bool, confidence: float = 0.95) -> None:
        super().__init__(engine_id=engine_id, tier=tier, config={})
        self.succeed = succeed
        self.confidence = confidence
        self.calls = 0

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        if self.succeed:
            return InferenceResponse(
                request_id=request.request_id, engine_id=self.engine_id, tier=self.tier,
                content=f"[{self.engine_id}] ok", confidence=self.confidence,
                cost_usd=0.0001, success=True, failure_mode=FailureMode.NONE,
            )
        return InferenceResponse(
            request_id=request.request_id, engine_id=self.engine_id, tier=self.tier,
            content="", confidence=0.0, cost_usd=0.0, success=False,
            failure_mode=FailureMode.INFRASTRUCTURE, error_message="down",
        )

    def estimated_cost(self, request: InferenceRequest) -> float:
        return 0.0001

    async def health_check(self) -> EngineStatus:
        return EngineStatus.HEALTHY


async def test_downgrades_to_local_when_cloud_fails():
    """min_tier=2 forces cloud tiers, but both fail — router must fall back to
    the skipped local tier-1 rather than returning a hard failure."""
    local = ScriptedEngine("local-1", tier=1, succeed=True)
    mid = ScriptedEngine("mid-2", tier=2, succeed=False)
    premium = ScriptedEngine("premium-3", tier=3, succeed=False)
    router = CascadeRouter([local, mid, premium], RouterConfig(enable_local_fallback=True))

    req = InferenceRequest(request_id="r2", prompt="logical query", min_tier=2)
    resp, decision = await router.route(req)

    assert resp.success
    assert resp.engine_id == "local-1"
    assert resp.was_escalated
    assert decision.final_tier == 1
    assert local.calls == 1
    assert any("local fallback" in r for r in decision.escalation_reasons)


async def test_no_fallback_when_disabled():
    local = ScriptedEngine("local-1", tier=1, succeed=True)
    mid = ScriptedEngine("mid-2", tier=2, succeed=False)
    router = CascadeRouter([local, mid], RouterConfig(enable_local_fallback=False))

    req = InferenceRequest(request_id="r3", prompt="q", min_tier=2)
    resp, decision = await router.route(req)

    assert not resp.success
    assert local.calls == 0, "local must NOT be tried when fallback is disabled"


async def test_no_fallback_when_higher_tier_succeeds():
    local = ScriptedEngine("local-1", tier=1, succeed=True)
    mid = ScriptedEngine("mid-2", tier=2, succeed=True)
    router = CascadeRouter([local, mid], RouterConfig(enable_local_fallback=True))

    req = InferenceRequest(request_id="r4", prompt="q", min_tier=2)
    resp, decision = await router.route(req)

    assert resp.success
    assert resp.engine_id == "mid-2"
    assert local.calls == 0, "fallback must not fire when a permitted tier succeeds"
