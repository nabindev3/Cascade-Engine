"""Tests for OrchestrationWrapper — covers the fairness-critical properties."""

import pytest

from python_core.engines.base import InferenceRequest
from python_core.router.cascade_router import CascadeRouter, RouterConfig
from python_core.router.intelligent_layers import IntelligentOrchestrator
from python_core.router.orchestration_wrapper import OrchestrationWrapper


pytestmark = pytest.mark.heavy


async def test_wrapper_caches_response(fake_engines, simple_request):
    """Second identical request must hit cache (no engine call)."""
    inner = CascadeRouter(engines=fake_engines, config=RouterConfig())
    wrapper = OrchestrationWrapper(inner, IntelligentOrchestrator(cache_threshold=0.95))

    response1, decision1 = await wrapper.route(simple_request)
    assert decision1.final_engine != "cache"
    calls_after_first = sum(len(e.calls) for e in fake_engines)

    second_request = InferenceRequest(request_id="test-002", prompt=simple_request.prompt)
    response2, decision2 = await wrapper.route(second_request)
    assert decision2.final_engine == "cache"
    assert response2.content == response1.content
    # No additional engine calls.
    assert sum(len(e.calls) for e in fake_engines) == calls_after_first


async def test_wrapper_isolates_state_per_instance(fake_engines, simple_request):
    """Two wrappers must NOT share cache state — required for benchmark fairness."""
    inner_a = CascadeRouter(engines=fake_engines, config=RouterConfig())
    inner_b = CascadeRouter(engines=fake_engines, config=RouterConfig())
    wrapper_a = OrchestrationWrapper(inner_a, IntelligentOrchestrator())
    wrapper_b = OrchestrationWrapper(inner_b, IntelligentOrchestrator())

    await wrapper_a.route(simple_request)
    calls_before_b = sum(len(e.calls) for e in fake_engines)

    second_request = InferenceRequest(request_id="test-003", prompt=simple_request.prompt)
    _, decision_b = await wrapper_b.route(second_request)

    # Wrapper B has no cache → must call an engine.
    assert decision_b.final_engine != "cache"
    assert sum(len(e.calls) for e in fake_engines) > calls_before_b


async def test_wrapper_masks_pii_before_routing(fake_engines):
    """Engine must receive the masked prompt, not the raw PII."""
    inner = CascadeRouter(engines=fake_engines, config=RouterConfig())
    wrapper = OrchestrationWrapper(inner, IntelligentOrchestrator())

    req = InferenceRequest(
        request_id="test-pii",
        prompt="My card is 4111-1111-1111-1111. Please help.",
    )
    await wrapper.route(req)

    seen_prompts = [p for e in fake_engines for p in e.calls]
    assert any(seen_prompts), "Engine was not called at all"
    for prompt in seen_prompts:
        assert "4111-1111-1111-1111" not in prompt, "Raw PII leaked to engine"
