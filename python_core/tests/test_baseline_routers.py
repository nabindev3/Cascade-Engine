"""Smoke tests for the FrugalGPT and RouteLLM baseline routers.

These tests verify the baselines load their real models and produce a routing
decision. They are slow (model downloads + CPU inference) — mark as heavy.
"""

import pytest

from python_core.router.baseline_routers import FrugalGPTRouter, RouteLLMRouter


pytestmark = pytest.mark.heavy


async def test_frugal_gpt_routes_with_real_scorer(fake_engines, simple_request):
    router = FrugalGPTRouter(engines=fake_engines, threshold=0.5)
    response, decision = await router.route(simple_request)
    # Must have actually called at least one engine.
    assert decision.engines_tried, "FrugalGPT did not call any engine"
    assert decision.final_engine in {e.engine_id for e in fake_engines}
    assert response.success


async def test_frugal_gpt_no_silent_mock_fallback():
    """If the reward model can't load, the constructor must raise — not mock."""
    from python_core.router.baseline_routers import _RewardModelScorer
    with pytest.raises((OSError, ImportError, ValueError)):
        _RewardModelScorer(model_name="this-model-does-not-exist-on-hf-12345")


async def test_routellm_routes_with_real_predictor(fake_engines, simple_request):
    router = RouteLLMRouter(engines=fake_engines)
    response, decision = await router.route(simple_request)
    assert decision.engines_tried, "RouteLLM did not call any engine"
    assert decision.final_engine in {e.engine_id for e in fake_engines}
    # Must include the win-rate probability in its decision trace.
    assert any("prob_strong=" in r for r in decision.escalation_reasons)


async def test_routellm_no_silent_mock_fallback():
    """Unknown router_type must raise — not silently degrade."""
    from python_core.router.baseline_routers import _RouteLLMPredictor
    with pytest.raises(ValueError):
        _RouteLLMPredictor(router_type="this-router-does-not-exist")
