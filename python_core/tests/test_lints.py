"""Tests for the Linear Contextual Thompson Sampling router (LinTS).

LinTS replaces the tabular ThompsonSamplingRouter's independent (bin, engine)
arms with a per-engine Bayesian linear reward model over a context vector, so
learning generalizes across similar prompts. These tests verify (1) the feature
map is well-formed, (2) each arm's posterior actually fits the reward it sees,
(3) the linear model GENERALIZES to unseen-but-similar contexts (the property a
tabular bandit cannot have), and (4) the router converges to the better engine.
"""

import numpy as np
import pytest

from python_core.router.learned_router import (
    LinTSArm,
    LinTSConfig,
    LinTSRouter,
    prompt_features,
)
from python_core.tests.conftest import FakeEngine
from python_core.engines.base import InferenceRequest


def test_prompt_features_shape_and_bounds():
    f_short = prompt_features("Hi")
    f_long = prompt_features("Analyze the following 500-word report and " * 30)
    assert f_short.shape == f_long.shape
    assert f_short[0] == 1.0 and f_long[0] == 1.0  # bias term
    # All features bounded to [0, 1].
    assert np.all(f_long >= 0.0) and np.all(f_long <= 1.0)
    # A longer prompt has a strictly larger normalized-length feature.
    assert f_long[1] > f_short[1]


def test_lints_arm_fits_constant_reward():
    """An arm fed a fixed (x, reward) must drive xᵀθ̂ toward that reward."""
    x = prompt_features("What is the capital of France?")
    arm = LinTSArm(d=len(x), prior_precision=1.0, exploration_scale=0.1)
    for _ in range(200):
        arm.update(x, 0.8)
    assert abs(arm.expected_reward(x) - 0.8) < 0.05


def test_lints_arm_generalizes_to_unseen_similar_context():
    """The defining property over a tabular bandit: train on two distinct
    contexts, then predict on a NEW context close to one of them and get the
    corresponding reward — generalization the bins cannot provide."""
    arm = LinTSArm(d=3, prior_precision=0.01, exploration_scale=0.0)
    hi = np.array([1.0, 1.0, 0.0])  # this pattern → high reward
    lo = np.array([1.0, 0.0, 1.0])  # this pattern → low reward
    for _ in range(300):
        arm.update(hi, 1.0)
        arm.update(lo, 0.0)
    # Unseen contexts that merely *resemble* the trained ones.
    near_hi = np.array([1.0, 0.9, 0.1])
    near_lo = np.array([1.0, 0.1, 0.9])
    assert arm.expected_reward(near_hi) > arm.expected_reward(near_lo) + 0.5


async def test_lints_router_routes_and_updates(fake_engines, simple_request):
    np.random.seed(0)
    router = LinTSRouter(engines=fake_engines, config=LinTSConfig())
    response, decision = await router.route(simple_request)
    assert decision.success
    assert decision.final_engine in {e.engine_id for e in fake_engines}
    # The engine it actually used must have had its posterior updated.
    assert router._arms[decision.final_engine].n_pulls >= 1


async def test_lints_router_learns_to_prefer_higher_reward_engine():
    """With a high-confidence cheap engine and an expensive low-confidence one,
    the net reward favors the cheap engine; LinTS should learn to prefer it."""
    np.random.seed(0)
    # 'cheap' wins on reward: high confidence, negligible cost.
    cheap = FakeEngine("cheap", tier=1, confidence=0.95, cost_per_call=0.0001)
    # 'pricey' loses: lower confidence AND a large cost penalty.
    pricey = FakeEngine("pricey", tier=2, confidence=0.6, cost_per_call=0.02)
    router = LinTSRouter(engines=[cheap, pricey], config=LinTSConfig())

    for i in range(150):
        await router.route(InferenceRequest(request_id=f"r{i}", prompt="Translate hello to French"))

    x = prompt_features("Translate hello to French")
    assert router._arms["cheap"].expected_reward(x) > router._arms["pricey"].expected_reward(x)
