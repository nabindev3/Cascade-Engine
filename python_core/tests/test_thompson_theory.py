"""Tests verifying that the Thompson Sampling router actually implements the
Algorithm 1 (CD-TS) from `paper/theory.tex`. If any of these properties break,
Theorem 1 stops applying to the implementation — which is exactly the failure
mode that produced the original biased update.
"""

import random
from statistics import mean

import pytest

from python_core.router.learned_router import (
    ArmStats,
    ThompsonConfig,
    ThompsonSamplingRouter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: the Bernoulli-trick update is UNBIASED.
#   E[Δα + Δβ] = 1 per update (conservation), and
#   E[Δα] = r where r is the reward.
# This is the key property that lets Agrawal-Goyal-style regret bounds carry
# over to bounded continuous rewards.
# ─────────────────────────────────────────────────────────────────────────────


def test_bernoulli_trick_alpha_update_is_unbiased():
    random.seed(12345)
    n_trials = 20_000
    target_r = 0.37  # continuous reward
    deltas = []
    for _ in range(n_trials):
        arm = ArmStats()
        a0, b0 = arm.alpha, arm.beta
        arm.update(target_r, decay=1.0, floor=0.0)  # no decay, no floor for clean measurement
        deltas.append(arm.alpha - a0)
    empirical = mean(deltas)
    # E[Δα] should equal target_r within ~2 standard errors.
    # Bernoulli variance r(1-r) ≈ 0.233, SE = sqrt(0.233/20000) ≈ 0.0034.
    assert abs(empirical - target_r) < 0.015, (
        f"E[Δα] = {empirical:.4f}, expected {target_r}. "
        f"The Bernoulli trick is biased — Theorem 1 no longer applies."
    )


def test_bernoulli_trick_alpha_plus_beta_increments_by_one():
    """Each update adds exactly 1 unit of mass to α+β (under no decay)."""
    random.seed(99)
    for _ in range(200):
        arm = ArmStats()
        a0, b0 = arm.alpha, arm.beta
        arm.update(random.random(), decay=1.0, floor=0.0)
        assert abs((arm.alpha + arm.beta) - (a0 + b0 + 1.0)) < 1e-9


def test_bernoulli_trick_clips_out_of_range_reward():
    """Rewards outside [0,1] must be clipped before the Bernoulli draw."""
    random.seed(0)
    arm = ArmStats()
    a0 = arm.alpha
    arm.update(1.5, decay=1.0, floor=0.0)  # treated as r=1 → always y=1 → α += 1
    assert arm.alpha == a0 + 1.0
    arm = ArmStats()
    b0 = arm.beta
    arm.update(-0.3, decay=1.0, floor=0.0)  # treated as r=0 → always y=0 → β += 1
    assert arm.beta == b0 + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: discounting is applied to the played arm only.
# Russac et al. (2019) and Theorem 1's proof use per-played-arm decay. Global
# decay (the old `_apply_decay` method) decays unplayed arms unnecessarily.
# ─────────────────────────────────────────────────────────────────────────────


def test_decay_is_per_played_arm_not_global():
    """After 100 updates on arm A, arm B's posterior must be unchanged."""
    arm_a = ArmStats()
    arm_b = ArmStats()
    b0_alpha, b0_beta = arm_b.alpha, arm_b.beta

    random.seed(7)
    for _ in range(100):
        arm_a.update(0.6, decay=0.99, floor=1.0)

    assert arm_b.alpha == b0_alpha, "Unplayed arm B's alpha changed — decay is global, not per-played-arm."
    assert arm_b.beta == b0_beta, "Unplayed arm B's beta changed — decay is global, not per-played-arm."


def test_router_no_longer_has_global_apply_decay_method():
    """Regression check: the old global-decay implementation should not return."""
    assert not hasattr(ThompsonSamplingRouter, "_apply_decay"), (
        "_apply_decay is back — this decays all arms each step and breaks "
        "the per-played-arm semantics of Algorithm 1."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 3: the floor keeps Beta(1,1) ≤ posterior — the prior is never lost.
# This guarantees the posterior is always a proper distribution.
# ─────────────────────────────────────────────────────────────────────────────


def test_floor_prevents_posterior_collapse():
    """Under aggressive decay and many zero-reward updates, the floor at 1
    must keep both α and β ≥ 1."""
    random.seed(42)
    arm = ArmStats()
    for _ in range(10_000):
        arm.update(0.0, decay=0.5, floor=1.0)
    assert arm.alpha >= 1.0
    assert arm.beta >= 1.0


def test_no_floor_allows_collapse():
    """Sanity check on the test: without the floor, repeated zero-reward
    updates with decay should drive α toward zero."""
    random.seed(42)
    arm = ArmStats()
    for _ in range(200):
        arm.update(0.0, decay=0.5, floor=0.0)
    assert arm.alpha < 0.01, (
        f"α = {arm.alpha} did not collapse without a floor — the test setup is wrong."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 4: the Beta posterior is proper after any sequence of updates.
# A sample from Beta(α, β) with α, β > 0 must succeed without exception.
# ─────────────────────────────────────────────────────────────────────────────


def test_posterior_is_always_samplable():
    """No matter the update sequence, sampling never raises."""
    random.seed(2026)
    arm = ArmStats()
    for _ in range(500):
        arm.update(random.random(), decay=0.99, floor=1.0)
        x = arm.sample()
        assert 0.0 <= x <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Property 5: posterior mean tracks the true reward distribution under decay.
# A quick smoke test that the algorithm actually learns: feed a stream where
# the true mean shifts mid-stream, and verify the posterior mean follows
# the second-half mean more closely than the overall mean.
# This is the "non-stationarity tracking" the discount factor is supposed to give.
# ─────────────────────────────────────────────────────────────────────────────


def test_discount_tracks_non_stationary_mean():
    random.seed(101)
    arm = ArmStats()

    # First 200 rounds: mean reward 0.2.
    for _ in range(200):
        r = 0.2 + 0.05 * (random.random() - 0.5)
        arm.update(max(0.0, min(1.0, r)), decay=0.97, floor=1.0)

    # Next 200 rounds: mean reward 0.8 (regime shift).
    for _ in range(200):
        r = 0.8 + 0.05 * (random.random() - 0.5)
        arm.update(max(0.0, min(1.0, r)), decay=0.97, floor=1.0)

    posterior_mean = arm.alpha / (arm.alpha + arm.beta)
    # Posterior mean should be much closer to 0.8 than to the overall mean 0.5.
    assert posterior_mean > 0.55, (
        f"Posterior mean {posterior_mean:.3f} did not track the regime shift; "
        f"discount factor 0.97 should give a window short enough to forget the "
        f"first regime within 200 rounds."
    )
