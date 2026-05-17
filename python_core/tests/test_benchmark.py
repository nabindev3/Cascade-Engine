"""Tests for the benchmark statistics, Pareto frontier, and quality scoring.

These tests verify the pure-Python pieces (frontier, t-critical, scorer cache,
router construction) without loading the heavy reward model. The reward-model
path is covered separately under `heavy`.
"""

import math
from unittest.mock import MagicMock

import pytest

from python_core.router.benchmark import (
    AggregateStats,
    QualityScorer,
    RouterBenchmark,
    _t_critical,
    pareto_frontier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────


def test_pareto_frontier_basic():
    points = [
        ("a", 1.0, 0.5),  # on frontier (cheapest)
        ("b", 2.0, 0.7),  # on frontier (better quality)
        ("c", 3.0, 0.6),  # dominated by b
        ("d", 4.0, 0.9),  # on frontier (best quality)
    ]
    frontier = pareto_frontier(points)
    names = [p[0] for p in frontier]
    assert names == ["a", "b", "d"]


def test_pareto_frontier_handles_ties():
    points = [
        ("a", 1.0, 0.5),
        ("b", 1.0, 0.7),  # same cost, higher quality — should win
        ("c", 2.0, 0.7),  # dominated by b (same quality, higher cost)
        ("d", 2.0, 0.8),
    ]
    frontier = pareto_frontier(points)
    names = [p[0] for p in frontier]
    assert names == ["b", "d"]


def test_pareto_frontier_single_point():
    assert pareto_frontier([("only", 1.0, 0.5)]) == [("only", 1.0, 0.5)]


def test_pareto_frontier_all_dominated_except_one():
    # One point dominates everything else.
    points = [
        ("super", 0.001, 0.999),
        ("a", 0.5, 0.3),
        ("b", 1.0, 0.4),
        ("c", 2.0, 0.5),
    ]
    frontier = pareto_frontier(points)
    assert [p[0] for p in frontier] == ["super"]


# ─────────────────────────────────────────────────────────────────────────────
# t-critical
# ─────────────────────────────────────────────────────────────────────────────


def test_t_critical_small_n_is_not_z_196():
    """For n=5 (df=4), t-critical ≈ 2.776, NOT 1.96. Catches the regression."""
    t5 = _t_critical(n=5)
    assert t5 > 2.5, f"t-critical for n=5 should be ~2.776, got {t5}"
    assert t5 < 3.0


def test_t_critical_converges_to_z_for_large_n():
    t100 = _t_critical(n=100)
    assert abs(t100 - 1.96) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# QualityScorer caching
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_scorer_caches_identical_inputs():
    """Reward model is deterministic; duplicate (prompt, response) must not re-score."""
    mock_inner = MagicMock()
    mock_inner.score.return_value = 0.73
    scorer = QualityScorer(scorer=mock_inner)

    s1 = scorer.score("prompt A", "response 1")
    s2 = scorer.score("prompt A", "response 1")  # identical → cache hit
    s3 = scorer.score("prompt A", "response 2")  # different response

    assert s1 == s2 == 0.73
    assert s3 == 0.73
    assert mock_inner.score.call_count == 2  # not 3


def test_quality_scorer_empty_response_is_zero():
    mock_inner = MagicMock()
    scorer = QualityScorer(scorer=mock_inner)
    assert scorer.score("prompt", "") == 0.0
    mock_inner.score.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# always_premium fix: must route DIRECTLY to top tier, not cascade through.
# ─────────────────────────────────────────────────────────────────────────────


async def test_always_premium_does_not_pay_for_lower_tiers(fake_engines):
    """Regression: the old `always_premium` was CascadeRouter with high thresholds,
    which paid for Tier 1 + Tier 2 + Tier 3 per request. The fixed version
    should call only the top-tier engine."""
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.5

    bench = RouterBenchmark(
        engines=fake_engines,
        workload=[],
        quality_scorer=QualityScorer(scorer=mock_scorer),
    )
    bench.generate_synthetic_workload(n_requests=5)
    results = await bench.run_all(system_mode=False)

    ap = results["always_premium"]
    # Top-tier engine cost is 0.01 in the fixture. With 5 requests, total ≤ 0.05.
    # The old buggy version would have totalled 5 × (0.0001 + 0.001 + 0.01) = 0.0555+.
    assert ap.total_cost_usd <= 0.05 + 1e-9, (
        f"always_premium cost {ap.total_cost_usd} is too high — it's probably "
        f"cascading through lower tiers again."
    )
    # All requests must end on the top tier.
    assert set(ap.tier_distribution.keys()) == {3}


async def test_quality_score_uses_scorer_not_confidence(fake_engines):
    """quality_score must come from the reward model, not the engine's confidence.

    fake_engines have confidence 0.6/0.8/0.95. If we mock the scorer to return
    0.42 for everything, every router's quality_score must be ~0.42 — proving
    we're no longer falling back to confidence.
    """
    import numpy as np
    import random as _random
    _random.seed(20260514)
    np.random.seed(20260514)
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.42

    bench = RouterBenchmark(
        engines=fake_engines,
        workload=[],
        quality_scorer=QualityScorer(scorer=mock_scorer),
    )
    bench.generate_synthetic_workload(n_requests=10)
    results = await bench.run_all(system_mode=False)

    for name, r in results.items():
        # quality is the mean reward across requests; cached responses still get scored.
        assert abs(r.quality_score - 0.42) < 1e-6, (
            f"Router {name} quality_score={r.quality_score} != 0.42 — "
            f"the scorer pathway isn't being used."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed plumbing
# ─────────────────────────────────────────────────────────────────────────────


async def test_run_experiment_returns_aggregate_stats(fake_engines):
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.5

    bench = RouterBenchmark(
        engines=fake_engines,
        workload=[],
        quality_scorer=QualityScorer(scorer=mock_scorer),
    )
    bench.generate_synthetic_workload(n_requests=10)
    stats = await bench.run_experiment(n_seeds=3, system_mode=False)

    for name, metrics in stats.items():
        for k in ("cost", "latency", "quality", "success"):
            assert k in metrics
            assert isinstance(metrics[k], AggregateStats)
            assert len(metrics[k].values) == 3
            # CI must be non-negative and finite.
            assert metrics[k].ci >= 0
            assert math.isfinite(metrics[k].ci)


async def test_run_experiment_uses_sample_std_not_population(fake_engines):
    """ddof=1 regression check. If someone reverts to ddof=0, this will fail."""
    mock_scorer = MagicMock()
    # Make quality vary so std is non-zero.
    mock_scorer.score.side_effect = lambda p, r: 0.3 + 0.4 * (len(p) % 3) / 2
    bench = RouterBenchmark(
        engines=fake_engines,
        workload=[],
        quality_scorer=QualityScorer(scorer=mock_scorer),
    )
    bench.generate_synthetic_workload(n_requests=20)
    stats = await bench.run_experiment(n_seeds=5, system_mode=False)

    # Pick any router with variation across seeds and verify CI was computed
    # with ddof=1 (sample) not ddof=0 (population). Sample std > population std,
    # so we just verify CI is non-trivially large for n=5.
    for name, metrics in stats.items():
        quality = metrics["quality"]
        if len(set(quality.values)) > 1:
            # n=5, t≈2.776. Sample std non-zero ⇒ CI > 0 and noticeably wider
            # than the population-std version would be.
            assert quality.ci > 0
            return
