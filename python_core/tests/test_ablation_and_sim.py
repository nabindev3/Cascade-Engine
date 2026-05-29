"""Tests for the ablation runner, calibrated simulation engine, the new
single-router multi-seed method, and the figure watermark integrity check.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from python_core.engines.base import InferenceRequest
from python_core.engines.simulated_engine import (
    SIM_PREFIX,
    CalibratedSimulatedEngine,
    build_calibrated_sim_engines,
)
from python_core.router.benchmark import QualityScorer, RouterBenchmark
from python_core.router.learned_router import ThompsonConfig, ThompsonSamplingRouter


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated simulation engine
# ─────────────────────────────────────────────────────────────────────────────


def test_sim_engine_id_is_always_watermarked():
    """The SIM: prefix must be unremovable so provenance survives downstream."""
    e = CalibratedSimulatedEngine(
        "premium-4o", tier=3,
        cost_per_input_token=2.5e-6, cost_per_output_token=1e-5,
        latency_p50_ms=900, latency_p99_ms=6000, competence=0.93,
    )
    assert e.engine_id.startswith(SIM_PREFIX)
    # Even if already prefixed, no double-prefix.
    e2 = CalibratedSimulatedEngine(
        "SIM:already", tier=1,
        cost_per_input_token=1e-7, cost_per_output_token=1e-7,
        latency_p50_ms=100, latency_p99_ms=500, competence=0.5,
    )
    assert e2.engine_id == "SIM:already"


def test_sim_engine_cost_uses_real_per_token_prices():
    e = CalibratedSimulatedEngine(
        "mid", tier=2,
        cost_per_input_token=1.5e-7, cost_per_output_token=6.0e-7,
        latency_p50_ms=400, latency_p99_ms=2500, competence=0.78,
        avg_output_tokens=100,
    )
    req = InferenceRequest(request_id="t", prompt="one two three four five")
    # 5 input tokens * 1.5e-7 + 100 output * 6e-7
    expected = 5 * 1.5e-7 + 100 * 6.0e-7
    assert abs(e.estimated_cost(req) - expected) < 1e-15


async def test_sim_engine_quality_ladder_is_monotone_in_competence():
    """Higher-competence tiers must score higher under a real-ish scorer.

    We approximate the reward model with response length (the synthetic
    response grows with competence by construction), which is exactly the
    signal a coherence-based reward model rewards.
    """
    low = CalibratedSimulatedEngine(
        "lo", tier=1, cost_per_input_token=1e-7, cost_per_output_token=1e-7,
        latency_p50_ms=100, latency_p99_ms=500, competence=0.2, failure_rate=0.0,
    )
    high = CalibratedSimulatedEngine(
        "hi", tier=3, cost_per_input_token=1e-7, cost_per_output_token=1e-7,
        latency_p50_ms=100, latency_p99_ms=500, competence=0.95, failure_rate=0.0,
    )
    req = InferenceRequest(request_id="t", prompt="Explain gradient descent.")
    r_low = await low.infer(req)
    r_high = await high.infer(req)
    assert r_low.success and r_high.success
    assert len(r_high.content) > len(r_low.content)


def test_build_calibrated_sim_engines_label_marks_simulation():
    engines, label = build_calibrated_sim_engines()
    assert len(engines) == 3
    assert "SIMULATION" in label.upper()
    assert "NOT real model calls" in label or "NOT REAL" in label.upper()
    assert all(e.engine_id.startswith(SIM_PREFIX) for e in engines)
    assert [e.tier for e in engines] == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Single-router multi-seed method (used by the ablation runner)
# ─────────────────────────────────────────────────────────────────────────────


async def test_run_single_router_experiment_shape(fake_engines):
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.5
    bench = RouterBenchmark(engines=fake_engines, workload=[],
                            quality_scorer=QualityScorer(scorer=mock_scorer))
    bench.generate_synthetic_workload(n_requests=8)

    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return ThompsonSamplingRouter(engines=fake_engines, config=ThompsonConfig())

    stats = await bench.run_single_router_experiment(factory, "ts", n_seeds=3)
    # Fresh router built once per seed (no state leak).
    assert calls["n"] == 3
    for metric in ("cost", "latency", "quality", "success"):
        assert metric in stats
        assert len(stats[metric].values) == 3
        assert stats[metric].ci >= 0


async def test_single_router_uses_same_t_distribution_stats(fake_engines):
    """The ablation path must reuse the centralized t-dist/ddof=1 aggregation."""
    mock_scorer = MagicMock()
    mock_scorer.score.side_effect = lambda p, r: 0.3 + 0.4 * (len(p) % 3) / 2
    bench = RouterBenchmark(engines=fake_engines, workload=[],
                            quality_scorer=QualityScorer(scorer=mock_scorer))
    bench.generate_synthetic_workload(n_requests=15)

    def factory():
        return ThompsonSamplingRouter(engines=fake_engines, config=ThompsonConfig())

    stats = await bench.run_single_router_experiment(factory, "ts", n_seeds=5)
    # With n=5 and variation, CI must be > 0 and finite (t-dist, ddof=1).
    if len(set(stats["quality"].values)) > 1:
        assert stats["quality"].ci > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner CLI smoke
# ─────────────────────────────────────────────────────────────────────────────


async def test_run_ablation_cli_smoke(tmp_path, monkeypatch):
    from python_core.scripts import run_ablation as runner

    fake_rows = [{"instruction": f"prompt {i}", "output": f"ref {i}"} for i in range(6)]

    class FakeDataset:
        def __getitem__(self, i): return fake_rows[i]
        def __len__(self): return len(fake_rows)
        def __iter__(self): return iter(fake_rows)

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: FakeDataset())

    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.5
    real_cls = runner.RouterBenchmark

    def patched(engines, workload):
        return real_cls(engines=engines, workload=workload,
                        quality_scorer=QualityScorer(scorer=mock_scorer))

    monkeypatch.setattr(runner, "RouterBenchmark", patched)

    rc = await runner.main([
        "--dataset", "fake/ds", "--config-name", "", "--split", "train",
        "--max-samples", "6", "--seeds", "2",
        "--output-dir", str(tmp_path), "--label", "abl-smoke",
        "--axes", "decay,bins",  # keep it short
    ])
    assert rc == 0

    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    rd = runs[0]
    for name in ("manifest.json", "ablation.json", "ablation.csv", "summary.txt"):
        assert (rd / name).exists(), f"missing {name}"

    abl = json.loads((rd / "ablation.json").read_text())
    assert "decay" in abl and "bins" in abl
    # Every decay value swept produced full metric stats.
    for val, metrics in abl["decay"].items():
        for m in ("cost", "quality", "latency", "success"):
            assert m in metrics and "mean" in metrics[m] and "ci" in metrics[m]
            assert len(metrics[m]["values"]) == 2

    csv_lines = (rd / "ablation.csv").read_text().strip().split("\n")
    assert csv_lines[0] == "axis,value,metric,mean,ci"
    assert len(csv_lines) > 1


# ─────────────────────────────────────────────────────────────────────────────
# Figure watermark integrity
# ─────────────────────────────────────────────────────────────────────────────


def test_make_figures_flags_simulation(tmp_path):
    from python_core.scripts import make_figures

    # Minimal sim manifest + a pareto.csv.
    rd = tmp_path / "20260514T000000Z"
    rd.mkdir()
    (rd / "manifest.json").write_text(json.dumps({
        "experiment": {"n_seeds": 3},
        "engines": {"label": "CALIBRATED SIMULATION (cost real ...) — NOT real model calls"},
    }))
    (rd / "pareto.csv").write_text(
        "router,cost_mean,cost_ci,quality_mean,quality_ci,on_frontier\n"
        "always_local,0.0001,0.0,0.42,0.0,1\n"
        "always_premium,0.01,0.0,0.55,0.0,1\n"
    )
    rc = make_figures.main([str(rd)])
    assert rc == 0
    assert (rd / "fig_pareto.pdf").exists()

    # Provenance helper must classify this as NOT real.
    manifest = json.loads((rd / "manifest.json").read_text())
    label, is_real = make_figures._engine_provenance(manifest)
    assert is_real is False
    assert "SIMULATION" in label.upper()


def test_make_figures_real_run_not_flagged(tmp_path):
    from python_core.scripts import make_figures
    manifest = {"engines": {"label": "real engines from config.yaml"}}
    label, is_real = make_figures._engine_provenance(manifest)
    assert is_real is True
