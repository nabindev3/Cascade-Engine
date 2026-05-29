"""Tests for the reproducibility runner.

The most important guarantee here: the dataset hash is deterministic and
sensitive to changes. If hashing breaks, manifests stop being verifiable.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python_core.engines.base import InferenceRequest
from python_core.router.benchmark import WorkloadItem, QualityScorer
from python_core.router.data_loader import hash_workload


def _make_workload(items):
    return [
        WorkloadItem(
            request=InferenceRequest(request_id=f"r{i}", prompt=p),
            reference_response=r,
        )
        for i, (p, r) in enumerate(items)
    ]


def test_hash_is_deterministic():
    w1 = _make_workload([("a", "x"), ("b", "y")])
    w2 = _make_workload([("a", "x"), ("b", "y")])
    assert hash_workload(w1) == hash_workload(w2)


def test_hash_distinguishes_different_prompts():
    w1 = _make_workload([("a", "x"), ("b", "y")])
    w2 = _make_workload([("a", "x"), ("c", "y")])
    assert hash_workload(w1) != hash_workload(w2)


def test_hash_distinguishes_different_references():
    w1 = _make_workload([("a", "x")])
    w2 = _make_workload([("a", "z")])
    assert hash_workload(w1) != hash_workload(w2)


def test_hash_distinguishes_order():
    """Order matters — different orderings are different experiments."""
    w1 = _make_workload([("a", "x"), ("b", "y")])
    w2 = _make_workload([("b", "y"), ("a", "x")])
    assert hash_workload(w1) != hash_workload(w2)


def test_hash_handles_none_reference():
    w1 = _make_workload([("a", None)])
    w2 = _make_workload([("a", None)])
    assert hash_workload(w1) == hash_workload(w2)
    # And differs from a workload with a non-None reference.
    w3 = _make_workload([("a", "")])
    # Empty string vs None should both hash the same (since we coalesce to ""),
    # which is documented behavior.
    assert hash_workload(w1) == hash_workload(w3)


async def test_run_experiment_cli_smoke(tmp_path, monkeypatch):
    """Run the CLI end-to-end with a mocked dataset + mocked quality scorer.
    Verifies manifest, raw_stats, summary, pareto.csv are all written and
    the manifest contains expected provenance fields."""

    from python_core.scripts import run_experiment as runner

    # Mock dataset loading.
    fake_rows = [{"instruction": f"prompt {i}", "output": f"ref {i}"} for i in range(8)]

    class FakeDataset:
        def __getitem__(self, i): return fake_rows[i]
        def __len__(self): return len(fake_rows)
        def __iter__(self): return iter(fake_rows)

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: FakeDataset())

    # Mock the heavy reward model: deterministic constant.
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = 0.5

    # Patch RouterBenchmark construction to inject the mock scorer.
    real_router_benchmark_cls = runner.RouterBenchmark

    def patched_benchmark(engines, workload):
        return real_router_benchmark_cls(
            engines=engines,
            workload=workload,
            quality_scorer=QualityScorer(scorer=mock_scorer),
        )

    monkeypatch.setattr(runner, "RouterBenchmark", patched_benchmark)

    # CLI args.
    argv = [
        "--dataset", "fake/dataset",
        "--config-name", "",  # None
        "--split", "train",
        "--max-samples", "8",
        "--seeds", "2",
        "--output-dir", str(tmp_path),
        "--label", "smoke-test",
    ]
    rc = await runner.main(argv)
    assert rc == 0

    # Locate output directory.
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]

    # All four artifacts present.
    for name in ("manifest.json", "raw_stats.json", "summary.txt", "pareto.csv"):
        assert (run_dir / name).exists(), f"missing {name}"

    # Manifest fields.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["label"] == "smoke-test"
    assert manifest["dataset"]["name"] == "fake/dataset"
    assert manifest["dataset"]["sha256"]
    assert len(manifest["dataset"]["sha256"]) == 64  # SHA-256 hex
    assert manifest["dataset"]["n_loaded"] == 8
    assert manifest["experiment"]["n_seeds"] == 2
    assert manifest["experiment"]["seeds"] == [0, 1]
    assert "python_version" in manifest
    assert "package_versions" in manifest
    assert "engines" in manifest

    # Raw stats has the expected routers.
    raw = json.loads((run_dir / "raw_stats.json").read_text())
    assert "static_cascade" in raw
    assert "always_premium" in raw
    for router_name, metrics in raw.items():
        for metric_name in ("cost", "quality", "latency", "success"):
            assert metric_name in metrics
            assert "mean" in metrics[metric_name]
            assert "ci" in metrics[metric_name]
            assert "values" in metrics[metric_name]
            assert len(metrics[metric_name]["values"]) == 2  # n_seeds=2

    # Pareto CSV has a header + at least one row.
    csv_lines = (run_dir / "pareto.csv").read_text().strip().split("\n")
    assert csv_lines[0] == "router,cost_mean,cost_ci,quality_mean,quality_ci,on_frontier"
    assert len(csv_lines) >= 2
