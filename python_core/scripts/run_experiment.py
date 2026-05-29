"""Single-command experiment runner with full provenance capture.

Reviewer-facing entry point. From a clean checkout:

    pip install -r python_core/requirements.txt
    python -m python_core.scripts.run_experiment \
        --dataset tatsu-lab/alpaca_eval \
        --max-samples 200 --seeds 5 \
        --output-dir results/

Outputs (under `results/{timestamp}/`):
    manifest.json   — git SHA, package versions, dataset hash, args, env
    raw_stats.json  — per-router mean/CI/values for cost/quality/latency/success
    summary.txt     — human-readable report including Pareto frontier
    pareto.csv      — (router, cost_mean, cost_ci, quality_mean, quality_ci, on_frontier)

The manifest pins everything needed to verify the run is reproducible:
the dataset SHA-256 fixes inputs; the package versions fix the toolchain;
the git SHA fixes the code under test; the seed list fixes the RNG.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engines.base import BaseEngine
from ..router.benchmark import (
    AggregateStats,
    RouterBenchmark,
)
from ..router.data_loader import hash_workload, load_prompt_workload


# Packages whose versions belong in the manifest — anything that can shift
# numerics or behavior across versions.
PINNED_PACKAGES = [
    "torch", "transformers", "sentence-transformers", "faiss-cpu",
    "presidio-analyzer", "presidio-anonymizer", "vaderSentiment",
    "routellm", "numpy", "scipy", "datasets",
]


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def _package_versions(packages: List[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _stats_to_dict(stats: Dict[str, Dict[str, AggregateStats]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for router, metrics in stats.items():
        out[router] = {
            metric: {
                "mean": agg.mean,
                "ci": agg.ci,
                "values": agg.values,
            }
            for metric, agg in metrics.items()
        }
    return out


def _write_pareto_csv(
    path: Path,
    stats: Dict[str, Dict[str, AggregateStats]],
    frontier_names: set,
) -> None:
    lines = ["router,cost_mean,cost_ci,quality_mean,quality_ci,on_frontier"]
    for router, metrics in sorted(stats.items(), key=lambda kv: kv[1]["cost"].mean):
        on_frontier = "1" if router in frontier_names else "0"
        lines.append(
            f"{router},"
            f"{metrics['cost'].mean:.8f},{metrics['cost'].ci:.8f},"
            f"{metrics['quality'].mean:.6f},{metrics['quality'].ci:.6f},"
            f"{on_frontier}"
        )
    path.write_text("\n".join(lines) + "\n")


def build_engines(mode: str = "auto") -> tuple:
    """Construct the engine list.

    mode:
      "auto" — try real engines from config.yaml, else FakeEngine (flagged).
      "sim"  — calibrated simulation (cost real, latency calibrated, quality
               simulated); explicitly watermarked in the manifest/figures.
      "fake" — deterministic FakeEngine test fixture.

    Returns (engines, label). The label is recorded in the manifest so
    simulation/fake runs can never be confused with real ones.
    """
    if mode == "sim":
        from ..engines.simulated_engine import build_calibrated_sim_engines
        return build_calibrated_sim_engines()
    if mode == "fake":
        from ..tests.conftest import FakeEngine
        return [
            FakeEngine("local-tier1", tier=1, confidence=0.6, cost_per_call=0.0001),
            FakeEngine("mid-tier2", tier=2, confidence=0.8, cost_per_call=0.001),
            FakeEngine("premium-tier3", tier=3, confidence=0.95, cost_per_call=0.01),
        ], "FakeEngine (smoke run — results not real)"
    try:
        from ..config.loader import load_config
        from ..engines.local_engine import LocalEngine
        from ..engines.cloud_engine import CloudEngine

        config = load_config()
        engines: List[BaseEngine] = []
        if config.engines.local.enabled:
            engines.append(LocalEngine(config=config.engines.local))
        if config.engines.mid.enabled:
            engines.append(CloudEngine(tier=2, config=config.engines.mid))
        if config.engines.premium.enabled:
            engines.append(CloudEngine(tier=3, config=config.engines.premium))
        if engines:
            return engines, "real engines from config.yaml"
    except Exception as e:
        print(f"[run_experiment] Real engine construction failed: {e}", file=sys.stderr)

    print("[run_experiment] WARNING: falling back to FakeEngine — numbers are not real.",
          file=sys.stderr)
    from ..tests.conftest import FakeEngine
    engines = [
        FakeEngine("local-tier1", tier=1, confidence=0.6, cost_per_call=0.0001),
        FakeEngine("mid-tier2", tier=2, confidence=0.8, cost_per_call=0.001),
        FakeEngine("premium-tier3", tier=3, confidence=0.95, cost_per_call=0.01),
    ]
    return engines, "FakeEngine (smoke run — results not real)"


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible benchmark runner for Cascade Engine.",
    )
    parser.add_argument("--dataset", default="tatsu-lab/alpaca_eval",
                        help="HuggingFace dataset id. Default: tatsu-lab/alpaca_eval")
    parser.add_argument("--config-name", default="alpaca_eval",
                        help="Dataset config name. Pass empty string for None.")
    parser.add_argument("--split", default="eval",
                        help="Dataset split (default: eval)")
    parser.add_argument("--prompt-field", default=None,
                        help="Override prompt field autodetection.")
    parser.add_argument("--reference-field", default=None,
                        help="Override reference field autodetection.")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Cap workload size (default: 200)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--system-mode", action="store_true",
                        help="Wrap each router with an IntelligentOrchestrator "
                             "(measures the full system stack, not pure routing).")
    parser.add_argument("--output-dir", default="results",
                        help="Where to write artifacts (default: ./results)")
    parser.add_argument("--label", default=None,
                        help="Optional human-readable run label.")
    parser.add_argument("--engines", choices=["auto", "sim", "fake"], default="auto",
                        help="Engine source. 'sim' = calibrated simulation "
                             "(watermarked); 'auto' = real-from-config else fake.")
    args = parser.parse_args(argv)

    config_name = args.config_name if args.config_name else None

    # Load workload + hash.
    workload = load_prompt_workload(
        dataset_name=args.dataset,
        config_name=config_name,
        split=args.split,
        max_samples=args.max_samples,
        prompt_field=args.prompt_field,
        reference_field=args.reference_field,
    )
    dataset_hash = hash_workload(workload)

    # Engines (real if available, fake otherwise — flagged in manifest).
    engines, engine_label = build_engines(mode=args.engines)

    # Run experiment.
    benchmark = RouterBenchmark(engines=engines, workload=workload)
    t_start = time.perf_counter()
    stats = await benchmark.run_experiment(
        n_seeds=args.seeds,
        system_mode=args.system_mode,
    )
    duration_s = time.perf_counter() - t_start

    # Pareto frontier and reports.
    frontier = benchmark.compute_pareto_frontier(stats)
    frontier_names = {p[0] for p in frontier}
    summary = benchmark.report_experiment(stats)

    # Output dir.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest.
    manifest = {
        "timestamp_utc": ts,
        "label": args.label,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(PINNED_PACKAGES),
        "dataset": {
            "name": args.dataset,
            "config": config_name,
            "split": args.split,
            "max_samples": args.max_samples,
            "n_loaded": len(workload),
            "sha256": dataset_hash,
            "prompt_field": args.prompt_field,
            "reference_field": args.reference_field,
        },
        "experiment": {
            "n_seeds": args.seeds,
            "seeds": list(range(args.seeds)),
            "system_mode": args.system_mode,
            "duration_s": duration_s,
        },
        "engines": {
            "label": engine_label,
            "tiers": [{"id": e.engine_id, "tier": e.tier} for e in engines],
        },
        "args": vars(args),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "raw_stats.json").write_text(json.dumps(_stats_to_dict(stats), indent=2))
    (out_dir / "summary.txt").write_text(summary + "\n")
    _write_pareto_csv(out_dir / "pareto.csv", stats, frontier_names)

    print(summary)
    print(f"\nArtifacts written to: {out_dir}")
    print(f"  manifest.json  — provenance (git SHA, deps, dataset hash)")
    print(f"  raw_stats.json — per-router aggregate stats")
    print(f"  summary.txt    — this report")
    print(f"  pareto.csv     — for plotting (see scripts/plot_pareto.py)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
