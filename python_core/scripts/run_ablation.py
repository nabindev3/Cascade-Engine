"""One-at-a-time ablation runner for the CD-TS router.

Answers the reviewer questions the theory raises:

  * How does regret/quality/cost vary with the discount factor gamma
    (decay_factor)? Theory predicts a U-shape: too small forgets useful
    history, too large fails to track drift (Theorem 1, the L^* tradeoff).
  * How does context granularity (n_complexity_bins, the M in the bound)
    affect cost/quality? The bound scales as M^{1/3}.
  * How do the reward weights (lambda = cost_penalty, mu = latency_penalty)
    move the operating point on the cost-quality-latency surface?

Each axis is swept one hyperparameter at a time, all others held at default.
Every setting is evaluated with the SAME multi-seed statistical treatment as
run_experiment (t-distribution 95% CI, ddof=1) via
RouterBenchmark.run_single_router_experiment, so ablation numbers are directly
comparable to the main results.

Usage (from repo root):

    python -m python_core.scripts.run_ablation \\
        --dataset tatsu-lab/alpaca_eval --max-samples 200 --seeds 5 \\
        --label "paper2-ablations"

Outputs under results/ablation_{timestamp}/:
    manifest.json   — provenance (git SHA, deps, dataset hash, swept grids)
    ablation.json   — {axis: {value: {metric: {mean, ci, values}}}}
    ablation.csv    — flat table: axis,value,metric,mean,ci
    summary.txt     — human-readable per-axis tables
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..router.benchmark import AggregateStats, RouterBenchmark
from ..router.data_loader import hash_workload, load_prompt_workload
from ..router.learned_router import ThompsonConfig, ThompsonSamplingRouter
from .run_experiment import (
    PINNED_PACKAGES,
    _git_sha,
    _git_dirty,
    _package_versions,
    build_engines,
)


# Default sweep grids. Override via CLI if needed.
GRID_DECAY = [0.90, 0.95, 0.98, 0.995, 1.0]
GRID_BINS = [1, 3, 5, 8, 12]
GRID_COST_PENALTY = [1.0, 5.0, 10.0, 25.0, 50.0]      # lambda
GRID_LATENCY_PENALTY = [0.0, 0.0005, 0.001, 0.005]    # mu


def _stats_to_dict(stats: Dict[str, AggregateStats]) -> dict:
    return {
        metric: {"mean": agg.mean, "ci": agg.ci, "values": agg.values}
        for metric, agg in stats.items()
    }


async def _sweep_axis(
    benchmark: RouterBenchmark,
    engines,
    axis_name: str,
    values: list,
    config_field: str,
    n_seeds: int,
) -> Dict[str, dict]:
    """Sweep one ThompsonConfig field, holding all others at default."""
    out: Dict[str, dict] = {}
    for v in values:
        kwargs = {config_field: v}
        # Build a fresh router each seed so learned posteriors never leak.
        def factory(_kwargs=kwargs):
            return ThompsonSamplingRouter(
                engines=engines, config=ThompsonConfig(**_kwargs)
            )

        stats = await benchmark.run_single_router_experiment(
            router_factory=factory,
            name=f"{axis_name}={v}",
            n_seeds=n_seeds,
        )
        out[str(v)] = _stats_to_dict(stats)
        print(
            f"  {axis_name}={v}: "
            f"cost={stats['cost'].mean:.5f}±{stats['cost'].ci:.5f}  "
            f"quality={stats['quality'].mean:.4f}±{stats['quality'].ci:.4f}  "
            f"latency={stats['latency'].mean:.0f}±{stats['latency'].ci:.0f}ms"
        )
    return out


def _summary(ablation: Dict[str, Dict[str, dict]]) -> str:
    lines = ["CD-TS ABLATION SUMMARY", "=" * 72]
    for axis, sweep in ablation.items():
        lines.append(f"\n[{axis}]")
        lines.append(f"  {'value':>10} {'cost':>16} {'quality':>16} {'latency(ms)':>16}")
        for val, metrics in sweep.items():
            c, q, l = metrics["cost"], metrics["quality"], metrics["latency"]
            lines.append(
                f"  {val:>10} "
                f"{c['mean']:.5f}±{c['ci']:.5f} "
                f"{q['mean']:.4f}±{q['ci']:.4f} "
                f"{l['mean']:.0f}±{l['ci']:.0f}"
            )
    return "\n".join(lines)


def _write_csv(path: Path, ablation: Dict[str, Dict[str, dict]]) -> None:
    rows = ["axis,value,metric,mean,ci"]
    for axis, sweep in ablation.items():
        for val, metrics in sweep.items():
            for metric, agg in metrics.items():
                rows.append(f"{axis},{val},{metric},{agg['mean']:.8f},{agg['ci']:.8f}")
    path.write_text("\n".join(rows) + "\n")


async def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CD-TS one-at-a-time ablation runner.")
    p.add_argument("--dataset", default="tatsu-lab/alpaca_eval")
    p.add_argument("--config-name", default="alpaca_eval")
    p.add_argument("--split", default="eval")
    p.add_argument("--prompt-field", default=None)
    p.add_argument("--reference-field", default=None)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--label", default=None)
    p.add_argument("--axes", default="decay,bins,cost_penalty,latency_penalty",
                   help="Comma-separated subset of axes to run.")
    args = p.parse_args(argv)

    config_name = args.config_name if args.config_name else None

    workload = load_prompt_workload(
        dataset_name=args.dataset,
        config_name=config_name,
        split=args.split,
        max_samples=args.max_samples,
        prompt_field=args.prompt_field,
        reference_field=args.reference_field,
    )
    dataset_hash = hash_workload(workload)
    engines, engine_label = build_engines()
    benchmark = RouterBenchmark(engines=engines, workload=workload)

    axes = set(a.strip() for a in args.axes.split(","))
    grids = {
        "decay": ("decay_factor", GRID_DECAY),
        "bins": ("n_complexity_bins", GRID_BINS),
        "cost_penalty": ("cost_penalty", GRID_COST_PENALTY),
        "latency_penalty": ("latency_penalty", GRID_LATENCY_PENALTY),
    }

    t0 = time.perf_counter()
    ablation: Dict[str, Dict[str, dict]] = {}
    for axis in ("decay", "bins", "cost_penalty", "latency_penalty"):
        if axis not in axes:
            continue
        field, grid = grids[axis]
        print(f"\nSweeping {axis} ({field}) over {grid} ...")
        ablation[axis] = await _sweep_axis(
            benchmark, engines, axis, grid, field, args.seeds
        )
    duration = time.perf_counter() - t0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": ts,
        "label": args.label,
        "kind": "ablation",
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(PINNED_PACKAGES),
        "dataset": {
            "name": args.dataset,
            "config": config_name,
            "split": args.split,
            "n_loaded": len(workload),
            "sha256": dataset_hash,
        },
        "experiment": {"n_seeds": args.seeds, "duration_s": duration, "axes": sorted(axes)},
        "engines": {"label": engine_label,
                    "tiers": [{"id": e.engine_id, "tier": e.tier} for e in engines]},
        "grids": {
            "decay": GRID_DECAY, "bins": GRID_BINS,
            "cost_penalty": GRID_COST_PENALTY, "latency_penalty": GRID_LATENCY_PENALTY,
        },
        "args": vars(args),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "ablation.json").write_text(json.dumps(ablation, indent=2))
    (out_dir / "summary.txt").write_text(_summary(ablation) + "\n")
    _write_csv(out_dir / "ablation.csv", ablation)

    print("\n" + _summary(ablation))
    print(f"\nArtifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
