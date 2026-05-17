"""Plot cost vs quality with the Pareto frontier highlighted.

Usage:
    python -m python_core.scripts.plot_pareto results/20260514T093000Z/pareto.csv

Reads the CSV emitted by `run_experiment.py` and writes a PNG next to it.
Adds 95% CI error bars on both axes and labels each router.
"""

import argparse
import csv
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to pareto.csv emitted by run_experiment.py")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to alongside the CSV.")
    parser.add_argument("--title", default="Cost vs Quality (95% CI, Pareto frontier highlighted)")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "router": row["router"],
                "cost_mean": float(row["cost_mean"]),
                "cost_ci": float(row["cost_ci"]),
                "quality_mean": float(row["quality_mean"]),
                "quality_ci": float(row["quality_ci"]),
                "on_frontier": row["on_frontier"] == "1",
            })

    fig, ax = plt.subplots(figsize=(8, 6))
    frontier_pts = sorted(
        [r for r in rows if r["on_frontier"]],
        key=lambda r: r["cost_mean"],
    )
    if frontier_pts:
        ax.plot(
            [p["cost_mean"] for p in frontier_pts],
            [p["quality_mean"] for p in frontier_pts],
            "--", color="tab:green", alpha=0.6, label="Pareto frontier",
        )
    for r in rows:
        color = "tab:green" if r["on_frontier"] else "tab:gray"
        marker = "o" if r["on_frontier"] else "x"
        ax.errorbar(
            r["cost_mean"], r["quality_mean"],
            xerr=r["cost_ci"], yerr=r["quality_ci"],
            fmt=marker, color=color, markersize=8, capsize=3,
        )
        ax.annotate(
            r["router"], (r["cost_mean"], r["quality_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )

    ax.set_xlabel("Cost per request (USD)")
    ax.set_ylabel("Quality (reward-model score)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_path = Path(args.output) if args.output else csv_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
