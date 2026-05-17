"""Generate publication figures from a results directory.

Produces PDF figures (vector, paper-ready) from artifacts written by
run_experiment.py and/or run_ablation.py:

    fig_pareto.pdf       — cost vs quality with Pareto frontier + 95% CI
    fig_ablation_*.pdf   — one curve per swept axis (decay, bins, ...)

INTEGRITY: every figure is stamped with the manifest's engine label. If the
run used simulated or fake engines, the stamp is a large red banner reading
"SIMULATION — NOT REAL MODEL CALLS" that cannot be removed via CLI flag. A
figure can therefore never silently misrepresent simulated numbers as measured.

Usage:
    python -m python_core.scripts.make_figures results/20260514T093000Z
    python -m python_core.scripts.make_figures results/ablation_20260514T100000Z
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def _engine_provenance(manifest: dict) -> tuple:
    """Return (label, is_real). is_real is False for sim/fake runs."""
    label = manifest.get("engines", {}).get("label", "unknown engine source")
    lowered = label.lower()
    is_real = not ("simulation" in lowered or "fakeengine" in lowered
                   or "not real" in lowered or "results not real" in lowered)
    return label, is_real


def _stamp(fig, label: str, is_real: bool) -> None:
    """Watermark every figure with engine provenance."""
    if is_real:
        fig.text(0.99, 0.01, label, ha="right", va="bottom",
                 fontsize=7, color="gray", alpha=0.7)
    else:
        # Unmissable banner for non-real runs.
        fig.text(0.5, 0.5, "SIMULATION\nNOT REAL MODEL CALLS",
                 ha="center", va="center", fontsize=34, color="red",
                 alpha=0.18, rotation=30, fontweight="bold", zorder=0)
        fig.text(0.99, 0.01, label, ha="right", va="bottom",
                 fontsize=7, color="red", alpha=0.85)


def _plot_pareto(plt, results_dir: Path, manifest: dict) -> Path:
    rows = []
    with (results_dir / "pareto.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "router": r["router"],
                "cost": float(r["cost_mean"]), "cost_ci": float(r["cost_ci"]),
                "q": float(r["quality_mean"]), "q_ci": float(r["quality_ci"]),
                "front": r["on_frontier"] == "1",
            })
    fig, ax = plt.subplots(figsize=(7, 5))
    front = sorted([r for r in rows if r["front"]], key=lambda r: r["cost"])
    if len(front) >= 2:
        ax.plot([p["cost"] for p in front], [p["q"] for p in front],
                "--", color="tab:green", alpha=0.6, label="Pareto frontier")
    for r in rows:
        ax.errorbar(r["cost"], r["q"], xerr=r["cost_ci"], yerr=r["q_ci"],
                    fmt="o" if r["front"] else "x",
                    color="tab:green" if r["front"] else "tab:gray",
                    markersize=7, capsize=3)
        ax.annotate(r["router"], (r["cost"], r["q"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Cost per request (USD)")
    ax.set_ylabel("Quality (reward-model score)")
    ax.set_title(f"Cost vs Quality — {manifest['experiment'].get('n_seeds','?')} seeds, 95% CI")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    label, is_real = _engine_provenance(manifest)
    _stamp(fig, label, is_real)
    out = results_dir / "fig_pareto.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_ablation(plt, results_dir: Path, manifest: dict) -> list:
    ablation = json.loads((results_dir / "ablation.json").read_text())
    label, is_real = _engine_provenance(manifest)
    outs = []
    for axis, sweep in ablation.items():
        xs, q_m, q_c, c_m, c_c = [], [], [], [], []
        for val, metrics in sweep.items():
            try:
                xv = float(val)
            except ValueError:
                xv = val
            xs.append(xv)
            q_m.append(metrics["quality"]["mean"]); q_c.append(metrics["quality"]["ci"])
            c_m.append(metrics["cost"]["mean"]); c_c.append(metrics["cost"]["ci"])
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.errorbar(range(len(xs)), q_m, yerr=q_c, fmt="o-",
                     color="tab:blue", capsize=3, label="Quality")
        ax1.set_xticks(range(len(xs)))
        ax1.set_xticklabels([str(x) for x in xs])
        ax1.set_xlabel(axis)
        ax1.set_ylabel("Quality (reward-model score)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.errorbar(range(len(xs)), c_m, yerr=c_c, fmt="s--",
                     color="tab:red", capsize=3, label="Cost")
        ax2.set_ylabel("Cost per request (USD)", color="tab:red")
        ax1.set_title(f"Ablation: {axis}")
        ax1.grid(True, alpha=0.3)
        _stamp(fig, label, is_real)
        out = results_dir / f"fig_ablation_{axis}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def _plot_regret(plt, results_dir: Path, manifest: dict) -> list:
    """Log-log cumulative-regret curves per policy, one figure per regime.

    The headline of the paper's decisive experiment: CD-TS stays a straight
    line of slope ≈ 2/3 (Theorem 1) while static-optimal bends to slope ≈ 1
    (linear regret) once the environment drifts past its frozen policy.
    """
    rows = []
    with (results_dir / "regret_curves.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append((r["regime"], int(r["t"]), r["policy"],
                         float(r["mean_cumulative_regret"])))
    regimes = sorted({r[0] for r in rows})
    label, is_real = _engine_provenance(manifest)
    colors = {"cd_ts": "tab:blue", "static_optimal": "tab:red",
              "round_robin": "tab:gray"}
    outs = []
    for regime in regimes:
        fig, ax = plt.subplots(figsize=(7, 5))
        for policy in ("cd_ts", "static_optimal", "round_robin"):
            pts = sorted((t, v) for rg, t, p, v in rows if rg == regime and p == policy)
            if not pts:
                continue
            ax.plot([t + 1 for t, _ in pts], [max(v, 1e-9) for _, v in pts],
                    label=policy, color=colors.get(policy), linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Round t (log)")
        ax.set_ylabel("Cumulative pseudo-regret (log)")
        ax.set_title(f"Non-stationary regret — regime: {regime}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
        _stamp(fig, label, is_real)
        out = results_dir / f"fig_regret_{regime}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)
    return outs


def _plot_scaling(plt, results_dir: Path, manifest: dict) -> list:
    """THE money figure: log(final regret) vs log(T), with fitted slopes.
    CD-TS slope ≲ 2/3 (sublinear, validates Theorem 1); static-optimal ≈ 1."""
    import math
    rows = []
    with (results_dir / "scaling.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append((r["regime"], int(r["T"]), r["policy"],
                         float(r["final_regret"])))
    regimes = sorted({r[0] for r in rows})
    label, is_real = _engine_provenance(manifest)
    colors = {"cd_ts": "tab:blue", "static_optimal": "tab:red",
              "round_robin": "tab:gray"}
    outs = []
    for regime in regimes:
        fig, ax = plt.subplots(figsize=(7, 5))
        for policy in ("cd_ts", "static_optimal", "round_robin"):
            pts = sorted((T, v) for rg, T, p, v in rows if rg == regime and p == policy)
            if not pts:
                continue
            Ts = [T for T, _ in pts]
            Rs = [v for _, v in pts]
            ax.plot(Ts, Rs, "o-", color=colors.get(policy), linewidth=2,
                    label=policy)
            # Fitted slope annotation.
            lx = [math.log(T) for T in Ts]
            ly = [math.log(max(v, 1e-9)) for v in Rs]
            mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
            sxx = sum((x - mx) ** 2 for x in lx)
            if sxx > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
                ax.annotate(f"{policy}: slope≈{slope:.2f}",
                            (Ts[-1], Rs[-1]), textcoords="offset points",
                            xytext=(-10, 8), fontsize=8,
                            color=colors.get(policy))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Horizon T (log)")
        ax.set_ylabel("Final cumulative regret (log)")
        ax.set_title(f"Regret vs horizon — {regime}  "
                     f"(Theorem 1: CD-TS ≲ 2/3, static ≈ 1)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
        _stamp(fig, label, is_real)
        out = results_dir / f"fig_scaling_{regime}.pdf"
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        outs.append(out)
    return outs


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="A results/<timestamp>, ablation_, or nonstationary_ dir")
    args = p.parse_args(argv)

    rd = Path(args.results_dir)
    if not (rd / "manifest.json").exists():
        print(f"No manifest.json in {rd}", file=sys.stderr)
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        return 1

    manifest = json.loads((rd / "manifest.json").read_text())
    written = []
    if (rd / "pareto.csv").exists():
        written.append(_plot_pareto(plt, rd, manifest))
    if (rd / "ablation.json").exists():
        written.extend(_plot_ablation(plt, rd, manifest))
    if (rd / "regret_curves.csv").exists():
        written.extend(_plot_regret(plt, rd, manifest))
    if (rd / "scaling.csv").exists():
        written.extend(_plot_scaling(plt, rd, manifest))

    if not written:
        print(f"No plottable artifacts (pareto.csv / ablation.json) in {rd}",
              file=sys.stderr)
        return 1

    label, is_real = _engine_provenance(manifest)
    for w in written:
        print(f"Wrote: {w}")
    if not is_real:
        print(f"\n⚠️  Figures watermarked SIMULATION — engine source: {label}")
        print("   These are framework-validation figures, NOT paper-eligible "
              "empirical results. Re-run with real engines for the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
