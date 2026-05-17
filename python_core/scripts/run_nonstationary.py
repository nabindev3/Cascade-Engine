"""CLI for the decisive non-stationary regret experiment.

Produces the numbers and figures that distinguish this work from the
offline-optimal paradigm of Dekoninck et al. (ICML 2025).

Two complementary outputs:

  1. HORIZON SCALING (the Theorem-1 validation). Variation budget V_T held
     fixed; horizon T swept; γ re-tuned per Theorem 1 at each horizon; fit
     log(final regret) vs log(T). CD-TS exponent ≲ 2/3 (sublinear); the frozen
     offline-optimal policy ≈ 1 (linear). This is the correct way to validate
     an O(T^{2/3}) bound — NOT a within-run slope.

  2. SINGLE-HORIZON regret curves (the intuition figure). Shows CD-TS recovers
     after a drift event while static-optimal stays stuck.

Usage (repo root):

    python -m python_core.scripts.run_nonstationary \\
        --regime periodic --seeds 8 --label "paper-nonstationary"

Regimes: abrupt | drift | periodic | all
Outputs under results/nonstationary_{timestamp}/:
    manifest.json         — provenance + V_T + scaling exponents
    scaling.csv           — regime, T, policy, final_regret, V_T, gamma
    regret_curves.csv     — regime, t, policy, mean_cumulative_regret (one T)
    summary.txt           — scaling exponents, degradation, recovery
    (figures via make_figures.py)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..router.nonstationary import (
    NonStationaryEnv,
    abrupt,
    linear_drift,
    periodic,
    run_horizon_scaling,
    run_single_horizon,
)
from .run_experiment import (
    PINNED_PACKAGES,
    _git_dirty,
    _git_sha,
    _package_versions,
)


def _env_at(regime: str):
    """Return env_factory_at(T): builds a horizon-T env whose variation budget
    V_T is (approximately) INDEPENDENT of T — required for the scaling test to
    be meaningful. We do this by fixing the number of change events (abrupt:
    one breakpoint; drift: one monotone sweep; periodic: a fixed period
    *count*, so oscillations don't multiply with T)."""
    if regime == "abrupt":
        def at(T):
            return NonStationaryEnv([
                abrupt(0.45, 0.45),
                linear_drift(0.58, 0.60),
                abrupt(0.85, 0.25, at_frac=0.5),  # one breakpoint ⇒ V_T≈const
            ], T)
    elif regime == "drift":
        def at(T):
            return NonStationaryEnv([
                linear_drift(0.45, 0.46),
                linear_drift(0.55, 0.75),
                linear_drift(0.85, 0.40),         # total variation ⇒ V_T≈const
            ], T)
    elif regime == "periodic":
        def at(T):
            return NonStationaryEnv([
                periodic(0.50, 0.08, periods=3),
                periodic(0.60, 0.22, periods=3),  # FIXED period count ⇒ V_T≈const
                periodic(0.70, 0.18, periods=3),
            ], T)
    else:
        raise ValueError(regime)
    return at


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Non-stationary regret experiment.")
    p.add_argument("--regime", choices=["abrupt", "drift", "periodic", "all"],
                   default="periodic")
    p.add_argument("--horizons", default="2000,4000,8000,16000,32000",
                   help="Comma-separated horizons for the scaling fit.")
    p.add_argument("--curve-T", type=int, default=8000,
                   help="Horizon for the single-run regret-curve figure.")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)

    regimes = ["abrupt", "drift", "periodic"] if args.regime == "all" else [args.regime]
    horizons = [int(x) for x in args.horizons.split(",")]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"nonstationary_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_rows = ["regime,T,policy,final_regret,V_T,gamma"]
    curve_rows = ["regime,t,policy,mean_cumulative_regret"]
    summary = ["NON-STATIONARY REGRET EXPERIMENT", "=" * 74]
    results = {}

    for regime in regimes:
        env_at = _env_at(regime)

        scal = run_horizon_scaling(env_at, horizons, args.seeds, args.calib_frac)
        curve = run_single_horizon(lambda: env_at(args.curve_T),
                                   T=args.curve_T, n_seeds=args.seeds,
                                   cdts_gamma=None, calib_frac=args.calib_frac)
        results[regime] = {"scaling": scal,
                           "single_horizon": {k: v for k, v in curve.items()
                                              if k != "avg_regret_curves"}}

        for T in horizons:
            ph = scal["per_horizon"][T]
            for pol in ("cd_ts", "static_optimal", "round_robin"):
                scaling_rows.append(
                    f"{regime},{T},{pol},{ph['final_regret'][pol]['mean']:.4f},"
                    f"{ph['variation_budget_V_T']:.4f},{ph['cdts_gamma']:.6f}")
        for pol, c in curve["avg_regret_curves"].items():
            step = max(1, len(c) // 400)
            for t in range(0, len(c), step):
                curve_rows.append(f"{regime},{t},{pol},{c[t]:.6f}")

        sx = scal["scaling_exponent"]
        summary += [
            f"\n[regime: {regime}]  horizons={horizons}  seeds={args.seeds}",
            f"  V_T held fixed: [{scal['v_budget_min']:.3f}, {scal['v_budget_max']:.3f}] "
            f"(relative drift {scal['v_budget_relative_drift']*100:.1f}% — must be ~0)",
            f"  Theorem-1 predicted CD-TS exponent: ≤ {scal['theorem1_predicted_exponent']:.3f}",
            f"  CD-TS   regret-vs-horizon exponent : {sx['cd_ts']['exponent']:.3f} "
            f"(R²={sx['cd_ts']['r2']:.3f})   ← sublinear",
            f"  static  regret-vs-horizon exponent : {sx['static_optimal']['exponent']:.3f} "
            f"(R²={sx['static_optimal']['r2']:.3f})   ← ≈ linear",
            f"  degradation (static/CD-TS) at largest T: "
            f"{scal['per_horizon'][horizons[-1]]['static_over_cdts_degradation']:.2f}x",
            f"  recovery (tail per-round regret @ T={args.curve_T}): "
            f"CD-TS={curve['tail_regret_rate']['cd_ts']['mean']:.4f}  "
            f"static={curve['tail_regret_rate']['static_optimal']['mean']:.4f}",
        ]

    manifest = {
        "timestamp_utc": ts, "label": args.label, "kind": "nonstationary_regret",
        "git_sha": _git_sha(), "git_dirty": _git_dirty(),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(PINNED_PACKAGES),
        "engines": {"label": "CONTROLLED-REWARD non-stationary bandit "
                             "(known drift schedule; required for exact regret). "
                             "Complementary to the reward-model real-prompt experiment."},
        "experiment": {"regimes": regimes, "horizons": horizons,
                       "curve_T": args.curve_T, "n_seeds": args.seeds,
                       "calib_frac": args.calib_frac},
        "results": results,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "scaling.csv").write_text("\n".join(scaling_rows) + "\n")
    (out_dir / "regret_curves.csv").write_text("\n".join(curve_rows) + "\n")
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print(f"\nArtifacts written to: {out_dir}")
    print(f"Figures:  python -m python_core.scripts.make_figures {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
