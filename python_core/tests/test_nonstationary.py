"""Tests for the decisive non-stationary regret experiment.

The central test, `test_horizon_scaling_validates_theorem1`, encodes the
paper's thesis as the scientifically-correct statistic: with V_T held fixed
and γ tuned per Theorem 1, CD-TS final regret scales SUBLINEARLY with the
horizon (exponent well below 1, consistent with the O(T^{2/3}) upper bound),
while the frozen offline-optimal policy scales ≈ linearly. If that inverts,
the thesis is false and this test fails loudly.
"""

import json
import math

import pytest

from python_core.router.nonstationary import (
    NonStationaryEnv,
    abrupt,
    fit_regret_exponent,
    linear_drift,
    loglog_slope,
    periodic,
    run_horizon_scaling,
    run_single_horizon,
    theory_gamma,
)


# ─────────────────────────────────────────────────────────────────────────────
# Drift schedules + variation budget
# ─────────────────────────────────────────────────────────────────────────────


def test_abrupt_variation_budget_is_single_breakpoint():
    assert abs(abrupt(0.9, 0.3, at_frac=0.5).variation_budget(1000) - 0.6) < 1e-9


def test_linear_drift_variation_budget_is_total_change():
    assert abs(linear_drift(0.2, 0.8).variation_budget(1000) - 0.6) < 1e-2


def test_periodic_fixed_period_count_keeps_V_T_horizon_independent():
    """The scaling test REQUIRES V_T independent of T. Fixed `periods` count
    must give ~constant variation across horizons."""
    s = periodic(0.5, 0.2, periods=3)
    v_small = s.variation_budget(2000)
    v_large = s.variation_budget(32000)
    # Within a few percent — the discrete-sum approximation of a fixed number
    # of sinusoid periods.
    assert abs(v_large - v_small) / v_small < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Log-log slope estimator
# ─────────────────────────────────────────────────────────────────────────────


def test_loglog_slope_recovers_known_exponents():
    xs = [math.log(t) for t in range(1, 50)]
    for true_b in (0.5, 2 / 3, 1.0):
        ys = [true_b * x + 0.3 for x in xs]
        b, r2 = loglog_slope(xs, ys)
        assert abs(b - true_b) < 1e-6
        assert r2 > 0.999


def test_theory_gamma_decreases_window_as_horizon_grows():
    """1-γ should shrink as T grows (longer effective memory window)."""
    g_small = theory_gamma(1000, V_T=5.0)
    g_large = theory_gamma(64000, V_T=5.0)
    assert 0 < (1 - g_large) < (1 - g_small) <= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Single-horizon behavior: CD-TS recovers, static-optimal does not
# ─────────────────────────────────────────────────────────────────────────────


def test_cdts_recovers_after_abrupt_drift_static_does_not():
    T = 4000

    def factory():
        return NonStationaryEnv([
            abrupt(0.45, 0.45),
            linear_drift(0.58, 0.60),
            abrupt(0.85, 0.25, at_frac=0.5),  # best, then collapses
        ], T)

    res = run_single_horizon(factory, T=T, n_seeds=4, cdts_gamma=None, calib_frac=0.2)
    cd_tail = res["tail_regret_rate"]["cd_ts"]["mean"]
    st_tail = res["tail_regret_rate"]["static_optimal"]["mean"]
    # After the collapse, CD-TS's per-round regret in the final 10% must be
    # well below the frozen policy's (CD-TS recovered; static stayed stuck).
    assert cd_tail < st_tail
    assert res["static_over_cdts_degradation"] > 1.3
    assert res["final_regret"]["static_optimal"]["mean"] > res["final_regret"]["cd_ts"]["mean"]


# ─────────────────────────────────────────────────────────────────────────────
# THE central claim: Theorem-1 horizon scaling
# ─────────────────────────────────────────────────────────────────────────────


def test_horizon_scaling_validates_theorem1():
    """With V_T fixed and γ tuned per Theorem 1, CD-TS final regret scales
    sublinearly in T (exponent comfortably < 1, consistent with the ≤2/3
    upper bound), while the frozen offline-optimal scales ≈ linearly."""

    def env_at(T):
        # FIXED period count ⇒ V_T independent of T (required for the test).
        return NonStationaryEnv([
            periodic(0.50, 0.08, periods=3),
            periodic(0.60, 0.22, periods=3),
            periodic(0.70, 0.18, periods=3),
        ], T)

    res = run_horizon_scaling(
        env_factory_at=env_at,
        horizons=[2000, 4000, 8000, 16000],
        n_seeds=4,
        calib_frac=0.2,
    )

    # V_T must actually be (near-)constant across horizons or the test is moot.
    assert res["v_budget_relative_drift"] < 0.05, (
        f"V_T drifted {res['v_budget_relative_drift']:.2%} across horizons — "
        f"the scaling comparison is information-theoretically meaningless."
    )

    cd = res["scaling_exponent"]["cd_ts"]["exponent"]
    st = res["scaling_exponent"]["static_optimal"]["exponent"]

    # CD-TS: sublinear, consistent with the O(T^{2/3}) upper bound (allow
    # slack up to ~0.85 for finite-horizon constants and Bernoulli noise).
    assert cd < 0.85, f"CD-TS scaling exponent {cd:.3f} not sublinear"
    # Static-optimal: ≈ linear (frozen policy ⇒ regret ∝ T).
    assert st > 0.80, f"static exponent {st:.3f} should be ≈ linear"
    # And CD-TS must scale strictly better than static.
    assert cd < st - 0.05, (
        f"CD-TS exponent {cd:.3f} must be clearly below static {st:.3f} — "
        f"this gap IS the paper's contribution."
    )


def test_run_single_horizon_output_shape():
    env = lambda: NonStationaryEnv([abrupt(0.7, 0.3), linear_drift(0.4, 0.6)], 500)
    res = run_single_horizon(env, T=500, n_seeds=3)
    for key in ("variation_budget_V_T", "final_regret", "tail_regret_rate",
                "static_over_cdts_degradation", "avg_regret_curves", "cdts_gamma"):
        assert key in res
    for p in ("cd_ts", "static_optimal", "round_robin"):
        assert len(res["avg_regret_curves"][p]) == 500


def test_within_run_exponent_is_documented_diagnostic_only():
    """fit_regret_exponent must still work (it's used for run-shape
    diagnostics) but is explicitly NOT the Theorem-1 statistic."""
    cum = [0.5 * t for t in range(1, 1000)]
    b, r2 = fit_regret_exponent(cum)
    assert abs(b - 1.0) < 0.05  # linear within-run ⇒ slope 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────────────


def test_run_nonstationary_cli_smoke(tmp_path):
    from python_core.scripts import run_nonstationary as runner

    rc = runner.main([
        "--regime", "periodic",
        "--horizons", "1000,2000",
        "--curve-T", "1500",
        "--seeds", "2",
        "--output-dir", str(tmp_path),
        "--label", "ns-smoke",
    ])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    rd = runs[0]
    for name in ("manifest.json", "scaling.csv", "regret_curves.csv", "summary.txt"):
        assert (rd / name).exists(), f"missing {name}"

    manifest = json.loads((rd / "manifest.json").read_text())
    assert manifest["kind"] == "nonstationary_regret"
    assert "periodic" in manifest["results"]
    sx = manifest["results"]["periodic"]["scaling"]["scaling_exponent"]
    assert "cd_ts" in sx and "static_optimal" in sx

    assert (rd / "scaling.csv").read_text().splitlines()[0] == \
        "regime,T,policy,final_regret,V_T,gamma"
    assert (rd / "regret_curves.csv").read_text().splitlines()[0] == \
        "regime,t,policy,mean_cumulative_regret"
