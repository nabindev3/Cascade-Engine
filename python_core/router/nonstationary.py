"""Non-stationary regret experiment — the paper's decisive experiment.

Why this file exists
--------------------
Dekoninck, Baader, Vechev (ICML 2025) prove the *per-query optimal* routing /
cascading rule **given a fixed quality estimator and a stationary query
distribution**. Their framework cannot, by construction, react when engine
reliability drifts (model deprecation, time-of-day API degradation, load).
Our contribution is precisely that regime. This module produces the experiment
that demonstrates it:

  * a controlled non-stationary reward environment with a KNOWN drift schedule,
    so pseudo-regret against the dynamic oracle is *exactly computable* (this
    is the standard and necessary methodology in the non-stationary bandit
    literature — you cannot measure regret without knowing the true optimum);
  * the real CD-TS core (`learned_router.ArmStats`: Beta posterior + Bernoulli
    trick + per-played-arm geometric discounting) — i.e. Algorithm 1 of
    `paper/theory.tex`, the exact object Theorem 1 bounds;
  * a faithful re-implementation of the *offline-optimal paradigm*
    (calibrate on a stationary prefix, commit, freeze) as the comparator —
    this is the failure mode their paradigm has under drift;
  * the dynamic oracle and round-robin as references;
  * the correct empirical check of the Theorem 1 exponent: a CROSS-HORIZON
    scaling experiment. We hold the variation budget V_T fixed (fixed number
    of change events), sweep the horizon T, re-tune γ per Theorem 1 at each
    horizon, and fit log(final_regret) vs log(T). CD-TS scales with exponent
    ≲ 2/3 (sublinear; the bound is an upper bound) while the frozen
    offline-optimal policy scales ≈ T^1. NOTE: the *within-run* cumulative-
    regret slope is NOT a valid check — a discounted bandit at its optimal
    constant per-round rate has within-run slope ≈ 1 irrespective of the
    horizon-scaling exponent; and with V_T ∝ T, linear regret is
    information-theoretically optimal (Besbes et al. 2014).

This is a *controlled-reward* experiment by necessity; it is complementary to
the reward-model-scored real-prompt experiment in `run_experiment.py`. One
validates adaptation/regret, the other validates quality on real prompts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .learned_router import ArmStats


# ─────────────────────────────────────────────────────────────────────────────
# Drift schedules.  mean_reward(t) ∈ [0, 1] is the TRUE expected reward of an
# engine at round t.  Each schedule also reports its exact variation budget
# V = Σ_t |μ(t) − μ(t−1)| so we can compare measured regret to the
# (MK)^{1/3} V^{1/3} T^{2/3} prediction of Theorem 1.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DriftSchedule:
    name: str
    mean_at: Callable[[int, int], float]   # (t, T) -> mean reward in [0,1]

    def variation_budget(self, T: int) -> float:
        return sum(
            abs(self.mean_at(t, T) - self.mean_at(t - 1, T)) for t in range(1, T)
        )


def abrupt(base: float, drop_to: float, at_frac: float = 0.5) -> DriftSchedule:
    """Healthy at `base`, then a one-shot collapse to `drop_to` (a model
    deprecation / outage). Variation budget ≈ |base - drop_to| (one breakpoint)."""
    def f(t: int, T: int) -> float:
        return base if t < int(at_frac * T) else drop_to
    return DriftSchedule(f"abrupt({base}->{drop_to})", f)


def linear_drift(start: float, end: float) -> DriftSchedule:
    """Slow monotone decay (gradual quality regression). Variation ≈ |end-start|."""
    def f(t: int, T: int) -> float:
        frac = t / max(T - 1, 1)
        return start + (end - start) * frac
    return DriftSchedule(f"drift({start}->{end})", f)


def periodic(center: float, amp: float, periods: float = 4.0) -> DriftSchedule:
    """Sinusoidal (time-of-day load). Variation grows with `periods` — the
    high-V_T regime where Theorem 1's V_T^{1/3} term dominates."""
    def f(t: int, T: int) -> float:
        v = center + amp * math.sin(2 * math.pi * periods * t / max(T, 1))
        return min(1.0, max(0.0, v))
    return DriftSchedule(f"periodic(c={center},a={amp},p={periods})", f)


# ─────────────────────────────────────────────────────────────────────────────
# Environment: K engines, each with its own drift schedule.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NonStationaryEnv:
    schedules: List[DriftSchedule]
    T: int

    @property
    def K(self) -> int:
        return len(self.schedules)

    def means(self, t: int) -> List[float]:
        return [s.mean_at(t, self.T) for s in self.schedules]

    def pull(self, arm: int, t: int) -> float:
        """Sample a reward in [0,1] (Bernoulli around the true mean)."""
        mu = self.schedules[arm].mean_at(t, self.T)
        return 1.0 if random.random() < mu else 0.0

    def best_mean(self, t: int) -> float:
        return max(self.means(t))

    def total_variation(self) -> float:
        return sum(s.variation_budget(self.T) for s in self.schedules)


# ─────────────────────────────────────────────────────────────────────────────
# Policies.  Each returns the arm index chosen at round t.  Policies that learn
# also receive the observed reward via .update().
# ─────────────────────────────────────────────────────────────────────────────


class Policy:
    name = "policy"

    def select(self, t: int) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:  # pragma: no cover
        pass


class OraclePolicy(Policy):
    """Dynamic optimum: knows the true means each round (zero regret)."""
    name = "oracle"

    def __init__(self, env: NonStationaryEnv):
        self.env = env
        self._t = 0

    def select(self, t: int) -> int:
        means = self.env.means(t)
        return max(range(len(means)), key=lambda i: means[i])


class CDTSPolicy(Policy):
    """Algorithm 1 of paper/theory.tex, driven by the real, unit-tested
    ArmStats (Bernoulli trick + per-played-arm geometric discount)."""
    name = "cd_ts"

    def __init__(self, K: int, gamma: float = 0.98, floor: float = 1.0):
        self.arms = [ArmStats() for _ in range(K)]
        self.gamma = gamma
        self.floor = floor

    def select(self, t: int) -> int:
        samples = [a.sample() for a in self.arms]
        return max(range(len(samples)), key=lambda i: samples[i])

    def update(self, arm: int, reward: float) -> None:
        self.arms[arm].update(reward, decay=self.gamma, floor=self.floor)


class StaticOptimalPolicy(Policy):
    """Re-implementation of the offline-optimal *paradigm* (Dekoninck et al.
    2025): estimate each engine's quality on a stationary calibration prefix,
    pick the cost-constrained optimum, then FREEZE.  This is not their exact
    code; it captures the essential property — a policy fitted offline on a
    stationary sample and committed — and therefore its failure mode under
    drift. Cost is zero in this controlled experiment, so 'cost-constrained
    optimum' reduces to 'best calibrated mean'."""
    name = "static_optimal"

    def __init__(self, K: int, calib_rounds: int):
        self.K = K
        self.calib_rounds = calib_rounds
        self._sum = [0.0] * K
        self._n = [0] * K
        self._frozen_arm = None

    def select(self, t: int) -> int:
        if t < self.calib_rounds:
            return t % self.K  # round-robin exploration during calibration
        if self._frozen_arm is None:
            means = [
                (self._sum[i] / self._n[i]) if self._n[i] else 0.0
                for i in range(self.K)
            ]
            self._frozen_arm = max(range(self.K), key=lambda i: means[i])
        return self._frozen_arm

    def update(self, arm: int, reward: float) -> None:
        if self._frozen_arm is None:  # only learn during calibration
            self._sum[arm] += reward
            self._n[arm] += 1


class RoundRobinPolicy(Policy):
    name = "round_robin"

    def __init__(self, K: int):
        self.K = K

    def select(self, t: int) -> int:
        return t % self.K


# ─────────────────────────────────────────────────────────────────────────────
# Experiment driver.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PolicyRun:
    name: str
    cum_regret: List[float] = field(default_factory=list)

    @property
    def final_regret(self) -> float:
        return self.cum_regret[-1] if self.cum_regret else 0.0


def _run_one_policy(env: NonStationaryEnv, policy: Policy, T: int) -> PolicyRun:
    run = PolicyRun(name=policy.name)
    cum = 0.0
    for t in range(T):
        arm = policy.select(t)
        reward = env.pull(arm, t)
        policy.update(arm, reward)
        means = env.means(t)
        # Pseudo-regret: use TRUE means, not the noisy sample (standard).
        cum += max(means) - means[arm]
        run.cum_regret.append(cum)
    return run


def loglog_slope(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """OLS slope and R² of ys vs xs (both already in log space).

    Returns (slope, r_squared). NaN if degenerate.
    """
    if len(xs) < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, r2


def fit_regret_exponent(cum_regret: List[float], lo_frac: float = 0.5) -> Tuple[float, float]:
    """Within-run log-log slope of cumulative regret over t ∈ [lo·T, T].

    DIAGNOSTIC ONLY. This is NOT the right statistic for validating an
    O(T^{2/3}) bound: a discounted bandit at its optimal *constant* per-round
    regret rate has within-run cumulative-regret slope ≈ 1 regardless of the
    horizon-scaling exponent. The correct Theorem-1 validation is the
    cross-horizon scaling exponent computed by `run_horizon_scaling`. We keep
    this only to characterize a single run's shape (e.g. show static-optimal is
    locally linear post-freeze).
    """
    n = len(cum_regret)
    lo = max(2, int(lo_frac * n))
    xs = [math.log(t + 1) for t in range(lo, n) if cum_regret[t] > 0]
    ys = [math.log(cum_regret[t]) for t in range(lo, n) if cum_regret[t] > 0]
    return loglog_slope(xs, ys)


def theory_gamma(T: int, V_T: float, M: int = 1, K: int = 3) -> float:
    """Discount factor from Theorem 1: 1-γ = (MK log T)^{1/3} (V_T/T)^{2/3},
    clipped to (0, 0.5]. This is the schedule under which CD-TS attains the
    O((MK log T)^{1/3} V_T^{1/3} T^{2/3}) rate. The cross-horizon experiment
    re-tunes γ at every horizon exactly per this formula."""
    inv = (M * K * math.log(max(T, 2))) ** (1 / 3) * (max(V_T, 1.0) / T) ** (2 / 3)
    return 1.0 - min(0.5, max(1e-3, inv))


def _mean_ci(xs: List[float], n_seeds: int) -> Tuple[float, float]:
    from .benchmark import _t_critical
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, _t_critical(n_seeds) * math.sqrt(var) / math.sqrt(len(xs))


def run_single_horizon(
    env_factory: Callable[[], NonStationaryEnv],
    T: int,
    n_seeds: int,
    cdts_gamma: Optional[float] = None,
    calib_frac: float = 0.2,
) -> Dict:
    """One horizon. Produces seed-averaged cumulative-regret curves (for the
    regret figure), final regrets ±CI, and a post-drift *recovery* metric:
    mean per-round regret in the final 10% of rounds. CD-TS should recover
    (small tail rate); the frozen static-optimal should not.

    If `cdts_gamma` is None it is set per Theorem 1 from the env's V_T.
    """
    policy_names = ["cd_ts", "static_optimal", "round_robin"]
    curves = {p: [] for p in policy_names}
    finals = {p: [] for p in policy_names}
    tail_rate = {p: [] for p in policy_names}

    sample_env = env_factory()
    V_T = sample_env.total_variation()
    K = sample_env.K
    M = 1
    gamma = cdts_gamma if cdts_gamma is not None else theory_gamma(T, V_T, M, K)

    for seed in range(n_seeds):
        env = env_factory()
        policies = [
            CDTSPolicy(K, gamma=gamma),
            StaticOptimalPolicy(K, calib_rounds=int(calib_frac * T)),
            RoundRobinPolicy(K),
        ]
        for pol in policies:
            random.seed(7919 * seed + 31 * policy_names.index(pol.name) + 1)
            run = _run_one_policy(env, pol, T)
            curves[pol.name].append(run.cum_regret)
            finals[pol.name].append(run.final_regret)
            tail_n = max(1, T // 10)
            tail = run.cum_regret[-1] - run.cum_regret[-tail_n - 1] if T > tail_n else run.cum_regret[-1]
            tail_rate[pol.name].append(tail / tail_n)

    avg_curves = {}
    for p in policy_names:
        L = min(len(c) for c in curves[p])
        avg_curves[p] = [sum(curves[p][s][t] for s in range(n_seeds)) / n_seeds for t in range(L)]

    fr = {p: _mean_ci(finals[p], n_seeds) for p in policy_names}
    tr = {p: _mean_ci(tail_rate[p], n_seeds) for p in policy_names}
    cdts_f, static_f = fr["cd_ts"][0], fr["static_optimal"][0]

    return {
        "T": T, "n_seeds": n_seeds, "K": K, "M": M,
        "variation_budget_V_T": V_T, "cdts_gamma": gamma,
        "final_regret": {p: {"mean": fr[p][0], "ci": fr[p][1]} for p in policy_names},
        "tail_regret_rate": {p: {"mean": tr[p][0], "ci": tr[p][1]} for p in policy_names},
        "static_over_cdts_degradation": (static_f / cdts_f) if cdts_f > 0 else float("inf"),
        "avg_regret_curves": avg_curves,
    }


def run_horizon_scaling(
    env_factory_at: Callable[[int], NonStationaryEnv],
    horizons: List[int],
    n_seeds: int,
    calib_frac: float = 0.2,
) -> Dict:
    """THE Theorem-1 validation: hold the variation budget V_T (the number of
    change events) fixed while sweeping the horizon T, re-tuning CD-TS's γ per
    Theorem 1 at each horizon, and fit log(final_regret) vs log(T).

    Prediction: CD-TS scaling exponent ≤ 2/3 (sublinear; the bound is an upper
    bound so the realized exponent may be lower), while the frozen
    static-optimal policy scales ≈ T^1 (linear — it cannot track drift, so its
    regret is a fixed per-round gap times T). `env_factory_at(T)` MUST keep V_T
    (approximately) constant across horizons; otherwise the comparison is
    information-theoretically meaningless (Besbes et al. 2014: with V_T ∝ T,
    linear regret is optimal).
    """
    policy_names = ["cd_ts", "static_optimal", "round_robin"]
    per_T: Dict[int, Dict] = {}
    log_T: List[float] = []
    log_R: Dict[str, List[float]] = {p: [] for p in policy_names}
    v_budgets: List[float] = []

    for T in horizons:
        res = run_single_horizon(
            env_factory=lambda T=T: env_factory_at(T),
            T=T, n_seeds=n_seeds, cdts_gamma=None, calib_frac=calib_frac,
        )
        per_T[T] = {
            "variation_budget_V_T": res["variation_budget_V_T"],
            "cdts_gamma": res["cdts_gamma"],
            "final_regret": res["final_regret"],
            "static_over_cdts_degradation": res["static_over_cdts_degradation"],
        }
        v_budgets.append(res["variation_budget_V_T"])
        log_T.append(math.log(T))
        for p in policy_names:
            log_R[p].append(math.log(max(res["final_regret"][p]["mean"], 1e-9)))

    scaling = {}
    for p in policy_names:
        b, r2 = loglog_slope(log_T, log_R[p])
        scaling[p] = {"exponent": b, "r2": r2}

    v_min, v_max = min(v_budgets), max(v_budgets)
    v_drift = (v_max - v_min) / max(v_min, 1e-9)

    return {
        "horizons": horizons,
        "n_seeds": n_seeds,
        "v_budget_min": v_min,
        "v_budget_max": v_max,
        "v_budget_relative_drift": v_drift,  # should be ≈0: V_T held fixed
        "theorem1_predicted_exponent": 2 / 3,
        "scaling_exponent": scaling,         # cd_ts ≲ 2/3 ; static ≈ 1
        "per_horizon": per_T,
    }
