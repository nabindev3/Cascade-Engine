"""
Router Benchmark Framework — compares routing policies on a Pareto-frontier basis.

Methodology:

- **Quality** is computed by a reward model on (prompt, routed_response), giving
  an absolute scalar in [0, 1]. This is independent of the router's self-reported
  confidence and so doesn't suffer from the circularity of "engine scores itself."
  Reward-model-as-judge is the standard in MT-Bench, RewardBench, AlpacaEval.

- **Cost** is the dollar cost of the routed call(s). Higher tiers cost more.

- Cost and Quality are reported as independent axes. Over-provisioning shows up
  as high cost without proportional quality gain. The Pareto frontier identifies
  the non-dominated set of routers.

- Multi-seed evaluation reports mean ± 95% CI using the t-distribution
  (t_{0.025, n-1}) and sample standard deviation (ddof=1). For small n (≤30),
  the normal-approximation z=1.96 understates the interval and is not used.

- `always_premium` baseline routes directly to the top tier (no escalation
  through lower tiers paying along the way). This is a true cost-upper-bound.
"""

import asyncio
import copy
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from ..engines.base import BaseEngine, InferenceRequest, InferenceResponse
from .cascade_router import CascadeRouter, RouterConfig
from .learned_router import (
    ThompsonSamplingRouter, ThompsonConfig, MDPRouter, MDPConfig,
    LinTSRouter, LinTSConfig,
)
from .baseline_routers import FrugalGPTRouter, RouteLLMRouter, BaselineConfig, _RewardModelScorer
from .orchestration_wrapper import OrchestrationWrapper
from .intelligent_layers import IntelligentOrchestrator


@dataclass
class BenchmarkResult:
    """Results from running one router on the full workload."""
    router_name: str
    n_requests: int = 0
    n_successes: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    tier_distribution: Dict[int, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    quality_score: float = 0.0  # Mean reward-model score on routed responses

    @property
    def success_rate(self) -> float:
        return self.n_successes / max(self.n_requests, 1)

    @property
    def avg_cost(self) -> float:
        return self.total_cost_usd / max(self.n_requests, 1)

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(self.n_requests, 1)


@dataclass
class WorkloadItem:
    """A single request in the benchmark workload."""
    request: InferenceRequest
    reference_response: Optional[str] = None  # Optional gold response (e.g., from alpaca_eval `output`)
    # Deprecated; retained for back-compat with older notebooks. Do NOT use for new metrics.
    ground_truth: Optional[str] = None
    optimal_tier: Optional[int] = None


class QualityScorer:
    """Wraps a reward model to give (prompt, response) → quality ∈ [0, 1].

    Memoizes results: identical (prompt, response) pairs are scored once per
    benchmark instance. The reward model is deterministic in inference mode,
    so caching is correct.
    """

    def __init__(self, scorer: Optional[_RewardModelScorer] = None):
        self._scorer = scorer or _RewardModelScorer()
        self._cache: Dict[Tuple[str, str], float] = {}

    def score(self, prompt: str, response: str) -> float:
        if not response:
            return 0.0
        key = (prompt, response)
        if key not in self._cache:
            self._cache[key] = self._scorer.score(prompt, response)
        return self._cache[key]


def pareto_frontier(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """Return the non-dominated subset of (name, cost, quality) points.

    A point is dominated if some other point has cost ≤ AND quality ≥, with
    strict inequality in at least one dimension. The frontier is the set of
    non-dominated points, returned sorted by cost ascending.
    """
    # Sort by cost asc, then by quality desc so that for equal-cost points
    # the higher-quality one wins.
    points_sorted = sorted(points, key=lambda p: (p[1], -p[2]))
    frontier: List[Tuple[str, float, float]] = []
    best_quality = -math.inf
    for name, cost, quality in points_sorted:
        if quality > best_quality + 1e-12:
            frontier.append((name, cost, quality))
            best_quality = quality
    return frontier


def _t_critical(n: int, alpha: float = 0.05) -> float:
    """Two-sided t critical value at confidence 1-alpha for n samples.

    Uses scipy when available; otherwise falls back to a small lookup table
    for common n. For n ≥ 30, the difference from z=1.96 is negligible.
    """
    df = max(n - 1, 1)
    try:
        from scipy.stats import t
        return float(t.ppf(1 - alpha / 2, df))
    except ImportError:
        # Fallback table for two-sided 95% (alpha=0.05).
        table_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
        if df in table_95:
            return table_95[df]
        # Linear interp / fallback to z.
        return 1.96 if df >= 30 else table_95[min(table_95.keys(), key=lambda k: abs(k - df))]


@dataclass
class AggregateStats:
    """Mean and 95% CI for one metric across seeds."""
    mean: float
    ci: float
    values: List[float] = field(default_factory=list)


class RouterBenchmark:
    """Run multiple routing strategies on the same workload and compare."""

    def __init__(
        self,
        engines: List[BaseEngine],
        workload: Optional[List[WorkloadItem]] = None,
        quality_scorer: Optional[QualityScorer] = None,
    ):
        self.engines = engines
        self.workload = workload or []
        # Lazy default — only construct when actually needed (model load is heavy).
        self._quality_scorer = quality_scorer

    @property
    def quality_scorer(self) -> QualityScorer:
        if self._quality_scorer is None:
            self._quality_scorer = QualityScorer()
        return self._quality_scorer

    def generate_synthetic_workload(self, n_requests: int = 500, complexity_distribution: str = "mixed") -> List[WorkloadItem]:
        """Synthetic workload — kept for unit testing only. For paper results,
        use `data_loader.load_prompt_workload` instead."""
        if complexity_distribution == "simple":
            probs = [0.7, 0.2, 0.1]
        elif complexity_distribution == "complex":
            probs = [0.1, 0.3, 0.6]
        else:
            probs = [0.4, 0.35, 0.25]

        simple_prompts = [
            "What is 2+2?", "Hello", "Translate 'yes' to Spanish",
            "Is this a question?", "What color is the sky?",
        ]
        medium_prompts = [
            "Summarize the key differences between Python and JavaScript for a beginner.",
            "Classify the sentiment: 'This is the worst hotel I've stayed in.'",
            "Extract sentiment, issues, and urgency (1-5) from the following review.",
        ]
        hard_prompts = [
            "Analyze this 500-word complaint and generate a personalized recovery response with appropriate compensation.",
            "Parse sarcasm vs genuine praise: 'Oh wonderful, the AC worked every other hour.' Generate a management response.",
            "Given Q4 metrics (occupancy 78%, ADR $245, RevPAR $191, NPS 42), generate a strategic analysis with three recommendations.",
        ]
        workload = []
        for i in range(n_requests):
            level = int(np.random.choice([0, 1, 2], p=probs))
            prompt = np.random.choice([simple_prompts, medium_prompts, hard_prompts][level])
            workload.append(WorkloadItem(
                request=InferenceRequest(
                    request_id=f"bench-{i:04d}",
                    prompt=str(prompt),
                    task_type=["classification", "generation", "extraction"][level],
                ),
            ))
        self.workload = workload
        return workload

    def _build_routers(self, system_mode: bool) -> Dict[str, object]:
        """Construct one fresh router per strategy. Each gets its own
        OrchestrationWrapper instance when system_mode is on."""
        def wrap(router: Any) -> Any:
            return OrchestrationWrapper(router, IntelligentOrchestrator()) if system_mode else router

        max_tier = max(e.tier for e in self.engines)
        top_tier_engines = [e for e in self.engines if e.tier == max_tier]
        bottom_tier_engines = [e for e in self.engines if e.tier == min(e.tier for e in self.engines)]

        routers: Dict[str, object] = {
            "static_cascade": wrap(CascadeRouter(
                engines=self.engines,
                config=RouterConfig(confidence_thresholds={1: 0.65, 2: 0.80}),
            )),
            "thompson_sampling": wrap(ThompsonSamplingRouter(engines=self.engines, config=ThompsonConfig())),
            "lints": wrap(LinTSRouter(engines=self.engines, config=LinTSConfig())),
            "mdp_qlearning": wrap(MDPRouter(engines=self.engines, config=MDPConfig())),
            # True always-premium: only the top-tier engines are available, so the
            # router routes directly there with no cascade-up cost.
            "always_premium": wrap(CascadeRouter(engines=top_tier_engines, config=RouterConfig())),
        }
        # Baselines may fail to construct if their heavy deps (transformers, routellm,
        # downloaded checkpoints) are missing. Skip with a loud warning rather than
        # crashing the entire benchmark — but never silently substitute a mock.
        try:
            routers["frugal_gpt"] = wrap(FrugalGPTRouter(engines=self.engines))
        except Exception as e:
            warnings.warn(f"Skipping FrugalGPT baseline — could not construct: {e}", RuntimeWarning)
        try:
            routers["routellm"] = wrap(RouteLLMRouter(engines=self.engines))
        except Exception as e:
            warnings.warn(f"Skipping RouteLLM baseline — could not construct: {e}", RuntimeWarning)

        if bottom_tier_engines and bottom_tier_engines != self.engines:
            routers["always_local"] = wrap(CascadeRouter(engines=bottom_tier_engines, config=RouterConfig()))
        return routers

    async def run_all(self, system_mode: bool = False) -> Dict[str, BenchmarkResult]:
        """Run every router on the current workload. See `run_experiment` for
        multi-seed statistical evaluation."""
        results: Dict[str, BenchmarkResult] = {}
        for name, router in self._build_routers(system_mode).items():
            results[name] = await self._run_router(router, name)
        return results

    async def _run_router(self, router: Any, name: str) -> BenchmarkResult:
        result = BenchmarkResult(router_name=name)
        for item in self.workload:
            # Each router sees a deep-copied request to avoid mutations leaking.
            req = copy.deepcopy(item.request)
            response, decision = await router.route(req)
            result.n_requests += 1
            if decision.success:
                result.n_successes += 1
            result.total_cost_usd += decision.total_cost_usd
            result.total_latency_ms += decision.total_latency_ms
            tier = decision.final_tier or 0
            result.tier_distribution[tier] = result.tier_distribution.get(tier, 0) + 1
            result.avg_confidence += response.confidence

            # Quality = reward model on the routed (prompt, response).
            # We score against the post-mask prompt the router actually saw so
            # the scorer sees the same input the engine did.
            quality = self.quality_scorer.score(req.prompt, response.content or "")
            result.quality_score += quality

        n = max(result.n_requests, 1)
        result.avg_confidence /= n
        result.quality_score /= n
        return result

    async def run_experiment(
        self,
        n_seeds: int = 5,
        system_mode: bool = False,
    ) -> Dict[str, Dict[str, AggregateStats]]:
        """Run benchmarks across n_seeds different workload orderings and
        aggregate. Returns per-router AggregateStats for cost / latency /
        quality / success_rate."""
        per_seed: Dict[str, Dict[str, List[float]]] = {}

        original_workload = self.workload
        try:
            for seed in range(n_seeds):
                random.seed(seed)
                np.random.seed(seed)
                # Seed torch too — reward model + sentence-transformers + the
                # routellm BERT predictor all use torch under the hood. Inference
                # is deterministic with no_grad, but kernel selection and dropout
                # initialization can introduce subtle variation otherwise.
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    pass
                shuffled = list(original_workload)
                random.shuffle(shuffled)
                self.workload = shuffled

                results = await self.run_all(system_mode=system_mode)
                for name, res in results.items():
                    bucket = per_seed.setdefault(name, {"cost": [], "latency": [], "quality": [], "success": []})
                    bucket["cost"].append(res.avg_cost)
                    bucket["latency"].append(res.avg_latency)
                    bucket["quality"].append(res.quality_score)
                    bucket["success"].append(res.success_rate)
        finally:
            self.workload = original_workload

        return self._aggregate_per_seed(per_seed, n_seeds)

    @staticmethod
    def _aggregate_per_seed(
        per_seed: Dict[str, Dict[str, List[float]]],
        n_seeds: int,
    ) -> Dict[str, Dict[str, AggregateStats]]:
        """Mean + t-distribution 95% CI (ddof=1) per (router, metric).

        Centralized so every code path — run_experiment, ablations — uses the
        identical statistical treatment.
        """
        t_crit = _t_critical(n_seeds)
        final_stats: Dict[str, Dict[str, AggregateStats]] = {}
        for name, metrics in per_seed.items():
            final_stats[name] = {}
            for metric, values in metrics.items():
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                ci = t_crit * std / math.sqrt(max(len(arr), 1))
                final_stats[name][metric] = AggregateStats(mean=mean, ci=ci, values=values)
        return final_stats

    async def run_single_router_experiment(
        self,
        router_factory: Callable[[], Any],
        name: str,
        n_seeds: int = 5,
    ) -> Dict[str, AggregateStats]:
        """Multi-seed evaluation of ONE router built fresh each seed.

        `router_factory()` must return a new router (or wrapped router) on every
        call so learned state never leaks across seeds. Used by the ablation
        runner to sweep a hyperparameter while holding everything else fixed.
        Returns the same AggregateStats shape as `run_experiment`, keyed by
        `name`.
        """
        per_seed: Dict[str, Dict[str, List[float]]] = {
            name: {"cost": [], "latency": [], "quality": [], "success": []}
        }
        original_workload = self.workload
        try:
            for seed in range(n_seeds):
                random.seed(seed)
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    pass
                shuffled = list(original_workload)
                random.shuffle(shuffled)
                self.workload = shuffled

                router = router_factory()
                res = await self._run_router(router, name)
                per_seed[name]["cost"].append(res.avg_cost)
                per_seed[name]["latency"].append(res.avg_latency)
                per_seed[name]["quality"].append(res.quality_score)
                per_seed[name]["success"].append(res.success_rate)
        finally:
            self.workload = original_workload
        return self._aggregate_per_seed(per_seed, n_seeds)[name]

    def compute_pareto_frontier(
        self,
        final_stats: Dict[str, Dict[str, AggregateStats]],
    ) -> List[Tuple[str, float, float]]:
        """Return the cost-vs-quality Pareto frontier from aggregated stats."""
        points = [
            (name, stats["cost"].mean, stats["quality"].mean)
            for name, stats in final_stats.items()
        ]
        return pareto_frontier(points)

    def report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Single-run report (one seed)."""
        lines = [
            "=" * 78,
            f"ROUTER BENCHMARK — single run, {len(self.workload)} prompts",
            "=" * 78,
            f"{'router':<22} {'success':>8} {'cost':>10} {'latency':>10} {'quality':>10}",
            "-" * 78,
        ]
        for name, r in sorted(results.items(), key=lambda kv: kv[1].avg_cost):
            lines.append(
                f"{name:<22} {r.success_rate:>7.1%} ${r.avg_cost:>9.5f} "
                f"{r.avg_latency:>9.0f}ms {r.quality_score:>10.4f}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    def report_experiment(self, final_stats: Dict[str, Dict[str, AggregateStats]]) -> str:
        """Multi-seed report with 95% CI (t-distribution) and Pareto frontier."""
        n_seeds = max(
            (len(s["cost"].values) for s in final_stats.values()),
            default=0,
        )
        lines = [
            "=" * 88,
            f"MULTI-SEED EXPERIMENT — n={n_seeds} seeds, 95% CI via t-distribution",
            "=" * 88,
            f"{'router':<22} {'cost ± CI':>20} {'quality ± CI':>22} {'latency ± CI':>22}",
            "-" * 88,
        ]
        for name, s in sorted(final_stats.items(), key=lambda kv: kv[1]["cost"].mean):
            lines.append(
                f"{name:<22} "
                f"${s['cost'].mean:>8.5f} ± {s['cost'].ci:<8.5f} "
                f"{s['quality'].mean:>10.4f} ± {s['quality'].ci:<8.4f} "
                f"{s['latency'].mean:>9.0f} ± {s['latency'].ci:<6.0f}ms"
            )

        frontier = self.compute_pareto_frontier(final_stats)
        frontier_names = {p[0] for p in frontier}
        lines.append("-" * 88)
        lines.append("Pareto frontier (non-dominated in cost × quality):")
        for name, cost, quality in frontier:
            lines.append(f"  • {name:<22}  cost=${cost:.5f}  quality={quality:.4f}")
        dominated = sorted(set(final_stats) - frontier_names)
        if dominated:
            lines.append("Dominated:")
            for name in dominated:
                lines.append(f"  · {name}")
        lines.append("=" * 88)
        return "\n".join(lines)
