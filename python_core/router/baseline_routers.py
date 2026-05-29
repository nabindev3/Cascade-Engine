"""
Baseline routers for benchmark comparison.

These are real reproductions of the published methods, not heuristic proxies:

- FrugalGPTRouter: implements the LLM-Cascade strategy from Chen et al. (2023).
  The (prompt, response) scorer uses a pretrained reward model
  (`OpenAssistant/reward-model-deberta-v3-large-v2`) as a stand-in for FrugalGPT's
  learned DistilBERT scorer. This is honest: a sigmoid-calibrated preference
  model is the same shape of object FrugalGPT trains, and avoids the unfair
  comparison of a hand-rule scorer vs a learned one.

- RouteLLMRouter: uses the official `routellm` package and the published
  BERT checkpoint (`routellm/bert_gpt4_1106_augmented`) to compute
  `calculate_strong_win_rate(prompt)`, then thresholds to pick a tier.
  No OpenAI-embedding dependency (we deliberately chose the BERT router over MF).

If a dependency is missing the constructor raises ImportError — never silently
fall back to a mock. Mock fallbacks were the original methodological bug.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)
from .cascade_router import RoutingDecision


@dataclass
class BaselineConfig:
    max_cost_per_request: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# FrugalGPT — LLM-Cascade with learned response scorer
# ═══════════════════════════════════════════════════════════════════════════════


class _RewardModelScorer:
    """Wraps a HuggingFace reward model so it scores (prompt, response) ∈ [0, 1]."""

    DEFAULT_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device

    def score(self, prompt: str, response: str) -> float:
        inputs = self.tokenizer(
            prompt, response, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        # Reward model outputs a scalar; squash to [0, 1] via sigmoid.
        return float(self._torch.sigmoid(logits).item())


class FrugalGPTRouter:
    """
    LLM-Cascade strategy from FrugalGPT (Chen et al., 2023).

    Algorithm:
        for engine in (cheapest → most expensive):
            response = engine.infer(prompt)
            score = scorer(prompt, response)
            if score >= threshold or engine is last:
                return response
            else:
                escalate

    The threshold is a calibrated hyperparameter; FrugalGPT learns it via
    constrained optimization. Here we accept it as a config value.
    """

    def __init__(
        self,
        engines: List[BaseEngine],
        config: BaselineConfig = None,
        threshold: float = 0.5,
        scorer: Optional[_RewardModelScorer] = None,
    ):
        self.config = config or BaselineConfig()
        self.engines = sorted(engines, key=lambda e: e.tier)
        self.threshold = threshold
        self.scorer = scorer or _RewardModelScorer()

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        decision = RoutingDecision(request_id=request.request_id)
        start = time.perf_counter()
        cumulative_cost = 0.0
        best_response: Optional[InferenceResponse] = None
        is_escalated = False

        for engine in self.engines:
            if engine.status == EngineStatus.UNAVAILABLE:
                continue
            if request.min_tier and engine.tier < request.min_tier:
                continue
            est = engine.estimated_cost(request)
            if cumulative_cost + est > self.config.max_cost_per_request:
                decision.escalation_reasons.append(f"{engine.engine_id}: budget")
                continue

            decision.tiers_attempted.append(engine.tier)
            decision.engines_tried.append(engine.engine_id)

            response = await engine.infer(request)
            cumulative_cost += response.cost_usd
            response.was_escalated = is_escalated

            if not response.success:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: failed ({response.failure_mode.value})"
                )
                best_response = response
                is_escalated = True
                continue

            score = self.scorer.score(request.prompt, response.content)

            if score >= self.threshold or engine is self.engines[-1]:
                decision.final_engine = engine.engine_id
                decision.final_tier = engine.tier
                decision.success = True
                decision.total_latency_ms = (time.perf_counter() - start) * 1000
                decision.total_cost_usd = cumulative_cost
                return response, decision

            decision.escalation_reasons.append(
                f"{engine.engine_id}: scorer={score:.3f} < {self.threshold:.3f}"
            )
            best_response = response
            is_escalated = True

        decision.total_latency_ms = (time.perf_counter() - start) * 1000
        decision.total_cost_usd = cumulative_cost
        if best_response is None:
            best_response = InferenceResponse(
                request_id=request.request_id, engine_id="none", tier=0,
                content="", success=False,
                failure_mode=FailureMode.INFRASTRUCTURE,
                error_message="FrugalGPT exhausted all engines",
            )
        return best_response, decision


# ═══════════════════════════════════════════════════════════════════════════════
# RouteLLM — preference-based direct routing
# ═══════════════════════════════════════════════════════════════════════════════


class _RouteLLMPredictor:
    """Wraps a routellm Router instance to expose `predict_strong_prob(prompt)`."""

    DEFAULT_ROUTER = "bert"
    DEFAULT_CHECKPOINT = "routellm/bert_gpt4_1106_augmented"

    def __init__(self, router_type: str = DEFAULT_ROUTER, checkpoint: str = DEFAULT_CHECKPOINT):
        from routellm.routers.routers import ROUTER_CLS

        if router_type not in ROUTER_CLS:
            raise ValueError(
                f"Unknown routellm router '{router_type}'. "
                f"Available: {list(ROUTER_CLS.keys())}"
            )
        router_cls = ROUTER_CLS[router_type]
        # The BERT router only needs the checkpoint path; other variants take more args.
        self.router = router_cls(checkpoint_path=checkpoint)

    def predict_strong_prob(self, prompt: str) -> float:
        """Return P(strong model produces better response) ∈ [0, 1]."""
        return float(self.router.calculate_strong_win_rate(prompt))


class RouteLLMRouter:
    """
    RouteLLM (Ong et al., 2024) — preference-based direct routing.

    Algorithm:
        prob_strong = predictor(prompt)
        select tier = map_prob_to_tier(prob_strong, thresholds)
        return engine[tier].infer(prompt)

    With three tiers, we use two thresholds (low_t, high_t) to split into
    weak / mid / strong. With two tiers, only `threshold` is used.
    """

    def __init__(
        self,
        engines: List[BaseEngine],
        config: BaselineConfig = None,
        threshold_low: float = 0.33,
        threshold_high: float = 0.66,
        predictor: Optional[_RouteLLMPredictor] = None,
    ):
        self.config = config or BaselineConfig()
        self.engines = sorted(engines, key=lambda e: e.tier)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.predictor = predictor or _RouteLLMPredictor()

    def _select_engine(self, prob_strong: float, request: InferenceRequest) -> BaseEngine:
        candidates = [
            e for e in self.engines
            if e.status != EngineStatus.UNAVAILABLE
            and (not request.min_tier or e.tier >= request.min_tier)
        ]
        if not candidates:
            return self.engines[-1]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 2:
            return candidates[1] if prob_strong > 0.5 else candidates[0]
        # 3+ tiers: bucket the probability.
        if prob_strong < self.threshold_low:
            return candidates[0]
        if prob_strong < self.threshold_high:
            return candidates[len(candidates) // 2]
        return candidates[-1]

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        decision = RoutingDecision(request_id=request.request_id)
        start = time.perf_counter()

        prob_strong = self.predictor.predict_strong_prob(request.prompt)
        engine = self._select_engine(prob_strong, request)

        decision.tiers_attempted.append(engine.tier)
        decision.engines_tried.append(engine.engine_id)
        decision.escalation_reasons.append(f"routellm prob_strong={prob_strong:.3f}")

        response = await engine.infer(request)
        decision.final_engine = engine.engine_id
        decision.final_tier = engine.tier
        decision.success = response.success
        decision.total_latency_ms = (time.perf_counter() - start) * 1000
        decision.total_cost_usd = response.cost_usd
        return response, decision
