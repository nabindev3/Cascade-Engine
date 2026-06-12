"""
OrchestrationWrapper — applies the intelligent layers uniformly around any router.

Separating orchestration from routing policy is the key fix for benchmark fairness:
- Routers do pure routing (no cache, no PII, no gatekeeper).
- The wrapper adds those layers as a separate, swappable concern.
- Every wrapper owns its own `IntelligentOrchestrator` — no shared state.

Order of operations (important):
  1. Privacy mask the request.prompt FIRST (so PII never becomes a cache key).
  2. Cache lookup on the masked prompt.
  3. Gatekeeper + sarcasm set request.min_tier hints.
  4. Delegate to inner router.
  5. Save successful responses to cache.
"""

from typing import Any, Optional, Tuple

from ..engines.base import InferenceRequest, InferenceResponse
from .cascade_router import RoutingDecision
from .intelligent_layers import IntelligentOrchestrator


class OrchestrationWrapper:
    """Generic wrapper. The inner object must expose `async def route(request) -> (response, decision)`."""

    def __init__(self, inner_router: Any, orchestrator: Optional[IntelligentOrchestrator] = None) -> None:
        self.inner: Any = inner_router
        self.orchestrator: IntelligentOrchestrator = orchestrator or IntelligentOrchestrator()

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        # 1. Privacy mask before anything touches the prompt.
        request.prompt = self.orchestrator.privacy.mask(request.prompt)

        # 2. Cache check on the masked prompt.
        cached: Optional[str] = self.orchestrator.cache.check_cache(request.prompt)
        if cached is not None:
            decision: RoutingDecision = RoutingDecision(request_id=request.request_id)
            decision.final_engine = "cache"
            decision.final_tier = 0
            decision.success = True
            response: InferenceResponse = InferenceResponse(
                request_id=request.request_id,
                engine_id="cache",
                tier=0,
                content=cached,
                confidence=1.0,
                success=True,
            )
            return response, decision

        # 3. Gatekeeper + sarcasm produce min_tier hints.
        task_class: str = self.orchestrator.classifier.classify(request.prompt)
        if self.orchestrator.sarcasm.is_high_intensity(request.prompt):
            request.min_tier = 3
        elif task_class == "Logical" and (request.min_tier or 0) < 2:
            request.min_tier = 2

        # 4. Premium-tier budget pre-check (skip premium if daily budget exhausted).
        if request.min_tier == 3:
            estimated_premium: float = 0.01  # conservative pre-estimate
            if not self.orchestrator.budget.can_afford(estimated_premium):
                request.min_tier = 2  # Downgrade rather than fail.

        # 5. Delegate.
        response, decision = await self.inner.route(request)

        # 6. Charge budget + save cache on success.
        if response.success:
            if decision.final_tier == 3:
                self.orchestrator.budget.charge(response.cost_usd)
            self.orchestrator.cache.save_cache(request.prompt, response.content)

        return response, decision

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

