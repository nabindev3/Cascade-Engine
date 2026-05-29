"""
Cloud engines — Mid-tier and Premium API providers.

Tier 2 (Mid): GPT-4o-mini, Claude Haiku, Gemini Flash — moderate cost, good capability.
Tier 3 (Premium): GPT-4o, Claude Opus, Gemini Pro — highest cost, best capability.

Both use the OpenAI-compatible API format for simplicity (most providers support it).
"""

import time
from typing import Optional

import httpx

from .base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)


class OpenAICompatibleEngine(BaseEngine):
    """
    Generic engine for any OpenAI-compatible API (OpenAI, Anthropic via proxy,
    Azure OpenAI, Together, Groq, etc.)

    Config keys:
        api_key: str (required)
        base_url: str (default "https://api.openai.com/v1")
        model: str (required, e.g., "gpt-4o-mini")
        timeout_s: float (default 60.0)
        cost_per_input_token: float (in USD)
        cost_per_output_token: float (in USD)
        max_retries: int (default 1)
    """

    def __init__(self, engine_id: str, tier: int, config: dict):
        super().__init__(engine_id=engine_id, tier=tier, config=config)

        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config["model"]
        self.timeout_s = config.get("timeout_s", 60.0)
        self.cost_per_input_token = config.get("cost_per_input_token", 0.0)
        self.cost_per_output_token = config.get("cost_per_output_token", 0.0)
        self.max_retries = config.get("max_retries", 1)

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        attempt = 0

        while attempt <= self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": request.prompt}],
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature,
                        },
                    )

                latency = (time.perf_counter() - start) * 1000

                # Handle rate limits
                if resp.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        self.record_failure(FailureMode.RATE_LIMIT)
                        return self._failure_response(
                            request, FailureMode.RATE_LIMIT, latency, "Rate limited"
                        )
                    # Exponential backoff
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue

                # Auth errors
                if resp.status_code in (401, 403):
                    self.record_failure(FailureMode.AUTH_ERROR)
                    return self._failure_response(
                        request, FailureMode.AUTH_ERROR, latency, f"Auth error: {resp.status_code}"
                    )

                # Server errors
                if resp.status_code >= 500:
                    self.record_failure(FailureMode.INFRASTRUCTURE)
                    return self._failure_response(
                        request, FailureMode.INFRASTRUCTURE, latency, f"Server error: {resp.status_code}"
                    )

                resp.raise_for_status()
                data = resp.json()

                # Parse response
                choices = data.get("choices", [])
                if not choices:
                    self.record_failure(FailureMode.PARSE_ERROR)
                    return self._failure_response(
                        request, FailureMode.PARSE_ERROR, latency, "Empty choices in response"
                    )

                content = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                token_input = usage.get("prompt_tokens", 0)
                token_output = usage.get("completion_tokens", 0)

                cost = (
                    token_input * self.cost_per_input_token
                    + token_output * self.cost_per_output_token
                )

                self.record_success()
                return InferenceResponse(
                    request_id=request.request_id,
                    engine_id=self.engine_id,
                    tier=self.tier,
                    content=content,
                    raw_output=data,
                    confidence=self._estimate_confidence(data, content),
                    latency_ms=latency,
                    cost_usd=cost,
                    token_count_input=token_input,
                    token_count_output=token_output,
                    success=True,
                )

            except httpx.TimeoutException:
                latency = (time.perf_counter() - start) * 1000
                self.record_failure(FailureMode.TIMEOUT)
                return self._failure_response(request, FailureMode.TIMEOUT, latency, "API timeout")

            except httpx.ConnectError:
                latency = (time.perf_counter() - start) * 1000
                self.record_failure(FailureMode.INFRASTRUCTURE)
                return self._failure_response(
                    request, FailureMode.INFRASTRUCTURE, latency, "Connection failed"
                )

            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                self.record_failure(FailureMode.INFRASTRUCTURE)
                return self._failure_response(
                    request, FailureMode.INFRASTRUCTURE, latency, str(e)
                )

        # Should not reach here, but safety net
        latency = (time.perf_counter() - start) * 1000
        return self._failure_response(
            request, FailureMode.INFRASTRUCTURE, latency, "Max retries exceeded"
        )

    def estimated_cost(self, request: InferenceRequest) -> float:
        est_input_tokens = len(request.prompt.split()) * 1.3
        return (
            est_input_tokens * self.cost_per_input_token
            + request.max_tokens * self.cost_per_output_token
        )

    async def health_check(self) -> EngineStatus:
        """Lightweight check — just verifies API is reachable with a tiny request."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                if resp.status_code == 200:
                    self._status = EngineStatus.HEALTHY
                elif resp.status_code == 429:
                    self._status = EngineStatus.DEGRADED
                else:
                    self._status = EngineStatus.UNAVAILABLE
        except Exception:
            self._status = EngineStatus.UNAVAILABLE
        return self._status

    def _estimate_confidence(self, data: dict, content: str) -> float:
        """
        Estimate confidence from API response.

        For research: this is where you'd plug in calibrated confidence
        (e.g., from logprobs, or a separate confidence classifier).
        """
        # If the API provides logprobs, use mean token probability
        choices = data.get("choices", [{}])
        logprobs = choices[0].get("logprobs")
        if logprobs and logprobs.get("content"):
            import math
            token_probs = [
                math.exp(t["logprob"]) for t in logprobs["content"] if "logprob" in t
            ]
            if token_probs:
                return sum(token_probs) / len(token_probs)

        # Heuristic fallback: higher tiers get higher base confidence
        if self.tier >= 3:
            return 0.9
        return 0.75

    def _failure_response(
        self, request: InferenceRequest, mode: FailureMode, latency: float, msg: str
    ) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            engine_id=self.engine_id,
            tier=self.tier,
            content="",
            confidence=0.0,
            latency_ms=latency,
            cost_usd=0.0,
            success=False,
            failure_mode=mode,
            error_message=msg,
        )


# ─── Convenience Factories ────────────────────────────────────────────────────


def create_mid_tier_engine(config: dict) -> OpenAICompatibleEngine:
    """Factory for Tier 2 engine (e.g., GPT-4o-mini)."""
    defaults = {
        "model": "gpt-4o-mini",
        "cost_per_input_token": 0.00000015,   # $0.15 / 1M tokens
        "cost_per_output_token": 0.0000006,   # $0.60 / 1M tokens
    }
    merged = {**defaults, **config}
    return OpenAICompatibleEngine(
        engine_id=config.get("engine_id", "openai-mid"),
        tier=2,
        config=merged,
    )


def create_premium_engine(config: dict) -> OpenAICompatibleEngine:
    """Factory for Tier 3 engine (e.g., GPT-4o)."""
    defaults = {
        "model": "gpt-4o",
        "cost_per_input_token": 0.0000025,    # $2.50 / 1M tokens
        "cost_per_output_token": 0.00001,     # $10.00 / 1M tokens
    }
    merged = {**defaults, **config}
    return OpenAICompatibleEngine(
        engine_id=config.get("engine_id", "openai-premium"),
        tier=3,
        config=merged,
    )
