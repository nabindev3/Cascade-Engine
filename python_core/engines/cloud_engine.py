"""
Cloud engines — Mid-tier and Premium API providers.

Tier 2 (Mid): GPT-4o-mini, Claude Haiku, Gemini Flash — moderate cost, good capability.
Tier 3 (Premium): Premium models — highest cost, best capability.

Both use the OpenAI-compatible API format for simplicity (most providers support it).
"""

import asyncio
import random
import time
from typing import Any, Optional

import httpx

from .base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)


class OpenAIEngine(BaseEngine):
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
        max_retries: int (default 2 — number of retries on transient failures)
        backoff_base_s: float (default 0.5 — base for exponential backoff)
        backoff_max_s: float (default 30.0 — ceiling for a single backoff sleep)
    """

    # HTTP statuses that are transient and worth retrying.
    _RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    def __init__(self, engine_id: str, tier: int, config: dict[str, Any]) -> None:
        super().__init__(engine_id=engine_id, tier=tier, config=config)

        self.api_key: str = config["api_key"]
        self.base_url: str = config.get("base_url", "https://api.openai.com/v1")
        self.model: str = config["model"]
        self.timeout_s: float = config.get("timeout_s", 60.0)
        self.cost_per_input_token: float = config.get("cost_per_input_token", 0.0)
        self.cost_per_output_token: float = config.get("cost_per_output_token", 0.0)
        self.max_retries: int = config.get("max_retries", 2)
        self.backoff_base_s: float = config.get("backoff_base_s", 0.5)
        self.backoff_max_s: float = config.get("backoff_max_s", 30.0)

    async def _backoff_sleep(self, attempt: int, retry_after: Optional[str]) -> None:
        """Sleep before a retry using jittered exponential backoff.

        Honors a server-provided ``Retry-After`` header (seconds) when present,
        otherwise uses ``backoff_base_s * 2**attempt`` capped at ``backoff_max_s``,
        plus full jitter to avoid thundering-herd retries against the provider.
        """
        delay: float
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.backoff_base_s * (2 ** attempt)
        else:
            delay = self.backoff_base_s * (2 ** attempt)
        delay = min(delay, self.backoff_max_s)
        # Full jitter: sample in [0, delay] (https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
        await asyncio.sleep(random.uniform(0.0, delay))

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        start: float = time.perf_counter()
        attempt: int = 0
        # Remembered transient failure so the final return reports the real cause.
        last_transient_mode: FailureMode = FailureMode.INFRASTRUCTURE
        last_transient_msg: str = "Max retries exceeded"

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

                latency: float = (time.perf_counter() - start) * 1000

                # Auth errors — never retryable, fail fast.
                if resp.status_code in (401, 403):
                    self.record_failure(FailureMode.AUTH_ERROR)
                    return self._failure_response(
                        request, FailureMode.AUTH_ERROR, latency, f"Auth error: {resp.status_code}"
                    )

                # Transient errors (rate limit + server errors) — back off and retry.
                if resp.status_code in self._RETRYABLE_STATUSES:
                    last_transient_mode = (
                        FailureMode.RATE_LIMIT if resp.status_code == 429
                        else FailureMode.INFRASTRUCTURE
                    )
                    last_transient_msg = (
                        "Rate limited" if resp.status_code == 429
                        else f"Server error: {resp.status_code}"
                    )
                    attempt += 1
                    if attempt > self.max_retries:
                        self.record_failure(last_transient_mode)
                        return self._failure_response(
                            request, last_transient_mode, latency, last_transient_msg
                        )
                    await self._backoff_sleep(attempt, resp.headers.get("Retry-After"))
                    continue

                resp.raise_for_status()
                data: dict[str, Any] = resp.json()

                # Parse response
                choices: list[dict[str, Any]] = data.get("choices", [])
                if not choices:
                    self.record_failure(FailureMode.PARSE_ERROR)
                    return self._failure_response(
                        request, FailureMode.PARSE_ERROR, latency, "Empty choices in response"
                    )

                content: str = choices[0].get("message", {}).get("content", "")
                usage: dict[str, Any] = data.get("usage", {})
                token_input: int = usage.get("prompt_tokens", 0)
                token_output: int = usage.get("completion_tokens", 0)

                cost: float = (
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

            # Network-level errors are transient — back off and retry.
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                latency = (time.perf_counter() - start) * 1000
                last_transient_mode = (
                    FailureMode.TIMEOUT if isinstance(e, httpx.TimeoutException)
                    else FailureMode.INFRASTRUCTURE
                )
                last_transient_msg = (
                    "API timeout" if isinstance(e, httpx.TimeoutException)
                    else "Connection failed"
                )
                attempt += 1
                if attempt > self.max_retries:
                    self.record_failure(last_transient_mode)
                    return self._failure_response(
                        request, last_transient_mode, latency, last_transient_msg
                    )
                await self._backoff_sleep(attempt, None)
                continue

            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                self.record_failure(FailureMode.INFRASTRUCTURE)
                return self._failure_response(
                    request, FailureMode.INFRASTRUCTURE, latency, str(e)
                )

        # All retries exhausted on a transient failure.
        latency = (time.perf_counter() - start) * 1000
        self.record_failure(last_transient_mode)
        return self._failure_response(
            request, last_transient_mode, latency, last_transient_msg
        )

    def estimated_cost(self, request: InferenceRequest) -> float:
        est_input_tokens: float = len(request.prompt.split()) * 1.3
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

    def _estimate_confidence(self, data: dict[str, Any], content: str) -> float:
        """
        Estimate confidence from API response.

        For research: this is where you'd plug in calibrated confidence
        (e.g., from logprobs, or a separate confidence classifier).
        """
        # If the API provides logprobs, use mean token probability
        choices: list[dict[str, Any]] = data.get("choices", [{}])
        logprobs: Optional[dict[str, Any]] = choices[0].get("logprobs")
        if logprobs and logprobs.get("content"):
            import math
            token_probs: list[float] = [
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


def create_mid_tier_engine(config: dict[str, Any]) -> OpenAIEngine:
    """Factory for Tier 2 engine (e.g., GPT-4o-mini)."""
    defaults: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "cost_per_input_token": 0.00000015,   # $0.15 / 1M tokens
        "cost_per_output_token": 0.0000006,   # $0.60 / 1M tokens
    }
    merged: dict[str, Any] = {**defaults, **config}
    return OpenAIEngine(
        engine_id=config.get("engine_id", "openai-mid"),
        tier=2,
        config=merged,
    )


def create_premium_engine(config: dict[str, Any]) -> OpenAIEngine:
    """Factory for Tier 3 engine (e.g., GPT-4o)."""
    defaults: dict[str, Any] = {
        "model": "gpt-4o",
        "cost_per_input_token": 0.0000025,    # $2.50 / 1M tokens
        "cost_per_output_token": 0.00001,     # $10.00 / 1M tokens
    }
    merged: dict[str, Any] = {**defaults, **config}
    return OpenAIEngine(
        engine_id=config.get("engine_id", "openai-premium"),
        tier=3,
        config=merged,
    )

