"""
Local engine — calls Ollama or vLLM running on the same machine / network.

Tier 1: Cheapest, fastest, but least capable. Good for simple classification,
entity extraction, and short-form generation.
"""

import asyncio
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


class OllamaEngine(BaseEngine):
    """
    Connects to an Ollama instance for local inference.

    Config keys:
        base_url: str (default "http://localhost:11434")
        model: str (default "llama3.2:3b")
        timeout_s: float (default 30.0)
        cost_per_token: float (default 0.0 — local is "free" but we track compute)
    """

    def __init__(self, engine_id: str = "ollama-local", config: dict = None):
        config = config or {}
        super().__init__(engine_id=engine_id, tier=1, config=config)

        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama3.2:3b")
        self.timeout_s = config.get("timeout_s", 30.0)
        self.cost_per_token = config.get("cost_per_token", 0.000001)  # ~free

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": request.prompt,
                        "stream": False,
                        "options": {
                            "num_predict": request.max_tokens,
                            "temperature": request.temperature,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            latency = (time.perf_counter() - start) * 1000
            content = data.get("response", "")
            token_count_output = data.get("eval_count", len(content.split()))
            token_count_input = data.get("prompt_eval_count", len(request.prompt.split()))

            self.record_success()
            return InferenceResponse(
                request_id=request.request_id,
                engine_id=self.engine_id,
                tier=self.tier,
                content=content,
                raw_output=data,
                confidence=self._estimate_confidence(content, request),
                latency_ms=latency,
                cost_usd=self.cost_per_token * (token_count_input + token_count_output),
                token_count_input=token_count_input,
                token_count_output=token_count_output,
                success=True,
            )

        except httpx.TimeoutException:
            latency = (time.perf_counter() - start) * 1000
            self.record_failure(FailureMode.TIMEOUT)
            return self._failure_response(request, FailureMode.TIMEOUT, latency, "Ollama timeout")

        except httpx.ConnectError:
            latency = (time.perf_counter() - start) * 1000
            self.record_failure(FailureMode.INFRASTRUCTURE)
            return self._failure_response(request, FailureMode.INFRASTRUCTURE, latency, "Cannot connect to Ollama")

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.record_failure(FailureMode.INFRASTRUCTURE)
            return self._failure_response(request, FailureMode.INFRASTRUCTURE, latency, str(e))

    def estimated_cost(self, request: InferenceRequest) -> float:
        # Rough estimate: input tokens + max output tokens
        est_input_tokens = len(request.prompt.split()) * 1.3
        return self.cost_per_token * (est_input_tokens + request.max_tokens)

    async def health_check(self) -> EngineStatus:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    self._status = EngineStatus.HEALTHY
                else:
                    self._status = EngineStatus.DEGRADED
        except Exception:
            self._status = EngineStatus.UNAVAILABLE
        return self._status

    def _estimate_confidence(self, content: str, request: InferenceRequest) -> float:
        """
        Heuristic confidence for local models.
        In production, this would use calibrated logprobs or a learned confidence head.
        """
        if not content or len(content.strip()) < 3:
            return 0.1
        # Longer, well-formed responses get higher base confidence
        word_count = len(content.split())
        if word_count > 5:
            return 0.6
        return 0.4

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
