"""
Event Logger — Structured instrumentation for every inference call.

This is the data backbone for Paper 1 ("Measurement Study"). Every call through
the cascade generates a structured event that captures:
- Input characteristics (length, task type, complexity signals)
- Routing decisions (which tiers attempted, why escalations happened)
- Outcomes (latency, cost, confidence, failure modes)
- Disagreements (when multiple tiers produce different answers)

Storage: JSON Lines (.jsonl) for easy analysis with pandas/DuckDB.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from ..engines.base import FailureMode, InferenceRequest, InferenceResponse
from ..router.cascade_router import RoutingDecision


@dataclass
class InferenceEvent:
    """A single instrumented inference event — one row in your research dataset."""

    # Identity
    event_id: str
    request_id: str
    timestamp: float = field(default_factory=time.time)

    # Input characteristics
    input_length_chars: int = 0
    input_length_tokens_est: int = 0
    task_type: str = "general"
    metadata: dict = field(default_factory=dict)

    # Routing path
    tiers_attempted: List[int] = field(default_factory=list)
    engines_tried: List[str] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)
    final_engine: Optional[str] = None
    final_tier: Optional[int] = None

    # Outcome
    success: bool = False
    confidence: float = 0.0
    failure_mode: str = "none"
    error_message: Optional[str] = None

    # Performance
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    token_count_input: int = 0
    token_count_output: int = 0

    # Disagreement tracking (for Paper 1: "who was right?")
    tier_responses: List[dict] = field(default_factory=list)
    has_disagreement: bool = False


class EventLogger:
    """
    Logs inference events to JSONL files for research analysis.

    Usage:
        logger = EventLogger(output_dir="./data/logs")
        event = logger.create_event(request, response, decision)
        logger.log(event)
    """

    def __init__(self, output_dir: str = "./data/logs", buffer_size: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[dict] = []
        self._buffer_size = buffer_size
        self._current_file = self._get_log_path()

    def create_event(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        decision: RoutingDecision,
        all_responses: Optional[List[InferenceResponse]] = None,
    ) -> InferenceEvent:
        """Build a structured event from the inference artifacts."""
        event = InferenceEvent(
            event_id=f"{request.request_id}-{int(time.time()*1000)}",
            request_id=request.request_id,
            input_length_chars=len(request.prompt),
            input_length_tokens_est=len(request.prompt.split()),
            task_type=request.task_type,
            metadata=request.metadata,
            tiers_attempted=decision.tiers_attempted,
            engines_tried=decision.engines_tried,
            escalation_reasons=decision.escalation_reasons,
            final_engine=decision.final_engine,
            final_tier=decision.final_tier,
            success=response.success,
            confidence=response.confidence,
            failure_mode=response.failure_mode.value,
            error_message=response.error_message,
            total_latency_ms=decision.total_latency_ms,
            total_cost_usd=decision.total_cost_usd,
            token_count_input=response.token_count_input,
            token_count_output=response.token_count_output,
        )

        # Track disagreements across tiers
        if all_responses and len(all_responses) > 1:
            successful = [r for r in all_responses if r.success and r.content]
            if len(successful) > 1:
                # Simple disagreement check: are the outputs meaningfully different?
                contents = [r.content.strip().lower() for r in successful]
                event.has_disagreement = len(set(contents)) > 1
                event.tier_responses = [
                    {
                        "engine_id": r.engine_id,
                        "tier": r.tier,
                        "confidence": r.confidence,
                        "content_preview": r.content[:200],
                    }
                    for r in successful
                ]

        return event

    def log(self, event: InferenceEvent) -> None:
        """Buffer and write event to disk."""
        self._buffer.append(asdict(event))
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered events to JSONL file."""
        if not self._buffer:
            return
        with open(self._current_file, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record, default=str) + "\n")
        self._buffer.clear()

    def _get_log_path(self) -> Path:
        """One file per day for easy partitioning."""
        date_str = time.strftime("%Y-%m-%d")
        return self.output_dir / f"inference_events_{date_str}.jsonl"

    def get_stats_summary(self) -> dict[str, Any]:
        """Quick summary of today's events for the dashboard."""
        path = self._get_log_path()
        if not path.exists():
            return {"total_events": 0}

        total: int = 0
        successes: int = 0
        total_cost: float = 0.0
        tier_counts: dict[int, int] = {}
        failure_modes: dict[str, int] = {}

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                event: dict[str, Any] = json.loads(line)
                total += 1
                if event.get("success"):
                    successes += 1
                total_cost += event.get("total_cost_usd", 0)
                tier: int = event.get("final_tier", 0)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                mode: str = event.get("failure_mode", "none")
                if mode != "none":
                    failure_modes[mode] = failure_modes.get(mode, 0) + 1

        return {
            "total_events": total,
            "success_rate": successes / total if total > 0 else 0,
            "total_cost_usd": round(total_cost, 6),
            "tier_distribution": tier_counts,
            "failure_mode_distribution": failure_modes,
        }

    def __del__(self) -> None:
        self.flush()

