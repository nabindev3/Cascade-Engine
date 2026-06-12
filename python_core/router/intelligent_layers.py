"""
Intelligent orchestration layers — cache, privacy, gatekeeper, sarcasm, budget.

These layers are a system contribution, separate from the routing policy.
Each `IntelligentOrchestrator` is a standalone instance; there is no shared
module-level singleton. This is intentional: in benchmarks, every router must
get its own orchestrator so cache state cannot leak across runs.
"""

import re
import sqlite3
from typing import Any, Optional



class PrivacyFilter:
    """Privacy-aware masking via Microsoft Presidio."""

    def __init__(self) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.entities = ["CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON",
                         "US_SSN", "IP_ADDRESS", "IBAN_CODE"]

    def mask(self, text: str) -> str:
        results: list[Any] = self.analyzer.analyze(text=text, entities=self.entities, language="en")
        return self.anonymizer.anonymize(text=text, analyzer_results=results).text


class TokenBudget:
    """Daily token-cost budget (production safety; not part of routing policy)."""

    def __init__(self, daily_budget_usd: float = 5.0):
        self.daily_budget_usd = daily_budget_usd
        self.spent_today = 0.0

    def can_afford(self, estimated_cost: float) -> bool:
        return (self.spent_today + estimated_cost) <= self.daily_budget_usd

    def charge(self, cost: float) -> None:
        self.spent_today += cost


class GatekeeperClassifier:
    """Zero-shot task classifier (DistilBERT-MNLI) — categorizes prompts as
    Logical / Creative / Simple to inform min-tier hints."""

    LABELS = ["Logical", "Creative", "Simple"]

    def __init__(self, device: int = -1):
        from transformers import pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=device,
        )

    def classify(self, prompt: str) -> str:
        result = self.classifier(prompt, candidate_labels=self.LABELS)
        return result["labels"][0]


class SarcasmDetector:
    """VADER-based emotional-intensity detector. High-intensity prompts often
    contain sarcasm/strong sentiment that benefits from the premium tier."""

    def __init__(self, threshold: float = 0.5):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def is_high_intensity(self, prompt: str) -> bool:
        return abs(self.analyzer.polarity_scores(prompt)["compound"]) >= self.threshold


class SemanticCache:
    """Embedding-based cache (FAISS IndexFlatIP + all-MiniLM-L6-v2).

    Each instance maintains its own index and response store. No state is shared
    across instances — critical for benchmark fairness.
    """

    def __init__(self, threshold: float = 0.95, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.responses: list[str] = []
        self.threshold = threshold

    def check_cache(self, prompt: str) -> Optional[str]:
        if self.index.ntotal == 0:
            return None
        embedding = self.encoder.encode([prompt], normalize_embeddings=True)
        similarities, indices = self.index.search(embedding, 1)
        score, idx = float(similarities[0][0]), int(indices[0][0])
        if score >= self.threshold and idx >= 0:
            return self.responses[idx]
        return None

    def save_cache(self, prompt: str, response: str) -> None:
        embedding = self.encoder.encode([prompt], normalize_embeddings=True)
        self.index.add(embedding)
        self.responses.append(response)


class IntelligentOrchestrator:
    """Bundles the four intelligent layers + the daily token budget.

    Every router that uses orchestration receives its own instance. Construction
    is expensive (loads spaCy + DistilBERT + MiniLM); reuse across requests
    within one router but never share across routers in a benchmark.
    """

    def __init__(
        self,
        daily_budget_usd: float = 5.0,
        cache_threshold: float = 0.95,
        sarcasm_threshold: float = 0.5,
        gatekeeper_device: int = -1,
    ):
        from typing import Any
        self.privacy = PrivacyFilter()
        self.budget = TokenBudget(daily_budget_usd)
        self.classifier = GatekeeperClassifier(device=gatekeeper_device)
        self.sarcasm = SarcasmDetector(threshold=sarcasm_threshold)
        self.cache = SemanticCache(threshold=cache_threshold)
