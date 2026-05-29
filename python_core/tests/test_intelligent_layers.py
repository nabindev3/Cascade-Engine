"""Smoke tests for the intelligent orchestration layers.

These tests deliberately do NOT skip on missing dependencies — a missing
dependency means the production stack is broken, which is exactly what these
tests must surface.
"""

import pytest

from python_core.router.intelligent_layers import (
    GatekeeperClassifier,
    IntelligentOrchestrator,
    PrivacyFilter,
    SarcasmDetector,
    SemanticCache,
)


pytestmark = pytest.mark.heavy


def test_privacy_filter_masks_pii():
    pf = PrivacyFilter()
    text = "My credit card is 4111-1111-1111-1111 and email is alice@example.com"
    masked = pf.mask(text)
    assert "4111-1111-1111-1111" not in masked
    assert "alice@example.com" not in masked


def test_gatekeeper_returns_valid_label():
    gk = GatekeeperClassifier()
    label = gk.classify("Calculate the sum of all primes less than 100.")
    assert label in GatekeeperClassifier.LABELS


def test_sarcasm_detector_high_intensity():
    sd = SarcasmDetector(threshold=0.5)
    assert sd.is_high_intensity("This is absolutely the worst experience I've ever had!") is True
    assert sd.is_high_intensity("The book is on the table.") is False


def test_semantic_cache_exact_match():
    cache = SemanticCache(threshold=0.95)
    cache.save_cache("What is the capital of France?", "Paris")
    assert cache.check_cache("What is the capital of France?") == "Paris"


def test_semantic_cache_miss_on_unrelated():
    cache = SemanticCache(threshold=0.95)
    cache.save_cache("What is the capital of France?", "Paris")
    assert cache.check_cache("How do I bake a chocolate cake?") is None


def test_orchestrator_instances_are_independent():
    """Critical for benchmark fairness: cache state must not leak across instances."""
    o1 = IntelligentOrchestrator()
    o2 = IntelligentOrchestrator()
    o1.cache.save_cache("test prompt about widgets", "answer 1")
    # o2 should NOT see o1's cache entry.
    assert o2.cache.check_cache("test prompt about widgets") is None
    # Budgets are independent too.
    o1.budget.charge(0.5)
    assert o2.budget.spent_today == 0.0


def test_no_module_level_singleton():
    """Importing the module must not create a shared `orchestrator` global."""
    import python_core.router.intelligent_layers as il
    assert not hasattr(il, "orchestrator"), (
        "Module-level singleton was reintroduced — this causes cache state to "
        "leak across benchmark runs and breaks fair comparison."
    )
