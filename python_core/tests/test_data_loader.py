"""Tests for the dataset loader. We focus on the structural guarantees that
matter for methodology: hard-fail behavior, no silent fallback, and correct
parsing of the configured fields."""

from unittest.mock import patch

import pytest

from python_core.router.data_loader import load_prompt_workload


def test_load_propagates_errors_no_silent_fallback():
    """If the dataset can't load, the loader MUST raise. Silent fallback to an
    unlabeled dataset is the methodological bug we just fixed."""

    def boom(*args, **kwargs):
        raise OSError("simulated network failure")

    with patch("datasets.load_dataset", side_effect=boom):
        with pytest.raises(OSError):
            load_prompt_workload(dataset_name="any/dataset", max_samples=10)


def test_load_autodetects_prompt_field():
    fake_rows = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "Capital of France?", "output": "Paris"},
    ]

    class FakeDataset:
        def __getitem__(self, i):
            return fake_rows[i]
        def __len__(self):
            return len(fake_rows)
        def __iter__(self):
            return iter(fake_rows)

    with patch("datasets.load_dataset", return_value=FakeDataset()):
        workload = load_prompt_workload(dataset_name="x", max_samples=10)

    assert len(workload) == 2
    assert workload[0].request.prompt == "What is 2+2?"
    assert workload[0].reference_response == "4"
    assert workload[1].reference_response == "Paris"
    # Critical: the broken winner→optimal_tier mapping must be gone.
    assert workload[0].optimal_tier is None
    assert workload[1].optimal_tier is None


def test_load_skips_empty_and_non_string_prompts():
    fake_rows = [
        {"instruction": "valid prompt", "output": "ref"},
        {"instruction": "", "output": "ref"},
        {"instruction": None, "output": "ref"},
        {"instruction": "  ", "output": "ref"},
        {"instruction": "second valid", "output": "ref"},
    ]

    class FakeDataset:
        def __getitem__(self, i):
            return fake_rows[i]
        def __len__(self):
            return len(fake_rows)
        def __iter__(self):
            return iter(fake_rows)

    with patch("datasets.load_dataset", return_value=FakeDataset()):
        workload = load_prompt_workload(dataset_name="x", max_samples=10)

    assert len(workload) == 2
    assert workload[0].request.prompt == "valid prompt"
    assert workload[1].request.prompt == "second valid"


def test_load_raises_when_no_prompt_field_found():
    fake_rows = [{"weird_field": "foo", "another": "bar"}]

    class FakeDataset:
        def __getitem__(self, i):
            return fake_rows[i]
        def __len__(self):
            return len(fake_rows)
        def __iter__(self):
            return iter(fake_rows)

    with patch("datasets.load_dataset", return_value=FakeDataset()):
        with pytest.raises(ValueError, match="Could not autodetect"):
            load_prompt_workload(dataset_name="x", max_samples=10)


def test_load_raises_when_all_items_filtered():
    fake_rows = [{"instruction": "", "output": "x"}] * 3

    class FakeDataset:
        def __getitem__(self, i):
            return fake_rows[i]
        def __len__(self):
            return len(fake_rows)
        def __iter__(self):
            return iter(fake_rows)

    with patch("datasets.load_dataset", return_value=FakeDataset()):
        with pytest.raises(RuntimeError, match="0 usable items"):
            load_prompt_workload(dataset_name="x", max_samples=10)


def test_load_respects_explicit_field_overrides():
    fake_rows = [{"my_custom_prompt": "hi", "extra": "y"}]

    class FakeDataset:
        def __getitem__(self, i):
            return fake_rows[i]
        def __len__(self):
            return len(fake_rows)
        def __iter__(self):
            return iter(fake_rows)

    with patch("datasets.load_dataset", return_value=FakeDataset()):
        workload = load_prompt_workload(
            dataset_name="x",
            prompt_field="my_custom_prompt",
            max_samples=10,
        )
    assert workload[0].request.prompt == "hi"
