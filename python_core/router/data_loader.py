"""
Dataset loaders for benchmark evaluation.

Methodological choices:

- Default dataset: `tatsu-lab/alpaca_eval` (805 prompts, fully open, no auth required,
  widely cited in LLM evaluation literature). Each item has a reference response
  from a strong model (`output` field) which we can use either as ground truth
  for win-rate comparison or just as a sanity check.

- We DO NOT use Chatbot Arena's winner labels to assign `optimal_tier`. The winner
  field encodes "Model A preferred over Model B" between two anonymous models —
  it does not encode whether a strong tier was required. That mapping is a
  methodological error.

- We DO NOT silently fall back to an unlabeled dataset on load failure. If the
  requested dataset can't load, we raise. Silent degradation hides the fact that
  the benchmark is measuring nothing meaningful.

Quality scoring (in benchmark.py) uses a reward model to score every routed
response, so we don't need per-prompt ground-truth labels here. The dataset's job
is to provide a diverse, real prompt distribution.
"""

import hashlib
from typing import List, Optional

from .benchmark import WorkloadItem
from ..engines.base import InferenceRequest


def hash_workload(workload: List[WorkloadItem]) -> str:
    """SHA-256 of the (prompt, reference) content for reproducibility manifests.

    Two workloads with the same prompts in the same order produce the same hash,
    so a paper can publish this hash and a reviewer can verify they've loaded
    identical inputs.
    """
    h = hashlib.sha256()
    for item in workload:
        h.update(b"P\x00")
        h.update((item.request.prompt or "").encode("utf-8"))
        h.update(b"R\x00")
        h.update((item.reference_response or "").encode("utf-8"))
    return h.hexdigest()


def load_prompt_workload(
    dataset_name: str = "tatsu-lab/alpaca_eval",
    config_name: Optional[str] = "alpaca_eval",
    split: str = "eval",
    max_samples: int = 1000,
    prompt_field: Optional[str] = None,
    reference_field: Optional[str] = None,
) -> List[WorkloadItem]:
    """
    Load a prompt benchmark from HuggingFace into `WorkloadItem` format.

    Args:
        dataset_name: HuggingFace dataset id. Default `tatsu-lab/alpaca_eval`.
        config_name: Optional dataset configuration name (some datasets require it).
        split: Split name. Default "eval".
        max_samples: Cap the workload size to keep benchmarks tractable.
        prompt_field: Override which field to use as the prompt. If None, autodetect
            from common names (`instruction`, `prompt`, `question`, `text`).
        reference_field: Override which field to use as the reference response.
            If None, autodetect from (`output`, `response`, `reference`).

    Raises:
        ImportError: if the `datasets` package isn't installed.
        Exception: any load failure is propagated (no silent fallback).
    """
    from datasets import load_dataset

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    prompt_candidates = [prompt_field] if prompt_field else ["instruction", "prompt", "question", "text"]
    ref_candidates = [reference_field] if reference_field else ["output", "response", "reference"]

    detected_prompt: Optional[str] = None
    detected_ref: Optional[str] = None
    first = dataset[0] if len(dataset) > 0 else {}
    for c in prompt_candidates:
        if c and c in first:
            detected_prompt = c
            break
    if detected_prompt is None:
        raise ValueError(
            f"Could not autodetect a prompt field in dataset {dataset_name}. "
            f"Available fields: {list(first.keys())}. Pass `prompt_field=...` explicitly."
        )
    for c in ref_candidates:
        if c and c in first:
            detected_ref = c
            break  # Reference is optional — None is OK.

    workload: List[WorkloadItem] = []
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
        prompt = row.get(detected_prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        reference = row.get(detected_ref) if detected_ref else None
        if reference is not None and not isinstance(reference, str):
            reference = None

        workload.append(WorkloadItem(
            request=InferenceRequest(
                request_id=f"{dataset_name.replace('/', '_')}-{i:05d}",
                prompt=prompt,
                task_type="generation",
            ),
            reference_response=reference,
        ))

    if not workload:
        raise RuntimeError(
            f"Loaded 0 usable items from {dataset_name}. "
            f"Check `prompt_field`/`reference_field` overrides."
        )
    return workload
