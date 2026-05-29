# Cascade Engine — Status vs. the 15-Step PhD-Impact Plan

This document reconciles the action plan in `analysis_results.md` with the
**actual state of the codebase**, records what was implemented in the most
recent work session, and lays out a realistic roadmap for the remaining
research-heavy steps.

> **Key finding:** the plan was written against an earlier snapshot. Several
> "to-do" items were already implemented (UCB-Q, real-dataset loading, the
> baselines, the non-stationary regret experiment). The work below filled the
> genuine, verifiable gaps; the rest is scoped honestly as multi-week research.

---

## Test baseline (foundation)

`python -m pytest` previously had **11 failures**. All were environmental, not
logic bugs:

- `transformers` eagerly imported TensorFlow → protobuf "file defined twice"
  crash on macOS. Fixed by forcing the torch backend (`USE_TF=0`) in
  `python_core/tests/conftest.py` before any transformers import. This recovered
  **9** tests (privacy filter, gatekeeper, semantic cache, orchestration
  wrapper, FrugalGPT baseline).
- The 2 RouteLLM tests require a paid `OPENAI_API_KEY` (the `routellm` package
  initializes an OpenAI client at import). These now **skip cleanly** without a
  key and **run for real** when one is present (`test_baseline_routers.py`).

Current state: **66 passed, 14 heavy deselected** (fast suite); heavy suite green
except the credential-gated RouteLLM tests.

---

## Step-by-step reconciliation

| # | Step | State | Notes |
|---|------|-------|-------|
| 1 | Optimistic UCB-Q | ✅ **Already done** | `MDPConfig.use_ucb_q`, optimistic init to `H`, dynamic LR `(H+1)/(H+t)`, bonus `c·√(H³/t)`; covered by `test_ucb_q.py`. |
| 2 | Semantic / linear bandits | ✅ **Implemented this session** | `LinTSRouter` + `LinTSArm` (Bayesian linear TS, Agrawal & Goyal 2013), dependency-free `prompt_features`, optional `make_embedding_feature_fn` for sentence-transformers. Wired into the benchmark as `lints`. Tests: `test_lints.py`. |
| 3 | Adaptive drift (no prior `V_T`) | ✅ **Implemented this session** | `CUSUMDetector` + `AdaptiveCDTSPolicy` (detect-and-restart, Cao et al. 2019). Added as a first-class comparator in the non-stationary experiment. **Matches/beats the oracle-tuned CD-TS without ever seeing `V_T`.** Tests: `test_nonstationary.py`. |
| 4 | DBV-25 unified cascading composition | ⬜ Not started | Research: compose the ICML-2025 per-query optimal rule on top of the online posteriors. |
| 5 | Risk-sensitive SLA constraints (CMDP) | ⬜ Not started | Research: constrained-bandit / CMDP objective with latency/budget caps. |
| 6 | Real-time React/Next.js dashboard | ⬜ Not started | Engineering: front-end over `/v1/stats` (Beta posteriors, savings, drift). |
| 7 | TS gateway test suite | ✅ **Implemented this session** | `typescript_api/tests/` (vitest + supertest): auth, validation, proxy, error/status propagation, batch limits, rate limiting. **19 tests pass**; `tsc --noEmit` clean. |
| 8 | Direct gateway cloud fallback (circuit breaker) | ⬜ Not started | Engineering: fallback path in `server.ts` when the core is unhealthy. |
| 9 | Redis distributed state sync | ⬜ Not started | Engineering: move posteriors/Q-values/EMAs into Redis (compose already has it). |
| 10 | Async predictive pre-warming | ⬜ Not started | Engineering: speculative next-tier spin-up to hide escalation latency. |
| 11 | Real NLP/serving benchmarks | ✅ **Already done** | `data_loader.load_prompt_workload` loads any HF dataset (alpaca_eval default; pass `dataset_name="gsm8k"` / `"cais/mmlu"` for those). Reward-model scoring, multi-seed t-CIs, Pareto frontier all present. |
| 12 | High-fidelity API replay + drift simulator | 🟡 Partial | The controlled non-stationary env exists; recording/replaying *real* engine responses with injected drift is not yet built. |
| 13 | Open-source packaging (pip/npm + CI) | ⬜ Not started | Engineering: `pyproject.toml`, GitHub Actions, docs. |
| 14 | LaTeX polish + appendix proofs | 🟡 Partial | The adaptive-drift result (Step 3) is now written into **both** `paper/main.tex` (new §"Adaptive Drift Tracking Without a Known $V_T$": Algorithms 2–3, Remark, Table comparing Adaptive vs oracle CD-TS vs static; abstract + contributions + limitations updated) and `paper/theory.tex`, with four new bib entries. Still open: a LinTS (Step 2) subsection + the semantic-embedding experiment. |
| 15 | Submission / arXiv / networking | ⬜ Not started | Strategy: ENLSP workshop or MLSys/ICML/NeurIPS full paper. |

Legend: ✅ done · 🟡 partial · ⬜ not started.

---

## What changed this session (files)

- `python_core/tests/conftest.py` — force torch backend (`USE_TF=0`) so the
  heavy ML stack imports cleanly on macOS.
- `python_core/tests/test_baseline_routers.py` — skip RouteLLM tests without
  `OPENAI_API_KEY` (paid credential, not a code dependency).
- `python_core/router/nonstationary.py` — **Step 3**: `CUSUMDetector`,
  `AdaptiveCDTSPolicy`; added `adaptive_cd_ts` to `run_single_horizon` and
  `run_horizon_scaling`.
- `python_core/scripts/run_nonstationary.py` — emit `adaptive_cd_ts` rows in
  `scaling.csv` and its exponent in the summary.
- `python_core/router/learned_router.py` — **Step 2**: `prompt_features`,
  `make_embedding_feature_fn`, `LinTSConfig`, `LinTSArm`, `LinTSRouter`.
- `python_core/router/benchmark.py` — register `lints` among the compared
  routers.
- `python_core/tests/test_nonstationary.py`, `python_core/tests/test_lints.py`
  — tests for Steps 3 and 2.
- `typescript_api/src/server.ts` — env-configurable rate limit; skip port-bind
  and pretty-log transport under `NODE_ENV=test` (testability).
- `typescript_api/package.json`, `typescript_api/tests/*` — **Step 7**: vitest +
  supertest gateway suite.

### Headline empirical result (Step 3)

On the controlled non-stationary benchmark, the **adaptive** policy — which
**never sees the variation budget `V_T`** — is competitive with, and often
better than, the oracle-tuned CD-TS:

| Regime | metric | `cd_ts` (γ from known V_T) | `adaptive_cd_ts` (no V_T) | `static_optimal` |
|--------|--------|---------------------------|---------------------------|------------------|
| abrupt | final regret | 75.8 | **43.0** | 868.4 |
| abrupt | tail per-round regret | 0.0091 | **0.0003** | 0.349 |
| periodic | regret-vs-horizon exponent | 0.549 | **0.263** | 0.999 |

This directly answers the plan's Area-for-Improvement A3 ("`V_T` assumed known a
priori") and is a clean paper contribution: assumption-free drift tracking at no
regret cost.

---

## How to reproduce

```bash
# Python — fast suite (no model downloads)
python -m pytest -m "not heavy"

# Python — heavy suite (downloads models; set OPENAI_API_KEY to include RouteLLM)
python -m pytest -m heavy

# Non-stationary experiment incl. adaptive (Step 3)
python -m python_core.scripts.run_nonstationary --regime all --seeds 8

# Benchmark incl. LinTS (Step 2), on a real dataset
python -m python_core.scripts.run_experiment   # see --help for dataset flags

# TypeScript gateway tests (Step 7)
cd typescript_api && npm install && npm test
```

---

## Recommended next steps (highest leverage first)

1. **Paper integration (Step 14, partial).** ✅ The adaptive-drift section
   (Step 3) is now in `main.tex` and `theory.tex`. Remaining: a LinTS
   subsection (Step 2) and the semantic-embedding experiment; optionally a
   scaling figure that also plots the `adaptive_cd_ts` curve (the result is
   currently presented as a table).
2. **Semantic LinTS experiment (Step 2, extend).** Run the benchmark with
   `make_embedding_feature_fn("all-MiniLM-L6-v2")` and report LinTS vs. tabular
   Thompson on the Pareto frontier — quantifies the generalization benefit.
3. **Step 12 (replay simulator).** Record real engine responses and replay with
   injected latency/availability drift — turns the controlled experiment into a
   real-trace one and strengthens empirical credibility.
4. **Step 8/9 (gateway fallback + Redis state).** Production-systems story for
   MLSys; both are self-contained engineering with clear test surfaces.
5. **Steps 4/5 (DBV-25 composition, CMDP/SLA).** The deepest theory; pursue once
   the above are written up.
