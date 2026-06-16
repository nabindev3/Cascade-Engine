# Cascade Engine

This repository provides the code for our framework **Cascade Engine: A Multi-Tier, Intelligent Routing Framework for Cost-Effective LLM Inference**. In this README, we guide you through installing the library, using the intelligent cascade routing effectively, and reproducing our results.

## Installation

Follow these steps to set up the environment and install the necessary dependencies.

1. **Clone the repository:**
```bash
git clone https://github.com/nabindev3/Cascade-Engine.git
cd Cascade-Engine
```

2. **Create and activate the environment:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install the package and dependencies:**
```bash
pip install -r python_core/requirements.txt
```

4. **Download required models (for intelligent layers):**
```bash
python -m spacy download en_core_web_lg
```

## Getting Started

This library allows you to run multi-tier cascading strategies with intelligent preprocessing layers for any downstream LLM application. 

The router is fully **async** and **dependency-injected**: you build the engines,
the router config, and (optionally) the intelligent-layer orchestrator, then pass
them in. All payloads are validated by Pydantic models (`InferenceRequest` /
`InferenceResponse`).

### Step 1. Initialize the Engines
Define the tiers of models you want to use. We typically use a 3-tier system: Local, Mid-Cloud, and Premium-Cloud. Cloud engines accept an OpenAI-compatible config dict; the factories fill in sensible per-token pricing defaults.

```python
from python_core.engines.local_engine import OllamaEngine
from python_core.engines.cloud_engine import create_mid_tier_engine, create_premium_engine

tier1 = OllamaEngine(config={"model": "llama3.2:3b"})
tier2 = create_mid_tier_engine({"api_key": "YOUR_KEY", "model": "gpt-4o-mini"})
tier3 = create_premium_engine({"api_key": "YOUR_KEY", "model": "gpt-4o"})

engines = [tier1, tier2, tier3]
```

### Step 2. Build the Router
`CascadeRouter` takes the engines and a `RouterConfig`. The config controls the
confidence-gated cascade plus the production safety nets: **exponential backoff**
(in the cloud engines), **downgrade-to-local fallback**, and **risk-sensitive SLA
constraints**.

```python
import asyncio
from python_core.engines.base import InferenceRequest
from python_core.router.cascade_router import CascadeRouter, RouterConfig

router = CascadeRouter(
    engines=engines,
    config=RouterConfig(
        confidence_thresholds={1: 0.65, 2: 0.80},
        max_cost_per_request=0.05,     # cost SLO (USD)
        enable_local_fallback=True,    # downgrade to a local tier if the cloud fails
        enable_sla_constraints=True,   # honor per-request latency SLOs
        sla_risk_aversion=0.5,         # 0 = budget against p50, 1 = against the tail (~p99)
    ),
)
```

### Step 3. Run Queries through the Router
`route()` is a coroutine returning `(InferenceResponse, RoutingDecision)`. The
decision records the full routing path, escalation reasons, cost, and any SLA
violation — the data behind the dashboard and the paper.

```python
async def main():
    request = InferenceRequest(
        request_id="demo-1",
        prompt="Write a python script to reverse a linked list.",
        max_cost=0.02,         # per-request cost ceiling
        latency_slo_ms=1500,   # per-request latency SLO
    )
    response, decision = await router.route(request)

    print(response.content)
    print(f"Routed to: {response.engine_used if hasattr(response, 'engine_used') else response.engine_id}")
    print(f"Tier: {response.tier}  Cost: ${decision.total_cost_usd:.6f}")
    print(f"Path: {decision.engines_tried}  SLA violated: {decision.sla_violated}")

asyncio.run(main())
```

### (Optional) Wrap with the Intelligent Layers
To add PII masking, semantic caching, and gatekeeper/sarcasm routing hints around
*any* router, wrap it in an `OrchestrationWrapper`. Each wrapper owns its own
`IntelligentOrchestrator` (Presidio + FAISS + DistilBERT + VADER) so cache state
never leaks across instances.

```python
from python_core.router.intelligent_layers import IntelligentOrchestrator
from python_core.router.orchestration_wrapper import OrchestrationWrapper

wrapped = OrchestrationWrapper(inner_router=router, orchestrator=IntelligentOrchestrator())
response, decision = await wrapped.route(request)  # masks PII → cache → gatekeeper → route
```

## Architecture & Request Flow

When a user submits a query, it undergoes a sequential triage process designed to minimize costs while maximizing safety and speed. This ensures that expensive premium models are only called when absolutely necessary.

```mermaid
flowchart TD
    %% Define styles
    classDef request fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef filter fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef router fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef tier1 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef tier2 fill:#fff8e1,stroke:#fbc02d,stroke-width:2px;
    classDef tier3 fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    
    A(["User Request"]):::request --> B["PrivacyFilter <br/>(Presidio)"]:::filter
    B --> C{"SemanticCache <br/>(FAISS)"}:::filter
    
    C -- "Cache Hit" --> D(["Return Cached Response"]):::request
    
    C -- "Cache Miss" --> E["Gatekeeper & Intent <br/>(DistilBERT + VADER)"]:::filter
    E --> F{"Routing Decision"}:::router
    
    F -- "Simple / Factual" --> G["Tier 1: Local Model <br/>llama3.2:3b"]:::tier1
    F -- "Moderate" --> H["Tier 2: Mid-Cloud <br/>gpt-4o-mini"]:::tier2
    F -- "Complex Reasoning" --> I["Tier 3: Premium <br/>gpt-4o"]:::tier3
    
    G --> J[("Update Cache")]
    H --> J
    I --> J
    J --> K(["Final Response"]):::request
    D --> K
```

## API Gateway & Live Dashboard

A production-facing TypeScript gateway (`typescript_api/`) sits in front of the
Python core. It handles API-key auth, per-client rate limiting, request tracking,
and — when the core is unhealthy — a **circuit breaker with a direct cloud
fallback** so the service degrades gracefully instead of failing hard.

```bash
cd typescript_api
npm install
npm run dev          # gateway on :3000, expects the Python core on :8000
npm test             # vitest + supertest suite
```

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `CORE_SERVICE_URL` | URL of the Python core (default `http://localhost:8000`) |
| `API_KEYS` | Comma-separated valid keys for `Bearer` auth |
| `CORE_CIRCUIT_FAILURE_THRESHOLD` / `CORE_CIRCUIT_COOLDOWN_MS` | Circuit-breaker tuning |
| `FALLBACK_OPENAI_API_KEY` / `FALLBACK_MODEL` | Enable the degraded-mode direct cloud fallback |

**Live dashboard.** With the gateway running, open
[`http://localhost:3000/dashboard`](http://localhost:3000/dashboard). It polls
`/health` and `/v1/stats` every few seconds and shows system health, the circuit
breaker state, per-engine reliability (EMA), today's success rate / cost, and the
tier & failure-mode distributions. Paste an API key in the top-right to load the
authenticated routing stats.

## Reproducing Results

To reproduce the results presented in the paper, including the Pareto frontier evaluations on Alpaca-Eval and the ablation studies:

### 1. Run the Test Suite
Ensure all components are functioning correctly:
```bash
# Fast tests (skips loading heavy models)
pytest -m "not heavy"

# Full test suite
pytest
```

### 2. Run Benchmarks
Run the experiment script to execute the inference pipeline against the baseline models (e.g., RouteLLM):
```bash
python python_core/scripts/run_experiment.py
```
This will create a timestamped folder inside the `results/` directory containing `pareto.csv` and `manifest.json`.

### 3. Generate Paper Figures
Once your experiments have finished, you can generate the exact PDF plots used in the manuscript:
```bash
python -m python_core.scripts.make_figures results/<YOUR_TIMESTAMP_DIR>
```

## Code Structure

Below is a high-level overview of the code in this repository:

- **`python_core/engines/`**: Connectors to the underlying LLMs. `local_engine.py` handles local open-source models (via Ollama), while `cloud_engine.py` handles standard APIs (OpenAI).
- **`python_core/router/`**: The core logic of the framework.
  - `cascade_router.py`: The Frugal and base routing logic.
  - `learned_router.py`: Implementation of Contextual Discounted Thompson Sampling (CD-TS).
  - `intelligent_layers.py`: The preprocessing modules (SemanticCache, PrivacyFilter, Gatekeeper).
  - `benchmark.py`: Evaluates the routers against Alpaca-Eval.
- **`python_core/scripts/`**: Experiment execution (`run_experiment.py`) and visualization generation (`make_figures.py`).
- **`typescript_api/`**: The public API gateway (auth, rate limiting, circuit-breaker fallback) and the live dashboard (`public/dashboard.html`).
- **`paper/`**: The LaTeX source files and a compiled Markdown version of our academic manuscript.

## Citation

If you use this codebase or find our framework useful, please cite our paper:

```bibtex
@article{cascadeengine2026,
  title={Cascade Engine: A Multi-Tier, Intelligent Routing Framework for Cost-Effective LLM Inference},
  author={Nabin Prasad Dev},
  year={2026},
  journal={arXiv preprint}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
