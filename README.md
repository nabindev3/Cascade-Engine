# Cascade Engine

Cascade Engine is a multi-tier LLM inference router designed to optimize the balance between cost, speed, and accuracy. It intelligently routes incoming requests across different model tiers (e.g., local 3B models, mid-tier cloud models, and premium frontier models) to significantly reduce serving costs while maintaining high quality.

## Features

- **Multi-Tier Routing**: Dynamically routes requests between Tier 1 (local, low cost), Tier 2 (mid-cloud, balanced), and Tier 3 (premium cloud, high accuracy).
- **Intelligent Processing Layers**:
  - **Gatekeeper (DistilBERT)**: Fast request classification.
  - **PrivacyFilter (Presidio)**: PII detection and redaction for compliance.
  - **SemanticCache (FAISS)**: Vector similarity search to reuse previous high-quality responses.
  - **Sarcasm/Sentiment Detector (VADER)**: Analyzes user intent to inform routing.
- **Pareto-Frontier Benchmarking**: Rigorous evaluation using `Alpaca-Eval` to measure cost-accuracy trade-offs against baselines like RouteLLM.

## Architecture & Request Flow

The system consists of:
- `python_core/router/`: Contains routing logic, benchmark scripts, and intelligent layers.
- `python_core/engines/`: Connectors to local models (Ollama) and cloud APIs (OpenAI).

### How It Works

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

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cascade-engine.git
cd cascade-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r python_core/requirements.txt
```

## Running Benchmarks

```bash
# Run the test suite (skipping heavy tests that load large models)
pytest -m "not heavy"

# To run the full test suite including heavy ML layers
pytest

# Run a Pareto-frontier benchmarking experiment
python python_core/scripts/run_experiment.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
