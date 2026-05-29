"""Regenerate the paper notebooks from canonical templates.

The notebooks consume artifacts produced by `python_core.scripts.run_experiment`
rather than embedding their own benchmark runs. This keeps figures and tables
in sync with the canonical results directory and removes the path where a
notebook could silently disagree with the experiment artifact.

Run from the repo root:

    python paper/build_notebooks.py

Writes:
    notebooks/paper2_learned_routing.ipynb   (full rewrite, artifact-driven)
    notebooks/paper1_measurement_study.ipynb (updates §4 Pareto frontier to
                                              optionally load from artifacts)
"""

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# ─────────────────────────────────────────────────────────────────────────────
# Paper 2: learned routing — full rewrite, driven by run_experiment artifacts.
# ─────────────────────────────────────────────────────────────────────────────

PAPER2_TITLE_MD = """\
# Paper 2 — Learned Routing in Heterogeneous LLM Cascades

This notebook is a **reproducibility-faithful analysis** of a single benchmark
run produced by `python -m python_core.scripts.run_experiment`. It does **not**
run its own experiments — that would re-introduce the silent-disagreement bug
between notebook and canonical artifact that we explicitly designed against in
Phase 3. Every table and figure below is derived from `manifest.json`,
`raw_stats.json`, and `pareto.csv` under a single `RESULTS_DIR`.

To regenerate the artifacts before re-running this notebook:

```bash
python -m python_core.scripts.run_experiment \\
    --dataset tatsu-lab/alpaca_eval \\
    --max-samples 200 --seeds 5 \\
    --label "paper2-final"
```

The theoretical framing for the routing policies appears in `paper/theory.tex`.
"""

PAPER2_IMPORTS = """\
import json
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Point this at the directory produced by run_experiment.py.
# Pick the most recent run by default; override explicitly for reproducibility.
RESULTS_ROOT = Path('../results')
runs = sorted(RESULTS_ROOT.glob('*/')) if RESULTS_ROOT.exists() else []
if not runs:
    raise FileNotFoundError(
        f"No experiment runs found under {RESULTS_ROOT.resolve()}. "
        f"Generate one with:\\n"
        f"  python -m python_core.scripts.run_experiment --max-samples 200 --seeds 5"
    )
RESULTS_DIR = runs[-1]
print(f"Using run: {RESULTS_DIR}")

MANIFEST = json.loads((RESULTS_DIR / 'manifest.json').read_text())
RAW_STATS = json.loads((RESULTS_DIR / 'raw_stats.json').read_text())
PARETO = pd.read_csv(RESULTS_DIR / 'pareto.csv')
"""

PAPER2_PROVENANCE_MD = """\
## 1. Run Provenance

The reviewer-verifiable identity of this run. The `dataset.sha256` fixes the
inputs; `git_sha` + `git_dirty` fix the code; `package_versions` fix the
toolchain. Together these uniquely identify the experiment.
"""

PAPER2_PROVENANCE_CODE = """\
print(f"Run label:      {MANIFEST.get('label', '(none)')}")
print(f"Timestamp (UTC): {MANIFEST['timestamp_utc']}")
print(f"Git SHA:        {MANIFEST['git_sha']} (dirty={MANIFEST['git_dirty']})")
print(f"Python:         {MANIFEST['python_version']}")
print(f"Dataset:        {MANIFEST['dataset']['name']} (split={MANIFEST['dataset']['split']})")
print(f"  Loaded:       {MANIFEST['dataset']['n_loaded']} prompts")
print(f"  SHA-256:      {MANIFEST['dataset']['sha256']}")
print(f"Seeds:          {MANIFEST['experiment']['seeds']}")
print(f"System mode:    {MANIFEST['experiment']['system_mode']}")
print(f"Engines:        {MANIFEST['engines']['label']}")
print()
print('Package versions:')
for k, v in MANIFEST['package_versions'].items():
    print(f'  {k:<25s} {v}')
"""

PAPER2_PARETO_MD = """\
## 2. Cost vs Quality Pareto Frontier

The headline figure. Each point is one router at the run's mean cost and mean
quality across the `n_seeds` seeds; error bars are 95% confidence intervals
computed from the sample standard deviation using the t-distribution (df=n−1).
The green dashed line connects the non-dominated set: routers on this frontier
are not strictly worse than any other router in both cost and quality
simultaneously. Routers off the frontier are dominated and would not be
recommended by a Pareto-rational decision-maker.
"""

PAPER2_PARETO_CODE = """\
fig, ax = plt.subplots(figsize=(8, 6))

# Frontier polyline.
frontier = PARETO[PARETO['on_frontier'] == 1].sort_values('cost_mean')
if len(frontier) >= 2:
    ax.plot(frontier['cost_mean'], frontier['quality_mean'],
            '--', color='tab:green', alpha=0.6, label='Pareto frontier')

for _, r in PARETO.iterrows():
    on_f = bool(r['on_frontier'])
    ax.errorbar(
        r['cost_mean'], r['quality_mean'],
        xerr=r['cost_ci'], yerr=r['quality_ci'],
        fmt='o' if on_f else 'x',
        color='tab:green' if on_f else 'tab:gray',
        markersize=8, capsize=3,
    )
    ax.annotate(r['router'], (r['cost_mean'], r['quality_mean']),
                textcoords='offset points', xytext=(6, 6), fontsize=9)

ax.set_xlabel('Cost per request (USD)')
ax.set_ylabel('Quality (reward-model score)')
ax.set_title(f"Cost vs Quality — n={MANIFEST['experiment']['n_seeds']} seeds, 95% CI")
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
plt.tight_layout()
plt.show()
"""

PAPER2_BARS_MD = """\
## 3. Per-Router Aggregate Stats

Four panels: average cost, quality, latency, and success rate per router,
with 95% CI error bars. The same numbers feed the comparison table in §4.
"""

PAPER2_BARS_CODE = """\
def _stat(router, metric):
    return RAW_STATS[router][metric]['mean'], RAW_STATS[router][metric]['ci']

routers = sorted(RAW_STATS.keys(), key=lambda r: RAW_STATS[r]['cost']['mean'])
metrics = [('cost', 'Cost (USD)'),
           ('quality', 'Quality'),
           ('latency', 'Latency (ms)'),
           ('success', 'Success rate')]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (key, label) in zip(axes.flat, metrics):
    means = [RAW_STATS[r][key]['mean'] for r in routers]
    cis = [RAW_STATS[r][key]['ci'] for r in routers]
    ax.bar(range(len(routers)), means, yerr=cis, capsize=4,
           color='tab:blue', alpha=0.75)
    ax.set_xticks(range(len(routers)))
    ax.set_xticklabels(routers, rotation=30, ha='right')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
"""

PAPER2_TABLE_MD = """\
## 4. Paper-Ready Comparison Table

Mean ± 95% CI for every (router × metric) cell. This table goes directly into
the paper — copy `df.to_latex()` into the manuscript.
"""

PAPER2_TABLE_CODE = """\
def _fmt(stats, key, prec):
    m = stats[key]['mean']
    c = stats[key]['ci']
    return f"{m:.{prec}f} ± {c:.{prec}f}"

rows = []
for r in routers:
    s = RAW_STATS[r]
    rows.append({
        'Router': r,
        'Cost ($/req)':   _fmt(s, 'cost', 5),
        'Quality':        _fmt(s, 'quality', 4),
        'Latency (ms)':   _fmt(s, 'latency', 0),
        'Success rate':   _fmt(s, 'success', 3),
        'On Pareto':      'yes' if any(
            (PARETO['router'] == r) & (PARETO['on_frontier'] == 1)
        ) else 'no',
    })
df = pd.DataFrame(rows)
df
"""

PAPER2_THEORY_MD = """\
## 5. Theoretical Predictions

The CD-TS regret analysis in `paper/theory.tex` predicts:

$$
\\mathbb{E}[R_T] \\;=\\; O\\!\\left((MK \\log T)^{1/3} \\cdot V_T^{1/3} \\cdot T^{2/3}\\right)
$$

For the deployment parameters here — $M = 5$ complexity bins, $K \\in \\{2, 3\\}$
engine tiers, and $T$ in the hundreds — the **rate constant matters more than
the asymptotic exponent**: the implied effective-window optimum is

$$
L^\\star \\;=\\; \\Theta\\!\\big((T \\sqrt{MK \\log T} / V_T)^{2/3}\\big)
$$

which for $T = 200$, $MK = 15$, $V_T \\approx 5$ gives $L^\\star \\approx 50$
rounds. The default `decay_factor = 0.995` corresponds to $L = 200$ — slightly
slow for this regime; tuning to $\\gamma \\approx 0.98$ would track drift more
sharply at the cost of higher stationary variance.

The MDP router has no comparable finite-sample guarantee under our current
$\\varepsilon$-greedy parameterization (asymptotic convergence only). For
sample-efficient deployment, CD-TS is the recommended policy.
"""

PAPER2_NEXT_MD = """\
## 6. Re-running this analysis

To swap the dataset, increase the seed count, or compare system-mode against
pure routing, regenerate the artifact and re-run this notebook:

```bash
# Pure routing comparison (apples-to-apples, fair for the policy contribution):
python -m python_core.scripts.run_experiment --max-samples 500 --seeds 10 \\
    --label "paper2-pure-routing"

# Full system stack (cache + privacy + gatekeeper applied to every router):
python -m python_core.scripts.run_experiment --max-samples 500 --seeds 10 \\
    --system-mode --label "paper2-system-mode"
```

Both runs produce independent `results/<timestamp>/` directories with their own
manifests. To analyze a specific one, set `RESULTS_DIR` explicitly in cell 1.
"""


def build_paper2_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(PAPER2_TITLE_MD),
        code(PAPER2_IMPORTS),
        md(PAPER2_PROVENANCE_MD),
        code(PAPER2_PROVENANCE_CODE),
        md(PAPER2_PARETO_MD),
        code(PAPER2_PARETO_CODE),
        md(PAPER2_BARS_MD),
        code(PAPER2_BARS_CODE),
        md(PAPER2_TABLE_MD),
        code(PAPER2_TABLE_CODE),
        md(PAPER2_THEORY_MD),
        md(PAPER2_NEXT_MD),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Paper 1: measurement study — keep most of it; rewrite §4 Pareto to consume
# the canonical artifact when available, and add a header pointing at Paper 2
# for the policy-comparison analysis.
# ─────────────────────────────────────────────────────────────────────────────


PAPER1_HEADER_MD = """\
# Paper 1 — Measurement Study: Multi-Tier LLM Inference Characterization

> **Note on scope.** This notebook analyzes the *operational characteristics*
> of a deployed multi-tier system from production event logs (`data/logs/*.jsonl`).
> The *policy comparison* (cost/quality Pareto across routing strategies) lives
> in `paper2_learned_routing.ipynb` and is driven by the canonical experiment
> artifact in `results/<timestamp>/`. The two notebooks are complementary:
> Paper 1 characterizes what the live system does; Paper 2 evaluates which
> policy would be best.

**Target venues:** IEEE Access, ICSE-SEIP, ASE
"""

PAPER1_LOAD_LOGS_MD = """\
## 1. Load Inference Events

Loads JSONL event logs from `data/logs/` (one file per day, produced by
`monitor/event_logger.py`). If no logs are present, falls back to a synthetic
event generator solely so this notebook is browseable on a fresh checkout —
**figures from synthetic data are clearly marked and must not appear in the
paper.**
"""

PAPER1_LOAD_LOGS_CODE = """\
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = Path('../data/logs')
SYNTHETIC = False

events = []
for f in sorted(LOG_DIR.glob('inference_events_*.jsonl')):
    with open(f) as fp:
        for line in fp:
            events.append(json.loads(line))

if not events:
    SYNTHETIC = True
    print('No production logs found — generating SYNTHETIC events for notebook preview.')
    print('Any figures below are NOT paper-eligible until real logs are present.')
    rng = np.random.default_rng(42)
    n = 2000
    final_tier = rng.choice([1, 2, 3], size=n, p=[0.55, 0.30, 0.15])
    events = []
    for i in range(n):
        tier = int(final_tier[i])
        cost = {1: 0.0001, 2: 0.001, 3: 0.01}[tier] * rng.uniform(0.5, 1.5)
        latency = {1: 50, 2: 300, 3: 800}[tier] * rng.uniform(0.7, 1.5)
        events.append({
            'event_id': f'syn-{i:04d}',
            'final_tier': tier,
            'total_cost_usd': cost,
            'total_latency_ms': latency,
            'success': bool(rng.random() > 0.05),
            'confidence': float(rng.uniform(0.5, 0.99)),
            'failure_mode': 'none' if rng.random() > 0.08 else rng.choice(
                ['timeout', 'rate_limit', 'parse_error', 'semantic_failure']),
            'has_disagreement': bool(rng.random() < 0.18),
        })

df = pd.DataFrame(events)
print(f'Loaded {len(df)} events. SYNTHETIC = {SYNTHETIC}.')
df.head()
"""

PAPER1_TIER_DIST_MD = """\
## 2. Tier Distribution

**Research question:** What fraction of requests are absorbed by lower tiers?
"""

PAPER1_TIER_DIST_CODE = """\
fig, ax = plt.subplots(figsize=(7, 4))
counts = df['final_tier'].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values,
       color=['#9bd99b', '#f9c069', '#e88989'])
ax.set_xlabel('Final tier')
ax.set_ylabel('Number of requests')
ax.set_title(f"Tier distribution (n={len(df)}) {'[SYNTHETIC]' if SYNTHETIC else ''}")
for i, v in enumerate(counts.values):
    ax.text(i, v, f'{v}\\n({v/len(df):.1%})', ha='center', va='bottom')
plt.tight_layout()
plt.show()
"""

PAPER1_FAILURE_MD = """\
## 3. Failure Mode Analysis

**Research question:** What are the dominant failure modes per tier?
"""

PAPER1_FAILURE_CODE = """\
fail = df[df['failure_mode'] != 'none']
if len(fail) == 0:
    print('No failures recorded in this dataset.')
else:
    pivot = fail.groupby(['final_tier', 'failure_mode']).size().unstack(fill_value=0)
    if pivot.empty or pivot.values.sum() == 0:
        print(f'Pivot is empty ({len(fail)} failure rows; no plottable data). '
              f'Sample failure rows:')
        print(fail[['final_tier', 'failure_mode']].head())
    else:
        pivot.plot(kind='bar', stacked=True, figsize=(8, 4), colormap='tab10')
        plt.title(f"Failure modes by tier {'[SYNTHETIC]' if SYNTHETIC else ''}")
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
        try:
            display(pivot)
        except NameError:
            print(pivot)
"""

PAPER1_PARETO_MD = """\
## 4. Cost / Quality Pareto Frontier (from canonical experiment)

This section **does not** re-run experiments. It loads the latest canonical
result emitted by `python -m python_core.scripts.run_experiment` and reproduces
the Pareto plot from that artifact. If you want a different artifact, set
`RESULTS_DIR` explicitly below.

For the original log-only Pareto analysis (each request's cost vs confidence,
not router-vs-router), see the supplementary cell at the end of this section.
"""

PAPER1_PARETO_CODE = """\
RESULTS_ROOT = Path('../results')
runs = sorted(RESULTS_ROOT.glob('*/')) if RESULTS_ROOT.exists() else []
if not runs:
    print('No experiment artifacts found under ../results/.')
    print('Generate one with: python -m python_core.scripts.run_experiment')
else:
    RESULTS_DIR = runs[-1]
    pareto = pd.read_csv(RESULTS_DIR / 'pareto.csv')
    manifest = json.loads((RESULTS_DIR / 'manifest.json').read_text())

    fig, ax = plt.subplots(figsize=(8, 6))
    frontier = pareto[pareto['on_frontier'] == 1].sort_values('cost_mean')
    if len(frontier) >= 2:
        ax.plot(frontier['cost_mean'], frontier['quality_mean'],
                '--', color='tab:green', alpha=0.6, label='Pareto frontier')
    for _, r in pareto.iterrows():
        on_f = bool(r['on_frontier'])
        ax.errorbar(r['cost_mean'], r['quality_mean'],
                    xerr=r['cost_ci'], yerr=r['quality_ci'],
                    fmt='o' if on_f else 'x',
                    color='tab:green' if on_f else 'tab:gray',
                    markersize=8, capsize=3)
        ax.annotate(r['router'], (r['cost_mean'], r['quality_mean']),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xlabel('Cost per request (USD)')
    ax.set_ylabel('Quality (reward-model score)')
    ax.set_title(f"Pareto frontier — {manifest['experiment']['n_seeds']} seeds, 95% CI")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.show()
    print(f"Source: {RESULTS_DIR}/pareto.csv (dataset SHA-256 {manifest['dataset']['sha256'][:16]}...)")
"""

PAPER1_LOG_PARETO_MD = """\
### 4a. Supplementary: per-request cost vs confidence (log-only)

This is the original log-based analysis. It shows the distribution of cost and
self-reported confidence across **individual requests**, not router strategies.
The router-vs-router comparison above is the paper-eligible result.
"""

PAPER1_LOG_PARETO_CODE = """\
ok = df[df['success'] == True]
fig, ax = plt.subplots(figsize=(7, 5))
for tier in sorted(ok['final_tier'].unique()):
    sub = ok[ok['final_tier'] == tier]
    ax.scatter(sub['total_cost_usd'], sub['confidence'], alpha=0.4,
               label=f'Tier {tier} (n={len(sub)})', s=15)
ax.set_xlabel('Cost per request (USD)')
ax.set_ylabel('Self-reported confidence')
ax.set_xscale('log')
ax.set_title(f"Cost vs confidence by tier {'[SYNTHETIC]' if SYNTHETIC else ''}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

PAPER1_SUMMARY_MD = """\
## 5. Summary Table (paper Section IV)
"""

PAPER1_SUMMARY_CODE = """\
summary = df.groupby('final_tier').agg(
    n_requests=('event_id', 'count'),
    avg_cost=('total_cost_usd', 'mean'),
    avg_latency=('total_latency_ms', 'mean'),
    success_rate=('success', 'mean'),
).round({'avg_cost': 5, 'avg_latency': 0, 'success_rate': 3})
summary['share'] = (summary['n_requests'] / summary['n_requests'].sum()).round(3)
summary
"""

PAPER1_COST_SAVINGS_MD = """\
## 6. Cost Savings

Headline claim: **X% of requests can be handled at Tier 1, saving Y% of costs**
versus an always-premium baseline. The cost-savings number here is computed
from production logs; the policy comparison that produces this distribution
lives in `paper2_learned_routing.ipynb`.
"""

PAPER1_COST_SAVINGS_CODE = """\
actual = df['total_cost_usd'].sum()
premium_only_per_call = 0.01
always_premium = premium_only_per_call * len(df)
print(f"Actual system spend:     ${actual:.4f}")
print(f"Always-premium baseline: ${always_premium:.4f}")
print(f"Savings vs premium:      {(1 - actual/always_premium):.1%}")
tier1_share = (df['final_tier'] == 1).mean()
print(f"Share routed at Tier 1:  {tier1_share:.1%}")
"""


def build_paper1_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(PAPER1_HEADER_MD),
        md(PAPER1_LOAD_LOGS_MD),
        code(PAPER1_LOAD_LOGS_CODE),
        md(PAPER1_TIER_DIST_MD),
        code(PAPER1_TIER_DIST_CODE),
        md(PAPER1_FAILURE_MD),
        code(PAPER1_FAILURE_CODE),
        md(PAPER1_PARETO_MD),
        code(PAPER1_PARETO_CODE),
        md(PAPER1_LOG_PARETO_MD),
        code(PAPER1_LOG_PARETO_CODE),
        md(PAPER1_SUMMARY_MD),
        code(PAPER1_SUMMARY_CODE),
        md(PAPER1_COST_SAVINGS_MD),
        code(PAPER1_COST_SAVINGS_CODE),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def main():
    out = Path(__file__).resolve().parent.parent / "notebooks"
    out.mkdir(exist_ok=True)
    nb2 = build_paper2_notebook()
    nb1 = build_paper1_notebook()
    nbf.write(nb2, out / "paper2_learned_routing.ipynb")
    nbf.write(nb1, out / "paper1_measurement_study.ipynb")
    print(f"Wrote {out / 'paper2_learned_routing.ipynb'} ({len(nb2.cells)} cells)")
    print(f"Wrote {out / 'paper1_measurement_study.ipynb'} ({len(nb1.cells)} cells)")


if __name__ == "__main__":
    main()
