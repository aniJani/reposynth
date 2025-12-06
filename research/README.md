# ContextLoom Research

## Directory Structure

```
research/
├── README.md                    # This file
├── literature/                  # Literature review
│   ├── arpo_notes.md
│   ├── uncert_cot_notes.md
│   └── related_work.md
│
├── entropy/                     # Core entropy implementation
│   ├── __init__.py
│   ├── calculator.py           # Basic entropy functions
│   ├── token_classifier.py     # Code vs language classification
│   ├── cce.py                  # Contrastive Code Entropy
│   ├── monitor.py              # Uncertainty monitor
│   ├── spike_detector.py       # Spike detection
│   └── measurement.py          # Measurement strategies
│
├── retrieval/                   # Adaptive retrieval
│   ├── __init__.py
│   ├── topic_inference.py      # Infer topic from uncertainty
│   ├── adaptive.py             # Adaptive retriever
│   ├── context_manager.py      # Context window management
│   └── generator.py            # Generation with adaptive retrieval
│
├── visualization/               # Visualization system
│   ├── __init__.py
│   ├── data.py                 # Data structures
│   ├── api.py                  # FastAPI endpoints
│   └── charts.py               # Chart generation
│
├── benchmarks/                  # Evaluation benchmarks
│   ├── __init__.py
│   ├── dataset.py              # Dataset definitions
│   ├── examples/               # Benchmark examples
│   │   ├── easy/
│   │   ├── medium/
│   │   └── hard/
│   └── ground_truth.json       # Ground truth answers
│
├── baselines/                   # Baseline implementations
│   ├── __init__.py
│   ├── no_context.py
│   ├── full_context.py
│   ├── bm25_context.py
│   ├── embedding_context.py
│   └── uncert_cot.py
│
├── evaluation/                  # Evaluation framework
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   ├── runner.py               # Experiment runner
│   └── statistics.py           # Statistical analysis
│
├── experiments/                 # Experiment configurations
│   ├── configs/
│   │   ├── exp1_cce_vs_entropy.yaml
│   │   ├── exp2_adaptive_retrieval.yaml
│   │   ├── exp3_measurement_points.yaml
│   │   ├── exp4_thresholds.yaml
│   │   └── exp5_ablation.yaml
│   ├── scripts/
│   │   ├── run_all.sh
│   │   └── reproduce.py
│   └── results/
│       └── .gitkeep
│
├── notebooks/                   # Analysis notebooks
│   ├── 01_entropy_exploration.ipynb
│   ├── 02_cce_validation.ipynb
│   ├── 03_experiment_analysis.ipynb
│   └── 04_visualization.ipynb
│
└── paper/                       # Paper drafts
    ├── main.tex
    ├── figures/
    ├── tables/
    └── supplementary/
```

## Quick Start

### Phase 1: Setup
```bash
cd research
pip install -r requirements.txt
```

### Phase 2: Run Entropy Exploration
```bash
python -m notebooks.01_entropy_exploration
```

### Phase 3: Run Experiments
```bash
python experiments/scripts/run_all.py --config configs/exp1_cce_vs_entropy.yaml
```

## Key Files

| File | Purpose |
|------|---------|
| `entropy/cce.py` | Core CCE implementation (novel contribution) |
| `retrieval/adaptive.py` | Adaptive context retrieval |
| `evaluation/metrics.py` | All evaluation metrics |
| `experiments/runner.py` | Experiment orchestration |

## Citation

If you use this work, please cite:
```bibtex
@article{contextloom2025,
  title={Contrastive Code Entropy: Uncertainty-Guided Adaptive Context Retrieval for LLM Code Understanding},
  author={...},
  journal={...},
  year={2025}
}
```
