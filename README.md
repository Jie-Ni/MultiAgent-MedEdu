# MultiAgent-MedEdu-Simulation


## Overview

A four-agent LLM architecture (Tutor, Patient, Assessment, Simulated Student)
that instantiates **knowledge-constrained generation** — explicit accessible
vocabularies (K⁺), forbidden vocabularies (K⁻), confusion tuples (C), and
maximum reasoning depth (d_max) — for simulating medical students at three
competence levels (Novice Y1, Intermediate Y3, Expert Y5) across 1,350
USMLE-style clinical-reasoning dialogues.

## Repository contents

```
.
├── run_experiment.py          # Main experiment entry point
├── config.yaml                # Knowledge specs per competence level, case mix, LLM config
├── requirements.txt
├── agents/
│   ├── base.py                # Base agent abstraction
│   ├── multi_agent.py         # Four-agent orchestration
│   └── prompts.py             # Role-specific system prompts
├── evaluation/
│   └── analyze_dialogues.py   # Fidelity metrics: Bloom, CoI, Q, BVR, TTR, etc.
├── utils/
├── slurm/
│   ├── run_experiment.sbatch  # SLURM array job: 3 conditions × 4 levels
│   └── analyze.sbatch         # Post-experiment metric computation
└── docs/
    └── research_design.md     # Experimental design rationale
```

## Requirements

- Python ≥ 3.10
- GPU with ≥ 24 GB VRAM (for Llama-3-8B-Instruct via vLLM)
- `pip install -r requirements.txt`
- HuggingFace token with access to `meta-llama/Meta-Llama-3-8B-Instruct`

## Quick start

```bash
# 1. Set your HuggingFace token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 2. Run one (condition, level) cell locally
python run_experiment.py \
    --config config.yaml \
    --condition multi_agent \
    --level novice_y1 \
    --output-dir ./results

# 3. Compute fidelity metrics
python evaluation/analyze_dialogues.py \
    --results-dir ./results \
    --output-dir ./analysis
```

On a SLURM cluster, edit `DATA` in `slurm/run_experiment.sbatch`, then:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
sbatch slurm/run_experiment.sbatch
sbatch --dependency=afterok:$PREV_JOBID slurm/analyze.sbatch
```

## Data

Dialogue corpus (1,350 JSON files, ~450 MB) and computed metrics are released
via a separate data archive linked from the paper.


## License

MIT License — see `LICENSE`.
