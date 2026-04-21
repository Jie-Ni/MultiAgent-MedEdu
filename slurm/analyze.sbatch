#!/bin/bash
#SBATCH --job-name=chb_analyze
#SBATCH --partition=zen4_0768
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Runs AFTER full experiment completes (submitted with --dependency)
set -euo pipefail
DATA="${DATA:-/path/to/your/scratch}"

eval "$($DATA/miniconda3/bin/conda shell.bash hook)"
conda activate "$DATA/envs/medrag-pilot"
export PYTHONUNBUFFERED=1

cd "${DATA}/chb_medagent/code"

echo "=== Running Analysis ==="
python -u evaluation/analyze_dialogues.py \
    --results-dir "${DATA}/chb_medagent/results" \
    --output-dir "${DATA}/chb_medagent/analysis"

echo "=== Analysis Complete $(date) ==="
