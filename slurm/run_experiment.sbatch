#!/bin/bash
#SBATCH --job-name=chb_medagent
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Array job: 3 conditions × 4 student levels = 12 tasks
# Each task runs all cases × all seeds for one (condition, level) pair
set -euo pipefail

DATA="${DATA:-/path/to/your/scratch}"
SCRATCH_DIR="${DATA}/chb_medagent"
CODE_DIR="${SCRATCH_DIR}/code/system"

CONDITIONS=("multi_agent" "single_agent" "direct_qa")
LEVELS=("novice_y1" "intermediate_y3" "advanced_y4" "expert_y5")

COND_IDX=$((SLURM_ARRAY_TASK_ID / 4))
LEVEL_IDX=$((SLURM_ARRAY_TASK_ID % 4))

CONDITION="${CONDITIONS[$COND_IDX]}"
LEVEL="${LEVELS[$LEVEL_IDX]}"

echo "=== CHB Multi-Agent Experiment ==="
echo "Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Condition: ${CONDITION}, Level: ${LEVEL}"
echo "Node: $(hostname)"
nvidia-smi -L 2>/dev/null | head -4
echo "Time: $(date)"

# Environment
eval "$($DATA/miniconda3/bin/conda shell.bash hook)"
conda activate "$DATA/envs/medrag-pilot"

export HF_HOME="${DATA}/hf_cache"
# HF_TOKEN must be set in your shell before sbatch submit:
#   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
: "${HF_TOKEN:?HF_TOKEN environment variable must be set before sbatch submission}"
export HF_TOKEN
export VLLM_CACHE_ROOT="${DATA}/vllm_cache"
export TMPDIR="${DATA}/tmp"
export XDG_CACHE_HOME="${DATA}/.cache"
mkdir -p "${SCRATCH_DIR}/results" "${DATA}/tmp"

cd "${CODE_DIR}"

python run_experiment.py \
    --config config.yaml \
    --data-dir "${DATA}/medrag_pilot/data" \
    --output-dir "${SCRATCH_DIR}/results" \
    --condition "${CONDITION}" \
    --level "${LEVEL}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} Complete ==="
