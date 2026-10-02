#!/bin/bash
#SBATCH --job-name=agri-weights
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
set -euo pipefail
: "${AGRI_WEIGHT_PACKAGE:?Missing package path}"
: "${AGRI_WEIGHT_OUTPUT:?Missing output path}"
: "${AGRI_WEIGHT_VENV:?Missing environment}"
cd "$AGRI_WEIGHT_PACKAGE"
source "$AGRI_WEIGHT_VENV/bin/activate"
source source/hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
KIND="${1:?Expected pilot, check or study}"
if [ "$KIND" = check ]; then
    python study.py pilot-check --output "$AGRI_WEIGHT_OUTPUT"
    exit
fi
: "${SLURM_ARRAY_TASK_ID:?Must run through a Slurm array}"
if [ "$KIND" = pilot ]; then
    PILOT_INDICES=(0 2 3 100 300 500 700 1500 1900)
    INDEX=${PILOT_INDICES[$SLURM_ARRAY_TASK_ID]}
elif [ "$KIND" = study ]; then
    INDEX=$((SLURM_ARRAY_TASK_ID + ${2:?Missing offset}))
else
    echo 'Unknown worker type' >&2; exit 2
fi
python study.py task --output "$AGRI_WEIGHT_OUTPUT" --index "$INDEX"
if [ "$KIND" = pilot ] && [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    REPEAT="$AGRI_WEIGHT_OUTPUT/reproducibility_repeat"
    if [ ! -f "$REPEAT/plan.json" ]; then python study.py prepare --output "$REPEAT"; fi
    python study.py task --output "$REPEAT" --index 0
fi
