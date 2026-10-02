#!/bin/bash
#SBATCH --job-name=agri-weights-collect
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "${AGRI_WEIGHT_PACKAGE:?Missing package}"
source "${AGRI_WEIGHT_VENV:?Missing venv}/bin/activate"
source source/hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
python study.py pilot-check --output "${AGRI_WEIGHT_OUTPUT:?Missing output}"
python analyze.py --output "${AGRI_WEIGHT_OUTPUT:?Missing output}"
AGRI_PARENT=$(dirname "$AGRI_WEIGHT_OUTPUT")
AGRI_NAME=$(basename "$AGRI_WEIGHT_OUTPUT")
# First provide a compact analysis package; the full archive retains all episode evidence.
tar -czf "$AGRI_WEIGHT_OUTPUT/../${AGRI_NAME}_analysis.tar.gz" -C "$AGRI_WEIGHT_OUTPUT" analysis plan.json settings.json tasks.json COMPLETE.json
tar -czf "$AGRI_PARENT/${AGRI_NAME}.tar.gz" -C "$AGRI_PARENT" "$AGRI_NAME" -C "$(dirname "$AGRI_WEIGHT_PACKAGE")" "$(basename "$AGRI_WEIGHT_PACKAGE")"
cd "$AGRI_PARENT"
sha256sum "${AGRI_NAME}.tar.gz" > "${AGRI_NAME}.tar.gz.sha256"
sha256sum "${AGRI_NAME}_analysis.tar.gz" > "${AGRI_NAME}_analysis.tar.gz.sha256"
echo "Download $AGRI_PARENT/${AGRI_NAME}_analysis.tar.gz first, plus its .sha256."
echo "Full evidence: $AGRI_PARENT/${AGRI_NAME}.tar.gz and .sha256"
