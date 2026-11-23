#!/bin/bash
# Create the locked Python 3.11 environment of the study.
# Usage, from a clean checkout of the tagged commit: bash study/setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
AGRI_STUDY_REPO=$(pwd -P)
export AGRI_STUDY_VENV="${AGRI_STUDY_VENV:-$(dirname "$AGRI_STUDY_REPO")/study_venv}"
AGRI_PYTHON="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
"$AGRI_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Load the Python 3.11 module first"'
if [ ! -f "$AGRI_STUDY_VENV/bin/activate" ]; then
    "$AGRI_PYTHON" -m venv "$AGRI_STUDY_VENV"
fi
source "$AGRI_STUDY_VENV/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install -r agribrain/backend/requirements-lock.txt
# Build outside the checkout, which must stay clean.
BUILD_DIR=$(mktemp -d "$(dirname "$AGRI_STUDY_REPO")/study_backend_build.XXXXXX")
cp -a agribrain/backend/. "$BUILD_DIR/"
python -m pip install "$BUILD_DIR" --no-deps
python -m pip check
printf 'export AGRI_STUDY_VENV=%q\n' "$AGRI_STUDY_VENV" > study/setup.env
source hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
python -m pytest study -q -p no:cacheprovider
echo 'Environment installed and study tests passed.'
echo 'Run: bash study/submit.sh [your sbatch account/partition options]'
