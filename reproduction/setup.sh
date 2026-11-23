#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AGRI_PACKAGE=$(pwd -P)
export AGRI_WEIGHT_VENV="${AGRI_WEIGHT_VENV:-$(dirname "$AGRI_PACKAGE")/weight_sensitivity_venv}"
AGRI_PYTHON="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
"$AGRI_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Load module python/3.11 first"'
if [ ! -f "$AGRI_WEIGHT_VENV/bin/activate" ]; then
    "$AGRI_PYTHON" -m venv "$AGRI_WEIGHT_VENV"
fi
source "$AGRI_WEIGHT_VENV/bin/activate"
export PIP_NO_CACHE_DIR=1
python -m pip install -r source/agribrain/backend/requirements-lock.txt
# Build outside the immutable source snapshot.
BUILD_DIR=$(mktemp -d "$(dirname "$AGRI_PACKAGE")/weight_backend_build.XXXXXX")
cp -a source/agribrain/backend/. "$BUILD_DIR/"
python -m pip install "$BUILD_DIR" --no-deps
python -m pip check
printf 'export AGRI_WEIGHT_VENV=%q\n' "$AGRI_WEIGHT_VENV" > setup.env
echo 'Environment installed. Run: bash submit.sh [your sbatch account/partition options]'
