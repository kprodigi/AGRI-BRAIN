#!/bin/bash
# =========================================================================
# Import HPC results and push to GitHub
#
# Usage:
#   1. Copy hpc_results.tar.gz to C:/AgriBrain/
#   2. Run: bash hpc_import.sh
# =========================================================================
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f "hpc_results.tar.gz" ]; then
    echo "ERROR: hpc_results.tar.gz not found in $(pwd)"
    echo "Copy it from HPC first: scp <hpc>:<path>/hpc_results.tar.gz ."
    exit 1
fi

echo "=== Extracting HPC results ==="
tar xzf hpc_results.tar.gz
echo "Extracted $(tar tzf hpc_results.tar.gz | wc -l) files"

echo ""
echo "=== Validating ==="
.venv/Scripts/python.exe mvp/simulation/validation/validate_results.py

echo ""
echo "=== Changed files ==="
git status -s mvp/simulation/results/ mvp/simulation/baseline_snapshot.json

echo ""
echo "=== Ready to commit ==="
echo "Review the changes above, then run:"
echo "  git add mvp/simulation/results/ mvp/simulation/baseline_snapshot.json"
echo "  git commit -m 'Regenerate results from 20-seed HPC run'"
echo "  git push origin main"
