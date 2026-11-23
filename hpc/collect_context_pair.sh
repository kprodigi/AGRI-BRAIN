#!/bin/bash
# Validate and archive ALL outputs, including routing ledgers and adaptation.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo 'Usage: bash hpc/collect_context_pair.sh /absolute/path/to/context_pair_RESULTS' >&2
    exit 2
fi
PAIR_RESULTS=$(cd "$1" && pwd -P)
source "$PAIR_RESULTS/run.env"
if [ "$NO_CONTEXT_OUTPUT_ROOT" != "$PAIR_RESULTS" ]; then
    echo "Result directory does not match its run.env" >&2
    exit 2
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
source hpc/ensure_git_available.sh
source "$AGRIBRAIN_VENV/bin/activate"
source hpc/publication_env.sh
python hpc/validate_source_checkout.py
python hpc/capture_publication_environment.py --validate-only
python -m mvp.simulation.benchmarks.verify_context_pair --output-root "$PAIR_RESULTS"
PAIR_ARCHIVE="${PAIR_RESULTS}.tar.gz"
if [ -e "$PAIR_ARCHIVE" ]; then
    echo "Archive already exists: $PAIR_ARCHIVE" >&2
    exit 2
fi
tar -czf "${PAIR_ARCHIVE}.partial" -C "$(dirname "$PAIR_RESULTS")" "$(basename "$PAIR_RESULTS")"
mv -- "${PAIR_ARCHIVE}.partial" "$PAIR_ARCHIVE"
sha256sum "$PAIR_ARCHIVE" > "${PAIR_ARCHIVE}.sha256"
echo "Download BOTH files:"
echo "$PAIR_ARCHIVE"
echo "${PAIR_ARCHIVE}.sha256"
