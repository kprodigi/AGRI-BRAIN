#!/bin/bash
# Deprecated compatibility entry point.
#
# The publication workflow is orchestrated by hpc/hpc_run.sh and aggregated by
# hpc/hpc_publish.sh after both the seed and stress arrays complete. Keeping an
# independent legacy aggregation pipeline here previously allowed stale and
# non-publication diagnostics to be mixed into a new result bundle.

set -euo pipefail

echo "BLOCK: hpc/hpc_aggregate.sh is deprecated."
echo "Use: AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh"
echo "The dependent publication job will invoke hpc/hpc_publish.sh."
exit 2
