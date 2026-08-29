#!/bin/bash
# Aggregate the completed benchmark and stress arrays into one validated,
# commit-stamped publication artifact set.
#SBATCH --job-name=agribrain-publish
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/publish_%j.out
#SBATCH --error=logs/publish_%j.err

set -euo pipefail
if [ -z "${AGRIBRAIN_SOURCE_SNAPSHOT:-}" ]; then
    echo "BLOCK: AGRIBRAIN_SOURCE_SNAPSHOT not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: publisher is not executing from the declared source snapshot."
    exit 1
fi

# Compute nodes on the target cluster expose Git through a module rather than
# the default batch PATH.  Load it before the fail-closed source identity gate.
source hpc/ensure_git_available.sh

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi
if [ "${AGRIBRAIN_VENV:-}" != ".publication_venvs/${RUN_TAG}" ]; then
    echo "BLOCK: AGRIBRAIN_VENV is missing or does not match RUN_TAG."
    exit 1
fi
if [ ! -f "$AGRIBRAIN_VENV/bin/activate" ]; then
    echo "BLOCK: run-scoped venv not found: ${AGRIBRAIN_VENV}"
    exit 1
fi
source "$AGRIBRAIN_VENV/bin/activate"
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: stochastic publication run expected."
    exit 1
fi

source hpc/publication_env.sh
export ARTIFACT_MANIFEST_INCLUDE_RAW=1
export ARTIFACT_RUN_TAG="$RUN_TAG"
python hpc/validate_publication_env.py
python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
python hpc/capture_publication_environment.py --validate-only

RESULTS_DIR="mvp/simulation/results"
SEEDS_DIR="${RESULTS_DIR}/benchmark_seeds/${RUN_TAG}"
STRESS_DIR="${RESULTS_DIR}/stress_runs/${RUN_TAG}"
H3_LEDGER_ROOT="${RESULTS_DIR}/decision_ledger_h3/${RUN_TAG}"
EXPECTED_CORE_SUBMISSION_RECEIPT="${RESULTS_DIR}/core_submission_receipts/${RUN_TAG}.json"
EXPECTED_SEEDS=(42 1337 2024 7 99 101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515)
export BENCHMARK_SEEDS="42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515"

echo "[publish tag=${RUN_TAG}] starting at $(date)"
if [ ! -d "$SEEDS_DIR" ] || [ ! -d "$STRESS_DIR" ] || [ ! -d "$H3_LEDGER_ROOT" ]; then
    echo "BLOCK: tagged benchmark, stress, or H3 ledger outputs are missing."
    exit 1
fi
if [ "${CORE_SUBMISSION_RECEIPT:-}" != "$EXPECTED_CORE_SUBMISSION_RECEIPT" ]; then
    echo "BLOCK: core Slurm submission receipt path is missing or mismatched."
    exit 1
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "BLOCK: publisher SLURM_JOB_ID is missing. Submit via hpc/hpc_run.sh."
    exit 1
fi
python hpc/core_submission_receipt.py validate \
    --receipt "$CORE_SUBMISSION_RECEIPT" \
    --run-tag "$RUN_TAG" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --publisher-slurm-job-id "$SLURM_JOB_ID"

echo "=== Capture post-job Slurm accounting for all 25 simulation workers ==="
SCHEDULER_ACCOUNTING="${SEEDS_DIR}/slurm_simulation_accounting.json"
python hpc/capture_slurm_accounting.py \
    --submission-receipt "$CORE_SUBMISSION_RECEIPT" \
    --output "$SCHEDULER_ACCOUNTING" \
    --kind core \
    --run-tag "$RUN_TAG" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --source-tree-sha256 "$AGRIBRAIN_SOURCE_TREE_SHA256"

echo "=== Validate raw run identity and exact experimental panels ==="
python hpc/validate_raw_publication_inputs.py \
    --seed-root "$SEEDS_DIR" \
    --stress-root "$STRESS_DIR" \
    --h3-ledger-root "$H3_LEDGER_ROOT" \
    --submission-receipt "$CORE_SUBMISSION_RECEIPT" \
    --publisher-slurm-job-id "$SLURM_JOB_ID" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --run-tag "$RUN_TAG"

# Preserve obsolete, non-regenerated diagnostics outside the publication
# results tree so they cannot be mistaken for outputs of this commit. Their
# obsolete producers are now fail-closed; the filenames remain here only to
# quarantine residue copied from an older worktree.
LEGACY_DIR="hpc_legacy_results/${RUN_TAG}"
mkdir -p "$LEGACY_DIR"
for name in channel_hashseed_stability.json channel_spec_curve.json over_steer_ablation.json publication_validation_receipt.json; do
    if [ -f "${RESULTS_DIR}/${name}" ]; then
        mv "${RESULTS_DIR}/${name}" "${LEGACY_DIR}/${name}"
    fi
done

echo "=== Verify and stage the complete 20-seed panel ==="
mkdir -p "${RESULTS_DIR}/benchmark_seeds"
for seed in "${EXPECTED_SEEDS[@]}"; do
    src="${SEEDS_DIR}/seed_${seed}.json"
    if [ ! -s "$src" ]; then
        echo "BLOCK: missing or empty seed output: $src"
        exit 1
    fi
    cp -f "$src" "${RESULTS_DIR}/benchmark_seeds/seed_${seed}.json"
done

echo "=== Aggregate H1/H2 benchmark statistics and tables ==="
export AGRIBRAIN_PUBLICATION_AGGREGATION=1
python mvp/simulation/benchmarks/aggregate_seeds.py \
    --seed-root "${RESULTS_DIR}/benchmark_seeds" \
    --output-dir "$RESULTS_DIR" \
    --publication

echo "=== Consolidate and verify per-seed decision ledgers ==="
LEDGER_ROOT="${RESULTS_DIR}/decision_ledger_per_seed/${RUN_TAG}"
mkdir -p "$LEDGER_ROOT"
for seed in "${EXPECTED_SEEDS[@]}"; do
    src="${SEEDS_DIR}/decision_ledger_${seed}"
    dst="${LEDGER_ROOT}/seed_${seed}"
    if [ ! -d "$src" ]; then
        echo "BLOCK: missing decision-ledger directory: $src"
        exit 1
    fi
    count=$(find "$src" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    recursive_count=$(find "$src" -type f -name '*.jsonl' | wc -l)
    if [ "$count" -ne 55 ]; then
        echo "BLOCK: seed ${seed} has ${count} ledgers; expected 55 (11 modes x 5 scenarios)."
        exit 1
    fi
    if [ "$recursive_count" -ne "$count" ]; then
        echo "BLOCK: seed ${seed} contains nested ledger files."
        exit 1
    fi
    mkdir -p "$dst"
    while IFS= read -r -d '' ledger; do
        cp -f "$ledger" "$dst/$(basename "$ledger")"
    done < <(find "$src" -maxdepth 1 -type f -name '*.jsonl' -print0)
done
python hpc/validate_decision_ledgers.py \
    --ledger-root "$LEDGER_ROOT" \
    --seed-root "$SEEDS_DIR"

echo "=== Decision-level channel attribution (seed-cluster inference) ==="
python mvp/simulation/benchmarks/aggregate_channel_attribution.py \
    --ledger-root "$LEDGER_ROOT" \
    --output "${RESULTS_DIR}/channel_attribution_aggregate.json" \
    --modes agribrain
python mvp/simulation/_h2_permutation_test.py \
    --ledger-root "$LEDGER_ROOT" \
    --output "${RESULTS_DIR}/channel_complementarity_test.json"
python mvp/simulation/analysis/channel_saturation_analysis.py \
    --seed-root "${RESULTS_DIR}/benchmark_seeds" \
    --output "${RESULTS_DIR}/channel_saturation_analysis.json" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --run-tag "$RUN_TAG"

echo "=== Aggregate scenario-parallel H3 stress outputs ==="
python mvp/simulation/benchmarks/aggregate_stress_outputs.py \
    --input-root "$STRESS_DIR" --output-dir "$RESULTS_DIR" --publication

echo "=== Explainability and provenance integrity ==="
python -m mvp.simulation.analysis.explainability_metrics \
    --ledger "$LEDGER_ROOT" \
    --output "${RESULTS_DIR}/explainability_metrics.json" \
    --threshold 0.10

echo "=== Render every figure from the completed cache ==="
FIGURE_STAGE="${AGRIBRAIN_VENV}/figure_stage"
if [ -e "$FIGURE_STAGE" ]; then
    echo "BLOCK: run-scoped figure staging path already exists: ${FIGURE_STAGE}"
    exit 1
fi
mkdir "$FIGURE_STAGE"
# Render only from the exact flat seed envelopes that the manifest and
# publication archive retain.  Using the tagged source directory here would
# leave a trace-to-pixel provenance gap if either copy changed after staging.
export FIGURE_SEED_ROOT="${RESULTS_DIR}/benchmark_seeds"
export FIGURE_OUTPUT_DIR="$FIGURE_STAGE"
unset AGRIBRAIN_PUBLICATION_RENDER || true
python mvp/simulation/regenerate_figures_from_cache.py
python hpc/validate_and_promote_figures.py \
    --staging-dir "$FIGURE_STAGE" \
    --results-dir "$RESULTS_DIR" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --run-tag "$RUN_TAG"

echo "=== Export paper evidence and validate ==="
(cd mvp/simulation && python analysis/export_paper_evidence.py)
(cd mvp/simulation && python validation/validate_results.py)

echo "=== Recreate leakage-free internal forecast-validation receipt ==="
python mvp/simulation/validation/validate_forecasts.py \
    --output-dir "$RESULTS_DIR" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --run-tag "$RUN_TAG"

echo "=== Capture version-resolved publication environment ==="
python hpc/capture_publication_environment.py

echo "=== Build pre-receipt manifest and run every semantic gate ==="
(cd mvp/simulation && python analysis/build_artifact_manifest.py)
python mvp/simulation/analysis/verify_manifest.py --strict-commit
python mvp/simulation/validation/validate_publication_artifacts.py --write-receipt

echo "=== Hash-bind the semantic receipt and re-run the final validator ==="
(cd mvp/simulation && python analysis/build_artifact_manifest.py)
python mvp/simulation/analysis/verify_manifest.py --strict-commit
python mvp/simulation/validation/validate_publication_artifacts.py

BUNDLE="publication_bundle_${RUN_TAG}"
ARCHIVE="${BUNDLE}/hpc_results_${RUN_TAG}.tar.gz"
ARCHIVE_RECEIPT="${BUNDLE}/publication_archive_receipt_${RUN_TAG}.json"
# The builder writes archive, receipt, and READY marker in a temporary sibling
# directory, verifies them, then atomically promotes the complete bundle.
python mvp/simulation/analysis/build_publication_archive.py \
    --results-dir "$RESULTS_DIR" \
    --output "$ARCHIVE" \
    --receipt "$ARCHIVE_RECEIPT"
echo "[publish] ready bundle: ${BUNDLE}"
echo "[publish] archive: ${ARCHIVE}"
echo "[publish] completion marker: ${BUNDLE}/READY.json"

echo "=== Build separately verified lossless future-analysis evidence archive ==="
COMPLETE_EVIDENCE_BUNDLE="${RESULTS_DIR}/complete_run_evidence/${RUN_TAG}"
python hpc/build_complete_run_evidence.py \
    --input-root "core=${SEEDS_DIR}" \
    --input-root "h3_results=${STRESS_DIR}" \
    --input-root "h3_ledgers=${H3_LEDGER_ROOT}" \
    --input-file "core_submission_receipt.json=${CORE_SUBMISSION_RECEIPT}" \
    --output "$COMPLETE_EVIDENCE_BUNDLE" \
    --run-tag "$RUN_TAG" \
    --source-commit "$AGRIBRAIN_GIT_COMMIT" \
    --source-tree-sha256 "$AGRIBRAIN_SOURCE_TREE_SHA256" \
    --expected-manifests 25 \
    --expected-groups 1600 \
    --expected-episodes 6100 \
    --expected-adaptation-ledgers 4500 \
    --expected-final-ledgers 1600 \
    --expected-runtime-receipts 25 \
    --expected-scheduler-tasks 25
echo "[publish] complete raw evidence: ${COMPLETE_EVIDENCE_BUNDLE}"
python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
echo "[publish] complete at $(date)"
