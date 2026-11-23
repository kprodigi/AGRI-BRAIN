#!/bin/bash
#SBATCH --job-name=agri-study-collect
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# Check the pilot again, verify and analyse every task, then archive the study.
# The analysis is complete when COMPLETE.json exists in the results folder. The
# collection is complete when <results>_COLLECTED.sha256 exists beside that
# folder: it is written last and holds the checksums of both archives.
set -euo pipefail
cd "${AGRI_STUDY_REPO:?Missing repository path}"
source hpc/ensure_git_available.sh
source "${AGRI_STUDY_VENV:?Missing environment}/bin/activate"
source hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
OUTPUT=$(cd "${AGRI_STUDY_OUTPUT:?Missing output path}" && pwd -P)
PARENT=$(dirname "$OUTPUT")
NAME=$(basename "$OUTPUT")
# A collection that stops early leaves no record of an earlier one: an earlier
# pass of the pilot check, an earlier analysis and an earlier collection do not
# stand in for this one.
rm -f "$PARENT/${NAME}_COLLECTED.sha256" "$OUTPUT/PILOT_PASSED.json" "$OUTPUT/COMPLETE.json"
python study/study.py pilot-check --output "$OUTPUT"
# Every task is verified again from its retained ledgers, in parallel.
python study/analyze.py --output "$OUTPUT" --workers "${SLURM_CPUS_PER_TASK:-1}"
cd "$PARENT"
pack() {  # archive, then the arguments of tar; the archive takes its name once it is whole
    local archive=$1
    shift
    rm -f "$archive.partial" "$archive.sha256"
    tar -czf "$archive.partial" "$@"
    mv -f "$archive.partial" "$archive"
    sha256sum "$archive" > "$archive.sha256"
}
# A compact package first: the tables and the records, without the ledgers of the
# controls. The full archive retains all the episode evidence.
pack "${NAME}_analysis.tar.gz" -C "$OUTPUT" --exclude='controls/*/*.jsonl.gz' analysis controls \
    plan.json settings.json tasks.json setup_environment.json PILOT_PASSED.json COMPLETE.json \
    job_ids.txt source.bundle
pack "${NAME}.tar.gz" "$NAME"
cat "${NAME}_analysis.tar.gz.sha256" "${NAME}.tar.gz.sha256" > "${NAME}_COLLECTED.sha256.partial"
mv -f "${NAME}_COLLECTED.sha256.partial" "${NAME}_COLLECTED.sha256"
echo "Collection complete: $PARENT/${NAME}_COLLECTED.sha256"
echo "Download $PARENT/${NAME}_analysis.tar.gz and its .sha256 first."
echo "Full evidence: $PARENT/${NAME}.tar.gz and its .sha256"
