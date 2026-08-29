"""Fail-closed contracts for the Slurm recovery launchers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_dual_recovery_orchestrator_holds_authorizes_then_releases() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    assert script.count("--hold") == 3
    assert "trap cancel_held_recovery_jobs_on_failure EXIT" in script
    assert 'scancel "$job_id"' in script
    assert script.count("require_user_held_job") >= 7
    assert '[[ ! "$CORE_RECOVERY_JOB" =~ ^[1-9][0-9]*$ ]]' in script
    assert '[ "$STRUCTURAL_RECOVERY_JOB" = "$CORE_RECOVERY_JOB" ]' in script
    release = script.index(
        'scontrol release "${CORE_RECOVERY_JOB},${STRUCTURAL_RECOVERY_JOB},'
        '${FINALIZER_RECOVERY_JOB}"'
    )
    assert script.rfind('publication_recovery_receipt.py" validate', 0, release) >= 0
    assert script.rfind('require_user_held_job "$CORE_RECOVERY_JOB"', 0, release) >= 0
    assert script.rfind('require_user_held_job "$STRUCTURAL_RECOVERY_JOB"', 0, release) >= 0
    assert script.rfind("require_held_finalizer_dependency", 0, release) >= 0
    assert script.rfind('finalizer_submission_authorization.py" validate', 0, release) >= 0
    assert script.rfind("--require-live-held", 0, release) >= 0
    assert '"afterok:${CORE_RECOVERY_JOB}:${STRUCTURAL_RECOVERY_JOB}"' in script
    assert '"afterok:${CORE_RECOVERY_JOB},afterok:${STRUCTURAL_RECOVERY_JOB}"' in script
    assert r"\(unfulfilled\)" in script and r"\(satisfied\)" in script
    assert "SLURM_STATE_MAX_ATTEMPTS=120" in script
    assert "SLURM_STATE_RETRY_SECONDS=1" in script
    assert script.count("attempt <= SLURM_STATE_MAX_ATTEMPTS") == 3
    assert script.count('sleep "$SLURM_STATE_RETRY_SECONDS"') == 3
    assert "did not settle as PENDING/User-held" in script
    assert "did not settle with the exact two-publisher afterok dependency" in script
    assert "slurm_job_fields()" in script
    assert 'JobId=*) observed_job_id="${field#JobId=}"' in script
    assert 'JobState=*) state="${field#JobState=}"' in script
    assert 'Reason=*) reason="${field#Reason=}"' in script
    assert script.count(
        'IFS="|" read -r observed_job_id state reason <<< "$observed"'
    ) == 2
    assert '*" JobState=PENDING "*" Reason=JobHeldUser "*' not in script


def test_slurm_state_parser_accepts_exact_tokens_and_rejects_held_release() -> None:
    if os.name == "nt":
        pytest.skip("behavioral Bash helper test runs on POSIX CI/HPC")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable")
    script = _text("hpc/publication_recovery_run.sh")
    start = script.index("SLURM_STATE_MAX_ATTEMPTS=")
    end = script.index("cancel_held_recovery_jobs_on_failure()")
    helpers = script[start:end]
    program = f"""
set -euo pipefail
{helpers}
scontrol() {{
    test "$1" = show
    test "$2" = job
    test "$3" = -o
    printf 'JobId=%s JobName=test JobState=%s Reason=%s Dependency=(null)\\n' \\
        "$4" "$FAKE_STATE" "$FAKE_REASON"
}}
SLURM_STATE_MAX_ATTEMPTS=1
FAKE_STATE=PENDING
FAKE_REASON=JobHeldUser
require_user_held_job 12345
FAKE_STATE=RUNNING
FAKE_REASON=None
require_not_user_held_job 12345
FAKE_STATE=PENDING
FAKE_REASON=JobHeldUser
if require_not_user_held_job 12345; then
    exit 91
fi
printf 'SLURM_STATE_PARSER_OK\\n'
"""
    completed = subprocess.run(
        [bash, "-c", program],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SLURM_STATE_PARSER_OK" in completed.stdout


def test_physical_regular_file_normalizes_ancestor_alias_and_rejects_leaf_link(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("behavioral Bash path test runs on POSIX CI/HPC")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable")
    real_logs = tmp_path / "physical" / "logs"
    real_logs.mkdir(parents=True)
    real_file = real_logs / "publish_103.out"
    real_file.write_text("failed\n", encoding="utf-8")
    alias = tmp_path / "scratch"
    alias.symlink_to(tmp_path / "physical", target_is_directory=True)
    leaf_alias = real_logs / "publish_alias.out"
    leaf_alias.symlink_to(real_file)

    script = _text("hpc/publication_recovery_run.sh")
    start = script.index("physical_regular_file()")
    end = script.index('CORE_RECOVERY_JOB=""', start)
    helper = script[start:end]
    program = f"""
set -euo pipefail
{helper}
observed="$(physical_regular_file "$1" test-log)"
test "$observed" = "$2"
if physical_regular_file "$3" leaf-link >/dev/null 2>&1; then
    exit 91
fi
printf 'PHYSICAL_REGULAR_FILE_OK\\n'
"""
    completed = subprocess.run(
        [
            bash,
            "-c",
            program,
            "path-test",
            str(alias / "logs" / real_file.name),
            str(real_file.resolve()),
            str(leaf_alias),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PHYSICAL_REGULAR_FILE_OK" in completed.stdout


def test_failed_log_paths_are_physically_normalized_before_submission() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    first_submission = script.index("CORE_SUBMISSION=\"$(sbatch")
    for variable in (
        "AGRIBRAIN_CORE_FAILED_STDOUT",
        "AGRIBRAIN_CORE_FAILED_STDERR",
        "AGRIBRAIN_STRUCTURAL_FAILED_STDOUT",
        "AGRIBRAIN_STRUCTURAL_FAILED_STDERR",
    ):
        normalization = script.index(
            f'{variable}="$(physical_regular_file',
        )
        assert normalization < first_submission


def test_finalizer_scheduler_authorization_is_persisted_and_consumed() -> None:
    orchestrator = _text("hpc/publication_recovery_run.sh")
    finalizer = _text("hpc/hpc_full_submission_recovery.sh")
    ready_validator = _text("hpc/validate_full_submission_ready.py")
    assert 'finalizer_submission_authorization.py" create' in orchestrator
    assert "AGRIBRAIN_FINALIZER_AUTHORIZATION" in orchestrator
    assert 'RUN_TAG="finalizer_${AGRIBRAIN_CORE_RUN_TAG}"' in orchestrator
    assert "AGRIBRAIN_FINALIZER_AUTHORIZATION" in finalizer
    assert "finalizer_submission_authorization.py validate" in finalizer
    assert "FINALIZER_SUBMISSION_AUTHORIZATION.json" in finalizer
    assert '"finalizer_scheduler_authorization"' in ready_validator
    authorization = _text("hpc/finalizer_submission_authorization.py")
    assert '"observed_held_scheduler_records"' in authorization
    for role in ("core", "structural", "finalizer"):
        assert f'"{role}"' in authorization


def test_finalizer_validates_exact_bundle_before_and_after_atomic_promotion() -> None:
    finalizer = _text("hpc/hpc_full_submission_recovery.sh")
    promotion = finalizer.index("os.rename(stage, target)")
    validator = "python hpc/validate_full_submission_ready.py validate"
    assert finalizer.count(validator) == 2
    assert finalizer.index(validator) < promotion < finalizer.rindex(validator)
    assert finalizer.count("python hpc/validate_lexical_path.py") >= 3
    assert 'validate_lexical_inputs "$BOOTSTRAP_PYTHON"' in finalizer
    assert 'validate_lexical_inputs\n' in finalizer


def test_uncertain_release_preserves_authorization_and_requests_cancellation() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    release = script.index(
        'scontrol release "${CORE_RECOVERY_JOB},${STRUCTURAL_RECOVERY_JOB},'
        '${FINALIZER_RECOVERY_JOB}"'
    )
    assert script.rfind("RELEASE_ATTEMPTED=true", 0, release) >= 0
    assert 'if [ "$RELEASE_ATTEMPTED" = true ]' in script
    assert "Preserving immutable recovery-attempt evidence" in script
    assert "Cancellation requested for publication-only recovery job" in script
    assert script.find('require_not_user_held_job "$CORE_RECOVERY_JOB"', release) > release


def test_recovery_orchestrator_never_submits_simulation_workers() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    for forbidden in (
        "hpc/hpc_seed.sh",
        "hpc/hpc_stress.sh",
        "hpc/hpc_sensitivity_task.sh",
        "run_single_seed.py",
        "run_stress_suite.py",
    ):
        assert forbidden not in script
    assert "hpc/hpc_publish_recovery.sh" in script
    assert "hpc/hpc_sensitivity_publish_recovery.sh" in script
    assert "hpc/hpc_full_submission_recovery.sh" in script


def test_combined_finalizer_is_publication_only_and_lossless() -> None:
    orchestrator = _text("hpc/publication_recovery_run.sh")
    finalizer = _text("hpc/hpc_full_submission_recovery.sh")
    assert '--dependency="afterok:${CORE_RECOVERY_JOB}:${STRUCTURAL_RECOVERY_JOB}"' in orchestrator
    for argument in (
        "--core-archive",
        "--core-receipt",
        "--core-ready",
        "--core-complete-archive",
        "--core-complete-receipt",
        "--core-complete-ready",
        "--structural-archive",
        "--structural-receipt",
    ):
        assert argument in finalizer
    assert "FULL_SUBMISSION_EVIDENCE_RECEIPT.json" in finalizer
    assert 'READY_NAME="READY.json"' in finalizer
    assert "simulation_rerun" in _text("hpc/validate_full_submission_ready.py")
    assert "os.rename(stage, target)" in finalizer
    for forbidden in (
        "hpc/hpc_seed.sh",
        "hpc/hpc_stress.sh",
        "hpc/hpc_sensitivity_task.sh",
        "run_single_seed.py",
        "run_stress_suite.py",
    ):
        assert forbidden not in finalizer


def test_three_jobs_use_independent_clean_publication_worktrees() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    for variable, directory in (
        ("CORE_PUBLICATION_SNAPSHOT", "publication_source_core"),
        ("STRUCTURAL_PUBLICATION_SNAPSHOT", "publication_source_structural"),
        ("FINALIZER_PUBLICATION_SNAPSHOT", "publication_source_finalizer"),
    ):
        assert f'{variable}="${{CONTROL_ROOT}}/{directory}"' in script
        assert f'--chdir="${variable}"' in script
    assert 'git worktree add --detach "$snapshot" "$PUBLICATION_COMMIT"' in script
    assert script.count('prepare_recovery_venv "') >= 3
    assert "source hpc/publication_env.sh" in script
    assert "source hpc/publication_env.sh" in _text("hpc/hpc_full_submission_recovery.sh")
    assert "source hpc/ensure_git_available.sh" in _text("hpc/hpc_full_submission_recovery.sh")
    assert "publication worktrees do not have one identical source digest" in script
    assert 'CORE_BUNDLE="${CORE_PUBLICATION_SNAPSHOT}/' in script


def test_finalizer_revalidates_and_persists_locked_runtime_environment() -> None:
    finalizer = _text("hpc/hpc_full_submission_recovery.sh")
    ready_validator = _text("hpc/validate_full_submission_ready.py")
    assert "python hpc/validate_publication_env.py" in finalizer
    assert finalizer.count("python hpc/capture_publication_environment.py --validate-only") >= 2
    assert "FINALIZER_PUBLICATION_ENVIRONMENT.json" in finalizer
    assert '--output "${STAGE}/${ENVIRONMENT_NAME}"' in finalizer
    assert '"finalizer_publication_environment"' in ready_validator


def test_structural_recovery_attempt_is_job_scoped_and_retry_safe() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    parsed_job = script.index(
        'STRUCTURAL_RECOVERY_JOB="${STRUCTURAL_SUBMISSION%%;*}"'
    )
    held = script.index(
        'require_user_held_job "$STRUCTURAL_RECOVERY_JOB"', parsed_job,
    )
    attempt = script.index(
        'STRUCTURAL_ATTEMPT_ROOT="${STRUCTURAL_RUN_DIR}/'
        'publication_recovery_attempts/${STRUCTURAL_RECOVERY_JOB}"',
        held,
    )
    finalizer_submission = script.index('FINALIZER_SUBMISSION="$(sbatch', attempt)
    assert parsed_job < held < attempt < finalizer_submission
    assert 'if attempt_root.exists() or attempt_root.is_symlink():' in script
    assert 'with target.open("xb") as stream:' in script
    assert (
        'STRUCTURAL_RECOVERY_RECEIPT="${STRUCTURAL_ATTEMPT_ROOT}/'
        'publication_recovery_receipts/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"'
    ) in script
    assert (
        'STRUCTURAL_ATTEMPT_RAW_MANIFEST="${STRUCTURAL_ATTEMPT_ROOT}/'
        'preserved_raw_manifests/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"'
    ) in script
    assert 'STRUCTURAL_CANONICAL_RAW' not in script
    assert 'STRUCTURAL_CANONICAL_RECEIPT' not in script
    assert 'rm -- "$STRUCTURAL_ATTEMPT_ROOT"' not in script
    assert 'Structural recovery attempt directory: ${STRUCTURAL_ATTEMPT_ROOT}' in script
    assert 'Structural recovery archive destination: ${STRUCTURAL_ARCHIVE}' in script
    assert (
        'echo "BLOCK: structural derived output already exists: '
        '${STRUCTURAL_RUN_DIR}/${output}"'
    ) not in script


def test_structural_attempt_creator_preserves_prior_attempt_and_legacy_outputs(
    tmp_path: Path,
) -> None:
    script = _text("hpc/publication_recovery_run.sh")
    section = script.split(
        "# The actual held Slurm job ID is the immutable attempt identifier.",
        1,
    )[1]
    program = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(program, "publication_recovery_run.sh:attempt_creator", "exec")

    run_root = tmp_path / "structural-run"
    run_root.mkdir()
    source = tmp_path / "sensitivity_tag.json"
    source.write_bytes(b'{"manifest":"immutable"}\n')

    def create(job_id: str) -> subprocess.CompletedProcess[str]:
        attempt = run_root / "publication_recovery_attempts" / job_id
        target = attempt / "preserved_raw_manifests" / source.name
        return subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(run_root),
                str(attempt),
                job_id,
                str(source),
                str(target),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    first_job = "15550001"
    first = create(first_job)
    assert first.returncode == 0, first.stdout + first.stderr
    first_manifest = (
        run_root
        / "publication_recovery_attempts"
        / first_job
        / "preserved_raw_manifests"
        / source.name
    )
    first_bytes = first_manifest.read_bytes()

    # These are intentionally retained bytes from an older fixed-path release.
    (run_root / "completion_status.json").write_text("legacy\n", encoding="utf-8")
    (run_root / "publication_recovery_receipts").mkdir()
    (run_root / "preserved_raw_manifests").mkdir()
    second = create("15550002")
    assert second.returncode == 0, second.stdout + second.stderr
    assert first_manifest.read_bytes() == first_bytes

    repeated = create(first_job)
    assert repeated.returncode != 0
    assert "structural recovery attempt already exists" in (
        repeated.stdout + repeated.stderr
    )
    assert first_manifest.read_bytes() == first_bytes


def test_structural_recovery_wrapper_derives_all_outputs_from_attempt_root() -> None:
    script = _text("hpc/hpc_sensitivity_publish_recovery.sh")
    assert (
        'RECOVERY_ATTEMPT_ROOT="${SENSITIVITY_RUN_DIR}/'
        'publication_recovery_attempts/${SLURM_JOB_ID}"'
    ) in script
    for filename in (
        "completion_status.json",
        "structural_sensitivity_analysis.json",
        "publication_environment.json",
        "slurm_simulation_accounting.json",
        "structural_sensitivity_artifact_manifest.json",
        "structural_sensitivity_archive_receipt.json",
        "structural_sensitivity_summary.csv",
        "structural_sensitivity_summary.png",
        "structural_sensitivity_summary.pdf",
        "structural_sensitivity_publication_receipt.json",
    ):
        assert f'="${{RECOVERY_ATTEMPT_ROOT}}/{filename}"' in script
    assert '"$ANALYSIS_PATH" "$RECOVERY_ATTEMPT_ROOT"' in script
    assert '--recovery-attempt-root "$RECOVERY_ATTEMPT_ROOT"' in script


def test_core_recovery_rebinds_manifest_to_consumed_staged_bytes() -> None:
    wrapper = _text("hpc/hpc_publish_recovery.sh")
    assert 'export AGRIBRAIN_RAW_SEEDS_DIR="$SEEDS_DIR"' in wrapper
    assert 'export AGRIBRAIN_RAW_STRESS_DIR="$STRESS_DIR"' in wrapper
    assert 'export AGRIBRAIN_RAW_H3_LEDGER_DIR="$H3_LEDGER_DIR"' in wrapper
    assert wrapper.count("preserved_raw_manifest.py validate") >= 2

    publisher = _text("hpc/hpc_publish.sh")
    assert "validate_live_preserved_raw" in publisher
    assert "SEMANTIC_RECOVERY_ARGS" in publisher
    assert publisher.count('"${SEMANTIC_RECOVERY_ARGS[@]}"') == 2
    assert "${AGRIBRAIN_VENV}/slurm_simulation_accounting.json" in publisher


def test_structural_recovery_keeps_live_logs_outside_preserved_run() -> None:
    script = _text("hpc/hpc_sensitivity_publish_recovery.sh")
    assert "AGRIBRAIN_RECOVERY_LOG_DIR" in script
    assert "Slurm {field} is outside AGRIBRAIN_RECOVERY_LOG_DIR" in script
    assert script.count("validate_preserved_raw_outputs") >= 3
    assert "simulation_rerun" in _text(
        "mvp/simulation/sensitivity/finalize_structural_sensitivity.py"
    )
