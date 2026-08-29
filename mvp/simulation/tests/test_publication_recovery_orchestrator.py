"""Static fail-closed contracts for the Slurm recovery launchers."""

from __future__ import annotations

from pathlib import Path

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
    assert "Preserving canonical recovery evidence" in script
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


def test_structural_canonical_copy_is_retry_safe() -> None:
    script = _text("hpc/publication_recovery_run.sh")
    first_copy = script.index(
        'copy_canonical_evidence "$STRUCTURAL_RAW_MANIFEST" "$STRUCTURAL_CANONICAL_RAW"'
    )
    first_flag = script.index("STRUCTURAL_CANONICAL_RAW_CREATED=true", first_copy)
    second_copy = script.index('copy_canonical_evidence "$STRUCTURAL_RECOVERY_RECEIPT"', first_flag)
    second_flag = script.index("STRUCTURAL_CANONICAL_RECEIPT_CREATED=true", second_copy)
    assert first_copy < first_flag < second_copy < second_flag
    assert 'cmp --silent -- "$STRUCTURAL_RECOVERY_RECEIPT"' in script
    assert 'cmp --silent -- "$STRUCTURAL_RAW_MANIFEST"' in script
    assert "canonical structural evidence has a symlink component" in script
    assert "canonical structural recovery parent is symlinked" in script


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
