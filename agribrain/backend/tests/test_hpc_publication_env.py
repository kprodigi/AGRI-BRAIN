"""Checks for the canonical Slurm publication environment contract."""
from __future__ import annotations

import importlib.util
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "hpc" / "validate_publication_env.py"
SPEC = importlib.util.spec_from_file_location("validate_publication_env", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SNAPSHOT_PATH = REPO_ROOT / "hpc" / "validate_source_snapshot.py"
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "validate_source_snapshot", SNAPSHOT_PATH,
)
assert SNAPSHOT_SPEC is not None and SNAPSHOT_SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT)


def test_exact_canonical_environment_is_accepted():
    env = dict(MODULE.EXPECTED)
    assert MODULE.errors_for_environment(env) == []


def test_poisoned_ambient_values_are_rejected():
    env = dict(MODULE.EXPECTED)
    env.update({
        "DATA_CSV": "/tmp/wrong.csv",
        "SIM_API_BASE": "https://ambient.invalid",
        "STOCH_TEMP_STD_C": "99",
        "AGRIBRAIN_ALLOW_DIRTY": "1",
    })
    errors = MODULE.errors_for_environment(env)
    assert any(error.startswith("DATA_CSV:") for error in errors)
    assert any(error.startswith("SIM_API_BASE:") for error in errors)
    assert any(error.startswith("STOCH_TEMP_STD_C:") for error in errors)
    assert any(error.startswith("AGRIBRAIN_ALLOW_DIRTY:") for error in errors)


def test_only_lock_verified_python_minors_are_accepted():
    assert MODULE.interpreter_error((3, 11, 9)) is None
    assert MODULE.interpreter_error((3, 13, 2)) is not None
    assert "not lock-verified" in MODULE.interpreter_error((3, 12, 10))


def test_publication_removes_wall_clock_throttling_and_parallel_reduction_drift():
    assert MODULE.EXPECTED["MCP_RATE_LIMITS"] == "disabled"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        assert MODULE.EXPECTED[name] == "1"


def test_orchestrator_atomically_claims_a_fresh_run_scoped_venv():
    orchestrator = (REPO_ROOT / "hpc" / "hpc_run.sh").read_text(encoding="utf-8")
    assert 'export AGRIBRAIN_VENV=".publication_venvs/${RUN_TAG}"' in orchestrator
    assert 'if ! mkdir "$AGRIBRAIN_VENV"; then' in orchestrator
    assert 'PUBLICATION_PYTHON_BIN="${AGRIBRAIN_PYTHON_BIN:-python3.11}"' in orchestrator
    assert '"$PUBLICATION_PYTHON_BIN" -m venv "$AGRIBRAIN_VENV"' in orchestrator
    assert 'BACKEND_BUILD_SRC="${AGRIBRAIN_VENV}/backend-build-source"' in orchestrator
    assert 'cp -a agribrain/backend/. "$BACKEND_BUILD_SRC/"' in orchestrator
    assert 'python -m pip install "$BACKEND_BUILD_SRC" --no-deps' in orchestrator
    install_pos = orchestrator.index(
        'python -m pip install "$BACKEND_BUILD_SRC" --no-deps',
    )
    assert orchestrator.index(
        "python hpc/validate_source_checkout.py",
        install_pos,
    ) > install_pos
    assert "capture_publication_environment.py --validate-only" in orchestrator
    assert "source .venv/bin/activate" not in orchestrator
    assert "git worktree add --detach" in orchestrator
    assert 'export AGRIBRAIN_SOURCE_SNAPSHOT=' in orchestrator
    assert 'export AGRIBRAIN_SOURCE_SNAPSHOT_MODE=' in orchestrator
    assert 'export AGRIBRAIN_SOURCE_TREE_SHA256=' in orchestrator
    assert "validate_source_snapshot.py --print-digest" in orchestrator
    assert '--chdir="$AGRIBRAIN_SOURCE_SNAPSHOT"' in orchestrator


def test_every_worker_activates_and_validates_the_exact_run_venv():
    for name in ("hpc_seed.sh", "hpc_stress.sh", "hpc_publish.sh"):
        script = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
        bootstrap = "source hpc/ensure_git_available.sh"
        source_gate = "validate_source_checkout.py --allow-run-artifacts"
        assert bootstrap in script
        assert script.index(bootstrap) < script.index(source_gate)
        assert '"${AGRIBRAIN_VENV:-}" != ".publication_venvs/${RUN_TAG}"' in script
        assert 'source "$AGRIBRAIN_VENV/bin/activate"' in script
        if name == "hpc_publish.sh":
            assert "capture_publication_environment()" in script
            assert "capture_publication_environment --validate-only" in script
        else:
            assert "capture_publication_environment.py --validate-only" in script
        assert script.count("python hpc/validate_source_snapshot.py") >= 2
        assert 'cd "$AGRIBRAIN_SOURCE_SNAPSHOT"' in script


def test_git_bootstrap_is_fail_closed_and_supports_cluster_module():
    script = (REPO_ROOT / "hpc" / "ensure_git_available.sh").read_text(
        encoding="utf-8",
    )
    assert "command -v git" in script
    assert "git/2.42.0" in script
    assert "module load" in script
    assert 'return 2' in script


def test_source_snapshot_digest_rejects_a_restamped_source_mutation(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "snapshot@example.invalid"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Snapshot Test"],
        cwd=tmp_path, check=True,
    )
    source = tmp_path / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    result = tmp_path / "mvp" / "simulation" / "results" / "output.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True,
    ).strip()
    source.chmod(stat.S_IREAD)
    digest, count = SNAPSHOT.tracked_source_digest(tmp_path)
    assert count == 1  # tracked results are deliberately outside source digest
    env = {
        "AGRIBRAIN_SOURCE_SNAPSHOT": str(tmp_path.resolve()),
        "AGRIBRAIN_SOURCE_SNAPSHOT_MODE": SNAPSHOT.SNAPSHOT_MODE,
        "AGRIBRAIN_SOURCE_TREE_SHA256": digest,
        "AGRIBRAIN_GIT_COMMIT": commit,
    }
    assert SNAPSHOT.validation_errors(env, repo_root=tmp_path) == []

    source.chmod(stat.S_IREAD | stat.S_IWRITE)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    source.chmod(stat.S_IREAD)
    assert any(
        "digest changed" in error
        for error in SNAPSHOT.validation_errors(env, repo_root=tmp_path)
    )
