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
