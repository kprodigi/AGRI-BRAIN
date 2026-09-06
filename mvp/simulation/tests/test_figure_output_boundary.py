"""Fail-closed publication-figure rendering boundaries."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = REPO_ROOT / "mvp" / "simulation"


def test_direct_figure_script_cannot_run_or_write(tmp_path: Path) -> None:
    script = SIM_DIR / "generate_figures.py"
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path,
        text=True, capture_output=True, check=False, timeout=20,
    )
    assert completed.returncode == 2
    assert "RETIRED" in completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_cache_renderer_requires_strict_identity_and_output(tmp_path: Path) -> None:
    script = SIM_DIR / "regenerate_figures_from_cache.py"
    env = dict(os.environ)
    for name in (
        "STRICT_VALIDATION", "AGRIBRAIN_GIT_COMMIT", "RUN_TAG",
        "BENCHMARK_SEEDS", "FIGURE_SEED_ROOT", "FIGURE_OUTPUT_DIR",
        "AGRIBRAIN_PUBLICATION_RENDER",
    ):
        env.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=False, timeout=20,
    )
    assert completed.returncode == 2
    assert "STRICT_VALIDATION=1 is mandatory" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_canonical_output_requires_hpc_publisher_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(SIM_DIR))
    try:
        import regenerate_figures_from_cache as renderer
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("STRICT_VALIDATION", "1")
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("RUN_TAG", "test_run")
    monkeypatch.setenv(
        "BENCHMARK_SEEDS",
        "42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,"
        "1010,1111,1212,1313,1414,1515",
    )
    monkeypatch.setenv("FIGURE_SEED_ROOT", str(SIM_DIR / "results" / "benchmark_seeds"))
    monkeypatch.setenv("FIGURE_OUTPUT_DIR", str(SIM_DIR / "results"))
    monkeypatch.delenv("AGRIBRAIN_PUBLICATION_RENDER", raising=False)
    monkeypatch.setattr(renderer, "_require_renderer_source_identity", lambda: None)

    assert renderer.main() == 2


def test_cache_renderer_rejects_changed_or_wrong_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(SIM_DIR))
    try:
        import regenerate_figures_from_cache as renderer
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(
        renderer,
        "_source_validation_errors",
        lambda **_kwargs: ["checkout HEAD differs from simulation source"],
    )
    with pytest.raises(RuntimeError, match="clean simulation commit"):
        renderer._require_renderer_source_identity()


def test_figure_provenance_separates_simulation_and_renderer_commits() -> None:
    from mvp.simulation import regenerate_figures_from_cache as renderer

    simulation = "a" * 40
    publication = "b" * 40
    identity = renderer._figure_source_identity({
        "AGRIBRAIN_GIT_COMMIT": simulation,
        "AGRIBRAIN_SIMULATION_COMMIT": simulation,
        "AGRIBRAIN_PUBLICATION_CODE_COMMIT": publication,
    })
    assert identity == {
        "source_commit": simulation,
        "source_commit_semantics": "raw_input_simulation_commit",
        "simulation_source_commit": simulation,
        "renderer_code_commit": publication,
        "dual_provenance": True,
    }
    with pytest.raises(RuntimeError, match="raw-input commit"):
        renderer._figure_source_identity({
            "AGRIBRAIN_GIT_COMMIT": publication,
            "AGRIBRAIN_SIMULATION_COMMIT": simulation,
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT": publication,
        })


def test_hpc_publisher_uses_fresh_staging_and_validated_promotion() -> None:
    source = (REPO_ROOT / "hpc" / "hpc_publish.sh").read_text(encoding="utf-8")
    assert 'export FIGURE_OUTPUT_DIR="$FIGURE_STAGE"' in source
    assert 'if [ -e "$FIGURE_STAGE" ]' in source
    assert "hpc/validate_and_promote_figures.py" in source
    assert 'export FIGURE_OUTPUT_DIR="$RESULTS_DIR"' not in source


def test_figure_promoter_file_entrypoint_is_importable() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "hpc" / "validate_and_promote_figures.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--staging-dir" in completed.stdout
