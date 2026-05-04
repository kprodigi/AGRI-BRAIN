"""Packaging integrity guard.

The 2026-05 packaging fix declared explicit ``[tool.setuptools.packages.find]``
so that an editable install exposes both top-level packages (``src`` and
``pirag``). Before the fix, only ``src`` was on the editable path and
``import pirag`` worked accidentally via ``--app-dir`` or cwd. This
test asserts the packages are installable and reachable from a clean
context, so any future regression in pyproject.toml fails CI loudly.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_distribution_metadata_advertises_both_packages():
    """top_level.txt must list both src and pirag.

    A regression that drops one of them (e.g. by reverting the explicit
    packages.find block to autodiscovery) would silently bisect the
    import surface; the assertion catches that at install time.
    """
    dist = md.distribution("agri-brain-backend")
    top = dist.read_text("top_level.txt") or ""
    advertised = {line.strip() for line in top.splitlines() if line.strip()}
    # ``src`` and ``pirag`` are the two real public packages. The
    # auto-generated ``__init__`` token sometimes appears alongside on
    # older setuptools; we only assert presence, not exclusivity.
    assert "src" in advertised, f"src not in top_level.txt: {advertised!r}"
    assert "pirag" in advertised, f"pirag not in top_level.txt: {advertised!r}"


def test_packages_import_directly():
    """Both packages must be importable in the current process."""
    src = importlib.import_module("src")
    pirag = importlib.import_module("pirag")
    assert hasattr(src, "__file__")
    assert hasattr(pirag, "__file__")
    # Sanity: one canonical module from each package surfaces.
    importlib.import_module("src.app")
    importlib.import_module("pirag.mcp.registry")


@pytest.mark.parametrize("module", ["src", "pirag", "src.app", "pirag.mcp.registry"])
def test_imports_work_from_unrelated_cwd(tmp_path: Path, module: str):
    """Subprocess from an unrelated cwd must import without --app-dir.

    This is the regression target: before the packaging fix a fresh
    Python from outside ``agribrain/backend/`` could not import
    ``pirag`` because the editable .pth pointed at ``backend/src/``
    only. We launch a subprocess in ``tmp_path`` so cwd cannot help.
    """
    env = os.environ.copy()
    # Strip PYTHONPATH so cwd-on-sys.path is the only mechanism we can
    # accidentally rely on — and that should be the temp dir, not the
    # repo. The proper fix must work without any of those crutches.
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__name__)"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"import {module} failed from unrelated cwd "
        f"(returncode={result.returncode}):\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert module.split(".")[0] in result.stdout
