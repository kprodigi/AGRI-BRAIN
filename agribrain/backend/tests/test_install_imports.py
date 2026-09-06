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
import importlib.resources as resources
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_BRANDING = {"custom.v2.css", "favicon.png", "logo.png"}
_EXPECTED_KNOWLEDGE_DOCUMENTS = {
    "animal_feed_diversion_standards.txt",
    "blockchain_audit_requirements.txt",
    "carbon_accounting_transport.txt",
    "composting_bioenergy_requirements.txt",
    "cooperative_governance_policy.txt",
    "cyber_outage_contingency.txt",
    "demand_volatility_response.txt",
    "emergency_rerouting_sop.txt",
    "green_ai_reporting.txt",
    "heatwave_contingency_plan.txt",
    "iot_sensor_spec.txt",
    "redistribution_food_bank_protocol.txt",
    "regulatory_fda_leafy_greens.txt",
    "slca_community_resilience_metrics.txt",
    "slca_guidelines.txt",
    "slca_labor_fairness_standards.txt",
    "slca_price_transparency_framework.txt",
    "sop_cold_chain.txt",
    "temperature_excursion_protocol.txt",
    "waste_hierarchy_protocol.txt",
}


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


def test_runtime_resources_are_package_local_and_readable():
    """Wheel-required data must resolve from its owning import package."""

    src_root = resources.files("src")
    dataset = src_root.joinpath("data_spinach.csv")
    assert dataset.is_file()
    assert dataset.read_text(encoding="utf-8").startswith("timestamp,tempC,RH,")

    branding_root = src_root.joinpath("static", "branding")
    observed_branding = {
        entry.name for entry in branding_root.iterdir() if entry.is_file()
    }
    assert observed_branding == _EXPECTED_BRANDING
    assert branding_root.joinpath("custom.v2.css").read_text(
        encoding="utf-8"
    ).startswith("/* 1) Color the top bar */")

    pirag_root = resources.files("pirag")
    policy = pirag_root.joinpath("configs", "policy.yaml")
    assert policy.is_file()
    assert "rate_limits:" in policy.read_text(encoding="utf-8")
    knowledge_root = pirag_root.joinpath("knowledge_base")
    observed_knowledge = {
        entry.name
        for entry in knowledge_root.iterdir()
        if entry.is_file() and entry.name.endswith(".txt")
    }
    assert observed_knowledge == _EXPECTED_KNOWLEDGE_DOCUMENTS


def test_pyproject_declares_every_runtime_resource_family():
    """Lock the wheel package-data rules that editable installs can mask."""

    metadata = tomllib.loads(
        (_BACKEND_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    setuptools = metadata["tool"]["setuptools"]
    assert setuptools["include-package-data"] is True
    package_data = setuptools["package-data"]
    assert set(package_data["src"]) == {
        "data_spinach.csv",
        "static/branding/*",
    }
    assert set(package_data["pirag"]) == {
        "configs/*.yaml",
        "configs/*.yml",
        "knowledge_base/*.txt",
        "knowledge_base/*.json",
        "knowledge_base/*.csv",
    }


def test_app_mounts_package_local_static_directory():
    """Import-time StaticFiles setup must use the wheel-owned directory."""

    from fastapi.testclient import TestClient

    app = importlib.import_module("src.app")
    expected = (
        Path(importlib.import_module("src").__file__).resolve().parent / "static"
    )
    assert app._STATIC_DIR == expected.resolve()
    assert (app._STATIC_DIR / "branding" / "favicon.png").is_file()
    client = TestClient(app.API)
    css = client.get("/static/branding/custom.v2.css")
    assert css.status_code == 200
    assert css.text.startswith("/* 1) Color the top bar */")
    assert client.get("/static/branding/logo.png").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


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
