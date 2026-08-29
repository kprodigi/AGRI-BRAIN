"""The unsupported neural-residual path must be absent from public code."""
from __future__ import annotations

from pathlib import Path

from src.models.policy import Policy


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_neural_residual_module_is_not_shipped() -> None:
    assert not (
        REPO_ROOT / "agribrain" / "backend" / "src" / "models" / "pinn_net.py"
    ).exists()


def test_submission_policy_exposes_only_mechanistic_kinetic_parameters() -> None:
    policy = Policy()
    for retired_name in ("k0", "alpha_decay", "T0"):
        assert not hasattr(policy, retired_name)
    for active_name in ("k_ref", "Ea_R", "T_ref_K", "beta_humidity", "lag_lambda"):
        assert hasattr(policy, active_name)


def test_predictions_endpoint_does_not_execute_residual_compatibility_path() -> None:
    source = (REPO_ROOT / "agribrain" / "backend" / "src" / "app.py").read_text(
        encoding="utf-8",
    )
    assert "compute_spoilage_pinn" not in source
    assert "_PINN_OVERLAY_CACHE" not in source
    assert "shelf_left_pinn" not in source
    assert "spoilage_risk_pinn" not in source
    assert "pinn_available" not in source
