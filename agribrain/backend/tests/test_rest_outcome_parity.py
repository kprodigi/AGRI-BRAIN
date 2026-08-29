"""REST /decide must use the same outcome/reward definitions as the benchmark."""
from __future__ import annotations

import pytest


pytest.importorskip("fastapi", reason="FastAPI required for REST route parity")


def test_rest_decide_uses_timed_footprint_and_canonical_reward():
    from src import app
    from src.models.reward import compute_reward

    prior_phase = app._phase.get_active_phase()
    app._phase.set_active_phase("monitoring")
    app.footprint_meter.reset()
    try:
        response = app.decide(app.DecideIn(
            agent_id="outcome-contract",
            role="farm",
            step=0,
            deterministic=True,
            mode="static",
        ))
    finally:
        app._phase.set_active_phase(prior_phase)

    memo = response["memo"]
    footprint = memo["footprint"]
    reward = memo["reward_decomposition"]

    # Passing elapsed_seconds is necessary: otherwise the meter deliberately
    # returns None for time-based energy/water and the old REST code crashed.
    assert footprint["elapsed_seconds"] > 0.0
    assert footprint["energy_J"] is not None
    assert footprint["water_L"] is not None
    assert footprint["estimate_basis"] == (
        "measured_elapsed_seconds_x_declared_rates"
    )
    assert footprint["measurement_scope"].startswith(
        "REST context construction plus select_action wall time"
    )
    assert footprint["proxy_step_unit"] == "REST decision request"

    assert reward["footprint_terms_in_total"] is False
    assert reward["energy_penalty_descriptive_only"] is True
    assert reward["water_penalty_descriptive_only"] is True
    assert reward["formula"] == (
        "SLCA - eta_w*waste - eta_rho*rho_environmental"
    )

    expected = compute_reward(
        memo["slca"],
        memo["waste"],
        memo["spoilage_risk"],
        eta=app.state["policy"].eta,
        eta_rho=app.state["policy"].eta_rho,
    )
    # Public memo scalars are rounded to four decimals, so compare at the
    # response precision rather than against hidden full-precision values.
    assert reward["total"] == pytest.approx(expected, abs=2e-4)


def test_rest_slca_carbon_component_uses_policy_per_opportunity_cap():
    from src import app

    prior_phase = app._phase.get_active_phase()
    prior_cap = app.state["policy"].carbon_cap
    app._phase.set_active_phase("monitoring")
    app.state["policy"].carbon_cap = 25.0
    try:
        response = app.decide(app.DecideIn(
            agent_id="slca-contract",
            role="farm",
            step=0,
            deterministic=True,
            mode="static",
        ))
    finally:
        app.state["policy"].carbon_cap = prior_cap
        app._phase.set_active_phase(prior_phase)

    memo = response["memo"]
    expected_c = max(0.0, 1.0 - memo["carbon_kg"] / 25.0)
    assert memo["slca_components"]["carbon"] == pytest.approx(
        expected_c,
        abs=1e-4,
    )


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    (
        (
            "post",
            "/decide",
            {
                "json": {
                    "agent_id": "http-outcome-contract",
                    "role": "farm",
                    "step": 0,
                    "deterministic": True,
                    "mode": "static",
                },
            },
        ),
        (
            "get",
            "/case/decide",
            {
                "params": {
                    "agent": "legacy-http-outcome-contract",
                    "role": "farm",
                    "step": 0,
                    "deterministic": True,
                    "mode": "static",
                },
            },
        ),
    ),
)
def test_public_decision_routes_share_canonical_outcome_path(
    method,
    path,
    kwargs,
):
    """Canonical and legacy-compatible HTTP routes cannot drift or TypeError."""
    from fastapi.testclient import TestClient
    from src import app

    prior_phase = app._phase.get_active_phase()
    app._phase.set_active_phase("monitoring")
    app.footprint_meter.reset()
    try:
        with TestClient(app.API) as client:
            response = getattr(client, method)(path, **kwargs)
    finally:
        app._phase.set_active_phase(prior_phase)

    assert response.status_code == 200, response.text
    memo = response.json()["memo"]
    reward = memo["reward_decomposition"]
    footprint = memo["footprint"]
    assert footprint["energy_J"] == pytest.approx(
        10.0 * footprint["elapsed_seconds"],
        abs=1e-8,
    )
    assert footprint["water_L"] == pytest.approx(
        1.8e-6 * footprint["elapsed_seconds"],
        abs=1e-12,
    )
    assert reward["formula"] == (
        "SLCA - eta_w*waste - eta_rho*rho_environmental"
    )
    assert reward["footprint_terms_in_total"] is False
