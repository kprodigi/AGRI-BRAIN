"""The MCP social-proxy lookup must expose the canonical carbon time basis."""
from __future__ import annotations

import pytest

from pirag.mcp.tools.slca_lookup import lookup_slca_weights
from src.models.policy import Policy
from src.models.slca import (
    DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY,
    slca_score,
)


def test_lookup_and_policy_share_one_per_opportunity_carbon_cap():
    lookup = lookup_slca_weights("spinach")
    cap = DEFAULT_CARBON_CAP_KG_PER_ROUTING_OPPORTUNITY

    assert Policy().carbon_cap == cap == 50.0
    assert lookup["carbon_cap"] == cap
    assert lookup[
        "carbon_cap_kg_per_standardized_routing_opportunity"
    ] == cap
    assert lookup["carbon_cap_time_basis"] == (
        "standardized_routing_opportunity"
    )
    assert lookup["carbon_cap_is_episode_cap"] is False


@pytest.mark.parametrize(
    ("carbon_kg", "carbon_cap"),
    ((-1.0, 50.0), (float("nan"), 50.0), (1.0, 0.0), (1.0, -1.0)),
)
def test_social_proxy_rejects_invalid_carbon_inputs(carbon_kg, carbon_cap):
    with pytest.raises(ValueError):
        slca_score(
            carbon_kg,
            "cold_chain",
            carbon_cap=carbon_cap,
        )
