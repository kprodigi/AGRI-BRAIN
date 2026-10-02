"""Regression guards for non-circular cyber-outage modelling."""
from __future__ import annotations

import inspect

from src.models import action_selection
from src.agents import coordinator


def test_no_mode_specific_cyber_probability_table():
    """Outage performance must not be assigned from the model label."""
    assert action_selection.CYBER_REROUTE_PROB == {}
    source = inspect.getsource(action_selection.select_action)
    assert "CYBER_REROUTE_PROB.get" not in source
    assert "scenario == \"cyber_outage\"" not in source


def test_coordinator_models_channel_availability():
    """The outage is represented as a channel failure in the normal path."""
    source = inspect.getsource(coordinator.AgentCoordinator._compute_step_context)
    assert 'scenario == "cyber_outage" and hour >= 24.0' in source
    assert '"_channel_unavailable"] = "cyber_outage"' in source
