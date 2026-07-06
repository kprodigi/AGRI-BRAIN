"""Inter-agent message protocol for supply chain coordination."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Iterable

import numpy as np


class MessageType(Enum):
    """Types of messages exchanged between supply chain agents."""
    SPOILAGE_ALERT = "spoilage_alert"
    SURPLUS_ALERT = "surplus_alert"
    CAPACITY_UPDATE = "capacity_update"
    REROUTE_REQUEST = "reroute_request"
    COORDINATION_UPDATE = "coordination_update"
    ACK = "ack"


@dataclass(frozen=True)
class InterAgentMessage:
    """Immutable message passed between supply chain agents.

    Parameters
    ----------
    sender : agent_id of the sending agent.
    recipient : agent_id of the target agent, or ``"broadcast"`` for all.
    msg_type : category of the message.
    payload : arbitrary data dict (spoilage risk, capacity, etc.).
    hour : simulation hour at which the message was created.
    """
    sender: str
    recipient: str
    msg_type: MessageType
    payload: Dict[str, Any]
    hour: float


# ---------------------------------------------------------------------------
# Message-to-bias mapping (consumed by the active agent's logits)
# ---------------------------------------------------------------------------

# Per-message-type logit nudges in (cold_chain, local_redistribute, recovery)
# space. Magnitudes ~0.05-0.15 — large enough to flip decisions when stacked
# with multiple alerts, small enough that a single ack does not dominate the
# softmax. Calibrated so the cumulative bias from a 5-message inbox stays
# bounded by ±0.30 per action.
_MESSAGE_TYPE_BIAS = {
    # Spoilage alerts: produce is in danger. Push local redistribution
    # (move it out fast) and recovery (compost / valorize as last
    # resort), suppress cold_chain (long-haul is too slow).
    MessageType.SPOILAGE_ALERT:    np.array([-0.10, +0.10, +0.05]),
    # Surplus alerts: too much inventory. Push local redistribution
    # (move surplus to nearby community), suppress cold_chain (no
    # point in long-haul for surplus).
    MessageType.SURPLUS_ALERT:     np.array([-0.05, +0.15, +0.00]),
    # Capacity updates from RecoveryAgent: tells the network how much
    # spare valorization capacity the recovery node has. The payload
    # carries `available_capacity` in [0, 1]; we scale the recovery
    # nudge by that capacity below.
    MessageType.CAPACITY_UPDATE:   np.array([+0.00, +0.00, +0.05]),
    # Reroute request: a downstream agent asks the upstream stage to
    # change its decision. Payload may carry `requested_action` ∈
    # {0,1,2}; scaling logic below applies +0.10 to that index.
    MessageType.REROUTE_REQUEST:   np.array([+0.00, +0.00, +0.00]),
    # Coordination: bookkeeping; no direct logit effect.
    MessageType.COORDINATION_UPDATE: np.array([+0.00, +0.00, +0.00]),
    # ACK: no direct logit effect.
    MessageType.ACK:               np.array([+0.00, +0.00, +0.00]),
}

_MESSAGE_BIAS_CAP = 0.30  # Per-action bound on total inbox-derived bias.


def message_bias_from_inbox(
    messages: Iterable[InterAgentMessage],
) -> np.ndarray:
    """Convert a flushed inbox into a logit-bias vector of shape (3,).

    The bias is added to the active agent's role_bias before softmax.
    See module-level ``_MESSAGE_TYPE_BIAS`` for the per-type mapping;
    REROUTE_REQUEST honors an optional ``requested_action`` payload
    field and CAPACITY_UPDATE scales by the sender's reported
    ``available_capacity``. The cumulative bias is clamped to
    ±``_MESSAGE_BIAS_CAP`` per action so a flooded inbox cannot
    completely override the policy.

    Returns ``np.zeros(3)`` for an empty inbox.
    """
    bias = np.zeros(3, dtype=np.float64)
    for m in messages:
        base = _MESSAGE_TYPE_BIAS.get(m.msg_type, np.zeros(3))
        if m.msg_type == MessageType.CAPACITY_UPDATE:
            # Preserve a zero capacity value as zero (the previous
            # `or 1.0` fallback treated 0.0 as falsy and silently
            # promoted it to full capacity, which inverted the
            # documented semantics). Default to full capacity only
            # when the field is genuinely missing or non-numeric.
            raw_cap = m.payload.get("available_capacity")
            try:
                cap = float(raw_cap) if raw_cap is not None else 1.0
            except (TypeError, ValueError):
                cap = 1.0
            bias = bias + base * max(0.0, min(1.0, cap))
        elif m.msg_type == MessageType.REROUTE_REQUEST:
            req = m.payload.get("requested_action")
            if isinstance(req, int) and 0 <= req <= 2:
                nudge = np.zeros(3, dtype=np.float64)
                nudge[req] = +0.10
                bias = bias + nudge
        else:
            bias = bias + base
    return np.clip(bias, -_MESSAGE_BIAS_CAP, _MESSAGE_BIAS_CAP)
