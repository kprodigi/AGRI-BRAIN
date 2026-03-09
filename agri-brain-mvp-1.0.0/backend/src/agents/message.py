"""Inter-agent message protocol for supply chain coordination."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class MessageType(Enum):
    """Types of messages exchanged between supply chain agents."""
    SPOILAGE_ALERT = "spoilage_alert"
    SURPLUS_ALERT = "surplus_alert"
    CAPACITY_UPDATE = "capacity_update"
    REROUTE_REQUEST = "reroute_request"
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
