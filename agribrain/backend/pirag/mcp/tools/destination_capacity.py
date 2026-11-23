"""Destination capacity query for the MCP server.

Returns the capacity that the redistribution outlets and the recovery facility
publish, together with the share of the current dispatch each can take. The
values are synthetic design quantities of the benchmark, not measured
capacities. In the simulator the published values are supplied with the call;
a deployment would read them from the destinations' own systems.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Dict


def _share(capacity: float, dispatch_size: float) -> float:
    if dispatch_size <= 0.0:
        return 1.0
    return float(min(1.0, max(capacity, 0.0) / dispatch_size))


def query_destination_capacity(
    redistribution_capacity: float,
    recovery_capacity: float,
    dispatch_size: float,
) -> Dict[str, Any]:
    """Report destination capacity for one dispatch.

    Parameters
    ----------
    redistribution_capacity : capacity published by the redistribution outlets,
        in reference dispatches.
    recovery_capacity : capacity published by the recovery facility, in
        reference dispatches.
    dispatch_size : size of the dispatch being routed, in reference dispatches.

    Returns
    -------
    Dict with the published capacities, the share of the dispatch each
    destination can take, the corresponding shortfall and whether the whole
    dispatch fits.
    """
    values = {
        "redistribution_capacity": float(redistribution_capacity),
        "recovery_capacity": float(recovery_capacity),
        "dispatch_size": float(dispatch_size),
    }
    for name, value in values.items():
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    size = values["dispatch_size"]
    redistribution_share = _share(values["redistribution_capacity"], size)
    recovery_share = _share(values["recovery_capacity"], size)
    return {
        "_status": "ok",
        "dispatch_size": size,
        "redistribution_capacity": values["redistribution_capacity"],
        "recovery_capacity": values["recovery_capacity"],
        "redistribution_share": redistribution_share,
        "recovery_share": recovery_share,
        "redistribution_shortfall": 1.0 - redistribution_share,
        "recovery_shortfall": 1.0 - recovery_share,
        "redistribution_has_room": bool(values["redistribution_capacity"] >= size),
        "recovery_has_room": bool(values["recovery_capacity"] >= size),
    }
