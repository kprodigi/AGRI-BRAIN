"""Parity check: deploy.js seed values must mirror Python THETA / THETA_CONTEXT.

The 2026-04 fix updated PolicyStore.sol to register THETA as (3, 10) and
THETA_CONTEXT as (3, 5), and deploy.js was retargeted to seed both
matrices via setPolicyMatrix. However, the THETA_CONTEXT_MILLI array in
deploy.js was not synced to the post-recalibration Python values --
context_to_logits.THETA_CONTEXT had been halved to land within the
+/-1 modifier clamp, but deploy.js still carried the pre-recalibration
magnitudes (roughly 2x off). An on-chain auditor reading
PolicyStore.getPolicyMatrix("THETA_CONTEXT") would have seen different
numbers than the running policy.

This test parses deploy.js for the two ``*_MILLI`` arrays and asserts
that every cell equals ``round(python_value * 1000)``. Adding this gate
to CI prevents the same drift from recurring silently.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.models.action_selection import THETA
from pirag.context_to_logits import THETA_CONTEXT

DEPLOY_JS = (
    Path(__file__).resolve().parent.parent.parent
    / "contracts" / "hardhat" / "scripts" / "deploy.js"
)


def _extract_milli_array(text: str, name: str) -> list[int]:
    """Pull a ``const <name> = [ ... ];`` integer array out of deploy.js."""
    m = re.search(rf"const\s+{re.escape(name)}\s*=\s*\[(.*?)\];", text, re.S)
    if not m:
        raise AssertionError(f"{name} not found in deploy.js")
    body = m.group(1)
    # Strip JS-style // comments so digits on comment lines are not parsed.
    body = re.sub(r"//[^\n]*", "", body)
    nums = re.findall(r"-?\d+", body)
    return [int(n) for n in nums]


@pytest.fixture(scope="module")
def deploy_text() -> str:
    if not DEPLOY_JS.exists():
        pytest.skip(f"deploy.js not found at {DEPLOY_JS}")
    return DEPLOY_JS.read_text(encoding="utf-8")


def test_theta_shape() -> None:
    assert THETA.shape == (3, 10), (
        f"Python THETA shape drift -- expected (3, 10), got {THETA.shape}. "
        "deploy.js seeds (3, 10) so any change to THETA's column count "
        "without also updating PolicyStore.sol + deploy.js will break "
        "on-chain audit."
    )


def test_theta_context_shape() -> None:
    assert THETA_CONTEXT.shape == (3, 5), (
        f"Python THETA_CONTEXT shape drift -- expected (3, 5), got "
        f"{THETA_CONTEXT.shape}."
    )


def test_theta_milli_parity(deploy_text: str) -> None:
    """deploy.js THETA_MILLI == round(Python THETA * 1000) cell-by-cell."""
    expected = (np.round(THETA * 1000).astype(int)).reshape(-1).tolist()
    actual = _extract_milli_array(deploy_text, "THETA_MILLI")
    assert len(actual) == 30, (
        f"THETA_MILLI must be 30 cells (3 rows x 10 cols); got {len(actual)}."
    )
    assert actual == expected, (
        "deploy.js THETA_MILLI is out of sync with Python THETA. "
        f"Diff (idx, py, js): {[(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]}. "
        "Update agribrain/contracts/hardhat/scripts/deploy.js to match "
        "agribrain/backend/src/models/action_selection.py::THETA, or vice versa. "
        "Both must be the same source-of-truth."
    )


def test_theta_context_milli_parity(deploy_text: str) -> None:
    """deploy.js THETA_CONTEXT_MILLI == round(Python THETA_CONTEXT * 1000)."""
    expected = (np.round(THETA_CONTEXT * 1000).astype(int)).reshape(-1).tolist()
    actual = _extract_milli_array(deploy_text, "THETA_CONTEXT_MILLI")
    assert len(actual) == 15, (
        f"THETA_CONTEXT_MILLI must be 15 cells (3 rows x 5 cols); got {len(actual)}."
    )
    assert actual == expected, (
        "deploy.js THETA_CONTEXT_MILLI is out of sync with Python "
        "THETA_CONTEXT. "
        f"Diff (idx, py, js): {[(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]}. "
        "Pre-2026-05 deploy.js carried roughly-2x-too-large values that "
        "predated the THETA_CONTEXT recalibration in "
        "agribrain/backend/pirag/context_to_logits.py. Update one "
        "source-of-truth and re-run this test."
    )
