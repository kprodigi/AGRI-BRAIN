import pytest

from src.chain.contracts import _send_tx


class _DummyHash:
    def hex(self):
        return "0xabc123"


class _DummySigned:
    raw_transaction = b"raw-tx"


class _DummyAccount:
    address = "0x0000000000000000000000000000000000000001"

    def sign_transaction(self, _tx):
        return _DummySigned()


class _DummyTxFn:
    def build_transaction(self, _params):
        return {"dummy": True}


class _DummyEth:
    def __init__(self, status):
        self._status = status

    def get_transaction_count(self, _address):
        return 7

    def send_raw_transaction(self, _raw):
        return b"\x12"

    def wait_for_transaction_receipt(self, _txh):
        return {"status": self._status, "transactionHash": _DummyHash()}


class _DummyW3:
    def __init__(self, status):
        self.eth = _DummyEth(status)

    def to_wei(self, value, _unit):
        return int(float(value) * 1_000_000_000)


def test_send_tx_returns_hash_on_success():
    txh = _send_tx(_DummyW3(status=1), _DummyAccount(), _DummyTxFn(), {"chain_id": 31337})
    assert txh == "0xabc123"


def test_send_tx_raises_on_revert():
    with pytest.raises(RuntimeError, match="reverted"):
        _send_tx(_DummyW3(status=0), _DummyAccount(), _DummyTxFn(), {"chain_id": 31337})


# ----------------------------------------------------------------------
# AgentRegistry ABI <-> Solidity signature pin (2026-05 critical fix).
# These tests gate against ABI drift -- a previous version of the ABI
# declared a 3-arg ``register(bytes32, string, string)`` that does not
# exist on the deployed contract; any chain submission would have
# reverted with "function selector not found". The Solidity source
# AgentRegistry.sol exposes ownerRegister and sponsorRegister, both
# 4-arg with leading ``address account``.
# ----------------------------------------------------------------------
def test_agent_registry_abi_matches_solidity_signatures():
    """ABI must declare ownerRegister + sponsorRegister with the 4-arg
    Solidity signatures. Drift here = silent on-chain failure.
    """
    from src.chain.contracts import AGENT_REGISTRY_ABI

    fn_specs = {
        entry["name"]: entry
        for entry in AGENT_REGISTRY_ABI
        if entry.get("type") == "function"
    }
    assert "register" not in fn_specs, (
        "Stale 3-arg register() ABI re-introduced; the deployed "
        "AgentRegistry.sol has no such function. Use sponsorRegister "
        "or ownerRegister."
    )
    for name in ("ownerRegister", "sponsorRegister"):
        assert name in fn_specs, f"AGENT_REGISTRY_ABI missing {name!r}"
        inputs = fn_specs[name]["inputs"]
        types = [arg["type"] for arg in inputs]
        assert types == ["address", "bytes32", "string", "string"], (
            f"{name} ABI signature {types} does not match Solidity "
            f"({'address, bytes32, string, string'!r})."
        )


def test_agent_register_wrapper_signature_takes_account():
    """The Python wrapper must accept ``account`` (the on-chain
    address being registered) as its first positional argument. The
    pre-2026-05 wrapper omitted this and silently called a non-existent
    Solidity ``register()`` with the wrong arity.
    """
    import inspect

    from src.chain.contracts import agent_register

    sig = inspect.signature(agent_register)
    params = list(sig.parameters.keys())
    assert params[0] == "account", (
        f"agent_register first parameter is {params[0]!r}; expected "
        f"'account' so callers can target a non-self address as "
        f"the contract requires."
    )
    assert "method" in sig.parameters, (
        "agent_register must accept a 'method' kwarg to switch "
        "between sponsorRegister (production) and ownerRegister "
        "(bootstrap-only)."
    )


def test_agent_register_wrapper_rejects_unknown_method():
    """Guard against typos / future signature drift."""
    import pytest

    from src.chain.contracts import agent_register

    # Pass chain_cfg=None so _get_contract returns None up-front; this
    # test is purely about argument validation.
    with pytest.raises(ValueError, match="method must be"):
        agent_register(
            "0x0000000000000000000000000000000000000001",
            "agent_id",
            "cooperative",
            "",
            chain_cfg={"will": "never_be_read_due_to_value_error"},
            method="register",  # the legacy 3-arg name; must be rejected
        )
