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
