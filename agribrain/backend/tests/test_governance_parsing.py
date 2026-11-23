import pytest
from fastapi import HTTPException

from src.routers.governance import _parse_policy_key


def test_parse_policy_key_accepts_hex_and_text():
    out_hex = _parse_policy_key("0x01")
    assert isinstance(out_hex, bytes)
    assert len(out_hex) == 32
    assert out_hex[0] == 0x01

    out_text = _parse_policy_key("min_shelf_reroute")
    assert isinstance(out_text, bytes)
    assert len(out_text) == 32


def test_parse_policy_key_rejects_invalid_hex():
    with pytest.raises(HTTPException):
        _parse_policy_key("0x" + "ab" * 40)
