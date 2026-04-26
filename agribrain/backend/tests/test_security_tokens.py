import base64
import hmac
import importlib
import time


def _reload_security(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("WS_REQUIRE_API_KEY", "true")
    monkeypatch.setenv("APP_API_KEY", "unit-test-secret")
    monkeypatch.setenv("WS_API_KEY", "unit-test-secret")

    import src.settings as settings_mod
    import src.security as security_mod

    importlib.reload(settings_mod)
    importlib.reload(security_mod)
    return security_mod


def test_issue_and_validate_ws_token(monkeypatch):
    security = _reload_security(monkeypatch)
    token = security.issue_ws_token(ttl_seconds=60)
    assert isinstance(token, str)
    assert security.validate_ws_token(token) is True


def test_validate_ws_token_rejects_expired(monkeypatch):
    security = _reload_security(monkeypatch)
    exp = int(time.time()) - 10
    sig = hmac.new(b"unit-test-secret", str(exp).encode("utf-8"), "sha256").hexdigest()
    raw = f"{exp}.{sig}".encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii")
    assert security.validate_ws_token(token) is False
