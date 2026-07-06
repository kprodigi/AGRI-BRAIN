"""Production-startup hardening tests.

Locks the rule that ``APP_ENV=prod`` combined with ``CORS_ORIGINS=*``
must refuse to start unless the operator explicitly opts in via
``ALLOW_PROD_WILDCARD_CORS=1``. The mitigation matters because the
allow-credentials auto-flip in app.py only addresses one of several
attack surfaces a wildcard CORS policy opens up; in prod, a typo or
copy-paste from a dev .env can otherwise expose response bodies to
any origin.
"""
from __future__ import annotations

import importlib

import pytest


def _reload_settings(monkeypatch, **env):
    """Reload the settings module under a controlled env so the
    module-level ``SETTINGS`` instance reflects the test scenario.
    """
    for k in (
        "APP_ENV", "CORS_ORIGINS", "ALLOW_PROD_WILDCARD_CORS",
        "APP_API_KEY", "REQUIRE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.settings as _settings_mod
    return importlib.reload(_settings_mod)


def test_dev_default_allows_wildcard_cors(monkeypatch):
    """The dev default must keep the wildcard CORS for local frontends."""
    mod = _reload_settings(monkeypatch, APP_ENV="dev")
    assert mod.SETTINGS.env == "dev"
    assert mod.SETTINGS.cors_origins == ["*"]


def test_prod_with_wildcard_cors_refuses_to_start(monkeypatch):
    """The prod + CORS=* combination must raise at import time."""
    with pytest.raises(RuntimeError, match="APP_ENV=prod with CORS_ORIGINS=\\*"):
        _reload_settings(
            monkeypatch, APP_ENV="prod", CORS_ORIGINS="*",
        )


def test_prod_with_wildcard_cors_can_be_explicitly_acknowledged(monkeypatch):
    """An operator who genuinely needs wildcard origins in prod must
    be able to opt in via ALLOW_PROD_WILDCARD_CORS=1, but the choice
    has to be auditable rather than implicit.
    """
    mod = _reload_settings(
        monkeypatch, APP_ENV="prod", CORS_ORIGINS="*",
        ALLOW_PROD_WILDCARD_CORS="1",
    )
    assert mod.SETTINGS.env == "prod"
    assert mod.SETTINGS.cors_origins == ["*"]


def test_prod_with_explicit_origins_starts_normally(monkeypatch):
    """A normal prod deployment with an explicit allowlist must start."""
    mod = _reload_settings(
        monkeypatch, APP_ENV="prod",
        CORS_ORIGINS="https://example.com,https://api.example.com",
    )
    assert mod.SETTINGS.env == "prod"
    assert mod.SETTINGS.cors_origins == [
        "https://example.com", "https://api.example.com",
    ]
