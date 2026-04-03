"""Centralized runtime settings for backend and MCP/piRAG services."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class RuntimeSettings:
    env: str
    cors_origins: List[str]
    require_api_key: bool
    api_key: str
    allow_local_without_api_key: bool
    enable_debug_routes: bool
    websocket_require_api_key: bool
    websocket_api_key: str
    chain_require_privkey: bool
    forecast_method: str
    online_learning: bool
    llm_provider: str
    data_csv: str
    sim_api_base: str


def load_settings() -> RuntimeSettings:
    env = os.getenv("APP_ENV", "dev").strip().lower()
    default_cors = "*" if env == "dev" else "http://localhost:5173"
    return RuntimeSettings(
        env=env,
        cors_origins=_csv("CORS_ORIGINS", default_cors),
        require_api_key=_bool("REQUIRE_API_KEY", env != "dev"),
        api_key=os.getenv("APP_API_KEY", ""),
        allow_local_without_api_key=_bool("ALLOW_LOCAL_WITHOUT_API_KEY", True),
        enable_debug_routes=_bool("ENABLE_DEBUG_ROUTES", env == "dev"),
        websocket_require_api_key=_bool("WS_REQUIRE_API_KEY", env != "dev"),
        websocket_api_key=os.getenv("WS_API_KEY", os.getenv("APP_API_KEY", "")),
        chain_require_privkey=_bool("CHAIN_REQUIRE_PRIVKEY", True),
        forecast_method=os.getenv("FORECAST_METHOD", "lstm"),
        online_learning=_bool("ONLINE_LEARNING", False),
        llm_provider=os.getenv("LLM_PROVIDER", "template"),
        data_csv=os.getenv("DATA_CSV", ""),
        sim_api_base=os.getenv("SIM_API_BASE", "http://127.0.0.1:8100"),
    )


SETTINGS = load_settings()

