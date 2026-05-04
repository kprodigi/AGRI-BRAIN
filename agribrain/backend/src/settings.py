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


VALID_DEPLOYMENT_PHASES = ("monitoring", "advisory", "autonomous")


def _phase(name: str, default: str) -> str:
    v = os.getenv(name, default).strip().lower()
    if v not in VALID_DEPLOYMENT_PHASES:
        return default
    return v


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
    deployment_phase: str
    dynamic_kb_feedback: bool
    protect_docs: bool
    # Optional scoped keys. When unset, scoped checks fall back to api_key
    # so existing deployments keep working with a single shared key.
    governance_api_key: str
    chain_api_key: str
    phase_api_key: str
    mcp_api_key: str

    def is_prod(self) -> bool:
        return self.env == "prod"


def load_settings() -> RuntimeSettings:
    env = os.getenv("APP_ENV", "dev").strip().lower()
    default_cors = "*" if env == "dev" else "http://localhost:5173"
    api_key = os.getenv("APP_API_KEY", "")
    return RuntimeSettings(
        env=env,
        cors_origins=_csv("CORS_ORIGINS", default_cors),
        require_api_key=_bool("REQUIRE_API_KEY", env != "dev"),
        api_key=api_key,
        allow_local_without_api_key=_bool("ALLOW_LOCAL_WITHOUT_API_KEY", env == "dev"),
        enable_debug_routes=_bool("ENABLE_DEBUG_ROUTES", env == "dev"),
        websocket_require_api_key=_bool("WS_REQUIRE_API_KEY", env != "dev"),
        websocket_api_key=os.getenv("WS_API_KEY", api_key),
        chain_require_privkey=_bool("CHAIN_REQUIRE_PRIVKEY", True),
        forecast_method=os.getenv("FORECAST_METHOD", "lstm"),
        online_learning=_bool("ONLINE_LEARNING", False),
        llm_provider=os.getenv("LLM_PROVIDER", "template"),
        data_csv=os.getenv("DATA_CSV", ""),
        # SIM_API_BASE: empty by default. The simulator runs as a
        # subprocess and does not expose a REST endpoint, and the MCP
        # `simulate` tool is intentionally unregistered when this is
        # empty (see pirag/mcp/registry.py). Set this only when a
        # separate FastAPI simulator process is running and you want
        # MCP-routed forward simulations to reach it. The previous
        # http://127.0.0.1:8100 default created surprising routing
        # behaviour: the live REST server was the FastAPI app at the
        # same port, so simulator routes silently looped back into the
        # main app.
        sim_api_base=os.getenv("SIM_API_BASE", ""),
        deployment_phase=_phase("DEPLOYMENT_PHASE", "autonomous"),
        dynamic_kb_feedback=_bool("DYNAMIC_KB_FEEDBACK", True),
        # PROTECT_DOCS: gate /docs, /redoc, /openapi.json behind the
        # API-key middleware. Defaults to true outside dev so production
        # does not leak the full route schema for reconnaissance via an
        # unauthenticated GET /openapi.json. Operators who terminate
        # docs upstream (reverse proxy, IP allowlist) can set
        # PROTECT_DOCS=false to keep them open here.
        protect_docs=_bool("PROTECT_DOCS", env != "dev"),
        governance_api_key=os.getenv("GOVERNANCE_API_KEY", api_key),
        chain_api_key=os.getenv("CHAIN_API_KEY", api_key),
        phase_api_key=os.getenv("PHASE_API_KEY", api_key),
        mcp_api_key=os.getenv("MCP_API_KEY", api_key),
    )


SETTINGS = load_settings()

