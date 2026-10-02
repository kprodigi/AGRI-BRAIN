from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from src.settings import SETTINGS
from src.security import enforce_api_key

router = APIRouter()

@router.get("/debug/routes")
def debug_routes(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    if not SETTINGS.enable_debug_routes:
        raise HTTPException(status_code=404, detail="Debug routes disabled")
    enforce_api_key(request, x_api_key)
    app: FastAPI = request.app
    return [
        {
            "path": getattr(r, "path", getattr(r, "path_regex", "?")),
            "methods": list(getattr(r, "methods", None) or []),
            "name": getattr(r, "name", ""),
        }
        for r in app.router.routes
        if hasattr(r, "path")
    ]


@router.get("/debug/config")
def debug_config(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    if not SETTINGS.enable_debug_routes:
        raise HTTPException(status_code=404, detail="Debug routes disabled")
    enforce_api_key(request, x_api_key)
    return {
        "env": SETTINGS.env,
        "cors_origins": SETTINGS.cors_origins,
        "require_api_key": SETTINGS.require_api_key,
        "allow_local_without_api_key": SETTINGS.allow_local_without_api_key,
        "enable_debug_routes": SETTINGS.enable_debug_routes,
        "websocket_require_api_key": SETTINGS.websocket_require_api_key,
        "chain_require_privkey": SETTINGS.chain_require_privkey,
    }
