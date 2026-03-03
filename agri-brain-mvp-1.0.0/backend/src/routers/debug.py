from fastapi import APIRouter, FastAPI, Request

router = APIRouter()

@router.get("/debug/routes")
def debug_routes(request: Request):
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
