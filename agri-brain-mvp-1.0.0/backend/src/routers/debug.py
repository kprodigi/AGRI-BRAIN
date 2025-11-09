from fastapi import APIRouter, FastAPI, Request

router = APIRouter()

@router.get("/debug/routes")
def debug_routes(request: Request):
    app: FastAPI = request.app
    return [
        {"path": r.path, "methods": list(r.methods or []), "name": r.name}
        for r in app.router.routes
    ]
