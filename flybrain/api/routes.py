"""REST routes. Registered onto the app by app.py factory."""
from fastapi import FastAPI
from fastapi.responses import FileResponse

from .control import apply_control, random_neurons, state_dict


def register_routes(app: FastAPI, rt) -> None:
    @app.get("/")
    async def root():
        return FileResponse("static/index.html")

    @app.get("/api/state")
    async def get_state():
        return state_dict(rt)

    @app.post("/api/control")
    async def control_simulation(action: dict):
        return await apply_control(rt, action)

    @app.get("/api/neurons")
    async def get_neurons(limit: int = 1000):
        return random_neurons(rt, limit)

    @app.get("/main.js")
    async def get_main_js():
        return FileResponse("static/main.js")

    @app.get("/api/annotations")
    async def get_annotations():
        if rt.brain_data.cached_annotations is not None:
            return rt.brain_data.cached_annotations
        return {"neurons": []}
