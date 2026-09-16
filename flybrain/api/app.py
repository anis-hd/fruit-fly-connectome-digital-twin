"""FastAPI app factory. Composition root for the live server."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import ServerConfig
from ..runtime import Runtime
from ..simulation import simulation_loop
from .routes import register_routes
from .ws import register_websocket


def create_app(rt: Runtime | None = None,
               cfg: ServerConfig | None = None) -> FastAPI:
    rt = rt or Runtime()
    cfg = cfg or ServerConfig()
    # Seed once at startup (was module-level in server.py).
    import numpy as np
    import torch
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        asyncio.create_task(simulation_loop(rt, cfg))
        yield

    app = FastAPI(title="Fly Brain Simulator", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="static"), name="static")
    register_routes(app, rt)
    register_websocket(app, rt)
    app.state.runtime = rt
    app.state.config = cfg
    return app
