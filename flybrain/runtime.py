"""Shared live-simulation runtime: state, brain data, websocket fan-out."""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from fastapi import WebSocket

from . import config as C


@dataclass
class SimState:
    running: bool = False
    paused: bool = False
    step: int = 0
    T_STEPS: int = 1200
    speed: float = 1.0
    stim_windows: List[tuple] = field(
        default_factory=lambda: [(300, 500), (700, 900)])
    stim_rate_hz: float = 60.0
    stim_fraction: float = 0.3
    i_ext: float = 2.0
    gain: float = 2.5
    noise: float = 0.06
    tau_mem: float = 10.0
    v_thr: float = 0.8
    manual_stim_sensory: int = 0
    manual_stim_all: int = 0
    # FlyGym body drive: named sense channels (0..1 levels).
    bridge: Dict[str, float] = field(default_factory=dict)


@dataclass
class BrainData:
    """Connectome-derived tensors (replaces server.py module globals)."""
    W = None
    universe = None
    lut = None
    id_ann = None
    sens_idx = None
    mot_idx = None
    mon = None
    mon_t = None
    stim_sub = None
    stim_sub_t = None
    bounds = None
    mon_ids_list = None
    cached_annotations = None
    bridge_sens_subs: dict = field(default_factory=dict)
    mot_left_t = None
    mot_right_t = None
    mot_left_n: int = 0
    mot_right_n: int = 0
    leg_pool_t: dict = field(default_factory=dict)
    desc_t = None
    descL_t = None
    descR_t = None
    abd_t = None
    head_t = None


@dataclass
class DriveState:
    """Per-run motor smoothing + histories + body handles."""
    bridge_motor: dict = field(default_factory=lambda: {
        "left": 0.0, "right": 0.0, "all": 0.0, "step": 0})
    drive_ema: dict = field(default_factory=lambda: {
        "ampL": 0.0, "ampR": 0.0, "turn": 0.0, "speed": 0.0})
    bridge_ts: dict = field(default_factory=dict)
    pop_rate_history: list = field(default_factory=list)
    mot_rate_history: list = field(default_factory=list)
    raster_buffer: list = field(default_factory=list)
    run_completed: bool = False
    loop: Optional[asyncio.AbstractEventLoop] = None
    flygym = None
    flygym_tele: dict = field(default_factory=dict)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        import json
        data = json.dumps(message)
        disconnected = set()
        for ws in tuple(self.active_connections):
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.active_connections.discard(ws)


@dataclass
class Runtime:
    """One object threading state through loop/api/control (no globals)."""
    state: SimState = field(default_factory=SimState)
    brain_data: BrainData = field(default_factory=BrainData)
    drive: DriveState = field(default_factory=DriveState)
    manager: ConnectionManager = field(default_factory=ConnectionManager)
    brain = None  # FlyBrainLIF, set after load
