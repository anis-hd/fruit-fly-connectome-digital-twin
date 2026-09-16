"""Back-compat re-export (split into loader.py + loop.py + body_drive.py)."""
from .body_drive import step_body
from .loader import load_connectome_into
from .loop import reset_run, simulation_loop

__all__ = ["load_connectome_into", "simulation_loop", "reset_run", "step_body"]
