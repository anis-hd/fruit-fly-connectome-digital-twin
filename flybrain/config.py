"""Central configuration. Previously duplicated at the top of app.py/server.py.

The offline pipeline and the live server intentionally use *different*
tunables (thresholds, gains, stimulus). They are kept as two frozen
dataclasses so a refactor can never silently unify them.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import torch


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- neurotransmitter keyword groups (shared) ---
EXC_KEYS = ["ach", "acetylcholine"]
INH_KEYS = ["gaba", "glut", "histamine"]
MOD_KEYS = ["serotonin", "5-ht", "5ht", "dopamine", "octopamine"]
MOD_SIGN = 1.0  # modulatory treated as weakly excitatory

# --- sensory / motor keyword matching on annotation text (shared) ---
SENS_KW = [
    "sensory", "receptor", "gustatory", "grn", "olfactory", "photoreceptor",
    "visual", "mechanosensory", "chordotonal", "johnston", "antennal",
    "thermosensory", "hygrosensory", "nociceptor", "ascending",
]
MOT_KW = [r"motor", r"\bmn\b", r"mn\d"]

CANDIDATE_TEXT_COLS = [
    "superclass", "class", "subclass", "type", "flywireType",
    "hemibrainType", "mancType", "receptorType", "group",
    "instance", "synonyms", "statusLabel",
]


@dataclass(frozen=True)
class OfflineConfig:
    """Tunables for the offline raster pipeline (original app.py values)."""

    ann_path: str = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path: str = "body-neurotransmitters-male-cns-v1.0.feather"
    conn_path: str = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    seed: int = 0
    dt: float = 1.0
    tau_mem: float = 10.0
    v_thr: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2
    background_noise: float = 0.03
    norm_mode: str = "input"
    gain: float = 1.6
    w_syn: float = 0.02
    min_weight: int = 1
    t_steps: int = 1200
    stim_windows: List[Tuple[int, int]] = field(
        default_factory=lambda: [(300, 500), (700, 900)]
    )
    stim_rate_hz: float = 40.0
    stim_fraction: float = 0.25
    i_ext: float = 1.5
    live: bool = True
    vis_every: int = 5
    monitor_cap: int = 1200


@dataclass(frozen=True)
class ServerConfig:
    """Tunables for the live FastAPI server (original server.py values)."""

    ann_path: str = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path: str = "body-neurotransmitters-male-cns-v1.0.feather"
    conn_path: str = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    seed: int = 0
    dt: float = 1.0
    tau_mem: float = 10.0
    v_thr: float = 0.8
    v_reset: float = 0.0
    refractory: int = 2
    background_noise: float = 0.06
    norm_mode: str = "input"
    gain: float = 2.5
    w_syn: float = 0.02
    min_weight: int = 5  # 26M edges @1 -> 6.3M @5
    t_steps: int = 1200
    stim_windows: List[Tuple[int, int]] = field(
        default_factory=lambda: [(300, 500), (700, 900)]
    )
    stim_rate_hz: float = 60.0
    stim_fraction: float = 0.3
    i_ext: float = 2.0
    monitor_cap: int = 1200
    vis_every: int = 10
    sim_speed: float = 1.0


# --- FlyGym bridge wiring (shared anatomical channel order) ---
BRIDGE_CHANNELS = (
    "smell", "left", "right", "touch",
    "taste", "wind", "light", "warm", "cool",
    "propL", "propR", "humid",
)
LEG_NERVES = ("ProLN", "MesoLN", "MetaLN")
LEG_SEGS = ("T1", "T2", "T3")
LIGHT_CLASSES = (
    "visual_projection", "ol_sensory", "cb_sensory", "visual_centrifugal",
)
BRIDGE_TIMEOUT_S = 5.0
MANUAL_STIM_DURATION = 120

# Motor-unit recruitment + CPG smoothing (calibrated 2026-09-16, see server.py).
AMP_EMA_ALPHA = 0.06
SPEED_EMA_ALPHA = 0.04
TURN_EMA_ALPHA = 0.08
TURN_DEADBAND = 0.10
DRIVE_EMA_ALPHA = AMP_EMA_ALPHA  # legacy alias
TURN_GAIN = 50.0
RECRUIT_GAIN = 60.0
RECRUIT_FLOOR = 0.0005


def recruit(x: float, gain: float = RECRUIT_GAIN,
            floor: float = RECRUIT_FLOOR) -> float:
    """Map a motor-pool spike fraction (0..~0.03) to CPG amplitude (0..1)."""
    return max(0.0, min(1.0, (float(x) - floor) * gain))
