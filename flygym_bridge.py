"""FlyGym closed loop — thin compat shim (was 533 lines).

Full implementation now lives in :mod:`flybrain.body`. This module only
re-exports the public surface so ``from flygym_bridge import ...`` keeps
working.
"""
from flybrain.body.bridge import FlyGymBridge
from flybrain.body.constants import (
    EAT_DIST,
    FOOD_R,
    FOOD_SPAWN_R,
    FRAME_EVERY,
    PHYS_SUBSTEPS,
)
from flybrain.body.math_utils import mat2quat as _mat2quat
from flybrain.body.math_utils import wrap_angle as _wrap

__all__ = [
    "FlyGymBridge",
    "FRAME_EVERY",
    "PHYS_SUBSTEPS",
    "FOOD_R",
    "EAT_DIST",
    "FOOD_SPAWN_R",
]
