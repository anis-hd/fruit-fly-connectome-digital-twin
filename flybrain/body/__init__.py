"""Body subpackage: MuJoCo fly driven by brain spikes, senses back."""
from .bridge import FlyGymBridge
from .constants import FRAME_EVERY

__all__ = ["FlyGymBridge", "FRAME_EVERY"]
