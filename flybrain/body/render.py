"""Follow-camera JPEG rendering."""
import io

import mujoco as mj
import numpy as np
from PIL import Image

from . import constants as K
from .math_utils import mat2quat


def render_follow_jpeg(sim, renderer, cam_id: int, fly_pos: np.ndarray):
    try:
        look = fly_pos + np.array([0.0, 0.0, 1.0])
        pos = fly_pos + K.CAM_OFFSET
        d = look - pos
        d /= np.linalg.norm(d)
        z_cam = -d
        x_cam = np.cross(np.array([0., 0., 1.]), z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        sim.mj_model.cam_pos[cam_id] = pos
        sim.mj_model.cam_quat[cam_id] = mat2quat(
            np.column_stack([x_cam, y_cam, z_cam]))
        sim.render_as_needed()
        frames = renderer.frames.get("arena", [])
        if not frames:
            return None
        buf = io.BytesIO()
        Image.fromarray(frames[-1]).save(
            buf, format="JPEG", quality=K.JPEG_QUALITY,
            subsampling=K.JPEG_SUBSAMPLING, optimize=True)
        renderer.frames["arena"] = []
        return buf.getvalue()
    except Exception as e:
        print(f"  [flygym] render failed: {e}", flush=True)
        return None
