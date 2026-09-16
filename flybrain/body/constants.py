"""Body constants. Extracted from flygym_bridge.py header."""
import math

import numpy as np

PHYS_SUBSTEPS = 10          # 10 x 0.1ms physics per brain step (1ms sim time)
FRAME_EVERY = 6             # render one video frame every N brain steps
FLYGYM_CAMERA_RES = (720, 960)
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 0        # 0 = 4:4:4, sharpest
FOOD_R = 1.5                # berry radius, mm
EAT_DIST = 3.0              # mm
FOOD_SPAWN_R = (12.0, 30.0)
CAM_OFFSET = np.array([13.0, -15.0, 12.0])
SPAWN_POS = np.array([0.0, 0.0, 1.5])
FOOD_Z_FLAT = FOOD_R
FOOD_Z_BLOCKS = 0.4 + FOOD_R

LEG_ORDER = ("lf", "lm", "lh", "rf", "rm", "rh")

# Virtual climate fields.
TEMP_AMBIENT = 25.0
WARM_X, WARM_Y, WARM_T, WARM_SIGMA = -6.0, 5.0, 33.0, 4.0
COOL_X, COOL_Y, COOL_T, COOL_SIGMA = 7.0, 6.0, 17.0, 4.0
TEMP_RANGE = 8.0
HUMID_X, HUMID_Y, HUMID_SIGMA = -7.0, -5.0, 5.0
HUMID_AMBIENT = 0.25

# Proprioception calibration (2026-09-16, joint-encoder reads).
PROP_DEV_REF = 0.15
PROP_VEL_REF = 3.0

# Natural gait gears + steering.
GEAR_FREQS = (5.0, 8.0, 12.0)  # amble / walk / stride (Hz)
TRIPOD_INIT = {"lf": 0.0, "lm": math.pi, "lh": 0.0,
               "rf": math.pi, "rm": 0.0, "rh": math.pi}
