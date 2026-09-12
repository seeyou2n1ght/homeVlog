"""Backward compatibility facade for src.hardware.frame_pool."""
import sys
from src.hardware import frame_pool as _real_module
from src.hardware.frame_pool import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
