"""Backward compatibility facade for src.hardware.monitor."""
import sys
from src.hardware import monitor as _real_module
from src.hardware.monitor import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
