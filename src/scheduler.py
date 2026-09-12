"""Backward compatibility facade for src.hardware.scheduler."""
import sys
from src.hardware import scheduler as _real_module
from src.hardware.scheduler import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
