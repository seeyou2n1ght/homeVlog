"""Backward compatibility facade for src.algorithms.motion_trace."""
import sys
from src.algorithms import motion_trace as _real_module
from src.algorithms.motion_trace import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
