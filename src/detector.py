"""Backward compatibility facade for src.stages.detector."""
import sys
from src.stages import detector as _real_module
from src.stages.detector import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
