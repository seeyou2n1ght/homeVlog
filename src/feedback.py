"""Backward compatibility facade for src.stages.feedback."""
import sys
from src.stages import feedback as _real_module
from src.stages.feedback import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
