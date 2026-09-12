"""Backward compatibility facade for src.stages.prescreen."""
import sys
from src.stages import prescreen as _real_module
from src.stages.prescreen import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
