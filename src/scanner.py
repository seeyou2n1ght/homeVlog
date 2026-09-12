"""Backward compatibility facade for src.stages.scanner."""
import sys
from src.stages import scanner as _real_module
from src.stages.scanner import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
