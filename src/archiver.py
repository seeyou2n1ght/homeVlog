"""Backward compatibility facade for src.stages.archiver."""
import sys
from src.stages import archiver as _real_module
from src.stages.archiver import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
