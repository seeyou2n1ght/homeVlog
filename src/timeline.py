"""Backward compatibility facade for src.stages.timeline."""
import sys
from src.stages import timeline as _real_module
from src.stages.timeline import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
