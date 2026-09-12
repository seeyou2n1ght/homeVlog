"""Backward compatibility facade for src.stages.renderer."""
import sys
from src.stages import renderer as _real_module
from src.stages.renderer import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
