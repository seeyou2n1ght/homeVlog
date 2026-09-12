"""Backward compatibility facade for src.hardware.render_cache."""
import sys
from src.hardware import render_cache as _real_module
from src.hardware.render_cache import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
