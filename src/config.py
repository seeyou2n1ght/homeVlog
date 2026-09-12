"""Backward compatibility facade for src.core.config."""
import sys
from src.core import config as _real_module
from src.core.config import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
