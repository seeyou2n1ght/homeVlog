"""Backward compatibility facade for src.algorithms.filters."""
import sys
from src.algorithms import filters as _real_module
from src.algorithms.filters import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
