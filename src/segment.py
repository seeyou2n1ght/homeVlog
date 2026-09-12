"""Backward compatibility facade for src.algorithms.segment."""
import sys
from src.algorithms import segment as _real_module
from src.algorithms.segment import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
