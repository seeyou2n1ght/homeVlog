"""Backward compatibility facade for src.algorithms.audio_features."""
import sys
from src.algorithms import audio_features as _real_module
from src.algorithms.audio_features import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
