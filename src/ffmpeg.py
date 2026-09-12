"""Backward compatibility facade for src.hardware.ffmpeg."""
import sys
from src.hardware import ffmpeg as _real_module
from src.hardware.ffmpeg import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
