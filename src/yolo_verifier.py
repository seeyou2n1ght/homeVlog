"""Backward compatibility facade for src.algorithms.yolo_verifier."""
import sys
from src.algorithms import yolo_verifier as _real_module
from src.algorithms.yolo_verifier import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
