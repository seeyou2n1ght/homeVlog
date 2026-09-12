"""Backward compatibility facade for src.core.utils."""
import sys
from src.core import utils as _real_module
from src.core.utils import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
