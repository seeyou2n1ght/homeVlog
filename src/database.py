"""Backward compatibility facade for src.core.database."""
import sys
from src.core import database as _real_module
from src.core.database import *  # noqa: F401, F403

sys.modules[__name__] = _real_module
