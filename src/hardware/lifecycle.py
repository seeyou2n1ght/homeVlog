"""Cleanup shared resources at the end of a processing run."""

import gc
import logging

from src.hardware.ffmpeg import FFmpegProcessRegistry


logger = logging.getLogger("homevlog")


def cleanup_resources(db=None):
    """Collect garbage, clear CUDA cache, checkpoint SQLite, and stop tracked FFmpeg processes."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("CUDA cache clear skipped: %s", exc)

    if db is not None:
        try:
            db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as exc:
            logger.debug("WAL checkpoint skipped: %s", exc)

    try:
        FFmpegProcessRegistry.kill_all()
    except Exception as exc:
        logger.warning("cleanup error: %s", exc)
