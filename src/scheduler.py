import logging

logger = logging.getLogger("homevlog")


def choose_encoder(cam_index: int) -> str:
    """Pick NVENC for cam0 (primary), QSV for cam1 (secondary)."""
    if cam_index == 0:
        return "nv"
    return "qsv"
