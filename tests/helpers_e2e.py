"""Deterministic two-camera footage with scanner-compatible names and durations."""

from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

DATE = "20260901"
CAMERA_MACS = ("B888805AA3CD", "B888806BD537")
CLIP_SECONDS = 8
FPS = 20
AUDIO_RATE = 16_000


def _pixels(scene: str, frame_index: int) -> np.ndarray:
    pixels = np.full((128, 128), 40, dtype=np.uint8)
    if scene == "motion_audio" and frame_index < 2 * FPS:
        left = 24 + frame_index // 2
        pixels[24:104, left:left + 42] = 225 if (frame_index // 4) % 2 else 60
    elif scene == "arrival":
        left = min(48, 12 + frame_index)
        pixels[30:106, left:left + 30] = 180
    elif scene == "motion":
        left = 24 + frame_index // 4
        pixels[24:104, left:left + 42] = 225 if (frame_index // 4) % 2 else 60
    return pixels


def _write_clip(path: Path, scene: str) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=FPS)
        video.width = video.height = 128
        video.pix_fmt = "yuv420p"
        video.codec_context.gop_size = 1
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono") if scene == "motion_audio" else None
        samples_per_frame = AUDIO_RATE // FPS
        for index in range(CLIP_SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(_pixels(scene, index), format="gray")
            frame.pts = index
            frame.time_base = Fraction(1, FPS)
            container.mux(video.encode(frame))
            if audio is not None:
                sample_start = index * samples_per_frame
                samples = np.zeros(samples_per_frame, dtype=np.float32)
                if 5 * FPS <= index < 7 * FPS:
                    time = (np.arange(samples_per_frame) + sample_start) / AUDIO_RATE
                    samples[:] = .5 * np.sin(2 * np.pi * 800 * time)
                audio_frame = av.AudioFrame.from_ndarray(samples[None], format="fltp", layout="mono")
                audio_frame.pts = sample_start
                audio_frame.sample_rate = AUDIO_RATE
                audio_frame.time_base = Fraction(1, AUDIO_RATE)
                container.mux(audio.encode(audio_frame))
        container.mux(video.encode())
        if audio is not None:
            container.mux(audio.encode())


def build_footage(root: Path) -> dict[str, Path]:
    """Make six local inputs, including an arrival/hold scene and a zero-byte source."""
    footage = {}
    start = datetime.strptime(f"{DATE}120000", "%Y%m%d%H%M%S")
    for cam, scenes in enumerate((("static", "motion_audio", "arrival", "empty", "motion"), ("secondary",))):
        folder = root / f"XiaomiCamera_{cam + 1:02}_{CAMERA_MACS[cam]}"
        folder.mkdir(parents=True)
        for index, scene in enumerate(scenes):
            clip_start = start + timedelta(seconds=CLIP_SECONDS * index)
            clip_end = clip_start + timedelta(seconds=CLIP_SECONDS)
            path = folder / f"00_{clip_start:%Y%m%d%H%M%S}_{clip_end:%Y%m%d%H%M%S}.mp4"
            if scene == "empty":
                path.touch()
            else:
                _write_clip(path, scene)
            footage[scene] = path
    return footage
