"""Deterministic synthetic footage and test harness for HomeVlog E2E verification.

Provides synthetic media generators, corrupted media fixtures, and test assertions.
Strictly non-mutating: does NOT monkeypatch or modify production classes/modules.
"""

from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from src.hardware import scheduler
from src.hardware.ffmpeg import FFmpegProcessRegistry

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
    create_synthetic_clip(
        path=path,
        scene=scene,
        duration_s=CLIP_SECONDS,
        fps=FPS,
        width=128,
        height=128,
        gop_size=1,
    )


def create_synthetic_clip(
    path: Path,
    scene: str = "static",
    duration_s: float = 4.0,
    fps: int = 20,
    width: int = 128,
    height: int = 128,
    gop_size: int = 20,
    motion_windows: list[tuple[float, float]] | None = None,
    audio_windows: list[tuple[float, float]] | None = None,
    codec: str = "mpeg4",
) -> Path:
    """Generate deterministic synthetic video with configurable GOP and temporal motion/audio windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration_s * fps)))
    samples_per_frame = AUDIO_RATE // fps

    with av.open(str(path), "w") as container:
        video = container.add_stream(codec, rate=fps)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        video.codec_context.gop_size = gop_size

        has_audio = audio_windows is not None or scene == "motion_audio"
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono") if has_audio else None

        for index in range(total_frames):
            t_curr = index / fps
            is_active = False
            if motion_windows:
                is_active = any(w_start <= t_curr < w_end for w_start, w_end in motion_windows)
            elif scene == "motion":
                is_active = True
            elif scene == "arrival":
                is_active = (index < fps * 2)
            elif scene == "dense_motion":
                is_active = (index < total_frames * 0.75)
            elif scene == "motion_audio":
                is_active = (index < fps * 2)

            pixels = np.full((height, width), 40, dtype=np.uint8)
            if is_active:
                left = 24 + (index // 2) % max(1, width - 50)
                pixels[24:min(104, height), left:min(width, left + 42)] = 225 if (index // 4) % 2 else 60

            frame = av.VideoFrame.from_ndarray(pixels, format="gray")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            container.mux(video.encode(frame))

            if audio is not None:
                sample_start = index * samples_per_frame
                samples = np.zeros(samples_per_frame, dtype=np.float32)
                audio_active = False
                if audio_windows:
                    audio_active = any(w_start <= t_curr < w_end for w_start, w_end in audio_windows)
                elif scene == "motion_audio" and (5.0 <= t_curr < 7.0):
                    audio_active = True

                if audio_active:
                    t_audio = (np.arange(samples_per_frame) + sample_start) / AUDIO_RATE
                    samples[:] = 0.5 * np.sin(2 * np.pi * 800 * t_audio)

                audio_frame = av.AudioFrame.from_ndarray(samples[None], format="fltp", layout="mono")
                audio_frame.pts = sample_start
                audio_frame.sample_rate = AUDIO_RATE
                audio_frame.time_base = Fraction(1, AUDIO_RATE)
                container.mux(audio.encode(audio_frame))

        container.mux(video.encode())
        if audio is not None:
            container.mux(audio.encode())
    return path


def create_corrupted_clip(path: Path, corruption_type: str = "zero_byte") -> Path:
    """Generate intentionally damaged or malformed files for boundary testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if corruption_type == "zero_byte":
        path.touch()
    elif corruption_type == "truncated_eof":
        valid = path.with_suffix(".temp.mp4")
        create_synthetic_clip(valid, duration_s=4.0)
        data = valid.read_bytes()
        valid.unlink()
        path.write_bytes(data[: max(1, len(data) // 3)])
    elif corruption_type == "garbage":
        path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00" + b"\xff" * 2048)
    return path


def build_footage(root: Path) -> dict[str, Path]:
    """Make six local inputs, including an arrival/hold scene and a zero-byte source."""
    footage = {}
    start = datetime.strptime(f"{DATE}120000", "%Y%m%d%H%M%S")
    for cam, scenes in enumerate((("static", "motion_audio", "arrival", "empty", "motion"), ("secondary",))):
        folder = root / f"XiaomiCamera_{cam + 1:02}_{CAMERA_MACS[cam]}"
        folder.mkdir(parents=True, exist_ok=True)
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


def build_custom_dataset(root: Path, specs: dict[int, list[dict]]) -> dict[str, Path]:
    """Build multi-camera mock share directory from a declarative specification."""
    files = {}
    for cam_idx, clip_list in specs.items():
        mac = CAMERA_MACS[cam_idx]
        folder = root / f"XiaomiCamera_{cam_idx + 1:02}_{mac}"
        folder.mkdir(parents=True, exist_ok=True)
        for clip in clip_list:
            path = folder / clip["name"]
            if clip.get("corruption"):
                create_corrupted_clip(path, clip["corruption"])
            else:
                create_synthetic_clip(
                    path,
                    scene=clip.get("scene", "static"),
                    duration_s=clip.get("duration_s", 4.0),
                    fps=clip.get("fps", 20),
                    width=clip.get("width", 128),
                    height=clip.get("height", 128),
                    gop_size=clip.get("gop", 20),
                    motion_windows=clip.get("motion"),
                    audio_windows=clip.get("audio"),
                )
            files[f"cam{cam_idx}_{clip['name']}"] = path
    return files


class MockVramController:
    """Allows test cases to dynamically simulate physical GPU VRAM usage and margin."""
    def __init__(self, free_mb: int = 4096, used_mb: int = 3500, watermark_mb: int = 6200):
        self.free_mb = free_mb
        self.used_mb = used_mb
        self.watermark_mb = watermark_mb

    def set_vram(self, free_mb: int, used_mb: int = 3500):
        self.free_mb = free_mb
        self.used_mb = used_mb

    def probe_used(self) -> int:
        return self.used_mb

    def probe_margin(self) -> int:
        return self.free_mb


def assert_resource_hygiene():
    """Verify all concurrency and subprocess semaphores and queues are cleanly released."""
    io = scheduler.get_disk_semaphore()
    if hasattr(io, "available") and hasattr(io, "limit"):
        assert io.available == io.limit, f"Disk semaphore leaked: {io.available} < {io.limit}"
    if hasattr(io, "waiters"):
        assert not io.waiters, "Disk semaphore waiters remained"
    for sem in (
        scheduler._nvdec_semaphore,
        scheduler._nvenc_semaphore,
        scheduler._qsv_semaphore,
        scheduler._qsv_render_semaphore,
    ):
        if sem is not None:
            if hasattr(sem, "available") and hasattr(sem, "limit"):
                assert sem.available == sem.limit, f"Semaphore leaked: {sem.available} < {sem.limit}"
                if hasattr(sem, "waiters"):
                    assert not sem.waiters, "Semaphore waiters remained"
            elif hasattr(sem, "_value"):
                assert sem._value > 0, f"Semaphore value non-positive: {sem._value}"
    assert not FFmpegProcessRegistry._processes, f"Leaked processes: {FFmpegProcessRegistry._processes}"
