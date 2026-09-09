"""Lossless in-memory frame storage; decompress only the frame being analyzed."""
import zlib
import numpy as np


class FramePool:
    def __init__(self, limit_bytes):
        self.limit_bytes = limit_bytes
        self.bytes_used = 0
        self._frames = []

    def append(self, frame):
        data = zlib.compress(frame.tobytes(), level=1)
        if self.bytes_used + len(data) > self.limit_bytes:
            raise MemoryError("Analysis frame budget exceeded; split source into shorter files")
        self._frames.append((frame.shape, data))
        self.bytes_used += len(data)

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        for shape, data in self._frames:
            yield np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(shape)
