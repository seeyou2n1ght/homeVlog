"""Streaming sufficient statistics for energy VAD, without retaining PCM."""
import numpy as np


class AudioFeatures:
    def __init__(self, sample_rate=16000, window_ms=50, limit_bytes=64 * 1024 * 1024):
        self.window_samples = max(1, int(sample_rate * window_ms / 1000))
        self.limit_bytes = limit_bytes
        self.size = 0
        self._pending = np.empty(0, np.float32)
        self._dbfs = []
        self._voiced = []
        self._windows = 0

    def append(self, samples):
        samples = np.asarray(samples, np.float32).reshape(-1)
        self.size += samples.size
        samples = np.concatenate((self._pending, samples))
        count = samples.size // self.window_samples
        if not count:
            self._pending = samples.copy()
            return
        if (self._windows + count) * 8 > self.limit_bytes:
            raise MemoryError("Audio feature budget exceeded")
        blocks = samples[:count * self.window_samples].reshape(count, self.window_samples)
        power = np.mean(blocks * blocks, axis=1)
        self._dbfs.append(20 * np.log10(np.sqrt(power) + 1e-7))
        cov = np.mean(blocks[:, 1:] * blocks[:, :-1], axis=1)
        self._voiced.append((cov / (power + 1e-9) >= .5) & (np.std(blocks, axis=1) > 1e-4))
        self._windows += count
        self._pending = samples[count * self.window_samples:].copy()

    def dbfs_windows(self):
        if self._dbfs:
            return np.concatenate(self._dbfs)
        if self._pending.size:
            rms = np.sqrt(np.mean(self._pending ** 2))
            return np.array([20 * np.log10(rms + 1e-7)], np.float32)
        return np.empty(0, np.float32)

    def voiced_windows(self):
        return np.concatenate(self._voiced) if self._voiced else np.array([False])
