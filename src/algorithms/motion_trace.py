"""Consume decoded frames immediately; retain only motion scalars and model state."""
from array import array

import cv2
import numpy as np


class MotionTrace:
    def __init__(self, detector, fps, is_night_mode=False):
        self.detector = detector
        self.dt = 1.0 / fps
        self.is_night_mode = bool(is_night_mode)
        self.ema = detector.create_ema_model()
        self.grid = detector.create_grid_filter()
        self.previous = None
        self.energies = array("d")
        self.confidences = array("d")

    def __len__(self):
        return len(self.energies)

    def append(self, frame):
        det = self.detector
        if frame.ndim == 3:
            gray = cv2.cvtColor(
                cv2.resize(frame, (det.width, det.height), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGR2GRAY,
            )
        elif frame.shape == (det.height, det.width):
            gray = frame
        else:
            gray = cv2.resize(frame, (det.width, det.height), interpolation=cv2.INTER_NEAREST)
        x, y = int(det.width * det.roi[0]), int(det.height * det.roi[1])
        w, h = int(det.width * det.roi[2]), int(det.height * det.roi[3])
        roi = gray[y:y + h, x:x + w]
        if det.ema_enabled:
            saliency, _, _ = self.ema.update(roi)
        else:
            saliency = (cv2.absdiff(roi, self.previous).astype(np.float32)
                        if self.previous is not None else np.zeros_like(roi, dtype=np.float32))
            # Decoder buffers may be reused after append returns.
            self.previous = roi.copy()
        energy, _, stats = self.grid.process_frame(
            saliency, self.dt, is_night_mode=self.is_night_mode
        )
        self.energies.append(energy)
        self.confidences.append(float(stats.get("max_confidence", 0.0)))
