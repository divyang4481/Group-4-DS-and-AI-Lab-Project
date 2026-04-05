from __future__ import annotations

from typing import Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np

from .base import FrameSource


class WebcamFrameSource(FrameSource):
    def __init__(self, camera_index: int = 0):
        self.camera_index = int(camera_index)
        self.source_id = f"webcam:{self.camera_index}"
        self._cap = None

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for webcam source")
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.camera_index}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Webcam source is not opened")
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
