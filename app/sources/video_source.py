from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np

from .base import FrameSource


class VideoFileFrameSource(FrameSource):
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.source_id = f"video:{self.video_path}"
        self._cap = None

    def open(self) -> None:
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video source")
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Video source is not opened")
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
