from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
    from PIL import Image

from .base import FrameSource


class FrameFolderSource(FrameSource):
    def __init__(self, folder_path: str, patterns=None):
        self.folder_path = Path(folder_path)
        self.patterns = patterns or ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.source_id = f"frame_folder:{self.folder_path}"
        self._frames: List[Path] = []
        self._idx = 0

    def open(self) -> None:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Frame folder not found: {self.folder_path}")
        paths = []
        for p in self.patterns:
            paths.extend(self.folder_path.glob(p))
        self._frames = sorted(paths)
        self._idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._idx >= len(self._frames):
            return False, None
        frame_path = self._frames[self._idx]
        self._idx += 1
        if cv2 is not None:
            frame = cv2.imread(str(frame_path))
        else:
            frame = np.array(Image.open(frame_path).convert("RGB"))[:, :, ::-1]
        if frame is None:
            return False, None
        return True, frame

    def close(self) -> None:
        self._frames = []
        self._idx = 0
