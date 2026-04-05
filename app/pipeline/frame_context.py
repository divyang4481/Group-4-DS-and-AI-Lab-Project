from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FrameContext:
    frame_idx: int
    source_id: str
    frame_bgr: np.ndarray
    frame_rgb: Optional[np.ndarray] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    depth_color: Optional[np.ndarray] = None
    zone_risks: Dict[str, float] = field(default_factory=lambda: {"left": 0.0, "center": 0.0, "right": 0.0})
    nav_command: str = ""
    annotated_frame: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
