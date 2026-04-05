from __future__ import annotations

import time
from typing import Dict, List


class MetricsTracker:
    def __init__(self):
        self.frame_metrics: List[Dict] = []
        self.resource_samples: List[Dict] = []
        self.started_at = time.time()

    def log_frame(self, ctx):
        row = {"frame_idx": ctx.frame_idx, **ctx.metrics}
        row["nav_command"] = ctx.nav_command
        row["left_risk"] = ctx.zone_risks.get("left", 0.0)
        row["center_risk"] = ctx.zone_risks.get("center", 0.0)
        row["right_risk"] = ctx.zone_risks.get("right", 0.0)
        self.frame_metrics.append(row)

    def log_resource(self, frame_idx: int, sample: Dict):
        self.resource_samples.append({"frame_idx": frame_idx, **sample})

    @property
    def duration_seconds(self):
        return time.time() - self.started_at
