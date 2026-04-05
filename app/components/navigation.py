from __future__ import annotations

import time

from app.mechanics.navigation_logic import NavigationLogic


class NavigationComponent:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.nav = None

    def run(self, ctx):
        if not self.enabled:
            return ctx
        if self.nav is None:
            self.nav = NavigationLogic(frame_width=ctx.frame_bgr.shape[1])

        nav_detections = [
            {
                "class": det["class_name"],
                "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "depth": det.get("depth_relative"),
                "distance": det.get("distance_relative"),
            }
            for det in ctx.detections
        ]
        start = time.perf_counter()
        zone_risks, command = self.nav.process_detections(nav_detections)
        ctx.metrics["navigation_latency_ms"] = (time.perf_counter() - start) * 1000.0
        ctx.metrics["deterministic_nav_latency_ms"] = ctx.metrics["navigation_latency_ms"]
        ctx.zone_risks = zone_risks
        ctx.nav_command = command
        ctx.metrics["left_risk"] = zone_risks.get("left", 0.0)
        ctx.metrics["center_risk"] = zone_risks.get("center", 0.0)
        ctx.metrics["right_risk"] = zone_risks.get("right", 0.0)
        return ctx
