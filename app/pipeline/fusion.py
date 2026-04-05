from __future__ import annotations

import time
from statistics import mean
from typing import Dict, List, Optional

from app.mechanics.depth_estimation import estimate_distance_from_depth


class DetectionDepthFusion:
    def run(self, ctx):
        start = time.perf_counter()
        depths = []
        for det in ctx.detections:
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            depth_m, relative_distance = estimate_distance_from_depth(ctx.depth_map, bbox)
            det["depth_relative"] = depth_m
            det["distance_relative"] = relative_distance
            if depth_m is not None:
                depths.append(float(depth_m))
        ctx.metrics["fusion_latency_ms"] = (time.perf_counter() - start) * 1000.0
        ctx.metrics["nearest_object_depth_m"] = min(depths) if depths else None
        ctx.metrics["farthest_object_depth_m"] = max(depths) if depths else None
        ctx.metrics["mean_detected_depth_m"] = mean(depths) if depths else None
        return ctx

    @staticmethod
    def nearest_depth(detections: List[Dict]) -> Optional[float]:
        vals = [d.get("depth_relative") for d in detections if d.get("depth_relative") is not None]
        return min(vals) if vals else None

    @staticmethod
    def farthest_depth(detections: List[Dict]) -> Optional[float]:
        vals = [d.get("depth_relative") for d in detections if d.get("depth_relative") is not None]
        return max(vals) if vals else None

    @staticmethod
    def mean_depth(detections: List[Dict]) -> Optional[float]:
        vals = [d.get("depth_relative") for d in detections if d.get("depth_relative") is not None]
        return mean(vals) if vals else None
