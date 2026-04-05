from __future__ import annotations

import time

from app.mechanics.object_detection import ObjectDetector


class DetectorComponent:
    def __init__(self, weights_path: str, enabled: bool = True):
        self.enabled = enabled
        self.detector = ObjectDetector(weights_path)

    def load(self):
        if self.enabled:
            self.detector.load_model()

    def run(self, ctx):
        if not self.enabled:
            return ctx
        start = time.perf_counter()
        _, detections = self.detector.predict(ctx.frame_bgr)
        ctx.detections = detections
        ctx.metrics["yolo_latency_ms"] = (time.perf_counter() - start) * 1000.0
        ctx.metrics["detection_count"] = len(detections)
        ctx.metrics["detected_classes"] = sorted({d["class_name"] for d in detections})
        return ctx
