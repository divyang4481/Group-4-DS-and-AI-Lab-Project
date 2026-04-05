from __future__ import annotations

import time

import cv2

from app.mechanics.object_detection import draw_centered_label


class VisualizationComponent:
    def __init__(self, enabled: bool = True, plot_stream_mode: int = 0):
        self.enabled = enabled
        self.plot_stream_mode = plot_stream_mode

    def run(self, ctx):
        if not self.enabled:
            return ctx
        start = time.perf_counter()
        plot_frame = ctx.frame_bgr.copy() if self.plot_stream_mode == 0 else (ctx.depth_color.copy() if ctx.depth_color is not None else ctx.frame_bgr.copy())

        for det in ctx.detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance_label = "dist: N/A"
            if det.get("distance_relative") is not None:
                distance_label = f"dist: {det['distance_relative']:.3f}"
            cv2.rectangle(plot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(plot_frame, f"{det['class_name']} {det['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            draw_centered_label(plot_frame, distance_label, center_x, center_y)

        ctx.annotated_frame = plot_frame
        ctx.metrics["visualization_latency_ms"] = (time.perf_counter() - start) * 1000.0
        return ctx
