from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .fusion import DetectionDepthFusion
from .modes import THREADED_PARALLEL


class PipelineExecutor:
    def __init__(self, detector, depth, navigation, visualization, tts, execution_mode="sequential"):
        self.detector = detector
        self.depth = depth
        self.navigation = navigation
        self.visualization = visualization
        self.tts = tts
        self.execution_mode = execution_mode
        self.fusion = DetectionDepthFusion()

    def run(self, ctx):
        loop_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        ctx.frame_rgb = ctx.frame_bgr[:, :, ::-1].copy()
        ctx.metrics["preprocess_latency_ms"] = (time.perf_counter() - preprocess_start) * 1000.0

        if self.execution_mode == THREADED_PARALLEL:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_det = pool.submit(self.detector.run, ctx)
                f_dep = pool.submit(self.depth.run, ctx)
                _ = f_det.result()
                _ = f_dep.result()
        else:
            self.detector.run(ctx)
            self.depth.run(ctx)

        self.fusion.run(ctx)
        self.navigation.run(ctx)
        self.visualization.run(ctx)
        self.tts.run(ctx)

        ctx.metrics["frame_total_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
        total_ms = ctx.metrics["frame_total_latency_ms"]
        ctx.metrics["fps_instant"] = (1000.0 / total_ms) if total_ms > 0 else None
        ctx.metrics.setdefault("nav_command", ctx.nav_command)
        return ctx
