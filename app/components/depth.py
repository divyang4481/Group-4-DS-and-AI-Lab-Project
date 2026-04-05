from __future__ import annotations

import time

from app.mechanics.depth_estimation import DepthEstimator


class DepthComponent:
    def __init__(self, model_path: str, device: str = "cpu", depth_repo_path=None, enabled: bool = True):
        self.enabled = enabled
        self.depth_estimator = DepthEstimator(model_path, device=device, repo_path=depth_repo_path)

    def load(self):
        if self.enabled:
            self.depth_estimator.load_model()

    def run(self, ctx):
        if not self.enabled:
            return ctx
        start = time.perf_counter()
        depth_float, depth_color = self.depth_estimator.predict(ctx.frame_rgb)
        ctx.depth_map = depth_float
        ctx.depth_color = depth_color
        ctx.metrics["depth_latency_ms"] = (time.perf_counter() - start) * 1000.0
        return ctx
