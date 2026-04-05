import numpy as np

from app.pipeline.executor import PipelineExecutor
from app.pipeline.frame_context import FrameContext


class DummyComp:
    def __init__(self, fn=None):
        self.fn = fn or (lambda ctx: ctx)

    def run(self, ctx):
        return self.fn(ctx)


def _det(ctx):
    ctx.detections = [{"class_name": "chair", "x1": 0, "y1": 0, "x2": 2, "y2": 2, "confidence": 0.9}]
    ctx.metrics["yolo_latency_ms"] = 1.0
    return ctx


def _dep(ctx):
    ctx.depth_map = np.ones((4, 4), dtype=float)
    ctx.depth_color = np.zeros((4, 4, 3), dtype=np.uint8)
    ctx.metrics["depth_latency_ms"] = 2.0
    return ctx


def _nav(ctx):
    ctx.zone_risks = {"left": 1.0, "center": 0.5, "right": 0.2}
    ctx.nav_command = "Path clear"
    return ctx


def test_sequential_executor_basic_behavior():
    ex = PipelineExecutor(DummyComp(_det), DummyComp(_dep), DummyComp(_nav), DummyComp(), DummyComp(), execution_mode="sequential")
    ctx = FrameContext(frame_idx=1, source_id="x", frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    out = ex.run(ctx)
    assert out.metrics["yolo_latency_ms"] == 1.0
    assert out.metrics["depth_latency_ms"] == 2.0
    assert out.nav_command == "Path clear"


def test_threaded_parallel_executor_basic_behavior():
    ex = PipelineExecutor(DummyComp(_det), DummyComp(_dep), DummyComp(_nav), DummyComp(), DummyComp(), execution_mode="threaded_parallel")
    ctx = FrameContext(frame_idx=1, source_id="x", frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    out = ex.run(ctx)
    assert out.metrics["yolo_latency_ms"] == 1.0
    assert out.metrics["depth_latency_ms"] == 2.0


def test_merge_detector_depth_outputs_via_fusion():
    ex = PipelineExecutor(DummyComp(_det), DummyComp(_dep), DummyComp(_nav), DummyComp(), DummyComp(), execution_mode="sequential")
    ctx = FrameContext(frame_idx=1, source_id="x", frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    out = ex.run(ctx)
    assert out.detections[0]["depth_relative"] == 1.0
    assert out.metrics["nearest_object_depth_m"] == 1.0
