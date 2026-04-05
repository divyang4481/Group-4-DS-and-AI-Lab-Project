from __future__ import annotations

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from app.pipeline.frame_context import FrameContext


class LiveRunner:
    def __init__(self, orchestrator, source, tracker, exporter=None, max_frames=None, stride=1, show_windows=False):
        self.orchestrator = orchestrator
        self.source = source
        self.tracker = tracker
        self.exporter = exporter
        self.max_frames = max_frames
        self.stride = max(1, stride)
        self.show_windows = show_windows

    def run(self):
        self.orchestrator.load()
        self.source.open()
        frame_idx = 0
        processed = 0
        try:
            while True:
                ok, frame = self.source.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % self.stride != 0:
                    continue
                ctx = FrameContext(frame_idx=frame_idx, source_id=self.source.source_id, frame_bgr=frame)
                self.orchestrator.executor.run(ctx)
                self.tracker.log_frame(ctx)
                processed += 1

                if self.show_windows and cv2 is not None and ctx.annotated_frame is not None:
                    cv2.imshow("Pipeline", ctx.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self.max_frames is not None and processed >= self.max_frames:
                    break
        finally:
            self.source.close()
            self.orchestrator.close()
            if cv2 is not None:
                cv2.destroyAllWindows()

        if self.exporter is not None:
            self.exporter.export(self.tracker.frame_metrics, self.tracker.resource_samples, self.tracker.duration_seconds)
        return self.tracker.frame_metrics
