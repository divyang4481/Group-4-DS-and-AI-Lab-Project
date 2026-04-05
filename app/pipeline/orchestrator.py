from __future__ import annotations

from app.pipeline.executor import PipelineExecutor


class _NoopComponent:
    def load(self):
        return None

    def run(self, ctx):
        return ctx

    def close(self):
        return None


class PipelineOrchestrator:
    def __init__(self, settings):
        self.settings = settings

        if settings.pipeline.enable_detection:
            from app.components.detector import DetectorComponent

            self.detector = DetectorComponent(settings.models.yolo_weights_path, enabled=True)
        else:
            self.detector = _NoopComponent()

        if settings.pipeline.enable_depth:
            from app.components.depth import DepthComponent

            self.depth = DepthComponent(
                settings.models.depth_model_path,
                device=settings.models.device,
                depth_repo_path=settings.models.depth_repo_path,
                enabled=True,
            )
        else:
            self.depth = _NoopComponent()

        if settings.pipeline.enable_navigation:
            from app.components.navigation import NavigationComponent

            self.navigation = NavigationComponent(enabled=True)
        else:
            self.navigation = _NoopComponent()

        if settings.pipeline.enable_visualization:
            from app.components.visualization import VisualizationComponent

            self.visualization = VisualizationComponent(enabled=True)
        else:
            self.visualization = _NoopComponent()

        if settings.pipeline.enable_tts:
            from app.components.tts import TTSComponent

            self.tts = TTSComponent(enabled=True)
        else:
            self.tts = _NoopComponent()

        self.executor = PipelineExecutor(
            self.detector,
            self.depth,
            self.navigation,
            self.visualization,
            self.tts,
            execution_mode=settings.pipeline.execution_mode,
        )

    def load(self):
        if hasattr(self.detector, "load"):
            self.detector.load()
        if hasattr(self.depth, "load"):
            self.depth.load()

    def close(self):
        if hasattr(self.tts, "close"):
            self.tts.close()
