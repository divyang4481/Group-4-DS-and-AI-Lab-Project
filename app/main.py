from __future__ import annotations

from pathlib import Path

from app.config.settings import build_settings_from_cli
from app.metrics.exporters import MetricsExporter
from app.metrics.tracker import MetricsTracker
from app.pipeline.orchestrator import PipelineOrchestrator
from app.runners.benchmark_runner import BenchmarkRunner
from app.runners.dataset_runner import DatasetRunner
from app.runners.live_runner import LiveRunner
from app.sources.frame_folder_source import FrameFolderSource
from app.sources.video_source import VideoFileFrameSource
from app.sources.webcam_source import WebcamFrameSource
from app.utils.kaggle_data import KaggleDatasetResolver
from app.utils.paths import timestamped_run_dir


def _resolve_dataset_source(settings):
    if settings.dataset == "egoblind" and not settings.pipeline.source_path:
        resolver = KaggleDatasetResolver(settings.kaggle.dataset_slug, settings.kaggle.cache_root)
        dataset_root = resolver.resolve(auto_download_if_missing=settings.kaggle.auto_download_if_missing)
        settings.pipeline.source_type = "frame_folder"
        settings.pipeline.source_path = str(dataset_root)


def _build_source(settings):
    if settings.pipeline.source_type == "webcam":
        index = int(settings.pipeline.source_path or 0)
        return WebcamFrameSource(index)
    if settings.pipeline.source_type == "video":
        return VideoFileFrameSource(settings.pipeline.source_path)
    return FrameFolderSource(settings.pipeline.source_path)


def run_with_settings(settings):
    if settings.mode in {"dataset_eval", "benchmark"}:
        _resolve_dataset_source(settings)

    run_dir = timestamped_run_dir(settings.output_root)

    if settings.mode == "benchmark":
        bench = BenchmarkRunner(settings, settings.pipeline.source_path, run_dir)
        bench.run()
        return run_dir

    orchestrator = PipelineOrchestrator(settings)
    source = _build_source(settings)
    tracker = MetricsTracker()
    exporter = MetricsExporter(run_dir)

    runner_cls = LiveRunner if settings.mode == "live" else DatasetRunner
    runner = runner_cls(
        orchestrator=orchestrator,
        source=source,
        tracker=tracker,
        exporter=exporter,
        max_frames=settings.max_frames,
        stride=settings.stride,
        show_windows=settings.pipeline.show_windows,
    )
    runner.run()
    return run_dir


def main(argv=None):
    settings = build_settings_from_cli(argv)
    run_dir = run_with_settings(settings)
    print(f"Run artifacts written to: {Path(run_dir)}")


if __name__ == "__main__":
    main()
