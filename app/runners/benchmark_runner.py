from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from app.config.pipeline_schema import AppSettings
from app.metrics.exporters import MetricsExporter
from app.metrics.tracker import MetricsTracker
from app.pipeline.orchestrator import PipelineOrchestrator
from app.runners.dataset_runner import DatasetRunner
from app.sources.frame_folder_source import FrameFolderSource
from app.sources.video_source import VideoFileFrameSource


class BenchmarkRunner:
    def __init__(self, settings: AppSettings, source_path: str, output_dir: Path):
        self.settings = settings
        self.source_path = source_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _source_for(self, source_type):
        if source_type == "video":
            return VideoFileFrameSource(self.source_path)
        return FrameFolderSource(self.source_path)

    def run(self):
        configs = [
            ("detection_only", True, False, "sequential", False),
            ("depth_only", False, True, "sequential", False),
            ("seq_detection_depth", True, True, "sequential", False),
            ("parallel_detection_depth", True, True, "threaded_parallel", False),
            ("seq_full_no_tts", True, True, "sequential", False),
            ("seq_full_with_tts", True, True, "sequential", True),
            ("parallel_full_no_tts", True, True, "threaded_parallel", False),
            ("parallel_full_with_tts", True, True, "threaded_parallel", True),
        ]
        results = []
        for name, det_on, dep_on, mode, tts_on in configs:
            run_settings = copy.deepcopy(self.settings)
            run_settings.pipeline.enable_detection = det_on
            run_settings.pipeline.enable_depth = dep_on
            run_settings.pipeline.execution_mode = mode
            run_settings.pipeline.enable_tts = tts_on

            run_dir = self.output_dir / name
            exporter = MetricsExporter(run_dir)
            tracker = MetricsTracker()
            orchestrator = PipelineOrchestrator(run_settings)
            source = self._source_for(run_settings.pipeline.source_type)
            runner = DatasetRunner(
                orchestrator=orchestrator,
                source=source,
                tracker=tracker,
                exporter=exporter,
                max_frames=run_settings.max_frames,
                stride=run_settings.stride,
                show_windows=False,
            )
            runner.run()
            with (run_dir / "run_summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["benchmark_case"] = name
            results.append(summary)

        if results:
            keys = sorted({k for r in results for k in r.keys() if not isinstance(r.get(k), (dict, list))})
            with (self.output_dir / "benchmark_comparison.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows([{k: r.get(k) for k in keys} for r in results])
            with (self.output_dir / "benchmark_comparison.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return results
