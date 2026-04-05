from __future__ import annotations

import csv
import json
from pathlib import Path

from .aggregator import MetricsAggregator


class MetricsExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agg = MetricsAggregator()

    def _write_csv(self, path: Path, rows):
        rows = list(rows)
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def export(self, frame_metrics, resource_samples, duration_seconds, include_summary=True):
        self._write_csv(self.output_dir / "frame_metrics.csv", frame_metrics)
        with (self.output_dir / "frame_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(frame_metrics, f, indent=2)

        self._write_csv(self.output_dir / "resource_samples.csv", resource_samples)

        cmd = self.agg.command_distribution(frame_metrics)
        cmd_rows = [{"command": k, "count": v} for k, v in cmd.items()]
        self._write_csv(self.output_dir / "command_distribution.csv", cmd_rows)

        slowest = self.agg.slowest_frames(frame_metrics)
        top_risk = self.agg.top_risk_frames(frame_metrics)
        self._write_csv(self.output_dir / "slowest_frames.csv", slowest)
        self._write_csv(self.output_dir / "top_risk_frames.csv", top_risk)

        latency_rows = []
        latency_keys = [
            "yolo_latency_ms", "depth_latency_ms", "fusion_latency_ms", "navigation_latency_ms", "tts_latency_ms", "visualization_latency_ms", "frame_total_latency_ms"
        ]
        for key in latency_keys:
            values = [f.get(key) for f in frame_metrics if f.get(key) is not None]
            latency_rows.append({
                "metric": key,
                "avg": sum(values) / len(values) if values else None,
                "p95": self.agg._p95(values) if values else None,
            })
        self._write_csv(self.output_dir / "latency_summary.csv", latency_rows)

        summary = None
        if include_summary:
            summary = self.agg.summarize(frame_metrics, resource_samples, duration_seconds)
            with (self.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        return summary
