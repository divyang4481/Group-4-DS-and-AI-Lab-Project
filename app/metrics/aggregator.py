from __future__ import annotations

from collections import Counter
from statistics import mean, median


class MetricsAggregator:
    @staticmethod
    def _p95(values):
        if not values:
            return None
        values = sorted(values)
        idx = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
        return values[idx]

    def summarize(self, frame_metrics, resource_samples, duration_seconds):
        if not frame_metrics:
            return {"total_frames": 0, "processed_frames": 0, "duration_seconds": duration_seconds}

        def vals(key):
            return [f[key] for f in frame_metrics if f.get(key) is not None]

        fps_vals = vals("fps_instant")
        yolo_vals = vals("yolo_latency_ms")
        depth_vals = vals("depth_latency_ms")
        fusion_vals = vals("fusion_latency_ms")
        nav_vals = vals("navigation_latency_ms")
        tts_vals = vals("tts_latency_ms")
        total_vals = vals("frame_total_latency_ms")

        cmd_counter = Counter([f.get("nav_command") for f in frame_metrics if f.get("nav_command")])
        summary = {
            "total_frames": len(frame_metrics),
            "processed_frames": len(frame_metrics),
            "duration_seconds": duration_seconds,
            "average_fps": mean(fps_vals) if fps_vals else None,
            "median_fps": median(fps_vals) if fps_vals else None,
            "p95_fps": self._p95(fps_vals),
            "average_yolo_latency_ms": mean(yolo_vals) if yolo_vals else None,
            "median_yolo_latency_ms": median(yolo_vals) if yolo_vals else None,
            "p95_yolo_latency_ms": self._p95(yolo_vals),
            "average_depth_latency_ms": mean(depth_vals) if depth_vals else None,
            "p95_depth_latency_ms": self._p95(depth_vals),
            "average_fusion_latency_ms": mean(fusion_vals) if fusion_vals else None,
            "average_navigation_latency_ms": mean(nav_vals) if nav_vals else None,
            "average_tts_latency_ms": mean(tts_vals) if tts_vals else None,
            "average_total_frame_latency_ms": mean(total_vals) if total_vals else None,
            "p95_total_frame_latency_ms": self._p95(total_vals),
            "max_total_frame_latency_ms": max(total_vals) if total_vals else None,
            "avg_detection_count": mean(vals("detection_count")) if vals("detection_count") else None,
            "avg_risk_by_zone": {
                "left": mean(vals("left_risk")) if vals("left_risk") else None,
                "center": mean(vals("center_risk")) if vals("center_risk") else None,
                "right": mean(vals("right_risk")) if vals("right_risk") else None,
            },
            "command_distribution": dict(cmd_counter),
            "peak_gpu_memory_mb": max([r.get("gpu_mem_allocated_mb") for r in resource_samples if r.get("gpu_mem_allocated_mb") is not None], default=None),
            "average_cpu": mean([r.get("cpu_percent") for r in resource_samples if r.get("cpu_percent") is not None]) if resource_samples else None,
            "average_ram": mean([r.get("ram_percent") for r in resource_samples if r.get("ram_percent") is not None]) if resource_samples else None,
        }
        return summary

    def command_distribution(self, frame_metrics):
        from collections import Counter

        return dict(Counter([f.get("nav_command") for f in frame_metrics if f.get("nav_command")]))

    def slowest_frames(self, frame_metrics, n=10):
        return sorted(frame_metrics, key=lambda f: f.get("frame_total_latency_ms") or -1, reverse=True)[:n]

    def top_risk_frames(self, frame_metrics, n=10):
        return sorted(frame_metrics, key=lambda f: max(f.get("left_risk", 0), f.get("center_risk", 0), f.get("right_risk", 0)), reverse=True)[:n]
