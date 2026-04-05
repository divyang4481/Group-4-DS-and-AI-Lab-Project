from app.metrics.aggregator import MetricsAggregator


def test_average_median_p95():
    agg = MetricsAggregator()
    vals = [{"fps_instant": x, "frame_total_latency_ms": x, "nav_command": "a", "left_risk": 1, "center_risk": 2, "right_risk": 3, "detection_count": 1} for x in [1, 2, 3, 4, 5]]
    summary = agg.summarize(vals, [], 1.0)
    assert summary["average_fps"] == 3
    assert summary["median_fps"] == 3
    assert summary["p95_fps"] == 5


def test_command_distribution():
    agg = MetricsAggregator()
    d = agg.command_distribution([{"nav_command": "left"}, {"nav_command": "left"}, {"nav_command": "right"}])
    assert d["left"] == 2
    assert d["right"] == 1


def test_slowest_and_top_risk_frames():
    agg = MetricsAggregator()
    frames = [
        {"frame_idx": 1, "frame_total_latency_ms": 10, "left_risk": 0, "center_risk": 1, "right_risk": 0},
        {"frame_idx": 2, "frame_total_latency_ms": 30, "left_risk": 4, "center_risk": 1, "right_risk": 0},
        {"frame_idx": 3, "frame_total_latency_ms": 20, "left_risk": 0, "center_risk": 2, "right_risk": 3},
    ]
    assert agg.slowest_frames(frames, n=1)[0]["frame_idx"] == 2
    assert agg.top_risk_frames(frames, n=1)[0]["frame_idx"] == 2
