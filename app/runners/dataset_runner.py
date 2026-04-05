from __future__ import annotations

from app.runners.live_runner import LiveRunner


class DatasetRunner(LiveRunner):
    """Dataset runner currently reuses live runner mechanics with non-webcam sources."""
