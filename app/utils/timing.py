from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(metrics: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = (time.perf_counter() - start) * 1000.0
