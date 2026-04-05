from __future__ import annotations

from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def timestamped_run_dir(output_root: str | Path, run_name: str | None = None) -> Path:
    output_root = Path(output_root)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or f"run_{stamp}"
    path = output_root / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path
