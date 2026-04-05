from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceSample:
    frame_idx: int
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    gpu_mem_allocated_mb: Optional[float] = None
    gpu_mem_reserved_mb: Optional[float] = None
