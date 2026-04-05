from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineSettings:
    source_type: str = "webcam"
    source_path: Optional[str] = None
    execution_mode: str = "sequential"
    enable_detection: bool = True
    enable_depth: bool = True
    enable_navigation: bool = True
    enable_tts: bool = False
    enable_visualization: bool = True
    save_annotated_video: bool = False
    show_windows: bool = False
    warmup_frames: int = 0
    sample_resources_every_n_frames: int = 10


@dataclass
class ModelSettings:
    yolo_weights_path: str = "model_training/object_detection/best-weights/YOLOv8n-uni.pt"
    depth_model_path: str = (
        "model_training/depth_estimation/model_weights/"
        "depth_anything_v2_metric_hypersim_vits.pth"
    )
    depth_repo_path: Optional[str] = None
    device: str = "cpu"
    confidence_threshold: float = 0.25


@dataclass
class BenchmarkSettings:
    export_csv: bool = True
    export_json: bool = True
    export_plots: bool = False
    collect_resource_metrics: bool = True
    max_frames: Optional[int] = None
    stride: int = 1
    enable_component_timing: bool = True


@dataclass
class KaggleSettings:
    dataset_slug: str = "quackphuc/egoblind-short-context-frames"
    cache_root: str = "data_cache/kaggle"
    extracted_folder_name: str = "extracted"
    auto_download_if_missing: bool = True


@dataclass
class AppSettings:
    mode: str = "live"
    dataset: Optional[str] = None
    output_dir: str = "outputs/egoblind_runs"
    max_frames: Optional[int] = None
    stride: int = 1
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    kaggle: KaggleSettings = field(default_factory=KaggleSettings)

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir)
