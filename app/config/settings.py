import argparse
from pathlib import Path

from .pipeline_schema import AppSettings


def _maybe_int(v):
    return None if v is None else int(v)


def build_settings_from_cli(argv=None) -> AppSettings:
    parser = argparse.ArgumentParser(description="Multimodal indoor navigation pipeline")
    parser.add_argument("--mode", choices=["live", "dataset_eval", "benchmark"], default="live")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source-type", default=None, choices=["webcam", "video", "frame_folder"])
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--execution-mode", choices=["sequential", "threaded_parallel"], default="sequential")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--enable-tts", action="store_true")
    parser.add_argument("--disable-detection", action="store_true")
    parser.add_argument("--disable-depth", action="store_true")
    parser.add_argument("--disable-navigation", action="store_true")
    parser.add_argument("--disable-visualization", action="store_true")
    parser.add_argument("--save-annotated-video", action="store_true")
    parser.add_argument("--show-windows", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--depth-model-path", default=None)
    parser.add_argument("--depth-repo-path", default=None)
    parser.add_argument("--kaggle-auto-download", action="store_true")
    parser.add_argument("--no-kaggle-auto-download", action="store_true")

    args = parser.parse_args(argv)

    settings = AppSettings()
    settings.mode = args.mode
    settings.dataset = args.dataset
    settings.max_frames = _maybe_int(args.max_frames)
    settings.stride = max(1, int(args.stride))

    if args.output_dir:
        settings.output_dir = args.output_dir
    if args.source_type:
        settings.pipeline.source_type = args.source_type
    if args.source_path:
        settings.pipeline.source_path = args.source_path

    settings.pipeline.execution_mode = args.execution_mode
    settings.pipeline.enable_tts = bool(args.enable_tts)
    if args.disable_detection:
        settings.pipeline.enable_detection = False
    if args.disable_depth:
        settings.pipeline.enable_depth = False
    if args.disable_navigation:
        settings.pipeline.enable_navigation = False
    if args.disable_visualization:
        settings.pipeline.enable_visualization = False
    settings.pipeline.show_windows = bool(args.show_windows)
    settings.pipeline.save_annotated_video = bool(args.save_annotated_video)

    if args.device:
        settings.models.device = args.device
    if args.yolo_weights:
        settings.models.yolo_weights_path = args.yolo_weights
    if args.depth_model_path:
        settings.models.depth_model_path = args.depth_model_path
    if args.depth_repo_path:
        settings.models.depth_repo_path = args.depth_repo_path

    if args.kaggle_auto_download:
        settings.kaggle.auto_download_if_missing = True
    if args.no_kaggle_auto_download:
        settings.kaggle.auto_download_if_missing = False

    repo_root = Path(__file__).resolve().parents[2]
    if not Path(settings.models.yolo_weights_path).is_absolute():
        settings.models.yolo_weights_path = str(repo_root / settings.models.yolo_weights_path)
    if not Path(settings.models.depth_model_path).is_absolute():
        settings.models.depth_model_path = str(repo_root / settings.models.depth_model_path)
    if settings.models.depth_repo_path and not Path(settings.models.depth_repo_path).is_absolute():
        settings.models.depth_repo_path = str(repo_root / settings.models.depth_repo_path)

    return settings
