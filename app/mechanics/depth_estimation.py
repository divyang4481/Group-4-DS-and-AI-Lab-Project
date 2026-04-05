import os
import sys

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np
import torch


def configure_depth_anything_path(repo_path=None):
    """Configure import paths for Depth-Anything-V2 metric module in a portable way."""
    candidate = (
        repo_path
        or os.getenv("DEPTH_ANYTHING_REPO_PATH")
        or os.getenv("DEPTH_MODEL_REPO_PATH")
    )
    if not candidate:
        return

    repo_path = os.path.abspath(candidate)
    metric_path = os.path.join(repo_path, "metric_depth")
    if os.path.exists(metric_path) and metric_path not in sys.path:
        sys.path.insert(0, metric_path)
    elif os.path.exists(repo_path) and repo_path not in sys.path:
        sys.path.append(repo_path)


configure_depth_anything_path()
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception:  # pragma: no cover - optional dependency at import time
    DepthAnythingV2 = None


class DepthEstimator:
    def __init__(self, model_dir, device="cpu", repo_path=None):
        configure_depth_anything_path(repo_path)
        self.model_dir = model_dir
        self.device = device
        self.model = None

        self.model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        filename = os.path.basename(model_dir).lower()
        self.encoder = "vitl"
        if "vits" in filename:
            self.encoder = "vits"
        elif "vitb" in filename:
            self.encoder = "vitb"

        self.dataset = "vkitti" if "vkitti" in filename else "hypersim"
        self.max_depth = 80 if self.dataset == "vkitti" else 20

    def load_model(self):
        print(f"Loading Depth-Anything-V2 ({self.encoder}, {self.dataset}) from {self.model_dir}...")
        if DepthAnythingV2 is None:
            raise ImportError("depth_anything_v2 is not importable. Set DEPTH_ANYTHING_REPO_PATH or install module.")
        config = self.model_configs[self.encoder]
        try:
            self.model = DepthAnythingV2(**{**config, "max_depth": self.max_depth})
        except TypeError:
            self.model = DepthAnythingV2(**config)
            self.model.max_depth = self.max_depth

        state_dict = torch.load(self.model_dir, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        return self.model

    def predict(self, frame_rgb):
        if self.model is None:
            self.load_model()

        with torch.no_grad():
            depth_float = self.model.infer_image(frame_rgb)

        depth_clipped = np.clip(depth_float, 0, self.max_depth)
        if cv2 is not None:
            depth_u8 = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        else:
            depth_u8 = (255 * (depth_clipped - np.min(depth_clipped)) / (np.ptp(depth_clipped) + 1e-8)).astype(np.uint8)
            depth_color = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)
        return depth_float, depth_color


def estimate_distance_from_depth(depth_map, bbox):
    if depth_map is None or bbox is None or len(bbox) != 4:
        return None, None

    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    depth_roi = depth_map[y1:y2, x1:x2]
    valid = depth_roi[np.isfinite(depth_roi)]
    if valid.size == 0:
        return None, None

    depth_value = float(np.median(valid))
    if not np.isfinite(depth_value):
        return None, None

    return depth_value, depth_value
