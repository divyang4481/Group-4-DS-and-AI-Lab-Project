from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class KaggleDatasetResolver:
    def __init__(self, slug: str, cache_root: str = "data_cache/kaggle"):
        self.slug = slug
        self.cache_root = Path(cache_root)
        self.dataset_dir = self.cache_root / slug.split("/")[-1]
        self.raw_dir = self.dataset_dir / "raw"
        self.extracted_dir = self.dataset_dir / "extracted"
        self.stamp_path = self.dataset_dir / "download_stamp.json"

    def resolve(self, auto_download_if_missing: bool = True) -> Path:
        if self._is_cached():
            return self.extracted_dir
        if not auto_download_if_missing:
            raise FileNotFoundError(
                f"Dataset cache missing at {self.extracted_dir}. Enable auto download or provide source path."
            )
        self.download_and_extract()
        return self.extracted_dir

    def _is_cached(self) -> bool:
        return self.extracted_dir.exists() and any(self.extracted_dir.iterdir())

    def _kaggle_cmd(self):
        return [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.slug,
            "-p",
            str(self.raw_dir),
            "--force",
        ]

    def download_and_extract(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("kaggle") is None:
            raise RuntimeError(
                "Kaggle CLI is not installed. Install kaggle package and configure credentials in ~/.kaggle/kaggle.json"
            )

        cmd = self._kaggle_cmd()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                "Kaggle download failed. Ensure ~/.kaggle/kaggle.json exists with valid API credentials. "
                f"Details: {stderr}"
            ) from exc

        zip_files = sorted(self.raw_dir.glob("*.zip"))
        if not zip_files:
            raise RuntimeError("Kaggle download finished but no zip file was found in raw cache directory")

        for z in zip_files:
            shutil.unpack_archive(str(z), str(self.extracted_dir))

        stamp = {
            "slug": self.slug,
            "raw_dir": str(self.raw_dir),
            "extracted_dir": str(self.extracted_dir),
            "downloaded": True,
        }
        self.stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
