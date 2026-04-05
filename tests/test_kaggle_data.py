import shutil
from pathlib import Path
from unittest.mock import patch

from app.utils.kaggle_data import KaggleDatasetResolver


def test_path_resolution(tmp_path):
    r = KaggleDatasetResolver("quackphuc/egoblind-short-context-frames", cache_root=str(tmp_path))
    assert r.dataset_dir.name == "egoblind-short-context-frames"


def test_cache_hit_no_download(tmp_path):
    r = KaggleDatasetResolver("quackphuc/egoblind-short-context-frames", cache_root=str(tmp_path))
    r.extracted_dir.mkdir(parents=True)
    (r.extracted_dir / "frame.jpg").write_text("x")
    with patch.object(r, "download_and_extract") as mocked:
        resolved = r.resolve(auto_download_if_missing=True)
    assert resolved == r.extracted_dir
    mocked.assert_not_called()


def test_no_download_when_cache_exists(tmp_path):
    r = KaggleDatasetResolver("quackphuc/egoblind-short-context-frames", cache_root=str(tmp_path))
    r.extracted_dir.mkdir(parents=True)
    (r.extracted_dir / "a.txt").write_text("ok")
    assert r._is_cached() is True


def test_kaggle_command_construction(tmp_path):
    r = KaggleDatasetResolver("quackphuc/egoblind-short-context-frames", cache_root=str(tmp_path))
    cmd = r._kaggle_cmd()
    assert cmd[:4] == ["kaggle", "datasets", "download", "-d"]
    assert "quackphuc/egoblind-short-context-frames" in cmd


def test_extract_logic_with_mocks(tmp_path):
    r = KaggleDatasetResolver("quackphuc/egoblind-short-context-frames", cache_root=str(tmp_path))
    r.raw_dir.mkdir(parents=True)
    zip_path = r.raw_dir / "data.zip"
    zip_path.write_text("fake")
    with patch("shutil.which", return_value="/usr/bin/kaggle"), patch("subprocess.run") as run, patch("shutil.unpack_archive") as unpack:
        r.download_and_extract()
    run.assert_called_once()
    unpack.assert_called()
    assert r.stamp_path.exists()
