from PIL import Image
import numpy as np

from app.sources.frame_folder_source import FrameFolderSource
from app.sources.video_source import VideoFileFrameSource


def test_video_source_init_missing_file(tmp_path):
    src = VideoFileFrameSource(str(tmp_path / "missing.mp4"))
    try:
        src.open()
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_frame_folder_ordering(tmp_path):
    for name, value in [("2.png", 2), ("10.png", 10), ("1.png", 1)]:
        img = np.full((10, 10, 3), value, dtype=np.uint8)
        Image.fromarray(img).save(tmp_path / name)

    src = FrameFolderSource(str(tmp_path))
    src.open()
    observed = []
    while True:
        ok, frame = src.read()
        if not ok:
            break
        observed.append(int(frame[0, 0, 0]))
    src.close()
    assert observed == [1, 10, 2]


def test_frame_folder_empty_handling(tmp_path):
    src = FrameFolderSource(str(tmp_path))
    src.open()
    ok, frame = src.read()
    src.close()
    assert ok is False
    assert frame is None
