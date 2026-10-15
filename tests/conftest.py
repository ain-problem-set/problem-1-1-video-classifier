import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def write_video(path, frames, fps=25):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


@pytest.fixture
def make_video(tmp_path):
    """kind: 'noise' (random frames), 'static' (all black), 'cut' (black then white)."""
    def _make(name="v.avi", n=100, size=(320, 240), fps=25, kind="noise"):
        rng = np.random.default_rng(0)
        w, h = size
        frames = []
        for i in range(n):
            if kind == "noise":
                f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            else:
                f = np.full((h, w, 3), 255 if (kind == "cut" and i >= n // 2) else 0, np.uint8)
            frames.append(f)
        return write_video(tmp_path / name, frames, fps)
    return _make
