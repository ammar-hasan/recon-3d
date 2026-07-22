from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from evals.benchmark.generate_benchmark import (
    REQUIRED_FILES,
    _postprocess_mask,
    verify_case,
)


def _image(path: Path, value: int = 255) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 10:22] = value
    cv2.imwrite(str(path), image)


def test_postprocess_mask_writes_binary_mask_and_bbox(tmp_path):
    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[7:25, 9:23] = 240
    cv2.imwrite(str(tmp_path / "mask_raw.png"), raw)

    _postprocess_mask(tmp_path)

    mask = cv2.imread(str(tmp_path / "mask.png"), cv2.IMREAD_GRAYSCALE)
    bbox = json.loads((tmp_path / "bbox.json").read_text())
    assert set(np.unique(mask)) == {0, 255}
    assert bbox == {
        "x0": 9, "y0": 7, "x1": 23, "y1": 25,
        "source": "rendered_mask", "image_size": [32, 32],
    }
    assert not (tmp_path / "mask_raw.png").exists()


def test_verify_case_requires_declared_calibrated_views(tmp_path):
    for name in REQUIRED_FILES:
        path = tmp_path / name
        if path.suffix == ".png":
            _image(path)
        else:
            path.write_text("{}")
    (tmp_path / "build_spec.json").write_text(json.dumps({
        "view_azimuth_offsets_deg": [45.0],
    }))

    problems = verify_case(tmp_path)
    assert any("views/view_001/input.png" in p for p in problems)

    view = tmp_path / "views" / "view_001"
    view.mkdir(parents=True)
    for name in ("input.png", "mask.png", "depth.png", "normals.png"):
        _image(view / name)
    (view / "bbox.json").write_text("{}")
    (view / "camera.json").write_text(json.dumps({
        "relative_azimuth_deg": 45.0,
    }))

    assert not [p for p in verify_case(tmp_path) if "view_001" in p]
