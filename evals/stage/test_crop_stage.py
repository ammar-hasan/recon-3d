"""Level B stage eval: crop + coordinate integrity (EVAL.md Eval 3).

Builds a synthetic segmentation, runs recon3d.crop.make_crop, and verifies
the recorded transform against the acceptance targets:
mean round-trip error < 0.1 px, max < 0.5 px, aspect error < 0.001,
100% object coverage.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("recon3d.crop")
import cv2  # noqa: E402

from recon3d import schemas as S  # noqa: E402
from recon3d.config import PipelineConfig  # noqa: E402
from recon3d.crop import make_crop  # noqa: E402

from evals import metrics as m  # noqa: E402


def _make_segmentation(tmp_path):
    """Synthetic 640x480 image with a known filled ellipse as the object."""
    h, w = 480, 640
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (320, 240), (150, 100), 25.0, 0, 360, 255, -1)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (30, 120, 200)
    rgba[..., 3] = mask
    original = np.full((h, w, 3), 220, np.uint8)

    mask_path = tmp_path / "mask.png"
    rgba_path = tmp_path / "rgba.png"
    orig_path = tmp_path / "original.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(rgba_path), rgba)
    cv2.imwrite(str(orig_path), original)

    ys, xs = np.where(mask > 0)
    seg = S.SegmentationResult(
        mask_path=str(mask_path), rgba_path=str(rgba_path),
        original_path=str(orig_path), confidence=1.0, backend="synthetic",
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        coverage=float((mask > 0).mean()))
    return seg, mask


@pytest.fixture()
def crop_result(tmp_path):
    seg, mask = _make_segmentation(tmp_path)
    cfg = PipelineConfig()
    out_dir = tmp_path / "seg_out"
    out_dir.mkdir()
    meta, crop_rgba, crop_mask = make_crop(seg, str(out_dir), cfg)
    return meta, crop_rgba, crop_mask, mask, cfg


class TestCropCoordinateIntegrity:
    def test_round_trip_error_targets(self, crop_result):
        meta, _, _, mask, _ = crop_result
        rng = np.random.RandomState(0)
        ys, xs = np.where(mask > 0)
        idx = rng.choice(len(xs), size=300, replace=False)
        pts = np.stack([xs[idx].astype(float), ys[idx].astype(float)], axis=1)
        errs = m.round_trip_errors(pts, meta.to_crop, meta.to_original)
        assert errs.mean() < 0.1, "mean round-trip error %.4f px" % errs.mean()
        assert errs.max() < 0.5, "max round-trip error %.4f px" % errs.max()

    def test_aspect_ratio_preserved(self, crop_result):
        meta, _, _, mask, _ = crop_result
        # the transform is isotropic by construction; verify on a test circle
        cx, cy = 320.0, 240.0
        ts = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        circle = np.stack([cx + 80 * np.cos(ts), cy + 80 * np.sin(ts)], axis=1)
        cropped = np.array([meta.to_crop(x, y) for x, y in circle])
        rx = float(np.ptp(cropped[:, 0]))
        ry = float(np.ptp(cropped[:, 1]))
        assert m.aspect_ratio_error(float(rx), 1.0, float(ry), 1.0) < 0.001

    def test_object_fully_inside_crop(self, crop_result):
        meta, _, _, mask, _ = crop_result
        w, h = meta.output_size
        coverage = m.mask_coverage(mask, (meta.offset[0], meta.offset[1],
                                          meta.offset[0] + w / meta.scale,
                                          meta.offset[1] + h / meta.scale))
        assert coverage == pytest.approx(1.0)

    def test_crop_files_written(self, crop_result, tmp_path):
        meta, crop_rgba, crop_mask, _, _ = crop_result
        import json
        from pathlib import Path
        assert Path(crop_rgba).exists()
        assert Path(crop_mask).exists()
        recorded = json.loads((tmp_path / "seg_out" / "crop_metadata.json").read_text())
        assert recorded["scale"] == pytest.approx(meta.scale)
        assert list(recorded["offset"]) == pytest.approx(list(meta.offset))

    def test_recorded_transform_reproducible(self, crop_result):
        # re-validating the same metadata must give identical transforms
        meta, _, _, mask, _ = crop_result
        meta2 = S.CropMetadata.model_validate(meta.model_dump())
        pts = np.array([[100.0, 100.0], [320.0, 240.0], [639.0, 479.0]])
        e = m.round_trip_errors(pts, meta2.to_crop, meta2.to_original)
        assert e.max() < 1e-9

    def test_crop_canvas_is_square(self, crop_result):
        meta, _, _, _, cfg = crop_result
        assert meta.output_size[0] == meta.output_size[1]
