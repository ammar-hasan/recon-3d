"""Level A unit tests for recon3d schema behaviour used by the evals.

These import recon3d directly (it is part of the repo) but remain defensive:
if a schema changes shape, the tests fail loudly rather than silently.
"""
from __future__ import annotations

import numpy as np
import pytest

schemas = pytest.importorskip("recon3d.schemas")
from recon3d import schemas as S  # noqa: E402

from evals import metrics as m  # noqa: E402


def _crop_meta():
    """CropMetadata matching the GOAL.md worked example."""
    return S.CropMetadata(
        source_image_size=(1920, 1080),
        source_bbox=(410, 160, 1480, 1010),
        padding=72,
        output_size=(1024, 1024),
        scale=0.71,
        offset=(-214.0, -48.0),
    )


class TestCropMetadataTransform:
    def test_round_trip_random_points(self):
        meta = _crop_meta()
        rng = np.random.RandomState(42)
        pts = rng.uniform(low=[410, 160], high=[1480, 1010], size=(200, 2))
        errs = m.round_trip_errors(pts, meta.to_crop, meta.to_original)
        assert errs.mean() < 0.1
        assert errs.max() < 0.5

    def test_norm_round_trip(self):
        meta = _crop_meta()
        rng = np.random.RandomState(7)
        pts = rng.uniform(low=[410, 160], high=[1480, 1010], size=(100, 2))
        for x, y in pts:
            u, v = meta.to_crop_norm(x, y)
            x2, y2 = meta.norm_to_original(u, v)
            assert abs(x2 - x) < 1e-6
            assert abs(y2 - y) < 1e-6

    def test_known_point(self):
        meta = _crop_meta()
        # point at the offset maps to crop origin
        assert meta.to_crop(-214.0, -48.0) == pytest.approx((0.0, 0.0))
        # and back
        assert meta.to_original(0.0, 0.0) == pytest.approx((-214.0, -48.0))

    def test_crop_is_isotropic(self):
        # a circle in the source must stay a circle in the crop: the transform
        # uses one scalar scale for both axes by construction
        meta = _crop_meta()
        cx, cy = 900.0, 500.0
        ts = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        circle = np.stack([cx + 100 * np.cos(ts), cy + 100 * np.sin(ts)], axis=1)
        cropped = np.array([meta.to_crop(x, y) for x, y in circle])
        rx = cropped[:, 0].max() - cropped[:, 0].min()
        ry = cropped[:, 1].max() - cropped[:, 1].min()
        assert m.aspect_ratio_error(rx, 1.0, ry, 1.0) < 0.001

    def test_serialisation_round_trip(self, tmp_path):
        meta = _crop_meta()
        p = S.SchemaIO.save_json(meta, tmp_path / "crop.json")
        loaded = S.SchemaIO.load_json(S.CropMetadata, p)
        assert loaded == meta


class TestEvidencedValue:
    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            S.EvidencedValue(value=1.0, confidence=1.5)
        with pytest.raises(Exception):
            S.EvidencedValue(value=1.0, confidence=-0.1)

    def test_sources(self):
        v = S.EvidencedValue(value=0.5, source=S.EvidenceSource.GENERATED_HYPOTHESIS,
                             confidence=0.3)
        assert v.source.value == "generated_hypothesis"


class TestSchemaIoYaml:
    def test_yaml_round_trip(self, tmp_path):
        prim = S.GeometricPrimitive(
            id="p1", type=S.PrimitiveType.CIRCLE,
            params={"center": [0.5, 0.5], "radius": 0.2},
            source_path="sp1", source_layer=S.TraceLayerName.SILHOUETTE)
        p = S.SchemaIO.save_yaml(prim, tmp_path / "prim.yaml")
        loaded = S.SchemaIO.load_yaml(S.GeometricPrimitive, p)
        assert loaded.id == "p1"
        assert loaded.type == S.PrimitiveType.CIRCLE
