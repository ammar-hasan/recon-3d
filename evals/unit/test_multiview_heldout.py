from recon3d.schemas import EvidenceSource, SegmentationResult
import numpy as np

from evals.multiview.run_multiview_eval import (
    align_surface_metrics,
    fixed_camera_from_primary,
)


def test_fixed_camera_uses_primary_bbox_only():
    seg = SegmentationResult(
        mask_path="m", rgba_path="r", original_path="i", confidence=1.0,
        backend="test", bbox=(50, 25, 150, 125), coverage=0.25,
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)
    framing = fixed_camera_from_primary(seg, (200, 200))
    assert framing["camera_type"] == "ORTHO"
    assert framing["ortho_scale"] == 2.0
    assert framing["camera_location"] == [0.0, -0.25, 2.0]


def test_fixed_perspective_camera_uses_primary_intrinsics_and_bbox_only():
    seg = SegmentationResult(
        mask_path="m", rgba_path="r", original_path="i", confidence=1.0,
        backend="test", bbox=(50, 25, 150, 125), coverage=0.25,
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)
    framing = fixed_camera_from_primary(seg, (200, 200), {
        "focal_length_px": 400.0,
        "lens_mm": 50.0,
        "sensor_width_mm": 36.0,
    })
    assert framing["camera_type"] == "PERSP"
    assert framing["ortho_scale"] is None
    assert framing["lens_mm"] == 50.0
    assert framing["sensor_width_mm"] == 36.0
    assert framing["camera_location"] == [0.0, -0.25, 4.0]


def test_surface_alignment_removes_similarity_and_axis_rotation():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(200, 3)) * np.asarray([1.0, 0.7, 0.3])
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rotation = np.asarray([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    generated = (reference @ rotation) * 2.7 + np.asarray([4.0, -2.0, 1.0])
    generated_normals = normals @ rotation
    metrics = align_surface_metrics(
        generated, generated_normals, reference, normals)
    assert metrics["normalized_surface_chamfer_distance"] < 1e-8
    assert metrics["surface_normal_consistency"] > 0.999999
