from recon3d.schemas import EvidenceSource, SegmentationResult
import numpy as np

from evals.multiview.run_multiview_eval import (
    _base_rotation,
    align_surface_metrics,
    exact_camera_in_primary_frame,
    fixed_camera_from_primary,
)
from recon3d.schemas import CameraEstimate, ConstructionPlan, EvidencedValue


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


def test_exact_camera_is_expressed_in_normalized_primary_frame():
    primary_matrix = np.eye(4)
    primary_matrix[:3, 3] = [0.0, 0.0, 2.0]
    secondary_matrix = np.eye(4)
    secondary_matrix[:3, 3] = [1.0, 0.0, 2.0]
    primary = {
        "camera_matrix_world": primary_matrix.tolist(),
        "look_at_target": [0.0, 0.0, 0.0],
        "focal_length_px": 100.0,
    }
    secondary = {
        "camera_matrix_world": secondary_matrix.tolist(),
        "lens_mm": 50.0,
        "sensor_width_mm": 36.0,
    }
    framing = exact_camera_in_primary_frame(primary, secondary, 100.0)
    assert framing["camera_type"] == "PERSP"
    assert np.allclose(np.asarray(framing["camera_matrix_world"])[:3, 3],
                       [0.5, 0.0, 1.0])
    assert np.allclose(np.asarray(framing["camera_matrix_world"])[:3, :3],
                       np.eye(3))


def test_visual_hull_ignores_parametric_base_rotation():
    plan = ConstructionPlan(
        object_id="parametric",
        camera=CameraEstimate(object_rotation_euler_deg=EvidencedValue(
            value=[1.0, 45.0, 2.0], confidence=1.0)),
    )
    assert _base_rotation(plan) == [1.0, 45.0, 2.0]
    plan.metadata = {"multiview_visual_hull": {"used": True}}
    assert _base_rotation(plan) == [0.0, 0.0, 0.0]


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
