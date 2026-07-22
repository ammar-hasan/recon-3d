from recon3d.schemas import EvidenceSource, SegmentationResult
from evals.multiview.run_multiview_eval import fixed_camera_from_primary


def test_fixed_camera_uses_primary_bbox_only():
    seg = SegmentationResult(
        mask_path="m", rgba_path="r", original_path="i", confidence=1.0,
        backend="test", bbox=(50, 25, 150, 125), coverage=0.25,
        selection_source=EvidenceSource.DIRECTLY_OBSERVED)
    framing = fixed_camera_from_primary(seg, (200, 200))
    assert framing["ortho_scale"] == 2.0
    assert framing["camera_location"] == [0.0, -0.25, 2.0]
