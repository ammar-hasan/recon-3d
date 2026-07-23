from pathlib import Path

import cv2
import numpy as np

from recon3d.config import PipelineConfig
from recon3d.construction_plan import validate_plan
from recon3d.multiview_visual_hull import (
    _View,
    _semantic_completion_hypothesis,
    augment_plan_with_visual_hull,
)
from recon3d.schemas import (
    ConstructionPlan,
    EvidenceSource,
    EvidencedValue,
    MultiViewObservation,
    MultiViewResult,
    OperatorCategory,
    PlanPart,
    CameraEstimate,
    ProjectionType,
    SegmentationResult,
)


def _write_box_mask(path: Path, width: int, height: int) -> tuple[int, int, int, int]:
    mask = np.zeros((128, 128), np.uint8)
    x0, y0 = (128 - width) // 2, (128 - height) // 2
    bbox = (x0, y0, x0 + width, y0 + height)
    cv2.rectangle(mask, bbox[:2], (bbox[2] - 1, bbox[3] - 1), 255, -1)
    assert cv2.imwrite(str(path), mask)
    return bbox


def _seg(path: Path, bbox: tuple[int, int, int, int]) -> SegmentationResult:
    return SegmentationResult(
        mask_path=str(path), rgba_path=str(path), original_path=str(path),
        confidence=1.0, backend="user_mask", bbox=bbox,
        coverage=((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 128**2),
        selection_source=EvidenceSource.USER_SUPPLIED,
    )


def test_calibrated_visual_hull_changes_render_geometry_but_keeps_guides(tmp_path):
    primary_path = tmp_path / "primary.png"
    side_path = tmp_path / "side.png"
    primary_bbox = _write_box_mask(primary_path, 80, 64)
    side_bbox = _write_box_mask(side_path, 40, 64)
    primary = _seg(primary_path, primary_bbox)
    observation = MultiViewObservation(
        view_id="view_001", image_path=str(side_path), mask_path=str(side_path),
        object_bbox=side_bbox, segmentation_confidence=1.0,
        scale_to_primary=EvidencedValue(
            value=1.0, source=EvidenceSource.FITTED_FROM_OBSERVATION,
            confidence=1.0),
    )
    result = MultiViewResult(
        enabled=True, observations=[observation],
        relative_camera_poses={
            "view_001": EvidencedValue(
                value=[0.0, 45.0, 0.0], unit="deg",
                source=EvidenceSource.USER_SUPPLIED, confidence=1.0),
        },
    )
    source = PlanPart(
        id="source_box", operator=OperatorCategory.PRIMITIVE,
        primitive_shape="cube", transform={"scale": [0.5, 0.4, 0.02]},
        evidence=EvidencedValue(source=EvidenceSource.FITTED_FROM_OBSERVATION,
                                confidence=0.9),
    )
    plan = ConstructionPlan(
        object_id="box", parts=[source], camera=CameraEstimate())
    cfg = PipelineConfig()
    cfg.multiview.visual_hull_grid_size = 32

    updated, result, used = augment_plan_with_visual_hull(
        plan, result, primary, cfg)

    assert used
    assert plan.parts[0].render_visible  # input plan was not mutated
    assert not updated.parts[0].render_visible
    hull = updated.parts[-1]
    assert hull.id == "multiview_visual_hull"
    assert hull.render_visible
    assert hull.profile["type"] == "mesh"
    assert len(hull.profile["vertices"]) > 8
    assert len(hull.profile["faces"]) > 8
    assert hull.evidence.source == EvidenceSource.GENERATED_HYPOTHESIS
    assert hull.evidence.confidence <= 0.5
    assert updated.camera.projection == ProjectionType.ORTHOGRAPHIC
    assert not validate_plan(updated)
    meta = updated.metadata["multiview_visual_hull"]
    assert meta["primary_observed_geometry_overwritten"] is False
    assert meta["unseen_view_risk"] == "medium"
    assert "planar semantic" in meta["unseen_view_risk_reason"]
    assert meta["generated_symmetry_hypothesis"]["source"] == "semantic_prior"
    assert meta["generated_symmetry_hypothesis"]["camera_azimuth_deg"] == 90.0
    assert meta["observed_view_reprojection_iou"]["view_000"] >= 0.8
    assert result.joint_optimization["visual_hull"]["used"] is True


def test_visual_hull_requires_user_supplied_pose(tmp_path):
    primary_path = tmp_path / "primary.png"
    side_path = tmp_path / "side.png"
    primary_bbox = _write_box_mask(primary_path, 80, 64)
    side_bbox = _write_box_mask(side_path, 40, 64)
    result = MultiViewResult(
        enabled=True,
        observations=[MultiViewObservation(
            view_id="view_001", image_path=str(side_path), mask_path=str(side_path),
            object_bbox=side_bbox,
            scale_to_primary=EvidencedValue(value=1.0, confidence=0.5),
        )],
        relative_camera_poses={
            "view_001": EvidencedValue(
                value=[0.0, 90.0, 0.0],
                source=EvidenceSource.ESTIMATED_FROM_CAMERA, confidence=0.5),
        },
    )
    plan = ConstructionPlan(object_id="box")

    unchanged, _, used = augment_plan_with_visual_hull(
        plan, result, _seg(primary_path, primary_bbox), PipelineConfig())

    assert not used
    assert unchanged is plan


def test_semantic_completion_distinguishes_axial_and_planar_priors():
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 6:26] = 1
    primary = _View("view_000", mask, (6, 8, 26, 24), 0.0, 20.0)
    secondary = _View("view_001", mask, (10, 8, 22, 24), 45.0, 20.0)
    cfg = PipelineConfig()

    axial, axial_note = _semantic_completion_hypothesis(
        ConstructionPlan(object_id="bottle"), [primary, secondary], cfg)
    assert axial.view_id == "generated_axial_invariance"
    assert np.array_equal(axial.mask, primary.mask)
    assert "axial" in axial_note

    planar, planar_note = _semantic_completion_hypothesis(
        ConstructionPlan(object_id="mug"), [primary, secondary], cfg)
    assert planar.view_id == "generated_planar_symmetry"
    assert np.array_equal(planar.mask, cv2.flip(primary.mask, 1))
    assert "mirrored" in planar_note
