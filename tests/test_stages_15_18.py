"""Tests for stages 15-18: blender codegen, sandboxed runner, render
validation and the refinement loop.

Tests marked ``blender`` launch real Blender background runs; deselect with
``-m 'not blender'``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from recon3d import blender_codegen, refinement, runner, validation
from recon3d.config import PipelineConfig
from recon3d.schemas import (CameraEstimate, ConstructionPlan, CropMetadata,
                             EvidenceSource, EvidencedValue, MaterialSpec,
                             OperatorCategory, PlanPart, SegmentationResult)

CANVAS = 512
REF_RADIUS_PX = 190


def _tyre_profile():
    pts = []
    for i in range(12):
        a = 2.0 * math.pi * i / 12
        pts.append([round(0.35 + 0.12 * math.cos(a), 5),
                    round(0.12 * math.sin(a), 5)])
    return pts


def make_wheel_plan(global_scale=None) -> ConstructionPlan:
    rubber = MaterialSpec(material_class="rubber",
                          base_color=(0.03, 0.03, 0.03), roughness=0.8)
    metal = MaterialSpec(material_class="metal",
                         base_color=(0.5, 0.52, 0.55), roughness=0.35,
                         metallic=0.9)
    parts = [
        PlanPart(id="tyre", operator=OperatorCategory.REVOLVE,
                 axis={"origin": [0, 0, 0], "direction": [0, 0, 1]},
                 profile={"type": "polyline", "points": _tyre_profile(),
                          "closed": True},
                 material=rubber),
        PlanPart(id="rim", operator=OperatorCategory.REVOLVE,
                 axis={"origin": [0, 0, 0], "direction": [0, 0, 1]},
                 profile={"type": "polyline",
                          "points": [[0.20, -0.06], [0.26, -0.06],
                                     [0.26, 0.06], [0.20, 0.06]],
                          "closed": True},
                 material=metal),
        PlanPart(id="spoke", operator=OperatorCategory.EXTRUDE,
                 profile={"type": "polygon",
                          "points": [[-0.035, 0.03], [0.035, 0.03],
                                     [0.035, 0.22], [-0.035, 0.22]],
                          "closed": True},
                 depth=0.08, material=metal),
        PlanPart(id="spoke_array", operator=OperatorCategory.RADIAL_ARRAY,
                 source_part="spoke", count=5, angle_degrees=360.0,
                 axis={"origin": [0, 0, 0], "direction": [0, 0, 1]}),
        PlanPart(id="hub", operator=OperatorCategory.PRIMITIVE,
                 primitive_shape="cylinder",
                 transform={"location": [0, 0, 0], "rotation_deg": [0, 0, 0],
                            "scale": [0.12, 0.12, 0.14]},
                 material=metal),
        PlanPart(id="hub_hole", operator=OperatorCategory.BOOLEAN,
                 primitive_shape="cylinder", boolean_target="hub",
                 boolean_operation="difference",
                 transform={"location": [0, 0, 0],
                            "scale": [0.05, 0.05, 0.30]}),
    ]
    metadata = {}
    if global_scale:
        metadata["global_scale"] = list(global_scale)
    return ConstructionPlan(object_id="test_wheel", parts=parts,
                            camera=CameraEstimate(), metadata=metadata)


def make_reference(project: Path):
    """Synthetic reference: a filled disc (the wheel silhouette) drawn with
    cv2, written as segmentation mask + rgba, with identity crop metadata."""
    seg_dir = project / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    c = CANVAS // 2
    mask = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.circle(mask, (c, c), REF_RADIUS_PX, 255, -1)
    rgba = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    rgba[mask > 0] = (60, 60, 60, 255)
    mask_path = seg_dir / "object_mask.png"
    rgba_path = seg_dir / "object_rgba.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(rgba_path), rgba)
    seg = SegmentationResult(
        mask_path=str(mask_path), rgba_path=str(rgba_path),
        original_path=str(mask_path), confidence=1.0, backend="synthetic",
        bbox=(c - REF_RADIUS_PX, c - REF_RADIUS_PX,
              c + REF_RADIUS_PX, c + REF_RADIUS_PX),
        coverage=float((mask > 0).mean()))
    crop_meta = CropMetadata(
        source_image_size=(CANVAS, CANVAS),
        source_bbox=(c - REF_RADIUS_PX, c - REF_RADIUS_PX,
                     c + REF_RADIUS_PX, c + REF_RADIUS_PX),
        padding=0, output_size=(CANVAS, CANVAS), scale=1.0, offset=(0.0, 0.0))
    return seg, crop_meta


def make_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.blender.timeout_seconds = 300
    cfg.refinement.max_iterations = 2
    cfg.refinement.max_renders = 6
    return cfg


# ---------------------------------------------------------------------------
# pure tests (no Blender launch)
# ---------------------------------------------------------------------------

def test_codegen_writes_deterministic_script(tmp_path):
    plan = make_wheel_plan()
    p1 = blender_codegen.generate_blender_script(plan, str(tmp_path), make_cfg())
    assert p1.endswith("build_model.py")
    text1 = Path(p1).read_text()
    p2 = blender_codegen.generate_blender_script(plan, str(tmp_path), make_cfg())
    assert Path(p2).read_text() == text1, "codegen must be deterministic"
    # generated script must itself pass the safety scan
    assert runner.scan_script_safety(text1, str(tmp_path)) == []
    # plan embedded
    assert "test_wheel" in text1


def test_codegen_orders_build_dependencies_before_consumers():
    target = PlanPart(
        id="body", operator=OperatorCategory.EXTRUDE,
        profile={"type": "polyline",
                 "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                 "closed": True}, depth=0.1)
    cutter = PlanPart(
        id="hole", operator=OperatorCategory.BOOLEAN,
        boolean_target="body", boolean_operation="difference",
        profile={"type": "polyline",
                 "points": [[0.2, 0.2], [0.3, 0.2], [0.25, 0.3]],
                 "closed": True}, depth=0.2)
    plan = ConstructionPlan(object_id="ordered", parts=[cutter, target])
    data = blender_codegen._ordered_plan_data(plan)
    assert [p["id"] for p in data["parts"]] == ["body", "hole"]


def test_pose_hypothesis_is_evidence_tracked():
    plan = ConstructionPlan(object_id="pose", parts=[])
    record = refinement._set_object_rotation(plan, 0, 30.0, make_cfg())
    assert record["object_rotation_x"]["new"] == 30.0
    assert plan.camera.object_rotation_euler_deg.value == [30.0, 0.0, 0.0]
    assert (plan.camera.object_rotation_euler_deg.source
            == refinement.EvidenceSource.ESTIMATED_FROM_CAMERA)


def test_refinement_never_replaces_user_supplied_object_pose():
    plan = ConstructionPlan(
        object_id="calibrated",
        parts=[PlanPart(id="body", operator=OperatorCategory.EXTRUDE,
                        profile={"type": "polygon", "points": [],
                                 "closed": True}, depth=0.1)],
        camera=CameraEstimate(object_rotation_euler_deg=EvidencedValue(
            value=[0.0, 43.0, 0.0], unit="deg",
            source=EvidenceSource.USER_SUPPLIED, confidence=1.0)),
    )
    assert not refinement._pose_search_eligible(plan, 0.2)


def test_refinement_prioritizes_visible_offset_and_skips_inapplicable_revolve():
    plan = ConstructionPlan(
        object_id="offset",
        parts=[PlanPart(id="body", operator=OperatorCategory.EXTRUDE,
                        profile={"type": "polygon", "points": [],
                                 "closed": True}, depth=0.1)],
    )
    ranked = [name for name, _ in refinement._rank_candidates({
        "width_ratio": 1.04,
        "height_ratio": 1.0,
        "dx_px": 0.0,
        "dy_px": -15.0,
        "canvas_w": 1024.0,
    }, plan)]
    assert ranked[0] == "camera_offset_y"
    assert "revolve_radius_scale" not in ranked


def test_refinement_offset_preserves_auto_framing():
    plan = ConstructionPlan(object_id="offset", parts=[])
    record = refinement._apply_candidate(
        plan, "camera_offset_y",
        {"dx_px": 0.0, "dy_px": -16.0, "ref_width_px": 800.0,
         "width_ratio": 1.0, "height_ratio": 1.0},
        1200.0, make_cfg())
    assert plan.camera is None
    assert plan.metadata["validation_camera_offset"] == [0.0, 0.02]
    assert record["camera_offset_y"]["new"] == 0.02


def test_refinement_scale_step_is_damped():
    plan = ConstructionPlan(object_id="scale", parts=[])
    record = refinement._apply_candidate(
        plan, "global_scale_x",
        {"width_ratio": 1.25, "height_ratio": 1.0},
        1200.0, make_cfg())
    assert plan.metadata["global_scale"] == [1.02, 1.0, 1.0]
    assert record["global_scale_x"]["new"] == 1.02


def test_safety_scan_rejects_dangerous_script(tmp_path):
    bad = "import subprocess\nsubprocess.run(['ls'])\n"
    violations = runner.scan_script_safety(bad, str(tmp_path))
    assert any("subprocess" in v for v in violations)
    bad_path = tmp_path / "evil.py"
    bad_path.write_text(bad)
    with pytest.raises(runner.ScriptSafetyError):
        runner.run_blender(str(bad_path), str(tmp_path), make_cfg())
    # writes outside the project dir are flagged
    bad2 = "open('/etc/passwd', 'w')\n"
    assert runner.scan_script_safety(bad2, str(tmp_path))


# ---------------------------------------------------------------------------
# Blender-backed tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_project(tmp_path_factory):
    """Build the wheel scene once; shared by the build/validation tests."""
    project = tmp_path_factory.mktemp("wheel_project")
    plan = make_wheel_plan()
    seg, crop_meta = make_reference(project)
    cfg = make_cfg()
    script = blender_codegen.generate_blender_script(
        plan, str(project / "blender"), cfg)
    manifest = runner.run_blender(script, str(project), cfg)
    return {
        "project": project, "plan": plan, "seg": seg,
        "crop_meta": crop_meta, "cfg": cfg, "manifest": manifest,
    }


@pytest.mark.blender
def test_run_blender_builds_scene(built_project):
    m = built_project["manifest"]
    assert m.success, "build failed: %s" % (m.errors[:1])
    assert Path(m.blend_path).exists()
    assert m.glb_path and Path(m.glb_path).exists()
    assert m.blender_version.startswith("5.2")

    names = {o.name for o in m.objects}
    # root + parts + array duplicates + cutter present, parts named by id
    assert "test_wheel" in names
    for part in ("tyre", "rim", "spoke", "hub", "hub_hole"):
        assert part in names, "missing part object %s" % part
    assert any(n.startswith("spoke_array") for n in names)

    by_name = {o.name: o for o in m.objects}
    # hierarchy: parts parented to the root empty
    assert by_name["tyre"].parent == "test_wheel"
    assert by_name["rim"].parent == "test_wheel"
    # array duplicates parented to the array anchor empty
    dup = by_name.get("spoke_array_00")
    assert dup is not None and dup.parent == "spoke_array"
    # collections per top-level semantic group
    assert "tyre" in m.collections
    # materials assigned and shared (no duplicates: rubber once, metal once)
    mats = sorted({mat for o in m.objects for mat in o.materials})
    assert mats == ["mat_metal_01", "mat_rubber_00"]
    assert by_name["tyre"].materials == ["mat_rubber_00"]
    assert by_name["rim"].materials == ["mat_metal_01"]
    # non-destructive: bevel + boolean modifiers retained
    assert "BEVEL" in by_name["spoke"].modifiers
    assert "BOOLEAN" in by_name["hub"].modifiers
    # mesh stats recorded
    assert by_name["tyre"].vertex_count > 0
    assert by_name["tyre"].face_count > 0
    assert by_name["tyre"].pivot is not None
    assert by_name["tyre"].location is not None
    assert by_name["tyre"].rotation_euler_deg is not None
    assert by_name["tyre"].scale is not None


@pytest.mark.blender
def test_validate_reconstruction(built_project):
    bp = built_project
    result = validation.validate_reconstruction(
        bp["manifest"], bp["plan"], bp["seg"], bp["crop_meta"],
        str(bp["project"]), bp["cfg"])
    m = result.metrics
    assert m.silhouette_iou is not None
    assert m.silhouette_iou > 0.70, "iou too low: %s" % m.silhouette_iou
    assert m.clay_silhouette_iou is not None
    assert m.contour_chamfer_distance is not None
    assert m.perceptual_similarity is not None
    assert m.color_region_agreement is not None
    # artifacts
    for p in (result.overlay_path, result.silhouette_comparison_path,
              result.depth_comparison_path, result.turntable_path):
        assert p and Path(p).exists(), "missing artifact %s" % p
    assert Path(result.turntable_path).stat().st_size > 0
    metrics_json = bp["project"] / "validation" / "metrics.json"
    assert metrics_json.exists()
    data = json.loads(metrics_json.read_text())
    assert data["silhouette_iou"] == pytest.approx(m.silhouette_iou)


@pytest.mark.blender
def test_refine_improves_and_terminates(tmp_path):
    # initial plan deliberately 15% too small -> refinement should scale up
    project = tmp_path / "refine_project"
    project.mkdir()
    plan = make_wheel_plan(global_scale=(0.85, 0.85, 0.85))
    seg, crop_meta = make_reference(project)
    cfg = make_cfg()

    best_plan, manifest, val_result, log = refinement.refine(
        plan, seg, crop_meta, str(project), cfg)

    assert manifest.success
    assert log.iterations >= 1
    assert len(log.actions) >= 1
    initial = log.initial_metrics["silhouette_iou"]
    final = log.final_metrics["silhouette_iou"]
    assert final is not None and initial is not None
    assert final >= initial - 1e-9, "refinement returned a worse plan"
    assert final > initial, "expected refinement to improve the shrunken plan"
    assert val_result.metrics.silhouette_iou == pytest.approx(final)
    # audit trail: params + metric change recorded per action
    for action in log.actions:
        assert action.observed_problem
        assert action.modified_parameters
        assert "silhouette_iou" in action.metric_change
    assert (project / "validation" / "refinement_log.json").exists()
    # hard caps respected even though target IoU is higher than reached
    assert log.iterations <= cfg.refinement.max_iterations
