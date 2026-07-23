"""Tests for stages 11-14 + material estimation.

Builds synthetic SketchGraphs in code; no other pipeline stage required.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from recon3d import camera as camera_mod
from recon3d import construction_plan as plan_mod
from recon3d import depth as depth_mod
from recon3d import materials as materials_mod
from recon3d import operators as operators_mod
from recon3d.config import PipelineConfig
from recon3d.schemas import (
    CameraEstimate,
    ConstraintType,
    CropMetadata,
    DepthEvidence,
    EvidenceSource,
    GeometricConstraint,
    GeometricPrimitive,
    InputSpec,
    MaterialSpec,
    OperatorCategory,
    PlanPart,
    PrimitiveType,
    ProjectionType,
    SegmentationResult,
    SemanticPart,
    SketchGraph,
    TraceLayerName,
)

CFG = PipelineConfig()
TILT_DEG = 30.0
TILT_RATIO = math.cos(math.radians(TILT_DEG))  # ~0.8660


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _ellipse(pid, cx, cy, rx, ry, rot=0.0):
    return GeometricPrimitive(
        id=pid,
        type=PrimitiveType.ELLIPSE,
        params={"center": [cx, cy], "radii": [rx, ry], "rotation_degrees": rot},
        source_path="path_" + pid,
        source_layer=TraceLayerName.SILHOUETTE,
        confidence=0.9,
    )


def _circle(pid, cx, cy, r):
    return GeometricPrimitive(
        id=pid,
        type=PrimitiveType.CIRCLE,
        params={"center": [cx, cy], "radius": r},
        source_path="path_" + pid,
        source_layer=TraceLayerName.SILHOUETTE,
        confidence=0.9,
    )


def _line(pid, x0, y0, x1, y1):
    return GeometricPrimitive(
        id=pid,
        type=PrimitiveType.LINE,
        params={"points": [[x0, y0], [x1, y1]]},
        source_path="path_" + pid,
        source_layer=TraceLayerName.STRUCTURAL_EDGES,
        confidence=0.8,
    )


def _rect_region(pid, x0, y0, x1, y1):
    return GeometricPrimitive(
        id=pid,
        type=PrimitiveType.RECTANGLE,
        params={"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]},
        source_path="path_" + pid,
        source_layer=TraceLayerName.SILHOUETTE,
        confidence=0.9,
    )


def _part(pid, cls, prim_ids):
    return SemanticPart(id=pid, part_class=cls, primitive_ids=list(prim_ids), confidence=0.8)


def make_wheel_graph():
    """Wheel: concentric ellipses (projected circles, tilt = TILT_DEG) + 5 spokes."""
    prims = [
        _ellipse("tyre_outer", 0.5, 0.5, 0.30, 0.30 * TILT_RATIO),
        _ellipse("rim_outer", 0.5, 0.5, 0.20, 0.20 * TILT_RATIO),
        _ellipse("hub_prim", 0.5, 0.5, 0.06, 0.06 * TILT_RATIO),
    ]
    spoke_ids = []
    for k in range(5):
        ang = math.radians(k * 72.0)
        sid = "spoke_%d" % (k + 1)
        spoke_ids.append(sid)
        prims.append(
            _line(sid, 0.5 + 0.07 * math.cos(ang), 0.5 + 0.07 * math.sin(ang),
                  0.5 + 0.19 * math.cos(ang), 0.5 + 0.19 * math.sin(ang))
        )
    constraints = [
        GeometricConstraint(
            type=ConstraintType.CONCENTRIC,
            entities=["tyre_outer", "rim_outer", "hub_prim"],
            confidence=0.95,
        ),
        GeometricConstraint(
            type=ConstraintType.ROTATIONAL_REPETITION,
            entities=spoke_ids,
            params={"count": 5, "center": [0.5, 0.5], "prototype": "spoke_1"},
            confidence=0.9,
        ),
    ]
    parts = [
        _part("tyre", "tyre", ["tyre_outer"]),
        _part("rim", "rim", ["rim_outer"]),
        _part("hub", "hub", ["hub_prim"]),
        _part("spokes", "spokes", spoke_ids),
    ]
    return SketchGraph(primitives=prims, constraints=constraints, parts=parts)


def make_bracket_graph():
    """Flat bracket plate with a circular hole cutter."""
    prims = [
        _rect_region("bracket_region", 0.2, 0.3, 0.8, 0.7),
        _circle("hole_prim", 0.5, 0.5, 0.05),
    ]
    constraints = [
        GeometricConstraint(
            type=ConstraintType.CONTAINMENT,
            entities=["hole_prim", "bracket_region"],
            confidence=0.9,
        )
    ]
    parts = [
        _part("bracket", "bracket", ["bracket_region"]),
        _part("hole", "hole", ["hole_prim"]),
    ]
    return SketchGraph(primitives=prims, constraints=constraints, parts=parts)


def make_seg():
    return SegmentationResult(
        mask_path="unused_mask.png",
        rgba_path="unused_rgba.png",
        original_path="unused_orig.png",
        confidence=0.9,
        backend="user_mask",
        bbox=(102, 102, 922, 922),  # 820 px wide
        coverage=0.5,
    )


def make_crop_meta():
    return CropMetadata(
        source_image_size=(1024, 1024),
        source_bbox=(102, 102, 922, 922),
        padding=48,
        output_size=(1024, 1024),
        scale=1.0,
        offset=(0.0, 0.0),
    )


def make_spec(**kw):
    kw.setdefault("image_paths", ["unused.png"])
    return InputSpec(**kw)


NO_DEPTH = DepthEvidence(backend="none", confidence=0.0)


# ---------------------------------------------------------------------------
# (a) wheel: operators + camera + scale
# ---------------------------------------------------------------------------

def test_wheel_operators():
    graph = operators_mod.classify_operators(make_wheel_graph(), NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in graph.parts}
    assert selected["tyre"] == OperatorCategory.REVOLVE.value
    assert selected["rim"] == OperatorCategory.REVOLVE.value
    assert selected["hub"] == OperatorCategory.REVOLVE.value
    assert selected["spokes"] == OperatorCategory.RADIAL_ARRAY.value
    for p in graph.parts:
        assert len(p.construction_candidates) >= 2  # multiple candidates kept
        confs = [c.confidence for c in p.construction_candidates]
        assert confs == sorted(confs, reverse=True)


def test_radial_symmetry_on_rings_does_not_beat_revolve():
    """A radial_symmetry constraint between concentric rings (no repetition
    count) must not turn ring parts into arrays of themselves."""
    graph = make_wheel_graph()
    graph.constraints.append(
        GeometricConstraint(
            type=ConstraintType.RADIAL_SYMMETRY,
            entities=["tyre_outer", "rim_outer", "hub_prim"],
            params={"center": [0.5, 0.5]},
            confidence=0.9,
        )
    )
    out = operators_mod.classify_operators(graph, NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in out.parts}
    assert selected["tyre"] == OperatorCategory.REVOLVE.value
    assert selected["rim"] == OperatorCategory.REVOLVE.value
    assert selected["hub"] == OperatorCategory.REVOLVE.value
    # spokes still get the array (count=5 repetition constraint)
    assert selected["spokes"] == OperatorCategory.RADIAL_ARRAY.value


def test_gear_body_extrudes_and_duplicate_tooth_trace_is_surface_detail():
    body = _rect_region("gear_outline", 0.2, 0.25, 0.8, 0.75)
    tooth = _rect_region("tooth_trace", 0.75, 0.44, 0.84, 0.56)
    parts = [
        _part("gear", "gear_body", [body.id]),
        SemanticPart(id="tooth", part_class="tooth",
                     primitive_ids=[tooth.id], parent_id="gear"),
    ]
    graph = operators_mod.classify_operators(
        SketchGraph(primitives=[body, tooth], parts=parts), NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in graph.parts}
    assert selected["gear"] == OperatorCategory.EXTRUDE.value
    assert selected["tooth"] == OperatorCategory.DISPLACEMENT.value


def test_crate_guided_roles_do_not_duplicate_root_silhouette():
    body = _rect_region("crate_outline", 0.2, 0.2, 0.8, 0.8)
    post = _rect_region("post_trace", 0.2, 0.2, 0.3, 0.8)
    slat = _rect_region("slat_trace", 0.3, 0.3, 0.8, 0.4)
    parts = [
        _part("crate", "bottom_panel", [body.id]),
        SemanticPart(id="post", part_class="corner_post",
                     primitive_ids=[post.id], parent_id="crate"),
        SemanticPart(id="slat", part_class="side_slat",
                     primitive_ids=[slat.id], parent_id="crate"),
    ]
    graph = operators_mod.classify_operators(
        SketchGraph(primitives=[body, post, slat], parts=parts), NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in graph.parts}
    assert selected["crate"] == OperatorCategory.EXTRUDE.value
    assert selected["post"] == OperatorCategory.DISPLACEMENT.value
    assert selected["slat"] == OperatorCategory.DISPLACEMENT.value


def test_sweep_plan_uses_medial_axis_as_path_and_observed_width():
    outline = GeometricPrimitive(
        id="pipe_outline",
        type=PrimitiveType.CLOSED_REGION,
        params={"points": [
            [0.20, 0.65], [0.30, 0.75], [0.55, 0.55], [0.70, 0.30],
            [0.62, 0.22], [0.47, 0.47], [0.25, 0.58],
        ]},
        fallback_points=[
            (0.20, 0.65), (0.30, 0.75), (0.55, 0.55), (0.70, 0.30),
            (0.62, 0.22), (0.47, 0.47), (0.25, 0.58),
        ],
        source_path="pipe_silhouette",
        source_layer=TraceLayerName.SILHOUETTE,
        confidence=0.9,
    )
    graph = SketchGraph(
        primitives=[outline], parts=[_part("pipe", "pipe", [outline.id])])
    graph = operators_mod.classify_operators(graph, NO_DEPTH, CFG)
    plan = plan_mod.build_plan(
        graph, CameraEstimate(), NO_DEPTH,
        make_spec(target_label="pipe", output_dir="/nonexistent-dir"), CFG)
    sweep = plan.parts[0]
    assert sweep.operator == OperatorCategory.SWEEP
    assert sweep.profile["type"] == "polyline"
    assert sweep.profile["closed"] is False
    assert len(sweep.profile["points"]) >= 3
    assert 0.006 <= sweep.depth <= 0.15


def test_boolean_target_ignores_non_geometric_details_part():
    body = _rect_region("body", 0.2, 0.2, 0.8, 0.8)
    hole = _circle("hole", 0.5, 0.5, 0.08)
    detail = _rect_region("detail", 0.1, 0.1, 0.9, 0.9)
    parts = [
        _part("body_part", "plate", [body.id]),
        SemanticPart(id="hole_part", part_class="center_bore",
                     primitive_ids=[hole.id], parent_id="body_part"),
        _part("detail_part", "details", [detail.id]),
    ]
    graph = operators_mod.classify_operators(
        SketchGraph(primitives=[body, hole, detail], parts=parts), NO_DEPTH, CFG)
    plan = plan_mod.build_plan(
        graph, CameraEstimate(), NO_DEPTH,
        make_spec(output_dir="/nonexistent-dir"), CFG)
    bore = next(p for p in plan.parts if p.id == "hole_part")
    assert bore.operator == OperatorCategory.BOOLEAN
    assert bore.boolean_target == "body_part"


def test_repetition_count_two_is_not_an_array():
    """rotational repetition needs count >= 3 to justify radial_array."""
    prims = [
        _ellipse("hub_prim", 0.5, 0.5, 0.1, 0.1 * TILT_RATIO),
        _rect_region("wing_a", 0.7, 0.45, 0.8, 0.55),
        _rect_region("wing_b", 0.2, 0.45, 0.3, 0.55),
    ]
    constraints = [
        GeometricConstraint(
            type=ConstraintType.ROTATIONAL_REPETITION,
            entities=["wing_a", "wing_b"],
            params={"count": 2, "center": [0.5, 0.5], "prototype": "wing_a"},
            confidence=0.9,
        ),
    ]
    parts = [
        _part("hub", "hub", ["hub_prim"]),
        _part("wings", "wings", ["wing_a", "wing_b"]),
    ]
    graph = SketchGraph(primitives=prims, constraints=constraints, parts=parts)
    out = operators_mod.classify_operators(graph, NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in out.parts}
    assert selected["wings"] != OperatorCategory.RADIAL_ARRAY.value


def test_camera_tilt_from_ellipse_ratio():
    est = camera_mod.estimate_camera(
        make_wheel_graph(), make_seg(), make_crop_meta(), make_spec(), CFG
    )
    rot = est.object_rotation_euler_deg
    assert rot.source == EvidenceSource.ESTIMATED_FROM_CAMERA
    assert rot.value is not None
    assert abs(rot.value[0] - TILT_DEG) <= 3.0
    assert est.projection == ProjectionType.PERSPECTIVE
    assert est.notes  # assumptions recorded


def test_camera_ignores_incidental_ellipses_for_labelled_bottle():
    est = camera_mod.estimate_camera(
        make_wheel_graph(), make_seg(), make_crop_meta(),
        make_spec(target_label="bottle"), CFG)
    assert est.object_rotation_euler_deg.value == [0.0, 0.0, 0.0]
    assert est.object_rotation_euler_deg.source == EvidenceSource.SEMANTIC_PRIOR


def test_camera_does_not_rotate_observed_gear_silhouette_twice():
    est = camera_mod.estimate_camera(
        make_wheel_graph(), make_seg(), make_crop_meta(),
        make_spec(target_label="gear"), CFG)
    assert est.object_rotation_euler_deg.value == [0.0, 0.0, 0.0]
    assert est.object_rotation_euler_deg.source == EvidenceSource.SEMANTIC_PRIOR


def test_scale_unknown_without_known_dimension():
    est = camera_mod.estimate_camera(
        make_wheel_graph(), make_seg(), make_crop_meta(), make_spec(), CFG
    )
    assert est.scale.source == EvidenceSource.UNKNOWN
    assert est.scale.value is None
    assert est.scale.confidence == 0.0


def test_scale_user_supplied_with_known_dimension():
    spec = make_spec(known_dimension=0.6, known_dimension_axis="diameter")
    est = camera_mod.estimate_camera(
        make_wheel_graph(), make_seg(), make_crop_meta(), spec, CFG
    )
    assert est.scale.source == EvidenceSource.USER_SUPPLIED
    expected = 0.6 / (820.0 / 1024.0)
    assert est.scale.value == pytest.approx(expected, rel=1e-3)
    assert est.scale.confidence > 0.5


def test_wheel_plan_valid():
    graph = operators_mod.classify_operators(make_wheel_graph(), NO_DEPTH, CFG)
    cam = camera_mod.estimate_camera(
        graph, make_seg(), make_crop_meta(), make_spec(output_dir="/nonexistent-dir"), CFG
    )
    plan = plan_mod.build_plan(graph, cam, NO_DEPTH, make_spec(output_dir="/nonexistent-dir"), CFG)
    assert plan.units == "normalized"
    assert plan.uncertainty["physical_scale"] == "unknown"
    ops = {p.id: p.operator for p in plan.parts}
    assert ops["tyre"] == OperatorCategory.REVOLVE
    assert ops["spokes"] == OperatorCategory.RADIAL_ARRAY
    errors = plan_mod.validate_plan(plan)
    assert errors == [], errors


def test_self_contained_array_gets_prototype_part():
    """When the array's prototype curve belongs to the array part itself,
    build_plan must split off a distinct prototype part (never self-reference)."""
    graph = operators_mod.classify_operators(make_wheel_graph(), NO_DEPTH, CFG)
    cam = camera_mod.estimate_camera(
        graph, make_seg(), make_crop_meta(), make_spec(output_dir="/nonexistent-dir"), CFG
    )
    plan = plan_mod.build_plan(graph, cam, NO_DEPTH, make_spec(output_dir="/nonexistent-dir"), CFG)
    by_id = {p.id: p for p in plan.parts}
    arr = by_id["spokes"]
    assert arr.source_part is not None and arr.source_part != "spokes"
    proto = by_id[arr.source_part]
    assert proto.operator == OperatorCategory.EXTRUDE
    assert proto.profile and len(proto.profile["points"]) >= 3
    assert arr.count == 5 and arr.angle_degrees == 360.0
    assert plan_mod.validate_plan(plan) == []


def test_revolve_profile_uses_observed_ring_radii():
    """Wheel cross-sections must come from the observed concentric rings:
    outer ring -> max radius, inner ring -> hollow section."""
    graph = operators_mod.classify_operators(make_wheel_graph(), NO_DEPTH, CFG)
    cam = camera_mod.estimate_camera(
        graph, make_seg(), make_crop_meta(), make_spec(output_dir="/nonexistent-dir"), CFG
    )
    plan = plan_mod.build_plan(graph, cam, NO_DEPTH, make_spec(output_dir="/nonexistent-dir"), CFG)
    by_id = {p.id: p for p in plan.parts}
    tyre_pts = by_id["tyre"].profile["points"]
    r_out = max(p[0] for p in tyre_pts)
    r_in = min(p[0] for p in tyre_pts)
    # tyre outer ring radius 0.30 over a 0.6-wide graph -> 0.5 object units
    assert r_out == pytest.approx(0.5, rel=0.02)
    # hollow: next ring in (rim at 0.20 -> 0.333 object units)
    assert r_in == pytest.approx(0.20 / 0.6, rel=0.05)
    # axis stays the object symmetry axis; tilt is applied at render time
    assert by_id["tyre"].axis["direction"] == [0.0, 0.0, 1.0]
    half_h = (max(p[1] for p in tyre_pts) - min(p[1] for p in tyre_pts)) / 2.0
    assert half_h > 0.02  # not the old degenerate 0.015 box


def test_bottle_revolve_uses_observed_side_silhouette_and_upright_axis():
    graph = SketchGraph(
        primitives=[_rect_region("body_outline", 0.35, 0.1, 0.65, 0.9)],
        parts=[_part("body", "bottle_body", ["body_outline"])])
    graph = operators_mod.classify_operators(graph, NO_DEPTH, CFG)
    plan = plan_mod.build_plan(
        graph, CameraEstimate(), NO_DEPTH,
        make_spec(target_label="bottle", output_dir="/nonexistent-dir"), CFG)
    body = next(p for p in plan.parts if p.id == "body")
    assert body.operator == OperatorCategory.REVOLVE
    assert body.axis["direction"] == [0.0, 1.0, 0.0]
    assert len(body.profile["points"]) >= 10
    assert max(p[0] for p in body.profile["points"]) == pytest.approx(0.5)


def test_validate_rejects_self_referencing_array():
    arr = PlanPart(
        id="arr",
        operator=OperatorCategory.RADIAL_ARRAY,
        source_part="arr",
        count=5,
        angle_degrees=360.0,
    )
    plan = _plan_with([arr])
    errors = plan_mod.validate_plan(plan)
    assert any("itself" in e for e in errors)


def test_validate_rejects_array_without_geometry_prototype():
    arr = PlanPart(
        id="arr2",
        operator=OperatorCategory.RADIAL_ARRAY,
        source_part="tex",
        count=5,
        angle_degrees=360.0,
    )
    tex = PlanPart(id="tex", operator=OperatorCategory.TEXTURE_ONLY)
    plan = _plan_with([tex, arr])
    errors = plan_mod.validate_plan(plan)
    assert any("prototype" in e for e in errors)


# ---------------------------------------------------------------------------
# (b) bracket: extrude + boolean
# ---------------------------------------------------------------------------

def test_bracket_operators_and_plan():
    graph = operators_mod.classify_operators(make_bracket_graph(), NO_DEPTH, CFG)
    selected = {p.id: p.selected_operator for p in graph.parts}
    assert selected["bracket"] == OperatorCategory.EXTRUDE.value
    assert selected["hole"] == OperatorCategory.BOOLEAN.value

    plan = plan_mod.build_plan(
        graph, CameraEstimate(), NO_DEPTH, make_spec(output_dir="/nonexistent-dir"), CFG
    )
    by_id = {p.id: p for p in plan.parts}
    assert by_id["bracket"].operator == OperatorCategory.EXTRUDE
    assert by_id["bracket"].depth is not None and by_id["bracket"].depth > 0
    assert by_id["hole"].operator == OperatorCategory.BOOLEAN
    assert by_id["hole"].boolean_target == "bracket"
    assert by_id["hole"].boolean_operation == "difference"
    errors = plan_mod.validate_plan(plan)
    assert errors == [], errors


# ---------------------------------------------------------------------------
# (c) validate_plan catches invalid plans
# ---------------------------------------------------------------------------

def _valid_extrude_part(pid, **kw):
    args = dict(
        id=pid,
        operator=OperatorCategory.EXTRUDE,
        profile={"type": "polyline",
                 "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                 "closed": True},
        depth=0.1,
    )
    args.update(kw)
    return PlanPart(**args)


def _plan_with(parts):
    return plan_mod.ConstructionPlan(object_id="t", parts=parts)


def test_validate_missing_parent():
    plan = _plan_with([_valid_extrude_part("a", parent="ghost")])
    assert plan_mod.validate_plan(plan)


def test_validate_parent_cycle():
    plan = _plan_with([
        _valid_extrude_part("a", parent="b"),
        _valid_extrude_part("b", parent="a"),
    ])
    assert plan_mod.validate_plan(plan)


def test_validate_boolean_self_target_and_dependency_cycle():
    self_cut = PlanPart(
        id="self_cut", operator=OperatorCategory.BOOLEAN,
        boolean_target="self_cut", boolean_operation="difference",
        profile={"type": "polyline", "points": [[0, 0], [1, 0], [0, 1]],
                 "closed": True}, depth=0.1)
    errors = plan_mod.validate_plan(_plan_with([self_cut]))
    assert any("itself" in e for e in errors)

    a = self_cut.model_copy(update={"id": "a", "boolean_target": "b"})
    b = self_cut.model_copy(update={"id": "b", "boolean_target": "a"})
    errors = plan_mod.validate_plan(_plan_with([a, b]))
    assert any("dependency cycle" in e for e in errors)


def test_boolean_target_prefers_larger_non_cutter_container():
    graph = make_bracket_graph()
    # Add a same-size sibling hole and constraints in the adversarial order
    # that previously produced cutter-to-cutter cycles.
    graph.primitives.append(_circle("hole_prim_2", 0.6, 0.5, 0.05))
    graph.parts.append(_part("hole_2", "hole", ["hole_prim_2"]))
    graph.constraints.insert(0, GeometricConstraint(
        type=ConstraintType.CONTAINMENT,
        entities=["hole_prim", "hole_prim_2"], confidence=0.6))
    graph.constraints.append(GeometricConstraint(
        type=ConstraintType.CONTAINMENT,
        entities=["hole_prim_2", "bracket_region"], confidence=0.9))
    graph = operators_mod.classify_operators(graph, NO_DEPTH, CFG)
    plan = plan_mod.build_plan(
        graph, CameraEstimate(), NO_DEPTH,
        make_spec(output_dir="/nonexistent-dir"), CFG)
    by_id = {p.id: p for p in plan.parts}
    assert by_id["hole"].boolean_target == "bracket"
    assert by_id["hole_2"].boolean_target == "bracket"
    assert plan_mod.validate_plan(plan) == []


def test_validate_zero_axis():
    part = PlanPart(
        id="r",
        operator=OperatorCategory.REVOLVE,
        axis={"origin": [0, 0, 0], "direction": [0.0, 0.0, 0.0]},
        profile={"type": "polyline",
                 "points": [[0.0, -0.1], [0.2, -0.1], [0.2, 0.1], [0.0, 0.1]],
                 "closed": True},
    )
    assert plan_mod.validate_plan(_plan_with([part]))


def test_validate_bad_array_count():
    part = PlanPart(
        id="arr",
        operator=OperatorCategory.RADIAL_ARRAY,
        source_part="src",
        count=1,
        angle_degrees=360.0,
    )
    plan = _plan_with([_valid_extrude_part("src"), part])
    assert plan_mod.validate_plan(plan)


def test_validate_negative_radius():
    part = PlanPart(
        id="r",
        operator=OperatorCategory.REVOLVE,
        axis={"origin": [0, 0, 0], "direction": [0, 0, 1]},
        profile={"type": "polyline",
                 "points": [[-0.1, -0.1], [0.2, -0.1], [0.2, 0.1]],
                 "closed": True},
    )
    assert plan_mod.validate_plan(_plan_with([part]))


def test_validate_non_finite_value():
    plan = _plan_with([_valid_extrude_part("a", depth=float("nan"))])
    assert plan_mod.validate_plan(plan)


def test_validate_material_out_of_range():
    part = _valid_extrude_part("a")
    part.material = MaterialSpec(base_color=(1.5, 0.5, 0.5))
    assert plan_mod.validate_plan(_plan_with([part]))


# ---------------------------------------------------------------------------
# (d) materials: highlights / shadows must not bake into base_color
# ---------------------------------------------------------------------------

def _write_highlight_shadow_crop(path, size=64):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 24, (128, 128, 128, 255), -1)
    cv2.circle(img, (24, 24), 5, (255, 255, 255, 255), -1)       # highlight blob
    cv2.circle(img, (42, 44), 6, (40, 40, 40, 255), -1)          # shadow blob
    cv2.imwrite(str(path), img)
    return str(path)


def test_materials_highlight_shadow_invariant(tmp_path):
    rgba = _write_highlight_shadow_crop(tmp_path / "crop_rgba.png")
    graph = SketchGraph(
        primitives=[_circle("body_prim", 0.5, 0.5, 24.0 / 64.0)],
        parts=[_part("body", "body", ["body_prim"])],
    )
    mats = materials_mod.estimate_materials(graph, rgba, CFG)
    spec = mats["body"]
    # mid-gray 128 sRGB -> ~0.212 linear; must stay near mid-gray, not
    # highlight-white (~1.0) or shadow-black (~0.02)
    for ch in spec.base_color:
        assert 0.1 < ch < 0.4, spec.base_color
    assert spec.material_class in ("rubber", "plastic", "metal")


def test_semantic_material_priors_change_class_but_preserve_observed_color():
    observed = MaterialSpec(
        material_class="metal", base_color=(0.12, 0.2, 0.3),
        source=EvidenceSource.FITTED_FROM_OBSERVATION)
    bottle = materials_mod.apply_semantic_class_priors(
        {"part_bottle_body_0": observed}, "bottle")
    assert bottle["part_bottle_body_0"].material_class == "glass"
    assert bottle["part_bottle_body_0"].base_color == observed.base_color
    assert bottle["part_bottle_body_0"].source == EvidenceSource.SEMANTIC_PRIOR
    assert observed.material_class == "metal"  # input was not mutated

    wheel = materials_mod.apply_semantic_class_priors({
        "part_tyre_0": observed, "part_rim_0": observed,
        "part_hub_0": observed}, "wheel")
    assert wheel["part_tyre_0"].material_class == "rubber"
    assert wheel["part_rim_0"].material_class == "metal"
    assert wheel["part_hub_0"].material_class == "plastic"


# ---------------------------------------------------------------------------
# depth heuristic (stage 12)
# ---------------------------------------------------------------------------

def test_depth_enabled_and_disabled(tmp_path):
    size = 64
    rgba_path = _write_highlight_shadow_crop(tmp_path / "crop_rgba.png", size)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), 24, 255, -1)
    mask_path = str(tmp_path / "crop_mask.png")
    cv2.imwrite(mask_path, mask)

    graph = SketchGraph(
        primitives=[_circle("body_prim", 0.5, 0.5, 24.0 / 64.0)],
        parts=[_part("body", "body", ["body_prim"])],
    )
    out_dir = str(tmp_path / "geom")
    ev = depth_mod.estimate_depth(rgba_path, mask_path, graph, out_dir, CFG)
    assert ev.backend == "silhouette_shading"
    assert ev.confidence > 0
    depth_img = cv2.imread(ev.depth_path, cv2.IMREAD_UNCHANGED)
    assert depth_img is not None and depth_img.dtype == np.uint16
    assert cv2.imread(ev.normals_path, cv2.IMREAD_UNCHANGED) is not None
    # dome: centre nearer than the rim of the mask
    assert depth_img[size // 2, size // 2] > depth_img[size // 2, size // 2 + 22]

    region = ev.region_estimates["body"]
    assert region.source == EvidenceSource.ESTIMATED_FROM_DEPTH
    assert region.confidence < 0.6
    assert region.value is not None

    cfg_off = PipelineConfig()
    cfg_off.depth.enabled = False
    ev_off = depth_mod.estimate_depth(rgba_path, mask_path, graph, out_dir, cfg_off)
    assert ev_off.backend == "none"
    assert ev_off.confidence == 0.0
    assert ev_off.depth_path is None
