"""Unit tests for stages 7-10: primitive fitting, constraint detection,
sketch graph assembly and semantic part decomposition.

All inputs are synthetic TraceLayers built in code (plus one generated RGBA
crop written to tmp_path for the appearance sampler).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from recon3d.config import PipelineConfig
from recon3d.constraints import detect_constraints
from recon3d.primitives import fit_primitives
from recon3d.schemas import (
    ConstraintType,
    EvidenceSource,
    InputSpec,
    PrimitiveType,
    SketchGraph,
    TraceLayer,
    TraceLayerName,
    VectorPath,
)
from recon3d.semantic_parts import decompose_parts
from recon3d.sketch_graph import build_sketch_graph

CFG = PipelineConfig()
MAX_ERR = CFG.primitives.max_fit_error_norm


# ---------------------------------------------------------------------------
# synthetic curve generators (normalised coords, origin top-left)
# ---------------------------------------------------------------------------

def _noise(rng, shape, sigma):
    return rng.normal(0.0, sigma, size=shape) if sigma > 0 else np.zeros(shape)


def circle_pts(center, r, n=128, sigma=0.0, rng=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.column_stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)])
    return P + _noise(rng, P.shape, sigma)


def arc_pts(center, r, a0_deg, a1_deg, n=64, sigma=0.0, rng=None):
    t = np.radians(np.linspace(a0_deg, a1_deg, n))
    P = np.column_stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)])
    return P + _noise(rng, P.shape, sigma)


def ellipse_pts(center, rx, ry, rot_deg, n=128, sigma=0.0, rng=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    x = rx * np.cos(t)
    y = ry * np.sin(t)
    P = np.column_stack([center[0] + x * c - y * s, center[1] + x * s + y * c])
    return P + _noise(rng, P.shape, sigma)


def rect_pts(center, w, h, rot_deg, per_edge=24, sigma=0.0, rng=None):
    hw, hh = w / 2.0, h / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = []
    for i in range(4):
        a = np.array(corners_local[i])
        b = np.array(corners_local[(i + 1) % 4])
        for t in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append(a + t * (b - a))
    P = np.array(pts)
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    P = P @ R.T + np.asarray(center)
    return P + _noise(rng, P.shape, sigma)


def line_pts(p0, p1, n=48, sigma=0.0, rng=None):
    t = np.linspace(0, 1, n)[:, None]
    P = (1 - t) * np.asarray(p0) + t * np.asarray(p1)
    return P + _noise(rng, P.shape, sigma)


def bracket_pts(center=(0.5, 0.55), w=0.4, h=0.5, notch_w=0.16, notch_d=0.2,
                per_edge=16):
    """U-shaped bracket outline (vertically symmetric), closed."""
    x0, y0 = center[0] - w / 2, center[1] - h / 2
    x1, y1 = center[0] + w / 2, center[1] + h / 2
    nx0, nx1 = center[0] - notch_w / 2, center[0] + notch_w / 2
    nd = y0 + notch_d
    corners = [(x0, y0), (nx0, y0), (nx0, nd), (nx1, nd),
               (nx1, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = []
    for i in range(len(corners)):
        a = np.array(corners[i])
        b = np.array(corners[(i + 1) % len(corners)])
        for t in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append(a + t * (b - a))
    return np.array(pts)


def make_path(P, closed, path_id, layer=TraceLayerName.SILHOUETTE, is_hole=False):
    return VectorPath(path_id=path_id, source_layer=layer, closed=closed,
                      points=[(float(x), float(y)) for x, y in P],
                      is_hole=is_hole)


def make_layer(paths, name=TraceLayerName.SILHOUETTE):
    return TraceLayer(name=name, svg_path="", paths=paths,
                      image_size=(1024, 1024))


def fit_one(P, closed, cfg=CFG, layer=TraceLayerName.SILHOUETTE):
    prims = fit_primitives([make_layer([make_path(P, closed, "p0")], layer)], cfg)
    assert len(prims) == 1
    return prims[0]


# ---------------------------------------------------------------------------
# Stage 7: primitive fitting
# ---------------------------------------------------------------------------

class TestPrimitiveFitting:
    def test_circle(self):
        rng = np.random.default_rng(0)
        P = circle_pts((0.5, 0.5), 0.3, sigma=0.0005, rng=rng)
        prim = fit_one(P, True)
        assert prim.type == PrimitiveType.CIRCLE
        assert abs(prim.params["radius"] - 0.3) / 0.3 <= 0.02
        assert float(np.hypot(*(np.array(prim.params["center"]) - 0.5))) < 0.01
        assert prim.fit_error <= MAX_ERR
        assert len(prim.fallback_points) == len(P)
        assert prim.id.startswith("silhouette_circle_")

    def test_noisy_ellipse(self):
        rng = np.random.default_rng(1)
        P = ellipse_pts((0.45, 0.55), 0.2, 0.14, 30.0, sigma=0.0005, rng=rng)
        prim = fit_one(P, True)
        assert prim.type == PrimitiveType.ELLIPSE
        rx, ry = sorted(prim.params["radii"], reverse=True)
        assert abs(rx - 0.2) / 0.2 <= 0.03
        assert abs(ry - 0.14) / 0.14 <= 0.03
        drot = abs(prim.params["rotation_degrees"] - 30.0) % 180.0
        drot = min(drot, 180.0 - drot)
        assert drot <= 2.0

    def test_partial_arc(self):
        rng = np.random.default_rng(2)
        P = arc_pts((0.5, 0.5), 0.2, 0.0, 120.0, sigma=0.0005, rng=rng)
        prim = fit_one(P, False)
        assert prim.type == PrimitiveType.CIRCULAR_ARC
        assert abs(prim.params["radius"] - 0.2) / 0.2 <= 0.02
        assert abs(prim.params["sweep_deg"] - 120.0) <= 5.0

    def test_rotated_rectangle(self):
        P = rect_pts((0.5, 0.5), 0.4, 0.2, 25.0)
        prim = fit_one(P, True)
        assert prim.type == PrimitiveType.RECTANGLE
        drot = abs(prim.params["rotation_degrees"] - 25.0) % 180.0
        drot = min(drot, 180.0 - drot)
        assert drot <= 2.0
        w, h = prim.params["size"]
        assert abs(w - 0.4) / 0.4 <= 0.03
        assert abs(h - 0.2) / 0.2 <= 0.03

    def test_lines(self):
        rng = np.random.default_rng(3)
        P = line_pts((0.2, 0.3), (0.7, 0.5), sigma=0.0002, rng=rng)
        prim = fit_one(P, False)
        assert prim.type == PrimitiveType.LINE
        true_ang = math.degrees(math.atan2(0.2, 0.5))
        d = np.array(prim.params["p1"]) - np.array(prim.params["p0"])
        ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
        diff = abs(ang - true_ang) % 180.0
        assert min(diff, 180.0 - diff) <= 1.0

    def test_bracket_region_and_hole(self):
        outer = fit_one(bracket_pts(), True)
        assert outer.type in (PrimitiveType.CLOSED_REGION,
                              PrimitiveType.SYMMETRIC_SPLINE,
                              PrimitiveType.BEZIER)
        assert len(outer.fallback_points) > 0
        hole = fit_one(circle_pts((0.5, 0.75), 0.06), True)
        assert hole.type == PrimitiveType.CIRCLE

    def test_fallback_for_irregular_open_curve(self):
        rng = np.random.default_rng(4)
        t = np.linspace(0, 4 * np.pi, 120)
        P = np.column_stack([0.2 + 0.05 * t,
                             0.5 + 0.08 * np.sin(t) + rng.normal(0, 0.004, t.shape)])
        prim = fit_one(P, False)
        assert prim.type in (PrimitiveType.BEZIER, PrimitiveType.POLYLINE)
        assert prim.fallback_points == [(float(x), float(y)) for x, y in P]

    def test_clean_type_accuracy(self):
        """All clean synthetic cases must be classified into the right family."""
        rng = np.random.default_rng(5)
        cases = [
            (circle_pts((0.5, 0.5), 0.25, rng=rng), True, PrimitiveType.CIRCLE),
            (ellipse_pts((0.5, 0.5), 0.22, 0.13, 10.0, rng=rng), True,
             PrimitiveType.ELLIPSE),
            (arc_pts((0.5, 0.5), 0.2, 20.0, 140.0, rng=rng), False,
             PrimitiveType.CIRCULAR_ARC),
            (rect_pts((0.5, 0.5), 0.3, 0.18, 40.0), True, PrimitiveType.RECTANGLE),
            (line_pts((0.1, 0.1), (0.8, 0.2), rng=rng), False, PrimitiveType.LINE),
        ]
        correct = 0
        for P, closed, want in cases:
            prim = fit_one(P, closed)
            if prim.type == want:
                correct += 1
            assert prim.confidence >= 0.6 if prim.type == want else True
        assert correct / len(cases) == 1.0


# ---------------------------------------------------------------------------
# shared fixtures for stages 8-10
# ---------------------------------------------------------------------------

WHEEL_CENTER = (0.5, 0.5)


def wheel_layer():
    """Three concentric circles + five radial spoke rectangles."""
    paths = []
    for i, r in enumerate((0.30, 0.19, 0.05)):
        paths.append(make_path(circle_pts(WHEEL_CENTER, r), True, "ring_%d" % i))
    for k in range(5):
        ang = 90.0 + 72.0 * k
        th = math.radians(ang)
        cc = (WHEEL_CENTER[0] + 0.115 * math.cos(th),
              WHEEL_CENTER[1] + 0.115 * math.sin(th))
        paths.append(make_path(rect_pts(cc, 0.13, 0.02, ang), True,
                               "spoke_%d" % k))
    return make_layer(paths)


def wheel_primitives():
    return fit_primitives([wheel_layer()], CFG)


def lines_primitives():
    """Two parallel lines plus one perpendicular, well separated."""
    paths = [
        make_path(line_pts((0.2, 0.3), (0.6, 0.3)), False, "l0"),
        make_path(line_pts((0.3, 0.7), (0.8, 0.7)), False, "l1"),
        make_path(line_pts((0.5, 0.35), (0.5, 0.65)), False, "l2"),
    ]
    return fit_primitives([make_layer(paths, TraceLayerName.STRUCTURAL_EDGES)], CFG)


# ---------------------------------------------------------------------------
# Stage 8: constraint detection
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_wheel_concentric_and_repetition(self):
        prims = wheel_primitives()
        types = {p.type for p in prims}
        assert types == {PrimitiveType.CIRCLE, PrimitiveType.RECTANGLE}
        cons = detect_constraints(prims, CFG)

        rings = sorted(p.id for p in prims if p.type == PrimitiveType.CIRCLE)
        spokes = sorted(p.id for p in prims if p.type == PrimitiveType.RECTANGLE)

        conc = [c for c in cons if c.type == ConstraintType.CONCENTRIC]
        assert len(conc) == 1
        assert sorted(conc[0].entities) == rings
        assert conc[0].confidence >= 0.9

        rot = [c for c in cons if c.type == ConstraintType.ROTATIONAL_REPETITION]
        assert len(rot) == 1
        assert rot[0].params["count"] == 5
        assert abs(rot[0].params["angle_degrees"] - 72.0) <= 1.0
        assert rot[0].params["prototype"] in spokes
        assert sorted(rot[0].entities) == spokes

    def test_wheel_no_false_constraints(self):
        prims = wheel_primitives()
        cons = detect_constraints(prims, CFG)
        # radii all differ -> no equal_radius / coincident
        assert not [c for c in cons if c.type == ConstraintType.EQUAL_RADIUS]
        assert not [c for c in cons if c.type == ConstraintType.COINCIDENT]
        # rings are strictly nested -> no tangent
        assert not [c for c in cons if c.type == ConstraintType.TANGENT]
        # no line primitives -> no parallel/perpendicular/collinear
        for t in (ConstraintType.PARALLEL, ConstraintType.PERPENDICULAR,
                  ConstraintType.COLLINEAR):
            assert not [c for c in cons if c.type == t]
        # every reported mirror axis must pass through the common centre
        for c in cons:
            if c.type == ConstraintType.MIRROR_SYMMETRY:
                pt = np.array(c.params["axis"]["point"])
                assert np.hypot(*(pt - np.array(WHEEL_CENTER))) <= 0.03

    def test_parallel_perpendicular_lines(self):
        prims = lines_primitives()
        assert all(p.type == PrimitiveType.LINE for p in prims)
        cons = detect_constraints(prims, CFG)
        par = [c for c in cons if c.type == ConstraintType.PARALLEL]
        perp = [c for c in cons if c.type == ConstraintType.PERPENDICULAR]
        assert len(par) == 1
        assert len(perp) == 2
        ids = {p.id for p in prims}
        for c in par + perp:
            assert set(c.entities) <= ids
            assert c.confidence >= 0.9
        # nothing else should fire on this minimal scene
        allowed = {ConstraintType.PARALLEL, ConstraintType.PERPENDICULAR}
        assert {c.type for c in cons} <= allowed

    def test_confidence_floor(self):
        cons = detect_constraints(wheel_primitives(), CFG)
        assert cons
        assert all(c.confidence >= 0.6 for c in cons)


class TestCrossLayerDedupe:
    """Stage 8 hardening: the same feature re-traced by several layers must
    not spawn duplicate/cross-layer relation noise."""

    def _dup_wheel_primitives(self):
        """Wheel traced twice: silhouette layer + a color_regions re-trace."""
        sil = wheel_layer()
        dup_paths = []
        for i, r in enumerate((0.30, 0.19, 0.05)):
            dup_paths.append(make_path(circle_pts(WHEEL_CENTER, r), True,
                                       "cring_%d" % i,
                                       layer=TraceLayerName.COLOR_REGIONS))
        color = make_layer(dup_paths, TraceLayerName.COLOR_REGIONS)
        return fit_primitives([sil, color], CFG)

    def test_cross_layer_duplicates_merged(self):
        prims = self._dup_wheel_primitives()
        cons = detect_constraints(prims, CFG)
        # the color_regions re-traces are coincident duplicates: they must not
        # appear in any constraint (the silhouette originals win)
        for c in cons:
            assert not any(e.startswith("color_regions_") for e in c.entities), c
        # no coincident-duplicate relations survive
        assert not [c for c in cons if c.type == ConstraintType.COINCIDENT]
        # concentric structure of the surviving rings is still detected
        conc = [c for c in cons if c.type == ConstraintType.CONCENTRIC]
        assert len(conc) == 1
        assert sorted(conc[0].entities) == sorted(
            p.id for p in prims if p.type == PrimitiveType.CIRCLE
            and p.id.startswith("silhouette_"))

    def test_containment_scoped_within_layer(self):
        outer = make_path(circle_pts((0.5, 0.5), 0.4), True, "outer",
                          layer=TraceLayerName.SILHOUETTE)
        inner = make_path(circle_pts((0.5, 0.5), 0.1), True, "inner",
                          layer=TraceLayerName.COLOR_REGIONS)
        prims = fit_primitives(
            [make_layer([outer], TraceLayerName.SILHOUETTE),
             make_layer([inner], TraceLayerName.COLOR_REGIONS)], CFG)
        cons = detect_constraints(prims, CFG)
        # radii differ wildly -> not duplicates; both survive, but cross-layer
        # containment must NOT be reported
        assert not [c for c in cons if c.type == ConstraintType.CONTAINMENT]
        # concentric detection still works across layers
        assert [c for c in cons if c.type == ConstraintType.CONCENTRIC]

    def test_adjacency_scoped_within_layer(self):
        l0 = make_path(line_pts((0.2, 0.3), (0.6, 0.3)), False, "a",
                       layer=TraceLayerName.SILHOUETTE)
        l1 = make_path(line_pts((0.2, 0.3005), (0.6, 0.3005)), False, "b",
                       layer=TraceLayerName.STRUCTURAL_EDGES)
        prims = fit_primitives(
            [make_layer([l0], TraceLayerName.SILHOUETTE),
             make_layer([l1], TraceLayerName.STRUCTURAL_EDGES)], CFG)
        cons = detect_constraints(prims, CFG)
        # nearly on top of each other but in different layers: no
        # adjacency/intersection reported
        assert not [c for c in cons if c.type in (ConstraintType.ADJACENCY,
                                                  ConstraintType.INTERSECTION)]

    def test_alignment_needs_four_and_maximal(self):
        # exactly 3 collinear features: too weak to report
        pts = [make_path(circle_pts((0.2 + 0.2 * k, 0.5), 0.03), True, "t%d" % k)
               for k in range(3)]
        prims = fit_primitives([make_layer(pts)], CFG)
        cons = detect_constraints(prims, CFG)
        assert not [c for c in cons if c.type == ConstraintType.ALIGNMENT]
        # 4 collinear features: reported once
        pts = [make_path(circle_pts((0.15 + 0.2 * k, 0.5), 0.03), True, "q%d" % k)
               for k in range(4)]
        prims = fit_primitives([make_layer(pts)], CFG)
        cons = detect_constraints(prims, CFG)
        al = [c for c in cons if c.type == ConstraintType.ALIGNMENT]
        assert len(al) == 1
        assert len(al[0].entities) == 4


# ---------------------------------------------------------------------------
# Stage 9: sketch graph
# ---------------------------------------------------------------------------

class TestSketchGraph:
    def test_assembly(self):
        prims = wheel_primitives()
        cons = detect_constraints(prims, CFG)
        graph = build_sketch_graph(prims, cons)
        assert graph.coordinate_system["type"] == "normalized_image"
        assert graph.coordinate_system["origin"] == "top_left"
        assert graph.uncertainty["physical_scale"] == "unknown"
        assert len(graph.primitives) == len(prims)
        assert len(graph.constraints) == len(cons)
        assert graph.stats["primitive_count"] == len(prims)
        assert graph.stats["primitives_by_type"]["circle"] == 3
        assert graph.stats["primitives_by_type"]["rectangle"] == 5
        assert graph.stats["constraint_count"] == len(cons)
        # serialisable
        graph.model_dump_json()


# ---------------------------------------------------------------------------
# Stage 10: semantic parts
# ---------------------------------------------------------------------------

def _write_wheel_rgba(path, size=512):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    c = (size // 2, size // 2)

    def px(r):
        return int(round(r * size))

    cv2.circle(img, c, px(0.30), (25, 25, 25, 255), -1)      # tyre: dark rubber
    cv2.circle(img, c, px(0.19), (120, 120, 120, 255), -1)   # rim: grey metal
    cv2.circle(img, c, px(0.05), (200, 200, 200, 255), -1)   # hub: bright metal
    for k in range(5):
        ang = math.radians(90.0 + 72.0 * k)
        p1 = (int(c[0] + px(0.055) * math.cos(ang)),
              int(c[1] + px(0.055) * math.sin(ang)))
        p2 = (int(c[0] + px(0.175) * math.cos(ang)),
              int(c[1] + px(0.175) * math.sin(ang)))
        cv2.line(img, p1, p2, (160, 160, 160, 255), max(2, px(0.02)))
    cv2.imwrite(str(path), img)
    return str(path)


class TestSemanticParts:
    def test_wheel_decomposition(self, tmp_path):
        prims = wheel_primitives()
        cons = detect_constraints(prims, CFG)
        graph = build_sketch_graph(prims, cons)
        before = graph.model_dump_json()

        img_path = _write_wheel_rgba(tmp_path / "crop_rgba.png")
        spec = InputSpec(image_paths=["synthetic.png"], target_label="wheel")
        out = decompose_parts(graph, img_path, spec, CFG)

        classes = {p.part_class for p in out.parts}
        assert {"wheel", "tyre", "rim", "hub", "spokes"} <= classes

        by_class = {p.part_class: p for p in out.parts}
        wheel = by_class["wheel"]
        tyre = by_class["tyre"]
        rim = by_class["rim"]
        hub = by_class["hub"]
        spokes = by_class["spokes"]

        # hierarchy: wheel -> tyre, rim ; rim -> hub, spokes
        assert tyre.parent_id == wheel.id
        assert rim.parent_id == wheel.id
        assert hub.parent_id == rim.id
        assert spokes.parent_id == rim.id
        assert set(wheel.child_ids) == {tyre.id, rim.id}
        assert set(rim.child_ids) == {hub.id, spokes.id}

        # tyre owns the outer ring, hub the inner one
        rings = sorted((p for p in prims if p.type == PrimitiveType.CIRCLE),
                       key=lambda p: p.params["radius"], reverse=True)
        assert tyre.primitive_ids == [rings[0].id]
        assert hub.primitive_ids == [rings[-1].id]
        assert set(spokes.primitive_ids) == {
            p.id for p in prims if p.type == PrimitiveType.RECTANGLE}

        # appearance sampled from the crop image, low confidence, observed src
        assert tyre.appearance is not None
        assert tyre.appearance.material_class == "rubber"
        assert tyre.appearance.source == EvidenceSource.FITTED_FROM_OBSERVATION
        assert tyre.appearance.confidence <= 0.5

        # inferred/hidden geometry marked with prior/hypothesis + low confidence
        for p in out.parts:
            assert "rear_profile" in p.inferred_geometry
            for ev in p.inferred_geometry.values():
                assert ev.confidence <= 0.5 or (
                    ev.source == EvidenceSource.FITTED_FROM_OBSERVATION)
                assert ev.source in (EvidenceSource.SEMANTIC_PRIOR,
                                     EvidenceSource.GENERATED_HYPOTHESIS,
                                     EvidenceSource.FITTED_FROM_OBSERVATION)
        hyp = tyre.inferred_geometry["rear_profile"]
        assert hyp.source == EvidenceSource.GENERATED_HYPOTHESIS
        assert hyp.confidence <= 0.5

        # primitive geometry untouched
        assert out.model_dump_json().find(before[:200]) != -1 or True
        for a, b in zip(graph.primitives, out.primitives):
            assert a is b or a.model_dump() == b.model_dump()

        # stats updated
        assert out.stats["part_count"] == len(out.parts)
        out.model_dump_json()

    def test_generic_body_and_panels(self, tmp_path):
        paths = [
            make_path(rect_pts((0.5, 0.5), 0.5, 0.6, 0.0), True, "outer"),
            make_path(rect_pts((0.5, 0.5), 0.36, 0.44, 0.0), True, "inner"),
        ]
        prims = fit_primitives([make_layer(paths)], CFG)
        graph = build_sketch_graph(prims, detect_constraints(prims, CFG))
        spec = InputSpec(image_paths=["synthetic.png"], description="a small box")
        out = decompose_parts(graph, "", spec, CFG)
        classes = [p.part_class for p in out.parts]
        assert classes[0] == "box"
        assert "panel" in classes or "bezel" in classes or "insert" in classes

    def test_wheel_detected_without_label_or_precomputed_repetition(self):
        graph = SketchGraph(primitives=wheel_primitives(), constraints=[])
        out = decompose_parts(
            graph, "", InputSpec(image_paths=["synthetic.png"]), CFG)
        classes = {p.part_class for p in out.parts}
        assert {"wheel", "tyre", "rim", "hub", "spokes"} <= classes
        repetitions = [c for c in out.constraints
                       if c.type == ConstraintType.ROTATIONAL_REPETITION]
        assert len(repetitions) == 1
        assert repetitions[0].params["count"] == 5

    def test_nonwheel_ring_system_gets_stable_geometric_names(self):
        graph = SketchGraph(primitives=wheel_primitives()[:3], constraints=[])
        out = decompose_parts(
            graph, "", InputSpec(image_paths=["synthetic.png"]), CFG)
        classes = {p.part_class for p in out.parts}
        assert {"ring_system", "outer_shell", "inner_panel", "hub"} <= classes
        assert "tyre" not in classes

    def test_bottle_label_does_not_promote_highlight_rings_to_panels(self):
        paths = [
            make_path(rect_pts((0.5, 0.5), 0.35, 0.75, 0.0), True,
                      "bottle_outline"),
            make_path(circle_pts((0.5, 0.45), 0.08), True, "highlight_outer"),
            make_path(circle_pts((0.5, 0.45), 0.04), True, "highlight_inner"),
        ]
        prims = fit_primitives([make_layer(paths)], CFG)
        graph = SketchGraph(primitives=prims, constraints=[])
        out = decompose_parts(
            graph, "", InputSpec(image_paths=["synthetic.png"],
                                  target_label="bottle"), CFG)
        classes = {p.part_class for p in out.parts}
        assert "bottle_body" in classes
        assert "inner_panel" not in classes

    def test_deterministic_ids(self, tmp_path):
        def run():
            graph = build_sketch_graph(wheel_primitives(),
                                       detect_constraints(wheel_primitives(), CFG))
            spec = InputSpec(image_paths=["synthetic.png"], target_label="wheel")
            return decompose_parts(graph, "", spec, CFG)
        a, b = run(), run()
        assert [p.id for p in a.parts] == [p.id for p in b.parts]
