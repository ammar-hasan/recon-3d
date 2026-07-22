"""Level B stage eval: constraint detection (EVAL.md Eval 8).

Builds synthetic primitive sets with known relationships and scores
recon3d.constraints.detect_constraints with precision/recall per constraint
type. Precision is prioritised over recall per EVAL.md.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("recon3d.constraints")

from recon3d import schemas as S  # noqa: E402
from recon3d.config import PipelineConfig  # noqa: E402
from recon3d.constraints import detect_constraints  # noqa: E402

from evals import metrics as m  # noqa: E402


def _circle(pid, cx, cy, r):
    return S.GeometricPrimitive(
        id=pid, type=S.PrimitiveType.CIRCLE,
        params={"center": [cx, cy], "radius": r},
        source_path="sp_" + pid, source_layer=S.TraceLayerName.SILHOUETTE)


def _ellipse(pid, cx, cy, ra, rb, rot=0.0):
    return S.GeometricPrimitive(
        id=pid, type=S.PrimitiveType.ELLIPSE,
        params={"center": [cx, cy], "radii": [ra, rb], "rotation_degrees": rot},
        source_path="sp_" + pid, source_layer=S.TraceLayerName.SILHOUETTE)


def _line(pid, p0, p1):
    return S.GeometricPrimitive(
        id=pid, type=S.PrimitiveType.LINE,
        params={"p0": list(p0), "p1": list(p1)},
        source_path="sp_" + pid, source_layer=S.TraceLayerName.SILHOUETTE)


def _detect(prims):
    return detect_constraints(prims, PipelineConfig())


def _found(constraints, ctype):
    return [c for c in constraints if c.type == ctype]


class TestConcentric:
    def test_concentric_circles_detected(self):
        prims = [_circle("a", 0.5, 0.5, 0.3),
                 _circle("b", 0.5, 0.5, 0.2),
                 _circle("c", 0.5, 0.5, 0.1)]
        cons = _found(_detect(prims), S.ConstraintType.CONCENTRIC)
        involved = {e for c in cons for e in c.entities}
        assert {"a", "b", "c"} <= involved

    def test_non_concentric_not_reported(self):
        prims = [_circle("a", 0.2, 0.2, 0.1),
                 _circle("b", 0.8, 0.8, 0.1)]
        cons = _found(_detect(prims), S.ConstraintType.CONCENTRIC)
        assert cons == []


class TestParallelPerpendicular:
    def test_parallel_lines(self):
        prims = [_line("l1", (0.1, 0.2), (0.9, 0.2)),
                 _line("l2", (0.1, 0.6), (0.9, 0.6))]
        cons = _found(_detect(prims), S.ConstraintType.PARALLEL)
        involved = {e for c in cons for e in c.entities}
        assert {"l1", "l2"} <= involved

    def test_perpendicular_lines(self):
        prims = [_line("l1", (0.1, 0.5), (0.9, 0.5)),
                 _line("l2", (0.5, 0.1), (0.5, 0.9))]
        cons = _found(_detect(prims), S.ConstraintType.PERPENDICULAR)
        involved = {e for c in cons for e in c.entities}
        assert {"l1", "l2"} <= involved

    def test_skew_lines_no_parallel(self):
        prims = [_line("l1", (0.1, 0.2), (0.9, 0.2)),
                 _line("l2", (0.1, 0.6), (0.9, 0.75))]
        cons = _found(_detect(prims), S.ConstraintType.PARALLEL)
        assert cons == []


class TestEqualRadius:
    def test_equal_radius_pair(self):
        prims = [_circle("a", 0.3, 0.5, 0.15),
                 _circle("b", 0.7, 0.5, 0.15)]
        cons = _found(_detect(prims), S.ConstraintType.EQUAL_RADIUS)
        involved = {e for c in cons for e in c.entities}
        assert {"a", "b"} <= involved


class TestContainment:
    def test_containment(self):
        prims = [_circle("outer", 0.5, 0.5, 0.4),
                 _circle("inner", 0.5, 0.5, 0.1)]
        types = {c.type for c in _detect(prims)}
        assert (S.ConstraintType.CONTAINMENT in types
                or S.ConstraintType.CONCENTRIC in types)


class TestPrecisionOverRecall:
    """A mixed scene: score recall of expected constraints and verify that no
    geometrically FALSE constraints are reported (precision priority)."""

    def test_scene_precision_recall(self):
        prims = [
            _circle("hub", 0.5, 0.5, 0.05),
            _circle("rim", 0.5, 0.5, 0.2),
            _circle("tyre", 0.5, 0.5, 0.35),
            _line("top", (0.2, 0.9), (0.8, 0.9)),
            _line("mid", (0.2, 0.95), (0.8, 0.95)),
            _ellipse("far", 0.9, 0.1, 0.05, 0.03),
        ]
        # expected detections (the scene also legitimately supports other true
        # constraints, e.g. containment of hub in tyre - those are NOT false)
        expected = {
            (S.ConstraintType.CONCENTRIC, "hub", "rim"),
            (S.ConstraintType.CONCENTRIC, "hub", "tyre"),
            (S.ConstraintType.CONCENTRIC, "rim", "tyre"),
            (S.ConstraintType.PARALLEL, "top", "mid"),
        }
        concentric_group = {"hub", "rim", "tyre"}
        parallel_group = {"top", "mid"}

        detected = _detect(prims)
        covered = set()
        false_constraints = []
        for c in detected:
            ents = set(c.entities)
            for (t, a, b) in expected:
                if c.type == t and {a, b} <= ents:
                    covered.add((t, a, b))
            if c.type == S.ConstraintType.CONCENTRIC and not ents <= concentric_group:
                false_constraints.append(c)
            if c.type == S.ConstraintType.PARALLEL and not ents <= parallel_group:
                false_constraints.append(c)
            if c.type == S.ConstraintType.PERPENDICULAR:
                false_constraints.append(c)  # no perpendicular lines exist

        recall = len(covered) / len(expected)
        assert recall >= 0.85, "expected-constraint recall %.2f" % recall
        false_rate = (len(false_constraints) / len(detected)) if detected else 0.0
        assert false_rate <= 0.08, "false constraints: %s" % (
            [(c.type.value, c.entities) for c in false_constraints],)
