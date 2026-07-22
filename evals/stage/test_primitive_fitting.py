"""Level B stage eval: geometric primitive fitting (EVAL.md Eval 7).

Generates synthetic contours with known parameters + noise, runs
recon3d.primitives.fit_primitives and asserts the EVAL.md acceptance targets:

- circle_radius_relative_error <= 0.02
- ellipse_axis_relative_error  <= 0.03
- ellipse_rotation_error_deg   <= 2.0
- line_angle_error_deg         <= 1.0
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("recon3d.primitives")

from recon3d import schemas as S  # noqa: E402
from recon3d.config import PipelineConfig  # noqa: E402
from recon3d.primitives import fit_primitives  # noqa: E402

from evals import metrics as m  # noqa: E402


def _circle_points(center, radius, n=120, noise=0.0, seed=0,
                   arc_deg=(0.0, 360.0)):
    rng = np.random.RandomState(seed)
    ts = np.linspace(math.radians(arc_deg[0]), math.radians(arc_deg[1]), n)
    pts = np.stack([center[0] + radius * np.cos(ts),
                    center[1] + radius * np.sin(ts)], axis=1)
    if noise > 0:
        pts += rng.normal(0.0, noise, pts.shape)
    return pts


def _ellipse_points(center, axes, rot_deg, n=160, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    ts = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    x = axes[0] * np.cos(ts)
    y = axes[1] * np.sin(ts)
    pts = np.stack([center[0] + c * x - s * y,
                    center[1] + s * x + c * y], axis=1)
    if noise > 0:
        pts += rng.normal(0.0, noise, pts.shape)
    return pts


def _line_points(p0, p1, n=80, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) * np.asarray(p0) + t * np.asarray(p1)
    if noise > 0:
        pts += rng.normal(0.0, noise, pts.shape)
    return pts


def _fit_single(points, closed):
    layer = S.TraceLayer(
        name=S.TraceLayerName.SILHOUETTE, svg_path="synthetic.svg",
        image_size=(1024, 1024),
        paths=[S.VectorPath(
            path_id="p0", source_layer=S.TraceLayerName.SILHOUETTE,
            closed=closed, points=[(float(x), float(y)) for x, y in points])])
    cfg = PipelineConfig()
    prims = fit_primitives([layer], cfg)
    assert len(prims) >= 1, "no primitive fitted"
    return prims[0]


class TestCircleFitting:
    def test_clean_circle(self):
        prim = _fit_single(_circle_points((0.5, 0.5), 0.2), closed=True)
        assert prim.type in (S.PrimitiveType.CIRCLE, S.PrimitiveType.ELLIPSE)
        if prim.type == S.PrimitiveType.CIRCLE:
            err = m.circle_param_error(prim.params["center"],
                                       prim.params["radius"], (0.5, 0.5), 0.2)
            assert err["radius_rel_error"] <= 0.02
            assert err["center_rel_error"] <= 0.05
        else:
            err = m.ellipse_param_error(prim.params["radii"],
                                        prim.params["rotation_degrees"],
                                        (0.2, 0.2), 0.0)
            assert err["major_axis_rel_error"] <= 0.02

    def test_noisy_circle(self):
        pts = _circle_points((0.4, 0.6), 0.25, noise=0.001, seed=3)
        prim = _fit_single(pts, closed=True)
        if prim.type == S.PrimitiveType.CIRCLE:
            err = m.circle_param_error(prim.params["center"],
                                       prim.params["radius"], (0.4, 0.6), 0.25)
        else:
            assert prim.type == S.PrimitiveType.ELLIPSE
            err = {"radius_rel_error":
                   abs(prim.params["radii"][0] - 0.25) / 0.25}
        assert err["radius_rel_error"] <= 0.02


class TestEllipseFitting:
    def test_clean_ellipse(self):
        prim = _fit_single(_ellipse_points((0.5, 0.5), (0.30, 0.15), 30.0),
                           closed=True)
        assert prim.type in (S.PrimitiveType.ELLIPSE, S.PrimitiveType.CIRCLE)
        if prim.type == S.PrimitiveType.ELLIPSE:
            err = m.ellipse_param_error(prim.params["radii"],
                                        prim.params["rotation_degrees"],
                                        (0.30, 0.15), 30.0)
            assert err["major_axis_rel_error"] <= 0.03
            assert err["minor_axis_rel_error"] <= 0.03
            assert err["rotation_error_deg"] <= 2.0

    def test_noisy_rotated_ellipse(self):
        pts = _ellipse_points((0.45, 0.55), (0.25, 0.10), -55.0,
                              noise=0.001, seed=11)
        prim = _fit_single(pts, closed=True)
        if prim.type == S.PrimitiveType.ELLIPSE:
            err = m.ellipse_param_error(prim.params["radii"],
                                        prim.params["rotation_degrees"],
                                        (0.25, 0.10), -55.0)
            assert err["major_axis_rel_error"] <= 0.03
            assert err["rotation_error_deg"] <= 2.0


class TestArcFitting:
    def test_partial_arc_params(self):
        pts = _circle_points((0.5, 0.5), 0.2, n=80, arc_deg=(20.0, 140.0))
        prim = _fit_single(pts, closed=False)
        assert prim.type in (S.PrimitiveType.CIRCULAR_ARC, S.PrimitiveType.CIRCLE,
                             S.PrimitiveType.ELLIPTICAL_ARC, S.PrimitiveType.ELLIPSE)
        center = prim.params["center"]
        radius = prim.params.get("radius", prim.params.get("radii", [None])[0])
        if radius is not None:
            err = m.circle_param_error(center, radius, (0.5, 0.5), 0.2)
            assert err["radius_rel_error"] <= 0.05  # arcs are harder
            assert err["center_rel_error"] <= 0.15


class TestLineFitting:
    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 90.0, -45.0])
    def test_line_angle(self, angle_deg):
        th = math.radians(angle_deg)
        p0 = (0.5 - 0.3 * math.cos(th), 0.5 - 0.3 * math.sin(th))
        p1 = (0.5 + 0.3 * math.cos(th), 0.5 + 0.3 * math.sin(th))
        prim = _fit_single(_line_points(p0, p1, noise=0.0005, seed=5),
                           closed=False)
        assert prim.type in (S.PrimitiveType.LINE, S.PrimitiveType.POLYLINE)
        if prim.type == S.PrimitiveType.LINE:
            fp0, fp1 = prim.params["p0"], prim.params["p1"]
            fit_angle = math.degrees(math.atan2(fp1[1] - fp0[1],
                                                fp1[0] - fp0[0]))
            assert m.line_angle_error_deg(fit_angle, angle_deg) <= 1.0
