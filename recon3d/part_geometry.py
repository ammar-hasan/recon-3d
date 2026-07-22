"""Internal geometry helpers shared by stages 11-14 and material estimation.

Not a pipeline stage module; only used by camera/depth/operators/materials/
construction_plan. All coordinates are normalised crop coordinates (0..1,
origin top-left, y down) unless stated otherwise.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .schemas import GeometricPrimitive, PrimitiveType, SemanticPart, SketchGraph

ELLIPSE_LIKE = {
    PrimitiveType.ELLIPSE,
    PrimitiveType.CIRCLE,
    PrimitiveType.ELLIPTICAL_ARC,
    PrimitiveType.CIRCULAR_ARC,
}
CLOSED_FLAT = {
    PrimitiveType.RECTANGLE,
    PrimitiveType.ROUNDED_RECTANGLE,
    PrimitiveType.REGULAR_POLYGON,
    PrimitiveType.CLOSED_REGION,
}
OPEN_PATH = {
    PrimitiveType.LINE,
    PrimitiveType.POLYLINE,
    PrimitiveType.BEZIER,
    PrimitiveType.SYMMETRIC_SPLINE,
}


def points_of(prim: GeometricPrimitive) -> List[Tuple[float, float]]:
    pts = prim.params.get("points")
    if pts:
        return [(float(p[0]), float(p[1])) for p in pts]
    if prim.fallback_points:
        return [(float(p[0]), float(p[1])) for p in prim.fallback_points]
    return []


def primitive_center(prim: GeometricPrimitive) -> Tuple[float, float]:
    c = prim.params.get("center")
    if c is not None:
        return float(c[0]), float(c[1])
    pts = points_of(prim)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.5, 0.5


def primitive_radii(prim: GeometricPrimitive) -> Optional[Tuple[float, float]]:
    """Return (rx, ry) in normalised units for ellipse/circle primitives."""
    r = prim.params.get("radii")
    if r is not None:
        return abs(float(r[0])), abs(float(r[1]))
    r = prim.params.get("radius")
    if r is not None:
        return abs(float(r)), abs(float(r))
    return None


def primitive_rotation_deg(prim: GeometricPrimitive) -> float:
    return float(prim.params.get("rotation_degrees", 0.0) or 0.0)


def primitive_bbox(prim: GeometricPrimitive) -> Tuple[float, float, float, float]:
    rad = primitive_radii(prim)
    if rad is not None:
        cx, cy = primitive_center(prim)
        return cx - rad[0], cy - rad[1], cx + rad[0], cy + rad[1]
    pts = points_of(prim)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)
    c = primitive_center(prim)
    return c[0], c[1], c[0], c[1]


def part_primitives(graph: SketchGraph, part: SemanticPart) -> List[GeometricPrimitive]:
    by_id = {p.id: p for p in graph.primitives}
    return [by_id[i] for i in part.primitive_ids if i in by_id]


def part_bbox(graph: SketchGraph, part: SemanticPart) -> Tuple[float, float, float, float]:
    prims = part_primitives(graph, part)
    if not prims:
        return 0.0, 0.0, 1.0, 1.0
    boxes = [primitive_bbox(p) for p in prims]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def graph_bbox(graph: SketchGraph) -> Tuple[float, float, float, float]:
    """Outermost bbox over all primitives; degenerate fallback = full canvas."""
    if not graph.primitives:
        return 0.0, 0.0, 1.0, 1.0
    boxes = [primitive_bbox(p) for p in graph.primitives]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return 0.0, 0.0, 1.0, 1.0
    return x0, y0, x1, y1


class ObjectFrame:
    """Maps normalised image coords (origin top-left, y down) to object units
    (origin at object centre, x right, y up; object width = 1.0).

    The same scale is used for x and y so aspect is preserved.
    """

    def __init__(self, bbox: Tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = bbox
        self.cx = (x0 + x1) / 2.0
        self.cy = (y0 + y1) / 2.0
        self.width = max(x1 - x0, 1e-9)

    def point(self, u: float, v: float) -> Tuple[float, float]:
        return (u - self.cx) / self.width, (self.cy - v) / self.width

    def length(self, norm_len: float) -> float:
        return norm_len / self.width


def ellipse_outline(
    prim: GeometricPrimitive, n: int = 24
) -> List[Tuple[float, float]]:
    """Polyline approximation of an ellipse/circle primitive (normalised)."""
    rad = primitive_radii(prim)
    if rad is None:
        return points_of(prim)
    cx, cy = primitive_center(prim)
    rot = math.radians(primitive_rotation_deg(prim))
    cr, sr = math.cos(rot), math.sin(rot)
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        ex, ey = rad[0] * math.cos(t), rad[1] * math.sin(t)
        pts.append((cx + ex * cr - ey * sr, cy + ex * sr + ey * cr))
    return pts


def outline_of(prim: GeometricPrimitive, n: int = 24) -> List[Tuple[float, float]]:
    """Best closed-outline polyline for a primitive (normalised coords)."""
    if prim.type in ELLIPSE_LIKE:
        return ellipse_outline(prim, n)
    return points_of(prim)


def rasterize_primitives(
    prims: Sequence[GeometricPrimitive], shape: Tuple[int, int]
) -> np.ndarray:
    """Rasterise primitive fills into a uint8 mask (255 = inside).

    ``shape`` is (height, width) in pixels. Open paths are drawn as thick
    polylines; closed shapes are filled.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    def px(pt: Tuple[float, float]) -> Tuple[int, int]:
        return int(round(pt[0] * w)), int(round(pt[1] * h))

    for prim in prims:
        rad = primitive_radii(prim)
        if rad is not None:
            cx, cy = primitive_center(prim)
            cv2.ellipse(
                mask,
                px((cx, cy)),
                (max(1, int(round(rad[0] * w))), max(1, int(round(rad[1] * h)))),
                primitive_rotation_deg(prim),
                0,
                360,
                255,
                -1,
            )
            continue
        pts = points_of(prim)
        if not pts:
            continue
        arr = np.array([px(p) for p in pts], dtype=np.int32).reshape(-1, 1, 2)
        if prim.type in CLOSED_FLAT and len(pts) >= 3:
            cv2.fillPoly(mask, [arr], 255)
        elif len(pts) >= 2:
            thickness = max(2, int(round(0.004 * w)))
            cv2.polylines(mask, [arr], False, 255, thickness)
        else:
            cv2.circle(mask, px(pts[0]), 2, 255, -1)
    return mask
