"""Minimal SVG path parsing / sampling utilities shared by vectorize + cleanup.

Supports M/L/H/V/C/S/Q/T/Z (absolute + relative) and ``transform`` attributes
of the form translate(tx,ty) / scale(s) / scale(sx,sy) / matrix(...) — the
subset emitted by vtracer and svgwrite. Béziers are sampled to polylines.

Coordinate convention at this stage is PIXELS (origin top-left, y down);
normalisation to 0..1 happens in ``svg_cleanup``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

Point = Tuple[float, float]

_TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtZzAa]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TRANSFORM_RE = re.compile(r"(translate|scale|matrix)\s*\(([^)]*)\)")

# samples per cubic/quadratic segment (before cleanup simplification)
_BEZIER_SAMPLES = 24


@dataclass
class Subpath:
    points: List[Point] = field(default_factory=list)
    closed: bool = False


def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, n: int) -> List[Point]:
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0]
        y = mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts


def _quad(p0: Point, p1: Point, p2: Point, n: int) -> List[Point]:
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def parse_path_d(d: str) -> List[Subpath]:
    """Parse an SVG ``d`` attribute into sampled polyline subpaths."""
    tokens = _TOKEN_RE.findall(d.replace(",", " "))
    i = 0
    cmd = ""
    cur: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    last_cubic_ctrl: Optional[Point] = None
    last_quad_ctrl: Optional[Point] = None
    subpaths: List[Subpath] = []
    cur_pts: List[Point] = []

    def is_cmd(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def num() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    def flush(closed: bool) -> None:
        nonlocal cur_pts
        if cur_pts:
            subpaths.append(Subpath(points=cur_pts, closed=closed))
        cur_pts = []

    n = len(tokens)
    while i < n:
        if is_cmd(tokens[i]):
            cmd = tokens[i]
            i += 1
        rel = cmd.islower()
        c = cmd.upper()

        if c == "M":
            flush(False)
            x, y = num(), num()
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = start = (x, y)
            cur_pts = [cur]
            cmd = "l" if rel else "L"  # implicit lineto after moveto
            last_cubic_ctrl = last_quad_ctrl = None
        elif c == "L":
            while i < n and not is_cmd(tokens[i]):
                x, y = num(), num()
                if rel:
                    x, y = cur[0] + x, cur[1] + y
                cur = (x, y)
                cur_pts.append(cur)
            last_cubic_ctrl = last_quad_ctrl = None
        elif c == "H":
            while i < n and not is_cmd(tokens[i]):
                x = num()
                if rel:
                    x += cur[0]
                cur = (x, cur[1])
                cur_pts.append(cur)
            last_cubic_ctrl = last_quad_ctrl = None
        elif c == "V":
            while i < n and not is_cmd(tokens[i]):
                y = num()
                if rel:
                    y += cur[1]
                cur = (cur[0], y)
                cur_pts.append(cur)
            last_cubic_ctrl = last_quad_ctrl = None
        elif c == "C":
            while i < n and not is_cmd(tokens[i]):
                x1, y1, x2, y2, x, y = num(), num(), num(), num(), num(), num()
                if rel:
                    x1, y1 = cur[0] + x1, cur[1] + y1
                    x2, y2 = cur[0] + x2, cur[1] + y2
                    x, y = cur[0] + x, cur[1] + y
                cur_pts.extend(_cubic(cur, (x1, y1), (x2, y2), (x, y), _BEZIER_SAMPLES))
                last_cubic_ctrl = (x2, y2)
                last_quad_ctrl = None
                cur = (x, y)
        elif c == "S":
            while i < n and not is_cmd(tokens[i]):
                x2, y2, x, y = num(), num(), num(), num()
                if rel:
                    x2, y2 = cur[0] + x2, cur[1] + y2
                    x, y = cur[0] + x, cur[1] + y
                if last_cubic_ctrl is not None:
                    x1, y1 = 2 * cur[0] - last_cubic_ctrl[0], 2 * cur[1] - last_cubic_ctrl[1]
                else:
                    x1, y1 = cur
                cur_pts.extend(_cubic(cur, (x1, y1), (x2, y2), (x, y), _BEZIER_SAMPLES))
                last_cubic_ctrl = (x2, y2)
                last_quad_ctrl = None
                cur = (x, y)
        elif c == "Q":
            while i < n and not is_cmd(tokens[i]):
                x1, y1, x, y = num(), num(), num(), num()
                if rel:
                    x1, y1 = cur[0] + x1, cur[1] + y1
                    x, y = cur[0] + x, cur[1] + y
                cur_pts.extend(_quad(cur, (x1, y1), (x, y), _BEZIER_SAMPLES))
                last_quad_ctrl = (x1, y1)
                last_cubic_ctrl = None
                cur = (x, y)
        elif c == "T":
            while i < n and not is_cmd(tokens[i]):
                x, y = num(), num()
                if rel:
                    x, y = cur[0] + x, cur[1] + y
                if last_quad_ctrl is not None:
                    x1, y1 = 2 * cur[0] - last_quad_ctrl[0], 2 * cur[1] - last_quad_ctrl[1]
                else:
                    x1, y1 = cur
                cur_pts.extend(_quad(cur, (x1, y1), (x, y), _BEZIER_SAMPLES))
                last_quad_ctrl = (x1, y1)
                last_cubic_ctrl = None
                cur = (x, y)
        elif c == "A":
            # Elliptical arcs are not emitted by our backends; approximate
            # with a straight line to the arc endpoint.
            while i < n and not is_cmd(tokens[i]):
                _rx, _ry, _rot, _laf, _sff = num(), num(), num(), num(), num()
                x, y = num(), num()
                if rel:
                    x, y = cur[0] + x, cur[1] + y
                cur = (x, y)
                cur_pts.append(cur)
            last_cubic_ctrl = last_quad_ctrl = None
        elif c == "Z":
            cur = start
            flush(True)
            last_cubic_ctrl = last_quad_ctrl = None
            cmd = ""
        else:
            # unknown command: skip token to avoid an infinite loop
            i += 1
    flush(False)
    return subpaths


def _parse_transform(transform: Optional[str]):
    """Return an affine (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f."""
    a, b, c_, d_, e, f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    if not transform:
        return a, b, c_, d_, e, f
    for name, argstr in _TRANSFORM_RE.findall(transform):
        args = [float(v) for v in re.split(r"[\s,]+", argstr.strip()) if v]
        if name == "translate":
            ta, tb, tc, td, te, tf = 1.0, 0.0, 0.0, 1.0, args[0], (args[1] if len(args) > 1 else 0.0)
        elif name == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            ta, tb, tc, td, te, tf = sx, 0.0, 0.0, sy, 0.0, 0.0
        else:  # matrix
            ta, tb, tc, td, te, tf = (args + [0.0] * 6)[:6]
        # compose: new = current * t  (transforms apply right-to-left in SVG,
        # but vtracer/svgwrite emit a single transform, so plain compose is fine)
        na = a * ta + c_ * tb
        nb = b * ta + d_ * tb
        nc = a * tc + c_ * td
        nd = b * tc + d_ * td
        ne = a * te + c_ * tf + e
        nf = b * te + d_ * tf + f
        a, b, c_, d_, e, f = na, nb, nc, nd, ne, nf
    return a, b, c_, d_, e, f


def _apply(m, p: Point) -> Point:
    a, b, c_, d_, e, f = m
    return (a * p[0] + c_ * p[1] + e, b * p[0] + d_ * p[1] + f)


def iter_svg_subpaths(svg_text: str) -> List[Subpath]:
    """Parse a full SVG document into transformed, sampled subpaths.

    Each ``M`` inside a ``d`` starts a new Subpath, so holes packed into the
    same path element come out as separate subpaths (containment is resolved
    later by svg_cleanup).
    """
    # strip namespace for simpler tag matching
    root = ET.fromstring(svg_text)
    out: List[Subpath] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "path":
            continue
        d = elem.get("d")
        if not d:
            continue
        m = _parse_transform(elem.get("transform"))
        for sp in parse_path_d(d):
            if m != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
                sp = Subpath(points=[_apply(m, p) for p in sp.points], closed=sp.closed)
            out.append(sp)
    return out


def parse_svg_file(path: str | Path) -> List[Subpath]:
    return iter_svg_subpaths(Path(path).read_text())


def points_to_d(points: List[Point], closed: bool, precision: int = 5) -> str:
    """Serialise a polyline back into a compact ``d`` attribute string."""
    if not points:
        return ""
    fmt = "{:.%df}" % precision

    def f(v: float) -> str:
        s = fmt.format(v).rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"

    parts = ["M%s %s" % (f(points[0][0]), f(points[0][1]))]
    parts.extend("L%s %s" % (f(x), f(y)) for x, y in points[1:])
    if closed:
        parts.append("Z")
    return " ".join(parts)


def shoelace_area(points: List[Point]) -> float:
    """Signed polygon area (positive = CCW in standard orientation)."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def polyline_length(points: List[Point], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    if closed:
        dx = points[0][0] - points[-1][0]
        dy = points[0][1] - points[-1][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total
