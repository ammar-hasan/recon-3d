"""Stage 8: constraint and relationship detection.

Detects geometric relationships between fitted primitives.  Tolerances are
relative to the overall object scale (bounding-box diagonal of all primitive
curves).  Precision is prioritised over recall: only constraints with
confidence >= 0.6 are returned, and confidences are derived from how far the
measured residual sits inside its tolerance band.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .schemas import (
    ConstraintType,
    GeometricConstraint,
    GeometricPrimitive,
    PrimitiveType,
)

_ROUND = 6
_CIRCLE_TYPES = {PrimitiveType.CIRCLE, PrimitiveType.CIRCULAR_ARC}
_ELLIPSE_TYPES = {PrimitiveType.ELLIPSE, PrimitiveType.ELLIPTICAL_ARC}
_CLOSED_TYPES = {
    PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE, PrimitiveType.RECTANGLE,
    PrimitiveType.ROUNDED_RECTANGLE, PrimitiveType.REGULAR_POLYGON,
    PrimitiveType.CLOSED_REGION, PrimitiveType.SYMMETRIC_SPLINE,
}
_CENTERED_TYPES = _CLOSED_TYPES | _CIRCLE_TYPES | _ELLIPSE_TYPES
_ORIENTED_TYPES = {
    PrimitiveType.LINE, PrimitiveType.ELLIPSE, PrimitiveType.ELLIPTICAL_ARC,
    PrimitiveType.RECTANGLE, PrimitiveType.ROUNDED_RECTANGLE,
    PrimitiveType.SYMMETRIC_SPLINE,
}


# ---------------------------------------------------------------------------
# per-primitive geometry accessors
# ---------------------------------------------------------------------------

def _center(p: GeometricPrimitive) -> Optional[np.ndarray]:
    prm = p.params
    if "center" in prm:
        return np.asarray(prm["center"], dtype=float)
    if p.type == PrimitiveType.LINE and "p0" in prm:
        return (np.asarray(prm["p0"]) + np.asarray(prm["p1"])) / 2.0
    if p.fallback_points:
        return np.asarray(p.fallback_points, dtype=float).mean(axis=0)
    return None


def _radius(p: GeometricPrimitive) -> Optional[float]:
    if "radius" in p.params:
        return float(p.params["radius"])
    if "radii" in p.params:
        return float(np.mean(p.params["radii"]))
    return None


def _direction_deg(p: GeometricPrimitive) -> Optional[float]:
    """Orientation angle (mod 180) for line-like / elongated primitives."""
    prm = p.params
    if p.type == PrimitiveType.LINE and "p0" in prm:
        d = np.asarray(prm["p1"], dtype=float) - np.asarray(prm["p0"], dtype=float)
        return math.degrees(math.atan2(d[1], d[0])) % 180.0
    if "rotation_degrees" in prm and p.type in _ORIENTED_TYPES:
        return float(prm["rotation_degrees"]) % 180.0
    if p.type == PrimitiveType.SYMMETRIC_SPLINE and "axis" in prm:
        return float(prm["axis"]["angle_degrees"]) % 180.0
    return None


def _length(p: GeometricPrimitive) -> Optional[float]:
    if p.type == PrimitiveType.LINE and "p0" in p.params:
        d = np.asarray(p.params["p1"], float) - np.asarray(p.params["p0"], float)
        return float(np.hypot(d[0], d[1]))
    if p.type == PrimitiveType.POLYLINE and "points" in p.params:
        pts = np.asarray(p.params["points"], float)
        if len(pts) >= 2:
            return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    return None


def _poly(p: GeometricPrimitive, max_pts: int = 96) -> np.ndarray:
    pts = np.asarray(p.fallback_points, dtype=float)
    if len(pts) > max_pts:
        idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[idx]
    return pts


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    """2D scalar cross product (np.cross on 2D vectors is deprecated)."""
    return float(u[0] * v[1] - u[1] * v[0])


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two orientations (mod 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _conf(residual: float, tol: float) -> float:
    return round(float(max(0.6, min(0.99, 1.0 - residual / max(tol, 1e-12)))), 4)


def _make(ctype: ConstraintType, entities: List[str],
          params: Optional[Dict[str, Any]] = None,
          confidence: float = 0.9) -> GeometricConstraint:
    return GeometricConstraint(type=ctype, entities=list(entities),
                               params=params or {}, confidence=confidence)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

def _group_center_sharing(prims: List[GeometricPrimitive],
                          centers: Dict[str, np.ndarray], tol: float,
                          ) -> List[Tuple[List[str], float, np.ndarray]]:
    """Union-find groups of centred primitives whose centres lie within tol."""
    centered = [p for p in prims if p.type in _CENTERED_TYPES and p.id in centers]
    uf = _UnionFind()
    worst: Dict[str, float] = {}
    for i in range(len(centered)):
        for j in range(i + 1, len(centered)):
            d = float(np.hypot(*(centers[centered[i].id] - centers[centered[j].id])))
            if d <= tol:
                uf.union(centered[i].id, centered[j].id)
                root = uf.find(centered[i].id)
                worst[root] = max(worst.get(root, 0.0), d)
    groups: Dict[str, List[str]] = {}
    for p in centered:
        root = uf.find(p.id)
        groups.setdefault(root, []).append(p.id)
    out = []
    for ids in groups.values():
        if len(ids) >= 2:
            c = np.mean([centers[i] for i in ids], axis=0)
            out.append((sorted(ids), worst.get(ids[0], 0.0), c))
    return out


def _detect_concentric(prims, centers, scale, out):
    tol = 0.02 * scale
    sub = [p for p in prims if p.type in (_CIRCLE_TYPES | _ELLIPSE_TYPES)]
    groups = _group_center_sharing(sub, centers, tol)
    for ids, w, c in groups:
        out.append(_make(ConstraintType.CONCENTRIC, ids,
                         {"center": [round(float(c[0]), _ROUND), round(float(c[1]), _ROUND)]},
                         _conf(w, tol)))
        full = [i for i in ids
                if next(p for p in prims if p.id == i).type in
                (PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE)]
        if len(full) >= 2:
            out.append(_make(ConstraintType.RADIAL_SYMMETRY, full,
                             {"center": [round(float(c[0]), _ROUND),
                                         round(float(c[1]), _ROUND)]},
                             _conf(w, tol)))


def _detect_shared_center(prims, centers, scale, out):
    tol = 0.02 * scale
    groups = _group_center_sharing(prims, centers, tol)
    for ids, w, c in groups:
        types = {next(p for p in prims if p.id == i).type for i in ids}
        if types - (_CIRCLE_TYPES | _ELLIPSE_TYPES):
            out.append(_make(ConstraintType.SHARED_CENTER, ids,
                             {"center": [round(float(c[0]), _ROUND),
                                         round(float(c[1]), _ROUND)]},
                             _conf(w, tol)))


def _detect_coincident(prims, centers, scale, out):
    tol = 0.01 * scale
    for i in range(len(prims)):
        for j in range(i + 1, len(prims)):
            a, b = prims[i], prims[j]
            if a.type != b.type or a.id not in centers or b.id not in centers:
                continue
            if float(np.hypot(*(centers[a.id] - centers[b.id]))) > tol:
                continue
            ra, rb = _radius(a), _radius(b)
            if ra is not None and rb is not None:
                if abs(ra - rb) > 0.01 * max(ra, rb):
                    continue
            la, lb = _length(a), _length(b)
            if la is not None and lb is not None:
                if abs(la - lb) > 0.01 * max(la, lb):
                    continue
            if ra is None and la is None:
                continue
            out.append(_make(ConstraintType.COINCIDENT, [a.id, b.id], {}, 0.9))


def _detect_line_angular(prims, centers, scale, out):
    """parallel / perpendicular / collinear between line primitives."""
    ang_tol = 3.0
    dist_tol = 0.01 * scale
    lines = [p for p in prims if p.type == PrimitiveType.LINE]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i], lines[j]
            da, db = _direction_deg(a), _direction_deg(b)
            if da is None or db is None:
                continue
            dd = _angle_diff(da, db)
            if dd <= ang_tol:
                p0 = np.asarray(a.params["p0"], float)
                q0 = np.asarray(b.params["p0"], float)
                q1 = np.asarray(b.params["p1"], float)
                v = q1 - q0
                L = np.hypot(v[0], v[1])
                dist = abs(_cross2(v, p0 - q0)) / max(L, 1e-12)
                if dist <= dist_tol:
                    out.append(_make(ConstraintType.COLLINEAR, [a.id, b.id],
                                     {"angle_degrees": round(da, 3)},
                                     _conf(dist, dist_tol)))
                else:
                    out.append(_make(ConstraintType.PARALLEL, [a.id, b.id],
                                     {"angle_degrees": round(da, 3)},
                                     _conf(dd, ang_tol)))
            elif abs(dd - 90.0) <= ang_tol:
                out.append(_make(ConstraintType.PERPENDICULAR, [a.id, b.id],
                                 {"angle_degrees": round((da + 90.0) % 180.0, 3)},
                                 _conf(abs(dd - 90.0), ang_tol)))


def _detect_equal_size(prims, out):
    circles = [(p, _radius(p)) for p in prims if p.type in _CIRCLE_TYPES]
    circles = [(p, r) for p, r in circles if r is not None]
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            (pa, ra), (pb, rb) = circles[i], circles[j]
            rel = abs(ra - rb) / max(ra, rb)
            if rel <= 0.02:
                out.append(_make(ConstraintType.EQUAL_RADIUS, [pa.id, pb.id],
                                 {"radius": round((ra + rb) / 2.0, _ROUND)},
                                 _conf(rel, 0.02)))
    lens = [(p, _length(p)) for p in prims]
    lens = [(p, L) for p, L in lens if L is not None and L > 1e-6]
    for i in range(len(lens)):
        for j in range(i + 1, len(lens)):
            (pa, la), (pb, lb) = lens[i], lens[j]
            rel = abs(la - lb) / max(la, lb)
            if rel <= 0.02:
                out.append(_make(ConstraintType.EQUAL_LENGTH, [pa.id, pb.id],
                                 {"length": round((la + lb) / 2.0, _ROUND)},
                                 _conf(rel, 0.02)))


def _detect_tangent(prims, centers, scale, out):
    tol = 0.02 * scale
    circles = [p for p in prims if p.type in _CIRCLE_TYPES and _radius(p)]
    lines = [p for p in prims if p.type == PrimitiveType.LINE]
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            a, b = circles[i], circles[j]
            ra, rb = _radius(a), _radius(b)
            d = float(np.hypot(*(centers[a.id] - centers[b.id])))
            if d <= 0.02 * scale:
                continue  # concentric, not tangent
            res = min(abs(d - (ra + rb)), abs(d - abs(ra - rb)))
            if res <= tol:
                kind = "external" if abs(d - (ra + rb)) < abs(d - abs(ra - rb)) else "internal"
                out.append(_make(ConstraintType.TANGENT, [a.id, b.id],
                                 {"kind": kind}, _conf(res, tol)))
    for c in circles:
        r = _radius(c)
        cc = centers[c.id]
        for ln in lines:
            q0 = np.asarray(ln.params["p0"], float)
            q1 = np.asarray(ln.params["p1"], float)
            v = q1 - q0
            L = np.hypot(v[0], v[1])
            if L < 1e-12:
                continue
            dist = abs(_cross2(v, cc - q0)) / L
            res = abs(dist - r)
            if res <= tol:
                out.append(_make(ConstraintType.TANGENT, [c.id, ln.id],
                                 {"kind": "circle_line"}, _conf(res, tol)))


def _detect_containment(prims, scale, out):
    tol = 0.002 * scale
    closed = [p for p in prims if p.type in _CLOSED_TYPES and len(p.fallback_points) >= 3]
    polys = {p.id: _poly(p) for p in closed}
    areas = {pid: abs(cv2.contourArea((pl * 4096.0).astype(np.float32)))
             for pid, pl in polys.items()}
    for big in closed:
        for small in closed:
            if big.id == small.id or areas[big.id] <= areas[small.id]:
                continue
            cnt = (polys[big.id] * 4096.0).astype(np.float32)
            pts = polys[small.id]
            ok = True
            for pt in pts[:: max(1, len(pts) // 24)]:
                d = cv2.pointPolygonTest(cnt, (float(pt[0] * 4096.0),
                                               float(pt[1] * 4096.0)), True)
                if d < -tol * 4096.0:
                    ok = False
                    break
            if ok:
                out.append(_make(ConstraintType.CONTAINMENT, [big.id, small.id],
                                 {}, 0.9))


def _segments_intersect(a1, a2, b1, b2) -> bool:
    def ccw(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1 = ccw(b1, b2, a1)
    d2 = ccw(b1, b2, a2)
    d3 = ccw(a1, a2, b1)
    d4 = ccw(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _polylines_cross(P: np.ndarray, Q: np.ndarray) -> bool:
    for i in range(len(P) - 1):
        for j in range(len(Q) - 1):
            if _segments_intersect(P[i], P[i + 1], Q[j], Q[j + 1]):
                return True
    return False


def _detect_spatial(prims, scale, out, containment_pairs):
    """intersection / adjacency between sampled polylines."""
    tol = 0.02 * scale
    polys = {p.id: _poly(p) for p in prims if len(p.fallback_points) >= 2}
    ids = list(polys)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (a, b) in containment_pairs or (b, a) in containment_pairs:
                continue
            P, Q = polys[a], polys[b]
            # cheap bbox reject
            if (P.min(axis=0) > Q.max(axis=0) + tol).any() or \
               (Q.min(axis=0) > P.max(axis=0) + tol).any():
                continue
            try:
                from scipy.spatial import cKDTree
                dmin = float(cKDTree(P).query(Q, k=1)[0].min())
            except Exception:
                dmin = float(np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min())
            if dmin > tol:
                continue
            if _polylines_cross(P, Q):
                out.append(_make(ConstraintType.INTERSECTION, [a, b], {},
                                 _conf(0.0, tol)))
            else:
                out.append(_make(ConstraintType.ADJACENCY, [a, b],
                                 {"min_distance": round(dmin, _ROUND)},
                                 _conf(dmin, tol)))


def _detect_alignment(prims, centers, scale, out):
    tol = 0.01 * scale
    items = [(p.id, centers[p.id]) for p in prims if p.id in centers]
    seen = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ia, ca), (ib, cb) = items[i], items[j]
            v = cb - ca
            L = float(np.hypot(v[0], v[1]))
            if L < 1e-9:
                continue
            group = []
            worst = 0.0
            for k, (ik, ck) in enumerate(items):
                d = abs(_cross2(v, ck - ca)) / L
                if d <= tol:
                    group.append((float(np.dot(ck - ca, v) / L), ik))
                    worst = max(worst, d)
            if len(group) >= 3:
                group.sort()
                key = tuple(sorted(g[1] for g in group))
                if key in seen:
                    continue
                seen.add(key)
                ids = [g[1] for g in group]
                angle = math.degrees(math.atan2(v[1], v[0])) % 180.0
                out.append(_make(ConstraintType.ALIGNMENT, ids,
                                 {"angle_degrees": round(angle, 3)},
                                 _conf(worst, tol)))
                gaps = np.diff([g[0] for g in group]) * L
                if len(gaps) >= 2 and gaps.mean() > 1e-6:
                    cv = float(gaps.std() / gaps.mean())
                    if cv <= 0.05:
                        out.append(_make(ConstraintType.EQUAL_SPACING, ids,
                                         {"spacing": round(float(gaps.mean()), _ROUND)},
                                         _conf(cv, 0.05)))


def _detect_rotational_repetition(prims, centers, scale, out):
    hub_types = (PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE)
    hubs = [p for p in prims if p.type in hub_types and p.id in centers]
    seen = set()
    for hub in hubs:
        hc = centers[hub.id]
        sats = []
        for p in prims:
            if p.id == hub.id or p.id not in centers:
                continue
            r = float(np.hypot(*(centers[p.id] - hc)))
            if r <= 0.05 * scale:
                continue
            sats.append((p, r))
        by_type: Dict[PrimitiveType, List[Tuple[GeometricPrimitive, float]]] = {}
        for p, r in sats:
            by_type.setdefault(p.type, []).append((p, r))
        for t, group in by_type.items():
            group.sort(key=lambda pr: pr[1])
            # split into radius clusters (5 % band)
            clusters: List[List[Tuple[GeometricPrimitive, float]]] = []
            for p, r in group:
                if clusters and abs(r - np.mean([g[1] for g in clusters[-1]])) \
                        <= 0.05 * max(r, 1e-9):
                    clusters[-1].append((p, r))
                else:
                    clusters.append([(p, r)])
            for cl in clusters:
                n = len(cl)
                if n < 3:
                    continue
                angs = sorted(math.degrees(math.atan2(centers[p.id][1] - hc[1],
                                                      centers[p.id][0] - hc[0])) % 360.0
                              for p, _ in cl)
                gaps = np.diff(angs + [angs[0] + 360.0])
                expected = 360.0 / n
                dev = float(np.abs(gaps - expected).max())
                if dev <= 0.1 * expected:
                    ids = [p.id for p, _ in cl]
                    key = frozenset(ids)
                    if key in seen:
                        continue
                    seen.add(key)
                    proto = ids[int(np.argmin(angs))]
                    out.append(_make(
                        ConstraintType.ROTATIONAL_REPETITION, ids,
                        {"prototype": proto,
                         "count": n,
                         "angle_degrees": round(expected, 3),
                         "center": [round(float(hc[0]), _ROUND),
                                    round(float(hc[1]), _ROUND)],
                         "radius": round(float(np.mean([g[1] for g in cl])), _ROUND)},
                        _conf(dev, 0.1 * expected)))


def _detect_mirror_symmetry(prims, centers, scale, out):
    match_tol = 0.02 * scale
    on_axis_tol = 0.01 * scale
    # candidate axis points: overall centroid + circle/ellipse centres
    pts = [c for c in centers.values()]
    if not pts:
        return
    candidates = [np.mean(pts, axis=0)]
    for p in prims:
        if p.type in (PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE) and p.id in centers:
            candidates.append(centers[p.id])
    dirs = {"vertical": 90.0, "horizontal": 0.0, "diag_pos": 45.0, "diag_neg": 135.0}
    best: Dict[Tuple[str, int], Tuple[float, List[str], np.ndarray, float]] = {}
    for cp in candidates:
        for name, ang in dirs.items():
            a = math.radians(ang)
            u = np.array([math.cos(a), math.sin(a)])
            n = np.array([-u[1], u[0]])
            matched: List[str] = []
            worst = 0.0
            ok = True
            used: set = set()
            for p in prims:
                if p.id not in centers:
                    continue
                c = centers[p.id]
                off = float(np.dot(c - cp, n))
                if abs(off) <= on_axis_tol:
                    matched.append(p.id)
                    continue
                cr = c - 2.0 * off * n
                found = None
                for q in prims:
                    if q.id == p.id or q.id in used or q.id not in centers:
                        continue
                    if q.type != p.type:
                        continue
                    dq = float(np.hypot(*(centers[q.id] - cr)))
                    if dq > match_tol:
                        continue
                    # orientation must mirror too (for oriented types)
                    dpa, dqa = _direction_deg(p), _direction_deg(q)
                    if dpa is not None and dqa is not None:
                        expect = (2.0 * ang - dpa) % 180.0
                        if _angle_diff(expect, dqa) > 5.0:
                            continue
                    found = (q, dq)
                    break
                if found is None:
                    ok = False
                    break
                used.add(found[0].id)
                matched.extend([p.id, found[0].id])
                worst = max(worst, found[1])
            if ok and len(matched) >= 2:
                key = (name, int(round(float(np.dot(cp, n)) / max(match_tol, 1e-9))))
                conf = _conf(worst, match_tol)
                cur = best.get(key)
                if cur is None or conf > cur[0]:
                    best[key] = (conf, sorted(set(matched)), cp, ang)
    for conf, ids, cp, ang in best.values():
        out.append(_make(ConstraintType.MIRROR_SYMMETRY, ids,
                         {"axis": {"point": [round(float(cp[0]), _ROUND),
                                             round(float(cp[1]), _ROUND)],
                                   "angle_degrees": ang}},
                         conf))


def _detect_shared_axis(prims, centers, scale, out):
    tol = 0.01 * scale
    ang_tol = 3.0
    oriented = [p for p in prims if p.type in
                (PrimitiveType.ELLIPSE, PrimitiveType.ELLIPTICAL_ARC,
                 PrimitiveType.RECTANGLE, PrimitiveType.ROUNDED_RECTANGLE)
                and p.id in centers and _direction_deg(p) is not None]
    for i in range(len(oriented)):
        for j in range(i + 1, len(oriented)):
            a, b = oriented[i], oriented[j]
            da, db = _direction_deg(a), _direction_deg(b)
            if _angle_diff(da, db) > ang_tol:
                continue
            th = math.radians((da + db) / 2.0)
            n = np.array([-math.sin(th), math.cos(th)])
            d = abs(float(np.dot(centers[a.id] - centers[b.id], n)))
            if d <= tol:
                out.append(_make(ConstraintType.SHARED_AXIS, [a.id, b.id],
                                 {"angle_degrees": round((da + db) / 2.0 % 180.0, 3)},
                                 _conf(d, tol)))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def detect_constraints(primitives: List[GeometricPrimitive],
                       cfg: PipelineConfig) -> List[GeometricConstraint]:
    """Detect all supported geometric constraints between primitives."""
    prims = list(primitives)
    centers: Dict[str, np.ndarray] = {}
    all_pts = []
    for p in prims:
        c = _center(p)
        if c is not None:
            centers[p.id] = c
        if p.fallback_points:
            all_pts.append(np.asarray(p.fallback_points, dtype=float))
    if all_pts:
        cat = np.concatenate(all_pts)
        scale = float(np.hypot(*(cat.max(axis=0) - cat.min(axis=0))))
    else:
        scale = 1.0
    if scale < 1e-9:
        scale = 1.0

    out: List[GeometricConstraint] = []
    _detect_concentric(prims, centers, scale, out)
    _detect_shared_center(prims, centers, scale, out)
    _detect_coincident(prims, centers, scale, out)
    _detect_line_angular(prims, centers, scale, out)
    _detect_equal_size(prims, out)
    _detect_tangent(prims, centers, scale, out)

    before = len(out)
    _detect_containment(prims, scale, out)
    containment_pairs = set()
    for c in out[before:]:
        if c.type == ConstraintType.CONTAINMENT:
            containment_pairs.add((c.entities[0], c.entities[1]))

    _detect_spatial(prims, scale, out, containment_pairs)
    _detect_alignment(prims, centers, scale, out)
    _detect_rotational_repetition(prims, centers, scale, out)
    _detect_mirror_symmetry(prims, centers, scale, out)
    _detect_shared_axis(prims, centers, scale, out)

    # dedupe + confidence filter (precision over recall)
    seen = set()
    result: List[GeometricConstraint] = []
    for c in out:
        key = (c.type.value, tuple(sorted(c.entities)))
        if key in seen:
            continue
        seen.add(key)
        if c.confidence >= 0.6:
            result.append(c)
    return result
