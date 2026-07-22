"""Stage 10: semantic part decomposition.

Groups sketch-graph primitives into labelled, hierarchical ``SemanticPart``s
with appearance estimates and explicitly-marked inferred hidden geometry.

The grouping logic lives behind the ``SemanticBackend`` protocol so a future
vision-language-model backend can replace the default heuristic one without
touching the pipeline: ``decompose_parts`` only talks to the backend
interface.  The heuristic backend is deterministic and never modifies
primitive geometry — it only labels, groups and proposes hypotheses.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .schemas import (
    AppearanceEstimate,
    ConstraintType,
    EvidenceSource,
    EvidencedValue,
    GeometricConstraint,
    GeometricPrimitive,
    InputSpec,
    PrimitiveType,
    SemanticPart,
    SketchGraph,
    TraceLayerName,
    Visibility,
)

_CLOSED_TYPES = {
    PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE, PrimitiveType.RECTANGLE,
    PrimitiveType.ROUNDED_RECTANGLE, PrimitiveType.REGULAR_POLYGON,
    PrimitiveType.CLOSED_REGION, PrimitiveType.SYMMETRIC_SPLINE,
}
_RING_TYPES = {PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE}
_KEYWORDS = ("wheel", "bottle", "cup", "mug", "vase", "knob", "gear",
             "lamp", "chair", "table", "box", "crate", "pipe_elbow",
             "pipe", "sign", "logo", "bracket")

_ROOT_CLASS_BY_KEYWORD = {
    "bottle": "bottle_body", "box": "enclosure_body",
    "bracket": "bracket_body", "gear": "gear_body",
    "knob": "knob_body", "mug": "mug_body", "cup": "cup_body",
    "vase": "vase_body", "sign": "plate", "logo": "plate",
    "pipe_elbow": "pipe", "pipe": "pipe", "crate": "bottom_panel",
    "lamp": "base", "chair": "seat", "table": "tabletop",
}

_GUIDED_ROLES = {
    "bottle": ("cap",), "box": ("lid",),
    "bracket": ("mounting_hole",), "gear": ("tooth", "center_bore"),
    "mug": ("handle",), "cup": ("handle",),
    "sign": ("logo_relief",), "logo": ("logo_relief",),
    "table": ("leg",),
    "pipe_elbow": ("flange", "flange"), "pipe": ("flange", "flange"),
    "chair": ("backrest", "leg"),
    "crate": ("corner_post", "side_slat"),
    "lamp": ("lower_arm", "upper_arm", "shade"),
}


# ---------------------------------------------------------------------------
# backend protocol + heuristic default
# ---------------------------------------------------------------------------

class SemanticBackend:
    """Pluggable semantic-grouping interface (heuristic or future VLM)."""

    name = "base"

    def decompose(self, graph: SketchGraph, image: Optional[np.ndarray],
                  spec: InputSpec, cfg: PipelineConfig) -> List[SemanticPart]:
        raise NotImplementedError


def _poly(p: GeometricPrimitive, max_pts: int = 96) -> np.ndarray:
    pts = np.asarray(p.fallback_points, dtype=float)
    if len(pts) > max_pts:
        idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[idx]
    return pts


def _center(p: GeometricPrimitive) -> Optional[np.ndarray]:
    if "center" in p.params:
        return np.asarray(p.params["center"], dtype=float)
    if p.fallback_points:
        return np.asarray(p.fallback_points, dtype=float).mean(axis=0)
    return None


def _radius(p: GeometricPrimitive) -> float:
    if "radius" in p.params:
        return float(p.params["radius"])
    if "radii" in p.params:
        return float(max(p.params["radii"]))
    if "size" in p.params:
        return float(max(p.params["size"])) / 2.0
    pts = _poly(p)
    c = _center(p)
    if c is None or len(pts) == 0:
        return 0.0
    return float(np.linalg.norm(pts - c, axis=1).max())


def _inside(container: np.ndarray, points: np.ndarray, tol_px: float) -> bool:
    cnt = (container * 4096.0).astype(np.float32)
    for pt in points[:: max(1, len(points) // 24)]:
        d = cv2.pointPolygonTest(cnt, (float(pt[0] * 4096.0),
                                       float(pt[1] * 4096.0)), True)
        if d < -tol_px:
            return False
    return True


class HeuristicSemanticBackend(SemanticBackend):
    """Containment-tree + radial-structure heuristics for hard-surface objects."""

    name = "heuristic"

    def __init__(self) -> None:
        self._class_counters: Dict[str, int] = {}
        self.generated_constraints: List[GeometricConstraint] = []
        self._target_label: Optional[str] = None

    # -- id / part helpers --------------------------------------------------
    def _pid(self, cls: str) -> str:
        i = self._class_counters.get(cls, 0)
        self._class_counters[cls] = i + 1
        return "part_%s_%d" % (cls, i)

    @staticmethod
    def _link(parent: SemanticPart, child: SemanticPart) -> None:
        child.parent_id = parent.id
        parent.child_ids.append(child.id)

    @staticmethod
    def _inferred_defaults(part: SemanticPart) -> None:
        part.inferred_geometry["rear_profile"] = EvidencedValue(
            value="mirrored_front_profile",
            source=EvidenceSource.GENERATED_HYPOTHESIS,
            confidence=0.3,
            note="rear/side surface unobserved; assumed to mirror the front")

    # -- appearance ---------------------------------------------------------
    @staticmethod
    def _appearance(image: Optional[np.ndarray],
                    prims: List[GeometricPrimitive],
                    subtract: Optional[List[GeometricPrimitive]] = None,
                    ) -> Optional[AppearanceEstimate]:
        """Mean sRGB + rough material guess over the part's region.

        ``subtract`` lists nested closed primitives owned by child parts, so
        e.g. a tyre samples its annulus, not the whole disc.
        """
        if image is None or not prims:
            return None
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in prims:
            pts = _poly(p)
            if len(pts) < 2:
                continue
            px = (pts * np.array([w, h])).astype(np.int32)
            if p.type in _CLOSED_TYPES and len(pts) >= 3:
                cv2.fillPoly(mask, [px], 255)
            else:
                cv2.polylines(mask, [px], False, 255, max(2, int(0.004 * w)))
        for p in subtract or []:
            pts = _poly(p)
            if len(pts) >= 3:
                px = (pts * np.array([w, h])).astype(np.int32)
                cv2.fillPoly(mask, [px], 0)
        sel = mask > 0
        if image.shape[2] == 4:
            sel &= image[:, :, 3] > 127
        if not sel.any():
            return None
        bgr = image[:, :, :3][sel].mean(axis=0)
        r, g, b = float(bgr[2]), float(bgr[1]), float(bgr[0])
        color = (int(round(r)), int(round(g)), int(round(b)))
        mx = max(r, g, b) / 255.0
        mn = min(r, g, b) / 255.0
        sat = 0.0 if mx < 1e-6 else (mx - mn) / mx
        if mx < 0.18 and sat < 0.35:
            mclass, rough, metal = "rubber", 0.85, 0.0
        elif sat > 0.4:
            mclass, rough, metal = "plastic", 0.5, 0.0
        elif mx >= 0.4 and sat <= 0.25:
            mclass, rough, metal = "metal", 0.35, 0.8
        else:
            mclass, rough, metal = "plastic", 0.6, 0.0
        return AppearanceEstimate(
            estimated_color_srgb=color,
            material_class=mclass,
            roughness=rough,
            metallic=metal,
            source=EvidenceSource.FITTED_FROM_OBSERVATION,
            confidence=0.35)

    # -- ring and radial structure ----------------------------------------
    @staticmethod
    def _ring_system(prims: List[GeometricPrimitive], scale: float
                     ) -> List[GeometricPrimitive]:
        """Largest connected group of plausible projected circular rings.

        The centre tolerance is deliberately looser than the constraint
        detector's: perspective projection and noisy cross-layer traces move
        fitted centres slightly. Slit-like detail ellipses are excluded.
        """
        rings = []
        for p in prims:
            if p.type not in _RING_TYPES or _center(p) is None:
                continue
            radii = p.params.get("radii")
            if radii is not None:
                major, minor = max(map(float, radii)), min(map(float, radii))
                # Strongly tilted small hubs can project below 0.5 aspect;
                # values below 0.35 are usually slits/highlights instead.
                if major <= 1e-9 or minor / major < 0.35:
                    continue
            rings.append(p)
        if not rings:
            return []

        parent = list(range(len(rings)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(rings)):
            ci = _center(rings[i])
            for j in range(i + 1, len(rings)):
                cj = _center(rings[j])
                if float(np.hypot(*(ci - cj))) <= 0.08 * scale:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        groups: Dict[int, List[GeometricPrimitive]] = {}
        for i, p in enumerate(rings):
            groups.setdefault(find(i), []).append(p)
        best = max(
            groups.values(),
            key=lambda g: (len(g), max((_radius(p) for p in g), default=0.0),
                           sorted(p.id for p in g)),
        )
        return sorted(best, key=lambda p: (-_radius(p), p.id))

    @staticmethod
    def _radius_groups(rings: List[GeometricPrimitive]
                       ) -> List[List[GeometricPrimitive]]:
        """Collapse cross-layer re-traces into distinct radius rungs."""
        if not rings:
            return []
        outer = max(_radius(p) for p in rings)
        groups: List[List[GeometricPrimitive]] = []
        for p in sorted(rings, key=lambda q: (-_radius(q), q.id)):
            r = _radius(p)
            if r < 0.06 * outer:
                continue
            if groups:
                mean_r = float(np.mean([_radius(q) for q in groups[-1]]))
                if abs(r - mean_r) <= 0.06 * max(mean_r, 1e-9):
                    groups[-1].append(p)
                    continue
            groups.append([p])
        return groups

    @staticmethod
    def _elongation(p: GeometricPrimitive) -> float:
        radii = p.params.get("radii")
        if radii is not None:
            major, minor = max(map(float, radii)), min(map(float, radii))
            return major / max(minor, 1e-9)
        size = p.params.get("size")
        if size is not None:
            major, minor = max(map(float, size)), min(map(float, size))
            return major / max(minor, 1e-9)
        pts = np.asarray(p.params.get("points") or p.fallback_points, dtype=float)
        if len(pts) < 2:
            return 1.0
        if len(pts) == 2:
            return 100.0
        vals = np.linalg.eigvalsh(np.cov(pts.T))
        return math.sqrt(max(float(vals[-1]), 1e-12)
                         / max(float(vals[0]), 1e-12))

    def _detect_spokes(self, graph: SketchGraph,
                       rings: List[GeometricPrimitive],
                       radius_groups: List[List[GeometricPrimitive]],
                       ) -> Optional[Dict[str, Any]]:
        """Detect repeated elongated radial evidence after un-foreshortening."""
        if len(radius_groups) < 2:
            return None
        ring_ids = {p.id for p in rings}
        r_out = max(_radius(p) for p in radius_groups[0])
        best = None
        for ring in rings:
            radii = ring.params.get("radii")
            if radii is None:
                rx = ry = _radius(ring)
            else:
                rx, ry = float(radii[0]), float(radii[1])
            major, minor = max(rx, ry), min(rx, ry)
            if major < 0.25 * r_out or minor <= 1e-9:
                continue
            stretch = major / minor if minor / major < 0.98 else 1.0
            theta = math.radians(
                float(ring.params.get("rotation_degrees", 0.0) or 0.0))
            if ry > rx:
                theta += math.pi / 2.0
            cc = _center(ring)
            ct, st = math.cos(-theta), math.sin(-theta)

            candidates = []
            for p in graph.primitives:
                if p.id in ring_ids or self._elongation(p) < 1.8:
                    continue
                pc = _center(p)
                if pc is None:
                    continue
                d = pc - cc
                u = ct * d[0] - st * d[1]
                v = st * d[0] + ct * d[1]
                ang = math.degrees(math.atan2(v * stretch, u)) % 360.0
                radius = float(math.hypot(u, v * stretch))
                if 0.10 * r_out < radius < 0.92 * r_out:
                    candidates.append((ang, p.id))
            if len(candidates) < 3:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]))
            clusters = [[candidates[0]]]
            for item in candidates[1:]:
                if item[0] - clusters[-1][-1][0] <= 25.0:
                    clusters[-1].append(item)
                else:
                    clusters.append([item])
            if (len(clusters) > 1
                    and clusters[0][0][0] + 360.0 - clusters[-1][-1][0] <= 25.0):
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()
            count = len(clusters)
            if not 3 <= count <= 12:
                continue
            means = sorted(float(np.mean([x[0] for x in group])) % 360.0
                           for group in clusters)
            step = 360.0 / count
            rel_dev = min(
                max(abs((ang - (off + i * step) + step / 2.0) % step
                        - step / 2.0) for i, ang in enumerate(means)) / step
                for off in np.arange(0.0, step, 0.5)
            )
            ids = sorted({pid for group in clusters for _, pid in group})
            candidate = (rel_dev, -count, ring.id, ids, count, cc, step)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None or best[0] > 0.45:
            return None
        by_id = {p.id: p for p in graph.primitives}
        shape_rank = {
            PrimitiveType.RECTANGLE: 0,
            PrimitiveType.ROUNDED_RECTANGLE: 0,
            PrimitiveType.CLOSED_REGION: 1,
            PrimitiveType.SYMMETRIC_SPLINE: 2,
            PrimitiveType.LINE: 3,
            PrimitiveType.POLYLINE: 3,
            PrimitiveType.CIRCULAR_ARC: 4,
            PrimitiveType.ELLIPTICAL_ARC: 4,
        }
        prototype = min(
            best[3], key=lambda pid: (
                shape_rank.get(by_id[pid].type, 5),
                -self._elongation(by_id[pid]), pid))
        outer_centers = np.asarray([_center(p) for p in radius_groups[0]],
                                   dtype=float)
        system_center = np.median(outer_centers, axis=0)
        return {
            "ids": best[3], "count": best[4],
            "prototype": prototype,
            "center": [float(system_center[0]), float(system_center[1])],
            "angle_degrees": float(best[6]), "confidence": 0.6,
        }

    @staticmethod
    def _dark_annulus(image: Optional[np.ndarray],
                      radius_groups: List[List[GeometricPrimitive]]) -> bool:
        if image is None or len(radius_groups) < 2:
            return False
        ring = radius_groups[0][0]
        c = _center(ring)
        radii = ring.params.get("radii")
        if c is None or radii is None:
            return False
        rx, ry = float(radii[0]), float(radii[1])
        major, minor = max(rx, ry), min(rx, ry)
        if minor <= 1e-9:
            return False
        theta = math.radians(float(ring.params.get("rotation_degrees", 0.0) or 0.0))
        if ry > rx:
            theta += math.pi / 2.0
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        dx, dy = xx / float(w) - c[0], yy / float(h) - c[1]
        ct, st = math.cos(-theta), math.sin(-theta)
        u = ct * dx - st * dy
        v = (st * dx + ct * dy) * (major / minor)
        rr = np.hypot(u, v)
        inner = max(_radius(p) for p in radius_groups[1])
        sel = (rr > inner * 1.08) & (rr < major * 0.95)
        if image.shape[2] == 4:
            sel &= image[:, :, 3] > 127
        if int(sel.sum()) < 32:
            return False
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        return float(gray[sel].mean()) / 255.0 < 0.45

    @staticmethod
    def _primary_keyword(keywords: set) -> Optional[str]:
        return next((k for k in _KEYWORDS if k in keywords), None)

    def _guided_role_parts(self, leftover: List[GeometricPrimitive],
                           root: SemanticPart, keyword: Optional[str],
                           image: Optional[np.ndarray], scale: float,
                           assigned: set) -> List[SemanticPart]:
        """Select conservative observed representatives for known major roles.

        A target label is an allowed Stage-1 input. It supplies vocabulary,
        while geometry still chooses which observed primitive supports each
        role. Near-duplicate cross-layer traces are grouped with the selected
        representative rather than emitted as separate solids.
        """
        roles = _GUIDED_ROLES.get(keyword or "", ())
        available = [p for p in leftover if _center(p) is not None]
        if not self._target_label or not roles or not available:
            return []
        root_prims = [self._primitive_by_id[i] for i in root.primitive_ids
                      if i in self._primitive_by_id]
        root_pts = [_poly(p) for p in root_prims if len(_poly(p))]
        all_pts = root_pts or [_poly(p) for p in available if len(_poly(p))]
        if all_pts:
            pts = np.concatenate(all_pts)
            object_center = pts.mean(axis=0)
            object_lo = pts.min(axis=0)
            object_hi = pts.max(axis=0)
        else:
            object_center = np.array([0.5, 0.5])
            object_lo = np.array([0.0, 0.0])
            object_hi = np.array([1.0, 1.0])

        def features(p):
            pts = _poly(p)
            c = _center(p)
            if len(pts):
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
                area = abs(float(cv2.contourArea(
                    (pts * 4096.0).astype(np.float32))))
            else:
                w = h = area = 0.0
            dist = float(np.hypot(*(c - object_center)))
            return c, w, h, area, dist

        def score(role: str, p: GeometricPrimitive):
            c, w, h, area, dist = features(p)
            elong = self._elongation(p)
            ring_penalty = 0.0 if p.type in _RING_TYPES else 1.0
            if role in ("cap", "shade"):
                return (c[1], ring_penalty, -area, p.id)
            if role == "backrest":
                return (c[1], -h, -area, p.id)
            if role == "base":
                return (-c[1], -area, p.id)
            if role in ("leg", "corner_post"):
                return (-c[1], -elong, -dist, p.id)
            if role == "side_slat":
                return (-w / max(h, 1e-9), -dist, p.id)
            if role == "center_bore":
                plausible = (dist <= 0.18 * scale
                             and _radius(p) >= 0.015 * scale)
                return (ring_penalty, 0.0 if plausible else 1.0,
                        -area, dist, p.id)
            if role == "mounting_hole":
                return (ring_penalty, dist, _radius(p), p.id)
            if role == "flange":
                inside = bool(np.all(c >= object_lo - 0.05 * scale)
                              and np.all(c <= object_hi + 0.05 * scale))
                return (ring_penalty, 0.0 if inside else 1.0,
                        -area, -dist, p.id)
            if role in ("handle", "tooth"):
                return (-dist, -elong, -area, p.id)
            if role == "logo_relief":
                return (dist, -area, p.id)
            if role == "lower_arm":
                return (abs(float(c[1]) - 0.62), -elong, p.id)
            if role == "upper_arm":
                return (abs(float(c[1]) - 0.38), -elong, p.id)
            if role in ("lid", "seat"):
                return (-w / max(h, 1e-9), -area, c[1], p.id)
            return (-area, p.id)

        result = []
        for role in roles:
            if not available:
                break
            chosen = min(available, key=lambda p: score(role, p))
            cc = _center(chosen)
            cr = _radius(chosen)
            group = []
            for p in list(available):
                pc = _center(p)
                same_kind = ((p.type in _RING_TYPES) ==
                             (chosen.type in _RING_TYPES))
                radius_close = (cr <= 1e-9 or
                                abs(_radius(p) - cr) <= 0.08 * max(cr, 1e-9))
                if (same_kind and radius_close
                        and float(np.hypot(*(pc - cc))) <= 0.02 * scale):
                    group.append(p)
                    available.remove(p)
            if not group:
                group = [chosen]
                available.remove(chosen)
            part = SemanticPart(
                id=self._pid(role), part_class=role,
                primitive_ids=[p.id for p in group],
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(image, group), confidence=0.5,
                notes=["role vocabulary from target label; supporting geometry "
                       "selected from observed primitives"])
            self._inferred_defaults(part)
            self._link(root, part)
            result.append(part)
            assigned.update(p.id for p in group)
        return result

    # -- main ---------------------------------------------------------------
    def decompose(self, graph: SketchGraph, image: Optional[np.ndarray],
                  spec: InputSpec, cfg: PipelineConfig) -> List[SemanticPart]:
        prims = list(graph.primitives)
        self._primitive_by_id = {p.id: p for p in prims}
        self._class_counters = {}
        self.generated_constraints = []
        self._target_label = (spec.target_label or "").strip().lower() or None
        text = ("%s %s" % (spec.target_label or "", spec.description or "")).lower()
        keywords = {k for k in _KEYWORDS if k in text}

        closed = [p for p in prims if p.type in _CLOSED_TYPES and len(p.fallback_points) >= 3]
        polys = {p.id: _poly(p) for p in closed}
        areas = {p.id: abs(cv2.contourArea((polys[p.id] * 4096.0).astype(np.float32)))
                 for p in closed}
        centers = {p.id: _center(p) for p in prims}
        scale = 1.0
        if prims:
            cat = np.concatenate([_poly(p) for p in prims if len(p.fallback_points)])
            if len(cat):
                scale = max(float(np.hypot(*(cat.max(axis=0) - cat.min(axis=0)))), 1e-9)

        # containment tree among closed primitives
        parent_of: Dict[str, Optional[str]] = {p.id: None for p in closed}
        for small in closed:
            best = None
            for big in closed:
                if big.id == small.id or areas[big.id] <= areas[small.id]:
                    continue
                if _inside(polys[big.id], polys[small.id], 0.002 * scale * 4096.0):
                    if best is None or areas[big.id] < areas[best.id]:
                        best = big
            parent_of[small.id] = best.id if best else None

        ring_cluster = self._ring_system(prims, scale)
        radius_groups = self._radius_groups(ring_cluster)

        rot = next((c for c in graph.constraints
                    if c.type == ConstraintType.ROTATIONAL_REPETITION
                    and int(c.params.get("count", 0)) >= 3), None)

        parts: List[SemanticPart] = []
        assigned: set = set()

        detected_spokes = (None if rot is not None else
                           self._detect_spokes(graph, ring_cluster, radius_groups))
        # Dense colour/detail tracing can fragment each visible spoke into
        # many tiny arcs.  A low-quality angular clustering of those
        # fragments is worse evidence than the conservative five-spoke wheel
        # prior: it produced four huge repeated trace fragments in Blender.
        # Keep clean observed repetition (one/few primitives per copy), but
        # explicitly mark the fallback as a semantic prior.
        if ("wheel" in keywords and detected_spokes is not None
                and len(detected_spokes["ids"]) > 2 * detected_spokes["count"]):
            detected_spokes = dict(detected_spokes)
            detected_spokes.update({
                "ids": [detected_spokes["prototype"]],
                "count": 5,
                "angle_degrees": 72.0,
                "confidence": 0.35,
                "source": EvidenceSource.SEMANTIC_PRIOR,
            })
        radial_wheel_evidence = (
            rot is not None or detected_spokes is not None
            or self._dark_annulus(image, radius_groups))
        # An explicit non-wheel target label disambiguates gear teeth, crate
        # slats, pipe flanges, and other repeated/concentric structures.
        wheel_like = len(radius_groups) >= 2 and (
            "wheel" in keywords or (not keywords and radial_wheel_evidence))
        if wheel_like:
            parts = self._decompose_wheel(
                graph, radius_groups, rot, detected_spokes,
                image, keywords, assigned)
        elif (len(radius_groups) >= 2
              and (not keywords or "knob" in keywords)):
            parts = self._decompose_ring_system(
                graph, radius_groups, image, keywords, assigned)
        else:
            parts = self._decompose_generic(
                graph, closed, areas, parent_of, centers, scale, image,
                keywords, assigned)

        # leftovers -> details part under the root
        leftover = [p for p in prims if p.id not in assigned]
        if leftover and parts:
            root = parts[0]
            role_parts = self._guided_role_parts(
                leftover, root, self._primary_keyword(keywords), image,
                scale, assigned)
            parts.extend(role_parts)
            leftover = [p for p in prims if p.id not in assigned]
        if leftover and parts:
            root = parts[0]
            det = SemanticPart(
                id=self._pid("details"),
                part_class="details",
                primitive_ids=[p.id for p in leftover],
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(image, leftover),
                confidence=0.4,
                notes=["unclassified geometry grouped as details"])
            self._inferred_defaults(det)
            self._link(root, det)
            parts.append(det)
            assigned.update(p.id for p in leftover)

        # appearance post-pass: sample each part's region, excluding nested
        # closed regions owned by descendant parts (e.g. tyre = annulus)
        prim_by_id = {p.id: p for p in prims}
        for part in parts:
            if not part.primitive_ids:
                continue
            own = [prim_by_id[i] for i in part.primitive_ids if i in prim_by_id]
            own_ids = set(part.primitive_ids)
            sub = []
            for q in closed:
                if q.id in own_ids:
                    continue
                cur = parent_of.get(q.id)
                while cur is not None:
                    if cur in own_ids:
                        sub.append(prim_by_id[q.id])
                        break
                    cur = parent_of.get(cur)
            part.appearance = self._appearance(image, own, sub)

        return parts
    def _decompose_wheel(self, graph, radius_groups, rot, detected_spokes,
                         image, keywords, assigned) -> List[SemanticPart]:
        root_cls = "wheel"  # structurally wheel-like: concentric rings + radial repetition
        ring_cluster = [p for group in radius_groups for p in group]
        ring_ids = [p.id for p in ring_cluster]
        spoke_ids: List[str] = []
        spoke_count = 0
        spoke_angle = None
        if rot is not None:
            spoke_ids = [i for i in rot.entities if i not in ring_ids]
            spoke_count = int(rot.params.get("count", len(spoke_ids)))
            spoke_angle = rot.params.get("angle_degrees")
        elif detected_spokes is not None:
            spoke_ids = list(detected_spokes["ids"])
            spoke_count = int(detected_spokes["count"])
            spoke_angle = float(detected_spokes["angle_degrees"])
            self.generated_constraints.append(GeometricConstraint(
                type=ConstraintType.ROTATIONAL_REPETITION,
                entities=spoke_ids,
                params={
                    "prototype": detected_spokes["prototype"],
                    "count": spoke_count,
                    "angle_degrees": round(spoke_angle, 3),
                    "center": [round(float(v), 6)
                               for v in detected_spokes["center"]],
                },
                confidence=float(detected_spokes["confidence"]),
                source=detected_spokes.get(
                    "source", EvidenceSource.FITTED_FROM_OBSERVATION),
            ))
        prim_by_id = {p.id: p for p in graph.primitives}

        root = SemanticPart(
            id=self._pid(root_cls),
            part_class=root_cls,
            primitive_ids=[],
            visibility=Visibility.VISIBLE,
            confidence=0.75 if "wheel" in keywords else 0.6,
            notes=["concentric ring system%s" %
                   (" with %d-fold rotational repetition" % spoke_count
                    if spoke_count else "")])
        root.inferred_geometry["axial_depth"] = EvidencedValue(
            value=round(0.5 * max(_radius(p) for p in radius_groups[0]), 6),
            unit="normalized",
            source=EvidenceSource.SEMANTIC_PRIOR,
            confidence=0.4,
            note="object depth unobserved; wheel-like prior")
        self._inferred_defaults(root)
        parts = [root]

        def ring_part(cls, prim_ids, parent, conf):
            part = SemanticPart(
                id=self._pid(cls),
                part_class=cls,
                primitive_ids=list(prim_ids),
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(
                    image, [prim_by_id[i] for i in prim_ids if i in prim_by_id]),
                confidence=conf)
            self._inferred_defaults(part)
            self._link(parent, part)
            return part

        outer_ids = [p.id for p in radius_groups[0]]
        tyre = ring_part("tyre", outer_ids, root, 0.7)
        parts.append(tyre)
        assigned.update(outer_ids)

        middle_groups = (radius_groups[1:-1] if len(radius_groups) >= 3
                         else [radius_groups[1]])
        middle = [p.id for group in middle_groups for p in group]
        rim = ring_part("rim", middle, root, 0.7)
        parts.append(rim)
        assigned.update(middle)

        if len(radius_groups) >= 3:
            hub_ids = [p.id for p in radius_groups[-1]]
            hub = ring_part("hub", hub_ids, rim, 0.65)
            parts.append(hub)
            assigned.update(hub_ids)

        if spoke_ids:
            spokes = SemanticPart(
                id=self._pid("spokes"),
                part_class="spokes",
                primitive_ids=list(spoke_ids),
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(
                    image, [prim_by_id[i] for i in spoke_ids if i in prim_by_id]),
                confidence=0.65,
                notes=["radial repetition count=%d angle=%s deg"
                       % (spoke_count, spoke_angle)])
            spokes.inferred_geometry["repetition_count"] = EvidencedValue(
                value=spoke_count,
                source=(rot.source if rot is not None else
                        detected_spokes.get(
                            "source", EvidenceSource.FITTED_FROM_OBSERVATION)),
                confidence=float(rot.confidence) if rot is not None else 0.6)
            self._inferred_defaults(spokes)
            self._link(rim, spokes)
            parts.append(spokes)
            assigned.update(spoke_ids)
        return parts

    def _decompose_ring_system(self, graph, radius_groups, image, keywords,
                               assigned) -> List[SemanticPart]:
        """Semantic structure for non-wheel concentric manufactured parts."""
        prim_by_id = {p.id: p for p in graph.primitives}
        keyword = self._primary_keyword(keywords)
        root_cls = (_ROOT_CLASS_BY_KEYWORD.get(keyword or "", keyword)
                    if self._target_label else keyword) or "ring_system"
        root_ids = ([p.id for p in radius_groups[0]] if keyword else [])
        root = SemanticPart(
            id=self._pid(root_cls), part_class=root_cls, primitive_ids=root_ids,
            visibility=Visibility.VISIBLE, confidence=0.55,
            notes=["concentric ring system; exact hidden cross-section unknown"])
        if root_ids:
            root.appearance = self._appearance(
                image, [prim_by_id[i] for i in root_ids if i in prim_by_id])
            assigned.update(root_ids)
        self._inferred_defaults(root)
        parts = [root]

        def add(cls, group, parent, conf):
            ids = [p.id for p in group]
            part = SemanticPart(
                id=self._pid(cls), part_class=cls, primitive_ids=ids,
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(
                    image, [prim_by_id[i] for i in ids if i in prim_by_id]),
                confidence=conf)
            self._inferred_defaults(part)
            self._link(parent, part)
            parts.append(part)
            assigned.update(ids)
            return part

        if not keyword:
            add("outer_shell", radius_groups[0], root, 0.6)
        middle_groups = (radius_groups[1:-1] if len(radius_groups) >= 3
                         else [radius_groups[1]])
        middle = add("inner_panel", [p for g in middle_groups for p in g],
                     root, 0.55)
        if len(radius_groups) >= 3:
            add("hub", radius_groups[-1], middle, 0.55)
        return parts

    # -- generic ------------------------------------------------------------
    def _decompose_generic(self, graph, closed, areas, parent_of, centers,
                           scale, image, keywords, assigned) -> List[SemanticPart]:
        prim_by_id = {p.id: p for p in graph.primitives}
        keyword = self._primary_keyword(keywords)
        root_cls = (_ROOT_CLASS_BY_KEYWORD.get(keyword or "", keyword)
                    if self._target_label else keyword) or "body"
        roots = [p for p in closed if parent_of.get(p.id) is None]
        roots.sort(key=lambda p: areas[p.id], reverse=True)

        if roots:
            root_prim = roots[0]
            root = SemanticPart(
                id=self._pid(root_cls),
                part_class=root_cls,
                primitive_ids=[root_prim.id],
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(image, [root_prim]),
                confidence=0.6,
                notes=["outermost closed region"])
            assigned.add(root_prim.id)
        else:
            root = SemanticPart(
                id=self._pid(root_cls),
                part_class=root_cls,
                primitive_ids=[],
                visibility=Visibility.PARTIAL,
                confidence=0.4,
                notes=["no closed silhouette primitive found"])
        self._inferred_defaults(root)
        parts = [root]

        # A crate's open slat spacing is not an inferred construction detail:
        # it is directly observed as child paths of the outer silhouette.
        # Preserve each fitted hole as its own boolean-capable semantic part
        # so the outer silhouette extrusion cannot turn the crate into a
        # solid filled panel.
        if keyword == "crate" and roots:
            outer_area = max(areas.get(roots[0].id, 0.0), 1e-9)
            root_polygon = _poly(roots[0])
            for p in graph.primitives:
                points = _poly(p)
                candidate_area = (abs(cv2.contourArea(
                    (points * 4096.0).astype(np.float32)))
                    if len(points) >= 3 else 0.0)
                center = _center(p)
                inside_root = (
                    center is not None and len(root_polygon) >= 3
                    and cv2.pointPolygonTest(
                        (root_polygon * 4096.0).astype(np.float32),
                        (float(center[0] * 4096.0),
                         float(center[1] * 4096.0)), False) >= 0
                )
                if (p.id in assigned
                        or p.source_layer != TraceLayerName.SILHOUETTE
                        or len(points) < 3
                        or not inside_root
                        # Cross-layer outer-silhouette retraces can be nested
                        # by a few pixels; they are not negative space.
                        or candidate_area / outer_area >= 0.25):
                    continue
                cutout = SemanticPart(
                    id=self._pid("cutout"),
                    part_class="cutout",
                    primitive_ids=[p.id],
                    visibility=Visibility.VISIBLE,
                    appearance=None,
                    confidence=max(0.5, float(p.confidence)),
                    notes=["negative space directly observed as a silhouette hole"],
                )
                self._inferred_defaults(cutout)
                self._link(root, cutout)
                parts.append(cutout)
                assigned.add(p.id)

        # nested closed primitives: rectangles -> panel/bezel, else insert
        nested = [p for p in closed if p.id not in assigned]
        nested.sort(key=lambda p: areas[p.id], reverse=True)
        depth_of: Dict[str, int] = {}
        for p in nested:
            d = 0
            cur = parent_of.get(p.id)
            while cur is not None:
                d += 1
                cur = parent_of.get(cur)
            depth_of[p.id] = d
        rect_types = {PrimitiveType.RECTANGLE, PrimitiveType.ROUNDED_RECTANGLE}
        panel_done = bezel_done = False
        root_area = areas.get(roots[0].id, 1.0) if roots else 1.0
        root_center = centers.get(roots[0].id) if roots else None
        for p in nested:
            # Cross-layer re-traces of the outer silhouette belong to the
            # root, while tiny nested fragments are surface detail rather
            # than dozens of independent solids.
            pc = centers.get(p.id)
            area_ratio = areas[p.id] / max(root_area, 1e-9)
            if (root_center is not None and pc is not None
                    and float(np.hypot(*(pc - root_center))) <= 0.02 * scale
                    and 0.85 <= area_ratio <= 1.15):
                root.primitive_ids.append(p.id)
                assigned.add(p.id)
                continue
            if area_ratio < 0.005:
                continue
            if self._target_label and keyword in _GUIDED_ROLES:
                # Major guided roles are selected once from the combined
                # leftover evidence after this structural pass.
                continue
            if p.type in rect_types and not panel_done:
                cls = "panel"
                panel_done = True
            elif p.type in rect_types and not bezel_done:
                cls = "bezel"
                bezel_done = True
            else:
                # Keep an unguided model compact: after two structural panels,
                # additional nested traces are grouped as details below.
                if any(q.part_class == "insert" for q in parts):
                    continue
                cls = "insert"
            part = SemanticPart(
                id=self._pid(cls),
                part_class=cls,
                primitive_ids=[p.id],
                visibility=Visibility.VISIBLE,
                appearance=self._appearance(image, [p]),
                confidence=0.55,
                notes=["nested closed region, depth %d" % depth_of[p.id]])
            self._inferred_defaults(part)
            self._link(root, part)
            parts.append(part)
            assigned.add(p.id)

        # attached side blobs: closed shapes whose centre sits outside the
        # root silhouette but that overlap its boundary -> handle/appendage
        if roots and (not self._target_label or keyword not in _GUIDED_ROLES
                      or keyword in ("mug", "cup")):
            root_poly = np.asarray(roots[0].fallback_points, dtype=float)
            handle_cls = "handle" if ({"cup", "mug"} & keywords) else "appendage"
            attached_done = False
            for p in graph.primitives:
                if attached_done:
                    break
                if p.id in assigned or p.id == roots[0].id:
                    continue
                c = centers.get(p.id)
                if c is None or p.type not in _CLOSED_TYPES:
                    continue
                d = cv2.pointPolygonTest(
                    (root_poly * 4096.0).astype(np.float32),
                    (float(c[0] * 4096.0), float(c[1] * 4096.0)), True)
                if d >= 0:
                    continue  # centre inside root: handled above
                pts = _poly(p)
                if pts.size and (np.abs(pts - np.clip(
                        pts, root_poly.min(axis=0), root_poly.max(axis=0)))
                        .max() < 0.05 * scale):
                    part = SemanticPart(
                        id=self._pid(handle_cls),
                        part_class=handle_cls,
                        primitive_ids=[p.id],
                        visibility=Visibility.VISIBLE,
                        appearance=self._appearance(image, [p]),
                        confidence=0.5,
                        notes=["closed blob attached outside the root silhouette"])
                    self._inferred_defaults(part)
                    self._link(root, part)
                    parts.append(part)
                    assigned.add(p.id)
                    attached_done = True
        return parts


def _default_backend(spec: InputSpec, cfg: PipelineConfig) -> SemanticBackend:
    """Backend selection hook.  A future VLM backend would be chosen here
    (e.g. via config); the heuristic backend is the deterministic default."""
    return HeuristicSemanticBackend()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def decompose_parts(graph: SketchGraph, crop_rgba_path: str, spec: InputSpec,
                    cfg: PipelineConfig) -> SketchGraph:
    """Group primitives into semantic parts.  Primitive geometry is never
    modified; only labels, hierarchy, appearance and inferred hypotheses are
    attached to the returned graph copy."""
    image = None
    if crop_rgba_path and os.path.exists(crop_rgba_path):
        img = cv2.imread(crop_rgba_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            image = img
    backend = _default_backend(spec, cfg)
    parts = backend.decompose(graph, image, spec, cfg)

    generated = list(getattr(backend, "generated_constraints", []))
    constraints = list(graph.constraints)
    existing = {(c.type, tuple(sorted(c.entities))) for c in constraints}
    for constraint in generated:
        key = (constraint.type, tuple(sorted(constraint.entities)))
        if key not in existing:
            constraints.append(constraint)
            existing.add(key)
    out = graph.model_copy(update={"parts": parts, "constraints": constraints})
    stats = dict(out.stats)
    stats["part_count"] = len(parts)
    stats["part_classes"] = sorted({p.part_class for p in parts})
    stats["semantic_backend"] = backend.name
    stats["semantic_generated_constraints"] = len(generated)
    out.stats = stats
    return out
