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
    GeometricPrimitive,
    InputSpec,
    PrimitiveType,
    SemanticPart,
    SketchGraph,
    Visibility,
)

_CLOSED_TYPES = {
    PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE, PrimitiveType.RECTANGLE,
    PrimitiveType.ROUNDED_RECTANGLE, PrimitiveType.REGULAR_POLYGON,
    PrimitiveType.CLOSED_REGION, PrimitiveType.SYMMETRIC_SPLINE,
}
_RING_TYPES = {PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE}
_KEYWORDS = ("wheel", "bottle", "cup", "lamp", "chair", "table", "box",
             "sign", "logo", "bracket")


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
    return float(np.hypot(*(pts - c)).max())


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

    # -- main ---------------------------------------------------------------
    def decompose(self, graph: SketchGraph, image: Optional[np.ndarray],
                  spec: InputSpec, cfg: PipelineConfig) -> List[SemanticPart]:
        prims = list(graph.primitives)
        self._class_counters = {}
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

        # concentric ring system (circles/ellipses sharing a centre)
        rings = [p for p in closed if p.type in _RING_TYPES and centers.get(p.id) is not None]
        ring_cluster: List[GeometricPrimitive] = []
        for seed in rings:
            sc = centers[seed.id]
            cl = [p for p in rings
                  if float(np.hypot(*(centers[p.id] - sc))) <= 0.02 * scale]
            if len(cl) > len(ring_cluster):
                ring_cluster = cl
        ring_cluster = sorted(ring_cluster, key=_radius, reverse=True)

        rot = next((c for c in graph.constraints
                    if c.type == ConstraintType.ROTATIONAL_REPETITION
                    and int(c.params.get("count", 0)) >= 3), None)

        parts: List[SemanticPart] = []
        assigned: set = set()

        wheel_like = len(ring_cluster) >= 2 and (
            rot is not None or "wheel" in keywords)
        if wheel_like:
            parts = self._decompose_wheel(
                graph, ring_cluster, rot, image, keywords, assigned)
        else:
            parts = self._decompose_generic(
                graph, closed, areas, parent_of, centers, scale, image,
                keywords, assigned)

        # leftovers -> details part under the root
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
    def _decompose_wheel(self, graph, ring_cluster, rot, image, keywords,
                         assigned) -> List[SemanticPart]:
        root_cls = "wheel"  # structurally wheel-like: concentric rings + radial repetition
        ring_ids = [p.id for p in ring_cluster]
        spoke_ids: List[str] = []
        spoke_count = 0
        spoke_angle = None
        if rot is not None:
            spoke_ids = [i for i in rot.entities if i not in ring_ids]
            spoke_count = int(rot.params.get("count", len(spoke_ids)))
            spoke_angle = rot.params.get("angle_degrees")
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
            value=round(0.5 * _radius(ring_cluster[0]), 6),
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

        tyre = ring_part("tyre", [ring_ids[0]], root, 0.7)
        parts.append(tyre)
        assigned.add(ring_ids[0])

        middle = ring_ids[1:-1] if len(ring_ids) >= 3 else [ring_ids[1]]
        rim = ring_part("rim", middle, root, 0.7)
        parts.append(rim)
        assigned.update(middle)

        if len(ring_ids) >= 3:
            hub = ring_part("hub", [ring_ids[-1]], rim, 0.65)
            parts.append(hub)
            assigned.add(ring_ids[-1])

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
                source=EvidenceSource.FITTED_FROM_OBSERVATION,
                confidence=float(rot.confidence) if rot is not None else 0.6)
            self._inferred_defaults(spokes)
            self._link(rim, spokes)
            parts.append(spokes)
            assigned.update(spoke_ids)
        return parts

    # -- generic ------------------------------------------------------------
    def _decompose_generic(self, graph, closed, areas, parent_of, centers,
                           scale, image, keywords, assigned) -> List[SemanticPart]:
        prim_by_id = {p.id: p for p in graph.primitives}
        root_cls = next((k for k in _KEYWORDS if k in keywords), "body")
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
        for p in nested:
            if p.type in rect_types and not panel_done:
                cls = "panel"
                panel_done = True
            elif p.type in rect_types and not bezel_done:
                cls = "bezel"
                bezel_done = True
            else:
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
        if roots:
            root_poly = np.asarray(roots[0].fallback_points, dtype=float)
            handle_cls = "handle" if ({"cup", "lamp"} & keywords) else "appendage"
            for p in graph.primitives:
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

    out = graph.model_copy(update={"parts": parts})
    stats = dict(out.stats)
    stats["part_count"] = len(parts)
    stats["part_classes"] = sorted({p.part_class for p in parts})
    stats["semantic_backend"] = backend.name
    out.stats = stats
    return out
