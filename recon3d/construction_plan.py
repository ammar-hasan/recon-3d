"""Stage 14: declarative parametric construction plan + plan validation.

Coordinate mapping: normalised image coords (origin top-left, y down) are
mapped to object units (origin at object centre, x right, y up, z toward
camera; object width = 1.0) using the root/outermost primitive bbox via
``part_geometry.ObjectFrame``.

Materials come from ``materials.estimate_materials`` when a crop RGBA is
discoverable under ``spec.output_dir`` (project layout); otherwise each part
falls back to its semantic appearance estimate, marked with its own source.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import materials as materials_mod
from .config import PipelineConfig
from .part_geometry import (
    CLOSED_FLAT,
    ELLIPSE_LIKE,
    ObjectFrame,
    graph_bbox,
    outline_of,
    part_bbox,
    part_primitives,
    primitive_bbox,
    primitive_center,
    primitive_radii,
)
from .schemas import (
    CameraEstimate,
    ConstraintType,
    ConstructionPlan,
    DepthEvidence,
    EvidencedValue,
    EvidenceSource,
    InputSpec,
    MaterialSpec,
    OperatorCategory,
    PlanPart,
    PrimitiveType,
    SemanticPart,
    SketchGraph,
)

_RADIAL_CONSTRAINTS = {ConstraintType.RADIAL_SYMMETRY, ConstraintType.ROTATIONAL_REPETITION}
_ALLOWED_UNITS = ("normalized", "meters")
_PRIMITIVE_SHAPES = ("cube", "cylinder", "sphere", "cone", "torus", "capsule")
_BOOLEAN_OPS = ("difference", "union", "intersect")

#: default extrusion slab depth when no depth evidence exists (fraction of
#: object width) — always marked as a generated hypothesis
_DEFAULT_EXTRUDE_DEPTH = 0.05
#: scale converting relative dome depth (0..1) into object units
_DEPTH_TO_UNITS = 0.3


def _find_crop_rgba(spec: InputSpec) -> Optional[str]:
    out = Path(spec.output_dir)
    if not out.is_dir():
        return None
    for rel in ("crop_rgba.png", "crop/crop_rgba.png", "segmentation/crop_rgba.png"):
        p = out / rel
        if p.is_file():
            return str(p)
    matches = sorted(out.rglob("*crop*rgba*.png"))
    return str(matches[0]) if matches else None


def _material_from_appearance(part: SemanticPart) -> MaterialSpec:
    ap = part.appearance
    spec = MaterialSpec(source=EvidenceSource.UNKNOWN)
    if ap is not None:
        if ap.estimated_color_srgb is not None:
            spec.base_color = tuple(
                float(c) / 255.0 for c in ap.estimated_color_srgb
            )  # type: ignore[assignment]
        spec.material_class = ap.material_class or "plastic"
        if ap.roughness is not None:
            spec.roughness = float(ap.roughness)
        if ap.metallic is not None:
            spec.metallic = float(ap.metallic)
        spec.source = ap.source
    return spec


def _frame_bbox(graph: SketchGraph) -> Tuple[float, float, float, float]:
    """Anchor bbox for the object frame: the true outline extent of the
    largest fully-closed primitive (the subject). The frame must be centred
    on the object — object rotation pivots on the frame origin at render
    time — and its width must match the observed silhouette, so stray
    detail fragments and outlier arcs must not stretch or shift it. Arcs
    are excluded: their parametric outline is not their true extent. Falls
    back to the whole-graph bbox when there are no closed primitives."""
    anchor_types = CLOSED_FLAT | {PrimitiveType.CIRCLE, PrimitiveType.ELLIPSE}
    best = None
    best_area = 0.0
    for p in graph.primitives:
        if p.type not in anchor_types:
            continue
        pts = outline_of(p, n=48)
        if len(pts) < 3:
            continue
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = (min(xs), min(ys), max(xs), max(ys))
    return best if best is not None else graph_bbox(graph)


class _PlanBuilder:
    def __init__(
        self,
        graph: SketchGraph,
        camera: CameraEstimate,
        depth: DepthEvidence,
        materials: Dict[str, MaterialSpec],
    ) -> None:
        self.graph = graph
        self.camera = camera
        self.depth = depth
        self.materials = materials
        self.frame = ObjectFrame(_frame_bbox(graph))

    # -- shared helpers -----------------------------------------------------

    def part_center_obj(self, part: SemanticPart) -> Tuple[float, float]:
        prims = part_primitives(self.graph, part)
        if not prims:
            return 0.0, 0.0
        centres = [primitive_center(p) for p in prims]
        u = sum(c[0] for c in centres) / len(centres)
        v = sum(c[1] for c in centres) / len(centres)
        return self.frame.point(u, v)

    def depth_estimate(self, part: SemanticPart) -> Optional[float]:
        ev = self.depth.region_estimates.get(part.id)
        if ev is None or ev.value is None:
            return None
        return float(ev.value)

    def z_offset(self, part: SemanticPart) -> float:
        est = self.depth_estimate(part)
        return _DEPTH_TO_UNITS * est if est is not None else 0.0

    def outline_obj(self, part: SemanticPart) -> List[List[float]]:
        """Largest-area primitive outline in object units (x right, y up)."""
        prims = part_primitives(self.graph, part)
        if not prims:
            return []
        prim = max(
            prims,
            key=lambda p: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(primitive_bbox(p)),
        )
        pts = outline_of(prim)
        # Some fitted open/symmetric primitives store a two-point parametric
        # summary while retaining the observed contour in fallback_points.
        # Closed construction operators need that contour, not the summary.
        if len(pts) < 3 and len(prim.fallback_points) >= 3:
            pts = [(float(x), float(y)) for x, y in prim.fallback_points]
        return [[round(x, 6), round(y, 6)] for x, y in (self.frame.point(u, v) for u, v in pts)]

    def extrusion_depth(self, part: SemanticPart, notes: List[str]) -> float:
        est = self.depth_estimate(part)
        if est is not None:
            notes.append(
                "extrusion depth from depth evidence (relative %.3f, low confidence)"
                % est
            )
            return max(0.02, _DEPTH_TO_UNITS * est)
        notes.append(
            "extrusion depth is a generated hypothesis (default %.3f of object "
            "width); no depth evidence" % _DEFAULT_EXTRUDE_DEPTH
        )
        return _DEFAULT_EXTRUDE_DEPTH

    def base_part(
        self, part: SemanticPart, op: OperatorCategory, confidence: float, notes: List[str]
    ) -> PlanPart:
        return PlanPart(
            id=part.id,
            operator=op,
            parent=part.parent_id,
            material=self.materials.get(part.id) or _material_from_appearance(part),
            visibility=part.visibility,
            evidence=EvidencedValue(
                value=op.value,
                source=EvidenceSource.FITTED_FROM_OBSERVATION,
                confidence=confidence,
                note="; ".join(notes) if notes else None,
            ),
        )

    def operator_of(self, part: SemanticPart) -> Tuple[OperatorCategory, float]:
        if part.selected_operator:
            op = OperatorCategory(part.selected_operator)
            conf = 0.5
            for cand in part.construction_candidates:
                if cand.operator == op:
                    conf = cand.confidence
                    break
            return op, conf
        if part.construction_candidates:
            top = max(part.construction_candidates, key=lambda c: c.confidence)
            return top.operator, top.confidence
        return OperatorCategory.FREEFORM, 0.2

    # -- per-operator builders ----------------------------------------------

    def _part_center_uv(self, part: SemanticPart) -> Tuple[float, float]:
        """Part centre in normalised image coords; for primitive-less root
        parts, the centre of the object frame's anchor bbox (the subject).
        A concentric-constraint centre is NOT used here: the largest group
        can be a junk cluster of arc fragments off the object centre."""
        prims = part_primitives(self.graph, part)
        if prims:
            centres = [primitive_center(p) for p in prims]
            return (sum(c[0] for c in centres) / len(centres),
                    sum(c[1] for c in centres) / len(centres))
        f = self.frame
        return f.cx, f.cy

    def _radius_ladder(self, cu: float, cv: float) -> List[float]:
        """Distinct observed ring radii (normalised, descending) of
        ellipse-like primitives centred (loosely) on the given system centre.
        Near-duplicate re-traces of the same edge collapse into one rung."""
        tol = 0.15 * self.frame.width
        radii = []
        for p in self.graph.primitives:
            if p.type not in ELLIPSE_LIKE:
                continue
            rad = primitive_radii(p)
            if not rad:
                continue
            px, py = primitive_center(p)
            if math.hypot(px - cu, py - cv) > tol:
                continue
            radii.append(max(rad))
        radii.sort(reverse=True)
        ladder: List[float] = []
        for r in radii:
            if not ladder or abs(r - ladder[-1]) > 0.02 * ladder[-1]:
                ladder.append(r)
        return ladder

    def _axial_revolve_profile(self, part: SemanticPart):
        """Observed side silhouette as a radius/height profile.

        Upright vessels revolve around image-up (object Y), unlike wheels
        whose visible circular face revolves around object Z.  Scanline
        intersections make the envelope stable even when the fitted contour
        has unevenly spaced points.
        """
        cls = (part.part_class or "").lower()
        axial = ("bottle", "vase", "mug", "cup", "knob", "cap")
        if not any(k in cls for k in axial):
            return None
        prims = part_primitives(self.graph, part)
        if cls == "cap":
            ellipses = [p for p in prims if primitive_radii(p)]
            if ellipses:
                prim = max(ellipses, key=lambda p: max(primitive_radii(p)))
                cu, cv = primitive_center(prim)
                cx, hy = self.frame.point(cu, cv)
                radius = self.frame.length(max(primitive_radii(prim)))
                half_h = max(0.05, 0.45 * radius)
                return ({
                    "type": "polyline",
                    "points": [[0.0, round(hy - half_h, 6)],
                               [round(radius, 6), round(hy - half_h, 6)],
                               [round(radius, 6), round(hy + half_h, 6)],
                               [0.0, round(hy + half_h, 6)]],
                    "closed": True,
                }, {"origin": [round(cx, 6), 0.0, 0.0],
                    "direction": [0.0, 1.0, 0.0]})
        candidates = []
        for prim in prims:
            if primitive_radii(prim) is not None:
                continue
            pts = outline_of(prim)
            if len(pts) < 3:
                continue
            bx = primitive_bbox(prim)
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            candidates.append((area, prim, pts))
        if not candidates:
            return None
        _, _, uv_points = max(candidates, key=lambda item: item[0])
        xy = [self.frame.point(float(u), float(v)) for u, v in uv_points]
        xmin = min(x for x, _ in xy)
        xmax = max(x for x, _ in xy)
        ymin = min(y for _, y in xy)
        ymax = max(y for _, y in xy)
        axis_x = 0.5 * (xmin + xmax)
        if ymax - ymin <= 1e-9:
            return None
        boundary = []
        closed = xy + [xy[0]]
        for i in range(33):
            h = ymin + (ymax - ymin) * i / 32.0
            intersections = []
            for (x0, y0), (x1, y1) in zip(closed, closed[1:]):
                if abs(y1 - y0) <= 1e-12:
                    if abs(h - y0) <= 1e-9:
                        intersections.extend([x0, x1])
                    continue
                if h < min(y0, y1) or h > max(y0, y1):
                    continue
                t = (h - y0) / (y1 - y0)
                intersections.append(x0 + t * (x1 - x0))
            if intersections:
                radius = max(abs(x - axis_x) for x in intersections)
                boundary.append([round(radius, 6), round(h, 6)])
        if len(boundary) < 4:
            return None
        profile = {
            "type": "polyline",
            "points": [[0.0, round(ymin, 6)]] + boundary
                      + [[0.0, round(ymax, 6)]],
            "closed": True,
        }
        axis = {"origin": [round(axis_x, 6), 0.0, 0.0],
                "direction": [0.0, 1.0, 0.0]}
        return profile, axis

    def build_revolve(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        axial = self._axial_revolve_profile(part)
        if axial is not None:
            profile, axis = axial
            notes.append(
                "axial radius profile fitted from the observed side silhouette")
            pp = self.base_part(part, OperatorCategory.REVOLVE, conf, notes)
            pp.axis = axis
            pp.profile = profile
            return pp
        prims = [p for p in part_primitives(self.graph, part) if p.type in ELLIPSE_LIKE]
        cu, cv = self._part_center_uv(part)
        cx, cy = self.frame.point(cu, cv)
        ladder = self._radius_ladder(cu, cv)
        own = sorted(
            (max(primitive_radii(p)) for p in prims if primitive_radii(p)),
            reverse=True,
        )
        if own:
            r_out = self.frame.length(own[0])
        elif ladder:
            # root/assembly part without own ring geometry: span the
            # observed concentric system (e.g. the wheel's outer tyre ring)
            r_out = self.frame.length(ladder[0])
            notes.append(
                "no own ring primitive; profile spans the observed concentric "
                "system's outer radius"
            )
        else:
            r_out = 0.1
            notes.append("no ring geometry observed; fallback radius hypothesis")
        # hollow section: the next observed ring strictly inside this one.
        # Assembly roots stay solid so they back their child rings and never
        # leave a see-through hole in the silhouette.
        r_in = 0.0
        if own:
            for r in ladder:
                rl = self.frame.length(r)
                if rl < 0.98 * r_out:
                    r_in = rl
                    notes.append(
                        "hollow cross-section: inner radius from the next "
                        "observed concentric ring"
                    )
                    break
        if own:
            half_h = 0.06 * r_out
            notes.append(
                "revolve half-height is a generated hypothesis (0.06 * radius); "
                "kept slim so the equator circle dominates the silhouette"
            )
        else:
            # assembly roots carry no observed cross-section: they become a
            # backing slab placed entirely behind z=0, deep enough that no
            # tilted sightline through the child rings' hollow centre can
            # pass underneath it, yet always behind the child front surfaces
            half_h = 0.08 * r_out
            notes.append(
                "backing-slab hypothesis for the ring-system root (deep "
                "behind-plane disc); hidden depth geometry inferred"
            )
        est = self.depth_estimate(part)
        if est is not None:
            half_h *= 1.0 + est
            notes.append(
                "revolve profile bulged by depth evidence (relative %.3f, "
                "low confidence)" % est
            )

        # the revolve axis is the object's symmetry axis; the camera stage's
        # object rotation (tilt from circle-unprojection) is applied to the
        # whole model at render time, so it must NOT be baked in here
        rot = self.camera.object_rotation_euler_deg
        if rot.value is not None:
            notes.append(
                "axis is the object symmetry axis; %.1f deg tilt carried by "
                "the camera stage's object rotation" % float(rot.value[0])
            )
        direction = [0.0, 0.0, 1.0]

        if r_in > 0.0:
            # tyre-like barrel cross-section: the maximum radius occurs only
            # at the equator so the projected silhouette keeps the observed
            # ellipse axis ratio instead of being fattened by the sidewall
            r_sh = max(r_in, 0.92 * r_out)
            pts = [
                [r_in, -half_h],
                [r_sh, -half_h],
                [r_out, 0.0],
                [r_sh, half_h],
                [r_in, half_h],
            ]
        else:
            # backing disc sits entirely behind the z=0 plane so its front
            # cap never covers the child rings' front surfaces
            pts = [
                [0.0, -2.0 * half_h],
                [r_out, -2.0 * half_h],
                [r_out, 0.0],
                [0.0, 0.0],
            ]
        profile = {
            "type": "polyline",
            "points": [[round(r, 6), round(z, 6)] for r, z in pts],
            "closed": True,
        }
        pp = self.base_part(part, OperatorCategory.REVOLVE, conf, notes)
        pp.axis = {"origin": [round(cx, 6), round(cy, 6), 0.0], "direction": direction}
        pp.profile = profile
        return pp

    def build_extrude(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        points = self.outline_obj(part)
        depth = self.extrusion_depth(part, notes)
        pp = self.base_part(part, OperatorCategory.EXTRUDE, conf, notes)
        pp.profile = {"type": "polyline", "points": points, "closed": True}
        pp.depth = round(depth, 6)
        return pp

    def build_primitive(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        cls = (part.part_class or "").lower()
        shape = "cube"
        for candidate in _PRIMITIVE_SHAPES:
            if candidate in cls or (candidate == "sphere" and "ball" in cls):
                shape = candidate
                break
        bx = part_bbox(self.graph, part)
        cx, cy = self.part_center_obj(part)
        sx = self.frame.length(bx[2] - bx[0])
        sy = self.frame.length(bx[3] - bx[1])
        sz = max(0.1 * max(sx, sy), 0.01)
        notes.append("depth extent of primitive is a generated hypothesis")
        pp = self.base_part(part, OperatorCategory.PRIMITIVE, conf, notes)
        pp.primitive_shape = shape
        pp.transform = {
            "location": [round(cx, 6), round(cy, 6), round(self.z_offset(part), 6)],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [round(sx, 6), round(sy, 6), round(sz, 6)],
        }
        return pp

    def build_sweep(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = ["sweep radius is a generated hypothesis"]
        prims = part_primitives(self.graph, part)
        prim = prims[0] if prims else None
        pts = outline_of(prim) if prim is not None else []
        if len(pts) >= 2:
            p0 = self.frame.point(*pts[0])
            p1 = self.frame.point(*pts[-1])
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            norm = math.hypot(dx, dy) or 1.0
            direction = [dx / norm, dy / norm, 0.0]
            origin = [round(p0[0], 6), round(p0[1], 6), 0.0]
        else:
            direction = [1.0, 0.0, 0.0]
            origin = [0.0, 0.0, 0.0]
        radius = 0.02
        circle = [
            [round(radius * math.cos(2 * math.pi * i / 12), 6),
             round(radius * math.sin(2 * math.pi * i / 12), 6)]
            for i in range(12)
        ]
        pp = self.base_part(part, OperatorCategory.SWEEP, conf, notes)
        pp.axis = {"origin": origin, "direction": [round(d, 6) for d in direction]}
        pp.profile = {"type": "circle", "points": circle, "closed": True}
        if prim is not None:
            pp.source_curve = prim.id
        return pp

    def build_radial_array(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        ids = set(part.primitive_ids)
        constraint = self._radial_constraint(part)
        count = len(part.primitive_ids) or 3
        source_part = part.id
        centre_uv = None
        if constraint is not None:
            count = int(constraint.params.get("count") or count)
            centre = constraint.params.get("center")
            if centre is not None:
                centre_uv = (float(centre[0]), float(centre[1]))
            proto = constraint.params.get("prototype")
            if proto is not None:
                owner = self._owner_of(str(proto), exclude=None)
                if owner is not None and owner != part.id:
                    source_part = owner
                    notes.append("array repeats prototype part '%s'" % owner)
        notes.append("array count %d from %s" % (
            count, "radial constraint" if constraint else "primitive count"))
        if centre_uv is not None:
            cx, cy = self.frame.point(*centre_uv)
        else:
            cx, cy = self.part_center_obj(part)
        pp = self.base_part(part, OperatorCategory.RADIAL_ARRAY, conf, notes)
        pp.source_part = source_part
        pp.count = max(count, 3)
        # full-circle repetition; the Blender builder interprets
        # angle_degrees as the total sweep
        pp.angle_degrees = 360.0
        pp.axis = {"origin": [round(cx, 6), round(cy, 6), 0.0], "direction": [0.0, 0.0, 1.0]}
        return pp

    def _prototype_outline(self, prim) -> List[List[float]]:
        """Closed 2D outline (object units) usable as an extrusion profile
        for one array copy; open curves become thin rectangles."""
        pts = outline_of(prim)
        if len(pts) >= 3 and prim.type in ELLIPSE_LIKE | CLOSED_FLAT:
            return [[round(x, 6), round(y, 6)]
                    for x, y in (self.frame.point(u, v) for u, v in pts)]
        if len(pts) >= 2:
            p0 = self.frame.point(*pts[0])
            p1 = self.frame.point(*pts[-1])
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                w = max(0.05 * length, 0.01)
                nx, ny = -dy / length * w, dx / length * w
                rect = [
                    (p0[0] + nx, p0[1] + ny),
                    (p1[0] + nx, p1[1] + ny),
                    (p1[0] - nx, p1[1] - ny),
                    (p0[0] - nx, p0[1] - ny),
                ]
                return [[round(x, 6), round(y, 6)] for x, y in rect]
        return []

    def build_array_prototype(self, part: SemanticPart, conf: float) -> PlanPart:
        """Extruded one-copy geometry for an array whose primitives all live
        on the array part itself: splits off the constraint's prototype curve
        as its own part so the array never references itself."""
        notes = ["prototype geometry extruded for radial array '%s'" % part.id]
        prims = part_primitives(self.graph, part)
        proto_prim = None
        constraint = self._radial_constraint(part)
        if constraint is not None:
            pid = constraint.params.get("prototype")
            proto_prim = next((p for p in prims if p.id == pid), None)
        if proto_prim is None and prims:
            proto_prim = prims[0]
        generated = (constraint is not None
                     and constraint.source == EvidenceSource.SEMANTIC_PRIOR)
        if generated:
            centre = constraint.params.get("center") or [self.frame.cx, self.frame.cy]
            cu, cv = float(centre[0]), float(centre[1])
            cx, cy = self.frame.point(cu, cv)
            ladder = self._radius_ladder(cu, cv)
            if len(ladder) >= 3:
                # Ignore near-duplicate outer-edge retraces and tiny highlight
                # ellipses.  The spoke runs from the first substantial hub
                # rung to the first substantial rim rung.
                outer = ladder[0]
                rim = next((r for r in ladder[1:] if r < 0.65 * outer),
                           0.48 * outer)
                hub = next((r for r in ladder if r < 0.65 * rim),
                           0.35 * rim)
                # Small overlaps prevent visible floating gaps where the
                # generated bar meets the revolved hub/rim surfaces.
                r0 = 0.75 * self.frame.length(hub)
                r1 = 1.05 * self.frame.length(rim)
            elif len(ladder) >= 2:
                r0 = 0.22 * self.frame.length(ladder[0])
                r1 = self.frame.length(ladder[1])
            else:
                r0, r1 = 0.12, 0.25
            if r1 <= 1.05 * r0:
                r1 = max(1.8 * r0, r0 + 0.08)
            half_width = max(0.012, 0.18 * (r1 - r0))
            points = [
                [round(cx + r0, 6), round(cy - half_width, 6)],
                [round(cx + r1, 6), round(cy - half_width, 6)],
                [round(cx + r1, 6), round(cy + half_width, 6)],
                [round(cx + r0, 6), round(cy + half_width, 6)],
            ]
            notes.append(
                "radial rectangle generated between observed hub and rim; "
                "trace fragments were not reliable prototype geometry")
        else:
            points = self._prototype_outline(proto_prim) if proto_prim is not None else []
        if proto_prim is not None and not generated:
            notes.append("prototype curve '%s'" % proto_prim.id)
        depth = self.extrusion_depth(part, notes)
        pp = PlanPart(
            id="%s_prototype" % part.id,
            operator=OperatorCategory.EXTRUDE,
            parent=part.parent_id,
            material=self.materials.get(part.id) or _material_from_appearance(part),
            visibility=part.visibility,
            evidence=EvidencedValue(
                value=OperatorCategory.EXTRUDE.value,
                source=(EvidenceSource.GENERATED_HYPOTHESIS if generated
                        else EvidenceSource.FITTED_FROM_OBSERVATION),
                confidence=round(max(0.3, conf - 0.2), 3),
                note="; ".join(notes),
            ),
        )
        pp.profile = {"type": "polyline", "points": points, "closed": True}
        pp.depth = round(depth, 6)
        return pp

    def build_mirror(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        ids = set(part.primitive_ids)
        source_part = part.id
        for c in self.graph.constraints:
            if c.type != ConstraintType.MIRROR_SYMMETRY or not ids.intersection(c.entities):
                continue
            for e in c.entities:
                owner = self._owner_of(e, exclude=part.id)
                if owner is not None:
                    source_part = owner
                    notes.append("mirrors part '%s' across the vertical axis" % owner)
                    break
            break
        cx, cy = self.part_center_obj(part)
        pp = self.base_part(part, OperatorCategory.MIRROR, conf, notes)
        pp.source_part = source_part
        pp.axis = {"origin": [round(cx, 6), round(cy, 6), 0.0], "direction": [1.0, 0.0, 0.0]}
        return pp

    def build_boolean(self, part: SemanticPart, conf: float) -> PlanPart:
        notes: List[str] = []
        target = self._boolean_target(part)
        if target is None:
            # A visually nested mark without a trustworthy larger solid is
            # surface evidence, not a valid destructive modeling operation.
            notes.append("boolean target unresolved; retained as surface detail")
            pp = self.base_part(part, OperatorCategory.DISPLACEMENT, conf, notes)
            prims = part_primitives(self.graph, part)
            if prims:
                pp.source_curve = prims[0].id
            return pp
        notes.append("cuts into part '%s'" % target)
        pp = self.base_part(part, OperatorCategory.BOOLEAN, conf, notes)
        pp.boolean_target = target
        pp.boolean_operation = "difference"
        pp.profile = {"type": "polyline", "points": self.outline_obj(part), "closed": True}
        pp.depth = 0.3
        return pp

    def build_freeform(self, part: SemanticPart, conf: float) -> PlanPart:
        notes = ["freeform shell; geometry largely a generated hypothesis"]
        pp = self.base_part(part, OperatorCategory.FREEFORM, conf, notes)
        pp.profile = {"type": "polyline", "points": self.outline_obj(part), "closed": True}
        return pp

    def build_loft(self, part: SemanticPart, conf: float) -> PlanPart:
        notes = ["loft between profiles; only primary profile recorded"]
        pp = self.base_part(part, OperatorCategory.LOFT, conf, notes)
        pp.profile = {"type": "polyline", "points": self.outline_obj(part), "closed": True}
        return pp

    def build_surface_detail(self, part: SemanticPart, op: OperatorCategory, conf: float) -> PlanPart:
        notes = ["surface detail carried by displacement/texture, not modelled geometry"]
        pp = self.base_part(part, op, conf, notes)
        prims = part_primitives(self.graph, part)
        if prims:
            pp.source_curve = prims[0].id
        cx, cy = self.part_center_obj(part)
        pp.transform = {"location": [round(cx, 6), round(cy, 6), 0.0]}
        return pp

    # -- lookups --------------------------------------------------------------

    def _radial_constraint(self, part: SemanticPart):
        ids = set(part.primitive_ids)
        candidates = [
            c for c in self.graph.constraints
            if c.type in _RADIAL_CONSTRAINTS
            and int(c.params.get("count") or 0) >= 3
            and ids.intersection(c.entities)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: (
            len(ids.intersection(c.entities)),
            int(c.params.get("prototype") in ids),
            float(c.confidence),
            tuple(sorted(c.entities)),
        ))

    def _owner_of(self, entity_id: str, exclude: Optional[str]) -> Optional[str]:
        for p in self.graph.parts:
            if exclude is not None and p.id == exclude:
                continue
            if entity_id in p.primitive_ids:
                return p.id
        return None

    def _boolean_target(self, part: SemanticPart) -> Optional[str]:
        """Smallest substantially larger non-cutter containing ``part``.

        Raw containment constraints can be duplicated or bidirectional across
        trace layers. Selecting their first partner created boolean cycles and
        same-size sibling targets. Geometry and operator role are authoritative
        here: a cutter must be <30% of a real container.
        """
        ids = set(part.primitive_ids)
        my_bbox = part_bbox(self.graph, part)
        my_area = max((my_bbox[2] - my_bbox[0]) * (my_bbox[3] - my_bbox[1]), 1e-12)
        my_cx = (my_bbox[0] + my_bbox[2]) / 2.0
        my_cy = (my_bbox[1] + my_bbox[3]) / 2.0
        constrained_owners = set()
        for c in self.graph.constraints:
            if c.type != ConstraintType.CONTAINMENT or not ids.intersection(c.entities):
                continue
            for e in c.entities:
                owner = self._owner_of(e, exclude=part.id)
                if owner is not None:
                    constrained_owners.add(owner)

        candidates: List[Tuple[int, float, str]] = []
        for other in self.graph.parts:
            if other.id == part.id:
                continue
            if other.selected_operator == OperatorCategory.BOOLEAN.value:
                continue
            bx = part_bbox(self.graph, other)
            if bx[0] <= my_cx <= bx[2] and bx[1] <= my_cy <= bx[3]:
                area = max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-12)
                if my_area / area < 0.3:
                    candidates.append(
                        (0 if other.id in constrained_owners else 1, area, other.id)
                    )
        return min(candidates)[2] if candidates else None


def build_plan(
    graph: SketchGraph,
    camera: CameraEstimate,
    depth: DepthEvidence,
    spec: InputSpec,
    cfg: PipelineConfig,
) -> ConstructionPlan:
    material_map: Dict[str, MaterialSpec] = {}
    material_source = "appearance_fallback"
    crop_rgba = _find_crop_rgba(spec)
    if crop_rgba is not None:
        try:
            material_map = materials_mod.estimate_materials(graph, crop_rgba, cfg)
            material_source = "image_sampling"
        except Exception:
            material_map = {}

    builder = _PlanBuilder(graph, camera, depth, material_map)
    builders = {
        OperatorCategory.REVOLVE: builder.build_revolve,
        OperatorCategory.EXTRUDE: builder.build_extrude,
        OperatorCategory.PRIMITIVE: builder.build_primitive,
        OperatorCategory.SWEEP: builder.build_sweep,
        OperatorCategory.RADIAL_ARRAY: builder.build_radial_array,
        OperatorCategory.MIRROR: builder.build_mirror,
        OperatorCategory.BOOLEAN: builder.build_boolean,
        OperatorCategory.FREEFORM: builder.build_freeform,
        OperatorCategory.LOFT: builder.build_loft,
    }

    plan_parts: List[PlanPart] = []
    for part in graph.parts:
        op, conf = builder.operator_of(part)
        if op in (OperatorCategory.DISPLACEMENT, OperatorCategory.TEXTURE_ONLY):
            plan_parts.append(builder.build_surface_detail(part, op, conf))
        elif op == OperatorCategory.RADIAL_ARRAY:
            arr = builder.build_radial_array(part, conf)
            if arr.source_part == part.id:
                # the repetition lives on the group part itself: split off a
                # standalone prototype part so the array has a valid,
                # non-self source
                proto = builder.build_array_prototype(part, conf)
                plan_parts.append(proto)
                arr.source_part = proto.id
                arr.evidence.note = (
                    (arr.evidence.note + "; " if arr.evidence.note else "")
                    + "array repeats prototype part '%s'" % proto.id
                )
            plan_parts.append(arr)
        else:
            plan_parts.append(builders[op](part, conf))

    scale_known = camera.scale.value is not None and spec.known_dimension
    uncertainty: Dict[str, Any] = {
        "rear_profile": "unobserved",
        "depth_source": depth.backend,
    }
    if scale_known:
        uncertainty["physical_scale"] = "user_supplied"
    else:
        uncertainty["physical_scale"] = "unknown"

    return ConstructionPlan(
        object_id=spec.target_label or "object",
        units="meters" if scale_known else "normalized",
        physical_width=float(camera.scale.value) if scale_known else None,
        parts=plan_parts,
        camera=camera,
        uncertainty=uncertainty,
        metadata={
            "stage": "construction_plan",
            "seed": cfg.seed,
            "material_source": material_source,
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_finite_seq(seq: Any, label: str, part_id: str, errors: List[str]) -> None:
    if not isinstance(seq, (list, tuple)):
        errors.append("part '%s': %s must be a sequence of numbers" % (part_id, label))
        return
    for v in seq:
        if not _is_finite_number(v):
            errors.append("part '%s': %s contains a non-finite value" % (part_id, label))
            return


def validate_plan(plan: ConstructionPlan) -> List[str]:
    """Schema-level sanity checks; returns human-readable errors ([] = valid)."""
    errors: List[str] = []

    if plan.units not in _ALLOWED_UNITS:
        errors.append("plan: units '%s' not in %s" % (plan.units, list(_ALLOWED_UNITS)))
    if plan.physical_width is not None and (
        not _is_finite_number(plan.physical_width) or plan.physical_width <= 0
    ):
        errors.append("plan: physical_width must be a positive finite number")

    ids = [p.id for p in plan.parts]
    if len(ids) != len(set(ids)):
        errors.append("plan: duplicate part ids: %s" % sorted({i for i in ids if ids.count(i) > 1}))
    idset = set(ids)

    for p in plan.parts:
        if p.parent is not None and p.parent not in idset:
            errors.append("part '%s': parent '%s' does not exist" % (p.id, p.parent))
        if p.source_part is not None and p.source_part not in idset:
            errors.append("part '%s': source_part '%s' does not exist" % (p.id, p.source_part))
        if p.boolean_target is not None and p.boolean_target not in idset:
            errors.append("part '%s': boolean_target '%s' does not exist" % (p.id, p.boolean_target))

    # parent cycles
    parent_of = {p.id: p.parent for p in plan.parts if p.parent is not None}
    for pid in parent_of:
        seen = set()
        cur: Optional[str] = pid
        while cur is not None and cur in parent_of:
            if cur in seen:
                errors.append("plan: parent cycle detected involving part '%s'" % pid)
                break
            seen.add(cur)
            cur = parent_of.get(cur)

    # Build dependencies must also be acyclic. Blender needs boolean targets
    # and array/mirror prototypes to exist before their dependent operation.
    dependency_of: Dict[str, List[str]] = {}
    for p in plan.parts:
        deps = []
        if p.boolean_target is not None:
            if p.boolean_target == p.id:
                errors.append("part '%s': boolean_target must not be the part itself" % p.id)
            deps.append(p.boolean_target)
        if p.source_part is not None:
            deps.append(p.source_part)
        dependency_of[p.id] = deps
    for pid in dependency_of:
        visiting = set()
        visited = set()

        def visit(cur: str) -> bool:
            if cur in visiting:
                return True
            if cur in visited:
                return False
            visiting.add(cur)
            for dep in dependency_of.get(cur, []):
                if dep in dependency_of and visit(dep):
                    return True
            visiting.remove(cur)
            visited.add(cur)
            return False

        if visit(pid):
            errors.append("plan: build dependency cycle detected involving part '%s'" % pid)

    for p in plan.parts:
        # finite transforms / axes / profiles
        for key, val in p.transform.items():
            _check_finite_seq(val, "transform.%s" % key, p.id, errors)
        if p.axis is not None:
            direction = p.axis.get("direction")
            origin = p.axis.get("origin")
            if direction is not None:
                _check_finite_seq(direction, "axis.direction", p.id, errors)
                if all(_is_finite_number(v) for v in direction):
                    if sum(float(v) ** 2 for v in direction) < 1e-12:
                        errors.append("part '%s': zero-length axis direction vector" % p.id)
            if origin is not None:
                _check_finite_seq(origin, "axis.origin", p.id, errors)
        if p.profile is not None:
            points = p.profile.get("points")
            if not isinstance(points, list) or len(points) < 3:
                errors.append(
                    "part '%s': profile must have at least 3 points (got %s)"
                    % (p.id, "none" if points is None else len(points))
                )
            else:
                for pt in points:
                    if (
                        not isinstance(pt, (list, tuple))
                        or len(pt) != 2
                        or not all(_is_finite_number(v) for v in pt)
                    ):
                        errors.append("part '%s': profile contains a non-finite point" % p.id)
                        break
                else:
                    if p.operator == OperatorCategory.REVOLVE:
                        for pt in points:
                            if float(pt[0]) < 0:
                                errors.append(
                                    "part '%s': revolve profile has a negative radius" % p.id
                                )
                                break
        if p.depth is not None and not _is_finite_number(p.depth):
            errors.append("part '%s': depth is not finite" % p.id)

        # operator-specific requirements
        if p.operator == OperatorCategory.EXTRUDE:
            if p.depth is None or not _is_finite_number(p.depth) or p.depth <= 0:
                errors.append("part '%s': extrude depth must be > 0" % p.id)
        elif p.operator == OperatorCategory.RADIAL_ARRAY:
            if p.count is None or not isinstance(p.count, int) or p.count < 3:
                errors.append("part '%s': radial_array count must be an integer >= 3" % p.id)
            if p.source_part is None:
                errors.append("part '%s': radial_array requires a source_part" % p.id)
            elif p.source_part == p.id:
                errors.append(
                    "part '%s': radial_array source_part must not be the part itself" % p.id)
            else:
                src = next((q for q in plan.parts if q.id == p.source_part), None)
                if src is not None and src.operator in (
                        OperatorCategory.RADIAL_ARRAY, OperatorCategory.MIRROR,
                        OperatorCategory.DISPLACEMENT, OperatorCategory.TEXTURE_ONLY):
                    errors.append(
                        "part '%s': radial_array source_part '%s' is not a "
                        "geometry prototype (operator %s)"
                        % (p.id, p.source_part, src.operator.value))
        elif p.operator == OperatorCategory.MIRROR:
            if p.source_part is None:
                errors.append("part '%s': mirror requires a source_part" % p.id)
        elif p.operator == OperatorCategory.BOOLEAN:
            if p.boolean_target is None:
                errors.append("part '%s': boolean requires a boolean_target" % p.id)
            if p.boolean_operation not in _BOOLEAN_OPS:
                errors.append(
                    "part '%s': boolean_operation '%s' not in %s"
                    % (p.id, p.boolean_operation, list(_BOOLEAN_OPS))
                )
        elif p.operator == OperatorCategory.PRIMITIVE:
            if p.primitive_shape not in _PRIMITIVE_SHAPES:
                errors.append(
                    "part '%s': primitive_shape '%s' not in %s"
                    % (p.id, p.primitive_shape, list(_PRIMITIVE_SHAPES))
                )
        elif p.operator == OperatorCategory.SWEEP:
            if not p.source_curve:
                errors.append("part '%s': sweep requires a source_curve" % p.id)

        # material ranges
        m = p.material
        for name, val in (
            ("base_color.r", m.base_color[0]),
            ("base_color.g", m.base_color[1]),
            ("base_color.b", m.base_color[2]),
            ("roughness", m.roughness),
            ("metallic", m.metallic),
            ("opacity", m.opacity),
            ("transmission", m.transmission),
        ):
            if not _is_finite_number(val) or not (0.0 <= float(val) <= 1.0):
                errors.append(
                    "part '%s': material %s must be in 0..1 (got %r)" % (p.id, name, val)
                )

    return errors
