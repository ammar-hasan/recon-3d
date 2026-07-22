"""Stage 13: construction-method classification.

Assigns each SemanticPart a ranked list of OperatorCandidates plus a
selected_operator. Rules combine semantic class keywords, primitive shapes,
constraint structure (concentric / containment / radial / mirror) and depth
evidence. Every OperatorCategory is covered by at least one rule; a low
confidence freeform fallback guarantees a non-empty candidate list.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .config import PipelineConfig
from .part_geometry import (
    CLOSED_FLAT,
    ELLIPSE_LIKE,
    OPEN_PATH,
    part_bbox,
    part_primitives,
    primitive_bbox,
    primitive_center,
)
from .schemas import (
    ConstraintType,
    DepthEvidence,
    GeometricConstraint,
    GeometricPrimitive,
    OperatorCandidate,
    OperatorCategory,
    SemanticPart,
    SketchGraph,
)

_REVOLVE_CLASSES = (
    "tyre", "tire", "rim", "hub", "wheel", "bottle", "bowl", "knob",
    "vase", "shade", "pulley", "cap", "barrel",
    "disk", "disc",
)
_EXTRUDE_CLASSES = (
    "plate", "bracket", "logo", "sign", "panel", "base", "lid", "seat",
    "backrest", "leg", "post", "slat", "tooth", "plaque", "gear",
    "flange",
)
_SWEEP_CLASSES = ("handle", "pipe", "arm", "rail", "cable", "tube", "hose")
_BOOLEAN_CLASSES = (
    "hole", "cutout", "cut-out", "vent", "socket", "recess", "bore", "slot",
)
_PRIMITIVE_CLASSES = ("box", "block", "case", "enclosure", "ball", "sphere",
                      "cone", "capsule", "torus")
_DETAIL_CLASSES = (
    "tread", "engraving", "embossing", "grip", "knurl", "pattern", "texture", "detail",
)

_RADIAL_CONSTRAINTS = {ConstraintType.RADIAL_SYMMETRY, ConstraintType.ROTATIONAL_REPETITION}
_CONCENTRIC_CONSTRAINTS = {
    ConstraintType.CONCENTRIC,
    ConstraintType.SHARED_CENTER,
    ConstraintType.SHARED_AXIS,
}


def _constraints_touching(
    graph: SketchGraph, part: SemanticPart, types
) -> List[GeometricConstraint]:
    ids = set(part.primitive_ids)
    return [
        c for c in graph.constraints if c.type in types and ids.intersection(c.entities)
    ]


def _concentric_partner_count(graph: SketchGraph, prims: List[GeometricPrimitive]) -> int:
    """How many ellipse-like primitives are concentric with this part's."""
    ellipse_like = [p for p in graph.primitives if p.type in ELLIPSE_LIKE]
    own = {p.id for p in prims}
    partners = set()
    for prim in prims:
        if prim.type not in ELLIPSE_LIKE:
            continue
        cx, cy = primitive_center(prim)
        for c in graph.constraints:
            if c.type in _CONCENTRIC_CONSTRAINTS and prim.id in c.entities:
                partners.update(e for e in c.entities if e != prim.id)
        for other in ellipse_like:
            if other.id in own:
                continue
            ox, oy = primitive_center(other)
            if math.hypot(cx - ox, cy - oy) < 0.02:
                partners.add(other.id)
    return len(partners)


def _has_depth_evidence(depth: DepthEvidence, part_id: str) -> bool:
    ev = depth.region_estimates.get(part_id)
    return ev is not None and ev.value is not None and ev.confidence > 0.0


def _contained_by(
    graph: SketchGraph, part: SemanticPart, prims: List[GeometricPrimitive]
) -> Optional[str]:
    """Id of another part whose closed region contains this one, if any.

    A contained part only qualifies as a boolean cutter when it is
    substantially smaller than its container (a hole inside a plate); large
    nested parts are structure, not cutouts.
    """
    my_bbox = part_bbox(graph, part)
    my_area = max((my_bbox[2] - my_bbox[0]) * (my_bbox[3] - my_bbox[1]), 1e-12)
    my_cx = (my_bbox[0] + my_bbox[2]) / 2.0
    my_cy = (my_bbox[1] + my_bbox[3]) / 2.0

    # constraint-driven containment first
    for c in _constraints_touching(graph, part, {ConstraintType.CONTAINMENT}):
        own = set(part.primitive_ids)
        for other_part in graph.parts:
            if other_part.id == part.id:
                continue
            if not (own.intersection(c.entities)
                    and set(other_part.primitive_ids).intersection(c.entities)):
                continue
            other_bbox = part_bbox(graph, other_part)
            other_area = max(
                (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1]),
                1e-12)
            if my_area / other_area < 0.3:
                return other_part.id

    # geometric containment inside another part's closed primitive
    for other_part in graph.parts:
        if other_part.id == part.id:
            continue
        for prim in part_primitives(graph, other_part):
            if prim.type not in CLOSED_FLAT and prim.type not in ELLIPSE_LIKE:
                continue
            bx = primitive_bbox(prim)
            if not (bx[0] <= my_cx <= bx[2] and bx[1] <= my_cy <= bx[3]):
                continue
            area = max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-12)
            if my_area / area < 0.3:
                return other_part.id
    return None


def classify_operators(graph: SketchGraph, depth: DepthEvidence, cfg: PipelineConfig) -> SketchGraph:
    g = graph.model_copy(deep=True)

    for part in g.parts:
        prims = part_primitives(g, part)
        cls = (part.part_class or "").lower()
        cands: Dict[OperatorCategory, float] = {}

        def add(op: OperatorCategory, conf: float) -> None:
            cands[op] = max(cands.get(op, 0.0), min(1.0, conf))

        has_depth = _has_depth_evidence(depth, part.id)
        ellipse_like = [p for p in prims if p.type in ELLIPSE_LIKE]
        flat_closed = [p for p in prims if p.type in CLOSED_FLAT]
        open_paths = [p for p in prims if p.type in OPEN_PATH]

        # --- revolve: concentric circle/ellipse systems, rotational parts ---
        if any(k in cls for k in _REVOLVE_CLASSES):
            add(OperatorCategory.REVOLVE, 0.85)
        n_concentric = _concentric_partner_count(g, prims)
        # ring-like = dominated by circle/ellipse primitives and small (a
        # ring part owns one ring, possibly re-traced a few times); mixed
        # scrap bags (e.g. unclassified details) must not qualify
        ring_like = (
            bool(ellipse_like)
            and len(prims) <= 4
            and 2 * len(ellipse_like) >= len(prims)
        )
        concentric_ring = ring_like and n_concentric >= 1
        if concentric_ring:
            add(OperatorCategory.REVOLVE, 0.8)
            part.notes.append(
                "part of a concentric circle/ellipse system; likely surface of revolution"
            )

        # --- extrude: flat closed region, optionally with uniform depth ---
        if any(k in cls for k in _EXTRUDE_CLASSES):
            add(OperatorCategory.EXTRUDE, 0.8)
        if flat_closed and not ellipse_like:
            conf = 0.65 + (0.05 if has_depth else 0.0)
            add(OperatorCategory.EXTRUDE, conf)

        # --- sweep: elongated curved path of ~constant width ---
        if any(k in cls for k in _SWEEP_CLASSES):
            add(OperatorCategory.SWEEP, 0.8)
        for prim in open_paths:
            bx = primitive_bbox(prim)
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]
            aspect = max(w, h) / max(min(w, h), 1e-9)
            if aspect > 4.0:
                add(OperatorCategory.SWEEP, 0.6)

        # --- primitive: shapes directly representable as 3D primitives ---
        if any(k in cls for k in _PRIMITIVE_CLASSES):
            add(OperatorCategory.PRIMITIVE, 0.7)

        # --- boolean: holes / cutouts contained in another part ---
        container = _contained_by(g, part, prims)
        if any(k in cls for k in _BOOLEAN_CLASSES):
            add(OperatorCategory.BOOLEAN, 0.85 if container else 0.7)
        elif container is not None:
            add(OperatorCategory.BOOLEAN, 0.7)
        if container is not None:
            part.notes.append("contained inside part '%s'; candidate boolean cutter" % container)

        # --- radial_array: repeated angular copies -------------------------
        # Only a genuine rotational repetition (3+ observed copies of a
        # prototype) justifies an array. A radial_symmetry constraint between
        # concentric rings carries no repetition count and must never turn a
        # ring into an array of itself.
        radial = [
            c for c in _constraints_touching(g, part, _RADIAL_CONSTRAINTS)
            if int(c.params.get("count") or 0) >= 3
        ]
        # parts whose primitives are concentric circles/ellipses centred on
        # the system centre are surfaces of revolution: revolve outranks
        # radial_array for them
        if radial and not concentric_ring:
            add(OperatorCategory.RADIAL_ARRAY, 0.85)
            count = radial[0].params.get("count")
            part.notes.append(
                "radial repetition constraint (count=%s); array of a prototype" % count
            )
        elif radial and concentric_ring:
            part.notes.append(
                "radial constraint touches a concentric ring part; "
                "revolve kept over radial_array"
            )

        # --- mirror: mirrored halves ---
        if _constraints_touching(g, part, {ConstraintType.MIRROR_SYMMETRY}):
            add(OperatorCategory.MIRROR, 0.8)
        elif any(k in cls for k in ("left", "right", "mirror", "wing")):
            add(OperatorCategory.MIRROR, 0.55)

        # --- loft: several non-concentric similar profiles ---
        profiles = [p for p in prims if p.type in ELLIPSE_LIKE or p.type in CLOSED_FLAT]
        if len(profiles) >= 2:
            centres = [primitive_center(p) for p in profiles]
            spread = max(
                math.hypot(a[0] - b[0], a[1] - b[1])
                for a in centres
                for b in centres
            )
            if spread > 0.03:
                add(OperatorCategory.LOFT, 0.55)

        # --- displacement / texture-only: small surface detail ---
        parent = next((p for p in g.parts if p.id == part.parent_id), None)
        gear_tooth_already_in_outline = (
            "tooth" in cls
            and parent is not None
            and "gear" in (parent.part_class or "").lower()
        )
        crate_role_already_in_outline = (
            cls in {"corner_post", "side_slat"}
            and parent is not None
            and "bottom_panel" in (parent.part_class or "").lower()
        )
        if gear_tooth_already_in_outline or crate_role_already_in_outline:
            # The gear root owns the directly observed toothed silhouette.
            # A guided tooth representative is semantic evidence, not an
            # additional solid (which would duplicate one trace fragment and
            # can create a large spur outside the gear body).
            add(OperatorCategory.DISPLACEMENT, 0.95)
            part.notes.append(
                "semantic role geometry is already encoded by the observed root silhouette"
            )
        elif any(k in cls for k in _DETAIL_CLASSES):
            # Aggregated residual traces are explicitly non-structural. They
            # must not become a noisy array/boolean merely because one member
            # touches a broad constraint.
            add(OperatorCategory.DISPLACEMENT, 0.95)
            add(OperatorCategory.TEXTURE_ONLY, 0.5)
        else:
            bx = part_bbox(g, part)
            if (bx[2] - bx[0]) * (bx[3] - bx[1]) < 0.002:
                add(OperatorCategory.DISPLACEMENT, 0.4)

        # --- freeform: irregular shells; also universal low fallback ---
        for prim in flat_closed:
            if prim.fit_error > cfg.primitives.max_fit_error_norm:
                add(OperatorCategory.FREEFORM, 0.45)
        add(OperatorCategory.FREEFORM, 0.2)

        ordered = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0].value))
        part.construction_candidates = [
            OperatorCandidate(operator=op, confidence=round(conf, 3))
            for op, conf in ordered
        ]
        part.selected_operator = ordered[0][0].value
        part.notes.append(
            "selected operator '%s' (confidence %.2f)"
            % (ordered[0][0].value, ordered[0][1])
        )

    return g
