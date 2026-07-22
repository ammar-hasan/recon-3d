"""Stage 11: camera estimation.

Strategy (no heavy models, single view):
- Circular features (concentric ellipse/circle systems such as wheels, rims,
  hubs, bottles) are treated as projected circles, not as 3D ellipses.
  Object tilt follows from the classic circle-unprojection:
  ``tilt = acos(minor_axis / major_axis)``.
- Perspective vs orthographic: under perspective, concentric circles of
  different radii project to ellipses whose centres drift apart and whose
  axis ratios vary with radius; under orthographic projection they stay
  perfectly concentric with identical axis ratios. When the signal is too
  weak to tell, we default to perspective with low confidence and say so.
- Focal length: no vanishing-point solver is available at this stage, so the
  configured default is reported at low confidence.
- Scale is UNKNOWN (confidence 0) unless the user supplied a known physical
  dimension, in which case units-per-normalised-width is derived from the
  segmentation bbox mapped through the crop transform.

Every output is an ``EvidencedValue`` with an honest source/confidence.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .config import PipelineConfig
from .part_geometry import (
    ELLIPSE_LIKE,
    graph_bbox,
    primitive_center,
    primitive_radii,
    primitive_rotation_deg,
)
from .schemas import (
    CameraEstimate,
    ConstraintType,
    CropMetadata,
    EvidencedValue,
    EvidenceSource,
    GeometricPrimitive,
    InputSpec,
    ProjectionType,
    SegmentationResult,
    SketchGraph,
)

_CONCENTRIC_CONSTRAINTS = {
    ConstraintType.CONCENTRIC,
    ConstraintType.SHARED_CENTER,
    ConstraintType.SHARED_AXIS,
}

#: centre drift (relative to mean radius) above which perspective is likely
_PERSPECTIVE_DRIFT_THRESHOLD = 0.02


def _constraint_groups(graph: SketchGraph) -> List[List[str]]:
    """Entity id groups linked by concentric/shared-centre constraints."""
    return [
        list(c.entities)
        for c in graph.constraints
        if c.type in _CONCENTRIC_CONSTRAINTS and len(c.entities) >= 2
    ]


def _circular_features(graph: SketchGraph) -> List[GeometricPrimitive]:
    """Ellipse/circle primitives likely to be projections of 3D circles.

    A primitive qualifies if it participates in a concentric constraint or is
    geometrically concentric with another ellipse-like primitive.
    """
    grouped = set()
    for ents in _constraint_groups(graph):
        grouped.update(ents)
    ellipse_like = [p for p in graph.primitives if p.type in ELLIPSE_LIKE]
    features = []
    for prim in ellipse_like:
        if prim.id in grouped:
            features.append(prim)
            continue
        cx, cy = primitive_center(prim)
        for other in ellipse_like:
            if other.id == prim.id:
                continue
            ox, oy = primitive_center(other)
            if math.hypot(cx - ox, cy - oy) < 0.02:
                features.append(prim)
                break
    return features


def _decide_projection(
    graph: SketchGraph, cfg: PipelineConfig, notes: List[str]
) -> Tuple[ProjectionType, float]:
    override = cfg.camera.assume_projection
    if override == "perspective":
        notes.append("projection forced to perspective by configuration")
        return ProjectionType.PERSPECTIVE, 0.7
    if override == "orthographic":
        notes.append("projection forced to orthographic by configuration")
        return ProjectionType.ORTHOGRAPHIC, 0.7

    prim_by_id = {p.id: p for p in graph.primitives}
    best_drift = 0.0
    for ents in _constraint_groups(graph):
        members = [prim_by_id[e] for e in ents if e in prim_by_id]
        members = [m for m in members if m.type in ELLIPSE_LIKE]
        if len(members) < 2:
            continue
        centres = [primitive_center(m) for m in members]
        radii = [primitive_radii(m) for m in members]
        mean_r = sum(max(r) for r in radii if r) / max(len(radii), 1)
        if mean_r <= 1e-9:
            continue
        drift = max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a in centres
            for b in centres
        ) / mean_r
        best_drift = max(best_drift, drift)
    if best_drift > _PERSPECTIVE_DRIFT_THRESHOLD:
        notes.append(
            "concentric circular features show centre drift %.4f > %.3f; "
            "consistent with perspective projection" % (best_drift, _PERSPECTIVE_DRIFT_THRESHOLD)
        )
        return ProjectionType.PERSPECTIVE, 0.7
    notes.append(
        "no measurable perspective cue (centre drift %.4f); "
        "defaulting to perspective with low confidence" % best_drift
    )
    return ProjectionType.PERSPECTIVE, 0.5


def _estimate_object_rotation(
    features: List[GeometricPrimitive], notes: List[str]
) -> EvidencedValue:
    """Tilt of circular features from ellipse axis ratios.

    Returns [rx_tilt_deg, ry_deg, rz_inplane_deg] where rx is the
    circle-unprojection tilt acos(minor/major) and rz is the in-plane
    ellipse rotation.
    """
    tilts = []
    weights = []
    rots = []
    for prim in features:
        rad = primitive_radii(prim)
        if rad is None:
            continue
        major, minor = max(rad), min(rad)
        if major <= 1e-9:
            continue
        ratio = min(1.0, minor / major)
        tilts.append(math.degrees(math.acos(ratio)))
        weights.append(max(prim.confidence, 1e-3) * major)
        rots.append(primitive_rotation_deg(prim))
    if not tilts:
        return EvidencedValue(
            value=None,
            unit="deg",
            source=EvidenceSource.UNKNOWN,
            confidence=0.0,
            note="no circular features available for circle-unprojection",
        )
    wsum = sum(weights)
    tilt = sum(t * w for t, w in zip(tilts, weights)) / wsum
    rot = sum(r * w for r, w in zip(rots, weights)) / wsum
    spread = math.sqrt(
        sum(w * (t - tilt) ** 2 for t, w in zip(tilts, weights)) / wsum
    )
    confidence = 0.75 if (len(tilts) >= 2 and spread < 5.0) else 0.55
    notes.append(
        "object tilt %.1f deg from ellipse axis ratios of %d circular "
        "feature(s) (spread %.1f deg)" % (tilt, len(tilts), spread)
    )
    return EvidencedValue(
        value=[tilt, 0.0, rot],
        unit="deg",
        source=EvidenceSource.ESTIMATED_FROM_CAMERA,
        confidence=confidence,
        note="tilt = acos(minor/major) of projected circles; "
        "assumes features are true circles in 3D",
    )


def _estimate_scale(
    graph: SketchGraph,
    seg: SegmentationResult,
    crop_meta: CropMetadata,
    spec: InputSpec,
    notes: List[str],
) -> EvidencedValue:
    if spec.known_dimension is None or spec.known_dimension <= 0:
        notes.append("no scale reference supplied; physical scale stays unknown")
        return EvidencedValue(
            value=None,
            unit="units_per_normalized_width",
            source=EvidenceSource.UNKNOWN,
            confidence=0.0,
            note="no reliable scale reference; physical scale unknown",
        )
    axis = spec.known_dimension_axis or "width"
    x0, y0, x1, y1 = seg.bbox
    if axis == "height":
        extent_px = float(y1 - y0)
        out = float(crop_meta.output_size[1])
    else:
        extent_px = float(x1 - x0)
        out = float(crop_meta.output_size[0])
    norm_extent = extent_px * crop_meta.scale / out if out > 0 else 0.0
    if norm_extent <= 1e-9:
        bx0, _, bx1, _ = graph_bbox(graph)
        norm_extent = max(bx1 - bx0, 1e-9)
        notes.append("segmentation bbox degenerate; used sketch-graph bbox for scale")
    value = spec.known_dimension / norm_extent
    note = "known %s %.6g over normalised extent %.4f" % (axis, spec.known_dimension, norm_extent)
    if axis not in ("width", "height", "diameter"):
        note += "; axis '%s' treated as width-like" % axis
    notes.append("scale from user-supplied dimension: " + note)
    return EvidencedValue(
        value=value,
        unit="units_per_normalized_width",
        source=EvidenceSource.USER_SUPPLIED,
        confidence=0.9,
        note=note,
    )


def estimate_camera(
    graph: SketchGraph,
    seg: SegmentationResult,
    crop_meta: CropMetadata,
    spec: InputSpec,
    cfg: PipelineConfig,
) -> CameraEstimate:
    notes: List[str] = []

    projection, proj_conf = _decide_projection(graph, cfg, notes)

    features = _circular_features(graph)
    object_rotation = _estimate_object_rotation(features, notes)

    focal = EvidencedValue(
        value=float(cfg.camera.default_focal_px),
        unit="px",
        source=EvidenceSource.UNKNOWN,
        confidence=0.3,
        note="configured default focal length; no vanishing points or known "
        "dimensions available to refine it",
    )
    notes.append("focal length assumed (default %.1f px); low confidence" % cfg.camera.default_focal_px)

    camera_rotation = EvidencedValue(
        value=[0.0, 0.0, 0.0],
        unit="deg",
        source=EvidenceSource.ESTIMATED_FROM_CAMERA,
        confidence=0.3,
        note="foreshortening attributed to object rotation; camera assumed frontal",
    )
    translation = EvidencedValue(
        value=None,
        unit="normalized",
        source=EvidenceSource.UNKNOWN,
        confidence=0.0,
        note="single view; camera translation not observable without scale",
    )
    scale = _estimate_scale(graph, seg, crop_meta, spec, notes)

    est = CameraEstimate(
        projection=projection,
        focal_length_px=focal,
        principal_point=(0.5, 0.5),
        rotation_euler_deg=camera_rotation,
        translation=translation,
        object_rotation_euler_deg=object_rotation,
        ground_plane=None,
        scale=scale,
        notes=notes,
    )
    est.notes.append("projection decision confidence %.2f" % proj_conf)
    est.notes.append("ground plane not estimated (no floor evidence used)")
    return est
