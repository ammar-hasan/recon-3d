"""Calibrated silhouette visual-hull reconstruction.

This module is deliberately limited to user-supplied relative camera
azimuths.  With uncalibrated views the correspondence is ambiguous, so the
pipeline retains the source-labelled hypotheses instead of silently treating
an estimated pose as measured geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage import measure

from .config import PipelineConfig
from .schemas import (ConstructionPlan, EvidenceSource, EvidencedValue,
                      MaterialSpec, MultiViewResult, OperatorCategory, PlanPart,
                      ProjectionType, SegmentationResult)


@dataclass
class _View:
    view_id: str
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    camera_azimuth_deg: float
    pixels_per_unit: float


def _read_mask(path: str, dilation: int) -> Optional[np.ndarray]:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 127).astype(np.uint8)
    if dilation:
        size = 2 * dilation + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    return mask


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _calibrated_views(result: MultiViewResult, primary: SegmentationResult,
                      cfg: PipelineConfig) -> List[_View]:
    dilation = cfg.multiview.visual_hull_mask_dilation_px
    pmask = _read_mask(primary.mask_path, dilation)
    if pmask is None:
        return []
    pbbox = tuple(int(v) for v in primary.bbox)
    pwidth = max(1.0, float(pbbox[2] - pbbox[0]))
    views = [_View("view_000", pmask, pbbox, 0.0, pwidth)]
    for observation in result.observations:
        pose = result.relative_camera_poses.get(observation.view_id)
        if (observation.status != "success" or not observation.mask_path
                or pose is None or pose.source != EvidenceSource.USER_SUPPLIED
                or not isinstance(pose.value, list) or len(pose.value) != 3):
            continue
        mask = _read_mask(observation.mask_path, dilation)
        if mask is None:
            continue
        bbox = observation.object_bbox or _bbox(mask)
        scale = observation.scale_to_primary.value
        if bbox is None or not isinstance(scale, (int, float)) or scale <= 0:
            continue
        views.append(_View(
            observation.view_id, mask, tuple(int(v) for v in bbox),
            float(pose.value[1]), pwidth / float(scale)))
    return views


def _box_symmetry_hypothesis(plan: ConstructionPlan, views: List[_View],
                             cfg: PipelineConfig) -> Optional[_View]:
    """Mirror the primary support about an observed box face direction.

    With only two silhouettes, their maximal visual hull is systematically
    too large in the unobserved direction. For enclosure-like semantics, a
    substantially narrower view near 45 degrees is evidence that the camera
    is facing a box side. Mirroring the primary support around that direction
    is a bounded, auditable compactness hypothesis; it is never described as
    observed evidence and can be disabled for ablation.
    """
    if (not cfg.multiview.visual_hull_box_symmetry_prior_enabled
            or len(views) != 2):
        return None
    label = plan.object_id.lower()
    if not any(token in label for token in ("box", "enclosure", "cabinet", "case")):
        return None
    primary, secondary = views
    primary_width = (primary.bbox[2] - primary.bbox[0]) / primary.pixels_per_unit
    secondary_width = ((secondary.bbox[2] - secondary.bbox[0])
                       / secondary.pixels_per_unit)
    angle = secondary.camera_azimuth_deg
    if not (25.0 <= abs(angle) <= 65.0 and secondary_width < 0.9 * primary_width):
        return None
    return _View(
        "generated_box_symmetry", primary.mask, primary.bbox,
        2.0 * angle, primary.pixels_per_unit)


def _inside(view: _View, x: np.ndarray, y: np.ndarray,
            z: np.ndarray) -> np.ndarray:
    # A +azimuth camera orbit is the same silhouette as a -azimuth object
    # yaw in the fixed primary camera.
    yaw = math.radians(-view.camera_azimuth_deg)
    projected_x = math.cos(yaw) * x + math.sin(yaw) * z
    x0, y0, x1, y1 = view.bbox
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    px = np.rint(cx + projected_x * view.pixels_per_unit).astype(np.int32)
    py = np.rint(cy - y * view.pixels_per_unit).astype(np.int32)
    h, w = view.mask.shape
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    result = np.zeros(px.shape, dtype=bool)
    result[valid] = view.mask[py[valid], px[valid]] > 0
    return result


def _project_occupancy(occupied: np.ndarray, coords: Tuple[np.ndarray, ...],
                       view: _View) -> np.ndarray:
    x, y, z = (axis[occupied] for axis in coords)
    yaw = math.radians(-view.camera_azimuth_deg)
    projected_x = math.cos(yaw) * x + math.sin(yaw) * z
    x0, y0, x1, y1 = view.bbox
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    px = np.rint(cx + projected_x * view.pixels_per_unit).astype(np.int32)
    py = np.rint(cy - y * view.pixels_per_unit).astype(np.int32)
    rendered = np.zeros_like(view.mask, dtype=np.uint8)
    valid = ((px >= 0) & (px < rendered.shape[1])
             & (py >= 0) & (py < rendered.shape[0]))
    rendered[py[valid], px[valid]] = 1
    # One voxel spans several image pixels. Expand point samples to their
    # conservative projected footprint before evaluating the silhouette.
    voxel_px = max(1, int(math.ceil(view.pixels_per_unit
                                    / max(1, occupied.shape[0] - 1))))
    kernel = np.ones((2 * voxel_px + 1, 2 * voxel_px + 1), np.uint8)
    return cv2.dilate(rendered, kernel)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0


def _mesh(occupied: np.ndarray, ranges: Tuple[np.ndarray, ...]
          ) -> Tuple[List[List[float]], List[List[int]]]:
    padded = np.pad(occupied.astype(np.float32), 1)
    steps = tuple(float(axis[1] - axis[0]) for axis in ranges)
    vertices, faces, _, _ = measure.marching_cubes(
        padded, level=0.5, spacing=steps, allow_degenerate=False)
    origin = np.asarray([axis[0] - step for axis, step in zip(ranges, steps)])
    vertices += origin
    return (np.round(vertices, 6).tolist(),
            faces.astype(np.int32).tolist())


def augment_plan_with_visual_hull(
    plan: ConstructionPlan,
    result: MultiViewResult,
    primary: SegmentationResult,
    cfg: PipelineConfig,
) -> Tuple[ConstructionPlan, MultiViewResult, bool]:
    """Add a measured mesh when at least one calibrated secondary view exists.

    Returns ``(plan, result, used)``. Existing parametric parts stay in the
    plan as editable, source-labelled guides but are hidden from render/export.
    """
    if (not result.enabled or not cfg.multiview.visual_hull_enabled):
        return plan, result, False
    views = _calibrated_views(result, primary, cfg)
    if len(views) < 2:
        return plan, result, False

    grid = cfg.multiview.visual_hull_grid_size
    pb = views[0].bbox
    height = max(1.0, float(pb[3] - pb[1])) / max(1.0, float(pb[2] - pb[0]))
    depth = cfg.multiview.visual_hull_depth_extent
    ranges = (
        np.linspace(-0.5, 0.5, grid, dtype=np.float32),
        np.linspace(-height / 2.0, height / 2.0, grid, dtype=np.float32),
        np.linspace(-depth / 2.0, depth / 2.0, grid, dtype=np.float32),
    )
    coords = np.meshgrid(*ranges, indexing="ij")
    occupied = np.ones((grid, grid, grid), dtype=bool)
    hypothesis_view = _box_symmetry_hypothesis(plan, views, cfg)
    carving_views = views + ([hypothesis_view] if hypothesis_view else [])
    for view in carving_views:
        occupied &= _inside(view, *coords)
    if int(occupied.sum()) < 8:
        result.warnings.append(
            "calibrated visual hull rejected: silhouette intersection is empty")
        return plan, result, False

    scores = {
        view.view_id: _iou(view.mask, _project_occupancy(occupied, coords, view))
        for view in views
    }
    if (scores["view_000"] < cfg.multiview.visual_hull_min_primary_iou
            or any(scores[v.view_id]
                   < cfg.multiview.visual_hull_min_secondary_iou
                   for v in views[1:])):
        result.warnings.append(
            "calibrated visual hull rejected: observed-view reprojection below "
            "threshold (%s)" % ", ".join(
                "%s=%.3f" % item for item in sorted(scores.items())))
        return plan, result, False

    vertices, faces = _mesh(occupied, ranges)
    updated = plan.model_copy(deep=True)
    if updated.camera is not None:
        updated.camera.projection = ProjectionType.ORTHOGRAPHIC
        updated.camera.notes.append(
            "calibrated visual hull uses an orthographic camera because only "
            "relative view azimuths, not full intrinsics/extrinsics, were supplied")
    source_ids = [part.id for part in updated.parts if part.render_visible]
    material = next((part.material.model_copy(deep=True) for part in updated.parts
                     if part.render_visible), None)
    for part in updated.parts:
        part.render_visible = False
    hull = PlanPart(
        id="multiview_visual_hull",
        operator=OperatorCategory.FREEFORM,
        profile={"type": "mesh", "vertices": vertices, "faces": faces},
        material=material if material is not None else MaterialSpec(),
        evidence=EvidencedValue(
            value={"view_ids": [view.view_id for view in views],
                   "reprojection_iou": scores},
            source=EvidenceSource.FITTED_FROM_OBSERVATION,
            confidence=min(scores.values()),
            note="voxel-carved intersection of calibrated observed silhouettes",
        ),
    )
    updated.parts.append(hull)
    updated.metadata = dict(updated.metadata)
    updated.metadata["multiview_visual_hull"] = {
        "used": True,
        "calibrated_view_count": len(views),
        "grid_size": grid,
        "depth_extent": depth,
        "occupied_voxels": int(occupied.sum()),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "observed_view_reprojection_iou": scores,
        "source_edit_guide_parts": source_ids,
        "primary_observed_geometry_overwritten": False,
        "generated_symmetry_hypothesis": (
            None if hypothesis_view is None else {
                "view_id": hypothesis_view.view_id,
                "camera_azimuth_deg": hypothesis_view.camera_azimuth_deg,
                "source": EvidenceSource.SEMANTIC_PRIOR.value,
                "note": ("primary support mirrored about the narrower "
                         "enclosure face direction; not an observed view"),
            }),
    }
    result.joint_optimization = dict(result.joint_optimization)
    result.joint_optimization["visual_hull"] = dict(
        updated.metadata["multiview_visual_hull"])
    return updated, result, True
