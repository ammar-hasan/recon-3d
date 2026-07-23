"""Stage 12: depth and surface-orientation evidence.

No heavy models are used. The fallback backend is a deterministic
shape-from-silhouette + shading heuristic:

- distance transform of the foreground mask gives a normalised "dome"
  (thickest image regions are assumed nearest the camera);
- the dome is optionally modulated by within-mask normalised luminance
  (brighter ~ more facing the light/camera) — a classic shape-from-shading
  crude cue;
- normals are derived from Sobel gradients of the depth map and encoded as
  an RGB normal map.

Outputs: ``depth.png`` (16-bit, 65535 = nearest) and ``normals.png`` in
``out_dir`` when ``cfg.depth.enabled``. Per-part mean relative depth is
reported in ``region_estimates`` at explicitly low confidence (< 0.6);
this evidence guides but never overrides observed vector evidence.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .part_geometry import part_primitives, rasterize_primitives
from .schemas import DepthEvidence, EvidencedValue, EvidenceSource, SketchGraph

#: blend weight of the luminance shading cue into the silhouette dome
_SHADING_WEIGHT = 0.25

_HEURISTIC_NOTE = (
    "shape-from-silhouette dome (distance transform, thickest = nearest) "
    "blended with normalised luminance shading at weight %.2f; relative "
    "depth only, never metric" % _SHADING_WEIGHT
)


def _load_inputs(crop_rgba_path: str, crop_mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    rgba = cv2.imread(crop_rgba_path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError("depth: could not read crop rgba image: %s" % crop_rgba_path)
    mask = cv2.imread(crop_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("depth: could not read crop mask image: %s" % crop_mask_path)
    if rgba.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgba, mask


def _relative_depth(rgba: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg = (mask > 127).astype(np.uint8)
    if fg.sum() == 0:
        raise ValueError("depth: foreground mask is empty")
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    dome = dist / dmax if dmax > 0 else np.zeros_like(dist)

    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    vals = gray[fg > 0]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo > 1e-6:
        lum_norm = (gray - lo) / (hi - lo)
    else:
        lum_norm = np.zeros_like(gray)

    depth = (1.0 - _SHADING_WEIGHT) * dome + _SHADING_WEIGHT * lum_norm
    depth = np.clip(depth, 0.0, 1.0) * fg
    return depth.astype(np.float32), fg


def _normals_from_depth(depth: np.ndarray, fg: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    strength = 0.5 * max(h, w)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(depth)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, 1e-9)
    rgb = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    rgb = ((rgb * 0.5 + 0.5) * 255.0).astype(np.uint8)
    rgb[fg == 0] = (128, 128, 255)  # flat background normal
    return rgb


def estimate_depth(
    crop_rgba_path: str,
    crop_mask_path: str,
    graph: SketchGraph,
    out_dir: str,
    cfg: PipelineConfig,
) -> DepthEvidence:
    depth_enabled = cfg.depth.enabled and cfg.depth.depth_enabled
    normals_enabled = cfg.depth.enabled and cfg.depth.normals_enabled
    if (not depth_enabled and not normals_enabled) or cfg.depth.backend == "none":
        return DepthEvidence(
            backend="none",
            confidence=0.0,
            notes=["depth estimation disabled by configuration"],
        )

    notes = [_HEURISTIC_NOTE]
    backend = cfg.depth.backend
    if backend not in ("auto", "shading", "silhouette_shading"):
        notes.append(
            "requested depth backend '%s' unavailable; using "
            "silhouette_shading fallback" % backend
        )
    backend = "silhouette_shading"

    rgba, mask = _load_inputs(crop_rgba_path, crop_mask_path)
    depth, fg = _relative_depth(rgba, mask)
    normals_rgb = _normals_from_depth(depth, fg) if normals_enabled else None

    os.makedirs(out_dir, exist_ok=True)
    depth_path = os.path.join(out_dir, "depth.png") if depth_enabled else None
    normals_path = os.path.join(out_dir, "normals.png") if normals_enabled else None
    if depth_path:
        cv2.imwrite(depth_path, (depth * 65535.0).astype(np.uint16))
        notes.append("wrote depth.png (16-bit, 65535 = nearest)")
    else:
        notes.append("depth output and per-part depth estimates disabled by ablation")
    if normals_path and normals_rgb is not None:
        cv2.imwrite(normals_path, normals_rgb[:, :, ::-1])  # RGB -> BGR
        notes.append("wrote normals.png")
    else:
        notes.append("normal output disabled by ablation")

    region_estimates: Dict[str, EvidencedValue] = {}
    h, w = depth.shape
    fg_count = int(fg.sum())
    for part in graph.parts if depth_enabled else []:
        prims = part_primitives(graph, part)
        if not prims:
            continue
        pmask = rasterize_primitives(prims, (h, w)) > 0
        pmask &= fg > 0
        count = int(pmask.sum())
        if count == 0:
            continue
        mean_depth = float(depth[pmask].mean())
        coverage = count / max(fg_count, 1)
        # low confidence by construction: larger regions are slightly more
        # trustworthy, but never trusted as measurements (< 0.6 always)
        confidence = float(min(0.55, max(0.2, 0.35 + 0.2 * min(1.0, coverage * 10.0))))
        region_estimates[part.id] = EvidencedValue(
            value=round(mean_depth, 4),
            unit="relative",
            source=EvidenceSource.ESTIMATED_FROM_DEPTH,
            confidence=confidence,
            note="mean normalised dome depth over part region; heuristic, "
            "guides but never overrides observed vector evidence",
        )

    return DepthEvidence(
        depth_path=depth_path,
        normals_path=normals_path,
        backend=(backend if depth_enabled and normals_enabled else
                 "depth_only" if depth_enabled else "normals_only"),
        region_estimates=region_estimates,
        confidence=0.45,
        notes=notes,
    )
