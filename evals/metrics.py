"""Pure, deterministic metric functions used across all eval levels.

Everything here is numpy/cv2/skimage only - no recon3d imports - so the
metrics can be unit-tested in isolation and reused by stage and e2e evals.

Conventions
-----------
- Binary masks are 2D arrays; anything ``> 0`` counts as foreground unless a
  function documents otherwise.
- "normalized" coordinates are 0..1 over the image/crop, origin top-left.
- Contours are ``(N, 2)`` float arrays of ``(x, y)`` points.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _as_bool_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Coerce an array to a 2D boolean mask, validating the shape."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError("%s must be a 2D array, got shape %r" % (name, arr.shape))
    return arr > 0


def _as_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    """Coerce an array to an ``(N, 2)`` float array of xy points."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("%s must have shape (N, 2), got %r" % (name, arr.shape))
    if arr.shape[0] == 0:
        raise ValueError("%s must contain at least one point" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains non-finite values" % name)
    return arr


# ---------------------------------------------------------------------------
# mask metrics
# ---------------------------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two binary masks.

    Returns 1.0 when both masks are empty (two empty masks agree perfectly),
    0.0 when exactly one is empty.
    """
    a = _as_bool_mask(mask_a, "mask_a")
    b = _as_bool_mask(mask_b, "mask_b")
    if a.shape != b.shape:
        raise ValueError("mask shapes differ: %r vs %r" % (a.shape, b.shape))
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def mask_precision_recall_f1(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Pixel precision/recall/F1 of ``pred`` foreground against ``ref``.

    A score of 1.0 is returned for precision (recall) when the prediction
    (reference) is empty and there is nothing to be wrong about.
    """
    p = _as_bool_mask(pred, "pred")
    r = _as_bool_mask(ref, "ref")
    if p.shape != r.shape:
        raise ValueError("mask shapes differ: %r vs %r" % (p.shape, r.shape))
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract the 1-px boundary band of a binary mask as a boolean mask."""
    m = _as_bool_mask(mask).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel)
    boundary = m - eroded
    return boundary > 0


def boundary_f_score(mask_a: np.ndarray, mask_b: np.ndarray,
                     tolerance_px: float = 2.0) -> float:
    """Boundary F-score between two binary masks.

    Boundary pixels of each mask are matched against the other mask's boundary
    within ``tolerance_px`` (distance transform). Returns the F1 of boundary
    precision and recall. Two empty masks score 1.0; one empty scores 0.0.
    """
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be >= 0")
    a = _as_bool_mask(mask_a, "mask_a")
    b = _as_bool_mask(mask_b, "mask_b")
    if a.shape != b.shape:
        raise ValueError("mask shapes differ: %r vs %r" % (a.shape, b.shape))
    ba = mask_boundary(a)
    bb = mask_boundary(b)
    na, nb = int(ba.sum()), int(bb.sum())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    # distance from every pixel to the nearest boundary pixel of the other mask
    dist_to_bb = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_ba = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_bb[ba] <= tolerance_px).mean())
    recall = float((dist_to_ba[bb] <= tolerance_px).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def hole_count(mask: np.ndarray) -> int:
    """Number of enclosed background holes inside the mask foreground."""
    m = _as_bool_mask(mask).astype(np.uint8)
    background = (m == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(background, connectivity=4)
    holes = 0
    h, w = m.shape
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        if (xs.min() > 0 and ys.min() > 0 and xs.max() < w - 1 and ys.max() < h - 1):
            holes += 1
    return holes


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight ``(x0, y0, x1, y1)`` bbox of the foreground, or None if empty."""
    m = _as_bool_mask(mask)
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_coverage(mask: np.ndarray, bbox: Sequence[float]) -> float:
    """Fraction of the mask's foreground that lies inside ``bbox`` (x0,y0,x1,y1)."""
    m = _as_bool_mask(mask)
    total = int(m.sum())
    if total == 0:
        return 0.0
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = m.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(m[y0:y1, x0:x1].sum()) / float(total)


# ---------------------------------------------------------------------------
# contour / chamfer metrics
# ---------------------------------------------------------------------------

def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask as ``(N, 2)`` xy float array."""
    m = _as_bool_mask(mask).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float64)


def _chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    a = _as_points(points_a, "points_a")
    b = _as_points(points_b, "points_b")
    # cdist-free chunked implementation to bound memory on large contours
    def _mean_nn(src: np.ndarray, dst: np.ndarray) -> float:
        total = 0.0
        for i in range(0, src.shape[0], 4096):
            chunk = src[i:i + 4096]
            d2 = ((chunk[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
            total += float(np.sqrt(d2.min(axis=1)).sum())
        return total / src.shape[0]
    return 0.5 * (_mean_nn(a, b) + _mean_nn(b, a))


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray,
                     normalize_by: Optional[float] = None) -> float:
    """Symmetric Chamfer distance between two point sets / contours.

    ``normalize_by`` divides the raw distance (e.g. object diagonal in pixels,
    or image width for normalized-space comparisons). Returns the raw mean
    distance when ``normalize_by`` is None.
    """
    d = _chamfer(points_a, points_b)
    if normalize_by is not None:
        if normalize_by <= 0:
            raise ValueError("normalize_by must be > 0")
        d /= float(normalize_by)
    return d


def chamfer_distance_masks(mask_a: np.ndarray, mask_b: np.ndarray,
                           normalize_by: Optional[str] = "diagonal") -> float:
    """Chamfer distance between the outer contours of two masks.

    ``normalize_by``: "diagonal" normalises by the reference (mask_b) object
    bbox diagonal; "image" by the image diagonal; None gives raw pixels.
    """
    ca = largest_contour(mask_a)
    cb = largest_contour(mask_b)
    if ca is None and cb is None:
        return 0.0
    if ca is None or cb is None:
        return float("inf")
    norm: Optional[float] = None
    if normalize_by == "diagonal":
        bb = mask_bbox(mask_b)
        if bb is None:
            return float("inf")
        norm = math.hypot(bb[2] - bb[0], bb[3] - bb[1])
        if norm <= 0:
            norm = 1.0
    elif normalize_by == "image":
        h, w = _as_bool_mask(mask_b).shape
        norm = math.hypot(w, h)
    elif normalize_by is not None:
        raise ValueError("unknown normalize_by: %r" % (normalize_by,))
    return chamfer_distance(ca, cb, normalize_by=norm)


# ---------------------------------------------------------------------------
# rasterization helpers
# ---------------------------------------------------------------------------

def rasterize_polygon(points_norm: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Rasterize a normalized-coordinate polygon into a uint8 mask (0/255).

    ``points_norm`` is ``(N, 2)`` with x,y in 0..1; ``size`` is (width, height).
    Degenerate polygons (< 3 points) produce an empty mask.
    """
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive, got %r" % (size,))
    pts = _as_points(points_norm, "points_norm")
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] < 3:
        return mask
    px = np.round(pts * np.array([w, h])).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [px], 255)
    return mask


def rasterize_paths(paths: Sequence[np.ndarray], size: Tuple[int, int],
                    holes: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """Rasterize several normalized polygons into one mask.

    ``paths`` are filled with 255; ``holes`` are then carved out with 0, which
    reproduces SVG even-odd/containment semantics for simple nestings.
    """
    w, h = int(size[0]), int(size[1])
    mask = np.zeros((h, w), np.uint8)
    for poly in paths:
        mask = np.maximum(mask, rasterize_polygon(poly, size))
    for hole in (holes or []):
        mask[mask_boundary(rasterize_polygon(hole, size))] = 255  # keep rims
        carved = rasterize_polygon(hole, size)
        mask[carved > 0] = 0
    return mask


def polyline_to_mask(points_norm: np.ndarray, size: Tuple[int, int],
                     thickness_px: int = 1) -> np.ndarray:
    """Rasterize an open normalized polyline with a stroke thickness."""
    w, h = int(size[0]), int(size[1])
    pts = _as_points(points_norm, "points_norm")
    mask = np.zeros((h, w), np.uint8)
    px = np.round(pts * np.array([w, h])).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [px], isClosed=False, color=255,
                  thickness=max(1, int(thickness_px)))
    return mask


# ---------------------------------------------------------------------------
# coordinate round-trip helpers
# ---------------------------------------------------------------------------

def round_trip_errors(points_xy: np.ndarray, to_crop, to_original) -> np.ndarray:
    """Per-point pixel error of original -> crop -> original transforms.

    ``to_crop`` / ``to_original`` are callables ``(x, y) -> (u, v)`` matching
    ``recon3d.schemas.CropMetadata``. Returns one error per input point.
    """
    pts = _as_points(points_xy, "points_xy")
    errors = np.empty(pts.shape[0], dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        u, v = to_crop(float(x), float(y))
        x2, y2 = to_original(float(u), float(v))
        errors[i] = math.hypot(x2 - x, y2 - y)
    return errors


def aspect_ratio_error(width_a: float, height_a: float,
                       width_b: float, height_b: float) -> float:
    """Relative difference between two width/height aspect ratios."""
    if min(width_a, height_a, width_b, height_b) <= 0:
        raise ValueError("dimensions must be positive")
    ra = width_a / height_a
    rb = width_b / height_b
    return abs(ra - rb) / ra


# ---------------------------------------------------------------------------
# colour / image-similarity metrics
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 ``(..., 3)`` or float 0..1 values to CIE L*a*b*."""
    arr = np.asarray(rgb)
    if arr.shape[-1] != 3:
        raise ValueError("rgb must have a trailing channel axis of size 3")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64)
        if arr.max(initial=0.0) > 1.0 + 1e-6:
            arr = arr / 255.0
    from skimage import color
    return color.rgb2lab(arr)


def color_delta_e76(rgb_a, rgb_b) -> float:
    """CIE76 (Euclidean Lab) difference between two sRGB colours.

    Accepts ``(r, g, b)`` tuples/lists/arrays in 0..255 or 0..1.
    """
    a = np.asarray(rgb_a, dtype=np.float64).reshape(1, 1, 3)
    b = np.asarray(rgb_b, dtype=np.float64).reshape(1, 1, 3)
    la = rgb_to_lab(a)
    lb = rgb_to_lab(b)
    return float(np.sqrt(((la - lb) ** 2).sum()))


def ssim(image_a: np.ndarray, image_b: np.ndarray,
         mask: Optional[np.ndarray] = None) -> float:
    """Structural similarity between two equally-sized images.

    Grayscale or RGB uint8/float images. When ``mask`` is given, both images
    are composited onto a black background first so background pixels do not
    inflate the score.
    """
    from skimage.metrics import structural_similarity
    a = np.asarray(image_a)
    b = np.asarray(image_b)
    if a.shape != b.shape:
        raise ValueError("image shapes differ: %r vs %r" % (a.shape, b.shape))
    if mask is not None:
        m = _as_bool_mask(mask)
        if a.ndim == 3:
            m = m[:, :, None]
        a = np.where(m, a, 0)
        b = np.where(m, b, 0)
    channel_axis = -1 if a.ndim == 3 else None
    data_range = 255.0 if a.dtype == np.uint8 else 1.0
    return float(structural_similarity(a, b, channel_axis=channel_axis,
                                       data_range=data_range))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two flattened arrays.

    Returns 0.0 when either array is constant (correlation undefined).
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("array sizes differ: %r vs %r" % (x.shape, y.shape))
    if x.size < 2:
        raise ValueError("need at least 2 samples")
    sx, sy = float(x.std()), float(y.std())
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank-order correlation between two flattened arrays."""
    from scipy import stats
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("array sizes differ: %r vs %r" % (x.shape, y.shape))
    if x.size < 2:
        raise ValueError("need at least 2 samples")
    rho, _ = stats.spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else 0.0


# ---------------------------------------------------------------------------
# primitive parameter error helpers
# ---------------------------------------------------------------------------

def circle_param_error(fit_center, fit_radius: float,
                       ref_center, ref_radius: float) -> Dict[str, float]:
    """Centre (relative to radius) and radius relative error of a circle fit."""
    if ref_radius <= 0:
        raise ValueError("ref_radius must be > 0")
    fc = np.asarray(fit_center, dtype=np.float64)
    rc = np.asarray(ref_center, dtype=np.float64)
    centre_err = float(np.linalg.norm(fc - rc)) / float(ref_radius)
    radius_err = abs(float(fit_radius) - float(ref_radius)) / float(ref_radius)
    return {"center_rel_error": centre_err, "radius_rel_error": radius_err}


def ellipse_param_error(fit_axes, fit_angle_deg: float,
                        ref_axes, ref_angle_deg: float) -> Dict[str, float]:
    """Axis relative errors and rotation error (deg) of an ellipse fit.

    ``axes`` are ``(semi_major, semi_minor)``. Axes are matched by size so a
    90-degree labelling flip does not inflate the error; the rotation error is
    taken modulo 180 degrees.
    """
    fa = sorted([float(v) for v in fit_axes], reverse=True)
    ra = sorted([float(v) for v in ref_axes], reverse=True)
    if min(ra) <= 0:
        raise ValueError("reference axes must be > 0")
    major_err = abs(fa[0] - ra[0]) / ra[0]
    minor_err = abs(fa[1] - ra[1]) / ra[1]
    rot_err = abs(angle_diff_deg(fit_angle_deg, ref_angle_deg, period=180.0))
    return {"major_axis_rel_error": major_err,
            "minor_axis_rel_error": minor_err,
            "rotation_error_deg": rot_err}


def angle_diff_deg(a: float, b: float, period: float = 360.0) -> float:
    """Smallest signed difference ``a - b`` in degrees, folded into period/2."""
    if period <= 0:
        raise ValueError("period must be > 0")
    d = math.fmod(float(a) - float(b), period)
    if d > period / 2.0:
        d -= period
    elif d < -period / 2.0:
        d += period
    return d


def line_angle_error_deg(angle_a_deg: float, angle_b_deg: float) -> float:
    """Absolute angle error between two undirected lines (modulo 180 deg)."""
    return abs(angle_diff_deg(angle_a_deg, angle_b_deg, period=180.0))


def line_endpoint_error(fit_p0, fit_p1, ref_p0, ref_p1) -> float:
    """Mean endpoint error allowing for endpoint swap (undirected segment)."""
    fa = np.asarray(fit_p0, dtype=np.float64)
    fb = np.asarray(fit_p1, dtype=np.float64)
    ra = np.asarray(ref_p0, dtype=np.float64)
    rb = np.asarray(ref_p1, dtype=np.float64)
    direct = 0.5 * (np.linalg.norm(fa - ra) + np.linalg.norm(fb - rb))
    swapped = 0.5 * (np.linalg.norm(fa - rb) + np.linalg.norm(fb - ra))
    return float(min(direct, swapped))


# ---------------------------------------------------------------------------
# detection / calibration metrics
# ---------------------------------------------------------------------------

def precision_recall_f1_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision/recall/F1 from raw true/false-positive/false-negative counts."""
    for name, v in (("tp", tp), ("fp", fp), ("fn", fn)):
        if v < 0:
            raise ValueError("%s must be >= 0" % name)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def match_sets(predicted: Sequence[str], reference: Sequence[str],
               key=None) -> Dict[str, object]:
    """Match predicted items to reference items, returning counts and pairs.

    ``key`` optionally maps an item to a canonical match key (default: the
    item itself). Matching is greedy one-to-one in the given order.
    """
    key = key or (lambda x: x)
    ref_remaining: Dict[object, int] = {}
    for r in reference:
        ref_remaining[key(r)] = ref_remaining.get(key(r), 0) + 1
    tp = 0
    matched: List[Tuple[object, object]] = []
    for p in predicted:
        k = key(p)
        if ref_remaining.get(k, 0) > 0:
            ref_remaining[k] -= 1
            tp += 1
            matched.append((p, k))
    fp = len(predicted) - tp
    fn = sum(ref_remaining.values())
    out = precision_recall_f1_counts(tp, fp, fn)
    out["matched"] = matched
    return out


def expected_calibration_error(confidences: Sequence[float],
                               correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    """Expected calibration error over uniformly spaced confidence bins.

    ECE = sum_b (|bin_b| / N) * |acc(bin_b) - conf(bin_b)|.
    Empty bins are skipped. Confidence == 1.0 falls in the last bin.
    """
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=bool)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError("confidences and correct must have equal length")
    if conf.shape[0] == 0:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.shape[0]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = float(corr[in_bin].mean())
        mean_conf = float(conf[in_bin].mean())
        ece += (count / n) * abs(acc - mean_conf)
    return float(ece)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error between predicted confidence and binary outcome."""
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=np.float64)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError("confidences and correct must have equal length")
    if conf.shape[0] == 0:
        raise ValueError("empty input")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    return float(np.mean((conf - corr) ** 2))


def selective_accuracy(confidences: Sequence[float], correct: Sequence[bool],
                       threshold: float) -> Dict[str, float]:
    """Accuracy and coverage when accepting only confidence >= threshold."""
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=bool)
    if conf.shape[0] != corr.shape[0] or conf.shape[0] == 0:
        raise ValueError("confidences and correct must be non-empty and equal length")
    keep = conf >= threshold
    coverage = float(keep.mean())
    accuracy = float(corr[keep].mean()) if keep.any() else 1.0
    return {"accuracy": accuracy, "coverage": coverage}


# ---------------------------------------------------------------------------
# depth / normals metrics
# ---------------------------------------------------------------------------

def depth_abs_rel_error(pred: np.ndarray, ref: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> float:
    """Mean absolute relative depth error |p - r| / r over valid pixels."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if p.shape != r.shape:
        raise ValueError("depth shapes differ: %r vs %r" % (p.shape, r.shape))
    valid = r > 0
    if mask is not None:
        valid &= _as_bool_mask(mask)
    if valid.sum() == 0:
        raise ValueError("no valid reference depth pixels")
    return float(np.mean(np.abs(p[valid] - r[valid]) / r[valid]))


def normals_angular_error_deg(pred: np.ndarray, ref: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Angular error stats between two ``(H, W, 3)`` normal maps.

    Returns mean/median degrees plus the fraction of pixels below the common
    11.25/22.5/30 degree thresholds. Zero-length normals are ignored.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if p.shape != r.shape:
        raise ValueError("normal map shapes differ")
    valid = (np.linalg.norm(p, axis=1) > 1e-9) & (np.linalg.norm(r, axis=1) > 1e-9)
    if mask is not None:
        valid &= _as_bool_mask(mask).ravel()
    if valid.sum() == 0:
        raise ValueError("no valid normal pixels")
    pn = p[valid] / np.linalg.norm(p[valid], axis=1, keepdims=True)
    rn = r[valid] / np.linalg.norm(r[valid], axis=1, keepdims=True)
    cos = np.clip((pn * rn).sum(axis=1), -1.0, 1.0)
    err = np.degrees(np.arccos(cos))
    return {
        "mean_deg": float(err.mean()),
        "median_deg": float(np.median(err)),
        "pct_below_11_25": float((err < 11.25).mean()),
        "pct_below_22_5": float((err < 22.5).mean()),
        "pct_below_30": float((err < 30.0).mean()),
    }
