"""Stage 2: foreground segmentation.

Backend priority (cfg.segmentation.backend == "auto"):
    user mask (spec.mask_path) > box/point-guided GrabCut > rembg > classical.

An explicit backend ("user_mask"|"grabcut"|"rembg"|"threshold") forces that
backend where applicable, with graceful fallback to the classical path.

All backends share post-processing that keeps the selected target component,
fills only tiny pinholes (meaningful holes are preserved), and applies small
morphological open/close.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .schemas import EvidenceSource, InputBundle, SegmentationResult

_BACKEND_PRIOR = {
    "user_mask": 0.99,
    "rembg": 0.85,
    "grabcut": 0.80,
    "classical": 0.65,
}

_REMBG_SESSIONS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------------

def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "RGBA":
        # composite over white so semi-transparent backgrounds do not confuse
        # colour statistics
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _mask_from_user(bundle: InputBundle, shape: Tuple[int, int]) -> np.ndarray:
    m = Image.open(bundle.spec.mask_path).convert("L")
    arr = np.asarray(m)
    if arr.shape[:2] != shape:
        raise ValueError(
            f"mask size {arr.shape[:2]} does not match image {shape} (should have "
            "been caught by load_input)"
        )
    return (arr > 127).astype(np.uint8)


def _grabcut(
    rgb: np.ndarray,
    iterations: int,
    rect: Optional[Tuple[int, int, int, int]] = None,
    init_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run GrabCut with either a rect or a GC-labelled init mask."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    if init_mask is not None:
        gc = init_mask.copy()
        cv2.grabCut(bgr, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    else:
        x0, y0, x1, y1 = rect
        r = (max(0, x0), max(0, y0), max(1, x1 - x0), max(1, y1 - y0))
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(bgr, gc, r, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def _grabcut_guided(
    rgb: np.ndarray,
    box: Optional[Tuple[float, float, float, float]],
    point: Optional[Tuple[float, float]],
    iterations: int,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        mask = _grabcut(rgb, iterations, rect=(x0, y0, x1, y1))
        if point is not None:
            # reinforce the user point as definite foreground
            px, py = int(round(point[0])), int(round(point[1]))
            if mask[py, px] == 0:
                gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
                gc[mask > 0] = cv2.GC_PR_FGD
                r = max(3, min(h, w) // 100)
                cv2.circle(gc, (px, py), r, int(cv2.GC_FGD), -1)
                mask = _grabcut(rgb, iterations, init_mask=gc)
        return mask
    # point only: seed a sure-foreground disk, sure-background border ring
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[0, :] = gc[-1, :] = cv2.GC_BGD
    gc[:, 0] = gc[:, -1] = cv2.GC_BGD
    px, py = int(round(point[0])), int(round(point[1]))
    r = max(3, min(h, w) // 50)
    cv2.circle(gc, (px, py), r, int(cv2.GC_FGD), -1)
    return _grabcut(rgb, iterations, init_mask=gc)


def _rembg_mask(rgb: np.ndarray, model: str) -> np.ndarray:
    from rembg import new_session, remove

    session = _REMBG_SESSIONS.get(model)
    if session is None:
        session = new_session(model)
        _REMBG_SESSIONS[model] = session
    out = remove(Image.fromarray(rgb), session=session)
    alpha = np.asarray(out.convert("RGBA"))[:, :, 3]
    return (alpha > 127).astype(np.uint8)


def _classical_mask(rgb: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Border-colour-distance Otsu, refined by one GrabCut pass."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    strip = max(2, min(h, w) // 40)
    border = np.concatenate(
        [
            lab[:strip].reshape(-1, 3),
            lab[-strip:].reshape(-1, 3),
            lab[:, :strip].reshape(-1, 3),
            lab[:, -strip:].reshape(-1, 3),
        ]
    )
    med = np.median(border, axis=0)
    dist = np.linalg.norm(lab - med, axis=2)
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, fg = cv2.threshold(dist8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coverage = float(fg.mean())
    if cfg.segmentation.min_coverage < coverage < cfg.segmentation.max_coverage:
        # refine with GrabCut seeded from the distance map
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[dist8 > t] = cv2.GC_FGD
        gc[dist8 < t * 0.4] = cv2.GC_BGD
        gc[(dist8 >= t * 0.4) & (dist8 <= t)] = cv2.GC_PR_BGD
        hi = np.percentile(dist8, 97)
        gc[dist8 >= hi] = cv2.GC_FGD
        try:
            refined = _grabcut(rgb, cfg.segmentation.grabcut_iterations, init_mask=gc)
            rc = float(refined.mean())
            if cfg.segmentation.min_coverage < rc < cfg.segmentation.max_coverage:
                return refined
        except cv2.error:
            pass
        return fg.astype(np.uint8)

    # degenerate Otsu result: centre-rect GrabCut as last resort
    mx, my = int(w * 0.08), int(h * 0.08)
    return _grabcut(rgb, cfg.segmentation.grabcut_iterations,
                    rect=(mx, my, w - mx, h - my))


def _rescue_undersegmented_rembg(
    rgb: np.ndarray, mask: np.ndarray, cfg: PipelineConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Recover a pale enclosing object when rembg selects only its emblem.

    A larger centre-rectangle GrabCut candidate is accepted only when the
    rembg component is small, is almost fully contained by the candidate,
    and the candidate has a plausible non-border extent.  The containment
    requirement prevents this fallback from replacing legitimate small
    isolated subjects.
    """
    h, w = mask.shape
    seed = _select_target(mask.astype(np.uint8), None, None)
    seed_cov = float(seed.mean())
    info: Dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "seed_coverage": seed_cov,
    }
    if not (cfg.segmentation.min_coverage < seed_cov < 0.07):
        return mask, info

    info["attempted"] = True
    # Keep the initial rectangle close to the canvas edge.  Pale products can
    # have boundaries well outside the conventional 10% centre rectangle;
    # GrabCut then treats the actual object as definite background.
    margin_x, margin_y = int(w * 0.02), int(h * 0.02)
    far_x, far_y = int(w * 0.98), int(h * 0.98)
    try:
        candidate = _grabcut(
            rgb,
            min(5, cfg.segmentation.grabcut_iterations),
            rect=(margin_x, margin_y, far_x, far_y),
        )
    except cv2.error:
        return mask, info
    candidate = _select_target(candidate, None, None)
    candidate_cov = float(candidate.mean())
    containment = float((candidate & seed).sum()) / max(float(seed.sum()), 1.0)
    info.update({
        "candidate_coverage": candidate_cov,
        "seed_containment": containment,
    })
    if candidate.any():
        ys, xs = np.nonzero(candidate)
        bw = (int(xs.max()) - int(xs.min()) + 1) / float(w)
        bh = (int(ys.max()) - int(ys.min()) + 1) / float(h)
        touches = (xs.min() == 0 or ys.min() == 0
                   or xs.max() == w - 1 or ys.max() == h - 1)
    else:
        bw = bh = 0.0
        touches = True
    plausible = (
        candidate_cov >= max(0.07, 2.0 * seed_cov)
        and candidate_cov <= min(0.55, 7.0 * seed_cov)
        and containment >= 0.85
        and bw >= 0.30
        and bh >= 0.20
        and not touches
    )
    if plausible:
        info["accepted"] = True
        candidate, edge_info = _expand_edge_bounded_enclosure(rgb, candidate)
        info["edge_enclosure"] = edge_info
        return candidate, info
    return mask, info


def _expand_edge_bounded_enclosure(
    rgb: np.ndarray, candidate: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extend a partial pale-object mask to strong enclosing straight edges.

    GrabCut can stop at a high-contrast logo printed on an otherwise white
    plate.  When long top, bottom, and right boundary lines form a plausible
    quadrilateral around the accepted candidate, union that enclosure with
    the mask.  This is deliberately downstream of the strict rembg
    containment check above, so ordinary textured scenes never enter it.
    """
    info: Dict[str, Any] = {"accepted": False}
    ys, xs = np.nonzero(candidate)
    if len(xs) < 16:
        return candidate, info
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pad = 0.20 * max(bw, bh)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=45,
        minLineLength=max(40, int(0.25 * min(bw, bh))),
        maxLineGap=max(12, int(0.10 * max(bw, bh))),
    )
    if lines is None:
        return candidate, info

    horizontal = []
    vertical = []
    for raw in lines[:, 0]:
        xa, ya, xb, yb = (float(v) for v in raw)
        if not (x0 - pad <= xa <= x1 + pad and x0 - pad <= xb <= x1 + pad
                and y0 - pad <= ya <= y1 + pad and y0 - pad <= yb <= y1 + pad):
            continue
        dx, dy = xb - xa, yb - ya
        length = float(np.hypot(dx, dy))
        if abs(dx) >= 1.7 * abs(dy) and abs(dx) > 1e-6:
            slope = dy / dx
            intercept = ya - slope * xa
            ymid = slope * cx + intercept
            left_x = min(xa, xb)
            if left_x <= x0 + 0.12 * bw and length >= 0.30 * bw:
                horizontal.append((ymid, -length, slope, intercept, raw))
        if abs(dy) >= 1.7 * abs(dx) and abs(dy) > 1e-6:
            slope = dx / dy
            intercept = xa - slope * ya
            xmid = slope * cy + intercept
            if xmid >= x1 - 0.12 * bw and length >= 0.30 * bh:
                vertical.append((xmid, -length, slope, intercept, raw))
    tops = [line for line in horizontal if line[0] < cy]
    bottoms = [line for line in horizontal if line[0] > cy]
    if not tops or not bottoms or not vertical:
        return candidate, info
    top = min(tops, key=lambda line: line[0])
    bottom = max(bottoms, key=lambda line: line[0])
    right = max(vertical, key=lambda line: line[0])

    def intersect(hline, vline):
        mh, bh0 = hline[2], hline[3]
        mv, bv0 = vline[2], vline[3]
        denom = 1.0 - mv * mh
        if abs(denom) < 1e-6:
            return None
        x = (mv * bh0 + bv0) / denom
        return [x, mh * x + bh0]

    def left_endpoint(line):
        xa, ya, xb, yb = (float(v) for v in line[4])
        return [xa, ya] if xa <= xb else [xb, yb]

    top_right = intersect(top, right)
    bottom_right = intersect(bottom, right)
    if top_right is None or bottom_right is None:
        return candidate, info
    polygon_f = np.asarray([
        left_endpoint(top), top_right, bottom_right, left_endpoint(bottom)
    ], dtype=np.float32)
    if (np.any(polygon_f[:, 0] < x0 - pad)
            or np.any(polygon_f[:, 0] > x1 + pad)
            or np.any(polygon_f[:, 1] < y0 - pad)
            or np.any(polygon_f[:, 1] > y1 + pad)
            or abs(float(cv2.contourArea(polygon_f))) < 0.5 * float(candidate.sum())):
        return candidate, info

    expanded = candidate.copy()
    cv2.fillPoly(expanded, [np.round(polygon_f).astype(np.int32)], 1)
    ratio = float(expanded.sum()) / max(float(candidate.sum()), 1.0)
    if ratio > 1.65:
        return candidate, info
    info.update({
        "accepted": True,
        "area_ratio": ratio,
        "polygon": np.round(polygon_f, 1).tolist(),
    })
    return expanded, info


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------

def _select_target(
    mask: np.ndarray,
    point: Optional[Tuple[float, float]],
    box: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Keep only the connected component matching the user's hint (or largest)."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    best = 1
    if point is not None:
        px, py = int(round(point[0])), int(round(point[1]))
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and labels[py, px] > 0:
            best = int(labels[py, px])
        else:
            d = np.hypot(centroids[1:, 0] - px, centroids[1:, 1] - py)
            best = int(np.argmin(d)) + 1
    elif box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        best_overlap = -1
        for lab in range(1, num):
            comp = labels == lab
            overlap = float(comp[y0:y1, x0:x1].sum()) / max(comp.sum(), 1)
            if overlap > best_overlap:
                best_overlap, best = overlap, lab
    else:
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == best).astype(np.uint8)


def _fill_pinholes(mask: np.ndarray, min_hole_area: int) -> np.ndarray:
    """Fill holes smaller than min_hole_area; preserve meaningful holes."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    out = mask.copy()
    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent >= 0 and cv2.contourArea(cnt) < min_hole_area:
            cv2.drawContours(out, contours, i, 1, thickness=-1)
    return out


def _postprocess(
    mask: np.ndarray,
    point: Optional[Tuple[float, float]],
    box: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = _select_target(mask, point, box)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    min_hole = max(16, int(0.0002 * mask.shape[0] * mask.shape[1]))
    mask = _fill_pinholes(mask, min_hole)
    return mask


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def segment(bundle: InputBundle, out_dir: str, cfg: PipelineConfig) -> SegmentationResult:
    spec = bundle.spec
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(bundle.images[0].path)
    h, w = rgb.shape[:2]
    warnings: List[str] = list(bundle.warnings)
    diagnostics: Dict[str, Any] = {}
    requested = cfg.segmentation.backend

    mask: Optional[np.ndarray] = None
    backend = ""

    if not cfg.segmentation.background_removal_enabled:
        mask = np.ones((h, w), dtype=np.uint8)
        backend = "disabled_background_removal"
        warnings.append(
            "background removal disabled by ablation; full image treated as foreground")
    elif spec.mask_path and requested in ("auto", "user_mask"):
        mask = _mask_from_user(bundle, (h, w))
        backend = "user_mask"
    elif (spec.box is not None or spec.point is not None) and requested in ("auto", "grabcut"):
        mask = _grabcut_guided(rgb, spec.box, spec.point, cfg.segmentation.grabcut_iterations)
        backend = "grabcut"
    elif requested == "grabcut":
        mx, my = int(w * 0.05), int(h * 0.05)
        mask = _grabcut(rgb, cfg.segmentation.grabcut_iterations,
                        rect=(mx, my, w - mx, h - my))
        backend = "grabcut"
    elif requested in ("auto", "rembg"):
        try:
            mask = _rembg_mask(rgb, cfg.segmentation.rembg_model)
            backend = "rembg"
        except Exception as exc:  # noqa: BLE001 - graceful degradation required
            warnings.append(f"rembg backend failed ({type(exc).__name__}: {exc}); "
                            "falling back to classical segmentation")
            if requested == "rembg":
                warnings.append("backend 'rembg' was explicitly requested")
            mask = _classical_mask(rgb, cfg)
            backend = "classical"
    else:  # "threshold" / "classical"
        mask = _classical_mask(rgb, cfg)
        backend = "classical"

    diagnostics["backend_requested"] = requested
    diagnostics["grabcut_iterations"] = cfg.segmentation.grabcut_iterations

    if backend == "rembg" and spec.box is None and spec.point is None:
        mask, rescue = _rescue_undersegmented_rembg(rgb, mask, cfg)
        diagnostics["undersegmentation_rescue"] = rescue
        if rescue.get("accepted"):
            warnings.append(
                "rembg selected a small component inside a larger plausible subject; "
                "accepted containment-checked GrabCut recovery"
            )

    mask = _postprocess(mask.astype(np.uint8), spec.point, spec.box)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"segmentation produced an empty mask (backend={backend})")

    # tight bbox + coverage
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    coverage = float(mask.mean())

    # hole count (meaningful negative spaces kept)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    n_holes = int(sum(1 for hh in (hierarchy[0] if hierarchy is not None else []) if hh[3] >= 0))
    diagnostics["hole_count"] = n_holes

    # confidence heuristic: backend prior, coverage sanity, border contact
    conf = _BACKEND_PRIOR.get(backend, 0.6)
    if coverage < cfg.segmentation.min_coverage:
        conf *= 0.5
        warnings.append(f"tiny object coverage {coverage:.4f} < {cfg.segmentation.min_coverage}")
    if coverage > cfg.segmentation.max_coverage:
        conf *= 0.5
        warnings.append(f"object coverage {coverage:.4f} exceeds {cfg.segmentation.max_coverage} "
                        "(mask may include background)")
    touches_border = x0 == 0 or y0 == 0 or x1 == w or y1 == h
    if touches_border:
        conf *= 0.85
        warnings.append("object touches image border; it may be truncated")
    confidence = float(min(0.99, max(0.05, conf)))

    # outputs
    mask_path = out / "object_mask.png"
    rgba_path = out / "object_rgba.png"
    cv2.imwrite(str(mask_path), mask * 255)
    rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
    Image.fromarray(rgba).save(rgba_path)

    if spec.mask_path and backend == "user_mask":
        source = EvidenceSource.USER_SUPPLIED
    elif spec.box is not None or spec.point is not None:
        source = EvidenceSource.USER_SUPPLIED
    else:
        source = EvidenceSource.FITTED_FROM_OBSERVATION

    return SegmentationResult(
        mask_path=str(mask_path),
        rgba_path=str(rgba_path),
        original_path=bundle.images[0].path,
        confidence=confidence,
        backend=backend,
        bbox=(x0, y0, x1, y1),
        coverage=coverage,
        diagnostics=diagnostics,
        selection_source=source,
        warnings=warnings,
    )
