"""Stage 4: preprocessing evidence layers.

Produces five deterministic layers from the normalised crop:
    silhouette.png            binary object boundary incl. internal holes
    color_quantized.png       seeded k-means colour regions (Lab), transparent bg
    structural_edges.png      geometric boundaries only (texture suppressed)
    details.png               high-frequency residual (tread/logos/fasteners)
    lighting_normalized.png   flat-field illumination correction
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from .config import PipelineConfig
from .schemas import PreprocessLayers


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _lighting_normalize(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divide each channel by a large-kernel Gaussian illumination estimate."""
    h, w = mask.shape
    k = _odd(max(31, min(h, w) // 8))
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32)
        illum = cv2.GaussianBlur(ch, (k, k), 0) + 1.0
        fg_mean = float(illum[mask > 0].mean()) if mask.any() else float(illum.mean())
        out[:, :, c] = ch * (fg_mean / illum)
    norm = np.clip(out, 0, 255).astype(np.uint8)
    norm[mask == 0] = 128  # neutral background outside the object
    return norm


def _color_quantize(
    rgb: np.ndarray, mask: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Seeded k-means in Lab space over masked pixels; transparent background."""
    filtered = cv2.bilateralFilter(rgb, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    samples = lab[mask > 0].astype(np.float32)
    k_eff = int(min(k, max(1, len(samples))))
    if k_eff > 1 and len(samples) > k_eff:
        cv2.setRNGSeed(seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            samples, k_eff, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        quant_lab = lab.copy()
        quant_lab[mask > 0] = centers[labels.flatten()].astype(lab.dtype)
    else:
        quant_lab = lab
    quant_rgb = cv2.cvtColor(quant_lab, cv2.COLOR_LAB2RGB)
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    return np.dstack([quant_rgb, alpha])


def _details_layer(ln_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """High-frequency residual (difference of Gaussians), masked to object."""
    g1 = cv2.GaussianBlur(ln_gray, (0, 0), 1.0).astype(np.float32)
    g2 = cv2.GaussianBlur(ln_gray, (0, 0), 4.0).astype(np.float32)
    dog = np.abs(g1 - g2)
    detail = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    detail[mask == 0] = 0
    return detail


def _structural_edges(
    ln_gray: np.ndarray,
    mask: np.ndarray,
    detail: np.ndarray,
    low: int,
    high: int,
) -> np.ndarray:
    """Canny on a smoothed, lighting-normalised image; texture edges suppressed."""
    base = cv2.GaussianBlur(ln_gray, (5, 5), 1.4)
    edges = cv2.Canny(base, low, high)
    edges[mask == 0] = 0

    # suppress pure-texture edges: pixels where the high-frequency detail
    # layer is strongly active are not geometric boundaries
    fg_vals = detail[mask > 0]
    if fg_vals.size:
        thr = max(10, int(np.percentile(fg_vals, 85)))
        texture = (detail > thr).astype(np.uint8)
        texture = cv2.dilate(texture, np.ones((3, 3), np.uint8))
        edges[texture > 0] = 0

    # the object outline (incl. internal holes) is always a geometric
    # boundary; the DoG-based suppression above removes it, so add it back
    kernel = np.ones((3, 3), np.uint8)
    outline = mask - cv2.erode(mask, kernel)
    edges[outline > 0] = 255

    # thin to single-pixel lines
    thin = skeletonize(edges > 0).astype(np.uint8) * 255
    return thin


def preprocess(
    crop_rgba_path: str, crop_mask_path: str, out_dir: str, cfg: PipelineConfig
) -> PreprocessLayers:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = cfg.preprocess

    rgba = np.asarray(Image.open(crop_rgba_path).convert("RGBA"))
    rgb = rgba[:, :, :3]
    mask = (np.asarray(Image.open(crop_mask_path).convert("L")) >= 128).astype(np.uint8)

    params: Dict[str, Any] = {
        "color_regions": p.color_regions,
        "edge_low_threshold": p.edge_low_threshold,
        "edge_high_threshold": p.edge_high_threshold,
        "detail_kernel": p.detail_kernel,
        "seed": cfg.seed,
    }

    # 1. silhouette: binary mask, internal holes preserved
    silhouette = mask * 255
    cv2.imwrite(str(out / "silhouette.png"), silhouette)

    # 2. lighting-normalised (flat-field per channel)
    ln = _lighting_normalize(rgb, mask)
    cv2.imwrite(str(out / "lighting_normalized.png"), cv2.cvtColor(ln, cv2.COLOR_RGB2BGR))
    ln_gray = cv2.cvtColor(ln, cv2.COLOR_RGB2GRAY)

    # 3. colour regions
    quant = _color_quantize(rgb, mask, p.color_regions, cfg.seed)
    Image.fromarray(quant).save(out / "color_quantized.png")

    # 4. details (DoG residual)
    detail = _details_layer(ln_gray, mask)
    if p.detail_kernel > 1:
        detail = cv2.GaussianBlur(detail, (_odd(p.detail_kernel), _odd(p.detail_kernel)), 0)
        detail[mask == 0] = 0
    cv2.imwrite(str(out / "details.png"), detail)

    # 5. structural edges (texture suppressed, thinned)
    edges = _structural_edges(
        ln_gray, mask, detail, p.edge_low_threshold, p.edge_high_threshold
    )
    cv2.imwrite(str(out / "structural_edges.png"), edges)

    return PreprocessLayers(
        silhouette_path=str(out / "silhouette.png"),
        color_quantized_path=str(out / "color_quantized.png"),
        structural_edges_path=str(out / "structural_edges.png"),
        details_path=str(out / "details.png"),
        lighting_normalized_path=str(out / "lighting_normalized.png"),
        params=params,
    )
