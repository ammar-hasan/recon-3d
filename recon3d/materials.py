"""Stage 16 support: per-part PBR material estimation.

Per part, pixels inside the rasterised primitive region are sampled from the
crop RGBA. To keep highlights and shadows out of the base colour:

- luminance is computed per pixel;
- the top and bottom luminance deciles (specular highlights, deep shadows)
  are trimmed;
- the median of the remaining pixels per channel becomes base_color.

Heuristics (deliberately modest — these are cues, not measurements):

- roughness: the fraction of pixels sitting in the trimmed top decile is a
  specular-highlight proxy; more highlight energy -> lower roughness.
  ``roughness = clamp(0.9 - 2.0 * highlight_fraction, 0.25, 0.9)``.
- metallic: metals show strong luminance variance with low saturation;
  ``metallic = clamp(4 * highlight_fraction, 0, 0.8)`` only when the base
  colour is desaturated, else 0.
- material_class: taken from the semantic appearance estimate when present;
  otherwise very dark + desaturated -> rubber, metallic-looking -> metal,
  everything else -> plastic.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .part_geometry import part_primitives, rasterize_primitives
from .schemas import EvidenceSource, MaterialSpec, SemanticPart, SketchGraph

#: luminance percentile trimmed from both ends (highlights / shadows)
_TRIM_PERCENTILE = 10.0


def _srgb_to_linearish(rgb01: np.ndarray) -> np.ndarray:
    """sRGB 0..1 -> approximately linear 0..1 (kept mild; MaterialSpec says
    'linear-ish')."""
    return np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)


def _sample_part_pixels(
    graph: SketchGraph, part: SemanticPart, rgb: np.ndarray, alpha: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]
    prims = part_primitives(graph, part)
    if not prims:
        return None
    mask = rasterize_primitives(prims, (h, w)) > 0
    if alpha is not None:
        mask &= alpha > 0
    if int(mask.sum()) < 10:
        return None
    return rgb[mask]


def _class_from_appearance(part: SemanticPart) -> Optional[str]:
    if part.appearance is not None and part.appearance.material_class:
        return part.appearance.material_class
    return None


def estimate_materials(
    graph: SketchGraph, crop_rgba_path: str, cfg: PipelineConfig
) -> Dict[str, MaterialSpec]:
    img = cv2.imread(crop_rgba_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("materials: could not read crop rgba image: %s" % crop_rgba_path)
    rgb = img[:, :, :3][:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    alpha = img[:, :, 3] if img.shape[2] == 4 else None

    result: Dict[str, MaterialSpec] = {}
    for part in graph.parts:
        pixels = _sample_part_pixels(graph, part, rgb, alpha)
        if pixels is None:
            # nothing to sample: fall back to semantic appearance if present
            spec = MaterialSpec(source=EvidenceSource.UNKNOWN)
            ap = part.appearance
            if ap is not None and ap.estimated_color_srgb is not None:
                spec.base_color = tuple(
                    float(c) / 255.0 for c in ap.estimated_color_srgb
                )  # type: ignore[assignment]
                spec.material_class = ap.material_class or "plastic"
                spec.roughness = float(ap.roughness) if ap.roughness is not None else 0.6
                spec.metallic = float(ap.metallic) if ap.metallic is not None else 0.0
                spec.source = ap.source
            result[part.id] = spec
            continue

        lum = (
            0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        )
        lo, hi = np.percentile(lum, _TRIM_PERCENTILE), np.percentile(
            lum, 100.0 - _TRIM_PERCENTILE
        )
        keep = (lum >= lo) & (lum <= hi)
        trimmed = pixels[keep]
        base_srgb = np.median(trimmed, axis=0)  # 0..1 sRGB per channel
        base = _srgb_to_linearish(base_srgb.astype(np.float32))

        # saturation of the base colour (0 = gray)
        bmax, bmin = float(base_srgb.max()), float(base_srgb.min())
        saturation = (bmax - bmin) / bmax if bmax > 1e-6 else 0.0

        # specular highlight proxy: share of pixels trimmed from the top decile
        highlight_fraction = float((lum > hi).mean())
        roughness = float(np.clip(0.9 - 2.0 * highlight_fraction, 0.25, 0.9))
        metallic = (
            float(np.clip(4.0 * highlight_fraction, 0.0, 0.8))
            if saturation < 0.15
            else 0.0
        )

        material_class = _class_from_appearance(part)
        mean_lum = float(base_srgb.mean())
        if material_class is None:
            if mean_lum < 0.18 and saturation < 0.2:
                material_class = "rubber"
            elif metallic > 0.4:
                material_class = "metal"
            else:
                material_class = "plastic"

        opacity = 1.0
        if alpha is not None:
            # mean alpha inside the region (usually fully opaque)
            prims = part_primitives(graph, part)
            mask = rasterize_primitives(prims, alpha.shape) > 0
            if int(mask.sum()) > 0:
                opacity = float(np.clip(alpha[mask].mean() / 255.0, 0.0, 1.0))

        transmission = 0.9 if material_class == "glass" else 0.0

        result[part.id] = MaterialSpec(
            material_class=material_class,
            base_color=(float(base[0]), float(base[1]), float(base[2])),
            roughness=round(roughness, 3),
            metallic=round(metallic, 3),
            opacity=round(opacity, 3),
            transmission=transmission,
            source=EvidenceSource.FITTED_FROM_OBSERVATION,
        )
    return result
