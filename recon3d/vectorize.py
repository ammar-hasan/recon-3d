"""Stage 5: raster-to-vector tracing, one SVG per preprocessing layer.

Primary backend: vtracer (binary mode for silhouette/edges/details, colour
mode for the quantized layer). Fallback: OpenCV findContours -> svgwrite
polylines. Produced SVGs are parsed back into TraceLayer/VectorPath objects
with points sampled in PIXELS (normalisation happens in svg_cleanup).

Note: vtracer binary mode traces DARK pixels, so binary inputs are rendered
as black-on-white before tracing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .schemas import (
    PreprocessLayers,
    TraceLayer,
    TraceLayerName,
    VectorPath,
)
from .svgpath import iter_svg_subpaths, points_to_d, shoelace_area

_LAYER_CONFIDENCE: Dict[TraceLayerName, float] = {
    TraceLayerName.SILHOUETTE: 0.9,
    TraceLayerName.COLOR_REGIONS: 0.8,
    TraceLayerName.STRUCTURAL_EDGES: 0.7,
    TraceLayerName.DETAILS: 0.6,
}


def _load_binary_fg(path: str, dilate_px: int = 0) -> np.ndarray:
    """Foreground mask (1=fg) from a grayscale/binary layer image."""
    arr = np.asarray(Image.open(path).convert("L"))
    fg = (arr > 63).astype(np.uint8)
    if dilate_px > 0 and fg.any():
        k = 2 * dilate_px + 1
        fg = cv2.dilate(fg, np.ones((k, k), np.uint8))
    return fg


def _fg_to_vtracer_pixels(fg: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """Black foreground on opaque white background (vtracer traces dark)."""
    h, w = fg.shape
    rgba = np.full((h, w, 4), 255, np.uint8)
    rgba[fg > 0, :3] = 0
    return [tuple(int(v) for v in px) for row in rgba for px in row], (w, h)


def _rgba_to_vtracer_pixels(rgba: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
    h, w = rgba.shape[:2]
    return [tuple(int(v) for v in px) for row in rgba for px in row], (w, h)


def _trace_vtracer_binary(fg: np.ndarray, cfg: PipelineConfig) -> str:
    import vtracer

    pixels, size = _fg_to_vtracer_pixels(fg)
    return vtracer.convert_pixels_to_svg(
        pixels,
        size,
        colormode="binary",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=int(cfg.vectorize.filter_speckle_px),
        corner_threshold=int(round(cfg.vectorize.corner_threshold_deg)),
        length_threshold=4,
        splice_threshold=45,
        path_precision=3,
    )


def _trace_vtracer_color(rgba: np.ndarray, cfg: PipelineConfig) -> str:
    import vtracer

    pixels, size = _rgba_to_vtracer_pixels(rgba)
    return vtracer.convert_pixels_to_svg(
        pixels,
        size,
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=int(cfg.vectorize.filter_speckle_px),
        color_precision=int(cfg.vectorize.color_precision),
        layer_difference=16,
        corner_threshold=int(round(cfg.vectorize.corner_threshold_deg)),
        length_threshold=4,
        splice_threshold=45,
        path_precision=3,
    )


def _trace_contours_binary(fg: np.ndarray, layer: TraceLayerName) -> str:
    """Fallback: OpenCV contours -> simple SVG polylines."""
    import svgwrite

    h, w = fg.shape
    contours, _hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    dwg = svgwrite.Drawing(size=(w, h))
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        dwg.add(dwg.path(d=points_to_d(pts, True), fill="black"))
    return dwg.tostring()


def _trace_contours_color(rgba: np.ndarray) -> str:
    """Fallback: trace each quantized colour region separately."""
    import svgwrite

    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    dwg = svgwrite.Drawing(size=(w, h))
    rgb = rgba[:, :, :3].reshape(-1, 3)
    keep = (alpha.reshape(-1) > 0)
    colors = np.unique(rgb[keep], axis=0)
    for color in colors:
        cmask = (
            (rgba[:, :, 0] == color[0])
            & (rgba[:, :, 1] == color[1])
            & (rgba[:, :, 2] == color[2])
            & (alpha > 0)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(cmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hexcol = "#%02x%02x%02x" % (int(color[0]), int(color[1]), int(color[2]))
        for cnt in contours:
            if len(cnt) < 3 or cv2.contourArea(cnt) < 4:
                continue
            pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
            dwg.add(dwg.path(d=points_to_d(pts, True), fill=hexcol))
    return dwg.tostring()


def _build_layer(
    name: TraceLayerName,
    svg_text: str,
    svg_path: Path,
    image_size: Tuple[int, int],
    backend: str,
) -> TraceLayer:
    svg_path.write_text(svg_text)
    subpaths = iter_svg_subpaths(svg_text)
    paths: List[VectorPath] = []
    for i, sp in enumerate(subpaths):
        if len(sp.points) < 2:
            continue
        paths.append(
            VectorPath(
                path_id=f"{name.value}_{i:03d}",
                source_layer=name,
                closed=sp.closed,
                confidence=_LAYER_CONFIDENCE[name],
                svg_d=points_to_d(sp.points, sp.closed),
                points=[(float(x), float(y)) for x, y in sp.points],
                area=shoelace_area(sp.points),
            )
        )
    return TraceLayer(
        name=name,
        svg_path=str(svg_path),
        paths=paths,
        image_size=image_size,
        stats={
            "backend": backend,
            "raw_path_count": len(paths),
            "raw_point_count": sum(len(p.points) for p in paths),
        },
    )


def vectorize(
    layers: PreprocessLayers, out_dir: str, cfg: PipelineConfig
) -> List[TraceLayer]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend_req = cfg.vectorize.backend

    # (name, raster path, colormode, dilate_px for thin lines)
    plan = [
        (TraceLayerName.SILHOUETTE, layers.silhouette_path, "binary", 0),
        (TraceLayerName.COLOR_REGIONS, layers.color_quantized_path, "color", 0),
        (TraceLayerName.STRUCTURAL_EDGES, layers.structural_edges_path, "binary", 1),
        (TraceLayerName.DETAILS, layers.details_path, "binary", 1),
    ]

    results: List[TraceLayer] = []
    for name, raster_path, colormode, dilate in plan:
        svg_text = ""
        backend = "vtracer"
        if backend_req in ("auto", "vtracer"):
            try:
                if colormode == "binary":
                    fg = _load_binary_fg(raster_path, dilate_px=dilate)
                    svg_text = _trace_vtracer_binary(fg, cfg)
                    size = (fg.shape[1], fg.shape[0])
                else:
                    rgba = np.asarray(Image.open(raster_path).convert("RGBA"))
                    svg_text = _trace_vtracer_color(rgba, cfg)
                    size = (rgba.shape[1], rgba.shape[0])
            except Exception:  # noqa: BLE001 - fall back per contract
                if backend_req == "vtracer":
                    raise
                svg_text = ""
        if backend_req == "contour" or not svg_text:
            backend = "contour"
            if colormode == "binary":
                fg = _load_binary_fg(raster_path, dilate_px=dilate)
                svg_text = _trace_contours_binary(fg, name)
                size = (fg.shape[1], fg.shape[0])
            else:
                rgba = np.asarray(Image.open(raster_path).convert("RGBA"))
                svg_text = _trace_contours_color(rgba)
                size = (rgba.shape[1], rgba.shape[0])

        results.append(
            _build_layer(name, svg_text, out / f"{name.value}.svg", size, backend)
        )
    return results
