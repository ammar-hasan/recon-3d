"""Stage 6: SVG cleanup and simplification.

Per layer: drop tiny paths, remove near-duplicate paths (Hausdorff), RDP
simplify (cv2.approxPolyDP), optional light Chaikin smoothing, near-closure
fixing, containment/hole hierarchy detection, winding normalisation, and
coordinate normalisation to 0..1. Topology is preserved: open paths stay
open, holes are never removed, components are never merged.

Outputs per layer: cleaned_<layer>.json (TraceLayer) + cleaned_<layer>.svg.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .schemas import SchemaIO, TraceLayer, VectorPath
from .svgpath import points_to_d, polyline_length, shoelace_area

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# geometry helpers (normalised coordinates)
# ---------------------------------------------------------------------------

def _subsample(points: List[Point], n: int = 64) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def _point_to_polyline_dist(p: np.ndarray, poly: np.ndarray, closed: bool) -> float:
    """Minimum distance from point p to a polyline."""
    segs_a = poly
    segs_b = np.roll(poly, -1, axis=0)
    if not closed:
        segs_a, segs_b = segs_a[:-1], segs_b[:-1]
    ab = segs_b - segs_a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-12
    t = ((p - segs_a) * ab).sum(axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = segs_a + t[:, None] * ab
    d = np.linalg.norm(proj - p, axis=1)
    return float(d.min())


def _hausdorff(a: List[Point], b: List[Point], closed_a: bool, closed_b: bool) -> float:
    pa, pb = _subsample(a), _subsample(b)
    d1 = max(_point_to_polyline_dist(p, pb, closed_b) for p in pa)
    d2 = max(_point_to_polyline_dist(p, pa, closed_a) for p in pb)
    return max(d1, d2)


def _rdp(points: List[Point], closed: bool, eps: float) -> List[Point]:
    """Ramer-Douglas-Peucker via cv2.approxPolyDP (deterministic)."""
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, float(eps), closed)
    out = [(float(p[0][0]), float(p[0][1])) for p in approx]
    # drop consecutive duplicates
    dedup: List[Point] = []
    for p in out:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    min_pts = 3 if closed else 2
    if len(dedup) < min_pts:
        return list(points)
    return dedup


def _chaikin(points: List[Point], closed: bool) -> List[Point]:
    if len(points) < 3:
        return list(points)
    out: List[Point] = []
    n = len(points)
    rng = range(n) if closed else range(n - 1)
    if not closed:
        out.append(points[0])
    for i in rng:
        p0 = points[i]
        p1 = points[(i + 1) % n]
        out.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
        out.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
    if not closed:
        out.append(points[-1])
    return out


def _interior_probe(points: List[Point]) -> Point:
    """A point guaranteed (near-)inside the polygon: vertex nudged to centroid."""
    arr = np.asarray(points, dtype=np.float64)
    centroid = arr.mean(axis=0)
    p = 0.98 * arr[0] + 0.02 * centroid
    return (float(p[0]), float(p[1]))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def cleanup_layers(
    layers: List[TraceLayer], out_dir: str, cfg: PipelineConfig
) -> List[TraceLayer]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = cfg.cleanup

    cleaned_layers: List[TraceLayer] = []
    for layer in layers:
        w, h = layer.image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"layer {layer.name} has invalid image_size {layer.image_size}")

        raw_paths = layer.paths
        raw_point_count = sum(len(p.points) for p in raw_paths)

        # normalise to 0..1 over the crop canvas
        work: List[VectorPath] = []
        for p in raw_paths:
            pts = [(x / w, y / h) for x, y in p.points]
            work.append(
                p.model_copy(update={"points": pts, "area": shoelace_area(pts)})
            )

        # 1. drop tiny paths
        min_len = 4.0 * (c.min_path_area_norm ** 0.5)
        kept: List[VectorPath] = []
        dropped_tiny = 0
        for p in work:
            if p.closed and abs(p.area) < c.min_path_area_norm:
                dropped_tiny += 1
                continue
            if not p.closed and polyline_length(p.points) < min_len:
                dropped_tiny += 1
                continue
            kept.append(p)

        # 2. remove near-duplicates (Hausdorff), keeping the larger/longer one
        dropped_dupes = 0
        surviving: List[VectorPath] = []
        for p in kept:
            dupe_of = None
            for q in surviving:
                if p.closed != q.closed:
                    continue
                if _hausdorff(p.points, q.points, p.closed, q.closed) < c.dedupe_distance_norm:
                    dupe_of = q
                    break
            if dupe_of is not None:
                dropped_dupes += 1
                # keep whichever carries more shape information
                if len(p.points) > len(dupe_of.points):
                    surviving.remove(dupe_of)
                    surviving.append(p)
                continue
            surviving.append(p)

        # 3-5. closure fix, optional Chaikin smoothing, RDP simplify.
        # Smoothing runs on the dense polyline BEFORE simplification: Chaikin
        # on an already-coarse polygon would cut corners by ~1/4 edge length.
        processed: List[VectorPath] = []
        for p in surviving:
            pts = list(p.points)
            closed = p.closed
            if not closed and len(pts) >= 4:
                gap = float(np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]))
                if gap < 3.0 * c.simplify_tolerance_norm:
                    closed = True  # near-closed -> closed
            if c.smooth:
                pts = _chaikin(pts, closed)
            pts = _rdp(pts, closed, c.simplify_tolerance_norm)
            processed.append(
                p.model_copy(update={"points": pts, "closed": closed,
                                     "area": shoelace_area(pts)})
            )

        # 6. containment hierarchy -> is_hole / parent_path_id
        closed_paths = [p for p in processed if p.closed and len(p.points) >= 3]
        contours = {
            p.path_id: np.asarray(p.points, dtype=np.float32).reshape(-1, 1, 2)
            for p in closed_paths
        }
        containers: dict = {p.path_id: [] for p in closed_paths}
        for p in closed_paths:
            probe = _interior_probe(p.points)
            for q in closed_paths:
                if q.path_id == p.path_id:
                    continue
                if cv2.pointPolygonTest(contours[q.path_id], probe, False) > 0:
                    containers[p.path_id].append(q)
        for p in processed:
            if not p.closed:
                continue
            parents = containers.get(p.path_id, [])
            is_hole = len(parents) % 2 == 1
            parent_id: Optional[str] = None
            if parents:
                parent_id = min(parents, key=lambda q: abs(q.area)).path_id
            p.is_hole = is_hole
            p.parent_path_id = parent_id

        # 7. winding normalisation: outers positive area, holes negative
        for p in processed:
            if not p.closed:
                continue
            if (not p.is_hole and p.area < 0) or (p.is_hole and p.area > 0):
                p.points = list(reversed(p.points))
                p.area = -p.area
            p.svg_d = points_to_d(p.points, p.closed)
        for p in processed:
            if not p.closed:
                p.svg_d = points_to_d(p.points, p.closed)

        cleaned_point_count = sum(len(p.points) for p in processed)
        stats = dict(layer.stats)
        stats.update(
            {
                "raw_path_count": len(raw_paths),
                "raw_point_count": raw_point_count,
                "cleaned_path_count": len(processed),
                "cleaned_point_count": cleaned_point_count,
                "point_reduction": (
                    1.0 - cleaned_point_count / raw_point_count if raw_point_count else 0.0
                ),
                "dropped_tiny": dropped_tiny,
                "dropped_duplicates": dropped_dupes,
                "coordinate_space": "normalized_0_1",
            }
        )

        # re-emit cleaned SVG in pixel coordinates
        svg_path = out / f"cleaned_{layer.name.value}.svg"
        parts = [
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        ]
        for p in processed:
            d_px = points_to_d([(x * w, y * h) for x, y in p.points], p.closed)
            fill = "black" if p.closed else "none"
            stroke = ' stroke="black" stroke-width="1"' if not p.closed else ""
            parts.append(f'<path d="{d_px}" fill="{fill}"{stroke}/>')
        parts.append("</svg>")
        svg_path.write_text("\n".join(parts))

        new_layer = TraceLayer(
            name=layer.name,
            svg_path=str(svg_path),
            paths=processed,
            image_size=layer.image_size,
            stats=stats,
        )
        SchemaIO.save_json(new_layer, out / f"cleaned_{layer.name.value}.json")
        cleaned_layers.append(new_layer)

    return cleaned_layers
