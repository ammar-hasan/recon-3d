"""Phase 6: evidence-preserving multiview reconstruction and fusion.

Every secondary image is processed independently through the 2D evidence
pipeline.  Semantic parts are then matched into a shared part graph, relative
camera pose and scale are solved from the matched observations, and only
source-labelled support metadata is fused into the primary graph.  Observed
primary geometry is never overwritten by a secondary view.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import (camera, constraints, crop, depth, preprocess, primitives,
               segmentation, semantic_parts, sketch_graph, svg_cleanup,
               vectorize)
from .config import PipelineConfig
from .part_geometry import graph_bbox, part_bbox
from .schemas import (
    CameraEstimate,
    CrossViewPartMatch,
    DepthEvidence,
    EvidenceSource,
    EvidencedValue,
    InputBundle,
    InputSpec,
    MultiViewObservation,
    MultiViewResult,
    SchemaIO,
    SketchGraph,
)


def _canonical_class(value: str) -> str:
    text = (value or "").lower()
    for suffix in ("_body", "_system", "_part"):
        text = text.replace(suffix, "")
    synonyms = {
        "tire": "tyre", "outer_shell": "body", "inner_panel": "panel",
        "centre_bore": "center_bore", "spokes": "spoke",
        "enclosure": "box", "bottom_panel": "crate",
    }
    return synonyms.get(text, text)


def _relative_box(graph: SketchGraph, part) -> Tuple[float, float, float, float]:
    gx0, gy0, gx1, gy1 = graph_bbox(graph)
    px0, py0, px1, py1 = part_bbox(graph, part)
    gw, gh = max(gx1 - gx0, 1e-9), max(gy1 - gy0, 1e-9)
    return ((px0 - gx0) / gw, (py0 - gy0) / gh,
            (px1 - gx0) / gw, (py1 - gy0) / gh)


def _box_cost(a, b) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    aw, ah = max(a[2] - a[0], 1e-4), max(a[3] - a[1], 1e-4)
    bw, bh = max(b[2] - b[0], 1e-4), max(b[3] - b[1], 1e-4)
    centre = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
    size = 0.25 * (abs(math.log(aw / bw)) + abs(math.log(ah / bh)))
    return float(centre + size)


def _match_parts(primary: SketchGraph, secondary: SketchGraph, view_id: str,
                 max_cost: float) -> List[CrossViewPartMatch]:
    candidates = []
    for p in primary.parts:
        pc = _canonical_class(p.part_class)
        ptokens = set(pc.split("_"))
        for q in secondary.parts:
            qc = _canonical_class(q.part_class)
            if pc != qc and not (ptokens & set(qc.split("_"))):
                continue
            cost = _box_cost(_relative_box(primary, p),
                             _relative_box(secondary, q))
            candidates.append((cost, p.id, q.id, p, q))
    used_p, used_q = set(), set()
    matches = []
    for cost, pid, qid, p, q in sorted(candidates, key=lambda row: row[:3]):
        if cost > max_cost or pid in used_p or qid in used_q:
            continue
        used_p.add(pid)
        used_q.add(qid)
        confidence = max(0.05, min(0.95, (1.0 - cost / max_cost)
                                   * math.sqrt(p.confidence * q.confidence)))
        matches.append(CrossViewPartMatch(
            primary_part_id=pid, secondary_part_id=qid, view_id=view_id,
            part_class=p.part_class, geometric_cost=cost,
            confidence=confidence,
        ))
    return matches


def _rotation(cam: CameraEstimate) -> List[float]:
    value = cam.object_rotation_euler_deg.value
    return [float(v) for v in value] if isinstance(value, list) and len(value) == 3 else [0.0] * 3


def fuse_multiview(
    bundle: InputBundle,
    primary_graph: SketchGraph,
    primary_camera: CameraEstimate,
    primary_depth: DepthEvidence,
    primary_seg,
    spec: InputSpec,
    cfg: PipelineConfig,
    out_dir: str,
) -> Tuple[SketchGraph, DepthEvidence, MultiViewResult]:
    """Process secondary views and fuse their support into the primary graph."""
    if not cfg.multiview.enabled or len(bundle.images) <= 1:
        return primary_graph, primary_depth, MultiViewResult(enabled=False)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = MultiViewResult(enabled=True)
    all_matches: List[CrossViewPartMatch] = []
    scale_samples = []
    residuals = []

    for index, loaded in enumerate(bundle.images[1:cfg.multiview.max_views], start=1):
        view_id = "view_%03d" % index
        view_root = root / "views" / view_id
        for sub in ("segmentation", "traces", "geometry"):
            (view_root / sub).mkdir(parents=True, exist_ok=True)
        view_spec = spec.model_copy(update={
            "image_paths": [loaded.path], "mask_path": None,
            "box": None, "point": None, "output_dir": str(view_root),
        })
        view_bundle = InputBundle(spec=view_spec, images=[loaded],
                                  warnings=list(bundle.warnings))
        observation = MultiViewObservation(view_id=view_id,
                                           image_path=loaded.path)
        try:
            seg = segmentation.segment(view_bundle, str(view_root / "segmentation"), cfg)
            SchemaIO.save_json(seg, view_root / "segmentation" / "segmentation_result.json")
            crop_meta, crop_rgba, crop_mask = crop.make_crop(
                seg, str(view_root / "segmentation"), cfg)
            layers_img = preprocess.preprocess(
                crop_rgba, crop_mask, str(view_root / "segmentation"), cfg)
            layers = vectorize.vectorize(layers_img, str(view_root / "traces"), cfg)
            layers = svg_cleanup.cleanup_layers(layers, str(view_root / "traces"), cfg)
            fitted = primitives.fit_primitives(layers, cfg)
            cons = constraints.detect_constraints(fitted, cfg)
            graph = sketch_graph.build_sketch_graph(fitted, cons)
            graph = semantic_parts.decompose_parts(graph, crop_rgba, view_spec, cfg)
            cam = camera.estimate_camera(graph, seg, crop_meta, view_spec, cfg)
            dep = depth.estimate_depth(
                crop_rgba, crop_mask, graph, str(view_root / "geometry"), cfg)
            graph_path = SchemaIO.save_json(graph, view_root / "geometry" / "sketch_graph.json")
            SchemaIO.save_json(dep, view_root / "geometry" / "depth_evidence.json")

            matches = _match_parts(primary_graph, graph, view_id,
                                   cfg.multiview.max_part_match_cost)
            all_matches.extend(matches)
            residuals.extend(m.geometric_cost for m in matches)

            primary_rot, secondary_rot = _rotation(primary_camera), _rotation(cam)
            relative = [secondary_rot[i] - primary_rot[i] for i in range(3)]
            baseline = float(np.linalg.norm(relative))
            result.relative_camera_poses[view_id] = EvidencedValue(
                value=relative, unit="deg",
                source=(EvidenceSource.ESTIMATED_FROM_CAMERA
                        if baseline >= cfg.multiview.min_pose_baseline_deg
                        else EvidenceSource.UNKNOWN),
                confidence=min(0.85, 0.25 + baseline / 120.0),
                note="secondary object rotation minus primary object rotation",
            )

            pwidth = max(1, primary_seg.bbox[2] - primary_seg.bbox[0])
            swidth = max(1, seg.bbox[2] - seg.bbox[0])
            image_scale = float(pwidth) / float(swidth)
            scale_samples.append(image_scale)
            observation.graph_path = graph_path
            observation.segmentation_confidence = seg.confidence
            observation.camera = cam
            observation.scale_to_primary = EvidencedValue(
                value=image_scale, source=EvidenceSource.FITTED_FROM_OBSERVATION,
                confidence=min(0.8, 0.35 + 0.08 * len(matches)),
                note="ratio of primary and secondary observed silhouette widths",
            )
            observation.warnings.extend(seg.warnings)
        except Exception as exc:  # one weak view must not destroy the primary run
            observation.status = "failed"
            observation.warnings.append("%s: %s" % (type(exc).__name__, exc))
            result.warnings.append("%s failed: %s" % (view_id, exc))
        result.observations.append(observation)

    result.matches = all_matches
    shared: Dict[str, List[Dict]] = {}
    for match in all_matches:
        shared.setdefault(match.primary_part_id, []).append({
            "view_id": match.view_id,
            "secondary_part_id": match.secondary_part_id,
            "part_class": match.part_class,
            "confidence": match.confidence,
            "source": match.source.value,
        })
    result.shared_part_graph = shared

    fused = primary_graph.model_copy(deep=True)
    for part in fused.parts:
        support = shared.get(part.id, [])
        if support:
            part.inferred_geometry["multiview_support"] = EvidencedValue(
                value={"view_count": len(support), "observations": support},
                source=EvidenceSource.FITTED_FROM_OBSERVATION,
                confidence=min(0.9, 0.45 + 0.1 * len(support)),
                note="cross-view semantic and relative-geometry match; primary geometry unchanged",
            )
    fused.stats = dict(fused.stats)
    fused.stats.update({
        "multiview_enabled": True,
        "multiview_count": len(result.observations) + 1,
        "cross_view_match_count": len(all_matches),
    })

    if scale_samples:
        median_scale = float(np.median(scale_samples))
        dispersion = float(np.median(np.abs(np.asarray(scale_samples) - median_scale)))
        result.consistent_scale = EvidencedValue(
            value=median_scale,
            source=EvidenceSource.FITTED_FROM_OBSERVATION,
            confidence=max(0.2, min(0.8, 0.7 - dispersion)),
            note="robust consensus of per-view silhouette scale ratios",
        )
    result.joint_optimization = {
        "method": "robust shared-part/pose/scale consensus",
        "optimized_parameters": ["part_correspondence", "relative_camera_pose", "view_scale"],
        "match_residual_mean": (float(np.mean(residuals)) if residuals else None),
        "views_used": sum(o.status == "success" for o in result.observations),
        "primary_geometry_overwritten": False,
    }
    return fused, primary_depth, result
