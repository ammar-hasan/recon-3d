"""Diagnostic report generation (GOAL.md: inspectability + report.md output).

Renders a per-run Markdown report from the artifacts already on disk:
what was observed vs inferred, stage diagnostics, validation metrics,
refinement audit, and explicit uncertainties.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .schemas import (BlenderManifest, ConstructionPlan, RefinementLog,
                      RunManifest, SchemaIO, SegmentationResult, SketchGraph,
                      ValidationMetrics)


def _yn(v: Optional[bool]) -> str:
    return "yes" if v else "no"


def _f(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def generate_report(project_dir: str | Path) -> str:
    """Build report.md in the project directory from its artifacts.

    Tolerates partially-completed runs (missing artifacts are reported as
    such). Returns the report path.
    """
    pdir = Path(project_dir)
    lines = ["# Reconstruction Report", ""]

    # --- manifest ---------------------------------------------------------
    manifest: Optional[RunManifest] = None
    if (pdir / "manifest.json").exists():
        manifest = SchemaIO.load_json(RunManifest, pdir / "manifest.json")
        lines += [
            f"- run id: `{manifest.run_id}`",
            f"- status: **{manifest.status}**",
            f"- started: {manifest.started_at}  finished: {manifest.finished_at}",
            f"- blender: {manifest.software.get('blender', 'n/a')}"
            if "blender" in manifest.software else "",
            "",
            "## Reproducibility",
            "",
        ]
        for k, v in manifest.input_hashes.items():
            lines.append(f"- input `{Path(k).name}` sha256 `{v[:16]}…`")
        lines.append(f"- pipeline seed: {manifest.seeds.get('pipeline', 'n/a')}")
        sw = ", ".join(f"{k} {v}" for k, v in sorted(manifest.software.items()))
        lines.append(f"- software: {sw}")
        lines.append("")

    # --- segmentation -----------------------------------------------------
    seg_path = pdir / "segmentation" / "segmentation_result.json"
    if seg_path.exists():
        seg = SchemaIO.load_json(SegmentationResult, seg_path)
        lines += [
            "## Segmentation",
            "",
            f"- backend: {seg.backend} (confidence {seg.confidence:.2f})",
            f"- foreground coverage: {seg.coverage:.3f}",
            f"- bbox: {list(seg.bbox)}",
        ]
        for w in seg.warnings:
            lines.append(f"- warning: {w}")
        lines.append("")

    # --- sketch graph -----------------------------------------------------
    graph: Optional[SketchGraph] = None
    if (pdir / "geometry" / "sketch_graph.json").exists():
        graph = SchemaIO.load_json(SketchGraph, pdir / "geometry" / "sketch_graph.json")
        n_prim = len(graph.primitives)
        by_type = {}
        for pr in graph.primitives:
            by_type[pr.type.value] = by_type.get(pr.type.value, 0) + 1
        lines += [
            "## 2D Geometry",
            "",
            f"- primitives fitted: {n_prim} "
            + "(" + ", ".join(f"{k}×{v}" for k, v in sorted(by_type.items())) + ")",
            f"- constraints: {len(graph.constraints)}",
        ]
        for c in graph.constraints[:20]:
            lines.append(f"  - {c.type.value} on {', '.join(c.entities)} "
                         f"(conf {c.confidence:.2f})")
        if len(graph.constraints) > 20:
            lines.append(f"  - … and {len(graph.constraints) - 20} more")
        lines += ["", "## Semantic Parts", ""]
        for part in graph.parts:
            vis = part.visibility.value
            op = part.selected_operator or "unclassified"
            lines.append(f"- `{part.id}` ({part.part_class}, {vis}, op={op}, "
                         f"conf {part.confidence:.2f})"
                         + (f" parent={part.parent_id}" if part.parent_id else ""))
            for key, ev in part.inferred_geometry.items():
                lines.append(f"    - inferred `{key}`: {ev.source.value} "
                             f"(conf {ev.confidence:.2f})")
        lines.append("")

    # --- construction plan ------------------------------------------------
    plan: Optional[ConstructionPlan] = None
    if (pdir / "geometry" / "construction_plan.yaml").exists():
        plan = SchemaIO.load_yaml(ConstructionPlan,
                                  pdir / "geometry" / "construction_plan.yaml")
        lines += [
            "## Construction Plan",
            "",
            f"- object: `{plan.object_id}` units={plan.units}",
            f"- physical width: {plan.physical_width if plan.physical_width else 'unknown'}",
            f"- parts: {len(plan.parts)}",
        ]
        for pp in plan.parts:
            lines.append(f"  - `{pp.id}` {pp.operator.value}"
                         + (f" ({pp.primitive_shape})" if pp.primitive_shape else "")
                         + (f" ×{pp.count}" if pp.count else "")
                         + f" mat={pp.material.material_class}"
                         + (f" [{pp.visibility.value}]"
                            if pp.visibility.value != "visible" else ""))
        if plan.uncertainty:
            lines += ["", "### Uncertainty", ""]
            for k, v in plan.uncertainty.items():
                lines.append(f"- {k}: {v}")
        lines.append("")

    # --- blender ----------------------------------------------------------
    bman: Optional[BlenderManifest] = None
    bman_path = pdir / "blender" / "blender_manifest.json"
    if bman_path.exists():
        try:
            # the on-disk manifest is written inside Blender and lacks the
            # runner-side script_path field; tolerate that
            data = json.loads(bman_path.read_text())
            data.setdefault("script_path", str(bman_path.parent / "build_model.py"))
            bman = BlenderManifest.model_validate(data)
        except Exception:
            bman = None
    if bman is not None:
        lines += [
            "## Blender Scene",
            "",
            f"- execution success: {_yn(bman.success)}",
            f"- blender version: {bman.blender_version or 'n/a'}",
            f"- objects: {len(bman.objects)}  collections: {len(bman.collections)}",
            f"- blend: `{Path(bman.blend_path).name}`  glb: "
            f"`{Path(bman.glb_path).name if bman.glb_path else 'n/a'}`",
        ]
        for e in bman.errors[:5]:
            lines.append(f"- error: {e}")
        lines.append("")

    # --- validation -------------------------------------------------------
    metrics: Optional[ValidationMetrics] = None
    metrics_path = pdir / "validation" / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = SchemaIO.load_json(ValidationMetrics, metrics_path)
        except Exception:
            metrics = None
    if metrics is not None:
        target = 0.9
        if manifest is not None:
            target = float((manifest.config.get("refinement") or {})
                           .get("target_silhouette_iou", target))
        lines += [
            "## Validation",
            "",
            f"- silhouette IoU: {_f(metrics.silhouette_iou)}",
            f"- clay silhouette IoU: {_f(metrics.clay_silhouette_iou)}",
            f"- contour chamfer: {_f(metrics.contour_chamfer_distance, 4)}",
            f"- feature alignment error (px): {_f(metrics.feature_alignment_error_px, 1)}",
            f"- depth correlation: {_f(metrics.depth_correlation)}",
            f"- color region agreement: {_f(metrics.color_region_agreement)}",
            f"- perceptual similarity (SSIM): {_f(metrics.perceptual_similarity)}",
            f"- passed: {_yn((metrics.silhouette_iou or 0.0) >= target)}",
            "",
        ]

    # --- refinement -------------------------------------------------------
    if (pdir / "validation" / "refinement_log.json").exists():
        try:
            rlog = SchemaIO.load_json(RefinementLog,
                                      pdir / "validation" / "refinement_log.json")
            lines += [
                "## Refinement",
                "",
                f"- iterations: {rlog.iterations}  converged: {_yn(rlog.converged)}",
                f"- initial silhouette IoU: "
                f"{_f(rlog.initial_metrics.get('silhouette_iou'))} → final: "
                f"{_f(rlog.final_metrics.get('silhouette_iou'))}",
                "",
                "| iter | problem | kept |",
                "|---|---|---|",
            ]
            for a in rlog.actions:
                lines.append(f"| {a.iteration} | {a.observed_problem} | {_yn(a.kept)} |")
            lines.append("")
        except Exception:
            pass

    # --- honesty section --------------------------------------------------
    lines += [
        "## Known Limitations",
        "",
        "A single RGB image cannot uniquely determine absolute scale, hidden",
        "surfaces, exact depth, internal structure, rear geometry, or material",
        "composition. All such properties in this reconstruction are marked as",
        "inferred/hypothesised with reduced confidence in the sketch graph and",
        "construction plan.",
        "",
    ]

    out = pdir / "report.md"
    out.write_text("\n".join(lines))
    return str(out)
