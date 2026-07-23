"""End-to-end pipeline orchestrator.

Usage:
    python -m recon3d.pipeline --image path/to/img.png [--label wheel]
        [--box x0 y0 x1 y1] [--point x y] [--mask m.png] [--dimension 0.65]
        [--out projects/run] [--config cfg.yaml]
"""
from __future__ import annotations

import argparse
import datetime
import json
import resource
import sys
import time
import traceback
import uuid
from pathlib import Path

from .config import PipelineConfig, software_versions
from .schemas import (InputSpec, RefinementLog, RunManifest, SchemaIO,
                      sha256_file)


def _ensure_dirs(project_dir: Path) -> None:
    for sub in ("input", "segmentation", "traces", "geometry", "blender", "validation"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)


def _result_status(validation_passed: bool, input_quality_risk: str) -> str:
    if validation_passed and input_quality_risk != "high":
        return "success"
    return "partial_success"


def run_pipeline(spec: InputSpec, cfg: PipelineConfig) -> RunManifest:
    from . import (blender_codegen, camera, constraints, construction_plan, crop,
                   depth, hypotheses, input_manager, input_quality, multiview,
                   multiview_refinement, multiview_visual_hull, operators,
                   preprocess, primitives,
                   refinement, runner, segmentation, semantic_parts, sketch_graph,
                   svg_cleanup, uncertainty, validation, vectorize)

    project_dir = Path(spec.output_dir)
    _ensure_dirs(project_dir)

    manifest = RunManifest(
        run_id=uuid.uuid4().hex[:12],
        software=software_versions(),
        seeds={"pipeline": cfg.seed},
        started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        config={},
    )
    manifest.config = json.loads(cfg.model_dump_json())

    quality = None
    run_clock = time.perf_counter()
    stage_clock = run_clock
    try:
        # Stage 1-3: input, segmentation, crop
        bundle = input_manager.load_input(spec)
        manifest.input_hashes = {img.path: img.sha256 for img in bundle.images}
        quality = input_quality.assess_input_quality(bundle)
        quality_path = SchemaIO.save_json(
            quality, project_dir / "input" / "quality_assessment.json")
        manifest.stage_outputs["input_quality_assessment"] = quality_path
        manifest.stage_outputs["input_quality_risk"] = quality.risk
        seg = segmentation.segment(bundle, str(project_dir / "segmentation"), cfg)
        SchemaIO.save_json(seg, project_dir / "segmentation" / "segmentation_result.json")
        crop_meta, crop_rgba, crop_mask = crop.make_crop(
            seg, str(project_dir / "segmentation"), cfg)
        quality = input_quality.assess_input_quality(bundle, seg)
        quality_path = SchemaIO.save_json(
            quality, project_dir / "input" / "quality_assessment.json")
        manifest.stage_outputs["input_quality_assessment"] = quality_path
        manifest.stage_outputs["input_quality_risk"] = quality.risk
        if quality.signals:
            quality_warnings = [
                "input quality %s: %s" % (signal.code, signal.evidence)
                for signal in quality.signals]
            seg = seg.model_copy(update={
                "warnings": list(seg.warnings) + quality_warnings})
            SchemaIO.save_json(
                seg, project_dir / "segmentation" / "segmentation_result.json")
        now = time.perf_counter()
        manifest.timings_seconds["input_segmentation_crop"] = round(
            now - stage_clock, 6)
        stage_clock = now

        # Stage 4-6: preprocess, vectorize, cleanup
        layers_img = preprocess.preprocess(crop_rgba, crop_mask,
                                           str(project_dir / "segmentation"), cfg)
        layers = vectorize.vectorize(layers_img, str(project_dir / "traces"), cfg)
        layers = svg_cleanup.cleanup_layers(layers, str(project_dir / "traces"), cfg)
        now = time.perf_counter()
        manifest.timings_seconds["preprocess_vectorize_cleanup"] = round(
            now - stage_clock, 6)
        stage_clock = now

        # Stage 7-10: primitives, constraints, sketch graph, semantic parts
        prims = primitives.fit_primitives(layers, cfg)
        cons = constraints.detect_constraints(prims, cfg)
        graph = sketch_graph.build_sketch_graph(prims, cons)
        graph = semantic_parts.decompose_parts(graph, crop_rgba, spec, cfg)
        SchemaIO.save_json(
            graph.model_copy(update={"parts": [], "constraints": []}),
            project_dir / "geometry" / "fitted_primitives.json")
        now = time.perf_counter()
        manifest.timings_seconds["primitives_constraints_semantics"] = round(
            now - stage_clock, 6)
        stage_clock = now

        # Stage 11-14: camera, depth, operators, plan
        cam = camera.estimate_camera(graph, seg, crop_meta, spec, cfg)
        dep = depth.estimate_depth(crop_rgba, crop_mask, graph,
                                   str(project_dir / "geometry"), cfg)

        # Phase 6: independent secondary-view reconstruction followed by
        # source-labelled shared-part, relative-pose, and scale fusion.
        graph, dep, mv_result = multiview.fuse_multiview(
            bundle, graph, cam, dep, seg, spec, cfg,
            str(project_dir / "geometry" / "multiview"))
        SchemaIO.save_json(mv_result, project_dir / "geometry" / "multiview.json")

        if not cfg.uncertainty.enabled:
            graph = uncertainty.disable_tracking(graph)
            cam = uncertainty.disable_tracking(cam)
            dep = uncertainty.disable_tracking(dep)
            mv_result = uncertainty.disable_tracking(mv_result)
            mv_result.warnings.append(
                "uncertainty tracking disabled by ablation; all input "
                "confidences forced to 1.0")
            SchemaIO.save_json(
                mv_result, project_dir / "geometry" / "multiview.json")

        graph = operators.classify_operators(graph, dep, cfg)

        # Phase 7: optional hidden-geometry proposals are explicitly scored,
        # accepted/rejected, and kept separate from observed primitives.
        graph, hypothesis_report = hypotheses.evaluate_hypotheses(
            graph, mv_result, cfg)
        SchemaIO.save_json(
            hypothesis_report, project_dir / "geometry" / "hypotheses.json")
        SchemaIO.save_json(graph, project_dir / "geometry" / "sketch_graph.json")
        plan = construction_plan.build_plan(graph, cam, dep, spec, cfg)
        if not cfg.uncertainty.enabled:
            plan = uncertainty.disable_tracking(plan)
            plan.metadata = dict(plan.metadata)
            plan.metadata["uncertainty_tracking"] = "disabled_ablation"
        plan, mv_result, visual_hull_used = (
            multiview_visual_hull.augment_plan_with_visual_hull(
                plan, mv_result, seg, cfg))
        if not cfg.uncertainty.enabled:
            plan = uncertainty.disable_tracking(plan)
            plan.metadata = dict(plan.metadata)
            plan.metadata["uncertainty_tracking"] = "disabled_ablation"
            mv_result = uncertainty.disable_tracking(mv_result)
        SchemaIO.save_json(mv_result, project_dir / "geometry" / "multiview.json")
        errors = construction_plan.validate_plan(plan)
        if errors:
            raise ValueError("construction plan invalid: " + "; ".join(errors))
        SchemaIO.save_yaml(plan, project_dir / "geometry" / "construction_plan.yaml")
        now = time.perf_counter()
        manifest.timings_seconds["camera_depth_multiview_planning"] = round(
            now - stage_clock, 6)
        stage_clock = now

        # Stage 15-16: Blender generation + execution
        script = blender_codegen.generate_blender_script(plan, str(project_dir / "blender"), cfg)
        bman = runner.run_blender(script, str(project_dir), cfg)
        if not bman.success:
            raise RuntimeError("blender execution failed: " + "; ".join(bman.errors))

        # Phase 6 joint geometry step: secondary silhouettes solve relative
        # yaw and the shared hidden depth extent.  Only inferred Z scale may
        # change; primary observed XY curves remain untouched.
        if visual_hull_used:
            rebuild = False
        else:
            plan, mv_result, rebuild = multiview_refinement.refine_multiview_geometry(
                plan, mv_result, str(project_dir), cfg, seg, crop_meta)
        SchemaIO.save_json(mv_result, project_dir / "geometry" / "multiview.json")
        if rebuild:
            SchemaIO.save_yaml(
                plan, project_dir / "geometry" / "construction_plan.yaml")
            script = blender_codegen.generate_blender_script(
                plan, str(project_dir / "blender"), cfg)
            bman = runner.run_blender(script, str(project_dir), cfg)
            if not bman.success:
                raise RuntimeError(
                    "multiview-refined blender execution failed: "
                    + "; ".join(bman.errors))
        now = time.perf_counter()
        manifest.timings_seconds["blender_build_and_multiview_geometry"] = round(
            now - stage_clock, 6)
        stage_clock = now

        # Stage 17-18: validation + refinement
        val = validation.validate_reconstruction(bman, plan, seg, crop_meta,
                                                 str(project_dir), cfg)
        if cfg.refinement.enabled:
            plan, bman, val, rlog = refinement.refine(
                plan, seg, crop_meta, str(project_dir), cfg)
        else:
            rlog = RefinementLog(
                initial_metrics={"silhouette_iou": val.metrics.silhouette_iou},
                final_metrics={"silhouette_iou": val.metrics.silhouette_iou},
                converged=val.passed, iterations=0)
        SchemaIO.save_yaml(plan, project_dir / "geometry" / "construction_plan.yaml")
        SchemaIO.save_json(rlog, project_dir / "validation" / "refinement_log.json")
        now = time.perf_counter()
        manifest.timings_seconds["validation_and_refinement"] = round(
            now - stage_clock, 6)

        manifest.status = _result_status(
            val.passed, quality.risk if quality is not None else "high")
    except input_manager.InputError as exc:
        # Invalid or insufficient source evidence is an input outcome, not a
        # reconstruction failure.  Keep the partial project and a machine-
        # readable reason so callers can distinguish it from failed geometry.
        manifest.status = "unsupported_input"
        manifest.stage_outputs["error"] = f"{type(exc).__name__}: {exc}"
        manifest.stage_outputs["unsupported_input_reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - orchestrator must never crash silently
        manifest.status = (
            "partial_success" if quality is not None and quality.risk == "high"
            else "failed_validation")
        manifest.stage_outputs["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        manifest.timings_seconds["total"] = round(
            time.perf_counter() - run_clock, 6)
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        manifest.resource_usage = {
            "peak_process_rss_mb": round(peak_rss / divisor, 3),
            "gpu_memory_mb": None,
            "model_calls": 0,
        }
        manifest.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        SchemaIO.save_json(manifest, project_dir / "manifest.json")
        try:
            from . import reporting
            reporting.generate_report(project_dir)
        except Exception:  # noqa: BLE001 - reporting must never mask the run
            traceback.print_exc()

    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recon3d")
    ap.add_argument("--image", action="append", required=True,
                    help="input image (repeatable for multiview)")
    ap.add_argument("--label", default=None, help="target object label")
    ap.add_argument("--description", default=None)
    ap.add_argument("--box", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--point", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--mask", default=None, help="user-supplied mask")
    ap.add_argument("--dimension", type=float, default=None, help="known physical size")
    ap.add_argument("--dimension-axis", default=None)
    ap.add_argument(
        "--view-azimuth", action="append", type=float, default=None,
        help="calibrated camera-orbit azimuth in degrees; repeat once per --image",
    )
    ap.add_argument(
        "--camera-json", action="append", default=None,
        help="full camera calibration JSON; repeat once per --image",
    )
    ap.add_argument("--out", default="projects/run")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    spec = InputSpec(
        image_paths=args.image,
        description=args.description,
        target_label=args.label,
        point=tuple(args.point) if args.point else None,
        box=tuple(args.box) if args.box else None,
        mask_path=args.mask,
        known_dimension=args.dimension,
        known_dimension_axis=args.dimension_axis,
        view_azimuths_deg=args.view_azimuth,
        camera_calibration_paths=args.camera_json,
        output_dir=args.out,
    )
    manifest = run_pipeline(spec, cfg)
    print(f"status: {manifest.status}")
    print(f"project: {args.out}")
    return 0 if manifest.status in ("success", "partial_success") else 1


if __name__ == "__main__":
    sys.exit(main())
