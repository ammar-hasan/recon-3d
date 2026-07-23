"""Stage 18: iterative refinement.

Loop: generate -> run Blender -> render+compare -> diagnose the largest
mismatch -> adjust the responsible plan parameter (bounded coordinate
descent) -> repeat. Keeps the best-valid plan, rolls back regressions, and
records a full audit trail in RefinementLog.

Hard guarantees: never loops forever (max_iterations, max_renders), and a
Blender failure stops the loop with the last good model restored.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import blender_codegen, runner, validation
from .config import PipelineConfig
from .schemas import (BlenderManifest, CameraEstimate, ConstructionPlan,
                      CropMetadata, EvidencedValue, EvidenceSource,
                      RefinementAction, RefinementLog, SchemaIO,
                      SegmentationResult, ValidationResult)

# cumulative bounds for adjusted parameters
_SCALE_BOUNDS = (0.5, 2.0)
_STEP_LIMIT = 1.25           # max multiplicative change per iteration
_SCALE_STEP_LIMIT = 1.02     # silhouette bbox ratios are noisy; avoid overshoot
_DIST_BOUNDS = (0.5, 20.0)


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------

def _mask_stats(mask: np.ndarray) -> Optional[Dict[str, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    h, w = mask.shape
    return {
        "x0": float(xs.min()), "x1": float(xs.max() + 1),
        "y0": float(ys.min()), "y1": float(ys.max() + 1),
        "cx": float(xs.mean()) / w, "cy": float(ys.mean()) / h,
        "w": float(xs.max() - xs.min() + 1), "h": float(ys.max() - ys.min() + 1),
        "canvas_w": float(w), "canvas_h": float(h),
    }


def _diagnose(ref: np.ndarray, render: np.ndarray) -> Optional[Dict[str, float]]:
    a = _mask_stats(ref)
    b = _mask_stats(render)
    if a is None or b is None:
        return None
    return {
        "width_ratio": a["w"] / max(b["w"], 1.0),     # >1: render too narrow
        "height_ratio": a["h"] / max(b["h"], 1.0),
        "dx_px": (b["cx"] - a["cx"]) * a["canvas_w"],  # render right of ref
        "dy_px": (b["cy"] - a["cy"]) * a["canvas_h"],  # render below ref
        "canvas_w": a["canvas_w"],
        "ref_width_px": a["w"],
    }


# ---------------------------------------------------------------------------
# parameter adjustment helpers
# ---------------------------------------------------------------------------

def _bounded_step(ratio: float) -> float:
    return min(_STEP_LIMIT, max(1.0 / _STEP_LIMIT, ratio))


def _bounded_scale_step(ratio: float) -> float:
    return min(_SCALE_STEP_LIMIT,
               max(1.0 / _SCALE_STEP_LIMIT, ratio))


def _get_global_scale(plan: ConstructionPlan) -> List[float]:
    gs = (plan.metadata or {}).get("global_scale")
    if gs:
        return [float(gs[0]), float(gs[1]), float(gs[2])]
    return [1.0, 1.0, 1.0]


def _set_global_scale(plan: ConstructionPlan, scale: List[float]) -> None:
    md = dict(plan.metadata or {})
    md["global_scale"] = [round(v, 6) for v in scale]
    plan.metadata = md


def _get_validation_camera_offset(plan: ConstructionPlan) -> List[float]:
    offset = (plan.metadata or {}).get("validation_camera_offset")
    if offset and len(offset) >= 2:
        return [float(offset[0]), float(offset[1])]
    return [0.0, 0.0]


def _set_validation_camera_offset(plan: ConstructionPlan,
                                  offset: List[float]) -> None:
    md = dict(plan.metadata or {})
    md["validation_camera_offset"] = [round(float(offset[0]), 6),
                                      round(float(offset[1]), 6)]
    plan.metadata = md


def _ensure_camera(plan: ConstructionPlan, cfg: PipelineConfig) -> CameraEstimate:
    if plan.camera is None:
        cam = CameraEstimate()
        cam.focal_length_px = EvidencedValue(
            value=cfg.camera.default_focal_px, unit="px",
            source=EvidenceSource.ESTIMATED_FROM_CAMERA, confidence=0.3)
        cam.translation = EvidencedValue(
            value=[0.0, 0.0, 2.0], unit="object_units",
            source=EvidenceSource.ESTIMATED_FROM_CAMERA, confidence=0.3)
        plan.camera = cam
    if plan.camera.translation.value is None:
        plan.camera.translation = EvidencedValue(
            value=[0.0, 0.0, 2.0], unit="object_units",
            source=EvidenceSource.ESTIMATED_FROM_CAMERA, confidence=0.3)
    if plan.camera.focal_length_px.value is None:
        plan.camera.focal_length_px = EvidencedValue(
            value=cfg.camera.default_focal_px, unit="px",
            source=EvidenceSource.ESTIMATED_FROM_CAMERA, confidence=0.3)
    return plan.camera


def _set_object_rotation(plan: ConstructionPlan, axis: int, angle: float,
                         cfg: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    cam = _ensure_camera(plan, cfg)
    current = list(cam.object_rotation_euler_deg.value or [0.0, 0.0, 0.0])
    while len(current) < 3:
        current.append(0.0)
    previous = float(current[axis])
    current[axis] = float(angle)
    cam.object_rotation_euler_deg = EvidencedValue(
        value=[round(float(v), 6) for v in current], unit="deg",
        source=EvidenceSource.ESTIMATED_FROM_CAMERA,
        confidence=0.45,
        note="render-validated object pose hypothesis; retained only when it "
             "improves reference silhouette agreement",
    )
    name = "object_rotation_x" if axis == 0 else "object_rotation_y"
    return {name: {"previous": previous, "new": float(angle)}}


def _apply_candidate(plan: ConstructionPlan, name: str, diag: Dict[str, float],
                     focal_px: float, cfg: PipelineConfig
                     ) -> Dict[str, Dict[str, Any]]:
    """Apply one candidate adjustment in place; return the param record."""
    record: Dict[str, Dict[str, Any]] = {}
    if name == "global_scale_x":
        scale = _get_global_scale(plan)
        new = min(_SCALE_BOUNDS[1], max(_SCALE_BOUNDS[0],
                                        scale[0] * _bounded_scale_step(
                                            diag["width_ratio"])))
        record["global_scale_x"] = {"previous": scale[0], "new": round(new, 6)}
        scale[0] = new
        _set_global_scale(plan, scale)
    elif name == "global_scale_y":
        scale = _get_global_scale(plan)
        new = min(_SCALE_BOUNDS[1], max(_SCALE_BOUNDS[0],
                                        scale[1] * _bounded_scale_step(
                                            diag["height_ratio"])))
        record["global_scale_y"] = {"previous": scale[1], "new": round(new, 6)}
        scale[1] = new
        _set_global_scale(plan, scale)
    elif name == "camera_distance":
        cam = _ensure_camera(plan, cfg)
        t = list(cam.translation.value)
        ratio = _bounded_step(1.0 / max(diag["width_ratio"], 1e-6))
        new = min(_DIST_BOUNDS[1], max(_DIST_BOUNDS[0], t[2] * ratio))
        record["camera_distance"] = {"previous": t[2], "new": round(new, 6)}
        t[2] = new
        cam.translation = cam.translation.model_copy(update={"value": t})
    elif name == "camera_offset_x":
        offset = _get_validation_camera_offset(plan)
        # Auto-framing uses distance=focal/reference_width, therefore a
        # world-space camera offset of dx/reference_width corrects dx pixels
        # without replacing the inferred distance or base centering.
        delta = diag["dx_px"] / max(diag.get("ref_width_px", focal_px), 1e-6)
        delta = max(-0.2, min(0.2, delta))
        record["camera_offset_x"] = {"previous": offset[0],
                                     "new": round(offset[0] + delta, 6)}
        offset[0] += delta
        _set_validation_camera_offset(plan, offset)
    elif name == "camera_offset_y":
        offset = _get_validation_camera_offset(plan)
        delta = -diag["dy_px"] / max(
            diag.get("ref_width_px", focal_px), 1e-6)  # y up vs image down
        delta = max(-0.2, min(0.2, delta))
        record["camera_offset_y"] = {"previous": offset[1],
                                     "new": round(offset[1] + delta, 6)}
        offset[1] += delta
        _set_validation_camera_offset(plan, offset)
    elif name == "revolve_radius_scale":
        ratio = _bounded_step(diag["width_ratio"])
        for part in plan.parts:
            if part.operator.value == "revolve" and part.profile:
                pts = part.profile.get("points") or []
                part.profile["points"] = [
                    [round(float(p[0]) * ratio, 6), p[1]] for p in pts]
        record["revolve_radius_scale"] = {"previous": 1.0,
                                          "new": round(ratio, 6)}
    return record


def _rank_candidates(diag: Dict[str, float],
                     plan: Optional[ConstructionPlan] = None
                     ) -> List[Tuple[str, float]]:
    """Order candidate adjustments by diagnosed mismatch magnitude."""
    canvas = diag.get("canvas_w", 1.0)
    scored = [
        ("global_scale_x", abs(math.log(max(diag["width_ratio"], 1e-6)))),
        ("global_scale_y", abs(math.log(max(diag["height_ratio"], 1e-6)))),
        # A centroid displacement contaminates bounding-box scale diagnosis:
        # correct alignment before spending the small render budget on size.
        # Three times the normalized displacement makes a visible 1--2% shift
        # competitive with the corresponding width/height mismatch.
        ("camera_offset_x", 3.0 * abs(diag["dx_px"]) / canvas),
        ("camera_offset_y", 3.0 * abs(diag["dy_px"]) / canvas),
        ("camera_distance",
         abs(math.log(max(diag["width_ratio"], 1e-6))) * 0.5),
        ("revolve_radius_scale",
         abs(math.log(max(diag["width_ratio"], 1e-6))) * 0.25),
    ]
    if plan is not None and not any(
            part.operator.value == "revolve" for part in plan.parts):
        scored = [item for item in scored
                  if item[0] != "revolve_radius_scale"]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def _pose_search_eligible(plan: ConstructionPlan, silhouette_iou: float) -> bool:
    """Return whether refinement may replace the current object pose.

    Explicit calibration is an input constraint, not a hypothesis for the
    single-view silhouette loop to overwrite. The loop may still tune scale,
    framing, and inferred geometry around that fixed pose.
    """
    pose = (plan.camera.object_rotation_euler_deg
            if plan.camera is not None else None)
    if pose is not None and pose.source == EvidenceSource.USER_SUPPLIED:
        return False
    op_names = [part.operator.value for part in plan.parts]
    return (silhouette_iou < 0.60
            and op_names.count("extrude") >= op_names.count("revolve"))


# ---------------------------------------------------------------------------
# artifact snapshot / restore (best-model rollback)
# ---------------------------------------------------------------------------

_SNAPSHOT_FILES = [
    ("blender", "scene.blend"),
    ("blender", "model.glb"),
    ("blender", "blender_manifest.json"),
    ("blender", "build_model.py"),
    ("validation", "render_silhouette.png"),
    ("validation", "render_shaded.png"),
    ("validation", "render_clay.png"),
    ("validation", "render_depth.png"),
    ("validation", "render_normal.png"),
    ("validation", "render_partid.png"),
    ("validation", "reference_overlay.png"),
    ("validation", "silhouette_comparison.png"),
    ("validation", "depth_comparison.png"),
    ("validation", "turntable.mp4"),
    ("validation", "metrics.json"),
]


def _snapshot(project: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for sub, name in _SNAPSHOT_FILES:
        src = project / sub / name
        if src.exists():
            tgt = dest / sub / name
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(tgt))


def _restore(project: Path, src_dir: Path) -> None:
    for sub, name in _SNAPSHOT_FILES:
        src = src_dir / sub / name
        if src.exists():
            tgt = project / sub / name
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(tgt))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def _focal_px(plan: ConstructionPlan, cfg: PipelineConfig) -> float:
    if plan.camera is not None and plan.camera.focal_length_px.value:
        return float(plan.camera.focal_length_px.value)
    return cfg.camera.default_focal_px


def refine(plan: ConstructionPlan, seg: SegmentationResult,
           crop_meta: CropMetadata, project_dir: str, cfg: PipelineConfig
           ) -> Tuple[ConstructionPlan, BlenderManifest, ValidationResult,
                      RefinementLog]:
    project = Path(project_dir).resolve()
    (project / "blender").mkdir(parents=True, exist_ok=True)
    (project / "validation").mkdir(parents=True, exist_ok=True)
    log = RefinementLog()
    rc = cfg.refinement

    # reference mask in crop space (for diagnosis)
    mask_img = cv2.imread(seg.mask_path, cv2.IMREAD_GRAYSCALE)
    ref_mask = validation._binarize(
        validation.remap_to_crop(mask_img, crop_meta))

    renders_used = 0

    def build_and_validate(p: ConstructionPlan):
        script = blender_codegen.generate_blender_script(
            p, str(project / "blender"), cfg)
        man = runner.run_blender(script, str(project), cfg)
        if not man.success:
            return man, None
        val = validation.validate_reconstruction(man, p, seg, crop_meta,
                                                 str(project), cfg)
        return man, val

    # ---- initial build --------------------------------------------------
    best_plan = plan.model_copy(deep=True)
    try:
        best_manifest, best_val = build_and_validate(best_plan)
    except runner.ScriptSafetyError as exc:
        man = BlenderManifest(
            blend_path=str(project / "blender" / "scene.blend"),
            script_path=str(project / "blender" / "build_model.py"),
            success=False, errors=["safety scan rejected script: %s" % exc])
        log.iterations = 0
        SchemaIO.save_json(log, project / "validation" / "refinement_log.json")
        return best_plan, man, ValidationResult(
            passed=False, notes=["safety scan rejected generated script"]), log
    renders_used += 1

    if best_val is None:
        # Blender failed on the initial plan: nothing to refine against.
        log.iterations = 0
        SchemaIO.save_json(log, project / "validation" / "refinement_log.json")
        return best_plan, best_manifest, ValidationResult(
            passed=False, notes=["initial blender build failed"]), log

    best_iou = best_val.metrics.silhouette_iou or 0.0
    log.initial_metrics = {"silhouette_iou": best_iou}
    snap_dir = project / "refinement" / "best"
    _snapshot(project, snap_dir)

    tried: set = set()
    iterations = 0
    stop_reason = "max_iterations"

    # Flat extruded objects often lack circular features, leaving the camera
    # stage unable to infer their out-of-plane pose. Before size/translation
    # tuning, render a bounded set of explicit tilt/spin hypotheses. Every
    # rejected pose rolls back and the initial model can never be made worse.
    pose_eligible = _pose_search_eligible(best_plan, best_iou)
    if pose_eligible:
        angles = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0]
        for axis, name in ((0, "object_rotation_x"),
                           (1, "object_rotation_y")):
            for angle in angles:
                if renders_used >= rc.max_renders:
                    break
                current_rot = (_ensure_camera(best_plan, cfg)
                               .object_rotation_euler_deg.value
                               or [0.0, 0.0, 0.0])
                if abs(float(current_rot[axis]) - angle) < 1e-9:
                    continue
                cand_plan = best_plan.model_copy(deep=True)
                record = _set_object_rotation(cand_plan, axis, angle, cfg)
                prev_iou = best_iou
                iterations += 1
                try:
                    cand_manifest, cand_val = build_and_validate(cand_plan)
                except runner.ScriptSafetyError as exc:
                    log.actions.append(RefinementAction(
                        iteration=iterations,
                        observed_problem="low silhouette agreement; testing %s" % name,
                        modified_parameters=record,
                        metric_change={"silhouette_iou": {
                            "previous": prev_iou, "new": None}}, kept=False))
                    stop_reason = "safety_error: %s" % exc
                    break
                renders_used += 1
                new_iou = (cand_val.metrics.silhouette_iou or 0.0
                           if cand_val is not None else 0.0)
                kept = bool(cand_val is not None and cand_manifest.success
                            and new_iou > best_iou + 1e-9)
                log.actions.append(RefinementAction(
                    iteration=iterations,
                    observed_problem="low silhouette agreement; testing %s" % name,
                    modified_parameters=record,
                    metric_change={"silhouette_iou": {
                        "previous": prev_iou,
                        "new": new_iou if cand_val is not None else None}},
                    kept=kept))
                if kept:
                    best_plan, best_manifest, best_val = (
                        cand_plan, cand_manifest, cand_val)
                    best_iou = new_iou
                    _snapshot(project, snap_dir)
            if renders_used >= rc.max_renders:
                break
        # The last trial may have been rejected; restore best render artifacts
        # before the ordinary diagnostic loop reads them.
        _restore(project, snap_dir)

    for it in range(1, rc.max_iterations + 1):
        if best_iou >= rc.target_silhouette_iou:
            stop_reason = "target_reached"
            break
        if renders_used >= rc.max_renders:
            stop_reason = "render_budget_exhausted"
            break

        sil_path = project / "validation" / "render_silhouette.png"
        render_sil = None
        if sil_path.exists():
            render_sil = validation._binarize(
                cv2.imread(str(sil_path), cv2.IMREAD_GRAYSCALE))
        diag = _diagnose(ref_mask, render_sil) if render_sil is not None else None
        if diag is None:
            stop_reason = "no_render_silhouette"
            break

        # pick the largest untried mismatch
        candidates = [c for c, _ in _rank_candidates(diag, best_plan)
                      if c not in tried]
        if not candidates:
            stop_reason = "no_candidates_left"
            break
        name = candidates[0]
        tried.add(name)

        cand_plan = best_plan.model_copy(deep=True)
        record = _apply_candidate(cand_plan, name, diag, _focal_px(cand_plan, cfg),
                                  cfg)
        problem = _describe_problem(name, diag)

        prev_iou = best_iou
        iterations += 1
        try:
            cand_manifest, cand_val = build_and_validate(cand_plan)
        except runner.ScriptSafetyError as exc:
            log.actions.append(RefinementAction(
                iteration=iterations, observed_problem=problem,
                modified_parameters=record,
                metric_change={"silhouette_iou": {"previous": prev_iou,
                                                  "new": None}},
                kept=False))
            stop_reason = "safety_error: %s" % exc
            break
        renders_used += 1

        if cand_val is None:
            # blender failed on the tweaked plan: roll back, try another knob
            log.actions.append(RefinementAction(
                iteration=iterations,
                observed_problem=problem + " (blender failed, rolled back)",
                modified_parameters=record,
                metric_change={"silhouette_iou": {"previous": prev_iou,
                                                  "new": None}},
                kept=False))
            continue

        new_iou = cand_val.metrics.silhouette_iou or 0.0
        kept = new_iou > prev_iou + 1e-9 and cand_manifest.success
        log.actions.append(RefinementAction(
            iteration=iterations,
            observed_problem=problem,
            modified_parameters=record,
            metric_change={"silhouette_iou": {"previous": prev_iou,
                                              "new": new_iou}},
            kept=kept))
        if kept:
            best_plan = cand_plan
            best_manifest = cand_manifest
            best_val = cand_val
            gain = new_iou - best_iou
            best_iou = new_iou
            _snapshot(project, snap_dir)
            # A small gain from one coordinate does not imply that independent
            # coordinates are exhausted.  Keep the bounded search moving; the
            # tried set and hard iteration/render caps provide termination.
            if gain < rc.min_iou_gain:
                stop_reason = "small_gain_continuing_coordinate_search"
        # not kept: best_plan/best_manifest/best_val untouched (rollback)

    log.iterations = iterations
    log.final_metrics = {"silhouette_iou": best_iou}
    log.converged = best_iou >= rc.target_silhouette_iou
    if log.converged and stop_reason == "max_iterations":
        stop_reason = "target_reached"

    # make sure on-disk artifacts match the best plan (rollback of files)
    manifest_path = project / "blender" / "blender_manifest.json"
    current_iou = None
    sil_path = project / "validation" / "render_silhouette.png"
    if sil_path.exists():
        cur = validation._binarize(cv2.imread(str(sil_path),
                                              cv2.IMREAD_GRAYSCALE))
        current_iou = validation._iou(ref_mask, cur)
    if current_iou is None or current_iou < best_iou - 1e-9:
        _restore(project, snap_dir)

    log.final_metrics = {"silhouette_iou": best_iou}
    SchemaIO.save_json(log, project / "validation" / "refinement_log.json")
    # stop reason recorded in an auxiliary field of the log file
    log_path = project / "validation" / "refinement_log.json"
    data = json.loads(log_path.read_text())
    data["stop_reason"] = stop_reason
    log_path.write_text(json.dumps(data, indent=2))
    return best_plan, best_manifest, best_val, log


def _describe_problem(name: str, diag: Dict[str, float]) -> str:
    if name == "global_scale_x":
        ratio = 1.0 / max(diag["width_ratio"], 1e-6)
        return ("silhouette too narrow (render/reference width %.3f)" % ratio
                if diag["width_ratio"] > 1 else
                "silhouette too wide (render/reference width %.3f)" % ratio)
    if name == "global_scale_y":
        return ("silhouette too short" if diag["height_ratio"] > 1
                else "silhouette too tall")
    if name == "camera_offset_x":
        return ("silhouette off-centre horizontally by %.1f px"
                % diag["dx_px"])
    if name == "camera_offset_y":
        return ("silhouette off-centre vertically by %.1f px"
                % diag["dy_px"])
    if name == "camera_distance":
        return "overall framing scale mismatch"
    return "revolve profile radius mismatch"
