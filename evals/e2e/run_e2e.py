"""Level C end-to-end evaluation runner (EVAL.md).

For every benchmark case this:
1. runs the pipeline CLI on the case's input.png (mask-guided, plus an
   un-guided run for easy cases to score real segmentation),
2. loads the produced project directory and scores every stage against the
   ground truth in the benchmark case directory,
3. writes a per-case report YAML in EVAL.md's per-case format,
4. runs the SVG-extrusion baseline and the no-refinement ablation,
5. aggregates everything into dashboard.json + dashboard.md and checks the
   MVP hard gates.

Robust by design: any individual case failure is recorded and the run
continues.

CLI:
    python -m evals.e2e.run_e2e --cases all \
        --dataset evals/benchmark/dataset --out evals/results/ --workers 1

    # smoke test without the pipeline / blender:
    python -m evals.e2e.run_e2e --cases fake_01 --dataset <ds> \
        --projects-root <projs> --skip-pipeline --skip-blender
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from evals import metrics as M
from evals.e2e.glbcheck import validate_glb

DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

# EVAL.md "Example Weighted Score"
SCORE_WEIGHTS = {
    "segmentation": 0.08,
    "vectorization": 0.07,
    "primitive_fitting": 0.08,
    "constraints": 0.05,
    "semantic_parts": 0.10,
    "camera": 0.07,
    "construction_plan": 0.08,
    "blender_execution": 0.07,
    "editability": 0.12,
    "silhouette": 0.12,
    "internal_features": 0.06,
    "multiview_geometry": 0.05,
    "materials": 0.02,
    "uncertainty": 0.03,
}

MVP_GATES = (
    "target_selection_correct",
    "segmentation_iou",
    "construction_plan_valid",
    "blender_execution_success",
    "blend_reopens",
    "glb_valid",
    "reference_silhouette_iou",
    "major_visible_part_recall",
    "safety_violations",
)


# ---------------------------------------------------------------------------
# case + project loading
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _read_mask(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def load_case(dataset_dir: Path, case_id: str) -> Dict[str, Any]:
    d = dataset_dir / case_id
    return {
        "case_id": case_id,
        "dir": d,
        "input_png": d / "input.png",
        "mask": _read_mask(d / "mask.png"),
        "meta": _read_json(d / "meta.json") or {},
        "parts": _read_json(d / "parts.json") or {},
        "camera": _read_json(d / "camera.json") or {},
        "bbox": _read_json(d / "bbox.json") or {},
        "dimensions": _read_json(d / "dimensions.json") or {},
    }


def list_cases(dataset_dir: Path) -> List[str]:
    return sorted(p.name for p in dataset_dir.iterdir()
                  if p.is_dir() and (p / "meta.json").exists())


# ---------------------------------------------------------------------------
# pipeline invocation
# ---------------------------------------------------------------------------

def run_pipeline(case: Dict[str, Any], project_dir: Path, use_mask: bool,
                 python: str, timeout: int = 1800) -> Dict[str, Any]:
    """Invoke the recon3d pipeline CLI; never raises."""
    cmd = [python, "-m", "recon3d.pipeline",
           "--image", str(case["input_png"]), "--out", str(project_dir)]
    if use_mask and case["mask"] is not None:
        cmd += ["--mask", str(case["dir"] / "mask.png")]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        (project_dir / "pipeline_stdout.log").write_text(
            (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""))
        return {"invoked": True, "returncode": proc.returncode,
                "seconds": round(time.time() - started, 1)}
    except subprocess.TimeoutExpired:
        return {"invoked": True, "returncode": None, "error": "timeout",
                "seconds": timeout}
    except Exception as exc:  # noqa: BLE001
        return {"invoked": True, "returncode": None, "error": repr(exc),
                "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------------------
# safety scan of the generated Blender script (EVAL.md Eval 15)
# ---------------------------------------------------------------------------

_BANNED_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
    ("os", "rmdir"), ("os", "removedirs"),
    ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "Popen"),
    ("subprocess", "check_output"),
    ("shutil", "rmtree"),
    ("socket", "create_connection"), ("socket", "socket"),
}
_BANNED_MODULES = {"subprocess", "socket", "urllib", "urllib.request",
                   "requests", "http", "ftplib", "telnetlib", "shutil"}
_BANNED_BUILTINS = {"eval", "exec", "__import__", "compile"}


def safety_scan(script_path: Path, project_dir: Path) -> List[str]:
    """AST scan of the generated build script for sandbox violations."""
    violations: List[str] = []
    try:
        tree = ast.parse(script_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return ["script unparseable: %r" % (exc,)]

    def _dotted(node: ast.AST) -> Optional[Tuple[str, str]]:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return (node.value.id, node.attr)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BANNED_MODULES:
                    violations.append("imports banned module '%s'" % alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BANNED_MODULES:
                violations.append("imports from banned module '%s'" % node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_BUILTINS:
                violations.append("calls banned builtin '%s'" % node.func.id)
            dotted = _dotted(node.func)
            if dotted and dotted in _BANNED_CALLS:
                violations.append("calls %s.%s" % dotted)
            # open() with an absolute path outside the project dir
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                for arg in node.args[:1]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        p = arg.value
                        if p.startswith("/") and not p.startswith(str(project_dir)):
                            violations.append("open() outside project dir: %s" % p)
    return sorted(set(violations))


# ---------------------------------------------------------------------------
# blender reopen probe
# ---------------------------------------------------------------------------

def blend_reopen_probe(blender: str, blend_path: Path,
                       timeout: int = 120) -> Dict[str, Any]:
    """Reopen a .blend in background Blender and report scene stats."""
    expr = (
        "import bpy; bpy.ops.wm.open_mainfile(filepath=%r); "
        "names=[o.name for o in bpy.data.objects]; "
        "print('REOPEN_OK', len(names), len(bpy.data.collections), "
        "len(bpy.data.materials), sorted(set(names)))"
    ) % str(blend_path)
    try:
        proc = subprocess.run(
            [blender, "--background", "--factory-startup", "--python-expr", expr],
            capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"reopens": False, "error": repr(exc)}
    m = re.search(r"REOPEN_OK (\d+) (\d+) (\d+) (\[.*\])", proc.stdout or "")
    if not m:
        return {"reopens": False,
                "error": (proc.stderr or proc.stdout or "")[-400:]}
    names = ast.literal_eval(m.group(4))
    defaultish = [n for n in names
                  if n.split(".")[0] in ("Cube", "Sphere", "Cylinder", "Plane",
                                         "Cone", "Torus", "Mesh")]
    return {"reopens": True,
            "object_count": int(m.group(1)),
            "collection_count": int(m.group(2)),
            "material_count": int(m.group(3)),
            "meaningful_name_rate":
                (len(names) - len(defaultish)) / max(1, len(names)),
            "part_separation": int(m.group(1)) >= 2}


# ---------------------------------------------------------------------------
# svg silhouette rasterisation (vectorization score)
# ---------------------------------------------------------------------------

def rasterize_svg_silhouette(svg_path: Path, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Best-effort rasterisation of an SVG's filled outline polygons.

    Self-contained mini-parser (absolute M/L/C/Q/Z commands, curves flattened
    coarsely). Returns None on any failure - the score is then skipped.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(svg_path)).getroot()
        polys: List[np.ndarray] = []
        holes: List[np.ndarray] = []
        for el in root.iter():
            if not el.tag.endswith("path"):
                continue
            d = el.attrib.get("d", "")
            pts = _svg_path_points(d)
            if pts is None or len(pts) < 3:
                continue
            fill = el.attrib.get("fill", "black").strip().lower()
            (holes if fill in ("#ffffff", "white") else polys).append(pts)
        if not polys:
            return None
        # normalise: svg viewport may be pixels or 0..1
        allpts = np.concatenate(polys + holes)
        vmax = float(np.abs(allpts).max())
        scale = 1.0 if vmax <= 1.5 else max(size)
        return M.rasterize_paths([p / scale for p in polys], size,
                                 holes=[p / scale for p in holes] or None)
    except Exception:  # noqa: BLE001
        return None


def _svg_path_points(d: str, curve_steps: int = 8) -> Optional[np.ndarray]:
    tokens = re.findall(r"[MLCQZmlcqz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    pts: List[List[float]] = []
    i = 0
    cur = (0.0, 0.0)
    cmd = ""
    while i < len(tokens):
        if re.fullmatch(r"[MLCQZmlcqz]", tokens[i]):
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                continue
        try:
            if cmd in ("M", "L"):
                cur = (float(tokens[i]), float(tokens[i + 1]))
                pts.append(list(cur))
                i += 2
            elif cmd == "C":
                p0 = np.array(cur)
                c1 = np.array([float(tokens[i]), float(tokens[i + 1])])
                c2 = np.array([float(tokens[i + 2]), float(tokens[i + 3])])
                p1 = np.array([float(tokens[i + 4]), float(tokens[i + 5])])
                for t in np.linspace(0, 1, curve_steps)[1:]:
                    q = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1
                         + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1)
                    pts.append([float(q[0]), float(q[1])])
                cur = tuple(p1)
                i += 6
            elif cmd == "Q":
                p0 = np.array(cur)
                c1 = np.array([float(tokens[i]), float(tokens[i + 1])])
                p1 = np.array([float(tokens[i + 2]), float(tokens[i + 3])])
                for t in np.linspace(0, 1, curve_steps)[1:]:
                    q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c1 + t ** 2 * p1
                    pts.append([float(q[0]), float(q[1])])
                cur = tuple(p1)
                i += 4
            else:  # relative / unsupported command
                return None
        except (IndexError, ValueError):
            return None
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# part-name matching
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _labels_match(gt_label: str, pred_name: str) -> bool:
    gt, pr = _tokens(gt_label), _tokens(pred_name)
    if not gt or not pr:
        return False
    if set(gt) & set(pr):
        return True
    return any(g in p or p in g for g in gt for p in pr)


def major_part_recall(gt_parts: Dict[str, Any],
                      graph_parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Recall of GT major parts among predicted sketch-graph parts."""
    major = [p for p in gt_parts.get("parts", []) if p.get("major")]
    pred_names = [str(p.get("id", "")) + " " + str(p.get("part_class", ""))
                  for p in graph_parts]
    found, missing = [], []
    for p in major:
        label = str(p.get("label", p.get("id", "")))
        if any(_labels_match(label, pn) or _labels_match(str(p.get("id", "")), pn)
               for pn in pred_names):
            found.append(p.get("id"))
        else:
            missing.append(p.get("id"))
    recall = len(found) / max(1, len(major))
    return {"expected_major_parts": len(major),
            "detected_major_parts": len(found),
            "missing": missing,
            "recall": recall}


# ---------------------------------------------------------------------------
# per-stage scorers (all tolerant of missing artifacts)
# ---------------------------------------------------------------------------

def score_segmentation(project_dir: Path, gt_mask: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pred = _read_mask(project_dir / "segmentation" / "object_mask.png")
    if pred is None or gt_mask is None:
        out["available"] = False
        return out
    if pred.shape != gt_mask.shape:
        pred = cv2.resize(pred, (gt_mask.shape[1], gt_mask.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    out.update({
        "available": True,
        "mask_iou": M.mask_iou(pred, gt_mask),
        "boundary_f_score": M.boundary_f_score(pred, gt_mask, tolerance_px=3.0),
        "gt_holes": M.hole_count(gt_mask),
        "pred_holes": M.hole_count(pred),
        "hole_recall": (1.0 if M.hole_count(gt_mask) == 0 else
                        min(1.0, M.hole_count(pred) / M.hole_count(gt_mask))),
    })
    out["passed"] = out["mask_iou"] >= 0.85
    return out


def score_crop(project_dir: Path, gt_mask: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    meta_path = project_dir / "segmentation" / "crop_metadata.json"
    data = _read_json(meta_path)
    if data is None or gt_mask is None:
        return out
    try:
        from recon3d.schemas import CropMetadata
        meta = CropMetadata.model_validate(data)
    except Exception:  # noqa: BLE001
        return out
    ys, xs = np.where(gt_mask > 0)
    if len(xs) == 0:
        return out
    rng = np.random.RandomState(0)
    idx = rng.choice(len(xs), size=min(300, len(xs)), replace=False)
    pts = np.stack([xs[idx].astype(float), ys[idx].astype(float)], axis=1)
    errs = M.round_trip_errors(pts, meta.to_crop, meta.to_original)
    out.update({
        "available": True,
        "mean_round_trip_error_px": float(errs.mean()),
        "max_round_trip_error_px": float(errs.max()),
        "passed": bool(errs.mean() < 0.1 and errs.max() < 0.5),
    })
    return out


def score_vectorization(project_dir: Path, gt_mask: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    svg = project_dir / "traces" / "silhouette.svg"
    if not svg.exists() or gt_mask is None:
        return out
    mask = rasterize_svg_silhouette(svg, (gt_mask.shape[1], gt_mask.shape[0]))
    if mask is None:
        return out
    out.update({
        "available": True,
        "silhouette_svg_iou": M.mask_iou(mask, gt_mask),
        "boundary_f_score": M.boundary_f_score(mask, gt_mask, tolerance_px=3.0),
    })
    out["passed"] = out["silhouette_svg_iou"] >= 0.90
    return out


def score_primitives(project_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    data = _read_json(project_dir / "geometry" / "fitted_primitives.json")
    if not data:
        return out
    prims = data.get("primitives", [])
    if not prims:
        return out
    type_counts: Dict[str, int] = {}
    raw_points = 0
    fit_errors = []
    for p in prims:
        type_counts[p.get("type", "?")] = type_counts.get(p.get("type", "?"), 0) + 1
        raw_points += len(p.get("fallback_points", []))
        fit_errors.append(float(p.get("fit_error", 0.0)))
    n = len(prims)
    reduction = (1.0 - n / raw_points) if raw_points > n else None
    good = sum(1 for e in fit_errors if e <= 0.01) / n
    out.update({
        "available": True,
        "detected": type_counts,
        "primitive_count": n,
        "control_points_raw": raw_points,
        "control_point_reduction": reduction,
        "mean_fit_error": float(np.mean(fit_errors)),
        "good_fit_rate": good,
        "primitive_accuracy": good,
        "passed": good >= 0.8,
    })
    return out


def score_parts(project_dir: Path, gt_parts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    data = _read_json(project_dir / "geometry" / "sketch_graph.json")
    if not data:
        return out
    parts = data.get("parts", [])
    rec = major_part_recall(gt_parts, parts)
    out.update({
        "available": True,
        "expected_major_parts": rec["expected_major_parts"],
        "detected_major_parts": rec["detected_major_parts"],
        "missing_major_parts": rec["missing"],
        "major_part_recall": rec["recall"],
        "constraint_count": len(data.get("constraints", [])),
        "uncertainty": data.get("uncertainty", {}),
        "passed": rec["recall"] >= 0.85,
    })
    return out


def score_plan(project_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "valid": None}
    plan_path = project_dir / "geometry" / "construction_plan.yaml"
    if not plan_path.exists():
        return out
    try:
        from recon3d.construction_plan import validate_plan
        from recon3d.schemas import ConstructionPlan
        plan = ConstructionPlan.model_validate(
            yaml.safe_load(plan_path.read_text()))
        errors = validate_plan(plan)
        cam = plan.camera
        out.update({
            "available": True,
            "valid": not errors,
            "errors": errors,
            "part_count": len(plan.parts),
            "focal_length_px": (cam.focal_length_px.value
                                if cam and cam.focal_length_px else None),
        })
    except Exception as exc:  # noqa: BLE001
        out.update({"available": True, "valid": False,
                    "errors": ["plan unparseable: %r" % (exc,)]})
    return out


def score_blender(project_dir: Path, blender: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    bdir = project_dir / "blender"
    blend = bdir / "scene.blend"
    glb = bdir / "model.glb"
    script = bdir / "build_model.py"
    manifest = _read_json(project_dir / "manifest.json") or {}
    status = manifest.get("status")
    out["pipeline_status"] = status
    out["blend_exists"] = blend.exists()
    out["execution_success"] = bool(blend.exists()) and status in (
        "success", "partial_success")
    if glb.exists():
        out["glb"] = validate_glb(str(glb))
    else:
        out["glb"] = {"valid": None, "errors": ["model.glb missing"]}
    if script.exists():
        out["safety_violations"] = safety_scan(script, project_dir)
    else:
        out["safety_violations"] = ["build_model.py missing"] if not blend.exists() else []
    if blender and blend.exists():
        out["reopen"] = blend_reopen_probe(blender, blend)
    else:
        out["reopen"] = None
    out["available"] = True
    return out


def score_visual(project_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    metrics = _read_json(project_dir / "validation" / "metrics.json")
    rlog = _read_json(project_dir / "validation" / "refinement_log.json")
    if metrics:
        m = metrics.get("metrics", metrics)
        out.update({
            "available": True,
            "silhouette_iou_final": m.get("silhouette_iou"),
            "contour_chamfer_distance": m.get("contour_chamfer_distance"),
            "depth_correlation": m.get("depth_correlation"),
            "part_mask_iou": m.get("part_mask_iou"),
            "perceptual_similarity": m.get("perceptual_similarity"),
            "clay_silhouette_iou": m.get("clay_silhouette_iou"),
        })
    if rlog:
        init = rlog.get("initial_metrics", {}) or {}
        fin = rlog.get("final_metrics", {}) or {}
        out["silhouette_iou_initial"] = init.get("silhouette_iou")
        if init.get("silhouette_iou") is not None and fin.get("silhouette_iou") is not None:
            out["refinement_iou_gain"] = fin["silhouette_iou"] - init["silhouette_iou"]
        out["refinement_iterations"] = rlog.get("iterations")
    return out


# ---------------------------------------------------------------------------
# baselines + ablations
# ---------------------------------------------------------------------------

def svg_extrusion_baseline(case: Dict[str, Any], out_dir: Path,
                           blender: Optional[str]) -> Dict[str, Any]:
    """Baseline 1: trace the GT silhouette -> flat extrusion -> score.

    With --skip-blender the reference-view silhouette of a camera-facing
    extrusion is exactly the input mask, so IoU is recorded as 1.0 with
    ``approximate: True``; the baseline still scores zero on part recall and
    editability, which is where the real pipeline must beat it.
    """
    result: Dict[str, Any] = {"baseline": "svg_extrusion"}
    gt_mask = case["mask"]
    if gt_mask is None:
        result["error"] = "no gt mask"
        return result
    contour = M.largest_contour(gt_mask)
    result["silhouette_contour_points"] = (0 if contour is None
                                           else int(len(contour)))
    if blender:
        try:
            from evals.e2e.baseline_extrusion import render_extrusion_mask
            rendered = render_extrusion_mask(case, out_dir, blender)
            if rendered is not None:
                result["reference_silhouette_iou"] = M.mask_iou(rendered, gt_mask)
                result["contour_error"] = M.chamfer_distance_masks(
                    rendered, gt_mask, normalize_by="diagonal")
                result["approximate"] = False
        except Exception as exc:  # noqa: BLE001
            result["error"] = "baseline render failed: %r" % (exc,)
    if "reference_silhouette_iou" not in result:
        result["reference_silhouette_iou"] = 1.0
        result["approximate"] = True
    result["major_visible_part_recall"] = 0.0   # one fused slab, no parts
    result["meaningful_editability"] = 0.0
    result["construction_plan_valid"] = None
    return result


def ablation_no_refinement(project_dir: Path) -> Dict[str, Any]:
    """Ablation: metrics before the refinement loop (initial vs final)."""
    result: Dict[str, Any] = {"ablation": "no_refinement", "available": False}
    rlog = _read_json(project_dir / "validation" / "refinement_log.json")
    if not rlog:
        return result
    init = rlog.get("initial_metrics", {}) or {}
    fin = rlog.get("final_metrics", {}) or {}
    result["available"] = True
    result["silhouette_iou_initial"] = init.get("silhouette_iou")
    result["silhouette_iou_final"] = fin.get("silhouette_iou")
    if init.get("silhouette_iou") is not None and fin.get("silhouette_iou") is not None:
        result["refinement_gain"] = fin["silhouette_iou"] - init["silhouette_iou"]
    result["iterations"] = rlog.get("iterations")
    return result


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _gate(value: Optional[bool]) -> Optional[bool]:
    return None if value is None else bool(value)


def build_report(case: Dict[str, Any], project_dir: Path,
                 nomask_project_dir: Optional[Path],
                 blender: Optional[str]) -> Dict[str, Any]:
    case_id = case["case_id"]
    gt_mask = case["mask"]
    meta = case["meta"]

    seg = score_segmentation(project_dir, gt_mask)
    nomask_seg = (score_segmentation(nomask_project_dir, gt_mask)
                  if nomask_project_dir and nomask_project_dir.exists()
                  else {"available": False})
    crop = score_crop(project_dir, gt_mask)
    vec = score_vectorization(project_dir, gt_mask)
    prim = score_primitives(project_dir)
    parts = score_parts(project_dir, case["parts"])
    plan = score_plan(project_dir)
    blend = score_blender(project_dir, blender)
    visual = score_visual(project_dir)

    # --- hard gates ---
    seg_iou_for_gates = (nomask_seg.get("mask_iou")
                         if nomask_seg.get("available")
                         else seg.get("mask_iou"))
    gates: Dict[str, Optional[bool]] = {
        "target_selection_correct":
            _gate(None if seg_iou_for_gates is None
                  else seg_iou_for_gates >= 0.5),
        "segmentation_iou":
            (_gate(nomask_seg["mask_iou"] >= 0.85)
             if nomask_seg.get("available") else None),
        "construction_plan_valid": _gate(plan.get("valid")),
        "blender_execution_success": _gate(blend.get("execution_success")),
        "blend_reopens": (_gate(blend["reopen"]["reopens"])
                          if blend.get("reopen") else None),
        "glb_valid": _gate(blend["glb"].get("valid")),
        "reference_silhouette_iou":
            _gate(None if visual.get("silhouette_iou_final") is None
                  else visual["silhouette_iou_final"] >= 0.80),
        "major_visible_part_recall":
            _gate(None if not parts.get("available")
                  else parts["major_part_recall"] >= 0.85),
        "safety_violations":
            _gate(blend.get("safety_violations") is not None
                  and len(blend["safety_violations"]) == 0),
    }
    blocking = [name for name, ok in gates.items() if ok is False]
    incomplete = any(ok is None for ok in gates.values())

    # --- weighted overall score (renormalised over available groups) ---
    gt_cam_focal = (case["camera"] or {}).get("focal_length_px")
    focal_score = None
    if plan.get("focal_length_px") and gt_cam_focal:
        rel = abs(plan["focal_length_px"] - gt_cam_focal) / gt_cam_focal
        focal_score = max(0.0, 1.0 - rel / 0.15)
    reopen = blend.get("reopen") or {}
    editability_score = None
    if reopen.get("reopens"):
        editability_score = (0.6 * float(reopen.get("part_separation", False))
                             + 0.4 * reopen.get("meaningful_name_rate", 0.0))
    group_scores: Dict[str, Optional[float]] = {
        "segmentation": seg.get("mask_iou"),
        "vectorization": vec.get("silhouette_svg_iou"),
        "primitive_fitting": prim.get("primitive_accuracy"),
        "constraints": (1.0 if parts.get("constraint_count") else None),
        "semantic_parts": parts.get("major_part_recall"),
        "camera": focal_score,
        "construction_plan": (1.0 if plan.get("valid") else
                              (0.0 if plan.get("valid") is False else None)),
        "blender_execution": (1.0 if blend.get("execution_success") else 0.0),
        "editability": editability_score,
        "silhouette": visual.get("silhouette_iou_final"),
        "internal_features": visual.get("part_mask_iou"),
        "multiview_geometry": None,
        "materials": visual.get("perceptual_similarity"),
        "uncertainty": (1.0 if parts.get("uncertainty") else None),
    }
    avail = {k: v for k, v in group_scores.items() if v is not None}
    wsum = sum(SCORE_WEIGHTS[k] for k in avail)
    overall = (sum(SCORE_WEIGHTS[k] * v for k, v in avail.items()) / wsum
               if wsum > 0 else 0.0)

    manifest = _read_json(project_dir / "manifest.json") or {}
    report = {
        "case_id": case_id,
        "status": manifest.get("status", "failed_validation"),
        "input": {
            "difficulty": meta.get("difficulty", "unknown"),
            "tags": meta.get("tags", []),
            "source_views": 1,
            "known_scale": False,
        },
        "stage_results": {
            "segmentation": seg,
            "segmentation_unguided": nomask_seg,
            "crop": crop,
            "vectorization": vec,
            "primitive_fitting": prim,
            "semantic_parts": parts,
            "construction_plan": plan,
            "blender": {k: v for k, v in blend.items() if k != "reopen"},
            "blender_reopen": reopen or None,
            "visual": visual,
        },
        "uncertainty": parts.get("uncertainty", {}),
        "group_scores": group_scores,
        "hard_gates": gates,
        "blocking_failures": blocking,
        "gates_incomplete": incomplete,
        "final_result": {
            # every evaluated gate passes; unknown gates are reported via
            # gates_incomplete but do not fail the case
            "passed_mvp": len(blocking) == 0,
            "overall_score": round(overall, 4),
        },
    }
    report["final_result"]["gates_evaluated"] = sum(
        1 for v in gates.values() if v is not None)
    report["final_result"]["gates_total"] = len(gates)
    return report


def evaluate_case(case: Dict[str, Any], projects_root: Path, out_dir: Path,
                  python: str, blender: Optional[str], skip_pipeline: bool,
                  timeout: int) -> Dict[str, Any]:
    case_id = case["case_id"]
    project_dir = projects_root / case_id
    nomask_dir = projects_root / (case_id + "_nomask")
    try:
        if not skip_pipeline:
            project_dir.mkdir(parents=True, exist_ok=True)
            run_pipeline(case, project_dir, use_mask=True,
                         python=python, timeout=timeout)
            if case["meta"].get("difficulty") == "easy":
                nomask_dir.mkdir(parents=True, exist_ok=True)
                run_pipeline(case, nomask_dir, use_mask=False,
                             python=python, timeout=timeout)
        report = build_report(case, project_dir, nomask_dir, blender)
        report["baseline_svg_extrusion"] = svg_extrusion_baseline(
            case, out_dir / "baseline" / case_id, blender)
        report["ablation_no_refinement"] = ablation_no_refinement(project_dir)
        report["evaluation_error"] = None
    except Exception as exc:  # noqa: BLE001
        report = {"case_id": case_id, "status": "evaluation_error",
                  "evaluation_error": repr(exc),
                  "blocking_failures": ["evaluation_error"],
                  "final_result": {"passed_mvp": False, "overall_score": 0.0}}
    rp = out_dir / "cases"
    rp.mkdir(parents=True, exist_ok=True)
    (rp / ("%s_report.yaml" % case_id)).write_text(
        yaml.safe_dump(_jsonable(report), sort_keys=False))
    return report


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def write_dashboard(reports: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for r in reports:
        sr = r.get("stage_results", {})
        cases.append({
            "case_id": r.get("case_id"),
            "difficulty": r.get("input", {}).get("difficulty"),
            "status": r.get("status"),
            "passed_mvp": r.get("final_result", {}).get("passed_mvp"),
            "overall_score": r.get("final_result", {}).get("overall_score"),
            "blocking_failures": r.get("blocking_failures", []),
            "segmentation_iou": sr.get("segmentation", {}).get("mask_iou"),
            "segmentation_iou_unguided":
                sr.get("segmentation_unguided", {}).get("mask_iou"),
            "trace_iou": sr.get("vectorization", {}).get("silhouette_svg_iou"),
            "control_point_reduction":
                sr.get("primitive_fitting", {}).get("control_point_reduction"),
            "primitive_accuracy":
                sr.get("primitive_fitting", {}).get("primitive_accuracy"),
            "major_part_recall":
                sr.get("semantic_parts", {}).get("major_part_recall"),
            "construction_plan_valid":
                sr.get("construction_plan", {}).get("valid"),
            "blender_execution_success":
                sr.get("blender", {}).get("execution_success"),
            "blend_reopens": (sr.get("blender_reopen") or {}).get("reopens"),
            "glb_valid": sr.get("blender", {}).get("glb", {}).get("valid"),
            "silhouette_iou": sr.get("visual", {}).get("silhouette_iou_final"),
            "contour_error": sr.get("visual", {}).get("contour_chamfer_distance"),
            "safety_violations":
                len(sr.get("blender", {}).get("safety_violations") or []),
            "uncertainty_warnings": bool(r.get("uncertainty")),
            "baseline_svg_extrusion_iou":
                r.get("baseline_svg_extrusion", {}).get("reference_silhouette_iou"),
            "no_refinement_iou":
                r.get("ablation_no_refinement", {}).get("silhouette_iou_initial"),
        })

    summary = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "n_passed_mvp": sum(1 for c in cases if c["passed_mvp"]),
        "reconstruction_status":
            "pass" if cases and all(c["passed_mvp"] for c in cases) else "fail",
        "geometry_score": _mean([c["silhouette_iou"] for c in cases]),
        "segmentation_iou_mean": _mean([c["segmentation_iou"] for c in cases]),
        "trace_iou_mean": _mean([c["trace_iou"] for c in cases]),
        "control_point_reduction_mean":
            _mean([c["control_point_reduction"] for c in cases]),
        "primitive_accuracy_mean": _mean([c["primitive_accuracy"] for c in cases]),
        "part_recall_mean": _mean([c["major_part_recall"] for c in cases]),
        "silhouette_iou_mean": _mean([c["silhouette_iou"] for c in cases]),
        "contour_error_mean": _mean([c["contour_error"] for c in cases]),
        "overall_score_mean": _mean([c["overall_score"] for c in cases]),
        "blender_success_rate": _mean([
            float(c["blender_execution_success"]) for c in cases
            if c["blender_execution_success"] is not None]),
        "glb_valid_rate": _mean([
            float(c["glb_valid"]) for c in cases
            if c["glb_valid"] is not None]),
        "safety_violations_total": sum(c["safety_violations"] for c in cases),
        "baseline_svg_extrusion_iou_mean":
            _mean([c["baseline_svg_extrusion_iou"] for c in cases]),
        "no_refinement_iou_mean": _mean([c["no_refinement_iou"] for c in cases]),
    }
    dashboard = {"summary": summary, "cases": cases}
    (out_dir / "dashboard.json").write_text(
        json.dumps(_jsonable(dashboard), indent=2))

    # markdown dashboard
    lines = ["# E2E Evaluation Dashboard", "",
             "- cases: %d | passed MVP: %d | status: %s"
             % (summary["n_cases"], summary["n_passed_mvp"],
                summary["reconstruction_status"]),
             "- segmentation IoU (mean): %s" % _fmt(summary["segmentation_iou_mean"]),
             "- trace IoU (mean): %s" % _fmt(summary["trace_iou_mean"]),
             "- control-point reduction (mean): %s" % _fmt(summary["control_point_reduction_mean"]),
             "- primitive accuracy (mean): %s" % _fmt(summary["primitive_accuracy_mean"]),
             "- part recall (mean): %s" % _fmt(summary["part_recall_mean"]),
             "- silhouette IoU (mean): %s" % _fmt(summary["silhouette_iou_mean"]),
             "- contour error (mean): %s" % _fmt(summary["contour_error_mean"]),
             "- blender success rate: %s" % _fmt(summary["blender_success_rate"]),
             "- GLB valid rate: %s" % _fmt(summary["glb_valid_rate"]),
             "- safety violations: %d" % summary["safety_violations_total"],
             "- baseline svg_extrusion IoU (mean): %s" % _fmt(summary["baseline_svg_extrusion_iou_mean"]),
             "- no-refinement IoU (mean): %s" % _fmt(summary["no_refinement_iou_mean"]),
             "",
             "| case | diff | status | MVP | score | seg IoU | trace IoU | prim acc | part recall | plan | blender | glb | sil IoU | baseline IoU |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["case_id"], c["difficulty"], c["status"],
            "PASS" if c["passed_mvp"] else "FAIL",
            _fmt(c["overall_score"]), _fmt(c["segmentation_iou"]),
            _fmt(c["trace_iou"]), _fmt(c["primitive_accuracy"]),
            _fmt(c["major_part_recall"]),
            _yn(c["construction_plan_valid"]),
            _yn(c["blender_execution_success"]),
            _yn(c["glb_valid"]),
            _fmt(c["silhouette_iou"]),
            _fmt(c["baseline_svg_extrusion_iou"])))
    (out_dir / "dashboard.md").write_text("\n".join(lines) + "\n")
    return dashboard


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else ("%.3f" % v)


def _yn(v: Optional[bool]) -> str:
    if v is None:
        return "?"
    return "Y" if v else "N"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_e2e")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--dataset",
                    default=str(Path(__file__).resolve().parents[1]
                                / "benchmark" / "dataset"))
    ap.add_argument("--projects-root", default="projects/e2e")
    ap.add_argument("--out", default="evals/results/e2e")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--skip-blender", action="store_true",
                    help="skip blender reopen probe + baseline render")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="score existing project dirs without running the pipeline")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    projects_root = Path(args.projects_root)
    blender = None if args.skip_blender else args.blender

    available = list_cases(dataset_dir)
    if args.cases.strip().lower() == "all":
        case_ids = available
    else:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        case_ids = [c for c in wanted if c in available]
        missing = set(wanted) - set(available)
        for cid in sorted(missing):
            print("[SKIP] %s not in dataset" % cid)

    reports: List[Dict[str, Any]] = []
    for i, cid in enumerate(case_ids):
        case = load_case(dataset_dir, cid)
        print("[%d/%d] %s ..." % (i + 1, len(case_ids), cid), flush=True)
        report = evaluate_case(case, projects_root, out_dir, args.python,
                               blender, args.skip_pipeline, args.timeout)
        fr = report.get("final_result", {})
        print("      -> %s score=%s blocking=%s"
              % ("PASS" if fr.get("passed_mvp") else "FAIL",
                 fr.get("overall_score"),
                 report.get("blocking_failures", [])), flush=True)
        reports.append(report)

    dashboard = write_dashboard(reports, out_dir)
    s = dashboard["summary"]
    print("\n=== E2E summary: %d/%d passed MVP | silhouette IoU mean %s | "
          "baseline IoU mean %s ==="
          % (s["n_passed_mvp"], s["n_cases"],
             _fmt(s["silhouette_iou_mean"]),
             _fmt(s["baseline_svg_extrusion_iou_mean"])))
    print("dashboard: %s" % (out_dir / "dashboard.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
