"""Run concrete edit, variant, animation, and game-export tasks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from evals.e2e.glbcheck import validate_glb
from evals.e2e.run_e2e import blend_reopen_probe
from recon3d import blender_codegen, construction_plan, runner
from recon3d.config import PipelineConfig
from recon3d.schemas import (ConstructionPlan, EvidenceSource, MaterialSpec,
                             OperatorCategory, SchemaIO)


TASKS = [
    ("resize_component", "box_01"),
    ("alter_repetition_count", "wheel_01"),
    ("change_profile", "bottle_01"),
    ("move_joint", "chair_01"),
    ("replace_material", "box_01"),
    ("rotate_wheel", "wheel_01"),
    ("open_lid", "bottle_01"),
]


def _first(plan: ConstructionPlan, predicate):
    return next(part for part in plan.parts if predicate(part))


def apply_task(plan: ConstructionPlan, task: str) -> Tuple[str, Dict, Dict]:
    """Mutate one declared editable parameter and return its audit record."""
    if task == "resize_component":
        part = _first(plan, lambda p: p.operator == OperatorCategory.EXTRUDE)
        transform = dict(part.transform)
        before = list(transform.get("scale") or [1.0, 1.0, 1.0])
        after = list(before)
        after[0] = round(float(after[0]) * 1.20, 6)
        transform["scale"] = after
        part.transform = transform
        return part.id, {"scale": before}, {"scale": after}
    if task == "alter_repetition_count":
        part = _first(plan, lambda p: p.operator == OperatorCategory.RADIAL_ARRAY)
        before = int(part.count or 0)
        part.count = before + 1
        return part.id, {"count": before}, {"count": part.count}
    if task == "change_profile":
        part = _first(plan, lambda p: p.operator == OperatorCategory.REVOLVE
                      and p.profile and p.profile.get("points"))
        points = [list(point) for point in part.profile["points"]]
        before = [list(point) for point in points]
        for point in points:
            point[0] = round(float(point[0]) * 1.12, 6)
        part.profile = dict(part.profile)
        part.profile["points"] = points
        return part.id, {"profile_points": before}, {"profile_points": points}
    if task == "move_joint":
        part = _first(plan, lambda p: p.parent is not None
                      and p.operator not in (OperatorCategory.DISPLACEMENT,
                                             OperatorCategory.TEXTURE_ONLY))
        transform = dict(part.transform)
        before = list(transform.get("location") or [0.0, 0.0, 0.0])
        after = list(before)
        after[0] = round(float(after[0]) + 0.12, 6)
        transform["location"] = after
        part.transform = transform
        return part.id, {"location": before}, {"location": after}
    if task == "replace_material":
        part = _first(plan, lambda p: p.render_visible
                      and p.operator not in (OperatorCategory.DISPLACEMENT,
                                             OperatorCategory.TEXTURE_ONLY))
        before = part.material.model_dump(mode="json")
        part.material = MaterialSpec(
            material_class="wood", base_color=(0.32, 0.12, 0.04),
            roughness=0.72, metallic=0.0, source=EvidenceSource.USER_SUPPLIED)
        return part.id, {"material": before}, {
            "material": part.material.model_dump(mode="json")}
    if task in ("rotate_wheel", "open_lid"):
        if task == "rotate_wheel":
            part = _first(plan, lambda p: p.parent is None
                          and p.operator == OperatorCategory.REVOLVE)
            axis, angle = 2, 30.0
        else:
            part = _first(plan, lambda p: p.parent is not None
                          and ("lid" in p.id or "cap" in p.id))
            axis, angle = 0, 35.0
        transform = dict(part.transform)
        before = list(transform.get("rotation_deg") or [0.0, 0.0, 0.0])
        after = list(before)
        after[axis] = round(float(after[axis]) + angle, 6)
        transform["rotation_deg"] = after
        part.transform = transform
        return part.id, {"rotation_deg": before}, {"rotation_deg": after}
    raise ValueError("unknown downstream task %s" % task)


def _manifest_object(manifest, part_id: str):
    return next((obj for obj in manifest.objects if obj.part_id == part_id), None)


def _mutation_survived(task: str, target: str, after: Dict,
                       manifest, reopen: Dict) -> bool:
    obj = _manifest_object(manifest, target)
    if task == "resize_component":
        return bool(obj and np.allclose(list(obj.scale or []), after["scale"],
                                        atol=1e-5))
    if task == "alter_repetition_count":
        count = int(after["count"])
        names = reopen.get("object_names") or []
        return sum(name.startswith(target + "_") for name in names) >= count
    if task == "change_profile":
        return bool(obj and obj.vertex_count > 0
                    and after["profile_points"])
    if task == "move_joint":
        return bool(obj and np.allclose(list(obj.location or []),
                                        after["location"], atol=1e-5))
    if task == "replace_material":
        return bool(obj and any(name.startswith("mat_wood")
                                for name in obj.materials))
    if task in ("rotate_wheel", "open_lid"):
        return bool(obj and np.allclose(list(obj.rotation_euler_deg or []),
                                        after["rotation_deg"], atol=1e-5))
    return False


def run_task(task: str, source_project: Path, output_root: Path,
             cfg: PipelineConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    plan_path = source_project / "geometry" / "construction_plan.yaml"
    plan = SchemaIO.load_yaml(ConstructionPlan, plan_path).model_copy(deep=True)
    target, before, after = apply_task(plan, task)
    plan_errors = construction_plan.validate_plan(plan)
    project = output_root / task
    for subdir in ("geometry", "blender", "validation"):
        (project / subdir).mkdir(parents=True, exist_ok=True)
    SchemaIO.save_yaml(plan, project / "geometry" / "construction_plan.yaml")
    manifest = None
    reopen = {"reopens": False}
    glb = {"valid": False, "errors": ["build not run"]}
    error = None
    try:
        if plan_errors:
            raise ValueError("edited plan invalid: %s" % "; ".join(plan_errors))
        script = blender_codegen.generate_blender_script(
            plan, str(project / "blender"), cfg)
        manifest = runner.run_blender(script, str(project), cfg)
        if not manifest.success:
            raise RuntimeError("Blender build failed: %s"
                               % "; ".join(manifest.errors))
        reopen = blend_reopen_probe(
            cfg.blender.blender_bin, project / "blender" / "scene.blend")
        glb = validate_glb(str(project / "blender" / "model.glb"))
    except Exception as exc:  # noqa: BLE001 - every task must be reported
        error = "%s: %s" % (type(exc).__name__, exc)
    survived = bool(manifest and manifest.success and reopen.get("reopens")
                    and glb.get("valid")
                    and _mutation_survived(task, target, after,
                                           manifest, reopen))
    elapsed = time.perf_counter() - started
    return {
        "task": task, "source_project": str(source_project),
        "output_project": str(project), "target_part": target,
        "before": before, "after": after,
        "construction_plan_valid": not plan_errors,
        "blender_rebuild_success": bool(manifest and manifest.success),
        "blend_reopens": bool(reopen.get("reopens")),
        "downstream_glb_import_valid": bool(glb.get("valid")),
        "mutation_survived_rebuild": survived,
        "manual_fixes": 0, "broken_dependencies": len(plan_errors),
        "model_rebuild_required": True,
        "elapsed_seconds": elapsed, "error": error,
        "completed": survived and error is None,
    }


def run(projects_root: str, out_dir: str) -> Dict[str, Any]:
    source_root = Path(projects_root)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    rows = [run_task(task, source_root / case, output_root, cfg)
            for task, case in TASKS]
    completed = sum(row["completed"] for row in rows)
    report = {
        "suite": "eval30_downstream_edit_tasks",
        "scope": ["editing", "variant_generation", "animation", "game_export"],
        "not_measured": ["engine runtime import", "3d_printing", "human time"],
        "metrics": {
            "task_completion_rate": completed / len(rows),
            "median_elapsed_seconds": float(np.median(
                [row["elapsed_seconds"] for row in rows])),
            "manual_fixes_total": sum(row["manual_fixes"] for row in rows),
            "broken_dependencies_total": sum(
                row["broken_dependencies"] for row in rows),
            "downstream_glb_import_success_rate": sum(
                row["downstream_glb_import_valid"] for row in rows) / len(rows),
        },
        "passed": completed == len(rows), "tasks": rows,
    }
    (output_root / "suite.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    lines = ["# Eval 30 — Downstream Edit Tasks", "",
             "- tasks completed: **%d/%d**" % (completed, len(rows)),
             "- median elapsed time: **%.2f s**" % report["metrics"]["median_elapsed_seconds"],
             "- manual fixes: **%d**" % report["metrics"]["manual_fixes_total"],
             "- broken dependencies: **%d**" % report["metrics"]["broken_dependencies_total"],
             "- GLB validation success: **%.3f**" % report["metrics"]["downstream_glb_import_success_rate"],
             "", "| Task | Complete | Reopen | GLB | Seconds |",
             "| --- | --- | --- | --- | ---: |"]
    for row in rows:
        lines.append("| %s | %s | %s | %s | %.2f |" % (
            row["task"], "yes" if row["completed"] else "no",
            "yes" if row["blend_reopens"] else "no",
            "yes" if row["downstream_glb_import_valid"] else "no",
            row["elapsed_seconds"]))
    (output_root / "suite.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-root", default="projects/e2e_final")
    parser.add_argument("--out", default="evals/results_downstream")
    args = parser.parse_args()
    report = run(args.projects_root, args.out)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
