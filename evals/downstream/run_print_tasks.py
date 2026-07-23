"""Evaluate watertight conversion, thickness probing, and STL export."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from recon3d.config import PipelineConfig


_BLENDER_SCRIPT = r'''
import bmesh
import bpy
import json
import math
import sys
import traceback

source, stl_path, result_path, voxel_raw = sys.argv[sys.argv.index("--") + 1:]
voxel_size = float(voxel_raw)
result = {"success": False, "errors": [], "voxel_size": voxel_size}
try:
    bpy.ops.wm.open_mainfile(filepath=source)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    converted = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated)
        mesh.transform(obj.matrix_world)
        copy = bpy.data.objects.new("print_" + obj.name, mesh)
        bpy.context.scene.collection.objects.link(copy)
        converted.append(copy)
    if not converted:
        raise RuntimeError("no render-visible mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in converted:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = converted[0]
    bpy.ops.object.join()
    printable = bpy.context.view_layer.objects.active
    printable.name = "printable_union"
    modifier = printable.modifiers.new("watertight_voxel_remesh", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = voxel_size
    modifier.adaptivity = 0.0
    modifier.use_remove_disconnected = False
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    mesh = printable.data
    mesh.validate(verbose=False)
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    non_manifold = sum(not edge.is_manifold for edge in bm.edges)
    boundary = sum(edge.is_boundary for edge in bm.edges)
    bm.free()

    thickness = []
    epsilon = max(1e-6, voxel_size * 0.02)
    polygon_step = max(1, len(mesh.polygons) // 1000)
    for polygon_index in range(0, len(mesh.polygons), polygon_step):
        polygon = mesh.polygons[polygon_index]
        origin = polygon.center - polygon.normal * epsilon
        hit, location, _normal, _index = printable.ray_cast(
            origin, -polygon.normal, distance=100.0)
        if hit:
            distance = (location - origin).length
            # Ignore immediate coplanar/self hits from voxel stair-stepping;
            # they are numerical surface intersections, not wall thickness.
            if distance >= voxel_size * 0.5:
                thickness.append(float(distance))
    min_thickness = min(thickness) if thickness else None

    bpy.ops.object.select_all(action="DESELECT")
    printable.select_set(True)
    bpy.context.view_layer.objects.active = printable
    bpy.ops.wm.stl_export(filepath=stl_path, export_selected_objects=True)
    stl_size = __import__("os").path.getsize(stl_path)
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.polygons)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=stl_path)
    imported = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    imported_non_manifold = 0
    for obj in imported:
        check = bmesh.new()
        check.from_mesh(obj.data)
        imported_non_manifold += sum(not edge.is_manifold for edge in check.edges)
        check.free()
    result.update({
        "success": non_manifold == 0 and boundary == 0
                   and imported_non_manifold == 0 and stl_size > 84,
        "vertex_count": vertex_count, "face_count": face_count,
        "non_manifold_edges": non_manifold, "boundary_edges": boundary,
        "minimum_ray_thickness": min_thickness,
        "minimum_thickness_threshold": voxel_size * 2.0,
        "minimum_thickness_pass": min_thickness is not None
                                  and min_thickness >= voxel_size * 2.0,
        "stl_size_bytes": stl_size, "stl_reimport_meshes": len(imported),
        "stl_reimport_non_manifold_edges": imported_non_manifold,
    })
    result["success"] = result["success"] and result["minimum_thickness_pass"]
except Exception as exc:
    result["errors"] = [str(exc), traceback.format_exc()]
with open(result_path, "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
'''


def run_case(case: str, source_blend: Path, output_root: Path,
             blender: str, voxel_size: float) -> dict:
    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    script = case_dir / "convert_to_printable.py"
    script.write_text(_BLENDER_SCRIPT)
    stl = case_dir / "printable.stl"
    result_path = case_dir / "result.json"
    started = time.perf_counter()
    process = subprocess.run(
        [blender, "--background", "--factory-startup", "--python", str(script),
         "--", str(source_blend), str(stl), str(result_path), str(voxel_size)],
        capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - started
    if result_path.is_file():
        result = json.loads(result_path.read_text())
    else:
        result = {"success": False, "errors": [
            (process.stderr or process.stdout or "no result")[-1000:]]}
    result.update({"case": case, "source_blend": str(source_blend),
                   "stl_path": str(stl), "elapsed_seconds": elapsed,
                   "blender_returncode": process.returncode})
    return result


def run(projects_root: str, out_dir: str, cases, voxel_size: float):
    cfg = PipelineConfig()
    root = Path(projects_root)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, root / case / "blender" / "scene.blend", output,
                     cfg.blender.blender_bin, voxel_size) for case in cases]
    report = {
        "suite": "eval30_print_tasks", "voxel_size": voxel_size,
        "metrics": {
            "watertight_conversion_rate": float(np.mean([
                row.get("non_manifold_edges") == 0
                and row.get("boundary_edges") == 0 for row in rows])),
            "minimum_thickness_pass_rate": float(np.mean([
                bool(row.get("minimum_thickness_pass")) for row in rows])),
            "stl_export_reimport_rate": float(np.mean([
                row.get("stl_reimport_non_manifold_edges") == 0
                and row.get("stl_reimport_meshes", 0) > 0 for row in rows])),
            "task_completion_rate": float(np.mean([
                bool(row.get("success")) for row in rows])),
            "median_elapsed_seconds": float(np.median([
                row["elapsed_seconds"] for row in rows])),
        },
        "passed": all(row.get("success") for row in rows), "cases": rows,
    }
    (output / "print_suite.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-root", default="projects/e2e_final")
    parser.add_argument("--out", default="evals/results_printing")
    parser.add_argument("--cases", default="box_01,bottle_01,gear_01")
    parser.add_argument("--voxel-size", type=float, default=0.015)
    args = parser.parse_args()
    cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    report = run(args.projects_root, args.out, cases, args.voxel_size)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
