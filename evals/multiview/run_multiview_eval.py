"""Render a completed reconstruction at a calibrated, held-out benchmark view.

The target view is never passed to reconstruction. Camera scale and offset are
fixed from the primary input; only the declared relative azimuth is applied.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from recon3d import runner, validation
from recon3d.config import PipelineConfig
from recon3d.schemas import ConstructionPlan, SchemaIO, SegmentationResult


_SCRIPT = '''\
import json
import math
import os
import sys
import traceback
import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
PROJECT_DIR, BLEND_PATH, GLB_PATH, MANIFEST_PATH = ARGV[:4]
P = json.loads(__PARAMS_JSON__)
manifest = {"blend_path": BLEND_PATH, "glb_path": GLB_PATH, "objects": [],
            "collections": [], "blender_version": bpy.app.version_string,
            "success": False, "errors": []}
try:
    bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)
    scene = bpy.context.scene
    try:
        scene.render.engine = P["render_engine"]
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(P["width"])
    scene.render.resolution_y = int(P["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("recon3d_heldout_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    scene.world = world
    material = bpy.data.materials.new("recon3d_heldout_mask")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    scene.view_layers[0].material_override = material

    root = next(o for o in bpy.data.objects if o.get("recon3d_object_id"))
    base = P["base_object_rotation_deg"]
    root.rotation_euler = (math.radians(float(base[0])),
                           math.radians(float(base[1]) - float(P["azimuth_deg"])),
                           math.radians(float(base[2])))
    camera_data = bpy.data.cameras.new("recon3d_heldout_camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(P["ortho_scale"])
    camera = bpy.data.objects.new("recon3d_heldout_camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = tuple(float(v) for v in P["camera_location"])
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
    scene.render.filepath = P["output_path"]
    bpy.ops.render.render(write_still=True)

    def sample_visible_meshes(objects, count, seed):
        triangles = []
        normals = []
        areas = []
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            if obj.type != "MESH" or obj.hide_render:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            matrix = obj.matrix_world
            for tri in mesh.loop_triangles:
                a, b, c = [matrix @ mesh.vertices[i].co for i in tri.vertices]
                cross = (b - a).cross(c - a)
                area = cross.length * 0.5
                if area <= 1e-12:
                    continue
                triangles.append(((a.x, a.y, a.z),
                                  (b.x, b.y, b.z),
                                  (c.x, c.y, c.z)))
                normal = cross.normalized()
                normals.append((normal.x, normal.y, normal.z))
                areas.append(area)
            evaluated.to_mesh_clear()
        if not triangles:
            return [], []
        tri = np.asarray(triangles, dtype=np.float64)
        norm = np.asarray(normals, dtype=np.float64)
        probability = np.asarray(areas, dtype=np.float64)
        probability /= probability.sum()
        rng = np.random.default_rng(seed)
        index = rng.choice(len(tri), size=count, p=probability)
        uv = rng.random((count, 2))
        over = uv.sum(axis=1) > 1.0
        uv[over] = 1.0 - uv[over]
        points = (tri[index, 0]
                  + uv[:, :1] * (tri[index, 1] - tri[index, 0])
                  + uv[:, 1:] * (tri[index, 2] - tri[index, 0]))
        return points.tolist(), norm[index].tolist()

    generated_points, generated_normals = sample_visible_meshes(
        list(bpy.data.objects), int(P["surface_sample_count"]), 1729)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=P["reference_glb"])
    reference_points, reference_normals = sample_visible_meshes(
        list(bpy.data.objects), int(P["surface_sample_count"]), 2718)
    with open(P["surface_samples_path"], "w") as fh:
        json.dump({"generated_points": generated_points,
                   "generated_normals": generated_normals,
                   "reference_points": reference_points,
                   "reference_normals": reference_normals}, fh)
    manifest["success"] = True
    manifest["collections"] = [c.name for c in bpy.data.collections]
except Exception as exc:
    manifest["errors"] = [str(exc), traceback.format_exc()]
with open(MANIFEST_PATH, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
'''


def fixed_camera_from_primary(seg: SegmentationResult,
                              image_shape: Tuple[int, int]) -> Dict:
    """Return orthographic framing derived only from the primary view."""
    height, width = image_shape
    x0, y0, x1, y1 = seg.bbox
    object_width = max(1, x1 - x0)
    view_width = float(width) / float(object_width)
    cx = (x0 + x1) * 0.5 / width
    cy = (y0 + y1) * 0.5 / height
    return {
        "ortho_scale": view_width * float(height) / float(width),
        "camera_location": [-(cx - 0.5) * view_width,
                            (cy - 0.5) * view_width, 2.0],
    }


def _base_rotation(plan: ConstructionPlan) -> list[float]:
    if plan.camera is not None:
        value = plan.camera.object_rotation_euler_deg.value
        if isinstance(value, list) and len(value) == 3:
            return [float(v) for v in value]
    return [0.0, 0.0, 0.0]


def _proper_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base * np.asarray(signs)[None, :]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def _kabsch(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def align_surface_metrics(generated_points: np.ndarray,
                          generated_normals: np.ndarray,
                          reference_points: np.ndarray,
                          reference_normals: np.ndarray) -> Dict[str, float]:
    """Similarity-normalize and rigidly align two sampled surfaces."""
    gp = np.asarray(generated_points, dtype=np.float64)
    gn = np.asarray(generated_normals, dtype=np.float64)
    rp = np.asarray(reference_points, dtype=np.float64)
    rn = np.asarray(reference_normals, dtype=np.float64)
    if min(len(gp), len(rp)) < 10:
        raise ValueError("surface metric requires at least ten points per mesh")
    gp -= gp.mean(axis=0)
    rp -= rp.mean(axis=0)
    gp /= max(1e-12, float(np.sqrt(np.mean(np.sum(gp * gp, axis=1)))))
    rp /= max(1e-12, float(np.sqrt(np.mean(np.sum(rp * rp, axis=1)))))
    tree = cKDTree(rp)
    best = None
    for initial in _proper_axis_rotations():
        points = gp @ initial
        normals = gn @ initial
        for _ in range(12):
            _, index = tree.query(points, k=1)
            delta, translation = _kabsch(points, rp[index])
            points = points @ delta + translation
            normals = normals @ delta
        distances, index = tree.query(points, k=1)
        score = float(np.mean(distances))
        if best is None or score < best[0]:
            best = (score, points, normals)
    _, aligned, aligned_normals = best
    ref_tree = cKDTree(rp)
    gen_tree = cKDTree(aligned)
    d_gr, i_gr = ref_tree.query(aligned, k=1)
    d_rg, i_rg = gen_tree.query(rp, k=1)
    reference_diagonal = float(np.linalg.norm(rp.max(axis=0) - rp.min(axis=0)))
    chamfer = float((np.mean(d_gr) + np.mean(d_rg)) * 0.5
                    / max(reference_diagonal, 1e-12))
    ng = aligned_normals / np.maximum(
        np.linalg.norm(aligned_normals, axis=1, keepdims=True), 1e-12)
    nr = rn / np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-12)
    consistency_gr = np.abs(np.sum(ng * nr[i_gr], axis=1)).mean()
    consistency_rg = np.abs(np.sum(nr * ng[i_rg], axis=1)).mean()
    return {
        "normalized_surface_chamfer_distance": chamfer,
        "surface_normal_consistency": float(
            0.5 * (consistency_gr + consistency_rg)),
        "surface_sample_count_per_mesh": int(min(len(gp), len(rp))),
    }


def evaluate_heldout(case_dir: str, project_dir: str,
                     heldout_view: str = "view_002",
                     cfg: PipelineConfig | None = None) -> Dict:
    case = Path(case_dir).resolve()
    project = Path(project_dir).resolve()
    cfg = cfg or PipelineConfig()
    target_path = case / "views" / heldout_view / "mask.png"
    camera_path = case / "views" / heldout_view / "camera.json"
    if not target_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError("held-out mask/camera missing for %s" % heldout_view)
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise ValueError("cannot read held-out mask %s" % target_path)
    target = validation._binarize(target)
    camera = json.loads(camera_path.read_text())
    azimuth = float(camera["relative_azimuth_deg"])
    seg = SegmentationResult.model_validate_json(
        (project / "segmentation" / "segmentation_result.json").read_text())
    plan = SchemaIO.load_yaml(
        ConstructionPlan, project / "geometry" / "construction_plan.yaml")
    framing = fixed_camera_from_primary(seg, target.shape)
    output_path = project / "validation" / ("heldout_%s.png" % heldout_view)
    reference_copy = project / "validation" / "heldout_reference.glb"
    shutil.copy2(case / "reference.glb", reference_copy)
    surface_samples_path = project / "validation" / "heldout_surface_samples.json"
    params = {
        "render_engine": cfg.blender.render_engine or "BLENDER_EEVEE",
        "width": int(target.shape[1]), "height": int(target.shape[0]),
        "azimuth_deg": azimuth,
        "base_object_rotation_deg": _base_rotation(plan),
        "ortho_scale": framing["ortho_scale"],
        "camera_location": framing["camera_location"],
        "output_path": str(output_path),
        "reference_glb": str(reference_copy),
        "surface_samples_path": str(surface_samples_path),
        "surface_sample_count": 3000,
    }
    script_text = _SCRIPT.replace(
        "__PARAMS_JSON__", json.dumps(json.dumps(params, sort_keys=True)))
    script_path = project / "blender" / ("render_heldout_%s.py" % heldout_view)
    script_path.write_text(script_text)
    manifest = runner.run_blender(str(script_path), str(project), cfg)
    if not manifest.success:
        raise RuntimeError("held-out render failed: %s" % "; ".join(manifest.errors))
    rendered = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if rendered is None:
        raise RuntimeError("held-out renderer did not write %s" % output_path)
    rendered = validation._binarize(rendered)
    report = {
        "case": case.name,
        "project_dir": str(project),
        "heldout_view": heldout_view,
        "heldout_relative_azimuth_deg": azimuth,
        "camera_policy": "primary scale/offset fixed; held-out azimuth exact",
        "heldout_silhouette_iou": validation._iou(target, rendered),
        "heldout_contour_chamfer_distance": validation._chamfer(target, rendered),
        "target_mask": str(target_path),
        "rendered_mask": str(output_path),
        "passed_iou_0_75": False,
    }
    samples = json.loads(surface_samples_path.read_text())
    report.update(align_surface_metrics(
        np.asarray(samples["generated_points"]),
        np.asarray(samples["generated_normals"]),
        np.asarray(samples["reference_points"]),
        np.asarray(samples["reference_normals"])))
    report["passed_iou_0_75"] = report["heldout_silhouette_iou"] >= 0.75
    report["passed_chamfer_0_05"] = (
        report["normalized_surface_chamfer_distance"] <= 0.05)
    report["passed_measured_eval20_subset"] = (
        report["passed_iou_0_75"] and report["passed_chamfer_0_05"])
    report_path = project / "validation" / ("heldout_%s_metrics.json" % heldout_view)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--heldout-view", default="view_002")
    args = parser.parse_args()
    report = evaluate_heldout(args.case, args.project, args.heldout_view)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed_measured_eval20_subset"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
