"""Render a completed reconstruction at a calibrated, held-out benchmark view.

The target view is never passed to reconstruction. Camera scale and offset are
fixed from the primary input; only the declared relative azimuth is applied.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2

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
    params = {
        "render_engine": cfg.blender.render_engine or "BLENDER_EEVEE",
        "width": int(target.shape[1]), "height": int(target.shape[0]),
        "azimuth_deg": azimuth,
        "base_object_rotation_deg": _base_rotation(plan),
        "ortho_scale": framing["ortho_scale"],
        "camera_location": framing["camera_location"],
        "output_path": str(output_path),
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
    report["passed_iou_0_75"] = report["heldout_silhouette_iou"] >= 0.75
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
    return 0 if report["passed_iou_0_75"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
