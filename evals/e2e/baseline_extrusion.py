"""SVG-extrusion baseline renderer (EVAL.md baseline 1).

Traces the benchmark case's GT silhouette, extrudes it into a flat slab
facing the reference camera, renders it from the exact GT camera and returns
the rendered mask for comparison against the GT mask.

Runs Blender in background mode; every step is deterministic.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_BLENDER_SCRIPT = r"""
import json, sys, os
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
spec = json.load(open(argv[0]))
out_png = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
cam_gt = spec["camera"]
res = cam_gt["resolution"][0]
focal = cam_gt["focal_length_px"]
d = cam_gt["camera_distance"]
M = Matrix(cam_gt["camera_matrix_world"])

# unproject contour pixels to a plane at camera distance d
verts_front, verts_back = [], []
depth = spec["slab_depth"]
cam_z = Matrix(cam_gt["camera_matrix_world"]).to_3x3() @ Vector((0, 0, 1))
for (px, py) in spec["contour_px"]:
    x_c = (px - res / 2.0) / focal * d
    y_c = -(py - res / 2.0) / focal * d
    w = M @ Vector((x_c, y_c, -d, 1.0))
    p = Vector((w.x, w.y, w.z)) / w.w
    verts_front.append(tuple(p - cam_z * (depth / 2.0)))
    verts_back.append(tuple(p + cam_z * (depth / 2.0)))

n = len(verts_front)
verts = verts_front + verts_back
faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
for i in range(n):
    j = (i + 1) % n
    faces.append((i, j, n + j, n + i))
mesh = bpy.data.meshes.new("Extrusion_mesh")
mesh.from_pydata(verts, [], faces)
mesh.update()
ob = bpy.data.objects.new("Extrusion", mesh)
scn.collection.objects.link(ob)

# white emission material
mat = bpy.data.materials.new("White")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
outn = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1, 1, 1, 1)
nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
mesh.materials.append(mat)

# black world
world = bpy.data.worlds.new("W")
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs[0].default_value = (0, 0, 0, 1)
bg.inputs[1].default_value = 0.0
scn.world = world

# GT camera
cam = bpy.data.cameras.new("Camera")
cam.lens = cam_gt["lens_mm"]
cam.sensor_width = cam_gt["sensor_width_mm"]
cam_ob = bpy.data.objects.new("Camera", cam)
scn.collection.objects.link(cam_ob)
cam_ob.matrix_world = M
scn.camera = cam_ob

scn.render.engine = "BLENDER_EEVEE"
scn.render.resolution_x = res
scn.render.resolution_y = res
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_mode = "RGB"
scn.view_settings.view_transform = "Standard"
scn.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("BASELINE_OK")
"""


def render_extrusion_mask(case, out_dir: Path, blender: str,
                          timeout: int = 300) -> Optional[np.ndarray]:
    """Render the extrusion baseline for one case; returns the mask or None."""
    gt_mask = case["mask"]
    cam_gt = case["camera"]
    if gt_mask is None or not cam_gt:
        return None
    contour = cv2.findContours(
        (gt_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contour:
        return None
    c = max(contour, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    bbox = case["bbox"]
    bbox_w = max(1, bbox.get("x1", 1) - bbox.get("x0", 0))
    res = cam_gt["resolution"][0]
    focal = cam_gt["focal_length_px"]
    dist = cam_gt["camera_distance"]
    slab_depth = 0.3 * (bbox_w / focal * dist)  # 30% of object width

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {"contour_px": approx.tolist(), "camera": cam_gt,
            "slab_depth": slab_depth}
    spec_path = out_dir / "baseline_spec.json"
    spec_path.write_text(json.dumps(spec))
    script_path = out_dir / "baseline_render.py"
    script_path.write_text(_BLENDER_SCRIPT)
    out_png = out_dir / "baseline_mask.png"

    proc = subprocess.run(
        [blender, "--background", "--factory-startup", "--python",
         str(script_path), "--", str(spec_path), str(out_png)],
        capture_output=True, text=True, timeout=timeout)
    (out_dir / "baseline_log.txt").write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""))
    if "BASELINE_OK" not in (proc.stdout or ""):
        return None
    img = cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8) * 255
