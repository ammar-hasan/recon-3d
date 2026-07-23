"""Evaluate a third-party mesh against benchmark reference geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evals.multiview.run_multiview_eval import align_surface_metrics
from recon3d import runner
from recon3d.config import PipelineConfig


_SCRIPT = '''\
import json
import sys

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
PROJECT_DIR, BLEND_PATH, _GLB_PATH, MANIFEST_PATH = ARGV[:4]
P = json.loads(__PARAMS_JSON__)
manifest = {"success": False, "errors": [], "collections": [], "objects": []}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def sample_meshes(count, seed):
        triangles, normals, areas = [], [], []
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in list(bpy.data.objects):
            if obj.type != "MESH" or obj.hide_render:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            matrix = obj.matrix_world
            for tri in mesh.loop_triangles:
                a, b, c = [matrix @ mesh.vertices[i].co for i in tri.vertices]
                cross = (b - a).cross(c - a)
                area = 0.5 * cross.length
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
            raise RuntimeError("mesh contains no sampleable triangles")
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

    bpy.ops.import_scene.gltf(filepath=P["generated_mesh"])
    generated_points, generated_normals = sample_meshes(P["count"], 1729)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=P["reference_mesh"])
    reference_points, reference_normals = sample_meshes(P["count"], 2718)
    with open(P["samples"], "w") as handle:
        json.dump({
            "generated_points": generated_points,
            "generated_normals": generated_normals,
            "reference_points": reference_points,
            "reference_normals": reference_normals,
        }, handle)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    manifest["success"] = True
except Exception as exc:
    import traceback
    manifest["errors"] = [str(exc), traceback.format_exc()]

with open(MANIFEST_PATH, "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
'''


def evaluate_mesh(generated_mesh: Path, reference_mesh: Path, output: Path,
                  sample_count: int = 3000,
                  cfg: PipelineConfig | None = None) -> dict:
    if sample_count < 100:
        raise ValueError("surface evaluation requires at least 100 samples")
    generated_mesh = generated_mesh.resolve()
    reference_mesh = reference_mesh.resolve()
    if not generated_mesh.is_file():
        raise FileNotFoundError(generated_mesh)
    if not reference_mesh.is_file():
        raise FileNotFoundError(reference_mesh)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "surface_samples.json"
    params = {
        "generated_mesh": str(generated_mesh),
        "reference_mesh": str(reference_mesh),
        "samples": str(samples_path),
        "count": sample_count,
    }
    script = output / "sample_meshes.py"
    script.write_text(_SCRIPT.replace(
        "__PARAMS_JSON__", json.dumps(json.dumps(params, sort_keys=True))))
    manifest = runner.run_blender(str(script), str(output),
                                  cfg or PipelineConfig())
    if not manifest.success:
        raise RuntimeError("mesh sampling failed: %s"
                           % "; ".join(manifest.errors))
    samples = json.loads(samples_path.read_text())
    metrics = align_surface_metrics(
        np.asarray(samples["generated_points"]),
        np.asarray(samples["generated_normals"]),
        np.asarray(samples["reference_points"]),
        np.asarray(samples["reference_normals"]))
    report = {
        "generated_mesh": str(generated_mesh),
        "reference_mesh": str(reference_mesh),
        **metrics,
    }
    (output / "metrics.json").write_text(json.dumps(
        report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    result = evaluate_mesh(
        Path(args.generated), Path(args.reference), Path(args.out),
        args.samples)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
