"""Score a directory of third-party meshes against benchmark references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_mesh import evaluate_mesh


def aggregate_case_metrics(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("cannot aggregate an empty baseline suite")
    chamfers = np.asarray([
        case["normalized_surface_chamfer_distance"] for case in cases])
    normals = np.asarray([
        case["surface_normal_consistency"] for case in cases])
    return {
        "case_count": len(cases),
        "median_normalized_surface_chamfer_distance": float(
            np.median(chamfers)),
        "median_surface_normal_consistency": float(np.median(normals)),
        "chamfer_pass_count_0_05": int(np.sum(chamfers <= 0.05)),
    }


def evaluate_suite(dataset: Path, baseline: Path, output: Path,
                   cases: str = "all", sample_count: int = 3000) -> dict:
    dataset = dataset.resolve()
    baseline = baseline.resolve()
    output = output.resolve()
    requested = set() if cases == "all" else set(cases.split(","))
    case_dirs = sorted(path for path in dataset.iterdir()
                       if path.is_dir() and (path / "reference.glb").is_file())
    if requested:
        case_dirs = [path for path in case_dirs if path.name in requested]
        missing = requested - {path.name for path in case_dirs}
        if missing:
            raise ValueError("unknown cases: %s" % ", ".join(sorted(missing)))
    results = []
    for case in case_dirs:
        mesh = baseline / case.name / "mesh.glb"
        metrics = evaluate_mesh(
            mesh, case / "reference.glb", output / case.name, sample_count)
        results.append({"case_id": case.name, **metrics})
    summary = {
        "schema_version": 1,
        "method": "external_mesh_baseline",
        "sample_count_per_mesh": sample_count,
        "aggregate": aggregate_case_metrics(results),
        "cases": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(
        summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    result = evaluate_suite(
        Path(args.dataset), Path(args.baseline), Path(args.out),
        args.cases, args.samples)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
