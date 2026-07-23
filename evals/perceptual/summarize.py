"""Aggregate reference-view perceptual metrics and required render passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REQUIRED_PASSES = (
    "render_silhouette.png", "render_clay.png", "render_normal.png",
    "render_depth.png", "render_partid.png", "render_materialid.png",
    "render_shaded.png",
)


def collect_case(case_id: str, project: Path) -> Dict:
    metrics_path = project / "validation" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    passes = {name: (project / "validation" / name).is_file()
              for name in REQUIRED_PASSES}
    shaded = metrics.get("perceptual_similarity")
    clay = metrics.get("clay_silhouette_iou")
    return {
        "case_id": case_id,
        "project_dir": str(project),
        "shaded_ssim": shaded,
        "color_region_agreement": metrics.get("color_region_agreement"),
        "clay_silhouette_iou": clay,
        "silhouette_iou": metrics.get("silhouette_iou"),
        "depth_correlation": metrics.get("depth_correlation"),
        "feature_alignment_error_px": metrics.get(
            "feature_alignment_error_px"),
        "render_passes": passes,
        "all_required_passes": all(passes.values()),
        "geometry_compensation_flag": (
            shaded is not None and clay is not None
            and float(shaded) >= 0.8 and float(clay) < 0.8),
    }


def _values(cases: Iterable[Dict], key: str) -> np.ndarray:
    return np.asarray([float(case[key]) for case in cases
                       if case.get(key) is not None], dtype=float)


def summarize(projects: Dict[str, Path]) -> Dict:
    cases = [collect_case(case_id, project)
             for case_id, project in sorted(projects.items())]
    aggregate = {}
    for key in ("shaded_ssim", "color_region_agreement",
                "clay_silhouette_iou", "silhouette_iou",
                "depth_correlation", "feature_alignment_error_px"):
        values = _values(cases, key)
        aggregate["mean_%s" % key] = (
            float(values.mean()) if len(values) else None)
        aggregate["median_%s" % key] = (
            float(np.median(values)) if len(values) else None)
    aggregate["geometry_compensation_flag_count"] = sum(
        case["geometry_compensation_flag"] for case in cases)
    aggregate["required_pass_coverage"] = {
        name: sum(case["render_passes"][name] for case in cases) / len(cases)
        for name in REQUIRED_PASSES
    } if cases else {}
    return {"case_count": len(cases), "cases": cases, "aggregate": aggregate}


def markdown(summary: Dict) -> str:
    a = summary["aggregate"]
    lines = [
        "# Eval 22 Perceptual Render Summary", "",
        "| Case | Shaded SSIM | Color agreement | Clay silhouette IoU | Geometry-compensation flag |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            "| `{case_id}` | {shaded_ssim:.3f} | {color_region_agreement:.3f} | "
            "{clay_silhouette_iou:.3f} | {geometry_compensation_flag} |".format(
                **case))
    lines += [
        "",
        "- mean shaded SSIM: %.3f" % a["mean_shaded_ssim"],
        "- mean color-region agreement: %.3f" % a["mean_color_region_agreement"],
        "- mean clay silhouette IoU: %.3f" % a["mean_clay_silhouette_iou"],
        "- geometry-compensation flags: %d/%d" % (
            a["geometry_compensation_flag_count"], summary["case_count"]),
        "- render-pass coverage: " + ", ".join(
            "%s %.0f%%" % (name.removeprefix("render_").removesuffix(".png"),
                            100.0 * coverage)
            for name, coverage in a["required_pass_coverage"].items()),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.projects_root)
    projects = {path.name: path for path in root.iterdir()
                if path.is_dir()
                and (path / "validation" / "metrics.json").is_file()
                and not path.name.endswith("_nomask")}
    summary = summarize(projects)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    output.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
