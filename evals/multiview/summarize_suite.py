"""Aggregate calibrated held-out metrics and uncertainty diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _ece(confidences: np.ndarray, outcomes: np.ndarray,
         bins: int = 5) -> float:
    total = len(confidences)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = ((confidences >= low)
                    & (confidences <= high if index == bins - 1
                       else confidences < high))
        if selected.any():
            value += (selected.sum() / total
                      * abs(float(confidences[selected].mean())
                            - float(outcomes[selected].mean())))
    return float(value)


def summarize_entries(entries: Dict[str, Path],
                      calibration_model: Optional[Dict] = None) -> Dict:
    from evals.multiview.calibration import predict_probability

    cases = []
    for case_id, project in entries.items():
        heldout = json.loads(
            (project / "validation" / "heldout_view_002_metrics.json").read_text())
        primary = json.loads((project / "validation" / "metrics.json").read_text())
        multiview = json.loads((project / "geometry" / "multiview.json").read_text())
        hull = (multiview.get("joint_optimization") or {}).get("visual_hull") or {}
        iou = float(heldout["heldout_silhouette_iou"])
        chamfer = float(heldout["normalized_surface_chamfer_distance"])
        confidence = float(hull.get("completion_confidence", 0.0))
        artifact_risk = str(hull.get("unseen_view_risk", "unknown"))
        rejected = any("visual hull rejected" in str(warning).lower()
                       for warning in multiview.get("warnings", []))
        risk = "high" if artifact_risk == "unknown" and rejected else artifact_risk
        cases.append({
            "case_id": case_id,
            "primary_silhouette_iou": float(primary["silhouette_iou"]),
            "heldout_silhouette_iou": iou,
            "normalized_surface_chamfer_distance": chamfer,
            "surface_normal_consistency": float(
                heldout["surface_normal_consistency"]),
            "hidden_completion_confidence": confidence,
            "artifact_unseen_view_risk": artifact_risk,
            "unseen_view_risk": risk,
            "silhouette_pass": iou >= 0.75,
            "surface_chamfer_pass": chamfer <= 0.05,
            "measured_eval20_subset_pass": iou >= 0.75 and chamfer <= 0.05,
        })
        if calibration_model is not None:
            cases[-1]["silhouette_pass_probability"] = predict_probability(
                calibration_model, cases[-1])
    ious = np.asarray([case["heldout_silhouette_iou"] for case in cases])
    chamfers = np.asarray(
        [case["normalized_surface_chamfer_distance"] for case in cases])
    normals = np.asarray([case["surface_normal_consistency"] for case in cases])
    hidden_confidences = np.asarray(
        [case["hidden_completion_confidence"] for case in cases])
    probabilities = (None if calibration_model is None else np.asarray(
        [case["silhouette_pass_probability"] for case in cases]))
    outcomes = np.asarray([float(case["silhouette_pass"]) for case in cases])
    actual_fail = ~outcomes.astype(bool)

    def detection(threshold: str) -> Dict[str, float]:
        ranks = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
        predicted = np.asarray([
            ranks.get(case["unseen_view_risk"], 0) >= ranks[threshold]
            for case in cases])
        return {
            "threshold": threshold,
            "failure_detection_rate": _rate(
                int(np.logical_and(predicted, actual_fail).sum()),
                int(actual_fail.sum())),
            "false_failure_rate_on_passing_cases": _rate(
                int(np.logical_and(predicted, ~actual_fail).sum()),
                int((~actual_fail).sum())),
        }

    return {
        "case_count": len(cases),
        "cases": cases,
        "aggregate": {
            "median_heldout_silhouette_iou": float(np.median(ious)),
            "silhouette_pass_rate": float(np.mean(ious >= 0.75)),
            "median_normalized_surface_chamfer_distance": float(
                np.median(chamfers)),
            "surface_chamfer_pass_rate": float(np.mean(chamfers <= 0.05)),
            "median_surface_normal_consistency": float(np.median(normals)),
            "measured_eval20_subset_pass_rate": float(np.mean(
                (ious >= 0.75) & (chamfers <= 0.05))),
            "hidden_completion_confidence_mean": float(
                np.mean(hidden_confidences)),
            "silhouette_calibration_status": (
                "measured_external_model" if calibration_model is not None
                else "not_measured"),
            "silhouette_confidence_ece": (
                None if probabilities is None else _ece(probabilities, outcomes)),
            "silhouette_confidence_brier": (
                None if probabilities is None else float(np.mean(
                    (probabilities - outcomes) ** 2))),
            "hidden_geometry_high_confidence_error_rate": _rate(
                int(np.logical_and(hidden_confidences >= 0.8, actual_fail).sum()),
                int((hidden_confidences >= 0.8).sum())),
            "failure_detection_high_only": detection("high"),
            "failure_detection_medium_or_high": detection("medium"),
        },
    }


def _markdown(summary: Dict) -> str:
    aggregate = summary["aggregate"]
    probability_column = any(
        "silhouette_pass_probability" in case for case in summary["cases"])
    heading = "| Case | Primary | Held-out | Chamfer | Risk"
    divider = "| --- | ---: | ---: | ---: | ---"
    if probability_column:
        heading += " | Pass probability"
        divider += " | ---:"
    lines = ["# Multiview Suite Summary", "", heading + " |", divider + " |"]
    for case in summary["cases"]:
        row = ("| `{case_id}` | {primary_silhouette_iou:.3f} | "
               "{heldout_silhouette_iou:.3f} | "
               "{normalized_surface_chamfer_distance:.3f} | "
               "{unseen_view_risk}").format(**case)
        if probability_column:
            row += " | %.3f" % case["silhouette_pass_probability"]
        lines.append(row + " |")
    lines += [
        "",
        "- median held-out silhouette IoU: %.3f" % aggregate[
            "median_heldout_silhouette_iou"],
        "- median normalized surface Chamfer: %.3f" % aggregate[
            "median_normalized_surface_chamfer_distance"],
        "- silhouette calibration status: %s" % aggregate[
            "silhouette_calibration_status"],
        "- silhouette confidence ECE: %s" % (
            "not measured" if aggregate["silhouette_confidence_ece"] is None
            else "%.3f" % aggregate["silhouette_confidence_ece"]),
        "- hidden-geometry high-confidence error rate: %.3f" % aggregate[
            "hidden_geometry_high_confidence_error_rate"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", required=True,
        help="case_id=project_dir; repeat for every evaluated case")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--calibration-model", default=None,
        help="model JSON fitted on a disjoint calibration cohort")
    args = parser.parse_args()
    entries = {}
    for raw in args.entry:
        if "=" not in raw:
            parser.error("--entry must be case_id=project_dir")
        case_id, path = raw.split("=", 1)
        entries[case_id] = Path(path)
    model = (None if args.calibration_model is None else
             json.loads(Path(args.calibration_model).read_text()))
    summary = summarize_entries(entries, calibration_model=model)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    output.with_suffix(".md").write_text(_markdown(summary))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
