import json
from pathlib import Path

import pytest

from evals.multiview.summarize_suite import summarize_entries


def _case(root: Path, name: str, iou: float, chamfer: float,
          confidence: float, risk: str) -> Path:
    project = root / name
    (project / "validation").mkdir(parents=True)
    (project / "geometry").mkdir()
    (project / "validation" / "heldout_view_002_metrics.json").write_text(
        json.dumps({"heldout_silhouette_iou": iou,
                    "normalized_surface_chamfer_distance": chamfer,
                    "surface_normal_consistency": 0.8}))
    (project / "validation" / "metrics.json").write_text(
        json.dumps({"silhouette_iou": 0.9}))
    (project / "geometry" / "multiview.json").write_text(json.dumps({
        "joint_optimization": {"visual_hull": {
            "completion_confidence": confidence, "unseen_view_risk": risk}},
    }))
    return project


def test_suite_aggregates_geometry_and_failure_detection(tmp_path):
    entries = {
        "pass": _case(tmp_path, "pass", 0.9, 0.04, 0.8, "low"),
        "fail": _case(tmp_path, "fail", 0.4, 0.08, 0.2, "high"),
    }
    summary = summarize_entries(entries)
    aggregate = summary["aggregate"]
    assert aggregate["median_heldout_silhouette_iou"] == pytest.approx(0.65)
    assert aggregate["median_normalized_surface_chamfer_distance"] == pytest.approx(0.06)
    assert aggregate["measured_eval20_subset_pass_rate"] == 0.5
    assert aggregate["failure_detection_high_only"]["failure_detection_rate"] == 1.0
    assert aggregate["failure_detection_high_only"][
        "false_failure_rate_on_passing_cases"] == 0.0
    assert aggregate["silhouette_confidence_ece"] == pytest.approx(0.2)
